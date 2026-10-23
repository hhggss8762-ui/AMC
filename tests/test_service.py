"""Boundary tests for the multipart API; no network or GPU is required."""
import json
import logging
import threading
import time

import httpx
import numpy as np
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from amc.service import create_app, install_amc


def metadata(signal_id="signal_0", sample_count=2048, **overrides):
    result = {"signal_id": signal_id, "dtype": "float32", "sample_rate_hz": 100000,
              "estimated_bandwidth_hz": 25000, "sample_count": sample_count}
    result.update(overrides)
    return result


def signal_parts(index=0, signal_id=None, count=2048, iq=None, meta=None):
    if iq is None:
        iq = np.ones((count, 2), dtype="<f4").tobytes()
    if meta is None:
        meta = metadata(signal_id or f"signal_{index}", count)
    text = meta if isinstance(meta, str) else json.dumps(meta)
    return [(f"iq_{index}", (f"signal_{index}.bin", iq, "application/octet-stream")),
            (f"metadata_{index}", (None, text))]


def multipart(parts=(), frame_id="frame_0001"):
    return [("frame_id", (None, frame_id)), *parts]


@pytest.fixture
def factory():
    class FakeClassifier:
        def __init__(self, checkpoint, **kwargs):
            self.checkpoint = checkpoint
            self.kwargs = kwargs
            self.device = "cpu"
            self.classes = ["AM", "QPSK"]
            self.warmups = 0
            self.calls = []
            self.active = 0
            self.max_active = 0
            self.fail = False
            self.fail_warmup = False

        def warmup(self):
            self.warmups += 1
            if self.fail_warmup:
                raise RuntimeError("warmup failure")

        def predict_frame(self, frame_id, signals):
            self.calls.append((frame_id, signals))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.002)
                if self.fail:
                    raise RuntimeError("internal model failure")
                response = {"frame_id": frame_id, "results": [
                    {"signal_id": signal.metadata.signal_id,
                     "modulation_type": "QPSK" if signal.metadata.sample_count >= 2048 and np.any(signal.iq) else None}
                    for signal in signals
                ]}
                return response, {"score": "internal-only"}
            finally:
                self.active -= 1

    class Factory:
        def __init__(self):
            self.instances = []
            self.fail_warmup = False

        def __call__(self, checkpoint, **kwargs):
            instance = FakeClassifier(checkpoint, **kwargs)
            instance.fail_warmup = self.fail_warmup
            self.instances.append(instance)
            return instance

    return Factory()


@pytest.fixture
def client(factory):
    with TestClient(create_app(classifier_factory=factory)) as test_client:
        yield test_client


def test_health_and_empty_frame_do_not_load_classifier(client, factory):
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/v1/amc/infer", files=multipart())
    assert response.status_code == 200
    assert response.json() == {"frame_id": "frame_0001", "results": []}
    assert factory.instances == []


def test_ready_loads_and_warms_once_then_inference_reuses_model(client, factory):
    assert client.get("/v1/amc/ready").json() == {"status": "ready", "device": "cpu", "classes": ["AM", "QPSK"]}
    assert client.get("/v1/amc/ready").status_code == 200
    response = client.post("/v1/amc/infer", files=multipart(signal_parts()))
    assert response.status_code == 200
    assert response.json() == {"frame_id": "frame_0001", "results": [{"signal_id": "signal_0", "modulation_type": "QPSK"}]}
    assert len(factory.instances) == 1
    assert factory.instances[0].warmups == 1
    assert factory.instances[0].kwargs == {"device": "auto", "window_batch_size": 64, "max_windows": 5}


def test_internal_diagnostics_use_uvicorn_logger_and_stay_out_of_response(client, caplog):
    caplog.set_level(logging.INFO, logger="uvicorn.error.amc")
    response = client.post("/v1/amc/infer", files=multipart(signal_parts()))
    assert response.status_code == 200
    records = [record for record in caplog.records if record.name == "uvicorn.error.amc"]
    assert any("internal-only" in record.getMessage() for record in records)
    assert "internal-only" not in response.text


def test_install_amc_preserves_existing_detector_routes_and_body_handling(factory):
    application = FastAPI(title="Existing detector")

    @application.get("/health")
    def original_health():
        return {"status": "ok", "source": "original"}

    @application.get("/ready")
    def original_ready():
        return {"status": "ready", "model": "original-detector"}

    @application.post("/v1/infer")
    async def original_infer(request: Request):
        return {"original_bytes": len(await request.body())}

    original_routes = tuple(application.routes)
    install_amc(application, classifier_factory=factory, max_request_bytes=1000)
    assert tuple(application.routes[:len(original_routes)]) == original_routes
    with TestClient(application) as client:
        assert client.get("/health").json() == {"status": "ok", "source": "original"}
        assert client.get("/ready").json()["model"] == "original-detector"
        # The AMC body cap must not change the existing detector's input limit.
        assert client.post("/v1/infer", content=b"x" * 2000).json() == {"original_bytes": 2000}
        assert client.post("/v1/amc/infer", files=multipart(signal_parts())).status_code == 413
        assert client.post("/v1/amc/infer", files=multipart()).json()["results"] == []
        assert factory.instances == []
        assert client.get("/v1/amc/ready").json()["status"] == "ready"
        assert len(factory.instances) == 1


def test_installed_amc_uses_the_same_batch_contract(factory):
    application = FastAPI()
    install_amc(application, classifier_factory=factory)
    with TestClient(application) as client:
        response = client.post("/v1/amc/infer", files=multipart(
            signal_parts(0) + signal_parts(1, count=1024)
        ))
        assert response.status_code == 200
        assert response.json() == {"frame_id": "frame_0001", "results": [
            {"signal_id": "signal_0", "modulation_type": "QPSK"},
            {"signal_id": "signal_1", "modulation_type": None},
        ]}


def test_batch_preserves_signal_order_and_little_endian_iq(client, factory):
    iq = np.arange(4096, dtype="<f4").reshape(2048, 2)
    parts = signal_parts(1, count=50) + signal_parts(0, iq=iq.tobytes())
    response = client.post("/v1/amc/infer", files=multipart(parts))
    assert response.status_code == 200
    assert response.json()["results"] == [
        {"signal_id": "signal_0", "modulation_type": "QPSK"},
        {"signal_id": "signal_1", "modulation_type": None},
    ]
    np.testing.assert_array_equal(factory.instances[0].calls[0][1][0].iq, iq)
    assert "internal-only" not in response.text


def test_zero_length_and_zero_power_are_per_signal_null(client):
    parts = signal_parts(0, count=0) + signal_parts(1, iq=np.zeros((2048, 2), dtype="<f4").tobytes()) + signal_parts(2)
    response = client.post("/v1/amc/infer", files=multipart(parts))
    assert response.status_code == 200
    assert [r["modulation_type"] for r in response.json()["results"]] == [None, None, "QPSK"]


@pytest.mark.parametrize("meta", [
    "{", "[]", "null", '{"sample_rate_hz":NaN}',
    "[" * 1100 + "0" + "]" * 1100,
    '{"signal_id":"a","signal_id":"b"}',
    metadata(dtype="int16"), metadata(sample_count=True), metadata(sample_count=-1),
    metadata(sample_rate_hz=0), metadata(estimated_bandwidth_hz=90000),
    metadata(extra="not allowed"), metadata(signal_id=" "), metadata(signal_id="x" * 129),
    metadata(iq_start_time_s=-1), metadata(sample_rate_hz="100000"),
])
def test_invalid_metadata_rejected_before_model_load(client, factory, meta):
    response = client.post("/v1/amc/infer", files=multipart(signal_parts(meta=meta)))
    assert response.status_code == 400, response.text
    assert factory.instances == []


@pytest.mark.parametrize("raw", [b"", b"\x00", b"\x00" * 16383,
    np.full((2048, 2), np.nan, dtype="<f4").tobytes(),
    np.full((2048, 2), np.inf, dtype="<f4").tobytes(),
], ids=["empty", "one-byte", "wrong-length", "nan", "inf"])
def test_bad_iq_rejected_before_model_load(client, factory, raw):
    response = client.post("/v1/amc/infer", files=multipart(signal_parts(iq=raw)))
    assert response.status_code == 400
    assert factory.instances == []


def test_all_signals_validated_before_any_inference(client, factory):
    response = client.post("/v1/amc/infer", files=multipart(signal_parts(0) + signal_parts(1, iq=b"bad")))
    assert response.status_code == 400
    assert factory.instances == []


@pytest.mark.parametrize("parts", [
    signal_parts(0)[:1], signal_parts(0)[1:], signal_parts(1),
    signal_parts(0) + signal_parts(1, signal_id="signal_0"),
    signal_parts(0) + [signal_parts(0)[0]],
    [("iq_00", ("s.bin", b"")), ("metadata_00", (None, "{}"))],
    [("anything", (None, "{}"))],
    [("iq_0", (None, "not a file")), signal_parts(0)[1]],
    [signal_parts(0)[0], ("metadata_0", ("meta.json", json.dumps(metadata()), "application/json"))],
])
def test_invalid_pairing_rejected(client, factory, parts):
    response = client.post("/v1/amc/infer", files=multipart(parts))
    assert response.status_code == 400, response.text
    assert factory.instances == []


@pytest.mark.parametrize("frame_id", ["", " ", "x" * 129])
def test_bad_frame_id(client, factory, frame_id):
    assert client.post("/v1/amc/infer", files=multipart(frame_id=frame_id)).status_code == 400
    assert factory.instances == []


def test_missing_or_duplicate_frame_id(client):
    assert client.post("/v1/amc/infer", files=signal_parts()).status_code == 400
    assert client.post("/v1/amc/infer", files=multipart() + multipart()).status_code == 400


def test_max_signals_limit(factory):
    with TestClient(create_app(classifier_factory=factory, max_signals=1)) as client:
        response = client.post("/v1/amc/infer", files=multipart(signal_parts(0) + signal_parts(1)))
    assert response.status_code == 413
    assert factory.instances == []


def test_declared_body_limit(factory):
    with TestClient(create_app(classifier_factory=factory, max_request_bytes=100)) as client:
        response = client.post("/v1/amc/infer", files=multipart(signal_parts()))
    assert response.status_code == 413
    assert factory.instances == []


def test_actual_streamed_body_limit_without_content_length(factory):
    request = httpx.Request("POST", "http://test/v1/amc/infer", files=multipart(signal_parts()))
    raw = request.read()
    with TestClient(create_app(classifier_factory=factory, max_request_bytes=1000)) as client:
        response = client.post("/v1/amc/infer", content=iter([raw[:500], raw[500:]]),
                               headers={"content-type": request.headers["content-type"]})
    assert response.status_code == 413
    assert factory.instances == []


def test_metadata_part_limit(client, factory):
    response = client.post("/v1/amc/infer", files=multipart(signal_parts(meta=" " * 16385)))
    assert response.status_code == 413
    assert factory.instances == []


def test_json_or_compressed_request_is_rejected(client, factory):
    assert client.post("/v1/amc/infer", json={"frame_id": "f"}).status_code == 400
    assert client.post("/v1/amc/infer", files=multipart(), headers={"content-encoding": "gzip"}).status_code == 400
    assert factory.instances == []


def test_temp_files_close_after_validation_failure(client, monkeypatch):
    import starlette.formparsers
    original = starlette.formparsers.SpooledTemporaryFile
    handles = []

    def tracked(*args, **kwargs):
        handle = original(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", tracked)
    response = client.post("/v1/amc/infer", files=multipart(signal_parts(iq=b"wrong")))
    assert response.status_code == 400
    assert handles and all(handle.closed for handle in handles)


def test_truncated_multipart_rejected_and_files_closed(client, monkeypatch, factory):
    import starlette.formparsers
    original = starlette.formparsers.SpooledTemporaryFile
    handles = []

    def tracked(*args, **kwargs):
        handle = original(*args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", tracked)
    request = httpx.Request("POST", "http://test/v1/amc/infer", files=multipart(signal_parts()))
    raw = request.read()
    response = client.post("/v1/amc/infer", content=raw[:-100], headers={"content-type": request.headers["content-type"]})
    assert response.status_code == 400
    assert handles and all(handle.closed for handle in handles)
    assert factory.instances == []


def test_model_failure_returns_503_not_partial_success(client, factory):
    client.get("/v1/amc/ready")
    factory.instances[0].fail = True
    response = client.post("/v1/amc/infer", files=multipart(signal_parts()))
    assert response.status_code == 503
    assert "results" not in response.json()
    assert "internal model failure" not in response.text


def test_warmup_failure_can_be_retried(client, factory):
    factory.fail_warmup = True
    assert client.get("/v1/amc/ready").status_code == 503
    assert client.get("/health").status_code == 200
    factory.fail_warmup = False
    assert client.get("/v1/amc/ready").status_code == 200
    assert len(factory.instances) == 2


def test_runtime_serializes_concurrent_inference(client, factory):
    client.get("/v1/amc/ready")
    responses = []

    def invoke():
        responses.append(client.post("/v1/amc/infer", files=multipart(signal_parts())))

    threads = [threading.Thread(target=invoke) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(responses) == 4 and all(response.status_code == 200 for response in responses)
    assert factory.instances[0].max_active == 1
    assert len(factory.instances) == 1


def test_openapi_shows_multipart_contract(client):
    schema = client.get("/openapi.json").json()
    assert "multipart/form-data" in schema["paths"]["/v1/amc/infer"]["post"]["requestBody"]["content"]


def test_environment_configuration(monkeypatch, factory):
    monkeypatch.setenv("AMC_CHECKPOINT", "chosen.pt")
    monkeypatch.setenv("AMC_DEVICE", "cpu")
    monkeypatch.setenv("AMC_WINDOW_BATCH_SIZE", "7")
    monkeypatch.setenv("AMC_MAX_SIGNALS", "2")
    monkeypatch.setenv("AMC_MAX_REQUEST_BYTES", "100000")
    with TestClient(create_app(classifier_factory=factory)) as client:
        assert client.get("/v1/amc/ready").status_code == 200
    instance = factory.instances[0]
    assert str(instance.checkpoint) == "chosen.pt"
    assert instance.kwargs["device"] == "cpu"
    assert instance.kwargs["window_batch_size"] == 7
