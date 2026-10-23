from copy import deepcopy
import struct
import numpy as np
import pytest
import torch
from amc.config import load_config
from amc.detector_adapter import DetectionIQAdapter, box_to_iq_bounds, validate_calibration
from amc.long_inference import LongIQClassifier
from amc.station_capture import load_station_capture
from pathlib import Path


SETTINGS = load_config(Path(__file__).parents[1] / "configs/baseline.yaml")["preprocess"]


def response(n=100000, fs=100000, center=1000000):
    return {"model": "signal-localization-beta", "class_names": {"0": "SIGNAL"},
            "image_width_px": 560, "image_height_px": 560,
            "calibration": {"time_window_ms": n / fs * 1000,
                            "freq_start_hz": center - fs / 2,
                            "freq_end_hz": center + fs / 2},
            "detections": [{"xyxy": [112, 168, 448, 224],
                            "class_id": 0, "class_name": "SIGNAL", "confidence": .95}]}


class RecordingClassifier:
    threshold = .6

    def predict(self, windows):
        self.windows = windows
        # Four weak A votes and one strong B; score mean must select B.
        scores = [[.51, .49]] * (len(windows) - 1) + [[.01, .99]]
        return [{"class_id": int(np.argmax(s)), "class_name": "AB"[int(np.argmax(s))],
                 "score": max(s), "class_scores": dict(zip("AB", s))} for s in scores]


def adapter():
    a = DetectionIQAdapter.__new__(DetectionIQAdapter)
    c = LongIQClassifier.__new__(LongIQClassifier)
    c.device = torch.device("cpu")
    c.settings = SETTINGS
    c.classes = ["A", "B"]
    c.window_samples = 2048
    c.window_classifier = RecordingClassifier()
    a.classifier = c
    a.max_windows = 5
    return a


def test_mapping_frequency_direction_and_time_are_full_capture():
    r = response()
    cal = validate_calibration(r, 100000, 100000, 1000000)
    b = box_to_iq_bounds(r["detections"][0]["xyxy"], cal, 100000, 100000, 1000000)
    assert b["start_time_s"] == pytest.approx(.2)
    assert b["end_time_s"] == pytest.approx(.8)
    assert b["start_sample"] == 20000
    assert b["end_sample_exclusive"] == 80000
    assert b["freq_low_hz"] == 1010000
    assert b["freq_high_hz"] == 1020000
    assert b["center_offset_hz"] == 15000


def test_context_is_excluded_score_mean_and_detector_response_is_preserved():
    a, r = adapter(), response()
    original = deepcopy(r)
    t = np.arange(100000) / 100000
    # Target at +15 kHz becomes DC; stronger neighbour at -20 kHz is rejected.
    iq = np.exp(2j * np.pi * 15000 * t) + 5 * np.exp(-2j * np.pi * 20000 * t)
    enriched, diagnostics = a.enrich(iq, 100000, 1000000, r)
    d = diagnostics[0]
    assert d["window_count"] == 5
    assert d["class_name"] == "B" and d["score"] == pytest.approx(.59)
    assert d["status"] == "uncertain"
    assert d["preprocessing"]["source_start_sample"] < 20000
    assert d["preprocessing"]["source_end_sample_exclusive"] > 80000
    for w in d["windows"]:
        assert .2 - 1e-10 <= w["capture_start_time_s"]
        assert w["capture_end_time_s_exclusive"] <= .8 + 1e-10
    z = a.classifier.window_classifier.windows[:, 0] + 1j * a.classifier.window_classifier.windows[:, 1]
    # The mixer starts at the extracted segment, so a constant phase rotation
    # is expected. Check DC/coherence and rejected neighbour, not absolute phase.
    assert np.max(np.abs(np.abs(z.mean(axis=1)) - 1)) < .002
    assert np.max(np.abs(z - z.mean(axis=1, keepdims=True))) < .002
    assert r == original
    for row in enriched["detections"]:
        assert row.pop("modulation_type") == "B"
        assert row.pop("modulation_score") == pytest.approx(.59)
    assert enriched == original


@pytest.mark.parametrize("box", [[-5, 168, 570, 224], [1, 168, 1.1, 224],
                                 [0, 0, 560, 560], [10, 10, 5, 20],
                                 [0, 0, 10, float("nan")], [0, 1, 560, 56]])
def test_short_edge_and_invalid_boxes_keep_all_detections(box):
    r, a = response(), adapter()
    r["detections"].append({"xyxy": box, "class_name": "SIGNAL"})
    x = np.exp(2j * np.pi * 15000 * np.arange(100000) / 100000)
    enriched, diagnostics = a.enrich(x, 100000, 1000000, r)
    assert len(enriched["detections"]) == len(diagnostics) == 2
    assert enriched["detections"][0]["modulation_type"] == "B"
    if box[0] != -5:
        assert enriched["detections"][1]["modulation_type"] is None
        assert diagnostics[1]["status"] in ("unavailable", "insufficient_length")


def test_calibration_mismatch_rejects_wrong_iq_instead_of_guessing():
    r, a = response(), adapter()
    x = np.ones(100000, dtype=np.complex64)
    with pytest.raises(ValueError, match="duration"):
        a.enrich(x[:-100], 100000, 1000000, r)
    with pytest.raises(ValueError, match="frequency"):
        a.enrich(x, 100000, 1000100, r)
    r["detections"] = []
    enriched, diagnostics = a.enrich(x, 100000, 1000000, r)
    assert enriched == r and diagnostics == []


def test_fractional_time_bounds_keep_complete_windows_inside_the_box():
    r, a = response(), adapter()
    r["detections"][0]["xyxy"] = [112.1234, 168, 448.7891, 224]
    x = np.exp(2j * np.pi * 15000 * np.arange(100000) / 100000)
    _, diagnostics = a.enrich(x, 100000, 1000000, r)
    d = diagnostics[0]
    assert d["window_count"] == 5
    for w in d["windows"]:
        assert w["capture_start_time_s"] >= d["extraction"]["start_time_s"] - 1e-10
        assert w["capture_end_time_s_exclusive"] <= d["extraction"]["end_time_s"] + 1e-10


def make_capture(path):
    payload = np.arange(40, dtype="<i2").tobytes()
    index = bytearray(100)
    for j in range(2):
        pos = j * 50
        struct.pack_into("<qqq", index, pos, 123456 + j, j + 1, j * 40)
        struct.pack_into("<dffi", index, pos + 25, 863512500., 250000., 320000., 10)
    (path / "Iq_test.ix").write_bytes(index)
    (path / "Iq_test.md").write_bytes(payload)
    return index


def test_station_reader_uses_all_samples_and_checks_every_record(tmp_path):
    index = make_capture(tmp_path)
    iq, metadata, info = load_station_capture(tmp_path)
    assert len(iq) == 20 and iq[-1] == 38 + 39j
    assert metadata["sample_rate_hz"] == 320000
    assert metadata["center_freq_hz"] == 863512500
    assert info["duration_s"] == 20 / 320000
    assert (tmp_path / "Iq_test.ix").read_bytes() == index
    struct.pack_into("<q", index, 50 + 16, 44)
    (tmp_path / "Iq_test.ix").write_bytes(index)
    with pytest.raises(ValueError, match="Non-contiguous"):
        load_station_capture(tmp_path)


def test_station_reader_rejects_changed_parameters_and_payload_size(tmp_path):
    index = make_capture(tmp_path)
    struct.pack_into("<f", index, 50 + 37, 640000.)
    (tmp_path / "Iq_test.ix").write_bytes(index)
    with pytest.raises(ValueError, match="change"):
        load_station_capture(tmp_path)
    make_capture(tmp_path)
    (tmp_path / "Iq_test.md").write_bytes(b"short")
    with pytest.raises(ValueError, match="size"):
        load_station_capture(tmp_path)
