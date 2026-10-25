from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
from amc.prepared_inference import (
    PreparedIQBatchClassifier, PreparedSignal, SignalMetadata, select_window_starts,
)
from amc.long_inference import window_starts


def metadata(signal_id="s0", count=12000, **kwargs):
    data = {"signal_id": signal_id, "dtype": "float32", "sample_rate_hz": 100000.,
            "estimated_bandwidth_hz": 25000., "sample_count": count}
    data.update(kwargs)
    return SignalMetadata.from_dict(data)


@pytest.mark.parametrize("length", [0, 1, 2047, 2048, 4095, 4096, 8192, 10239, 10240, 50000])
def test_prepared_windows_match_existing_baseline(length):
    starts = select_window_starts(length)
    assert starts == window_starts(length)
    assert all(b - a >= 2048 for a, b in zip(starts, starts[1:]))


@pytest.mark.parametrize("change", [
    {"signal_id": ""}, {"signal_id": " "}, {"signal_id": 123}, {"dtype": "int16"},
    {"sample_rate_hz": True}, {"sample_rate_hz": "100000"},
    {"sample_rate_hz": float("nan")}, {"sample_rate_hz": float("inf")},
    {"sample_rate_hz": 0}, {"sample_rate_hz": 30000},
    {"estimated_bandwidth_hz": -1}, {"estimated_bandwidth_hz": False},
    {"sample_count": True}, {"sample_count": -1}, {"sample_count": 2048.0},
    {"iq_start_time_s": -1}, {"iq_start_time_s": "0"}, {"iq_start_time_s": float("inf")},
    {"center_freq_hz": 1000000},
])
def test_metadata_rejects_invalid_contract(change):
    with pytest.raises(ValueError):
        metadata(**change)


def test_metadata_missing_fields_and_source_capped_output_rate():
    with pytest.raises(ValueError, match="Missing"):
        SignalMetadata.from_dict({"signal_id": "a"})
    with pytest.raises(ValueError, match="JSON object"):
        SignalMetadata.from_dict([])
    # When the original source Fs limits min(Fs,4B), Fs_out/B need not equal4.
    assert metadata(sample_rate_hz=60000).sample_rate_hz == 60000


def test_prepared_layout_count_and_finiteness():
    values = np.ones((2048, 2), dtype=np.float32)
    PreparedSignal(metadata(count=2048), values)
    for bad in [values.T, values.astype(np.float64), values[:2000], values.astype(np.complex64)]:
        with pytest.raises(ValueError, match="float32"):
            PreparedSignal(metadata(count=2048), bad)
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        PreparedSignal(metadata(count=2048), values)


class FixedWindowScores:
    threshold = .6

    def predict(self, windows, batch_size):
        self.received = windows.copy()
        self.batch_size = batch_size
        scores = [[.51, .49]] * 4 + [[.01, .99]] + [[.9, .1]] * 2
        assert len(windows) == len(scores)
        return [{"class_id": int(np.argmax(s)), "class_scores": dict(zip(["A", "B"], s))} for s in scores]


def fake_classifier():
    c = PreparedIQBatchClassifier.__new__(PreparedIQBatchClassifier)
    c.device = torch.device("cpu")
    c.classes, c.window_samples, c.window_batch_size, c.max_windows = ["A", "B"], 2048, 4, 5
    c.window_classifier = FixedWindowScores()
    c.score_threshold = None
    return c


def test_cross_signal_batch_keeps_mapping_and_averages_scores_not_votes():
    c = fake_classifier()
    # Nonzero carrier/DC and unequal I/Q amplitudes must reach WindowClassifier
    # unchanged; the service must neither shift/filter nor remove the mean.
    a = np.stack([np.arange(12000) + 10, np.arange(12000) * .3 + 2], axis=1).astype(np.float32)
    b = np.full((4096, 2), [3, 7], dtype=np.float32)
    before = a.copy()
    signals = [PreparedSignal(metadata("A", len(a)), a),
               PreparedSignal(metadata("B", len(b)), b),
               PreparedSignal(metadata("short", 2047), np.ones((2047, 2), dtype=np.float32)),
               PreparedSignal(metadata("zero", 2048), np.zeros((2048, 2), dtype=np.float32))]
    output, detail = c.predict_frame("frame", signals)
    assert output == {"frame_id": "frame", "results": [
        {"signal_id": "A", "modulation_type": "B"},
        {"signal_id": "B", "modulation_type": "A"},
        {"signal_id": "short", "modulation_type": None},
        {"signal_id": "zero", "modulation_type": None}]}
    assert detail["signals"][0]["score"] == pytest.approx(.59)
    assert detail["signals"][0]["status"] == "uncertain"  # Still returns top1.
    assert detail["signals"][1]["score"] == pytest.approx(.9)
    assert detail["inferred_window_count"] == 7 and detail["network_batches"] == 2
    assert c.window_classifier.batch_size == 4
    assert np.array_equal(c.window_classifier.received[0], a[:2048].T)
    assert np.array_equal(c.window_classifier.received[5], b[:2048].T)
    assert np.array_equal(a, before)


def test_null_empty_frame_and_duplicate_identifiers():
    c = fake_classifier()
    output, detail = c.predict_frame("empty", [])
    assert output == {"frame_id": "empty", "results": []}
    assert detail["network_batches"] == detail["inferred_window_count"] == 0
    x = PreparedSignal(metadata(count=0), np.empty((0, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="unique"):
        c.predict_frame("frame", [x, x])
    with pytest.raises(ValueError, match="frame_id"):
        c.predict_frame("", [])


def test_unknown_is_decided_after_fusion_and_preserves_null():
    c = fake_classifier()
    c.score_threshold = .6
    a = np.ones((12000, 2), dtype=np.float32)
    b = np.ones((4096, 2), dtype=np.float32)
    output, details = c.predict_frame("f", [
        PreparedSignal(metadata("a", len(a)), a),
        PreparedSignal(metadata("b", len(b)), b),
        PreparedSignal(metadata("short", 1), a[:1]),
        PreparedSignal(metadata("zero", 2048), np.zeros((2048, 2), np.float32)),
    ])
    assert [r["modulation_type"] for r in output["results"]] == ["unknown", "A", None, None]
    assert details["signals"][0]["status"] == "unknown"
    assert details["signals"][0]["class_name"] == "B"
    assert details["signals"][0]["score"] == pytest.approx(.59)


def test_unknown_threshold_accepts_exact_boundary():
    c = fake_classifier()
    c.score_threshold = .5

    class Scores:
        threshold = .5

        def predict(self, windows, batch_size):
            return [{"class_id": 0, "class_scores": dict(A=.5, B=.5)}]

    c.window_classifier = Scores()
    output, _ = c.predict_frame("f", [PreparedSignal(metadata(count=2048), np.ones((2048, 2), np.float32))])
    assert output["results"][0]["modulation_type"] == "A"


@pytest.mark.parametrize("value", [-.1, 1.1, True, float("nan"), float("inf")])
def test_invalid_unknown_threshold_rejected_before_loading(value):
    with pytest.raises(ValueError, match="score_threshold"):
        PreparedIQBatchClassifier("nonexistent.pt", score_threshold=value)


def test_mixed_zero_and_nonzero_windows_do_not_abort_other_signals():
    c = fake_classifier()
    x = np.ones((4096, 2), dtype=np.float32)
    x[:2048] = 0
    output, detail = c.predict_frame("f", [PreparedSignal(metadata(count=len(x)), x)])
    assert output["results"][0]["modulation_type"] is None
    assert detail["signals"][0]["status"] == "zero_power"
    assert detail["inferred_window_count"] == 0


def test_checkpoint_batch_split_matches_individual_window_reference(tmp_path):
    from amc.config import load_config
    from amc.model import build_model

    torch.set_num_threads(2)
    # Exercise real network batching without requiring a distributed weight file.
    config = load_config(Path(__file__).parents[1] / "configs/baseline.yaml")
    torch.manual_seed(92)
    model = build_model(config).eval()
    checkpoint = tmp_path / "batch_reference.pt"
    torch.save({"format_version": 1, "epoch": 0, "classes": config["classes"],
                "config": config, "model_state": model.state_dict()}, checkpoint)
    c = PreparedIQBatchClassifier(checkpoint, "cpu", window_batch_size=2)
    rng = np.random.default_rng(92)
    signals = [PreparedSignal(metadata(f"s{i}", n), rng.standard_normal((n, 2)).astype(np.float32))
               for i, n in enumerate([12000, 5000, 2048])]
    before = [s.iq.copy() for s in signals]
    output, details = c.predict_frame("reference", signals)
    assert details["inferred_window_count"] == 8 and details["network_batches"] == 4
    for result, diagnostic, signal, original in zip(output["results"], details["signals"], signals, before):
        windows = np.stack([signal.iq[s:s+2048].T for s in window_starts(len(signal.iq))])
        individual = c.window_classifier.predict(windows, batch_size=1)
        avg = np.mean([[p["class_scores"][name] for name in c.classes] for p in individual], axis=0)
        assert result["modulation_type"] == c.classes[int(avg.argmax())]
        assert np.allclose(avg, list(diagnostic["class_scores"].values()), rtol=0, atol=2e-6)
        assert np.array_equal(signal.iq, original)
        assert set(result) == {"signal_id", "modulation_type"}
