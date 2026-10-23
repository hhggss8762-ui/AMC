from pathlib import Path
import json
import numpy as np
import pytest
import torch
from amc.config import load_config
from amc.long_dataset import generate_long_record, load_long_settings
from amc.long_inference import LongIQClassifier, load_long_input, prepare_long_iq, window_starts
from amc.preprocess import band_power

ROOT = Path(__file__).parents[1]
CONFIG = load_config(ROOT / "configs/baseline.yaml")
SETTINGS = load_long_settings(ROOT / "configs/long_signal_eval.yaml")
CASES = [(name, subtype) for name in CONFIG["classes"]
         for subtype in CONFIG["dataset"]["subtypes"].get(name, [name])]


@pytest.mark.parametrize("name,subtype", CASES)
@pytest.mark.parametrize("snr", [0, 30])
def test_continuous_sources_replay_with_global_snr_and_distinct_windows(name, subtype, snr):
    raw, metadata, refs = generate_long_record(
        name, subtype, snr, [91, 3, CASES.index((name, subtype)), snr], CONFIG, SETTINGS)
    replay, pre = prepare_long_iq(raw, settings=CONFIG["preprocess"], **metadata["input"])
    clean, _ = prepare_long_iq(refs["raw_clean"], settings=CONFIG["preprocess"], **metadata["input"])
    noise, _ = prepare_long_iq(refs["raw_noise"], settings=CONFIG["preprocess"], **metadata["input"])
    assert raw.dtype == np.complex64 and len(replay) == 16384
    assert pre == metadata["preprocessing"]
    assert np.allclose(replay, clean + noise, atol=2e-6, rtol=2e-6)
    measured = 10 * np.log10(band_power(clean, pre["output_sample_rate_hz"], metadata["nominal_bandwidth_hz"])
                             / band_power(noise, pre["output_sample_rate_hz"], metadata["nominal_bandwidth_hz"]))
    assert measured == pytest.approx(snr, abs=0.002)
    assert metadata["window_starts"] == [0, 3584, 7168, 10752, 14336]
    assert not np.allclose(replay[:2048], replay[3584:5632])
    assert np.ptp(metadata["window_measured_snr_db"]) > 1e-5


@pytest.mark.parametrize("length,count", [(0, 0), (2047, 0), (2048, 1), (4095, 1),
                                         (4096, 2), (8192, 4), (10240, 5), (16384, 5)])
def test_short_record_window_counts_and_nonoverlap(length, count):
    starts = window_starts(length)
    assert len(starts) == count
    assert all(0 <= s <= length - 2048 for s in starts)
    assert all(b - a >= 2048 for a, b in zip(starts, starts[1:]))
    if count == 1:
        assert starts == [(length - 2048) // 2]


class FixedScores:
    threshold = 0.6

    def predict(self, windows, batch_size):
        assert windows.shape == (5, 2, 2048)
        # Four weak A votes, one strong B: probability average must choose B.
        scores = [[.51, .49]] * 4 + [[.01, .99]]
        return [{"row": i, "class_id": int(np.argmax(s)), "class_name": "AB"[int(np.argmax(s))],
                 "score": max(s), "class_scores": dict(zip("AB", s))}
                for i, s in enumerate(scores)]


def test_long_pipeline_averages_scores_not_votes_and_has_exact_middle():
    classifier = LongIQClassifier.__new__(LongIQClassifier)
    classifier.device = torch.device("cpu")
    classifier.classes = ["A", "B"]
    classifier.window_samples = 2048
    classifier.settings = CONFIG["preprocess"]
    classifier.window_classifier = FixedScores()
    rng = np.random.default_rng(1)
    raw = (rng.standard_normal(40000) + 1j * rng.standard_normal(40000)).astype(np.complex64)
    result = classifier.predict(raw, 400000, 50000, effective_samples=16384)
    assert result["class_name"] == "B" and result["score"] == pytest.approx(.59)
    assert result["status"] == "uncertain"  # Still retains a top-1 class.
    assert result["consistency"] == .2
    assert result["middle_window"]["start_sample"] == 7168
    assert result["middle_window"]["class_name"] == "A"
    assert result["window_count"] == result["windows_inferred"] == 5
    assert result["timing_ms"]["total_processing"] > 0


def test_input_acquisition_metadata_excludes_true_labels(tmp_path):
    path = tmp_path / "raw.npz"
    inputs = {"sample_rate_hz": 100000, "estimated_bandwidth_hz": 10000, "center_offset_hz": 0}
    np.savez(path, raw_iq=np.ones(30000, dtype=np.complex64),
             input_metadata_json=np.array(json.dumps(inputs)))
    raw, loaded = load_long_input(path)
    assert loaded == inputs and len(raw) == 30000
    inputs["class_name"] = "AM"
    np.savez(path, raw_iq=raw, input_metadata_json=np.array(json.dumps(inputs)))
    with pytest.raises(ValueError, match="acquisition parameters only"):
        load_long_input(path)


def test_long_input_layout_and_insufficient_length_handling():
    t = np.arange(40000) / 400000
    x = np.exp(2j * np.pi * 3000 * t)
    a, _ = prepare_long_iq(x, 400000, 50000, CONFIG["preprocess"], effective_samples=16384)
    b, _ = prepare_long_iq(np.stack([x.real, x.imag]), 400000, 50000,
                           CONFIG["preprocess"], effective_samples=16384)
    assert np.allclose(a, b)
    with pytest.raises(ValueError, match="Insufficient effective samples"):
        prepare_long_iq(x[:5000], 400000, 50000, CONFIG["preprocess"], effective_samples=16384)
    with pytest.raises(ValueError, match="complex"):
        prepare_long_iq(x.real, 400000, 50000, CONFIG["preprocess"])
    with pytest.raises(ValueError, match="finite"):
        prepare_long_iq(x, 400000, 50000, CONFIG["preprocess"], center_offset_hz=float("nan"))


def test_insufficient_signal_returns_status_without_padding_or_model_call():
    classifier = LongIQClassifier.__new__(LongIQClassifier)
    classifier.device = torch.device("cpu")
    classifier.classes = ["A", "B"]
    classifier.window_samples = 2048
    classifier.settings = CONFIG["preprocess"]
    classifier.window_classifier = FixedScores()  # Would reject anything except five windows.
    raw = np.ones(3000, dtype=np.complex64)
    result = classifier.predict(raw, 400000, 50000)
    assert result["status"] == "insufficient_length" and result["class_id"] is None
    assert result["windows"] == []


def test_even_selected_windows_do_not_include_extra_middle_in_average():
    class ThreeScores:
        threshold = .6

        def predict(self, windows, batch_size):
            assert windows.shape == (3, 2, 2048)
            scores = [[.9, .1], [.9, .1], [.01, .99]]
            return [{"class_id": int(np.argmax(s)), "class_name": "AB"[int(np.argmax(s))],
                     "score": max(s), "class_scores": dict(zip("AB", s))} for s in scores]

    classifier = LongIQClassifier.__new__(LongIQClassifier)
    classifier.device = torch.device("cpu")
    classifier.classes = ["A", "B"]
    classifier.window_samples = 2048
    classifier.settings = CONFIG["preprocess"]
    classifier.window_classifier = ThreeScores()
    result = classifier.predict(np.ones(12000, dtype=np.complex64), 400000, 50000,
                                effective_samples=5000)
    assert result["window_count"] == 2 and result["windows_inferred"] == 3
    assert result["class_name"] == "A" and result["score"] == pytest.approx(.9)
    assert result["middle_window"]["class_name"] == "B"
