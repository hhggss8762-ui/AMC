"""Independently recompute saved geometry, score fusion and source identities."""
import argparse
import json
from pathlib import Path
import numpy as np
from amc.config import load_config
from amc.station_capture import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    root = Path(args.report).resolve()
    load = lambda name: json.loads((root / name).read_text(encoding="utf-8"))
    source, summary = load("source.json"), load("summary.json")
    raw, enriched, diagnostics = load("detector_response.json"), load("response.json"), load("diagnostics.json")
    settings = load_config(Path(__file__).resolve().parents[1] / "configs/baseline.yaml")["preprocess"]
    recovered = json.loads(json.dumps(enriched))
    assert len(raw["detections"]) == len(enriched["detections"]) == len(diagnostics)
    cal = raw["calibration"]
    fs = source["sample_rate_hz"]
    duration = source["complex_samples"] / fs
    assert abs(duration - cal["time_window_ms"] / 1000) < 1e-9
    assert cal["sample_rate_hz"] == fs
    assert cal["freq_start_hz"] == source["center_freq_hz"] - fs / 2
    assert cal["freq_end_hz"] == source["center_freq_hz"] + fs / 2
    width, height = raw["image_width_px"], raw["image_height_px"]
    total_windows = 0
    max_rate_error = 0.
    for i, (before, after, diagnostic) in enumerate(zip(raw["detections"], recovered["detections"], diagnostics)):
        assert set(after) - set(before) == {"modulation_type", "modulation_score"}
        label, score = after.pop("modulation_type"), after.pop("modulation_score")
        assert before == after
        assert diagnostic["detection_index"] == i
        if label is None:
            assert score is None and diagnostic["status"] in ("unavailable", "insufficient_length")
            continue
        x1, y1, x2, y2 = before["xyxy"]
        start, end = np.clip([x1, x2], 0, width) / width * duration
        high, low = cal["freq_end_hz"] - np.clip([y1, y2], 0, height) / height * fs
        bounds = diagnostic["extraction"]
        assert np.allclose([start, end, low, high], [bounds["start_time_s"], bounds["end_time_s"], bounds["freq_low_hz"], bounds["freq_high_hz"]], rtol=0, atol=1e-8)
        assert abs((low + high) / 2 - source["center_freq_hz"]) + .6 * (high - low) < fs / 2
        pre = diagnostic["preprocessing"]
        assert pre["requested_output_sample_rate_hz"] == min(fs, settings["output_rate_over_estimated_bandwidth"] * (high - low))
        actual_rate = fs * pre["resample_up"] / pre["resample_down"]
        assert actual_rate == pre["output_sample_rate_hz"]
        assert pre["resample_down"] <= settings["resampling_max_denominator"] == 512
        max_rate_error = max(max_rate_error, pre["resampling_rate_relative_error"])
        classes = list(diagnostic["class_scores"])
        windows = diagnostic["windows"]
        assert len(windows) == diagnostic["window_count"] <= 5
        assert len(windows) == min(5, pre["box_valid_processed_samples"] // 2048)
        average = np.mean([[w["class_scores"][name] for name in classes] for w in windows], axis=0)
        assert np.allclose(average, list(diagnostic["class_scores"].values()), rtol=0, atol=1e-12)
        assert classes[int(average.argmax())] == label
        assert abs(float(max(average)) - score) < 1e-12
        previous_end = None
        for window in windows:
            assert window["capture_start_time_s"] >= start - 1e-9
            assert window["capture_end_time_s_exclusive"] <= end + 1e-9
            assert abs(window["capture_end_time_s_exclusive"] - window["capture_start_time_s"] - 2048 / actual_rate) < 1e-9
            if previous_end is not None:
                assert window["capture_start_time_s"] >= previous_end - 1e-9
            previous_end = window["capture_end_time_s_exclusive"]
        total_windows += len(windows)
    assert recovered == raw
    assert file_sha256(source["payload_file"]) == source["payload_sha256"]
    assert file_sha256(source["index_file"]) == source["index_sha256"]
    assert file_sha256(summary["checkpoint"]) == summary["checkpoint_sha256"] == "4f22f777fb859c406fab04b59cfdf67805c1145031ec34a3d10b5af3748d1da0"
    result = {"boxes": len(diagnostics), "selected_windows": total_windows,
              "original_response_preserved": True, "all_window_times_inside_original_boxes": True,
              "window_nonoverlap_checked": True, "averaged_scores_recomputed": True,
              "geometry_recomputed": True, "frozen_resampling_settings_checked": True,
              "max_actual_rate_relative_error": max_rate_error,
              "source_index_and_payload_unchanged": True, "checkpoint_unchanged": True,
              "no_ground_truth_accuracy_claim": True}
    (root / "result_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
