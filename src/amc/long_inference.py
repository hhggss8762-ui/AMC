"""Continuous single-target IQ -> shared preprocessing -> window score average.

No detector coordinates or ground-truth labels are consumed by this module.
"""
import json
from pathlib import Path
import time
import numpy as np
import torch
from .inference import WindowClassifier
from .preprocess import preprocess_iq, to_channels


def window_starts(length, window_samples=2048, max_windows=5):
    if window_samples < 1 or not 1 <= max_windows <= 5 or length < 0:
        raise ValueError("Invalid length/window settings; max_windows must be 1..5.")
    count = min(max_windows, length // window_samples)
    if count == 0:
        return []
    if count == 1:
        return [(length - window_samples) // 2]
    return np.linspace(0, length - window_samples, count).astype(int).tolist()


def prepare_long_iq(raw_iq, sample_rate_hz, estimated_bandwidth_hz, settings,
                    center_offset_hz=0.0, effective_samples=None):
    raw = np.asarray(raw_iq)
    if not np.isfinite(center_offset_hz):
        raise ValueError("Center offset must be finite.")
    if raw.ndim == 2 and raw.shape[0] == 2 and not np.iscomplexobj(raw):
        raw = raw[0].astype(np.float64) + 1j * raw[1].astype(np.float64)
    if raw.ndim != 1 or not np.iscomplexobj(raw):
        raise ValueError("Raw IQ must be a complex [N] vector or real [2,N] I/Q array.")
    processed, metadata = preprocess_iq(
        raw, sample_rate_hz, estimated_bandwidth_hz, settings, center_offset_hz
    )
    available = len(processed)
    if effective_samples is not None:
        if not isinstance(effective_samples, (int, np.integer)) or effective_samples < 1:
            raise ValueError("effective_samples must be a positive integer or None.")
        if available < effective_samples:
            raise ValueError(f"Insufficient effective samples: {available} < {effective_samples}.")
        start = (available - effective_samples) // 2
        processed = processed[start:start + effective_samples]
    else:
        start = 0
    origin_s = ((metadata["filter_taps"] - 1) / (2 * sample_rate_hz)
                + (settings["edge_guard_output_samples"] + start)
                / metadata["output_sample_rate_hz"])
    metadata.update({"available_processed_samples": available,
                     "crop_start_processed_sample": start,
                     "effective_samples": len(processed),
                     "effective_duration_s": len(processed) / metadata["output_sample_rate_hz"],
                     "effective_origin_in_raw_s": origin_s})
    return processed, metadata


def load_long_input(path):
    """NPZ inputs intentionally contain acquisition parameters, not true labels."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Use .npz with raw_iq and input_metadata_json.")
    with np.load(path, allow_pickle=False) as data:
        raw = data["raw_iq"].copy()
        inputs = json.loads(str(data["input_metadata_json"]))
    required = {"sample_rate_hz", "estimated_bandwidth_hz", "center_offset_hz"}
    if not required.issubset(inputs) or set(inputs) - required - {"effective_samples"}:
        raise ValueError("Input metadata must contain acquisition parameters only.")
    return raw, inputs


class LongIQClassifier:
    def __init__(self, checkpoint, device="auto", score_threshold=0.6):
        self.window_classifier = WindowClassifier(checkpoint, device, score_threshold)
        self.classes = self.window_classifier.classes
        self.device = self.window_classifier.device
        self.window_samples = self.window_classifier.window_samples
        self.settings = self.window_classifier.checkpoint["config"]["preprocess"]

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def predict(self, raw_iq, sample_rate_hz, estimated_bandwidth_hz,
                center_offset_hz=0.0, effective_samples=None, max_windows=5,
                batch_size=64):
        self._synchronize()
        started = time.perf_counter()
        try:
            processed, preprocessing = prepare_long_iq(
                raw_iq, sample_rate_hz, estimated_bandwidth_hz, self.settings,
                center_offset_hz, effective_samples,
            )
        except ValueError as exc:
            if "too short" not in str(exc).lower():
                raise
            return {"status": "insufficient_length", "class_id": None,
                    "reason": str(exc), "windows": [], "window_count": 0}
        preprocessed_at = time.perf_counter()
        starts = window_starts(len(processed), self.window_samples, max_windows)
        if not starts:
            return {"status": "insufficient_length", "class_id": None,
                    "reason": "Fewer than 2048 valid output samples.", "windows": [],
                    "window_count": 0, "preprocessing": preprocessing}
        middle_start = (len(processed) - self.window_samples) // 2
        inference_starts = starts + ([middle_start] if middle_start not in starts else [])
        windows = np.stack([to_channels(processed[s:s + self.window_samples])
                            for s in inference_starts])
        selected_at = time.perf_counter()
        predictions = self.window_classifier.predict(windows, batch_size)
        self._synchronize()
        inferred_at = time.perf_counter()
        selected = predictions[:len(starts)]
        scores = np.array([[p["class_scores"][c] for c in self.classes] for p in selected])
        average = scores.mean(axis=0)
        class_id = int(average.argmax())
        fs = preprocessing["output_sample_rate_hz"]
        for start, prediction in zip(inference_starts, predictions):
            prediction.update({"start_sample": start,
                               "end_sample_exclusive": start + self.window_samples,
                               "start_time_s": start / fs})
        middle = predictions[inference_starts.index(middle_start)]
        result = {
            "class_id": class_id, "class_name": self.classes[class_id],
            "score": float(average[class_id]),
            "status": ("classified" if average[class_id] >= self.window_classifier.threshold
                       else "uncertain"),
            "class_scores": dict(zip(self.classes, average.tolist())),
            "aggregation": "unweighted_mean_of_all_selected_window_scores",
            "window_count": len(starts), "windows_inferred": len(inference_starts),
            "window_starts": starts, "windows": selected, "middle_window": middle,
            "consistency": float(np.mean([p["class_id"] == class_id for p in selected])),
            "score_threshold": self.window_classifier.threshold,
            "preprocessing": preprocessing,
        }
        ended = time.perf_counter()
        result["timing_ms"] = {
            "preprocessing": (preprocessed_at - started) * 1000,
            "window_selection": (selected_at - preprocessed_at) * 1000,
            "normalization_and_inference": (inferred_at - selected_at) * 1000,
            "aggregation": (ended - inferred_at) * 1000,
            "total_processing": (ended - started) * 1000,
        }
        return result
