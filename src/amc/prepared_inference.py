"""Batch classification of receiver-preprocessed IQ; no detector or DDC work.

IQ arrays contain real float32 [complex_sample, I_then_Q] rows. One frame can
contain different signal lengths and acquisition scales. Only 2048-point windows
are stacked; softmax scores are averaged within each signal, never across signals.
"""
from dataclasses import dataclass, fields
import math
import time
import numpy as np
import torch
from .inference import WindowClassifier


def validate_identifier(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError(f"{name} must be a nonempty string of at most 128 characters.")
    return value


def _positive_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number.")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number.") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number.")
    return result


@dataclass(frozen=True)
class SignalMetadata:
    signal_id: str
    dtype: str
    sample_rate_hz: float
    estimated_bandwidth_hz: float
    sample_count: int
    iq_start_time_s: float | None = None

    def __post_init__(self):
        validate_identifier(self.signal_id, "signal_id")
        if self.dtype != "float32":
            raise ValueError("dtype must be float32 (little-endian interleaved I,Q).")
        fs = _positive_number(self.sample_rate_hz, "sample_rate_hz")
        bandwidth = _positive_number(self.estimated_bandwidth_hz, "estimated_bandwidth_hz")
        if bandwidth >= fs / 1.2:
            raise ValueError("sample_rate_hz must exceed 1.2 * estimated_bandwidth_hz to retain the agreed filter margin.")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 0:
            raise ValueError("sample_count must be a nonnegative integer counting complex samples.")
        if self.iq_start_time_s is not None:
            value = self.iq_start_time_s
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("iq_start_time_s must be a finite nonnegative number.")
            try:
                valid = math.isfinite(float(value)) and value >= 0
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError("iq_start_time_s must be a finite nonnegative number.")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Each metadata part must be a JSON object.")
        allowed = {f.name for f in fields(cls)}
        required = allowed - {"iq_start_time_s"}
        missing, extra = required - set(value), set(value) - allowed
        if missing:
            raise ValueError(f"Missing metadata fields: {', '.join(sorted(missing))}.")
        if extra:
            raise ValueError(f"Unexpected metadata fields: {', '.join(sorted(extra))}.")
        return cls(**value)


@dataclass(frozen=True)
class PreparedSignal:
    metadata: SignalMetadata
    iq: np.ndarray

    def __post_init__(self):
        if not isinstance(self.metadata, SignalMetadata):
            raise ValueError("metadata must be validated SignalMetadata.")
        if (not isinstance(self.iq, np.ndarray) or self.iq.ndim != 2
                or self.iq.shape != (self.metadata.sample_count, 2)
                or self.iq.dtype.kind != "f" or self.iq.dtype.itemsize != 4):
            raise ValueError("IQ must be float32 [sample_count, 2] I/Q rows.")
        if not np.isfinite(self.iq).all():
            raise ValueError("IQ contains NaN or Inf.")


def validate_frame(frame_id, signals):
    validate_identifier(frame_id, "frame_id")
    if not isinstance(signals, (list, tuple)) or any(not isinstance(s, PreparedSignal) for s in signals):
        raise ValueError("signals must be a list of validated PreparedSignal objects.")
    ids = [s.metadata.signal_id for s in signals]
    if len(ids) != len(set(ids)):
        raise ValueError("signal_id must be unique within one frame request.")


def select_window_starts(length, window_samples=2048, max_windows=5):
    # Same fixed rule as long_inference.window_starts, without importing raw-IQ
    # filtering or synthesis modules into the classification service.
    if not isinstance(max_windows, int) or isinstance(max_windows, bool) or not 1 <= max_windows <= 5:
        raise ValueError("max_windows must be an integer in 1..5.")
    count = min(max_windows, length // window_samples)
    if not count:
        return []
    if count == 1:
        return [(length - window_samples) // 2]
    return np.linspace(0, length - window_samples, count).astype(int).tolist()


class PreparedIQBatchClassifier:
    def __init__(self, checkpoint, device="auto", window_batch_size=64, max_windows=5,
                 score_threshold=None):
        if isinstance(window_batch_size, bool) or not isinstance(window_batch_size, int) or window_batch_size < 1:
            raise ValueError("window_batch_size must be a positive integer.")
        select_window_starts(0, max_windows=max_windows)
        if score_threshold is not None:
            if (isinstance(score_threshold, bool) or not isinstance(score_threshold, (int, float))
                    or not math.isfinite(score_threshold) or not 0 <= score_threshold <= 1):
                raise ValueError("score_threshold must be a finite number in [0,1] or None.")
            score_threshold = float(score_threshold)
        # None preserves the unthresholded classifier used by historical replays.
        # The HTTP service explicitly supplies its configured rejection threshold.
        self.score_threshold = score_threshold
        self.window_classifier = WindowClassifier(
            checkpoint, device, **({"score_threshold": score_threshold} if score_threshold is not None else {}),
        )
        self.device = self.window_classifier.device
        self.classes = self.window_classifier.classes
        self.window_samples = self.window_classifier.window_samples
        self.window_batch_size = window_batch_size
        self.max_windows = max_windows

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def warmup(self):
        self.window_classifier.predict(np.ones((1, 2, self.window_samples), dtype=np.float32), self.window_batch_size)
        self._synchronize()

    def predict_frame(self, frame_id, signals):
        validate_frame(frame_id, signals)
        self._synchronize()
        started = time.perf_counter()
        output = {"frame_id": frame_id, "results": []}
        diagnostics, windows, offsets = [], [], []
        for signal in signals:
            metadata = signal.metadata
            result = {"signal_id": metadata.signal_id, "modulation_type": None}
            output["results"].append(result)
            starts = select_window_starts(metadata.sample_count, self.window_samples, self.max_windows)
            diagnostic = {"signal_id": metadata.signal_id, "sample_count": metadata.sample_count,
                          "sample_rate_hz": metadata.sample_rate_hz,
                          "estimated_bandwidth_hz": metadata.estimated_bandwidth_hz,
                          "window_count": len(starts), "window_starts": starts}
            diagnostics.append(diagnostic)
            if not starts:
                diagnostic.update(status="insufficient_length", reason="Fewer than 2048 valid complex samples.")
                offsets.append(None)
                continue
            selected = np.stack([signal.iq[s:s + self.window_samples].T for s in starts]).astype(np.float32, copy=False)
            power = np.mean(np.sum(selected.astype(np.float64) ** 2, axis=1), axis=1)
            if np.any(power <= 1e-12):
                diagnostic.update(status="zero_power", reason="At least one selected window has no usable power.")
                offsets.append(None)
                continue
            if np.any(np.sqrt(power) > np.finfo(np.float32).max):
                diagnostic.update(status="invalid_amplitude", reason="Selected-window RMS cannot be represented as float32.")
                offsets.append(None)
                continue
            begin = len(windows)
            windows.extend(selected)
            offsets.append((begin, len(windows)))
        selected_at = time.perf_counter()
        if windows:
            # Existing WindowClassifier performs shared-complex RMS normalization
            # and splits this stack into bounded forwards on the selected device.
            predictions = self.window_classifier.predict(np.stack(windows), self.window_batch_size)
            if len(predictions) != len(windows):
                raise RuntimeError("Classifier did not return one score vector per window.")
        else:
            predictions = []
        self._synchronize()
        inferred_at = time.perf_counter()
        for result, diagnostic, bounds in zip(output["results"], diagnostics, offsets):
            if bounds is None:
                continue
            selected_predictions = predictions[bounds[0]:bounds[1]]
            average = np.mean([[p["class_scores"][name] for name in self.classes] for p in selected_predictions], axis=0)
            if not np.isfinite(average).all():
                raise RuntimeError("Classifier returned nonfinite fused scores.")
            label = int(average.argmax())
            rejected = self.score_threshold is not None and average[label] < self.score_threshold
            result["modulation_type"] = "unknown" if rejected else self.classes[label]
            diagnostic.update(
                status="unknown" if rejected else (
                    "classified" if average[label] >= self.window_classifier.threshold else "uncertain"),
                class_name=self.classes[label], score=float(average[label]),
                class_scores=dict(zip(self.classes, average.tolist())),
                consistency=float(np.mean([p["class_id"] == label for p in selected_predictions])),
            )
        ended = time.perf_counter()
        details = {"frame_id": frame_id, "signal_count": len(signals),
                   "selected_window_count": sum(d["window_count"] for d in diagnostics),
                   "inferred_window_count": len(windows),
                   "network_batches": math.ceil(len(windows) / self.window_batch_size),
                   "signals": diagnostics,
                   "timing_ms": {"window_selection": (selected_at - started) * 1000,
                                 "normalization_and_inference": (inferred_at - selected_at) * 1000,
                                 "fusion": (ended - inferred_at) * 1000,
                                 "total_processing": (ended - started) * 1000}}
        return output, details
