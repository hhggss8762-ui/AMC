"""0.2.1 full-view pixel boxes -> original IQ -> frozen AMC classification.

The public response is a copy of the detector response with exactly two new
fields per detection. Geometry, extraction details and errors stay in diagnostics.
No detection threshold, crop enlargement, class mapping or model weight is tuned.
"""
from copy import deepcopy
from fractions import Fraction
import time
import numpy as np
from scipy.signal import kaiserord
from .long_inference import LongIQClassifier, prepare_long_iq, window_starts
from .preprocess import to_channels


def validate_calibration(response, sample_count, sample_rate_hz, center_freq_hz):
    fs, center = float(sample_rate_hz), float(center_freq_hz)
    if not np.isfinite([fs, center]).all() or fs <= 0 or sample_count < 1:
        raise ValueError("Invalid original IQ acquisition parameters.")
    width, height = float(response["image_width_px"]), float(response["image_height_px"])
    cal = response["calibration"]
    duration = float(cal["time_window_ms"]) / 1000
    low, high = float(cal["freq_start_hz"]), float(cal["freq_end_hz"])
    if not np.isfinite([width, height, duration, low, high]).all():
        raise ValueError("Missing/nonfinite detector calibration.")
    if width <= 0 or height <= 0 or high <= low or duration <= 0:
        raise ValueError("Invalid detector calibration extent.")
    if abs(duration - sample_count / fs) > max(1 / fs, 1e-9):
        raise ValueError("Detector duration does not match this complete IQ payload.")
    if abs(low - (center - fs / 2)) > 1 or abs(high - (center + fs / 2)) > 1:
        raise ValueError("Detector frequency calibration does not match the original IQ passband.")
    for key, expected in (("img_width_px", width), ("img_height_px", height)):
        if key in cal and float(cal[key]) != expected:
            raise ValueError("Inconsistent detector image/calibration dimensions.")
    if "sample_rate_hz" in cal and not np.isclose(float(cal["sample_rate_hz"]), fs, rtol=0, atol=1e-6):
        raise ValueError("Detector sample rate does not match IQ.")
    if not isinstance(response["detections"], list):
        raise ValueError("detections must be a list.")
    return width, height, duration, low, high


def box_to_iq_bounds(xyxy, calibration, sample_count, sample_rate_hz, center_freq_hz):
    width, height, duration, low, high = calibration
    box = np.asarray(xyxy, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Box must have four finite xyxy coordinates.")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Box has reversed or empty bounds.")
    x1, y1, x2, y2 = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Box does not overlap the capture.")
    start_s, end_s = x1 / width * duration, x2 / width * duration
    f_low, f_high = high - y2 / height * (high - low), high - y1 / height * (high - low)
    # Cover the requested interval on the input grid; output centers are trimmed
    # to exact box times after filtering, so rounding never selects outside IQ.
    start = max(0, int(np.floor(start_s * sample_rate_hz)))
    end = min(sample_count, int(np.ceil(end_s * sample_rate_hz)))
    return {
        "clipped_xyxy": [float(x1), float(y1), float(x2), float(y2)],
        "start_time_s": start_s, "end_time_s": end_s,
        "start_sample": start, "end_sample_exclusive": end,
        "freq_low_hz": f_low, "freq_high_hz": f_high,
        "estimated_bandwidth_hz": f_high - f_low,
        "center_offset_hz": (f_low + f_high) / 2 - center_freq_hz,
    }


def extraction_context_samples(fs, bandwidth, settings):
    """Match the frozen FIR and rational resampler, without changing settings."""
    if not 0 < bandwidth < fs:
        raise ValueError("Need 0 < estimated bandwidth < input sample rate.")
    pass_edge = (0.5 + settings["frequency_margin_each_side"]) * bandwidth
    stop_edge = min(pass_edge + 0.2 * bandwidth, 0.49 * fs)
    if stop_edge <= pass_edge:
        raise ValueError("Insufficient Nyquist margin for target filter.")
    count, _ = kaiserord(settings["filter_attenuation_db"], (stop_edge - pass_edge) / (0.5 * fs))
    count = max(31, int(count) | 1)
    requested = min(fs, settings["output_rate_over_estimated_bandwidth"] * bandwidth)
    ratio = Fraction(requested / fs).limit_denominator(settings["resampling_max_denominator"])
    if ratio.numerator == 0:
        raise ValueError("Current denominator limit cannot represent the requested output rate.")
    actual = fs * ratio.numerator / ratio.denominator
    if actual / 2 <= stop_edge:
        raise ValueError("Resampling would alias the selected frequency band.")
    # FIR group delay plus existing resampler guard, with two output samples
    # for integer rounding. Source edges are clipped, never zero padded here.
    context = int(np.ceil((count - 1) / 2 + (settings["edge_guard_output_samples"] + 2) * fs / actual))
    return context


class DetectionIQAdapter:
    def __init__(self, checkpoint, device="auto", score_threshold=0.6, max_windows=5):
        if not 1 <= max_windows <= 5:
            raise ValueError("max_windows must be 1..5.")
        self.classifier = LongIQClassifier(checkpoint, device, score_threshold)
        self.max_windows = max_windows

    def classify_box(self, iq, bounds, fs):
        c = self.classifier
        c._synchronize()
        started = time.perf_counter()
        bandwidth, offset = bounds["estimated_bandwidth_hz"], bounds["center_offset_hz"]
        if abs(offset) + bandwidth / 2 >= fs / 2:
            raise ValueError("Target band reaches or crosses the IQ Nyquist boundary.")
        if abs(offset) + (0.5 + c.settings["frequency_margin_each_side"]) * bandwidth >= fs / 2:
            raise ValueError("Required frequency margin is not fully captured; do not wrap the opposite IQ edge.")
        context = extraction_context_samples(fs, bandwidth, c.settings)
        source_start = max(0, bounds["start_sample"] - context)
        source_end = min(len(iq), bounds["end_sample_exclusive"] + context)
        processed, pre = prepare_long_iq(
            iq[source_start:source_end], fs, bandwidth, c.settings, center_offset_hz=offset)
        output_fs = pre["output_sample_rate_hz"]
        origin = source_start / fs + pre["effective_origin_in_raw_s"]
        # Keep complete output sample intervals within the ORIGINAL box times.
        left = max(0, int(np.ceil((bounds["start_time_s"] - origin) * output_fs - 1e-8)))
        right = min(len(processed), int(np.floor((bounds["end_time_s"] - origin) * output_fs + 1e-8)))
        processed = processed[left:max(left, right)]
        valid_origin = origin + left / output_fs
        starts = window_starts(len(processed), c.window_samples, self.max_windows)
        pre.update({"source_start_sample": source_start, "source_end_sample_exclusive": source_end,
                    "requested_context_samples_each_side": context,
                    "box_valid_processed_samples": len(processed),
                    "box_valid_origin_in_capture_s": valid_origin,
                    "resampling_rate_relative_error": abs(output_fs - pre["requested_output_sample_rate_hz"]) / pre["requested_output_sample_rate_hz"]})
        if not starts:
            return {"status": "insufficient_length", "reason": "Fewer than 2048 valid target samples.",
                    "window_count": 0, "preprocessing": pre,
                    "timing_ms": {"total_processing": (time.perf_counter() - started) * 1000}}
        prepared = time.perf_counter()
        windows = np.stack([to_channels(processed[s:s + c.window_samples]) for s in starts])
        predictions = c.window_classifier.predict(windows)
        c._synchronize()
        inferred = time.perf_counter()
        average = np.mean([[p["class_scores"][name] for name in c.classes] for p in predictions], axis=0)
        label = int(np.argmax(average))
        for s, prediction in zip(starts, predictions):
            prediction.update({"start_processed_sample": s,
                               "capture_start_time_s": valid_origin + s / output_fs,
                               "capture_end_time_s_exclusive": valid_origin + (s + c.window_samples) / output_fs})
        return {
            "class_name": c.classes[label], "score": float(average[label]),
            "status": "classified" if average[label] >= c.window_classifier.threshold else "uncertain",
            "window_count": len(starts), "windows": predictions,
            "class_scores": dict(zip(c.classes, average.tolist())),
            "consistency": float(np.mean([p["class_id"] == label for p in predictions])),
            "preprocessing": pre,
            "timing_ms": {"preprocessing": (prepared - started) * 1000,
                          "window_selection_normalization_and_inference": (inferred - prepared) * 1000,
                          "total_processing": (time.perf_counter() - started) * 1000},
        }

    def enrich(self, iq, sample_rate_hz, center_freq_hz, detector_response):
        iq = np.asarray(iq)
        if iq.ndim != 1 or not np.iscomplexobj(iq) or not np.isfinite(iq).all():
            raise ValueError("Original IQ must be a finite complex [N] vector.")
        fs, center = float(sample_rate_hz), float(center_freq_hz)
        calibration = validate_calibration(detector_response, len(iq), fs, center)
        response = deepcopy(detector_response)
        diagnostics = []
        for i, detection in enumerate(response["detections"]):
            detection.update(modulation_type=None, modulation_score=None)
            diagnostic = {"detection_index": i}
            try:
                bounds = box_to_iq_bounds(detection["xyxy"], calibration, len(iq), fs, center)
                diagnostic["extraction"] = bounds
                result = self.classify_box(iq, bounds, fs)
                diagnostic.update(result)
                if result["status"] in ("classified", "uncertain"):
                    detection.update(modulation_type=result["class_name"], modulation_score=result["score"])
            except ValueError as exc:
                diagnostic.update(status="unavailable", reason=str(exc))
            diagnostics.append(diagnostic)
        return response, diagnostics
