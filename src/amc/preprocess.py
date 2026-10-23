"""Shared baseband filtering, rate normalization and IQ window operations."""
from fractions import Fraction
import numpy as np
from scipy.signal import fftconvolve, firwin, kaiserord, resample_poly


def preprocess_iq(iq, sample_rate_hz, estimated_bandwidth_hz, settings,
                  center_offset_hz=0.0):
    """Process one continuous signal; output excludes FIR boundary transients.

    The caller must supply actual original samples beyond its desired window.
    This module currently prepares synthetic single signals, not detection boxes.
    """
    x = np.asarray(iq, dtype=np.complex128)
    fs, bandwidth = float(sample_rate_hz), float(estimated_bandwidth_hz)
    if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("IQ must be a nonempty finite complex vector.")
    if not np.isfinite(fs) or not np.isfinite(bandwidth) or not 0 < bandwidth < fs:
        raise ValueError("Need 0 < estimated bandwidth < input sample rate.")
    if abs(center_offset_hz) + 0.5 * bandwidth >= 0.5 * fs:
        raise ValueError("Target frequency band is not fully captured.")
    time = np.arange(len(x), dtype=np.float64) / fs
    shifted = x * np.exp(-2j * np.pi * center_offset_hz * time)
    pass_edge = (0.5 + settings["frequency_margin_each_side"]) * bandwidth
    stop_edge = min(pass_edge + 0.2 * bandwidth, 0.49 * fs)
    if stop_edge <= pass_edge:
        raise ValueError("Insufficient Nyquist margin for target filter.")
    taps_count, beta = kaiserord(
        settings["filter_attenuation_db"], (stop_edge - pass_edge) / (0.5 * fs)
    )
    taps_count = max(31, int(taps_count) | 1)
    if len(x) <= taps_count:
        raise ValueError("Source IQ is too short for filtering.")
    taps = firwin(taps_count, (pass_edge + stop_edge) / 2,
                  window=("kaiser", beta), fs=fs)
    filtered = fftconvolve(shifted, taps, mode="valid")
    requested_fs = min(fs, settings["output_rate_over_estimated_bandwidth"] * bandwidth)
    ratio = Fraction(requested_fs / fs).limit_denominator(
        settings["resampling_max_denominator"]
    )
    actual_fs = fs * ratio.numerator / ratio.denominator
    if actual_fs / 2 <= stop_edge:
        raise ValueError("Resampling would alias the selected frequency band.")
    result = resample_poly(filtered, ratio.numerator, ratio.denominator)
    guard = int(settings["edge_guard_output_samples"])
    if len(result) <= 2 * guard:
        raise ValueError("Source IQ is too short after resampling.")
    result = result[guard:-guard] if guard else result
    return result, {
        "input_sample_rate_hz": fs,
        "estimated_bandwidth_hz": bandwidth,
        "requested_output_sample_rate_hz": requested_fs,
        "output_sample_rate_hz": actual_fs,
        "resample_up": ratio.numerator,
        "resample_down": ratio.denominator,
        "filter_taps": taps_count,
        "filter_pass_edge_hz": pass_edge,
        "filter_stop_edge_hz": stop_edge,
        "removed_resampler_edge_samples_each_side": guard,
    }


def band_power(iq, sample_rate_hz, bandwidth_hz):
    """FFT-bin-integrated mean power in [-B/2, B/2], using a rectangular window."""
    x = np.asarray(iq)
    bins = np.fft.fftfreq(len(x), d=1 / sample_rate_hz)
    selected = np.abs(bins) <= bandwidth_hz / 2
    return float(np.sum(np.abs(np.fft.fft(x)[selected]) ** 2) / len(x) ** 2)


def rms_normalize(iq):
    x = np.asarray(iq)
    scale = float(np.sqrt(np.mean(np.abs(x) ** 2)))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("IQ has no usable power.")
    # Never independently scale I/Q or subtract the mean: AM carrier is useful.
    return (x / scale).astype(np.complex64), scale


def to_channels(iq):
    x = np.asarray(iq)
    return np.stack((x.real, x.imag)).astype(np.float32)
