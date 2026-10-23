"""Physical sampled waveforms; no spectrogram-texture synthesis."""
from dataclasses import dataclass
import numpy as np
from scipy.signal import fftconvolve, firwin, hilbert, upfirdn
from .preprocess import band_power, preprocess_iq, rms_normalize, to_channels


@dataclass
class Waveform:
    iq: np.ndarray
    sample_rate_hz: float
    bandwidth_hz: float
    center_offset_hz: float
    parameters: dict


def _draw(rng, bounds, logarithmic=False):
    low, high = map(float, bounds)
    return float(np.exp(rng.uniform(np.log(low), np.log(high))) if logarithmic
                 else rng.uniform(low, high))


def rrc_taps(sps=8, beta=0.35, span_symbols=10):
    t = np.arange(-span_symbols * sps // 2, span_symbols * sps // 2 + 1) / sps
    result = np.empty_like(t)
    zero = np.isclose(t, 0)
    singular = np.isclose(np.abs(t), 1 / (4 * beta))
    regular = ~(zero | singular)
    u = t[regular]
    result[regular] = (
        np.sin(np.pi * u * (1 - beta)) + 4 * beta * u * np.cos(np.pi * u * (1 + beta))
    ) / (np.pi * u * (1 - (4 * beta * u) ** 2))
    result[zero] = 1 + beta * (4 / np.pi - 1)
    result[singular] = beta / np.sqrt(2) * (
        (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
    )
    return result / np.sqrt(np.sum(result ** 2))


def _message(rng, count, fs, bandwidth):
    taps_count = max(129, int(np.ceil(8 * fs / bandwidth)) | 1)
    taps = firwin(taps_count, bandwidth, fs=fs)
    message = fftconvolve(rng.standard_normal(count + taps_count - 1),
                          taps, mode="valid")
    message -= message.mean()
    return message / max(float(np.max(np.abs(message))), 1e-12)


def _ofdm_waveform(rng, settings):
    """Random payload CP-OFDM, with DC/guard bins empty; not a protocol transmitter."""
    fft_size = int(rng.choice(settings["fft_sizes"]))
    cp_samples = int(round(fft_size * float(rng.choice(settings["cp_fractions"]))))
    oversampling = int(settings["source_oversampling"])
    spacing = _draw(rng, settings["subcarrier_spacing_hz"], True)
    modulation = str(rng.choice(settings["subcarrier_modulations"]))
    half_active = int(np.floor(fft_size * settings["active_subcarrier_fraction"] / 2))
    active_bins = np.concatenate((np.arange(-half_active, 0), np.arange(1, half_active + 1)))
    active_count = len(active_bins)
    n_symbols = int(settings["source_ofdm_symbols"])
    shape = (n_symbols, active_count)
    if modulation == "QPSK":
        symbols = np.exp(2j * np.pi * rng.integers(0, 4, shape) / 4 + 1j * np.pi / 4)
    else:
        order = 16 if modulation == "16QAM" else 64
        side = int(np.sqrt(order))
        real = 2 * rng.integers(0, side, shape) - (side - 1)
        imag = 2 * rng.integers(0, side, shape) - (side - 1)
        symbols = (real + 1j * imag) / np.sqrt(2 * (order - 1) / 3)
    # Zero-pad the frequency grid, not the time samples: preserve orthogonality.
    source_fft = fft_size * oversampling
    grid = np.zeros((n_symbols, source_fft), dtype=np.complex128)
    grid[:, source_fft // 2 + active_bins] = symbols
    useful = np.fft.ifft(np.fft.ifftshift(grid, axes=1), axis=1)
    useful *= source_fft / np.sqrt(active_count)
    cp_source_samples = cp_samples * oversampling
    blocks = np.concatenate((useful[:, -cp_source_samples:], useful), axis=1)
    fs = source_fft * spacing
    # Nominal band extends half a subcarrier spacing beyond the edge carrier.
    bandwidth = (2 * half_active + 1) * spacing
    parameters = {
        "fft_size": fft_size, "cyclic_prefix_samples": cp_samples,
        "active_subcarriers": active_count, "dc_null": True,
        "subcarrier_modulation": modulation, "subcarrier_spacing_hz": spacing,
        "source_oversampling": oversampling, "source_ofdm_symbols": n_symbols,
        "ofdm_symbol_duration_s": (fft_size + cp_samples) / (fft_size * spacing),
        "protocol": None,
    }
    return Waveform(blocks.reshape(-1), fs, bandwidth, 0.0, parameters)


def generate_waveform(class_name, subtype, rng, config, *, analog_sample_count=None):
    s = config["synthesis"]
    if class_name == "OFDM":
        return _ofdm_waveform(rng, s["ofdm"])
    if class_name in ("AM", "FM"):
        message_bw = _draw(rng, s["message_bandwidth_hz"], True)
        if class_name == "AM":
            fs = 16 * message_bw
            message = _message(rng, 16384 if analog_sample_count is None else analog_sample_count,
                               fs, message_bw)
            parameters = {"message_bandwidth_hz": message_bw}
            if subtype == "DSB_LC":
                index = _draw(rng, s["am_index"])
                iq = (1 + index * message).astype(np.complex128)
                bandwidth, center = 2 * message_bw, 0.0
                parameters["am_index"] = index
            elif subtype == "SSB":
                side = str(rng.choice(["USB", "LSB"]))
                iq = hilbert(message)
                if side == "LSB":
                    iq = iq.conj()
                bandwidth = message_bw
                center = (1 if side == "USB" else -1) * message_bw / 2
            else:
                raise ValueError("Unknown AM subtype.")
            if subtype == "SSB":
                parameters["sideband"] = side
        else:
            deviation = message_bw * _draw(rng, s["fm_deviation_over_message_bandwidth"], True)
            bandwidth = 2 * (deviation + message_bw)  # nominal Carson bandwidth
            fs, center = 8 * bandwidth, 0.0
            message = _message(rng, 8192 if analog_sample_count is None else analog_sample_count,
                               fs, message_bw)
            iq = np.exp(2j * np.pi * np.cumsum(deviation * message) / fs)
            parameters = {"message_bandwidth_hz": message_bw, "peak_deviation_hz": deviation}
        return Waveform(iq, fs, bandwidth, center, parameters)

    rate = _draw(rng, s["symbol_rate_hz"], True)
    sps, count = int(s["samples_per_symbol"]), int(s["source_symbols"])
    fs = sps * rate
    parameters = {"symbol_rate_hz": rate, "source_samples_per_symbol": sps}
    if class_name in ("2FSK", "4FSK", "2GFSK"):
        order = 4 if class_name == "4FSK" else 2
        index = _draw(rng, s["fsk4_index"] if order == 4 else s["fsk2_index"])
        levels = 2 * rng.integers(0, order, size=count + 16) - (order - 1)
        drive = np.repeat(levels, sps).astype(np.float64)
        if class_name == "2GFSK":
            bt = _draw(rng, s["gfsk_bt"])
            t = np.arange(-4 * sps, 4 * sps + 1) / sps
            taps = np.exp(-2 * np.pi ** 2 * bt ** 2 * t ** 2 / np.log(2))
            drive = fftconvolve(drive, taps / taps.sum(), mode="same")
            parameters["gaussian_bt"] = bt
        drive = drive[8 * sps:(8 + count) * sps]
        frequency = drive * index * rate / 2
        iq = np.exp(2j * np.pi * np.cumsum(frequency) / fs)
        bandwidth = ((order - 1) * index + 1) * rate
        parameters.update({"modulation_index": index, "continuous_phase": True})
        return Waveform(iq, fs, bandwidth, 0.0, parameters)

    guard = 12
    n_symbols = count + 2 * guard
    if class_name == "BPSK":
        symbols = (2 * rng.integers(0, 2, n_symbols) - 1).astype(np.complex128)
    elif class_name in ("QPSK", "8PSK"):
        order = 4 if class_name == "QPSK" else 8
        symbols = np.exp(2j * np.pi * rng.integers(0, order, n_symbols) / order + 1j * np.pi / order)
    elif class_name == "PI4_DQPSK":
        increments = rng.choice(np.array([-3, -1, 1, 3]) * np.pi / 4, n_symbols)
        symbols = np.exp(1j * np.cumsum(increments))
    elif class_name in ("16QAM", "64QAM"):
        order = 16 if class_name == "16QAM" else 64
        side = int(np.sqrt(order))
        real = 2 * rng.integers(0, side, n_symbols) - (side - 1)
        imag = 2 * rng.integers(0, side, n_symbols) - (side - 1)
        symbols = (real + 1j * imag) / np.sqrt(2 * (order - 1) / 3)
    elif class_name == "ASK":
        low = 0.0 if subtype == "OOK" else _draw(rng, s["ask_low_to_high_amplitude"])
        symbols = np.where(rng.integers(0, 2, n_symbols), 1.0, low).astype(np.complex128)
        parameters["low_to_high_amplitude"] = low
    else:
        raise ValueError(f"Unsupported class: {class_name}")
    rolloff = float(s["rrc_rolloff"])
    taps = rrc_taps(sps, rolloff, int(s["rrc_span_symbols"]))
    shaped = upfirdn(taps, symbols, up=sps)
    first = guard * sps + (len(taps) - 1) // 2
    iq = shaped[first:first + count * sps]
    parameters["rrc_rolloff"] = rolloff
    return Waveform(iq, fs, (1 + rolloff) * rate, 0.0, parameters)


def generate_sample(class_name, subtype, snr_db, seed_words, config):
    """Return a normalized noisy IQ window, auditable metadata and clean/noise refs.

    SNR is scaled using power in the nominal true occupied band AFTER shared
    preprocessing and cropping, before normalization; total-band SNR differs.
    """
    rng = np.random.default_rng(np.random.SeedSequence(seed_words))
    waveform = generate_waveform(class_name, subtype, rng, config)
    phase = float(rng.uniform(-np.pi, np.pi))
    cfo = float(rng.uniform(-1, 1) * config["synthesis"]["residual_frequency_offset_fraction"]
                * waveform.bandwidth_hz)
    time = np.arange(len(waveform.iq)) / waveform.sample_rate_hz
    impaired = waveform.iq * np.exp(1j * (phase + 2 * np.pi * cfo * time))
    error = float(rng.uniform(-1, 1) * config["preprocess"]["bandwidth_estimate_relative_error"])
    estimated_bw = waveform.bandwidth_hz * (1 + error)
    args = (waveform.sample_rate_hz, estimated_bw, config["preprocess"])
    clean, preprocessing = preprocess_iq(impaired, *args,
                                         center_offset_hz=waveform.center_offset_hz)
    white = (rng.standard_normal(len(impaired)) + 1j * rng.standard_normal(len(impaired))) / np.sqrt(2)
    noise, _ = preprocess_iq(white, *args, center_offset_hz=waveform.center_offset_hz)
    length = int(config["dataset"]["window_samples"])
    if len(clean) < length:
        raise ValueError("Generated source does not contain a full valid window.")
    start = int(rng.integers(0, len(clean) - length + 1))
    clean, noise = clean[start:start + length], noise[start:start + length]
    fs_out = preprocessing["output_sample_rate_hz"]
    signal_power = band_power(clean, fs_out, waveform.bandwidth_hz)
    noise_power = band_power(noise, fs_out, waveform.bandwidth_hz)
    if signal_power <= 1e-12 or noise_power <= 1e-12:
        raise ValueError("Degenerate generated signal/noise power.")
    noise *= np.sqrt(signal_power / (noise_power * 10 ** (float(snr_db) / 10)))
    normalized, rms = rms_normalize(clean + noise)
    metadata = {
        "class_name": class_name, "subtype": subtype, "snr_db": float(snr_db),
        "seed_words": [int(v) for v in seed_words],
        "nominal_bandwidth_hz": waveform.bandwidth_hz,
        "source_center_offset_hz": waveform.center_offset_hz,
        "source_sample_count": len(waveform.iq),
        "source_parameters": waveform.parameters, "initial_phase_rad": phase,
        "residual_frequency_offset_hz": cfo, "bandwidth_estimate_relative_error": error,
        "crop_start_processed_sample": start, "window_samples": length,
        "window_duration_s": length / fs_out, "normalization_rms": rms,
        "measured_inband_snr_db": float(10 * np.log10(
            band_power(clean, fs_out, waveform.bandwidth_hz)
            / band_power(noise, fs_out, waveform.bandwidth_hz))),
        "snr_definition": "rectangular-window FFT power within nominal true band after preprocessing",
        **preprocessing,
    }
    reference = {"clean": to_channels(clean / rms), "noise": to_channels(noise / rms)}
    if class_name == "OFDM":
        metadata["window_ofdm_symbol_durations"] = (
            metadata["window_duration_s"] / waveform.parameters["ofdm_symbol_duration_s"]
        )
    return to_channels(normalized), metadata, reference
