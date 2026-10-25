"""Versioned 20-label synthesis, independent of the deployed baseline generator.

Protocol-style OFDM is a structural approximation, not a standards transmitter.
Unspread random-payload BPSK/DBPSK and QPSK/DQPSK are statistically overlapping;
their encoding provenance is retained without inventing label-specific artifacts.
"""
import hashlib
import numpy as np
from scipy.signal import fftconvolve, upfirdn

from .synthesis import Waveform, generate_waveform, rrc_taps, _draw
from .preprocess import preprocess_iq, band_power, rms_normalize, to_channels


CLASSES20 = ("AM", "FM", "ASK", "BPSK", "QPSK", "8PSK", "PI4_DQPSK",
             "16QAM", "64QAM", "2FSK", "4FSK", "2GFSK", "OFDM", "OQPSK",
             "DBPSK", "DQPSK", "MSK", "GMSK", "32QAM", "256QAM")
BARKER11 = np.array([1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1])


def constellation(name):
    if name in ("BPSK", "QPSK"):
        m = 2 if name == "BPSK" else 4
        return np.exp(2j * np.pi * np.arange(m) / m + (0 if m == 2 else 1j*np.pi/4))
    m = int(name.replace("QAM", ""))
    side = 6 if m == 32 else int(np.sqrt(m))
    v = np.arange(1-side, side, 2)
    points = (v[:, None] + 1j*v[None, :]).reshape(-1)
    if m == 32:
        points = points[~((np.abs(points.real) == 5) & (np.abs(points.imag) == 5))]
    if len(points) != m:
        raise ValueError("Invalid constellation order")
    return points / np.sqrt(np.mean(np.abs(points)**2))


def _new_digital(name, subtype, rng, config):
    s = config["synthesis"]
    rate = _draw(rng, s["symbol_rate_hz"], True)
    sps, count, guard = int(s["samples_per_symbol"]), int(s["source_symbols"]), 16
    n = count + 2*guard
    p = {"symbol_rate_hz": rate, "source_samples_per_symbol": sps}
    if name in ("MSK", "GMSK"):
        drive = np.repeat(2*rng.integers(0, 2, n)-1, sps).astype(float)
        if name == "GMSK":
            bt = _draw(rng, s["gmsk_bt"])
            t = np.arange(-4*sps, 4*sps+1)/sps
            h = np.exp(-2*np.pi**2*bt**2*t**2/np.log(2))
            drive = fftconvolve(drive, h/h.sum(), mode="same")
            p["gaussian_bt"] = bt
        drive = drive[guard*sps:(guard+count)*sps]
        iq = np.exp(2j*np.pi*np.cumsum(drive*0.5*rate/2)/(sps*rate))
        p.update(modulation_index=0.5, continuous_phase=True)
        return Waveform(iq, sps*rate, 1.5*rate, 0., p)
    if name == "DBPSK":
        increments = np.pi*rng.integers(0, 2, n)
        symbols = np.exp(1j*np.cumsum(increments))
        p["differential_phase_alphabet_rad"] = [0, float(np.pi)]
        if subtype == "DIFFERENTIAL_BARKER11":
            symbols = (symbols[:, None]*BARKER11).reshape(-1)
            # Rate denotes information-symbol rate; pulse shaping is at chip rate.
            p.update(spreading_code=BARKER11.tolist(), spreading_factor=11,
                     chip_rate_hz=rate*11, protocol=None)
            rate *= 11
            count *= 11
            guard *= 11
    elif name == "DQPSK":
        increments = np.pi/2*rng.integers(0, 4, n)
        symbols = np.exp(1j*np.cumsum(increments))
        p["differential_phase_alphabet_rad"] = [0, float(np.pi/2), float(np.pi), float(3*np.pi/2)]
    elif name == "OQPSK":
        symbols = ((2*rng.integers(0, 2, n)-1) + 1j*(2*rng.integers(0, 2, n)-1))/np.sqrt(2)
    else:
        points = constellation(name)
        symbols = points[rng.integers(0, len(points), n)]
        p["constellation"] = "cross32" if name == "32QAM" else "square256"
    beta = float(s["rrc_rolloff"])
    taps = rrc_taps(sps, beta, int(s["rrc_span_symbols"]))
    if name == "OQPSK":
        a = upfirdn(taps, symbols.real, up=sps)
        b = upfirdn(taps, symbols.imag, up=sps)
        delay = sps//2
        shaped = np.pad(a, (0, delay)) + 1j*np.pad(b, (delay, 0))
        p["quadrature_delay_symbols"] = .5
    else:
        shaped = upfirdn(taps, symbols, up=sps)
    first = guard*sps + (len(taps)-1)//2
    iq = shaped[first:first+count*sps]
    p["rrc_rolloff"] = beta
    return Waveform(iq, sps*rate, (1+beta)*rate, 0., p)


def _allocation(rng, available, load, mode):
    k = len(available)
    n = max(16, int(round(load*k/12))*12)
    n = min(n, k)
    if mode == "contiguous":
        first = int(rng.integers(0, k-n+1))
        return available[first:first+n]
    blocks = np.array_split(available, max(1, k//12))
    wanted = max(1, int(np.ceil(n/12)))
    ids = rng.choice(len(blocks), min(len(blocks), wanted), replace=False)
    return np.sort(np.concatenate([blocks[i] for i in ids]))


def generate_ofdm(rng, group, config):
    s = config["synthesis"]["ofdm"]
    g = s["groups"][group]
    nfft = int(rng.choice(g["fft_sizes"]))
    cp_fraction = float(rng.choice(g["cp_fractions"]))
    os = 4 if nfft <= 512 else int(rng.choice([2, 4]))
    spacing = float(rng.choice(s["subcarrier_spacings_hz"]))
    modulation = str(rng.choice(s["subcarrier_modulations"]))
    if group == "wlan_style":
        fraction = float(rng.choice([.75, .8125, .875]))
        load, layout, pilot_pattern = 1., "contiguous", "fixed_every_12"
    elif group == "lte_style":
        fraction = float(rng.choice([.5859375, .75, .87890625]))
        load = float(rng.choice([.08, .15, .3, .6, 1.]))
        layout, pilot_pattern = str(rng.choice(["contiguous", "blocks"])), "comb6_symbols0_4"
    elif group == "sparse_pilot":
        fraction = float(rng.choice([.5, .75, .875]))
        load = float(rng.uniform(.05, .3))
        layout, pilot_pattern = "blocks", str(rng.choice(["comb6_symbols0_4", "comb6_all_symbols"]))
    elif group == "multiscale":
        fraction = float(rng.choice([.5, .75, .875]))
        load = float(rng.choice([.15, .35, .7, 1.]))
        layout, pilot_pattern = str(rng.choice(["contiguous", "blocks"])), "comb6_symbols0_4"
    else:
        fraction = float(rng.uniform(.5, .9))
        load = float(rng.choice([.3, .6, 1.]))
        layout, pilot_pattern = str(rng.choice(["contiguous", "blocks"])), "fixed_every_12"
    half = max(20, int(nfft*fraction/2))
    available = np.r_[np.arange(-half, 0), np.arange(1, half+1)]
    source_fft = nfft*os
    # Enough complete source symbols for filtering and an arbitrary 2048-point crop.
    nsym = max(16, int(np.ceil(4096/(2*half)))+8)
    grid = np.zeros((nsym, source_fft), dtype=np.complex128)
    points = constellation(modulation)
    initial_slot = int(rng.integers(0, 20))
    cell = int(rng.integers(0, 504))
    pilot_gain_db = float(rng.uniform(-1, 6))
    burst = bool(rng.random() < .3)
    gate = np.ones(nsym, bool)
    if burst:
        # Keep several successive active OFDM symbols, with explicit idle intervals.
        period = int(rng.integers(7, 15))
        on = int(rng.integers(max(4, period//2), period))
        gate = (np.arange(nsym) + int(rng.integers(period))) % period < on
    loads, pilot_counts, data_counts = [], [], []
    last_data = None
    training_bins = available[available % 4 == 0]
    training_values = rng.choice([-1., 1.], size=len(training_bins))
    for l in range(nsym):
        if not gate[l]:
            loads.append(0.); pilot_counts.append(0); data_counts.append(0)
            continue
        if last_data is None or l % 7 == 0:
            last_data = _allocation(rng, available, load, layout)
        data = last_data
        pilots = np.array([], dtype=int)
        if pilot_pattern == "fixed_every_12":
            pilots = available[(available-cell) % 12 == 0]
        elif pilot_pattern == "comb6_all_symbols" or l % 7 in (0, 4):
            shift = (cell + (3 if l % 7 == 0 else 0)) % 6
            pilots = available[(available-shift) % 6 == 0]
        grid[l, (data % source_fft)] = points[rng.integers(len(points), size=len(data))]
        if len(pilots):
            q = constellation("QPSK")
            grid[l, pilots % source_fft] = q[rng.integers(4, size=len(pilots))]*10**(pilot_gain_db/20)
        if group == "wlan_style" and l in (0, 1):
            # Repeated sparse training symbols with no claim of an IEEE-valid preamble.
            grid[l] = 0
            grid[l, training_bins % source_fft] = training_values
        used = np.flatnonzero(np.abs(grid[l]) > 0)
        loads.append(len(used)/len(available))
        pilot_counts.append(len(pilots)); data_counts.append(len(data))
    useful = np.fft.ifft(grid, axis=1)*source_fft/np.sqrt(len(available))
    cp = np.full(nsym, max(1, int(round(cp_fraction*nfft)))*os, dtype=int)
    variable_cp = bool(group in ("lte_style", "multiscale") and np.isclose(cp_fraction, .0703125))
    if variable_cp:
        cp[np.arange(nsym) % 7 == 0] = max(1, int(round(nfft*.078125)))*os
    blocks = [np.r_[row[-c:], row] for row, c in zip(useful, cp)]
    iq = np.concatenate(blocks)
    fs, parent_bw = source_fft*spacing, (2*half+1)*spacing
    bw, center = parent_bw, 0.
    retained = 1.
    if group == "partial_band":
        retained = float(rng.uniform(*s["partial_band_fraction"]))
        bw = parent_bw*retained
        center = float(rng.uniform(-1, 1)*(parent_bw-bw)/2)
    params = {
        "ofdm_group": group, "protocol": None, "protocol_compliant": False,
        "reference_sequence": "random QPSK; structural reference pattern only",
        "fft_size": nfft, "source_fft_samples": source_fft, "source_oversampling": os,
        "subcarrier_spacing_hz": spacing, "subcarrier_modulation": modulation,
        "cp_fraction": cp_fraction, "cp_samples_per_symbol": cp.tolist(),
        "variable_cyclic_prefix": variable_cp, "source_ofdm_symbols": nsym,
        "usable_subcarriers": len(available), "parent_bandwidth_hz": parent_bw,
        "data_resource_load": load, "allocation_layout": layout,
        "pilot_pattern": pilot_pattern, "pilot_gain_db": pilot_gain_db,
        "pilot_counts_per_symbol": pilot_counts, "data_counts_per_symbol": data_counts,
        "occupied_resource_fraction_per_symbol": loads,
        "initial_slot": initial_slot, "cell_pattern_id": cell,
        "burst": burst, "active_symbol_mask": gate.tolist(),
        "partial_band_fraction": retained, "partial_band_center_hz": center,
        "useful_symbol_duration_s": 1/spacing,
    }
    return Waveform(iq, fs, bw, center, params)


def _channel(x, waveform, name, rng, config):
    c = config["channel"]
    kind = str(rng.choice(c["kinds"], p=c["probabilities"]))
    p = {"kind": kind}
    if kind == "flat_fading":
        h = (rng.normal()+1j*rng.normal())/np.sqrt(2)
        x = x*h
        p["flat_gain"] = [float(h.real), float(h.imag)]
    elif kind == "multipath":
        if name == "OFDM":
            max_delay = max(1, int(.8*min(waveform.parameters["cp_samples_per_symbol"])))
        else:
            max_delay = max(1, int(config["synthesis"]["samples_per_symbol"]*.5))
        delays = np.unique(np.r_[0, rng.integers(1, max_delay+1, size=int(rng.integers(1, 4)))])
        gains = (rng.normal(size=len(delays))+1j*rng.normal(size=len(delays)))*np.exp(-delays/max_delay)
        gains /= np.sqrt(np.sum(np.abs(gains)**2))
        y = np.zeros_like(x)
        for d, h in zip(delays, gains):
            if d:
                y[d:] += h*x[:-d]
            else:
                y += h*x
        x = y
        p.update(delays_samples=delays.tolist(), gains=[[float(v.real), float(v.imag)] for v in gains])
    if rng.random() < c["iq_imbalance_probability"]:
        db = float(rng.uniform(*c["iq_gain_error_db"]))
        phase = float(np.deg2rad(rng.uniform(*c["iq_phase_error_degrees"])))
        x = x.real + 1j*10**(db/20)*(x.imag*np.cos(phase)+x.real*np.sin(phase))
        p.update(iq_gain_error_db=db, iq_phase_error_rad=phase)
    rms = float(np.sqrt(np.mean(np.abs(x)**2)))
    if rng.random() < c["dc_probability"]:
        a = float(rng.uniform(*c["dc_amplitude_over_rms"]))
        z = a*rms*np.exp(1j*rng.uniform(-np.pi, np.pi))
        x = x+z
        p["dc_over_rms"] = a
    if rng.random() < c["clipping_probability"]:
        threshold = float(rng.uniform(*c["clipping_rms"]))
        x = x*np.minimum(1, threshold*rms/np.maximum(np.abs(x), 1e-15))
        p["clipping_rms"] = threshold
    return x, p


def generate_sample20(name, subtype, snr, seed_words, config):
    rng = np.random.default_rng(np.random.SeedSequence(seed_words))
    local = {**config, "synthesis": {**config["synthesis"]}}
    local["synthesis"]["rrc_rolloff"] = float(rng.uniform(*config["synthesis"]["rrc_rolloff_range"]))
    if name == "OFDM":
        wave = generate_ofdm(rng, subtype, local)
    elif name in ("OQPSK", "DBPSK", "DQPSK", "MSK", "GMSK", "32QAM", "256QAM"):
        wave = _new_digital(name, subtype, rng, local)
    else:
        wave = generate_waveform(name, subtype, rng, local)
    phase = float(rng.uniform(-np.pi, np.pi))
    cfo = (float(rng.uniform(-1, 1))*config["channel"]["ofdm_max_cfo_over_spacing"]*
           wave.parameters["subcarrier_spacing_hz"] if name == "OFDM" else
           float(rng.uniform(-1, 1))*config["synthesis"]["residual_frequency_offset_fraction"]*wave.bandwidth_hz)
    # Local source time resets only once per independent generated waveform.
    time = np.arange(len(wave.iq))/wave.sample_rate_hz
    impaired = wave.iq*np.exp(1j*(phase+2*np.pi*cfo*time))
    impaired, channel = _channel(impaired, wave, name, rng, config)
    bw_error = float(rng.uniform(-1, 1)*config["preprocess"]["bandwidth_estimate_relative_error"])
    estimated_bw = wave.bandwidth_hz*(1+bw_error)
    center_error = float(rng.uniform(-1, 1)*config["channel"]["center_error_over_bandwidth"]*wave.bandwidth_hz)
    args = (wave.sample_rate_hz, estimated_bw, config["preprocess"])
    center = wave.center_offset_hz+center_error
    clean, prep = preprocess_iq(impaired, *args, center_offset_hz=center)
    white = (rng.normal(size=len(impaired))+1j*rng.normal(size=len(impaired)))/np.sqrt(2)
    noise, _ = preprocess_iq(white, *args, center_offset_hz=center)
    length = int(config["dataset"]["window_samples"])
    if len(clean) < length:
        raise ValueError(f"Source too short: {name}/{subtype}: {len(clean)}")
    fs = prep["output_sample_rate_hz"]
    full_power = float(np.mean(np.abs(clean)**2))
    for attempt in range(40):
        start = int(rng.integers(0, len(clean)-length+1))
        target = clean[start:start+length]
        signal_power = band_power(target, fs, wave.bandwidth_hz)
        if signal_power > max(1e-12, .02*full_power):
            break
    else:
        raise ValueError(f"No active target window: {name}/{subtype}")
    noise = noise[start:start+length]
    noise *= np.sqrt(signal_power/(band_power(noise, fs, wave.bandwidth_hz)*10**(snr/10)))
    iq, scale = rms_normalize(target+noise)
    measured = float(10*np.log10(signal_power/band_power(noise, fs, wave.bandwidth_hz)))
    group = ("BPSK_DBPSK_random_payload" if name in ("BPSK", "DBPSK") and subtype != "DIFFERENTIAL_BARKER11"
             else "QPSK_DQPSK_random_payload" if name in ("QPSK", "DQPSK") else None)
    meta = {
        "class_name": name, "subtype": subtype, "snr_db": float(snr),
        "seed_words": list(map(int, seed_words)), "window_samples": length,
        "nominal_bandwidth_hz": wave.bandwidth_hz, "source_center_offset_hz": wave.center_offset_hz,
        "source_sample_count": len(wave.iq), "source_parameters": wave.parameters,
        "initial_phase_rad": phase, "residual_frequency_offset_hz": cfo,
        "center_estimate_error_hz": center_error, "bandwidth_estimate_relative_error": bw_error,
        "channel": channel, "crop_start_processed_sample": start, "crop_attempts": attempt+1,
        "normalization_rms": scale, "window_duration_s": length/fs,
        "measured_inband_snr_db": measured,
        "snr_definition": "FFT power inside nominal selected band after preprocessing and crop, before normalization",
        "statistically_overlapping_label_group": group, **prep,
    }
    if name == "OFDM":
        p = wave.parameters
        meta.update(window_useful_symbol_durations=length/fs*p["subcarrier_spacing_hz"],
                    equivalent_useful_symbol_output_samples=fs/p["subcarrier_spacing_hz"])
    channels = to_channels(iq)
    meta["iq_blake2b128"] = hashlib.blake2b(channels.tobytes(), digest_size=16).hexdigest()
    ref = {"clean": to_channels(target/scale), "noise": to_channels(noise/scale)}
    return channels, meta, ref
