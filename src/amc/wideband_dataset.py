"""Fresh payloads at fixed nominal bandwidths, with real acquisition resampling."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
import yaml
from .dataset import write_json
from .long_dataset import generate_long_record, sha256_file
from .synthesis import Waveform, generate_waveform


def load_wideband_settings(path):
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if settings["seed_namespace"] != 4 or settings["max_windows"] != 5:
        raise ValueError("Bandwidth sweep fixes seed namespace 4 and five windows.")
    count = settings["per_class_per_snr"]
    if not isinstance(count, int) or count < 2 or count % 2:
        raise ValueError("Counts must be positive even integers.")
    if settings["effective_samples"] < 2048 * settings["max_windows"]:
        raise ValueError("Insufficient effective length.")
    if (not settings["bandwidth_hz"] or len(set(settings["bandwidth_hz"])) != len(settings["bandwidth_hz"])
            or min(settings["bandwidth_hz"]) <= 0):
        raise ValueError("Need unique positive nominal bandwidths.")
    if min(settings["acquisition_rate_over_bandwidth"]) < 5:
        raise ValueError("This sweep tests capture rates >=5B, not undersampled acquisition.")
    return settings


def generate_bandwidth_waveform(name, subtype, rng, config, settings, bandwidth_hz, acquisition_ratio):
    """Change symbol/message rates in the physical generator, then sample the capture.

    Relative modulation parameters are kept within the trained family. This is
    time-scale coverage, not new protocols or additional channel impairments.
    """
    if acquisition_ratio not in settings["acquisition_rate_over_bandwidth"]:
        raise ValueError("Unexpected acquisition ratio.")
    cfg = deepcopy(config)
    s = cfg["synthesis"]
    bw = float(bandwidth_hz)
    if not np.isfinite(bw) or bw <= 0:
        raise ValueError("Bandwidth must be finite and positive.")
    sizes = settings["source_lengths"]
    s["source_symbols"] = sizes["digital_symbols"]
    analog_count = None
    if name == "AM":
        message_bw = bw / 2 if subtype == "DSB_LC" else bw
        s["message_bandwidth_hz"] = [message_bw, message_bw]
        analog_count = sizes["am_samples"]
    elif name == "FM":
        low, high = s["fm_deviation_over_message_bandwidth"]
        ratio = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        message_bw = bw / (2 * (ratio + 1))
        s["message_bandwidth_hz"] = [message_bw, message_bw]
        s["fm_deviation_over_message_bandwidth"] = [ratio, ratio]
        analog_count = sizes["fm_samples"]
    elif name == "OFDM":
        fft_size = int(rng.choice(s["ofdm"]["fft_sizes"]))
        half_active = int(np.floor(fft_size * s["ofdm"]["active_subcarrier_fraction"] / 2))
        spacing = bw / (2 * half_active + 1)
        s["ofdm"]["fft_sizes"] = [fft_size]
        s["ofdm"]["subcarrier_spacing_hz"] = [spacing, spacing]
        s["ofdm"]["source_ofdm_symbols"] = sizes["ofdm_symbols"]
    else:
        if name in ("2FSK", "4FSK", "2GFSK"):
            key = "fsk4_index" if name == "4FSK" else "fsk2_index"
            index = float(rng.uniform(*s[key]))
            s[key] = [index, index]
            order = 4 if name == "4FSK" else 2
            rate = bw / (1 + (order - 1) * index)
        else:
            rate = bw / (1 + s["rrc_rolloff"])
        s["symbol_rate_hz"] = [rate, rate]
    source = generate_waveform(name, subtype, rng, cfg, analog_sample_count=analog_count)
    if not np.isclose(source.bandwidth_hz, bw, rtol=2e-12, atol=1e-6):
        raise AssertionError("Physical generator did not produce the requested nominal bandwidth.")
    # Center SSB before the acquisition sample-rate conversion.
    time = np.arange(len(source.iq)) / source.sample_rate_hz
    centered = source.iq * np.exp(-2j * np.pi * source.center_offset_hz * time)
    requested_fs = float(acquisition_ratio * bw)
    denominator = int(config["preprocess"]["resampling_max_denominator"])
    ratio = Fraction(requested_fs / source.sample_rate_hz).limit_denominator(denominator)
    if ratio.numerator <= 0:
        raise ValueError("Acquisition resampling ratio rounded to zero.")
    actual_fs = source.sample_rate_hz * ratio.numerator / ratio.denominator
    sampled = resample_poly(centered, ratio.numerator, ratio.denominator)
    # Exercise nonzero target locations, within a safe known capture band.
    center = float(rng.uniform(-.1, .1) * actual_fs)
    sampled *= np.exp(2j * np.pi * center * np.arange(len(sampled)) / actual_fs)
    parameters = dict(source.parameters)
    parameters.update({
        "bandwidth_sweep_target_hz": bw,
        "generator_sample_rate_hz": source.sample_rate_hz,
        "generator_sample_count": len(source.iq),
        "generator_center_offset_hz": source.center_offset_hz,
        "acquisition_requested_rate_over_bandwidth": float(acquisition_ratio),
        "acquisition_requested_sample_rate_hz": requested_fs,
        "acquisition_resample_up": ratio.numerator,
        "acquisition_resample_down": ratio.denominator,
        "acquisition_sample_rate_hz": actual_fs,
        "acquisition_relative_rate_error": abs(actual_fs / requested_fs - 1),
        "acquisition_target_center_offset_hz": center,
    })
    return Waveform(sampled, actual_fs, source.bandwidth_hz, center, parameters)


def generate_wideband_record(name, subtype, snr_db, seed_words, config, settings,
                             bandwidth_hz, acquisition_ratio):
    return generate_long_record(
        name, subtype, snr_db, seed_words, config, settings,
        waveform_factory=lambda rng: generate_bandwidth_waveform(
            name, subtype, rng, config, settings, bandwidth_hz, acquisition_ratio),
    )


def create_wideband_dataset(config, settings, output, profile="full"):
    if config["preprocess"]["resampling_max_denominator"] != 512:
        raise ValueError("Keep the agreed denominator limit 512 unchanged.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty dataset: {output}")
    if profile not in ("preview", "full"):
        raise ValueError("Invalid profile.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "signals").mkdir()
    (output / "references").mkdir()
    count = 2 if profile == "preview" else settings["per_class_per_snr"]
    levels = [15] if profile == "preview" else config["dataset"]["snr_db"]
    manifest = {
        "status": "building", "kind": "bandwidth_sweep_continuous_single_signal_raw_iq",
        "profile": profile, "classes": config["classes"], "snr_db": levels,
        "bandwidth_hz": settings["bandwidth_hz"], "per_class_per_snr": count,
        "effective_samples": settings["effective_samples"], "seed": settings["seed"],
        "seed_namespace": settings["seed_namespace"], "profile_seed_namespace": 1 if profile == "preview" else 2,
        "raw_dtype": "complex64", "code_sha256": {
            p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")},
        "scope": "fresh same-family waveforms at target bandwidths; acquisition resampled; no retraining",
        "resampling_max_denominator": 512,
    }
    write_json(output / "manifest.json", manifest)
    (output / "config_resolved.yaml").write_text(
        yaml.safe_dump({"baseline": config, "long_signal": settings}, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    row, payload = 0, 0
    with (output / "metadata.jsonl").open("w", encoding="utf-8") as stream:
        for bandwidth_index, bandwidth in enumerate(settings["bandwidth_hz"]):
            for class_id, name in enumerate(config["classes"]):
                subtypes = config["dataset"]["subtypes"].get(name, [name])
                for snr_index, snr in enumerate(levels):
                    for i in range(count):
                        subtype = subtypes[i % len(subtypes)]
                        ratios = settings["acquisition_rate_over_bandwidth"]
                        capture_ratio = ratios[(i + class_id + snr_index + bandwidth_index) % len(ratios)]
                        seed = [settings["seed"], settings["seed_namespace"], manifest["profile_seed_namespace"],
                                bandwidth_index, class_id, snr_index, i]
                        raw, metadata, refs = generate_wideband_record(
                            name, subtype, snr, seed, config, settings, bandwidth, capture_ratio)
                        relative = f"signals/{row:06d}.npz"
                        np.savez(output / relative, raw_iq=raw,
                                 input_metadata_json=np.array(json.dumps(metadata["input"])))
                        metadata.update({"row": row, "class_id": class_id, "path": relative,
                                         "bandwidth_hz": bandwidth, "bandwidth_index": bandwidth_index,
                                         "file_sha256": sha256_file(output / relative)})
                        if snr == 15 and i < len(subtypes):
                            np.savez(output / "references" / f"B{bandwidth}_{name}_{subtype}.npz", **refs,
                                     metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)))
                        stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                        row += 1
                        payload += raw.nbytes
                stream.flush()
                print(f"{profile}: B={bandwidth:g} Hz {name} complete; total={row}", flush=True)
    manifest.update({"status": "complete", "total_signals": row, "raw_payload_bytes": payload})
    write_json(output / "manifest.json", manifest)
    return manifest
