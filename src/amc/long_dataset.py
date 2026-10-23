"""Independent continuous-waveform dataset; global SNR, raw complex64 storage."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from .dataset import write_json
from .long_inference import load_long_input, prepare_long_iq, window_starts
from .preprocess import band_power
from .synthesis import generate_waveform


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_long_settings(path):
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    count = settings["per_class_per_snr"]
    if not isinstance(count, int) or count < 2 or count % 2:
        raise ValueError("Long signal counts must be positive even integers.")
    if settings["effective_samples"] < settings["max_windows"] * 2048:
        raise ValueError("Need enough samples for nonoverlapping windows.")
    if settings["max_windows"] != 5 or settings["seed_namespace"] != 3:
        raise ValueError("This long baseline fixes five windows and seed namespace 3.")
    return settings


def _snr(clean, noise, fs, bandwidth):
    a, b = band_power(clean, fs, bandwidth), band_power(noise, fs, bandwidth)
    if min(a, b) <= 1e-14:
        raise ValueError("Degenerate signal/noise component power.")
    return float(10 * np.log10(a / b))


def generate_long_record(name, subtype, snr_db, seed_words, config, settings, *, waveform_factory=None):
    rng = np.random.default_rng(np.random.SeedSequence(seed_words))
    expanded = deepcopy(config)
    sizes = settings["source_lengths"]
    expanded["synthesis"]["source_symbols"] = sizes["digital_symbols"]
    expanded["synthesis"]["ofdm"]["source_ofdm_symbols"] = sizes["ofdm_symbols"]
    analog_count = (sizes["am_samples"] if name == "AM" else
                    sizes["fm_samples"] if name == "FM" else None)
    waveform = (generate_waveform(name, subtype, rng, expanded,
                                  analog_sample_count=analog_count)
                if waveform_factory is None else waveform_factory(rng))
    phase = float(rng.uniform(-np.pi, np.pi))
    cfo = float(rng.uniform(-1, 1)
                * config["synthesis"]["residual_frequency_offset_fraction"]
                * waveform.bandwidth_hz)
    error = float(rng.uniform(-1, 1)
                  * config["preprocess"]["bandwidth_estimate_relative_error"])
    time = np.arange(len(waveform.iq)) / waveform.sample_rate_hz
    clean_raw = (waveform.iq * np.exp(1j * (phase + 2 * np.pi * cfo * time))).astype(np.complex64)
    noise_raw = (rng.standard_normal(len(clean_raw))
                 + 1j * rng.standard_normal(len(clean_raw))) / np.sqrt(2)
    inputs = {
        "sample_rate_hz": waveform.sample_rate_hz,
        "estimated_bandwidth_hz": waveform.bandwidth_hz * (1 + error),
        "center_offset_hz": waveform.center_offset_hz,
        "effective_samples": int(settings["effective_samples"]),
    }
    clean, preprocessing = prepare_long_iq(clean_raw, settings=config["preprocess"], **inputs)
    noise, _ = prepare_long_iq(noise_raw, settings=config["preprocess"], **inputs)
    fs, bandwidth = preprocessing["output_sample_rate_hz"], waveform.bandwidth_hz
    scale = np.sqrt(band_power(clean, fs, bandwidth)
                    / (band_power(noise, fs, bandwidth) * 10 ** (snr_db / 10)))
    # One scalar for the entire raw record, never per-window noise adjustment.
    noise_raw = (noise_raw * scale).astype(np.complex64)
    noise, _ = prepare_long_iq(noise_raw, settings=config["preprocess"], **inputs)
    raw = (clean_raw + noise_raw).astype(np.complex64)
    replay, _ = prepare_long_iq(raw, settings=config["preprocess"], **inputs)
    if not np.isfinite(replay).all() or not np.allclose(replay, clean + noise, atol=2e-6, rtol=2e-6):
        raise AssertionError("Raw storage does not reproduce the component preprocessing.")
    starts = window_starts(len(clean), 2048, settings["max_windows"])
    metadata = {
        "class_name": name, "subtype": subtype, "snr_db": float(snr_db),
        "seed_words": list(map(int, seed_words)),
        "source_id": "-".join(map(str, seed_words)),
        "source_sample_count": len(raw), "source_parameters": waveform.parameters,
        "nominal_bandwidth_hz": bandwidth, "initial_phase_rad": phase,
        "residual_frequency_offset_hz": cfo, "bandwidth_estimate_relative_error": error,
        "input": inputs, "preprocessing": preprocessing,
        "noise_scale_applied_to_whole_record": float(scale),
        "measured_inband_snr_db": _snr(clean, noise, fs, bandwidth),
        "window_starts": starts,
        "window_measured_snr_db": [_snr(clean[s:s + 2048], noise[s:s + 2048], fs, bandwidth)
                                   for s in starts],
        "snr_definition": "nominal true band power over full effective segment after preprocessing",
    }
    references = {"raw_clean": clean_raw, "raw_noise": noise_raw,
                  "clean": clean.astype(np.complex64), "noise": noise.astype(np.complex64),
                  "processed_iq": replay.astype(np.complex64)}
    return raw, metadata, references


def create_long_dataset(config, settings, output, profile="full"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty dataset: {output}")
    if profile not in ("preview", "full"):
        raise ValueError("Invalid long dataset profile.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "signals").mkdir()
    (output / "references").mkdir()
    count = 2 if profile == "preview" else settings["per_class_per_snr"]
    levels = [15] if profile == "preview" else config["dataset"]["snr_db"]
    manifest = {
        "status": "building", "kind": "continuous_single_signal_raw_iq",
        "profile": profile, "classes": config["classes"], "snr_db": levels,
        "per_class_per_snr": count, "effective_samples": settings["effective_samples"],
        "seed": settings["seed"], "seed_namespace": settings["seed_namespace"],
        "profile_seed_namespace": 1 if profile == "preview" else 2,
        "raw_dtype": "complex64", "code_sha256": {
            p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")},
        "notes": "One continuous waveform per record; one global noise scale; no classifier training.",
    }
    write_json(output / "manifest.json", manifest)
    (output / "config_resolved.yaml").write_text(
        yaml.safe_dump({"baseline": config, "long_signal": settings},
                       allow_unicode=True, sort_keys=False), encoding="utf-8")
    row, raw_payload_bytes = 0, 0
    with (output / "metadata.jsonl").open("w", encoding="utf-8") as stream:
        for class_id, name in enumerate(config["classes"]):
            subtypes = config["dataset"]["subtypes"].get(name, [name])
            for snr_index, snr in enumerate(levels):
                for i in range(count):
                    subtype = subtypes[i % len(subtypes)]
                    seed = [settings["seed"], settings["seed_namespace"],
                            manifest["profile_seed_namespace"], class_id, snr_index, i]
                    raw, metadata, refs = generate_long_record(name, subtype, snr, seed, config, settings)
                    relative = f"signals/{row:06d}.npz"
                    np.savez(output / relative, raw_iq=raw,
                             input_metadata_json=np.array(json.dumps(metadata["input"])))
                    metadata.update({"row": row, "class_id": class_id, "path": relative,
                                     "file_sha256": sha256_file(output / relative)})
                    if snr == 15 and i < len(subtypes):
                        np.savez(output / "references" / f"{name}_{subtype}.npz", **refs,
                                 metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)))
                    stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                    row += 1
                    raw_payload_bytes += raw.nbytes
                stream.flush()
                print(f"{profile}: {name} SNR={snr} generated {count}; total={row}", flush=True)
    manifest.update({"status": "complete", "total_signals": row,
                     "raw_payload_bytes": raw_payload_bytes})
    write_json(output / "manifest.json", manifest)
    return manifest


def validate_long_dataset(folder, prior_datasets=()):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Long generation has not completed.")
    resolved = yaml.safe_load((folder / "config_resolved.yaml").read_text(encoding="utf-8"))
    config, settings = resolved["baseline"], resolved["long_signal"]
    wide = manifest["kind"] == "bandwidth_sweep_continuous_single_signal_raw_iq"
    if config["classes"] != manifest["classes"]:
        raise AssertionError("Class order mismatch.")
    prior_ids, prior_counts = set(), {}
    for prior in prior_datasets:
        prior = Path(prior)
        if not prior.exists():
            continue
        paths = ([prior / "metadata.jsonl"] if (prior / "metadata.jsonl").exists() else
                 [prior / split / "metadata.jsonl" for split in ("train", "validation", "test")])
        count = 0
        for path in paths:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    prior_ids.add(json.loads(line)["source_id"])
                    count += 1
        prior_counts[str(prior)] = count
    ids, cells, subtypes, paths = set(), {}, {}, set()
    max_snr_error, raw_bytes, max_rate_error = 0.0, 0, 0.0
    with (folder / "metadata.jsonl").open(encoding="utf-8") as stream:
        for row, line in enumerate(stream):
            metadata = json.loads(line)
            source_id = metadata["source_id"]
            if source_id in ids or source_id in prior_ids:
                raise AssertionError("Repeated source seed.")
            ids.add(source_id)
            seed = metadata["seed_words"]
            expected_seed = [manifest["seed"], manifest["seed_namespace"],
                             manifest["profile_seed_namespace"]]
            if wide:
                expected_seed.append(metadata["bandwidth_index"])
            expected_seed += [metadata["class_id"],
                             manifest["snr_db"].index(metadata["snr_db"])]
            if seed[:len(expected_seed)] != expected_seed or source_id != "-".join(map(str, seed)):
                raise AssertionError("Wrong long signal seed namespace.")
            if metadata["row"] != row or config["classes"][metadata["class_id"]] != metadata["class_name"]:
                raise AssertionError("Label/row mismatch.")
            path = folder / metadata["path"]
            if path in paths or sha256_file(path) != metadata["file_sha256"]:
                raise AssertionError("Repeated path or raw file hash mismatch.")
            paths.add(path)
            raw, inputs = load_long_input(path)
            if (inputs != metadata["input"] or raw.dtype != np.complex64 or raw.ndim != 1
                    or len(raw) != metadata["source_sample_count"] or not np.isfinite(raw).all()):
                raise AssertionError("Invalid raw IQ or acquisition metadata.")
            raw_bytes += raw.nbytes
            processed, replay_meta = prepare_long_iq(raw, settings=config["preprocess"], **inputs)
            if len(processed) != manifest["effective_samples"] or not np.isfinite(processed).all():
                raise AssertionError("Invalid processed continuous IQ.")
            if replay_meta != metadata["preprocessing"]:
                raise AssertionError("Raw replay metadata differs.")
            if wide:
                bw = manifest["bandwidth_hz"][metadata["bandwidth_index"]]
                if metadata["bandwidth_hz"] != bw or not np.isclose(
                        metadata["nominal_bandwidth_hz"], bw, rtol=2e-12, atol=1e-6):
                    raise AssertionError("Wrong nominal sweep bandwidth.")
                p = metadata["source_parameters"]
                acquisition_fs = (p["generator_sample_rate_hz"] * p["acquisition_resample_up"]
                                  / p["acquisition_resample_down"])
                if not np.isclose(acquisition_fs, inputs["sample_rate_hz"], rtol=1e-12):
                    raise AssertionError("Acquisition resampling rate mismatch.")
                if p["acquisition_requested_rate_over_bandwidth"] not in settings["acquisition_rate_over_bandwidth"]:
                    raise AssertionError("Unexpected acquisition sample rate ratio.")
                rate_error = abs(replay_meta["output_sample_rate_hz"]
                                 / replay_meta["requested_output_sample_rate_hz"] - 1)
                if rate_error > settings["max_output_rate_relative_error"]:
                    raise AssertionError("Output sample rate approximation exceeds declared tolerance.")
                max_rate_error = max(max_rate_error, rate_error)
            if not float(np.mean(abs(processed) ** 2)) > 1e-12:
                raise AssertionError("Zero-power processed record.")
            starts = window_starts(len(processed), 2048, settings["max_windows"])
            if starts != metadata["window_starts"] or len(starts) != 5:
                raise AssertionError("Wrong window positions.")
            if any(b - a < 2048 for a, b in zip(starts, starts[1:])):
                raise AssertionError("Overlapping windows.")
            snr_error = abs(metadata["measured_inband_snr_db"] - metadata["snr_db"])
            if snr_error > 0.002 or len(metadata["window_measured_snr_db"]) != 5:
                raise AssertionError("Whole-record SNR failed.")
            max_snr_error = max(max_snr_error, snr_error)
            cell = (metadata["class_name"], metadata["snr_db"])
            if wide:
                cell += (metadata["bandwidth_hz"],)
            cells[cell] = cells.get(cell, 0) + 1
            sub = (*cell, metadata["subtype"])
            subtypes[sub] = subtypes.get(sub, 0) + 1
    bandwidths = manifest["bandwidth_hz"] if wide else [None]
    expected = len(config["classes"]) * len(manifest["snr_db"]) * manifest["per_class_per_snr"] * len(bandwidths)
    if len(ids) != expected or manifest["total_signals"] != expected or raw_bytes != manifest["raw_payload_bytes"]:
        raise AssertionError("Signal count/payload mismatch.")
    if len(list((folder / "signals").glob("*.npz"))) != expected:
        raise AssertionError("Unexpected signal files.")
    for name in config["classes"]:
        names = config["dataset"]["subtypes"].get(name, [name])
        for snr in manifest["snr_db"]:
            for bandwidth in bandwidths:
                cell = (name, snr, bandwidth) if wide else (name, snr)
                if cells.get(cell) != manifest["per_class_per_snr"]:
                    raise AssertionError("Class/SNR/bandwidth imbalance.")
                for subtype in names:
                    if subtypes.get((*cell, subtype)) != manifest["per_class_per_snr"] // len(names):
                        raise AssertionError("Subtype imbalance.")
    reference_errors = []
    for path in (folder / "references").glob("*.npz"):
        with np.load(path, allow_pickle=False) as ref:
            m = json.loads(str(ref["metadata_json"]))
            clean, pre = prepare_long_iq(ref["raw_clean"], settings=config["preprocess"], **m["input"])
            noise, _ = prepare_long_iq(ref["raw_noise"], settings=config["preprocess"], **m["input"])
            raw, _ = load_long_input(folder / m["path"])
            replay, _ = prepare_long_iq(raw, settings=config["preprocess"], **m["input"])
            error = abs(_snr(clean, noise, pre["output_sample_rate_hz"], m["nominal_bandwidth_hz"]) - m["snr_db"])
            if error > 0.002 or not np.allclose(replay, clean + noise, atol=2e-6, rtol=2e-6):
                raise AssertionError("Independent raw component replay/SNR failed.")
            if not np.allclose(replay, ref["processed_iq"], atol=2e-6, rtol=2e-6):
                raise AssertionError("Saved processed reference differs.")
            reference_errors.append(error)
    expected_refs = sum(len(config["dataset"]["subtypes"].get(c, [c])) for c in config["classes"]) * len(bandwidths)
    if len(reference_errors) != expected_refs:
        raise AssertionError("Missing component references.")
    report = {
        "passed": True, "total_signals": expected, "selected_windows": expected * 5,
        "class_snr_cells": len(cells), "unique_sources": len(ids),
        "prior_datasets_checked": prior_counts, "prior_source_overlap": 0,
        "max_metadata_snr_error_db": max_snr_error,
        "reference_files_checked": len(reference_errors),
        "max_reference_snr_error_db": max(reference_errors), "raw_payload_bytes": raw_bytes,
        "total_directory_bytes": sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()),
        "checks": ["raw complex64 finite IQ", "raw file SHA256", "raw preprocessing replay",
                   "five nonoverlapping windows", "class/SNR/subtype counts", "unique source seeds",
                   "acquisition metadata excludes labels", "whole-record SNR and reference components"],
    }
    if wide:
        report.update({"bandwidth_hz": bandwidths,
                       "max_output_rate_relative_error": max_rate_error,
                       "preprocess_resampling_max_denominator": config["preprocess"]["resampling_max_denominator"]})
    return report
