"""Small preview/full dataset writer and audit; never silently overwrite data."""
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from .config import CLASSES, SPLITS
from .preprocess import band_power
from .synthesis import generate_sample


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def create_dataset(config, output, profile="preview"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    counts = ({"train": 4, "validation": 2, "test": 2} if profile == "preview"
              else dict(config["dataset"]["per_class_per_snr"]))
    levels = config["dataset"]["snr_db"]
    classes = tuple(config["classes"])
    seed_namespace = 1 if profile == "preview" else 2
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    manifest = {
        "status": "building", "profile": profile, "classes": list(classes),
        "snr_db": levels, "counts_per_class_per_snr": counts,
        "window_samples": config["dataset"]["window_samples"],
        "layout": "[window, IQ channel (I then Q), time]",
        "dtype": "float32", "code_sha256": source_hashes,
        "seed_namespace": seed_namespace,
        "scope": "synthetic single-signal windows; no receiver integration or classifier training",
    }
    write_json(output / "manifest.json", manifest)
    (output / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    previews = output / "preview_references"
    previews.mkdir()
    total = 0
    for split_index, split in enumerate(SPLITS):
        per_cell = counts[split]
        n = len(classes) * len(levels) * per_cell
        folder = output / split
        folder.mkdir()
        x = np.lib.format.open_memmap(
            folder / "iq.npy", mode="w+", dtype=np.float32,
            shape=(n, 2, config["dataset"]["window_samples"])
        )
        y = np.lib.format.open_memmap(folder / "labels.npy", mode="w+",
                                     dtype=np.int64, shape=(n,))
        row = 0
        with (folder / "metadata.jsonl").open("w", encoding="utf-8") as stream:
            for class_id, name in enumerate(classes):
                subtypes = config["dataset"]["subtypes"].get(name, [name])
                for snr_index, snr in enumerate(levels):
                    for i in range(per_cell):
                        subtype = subtypes[i % len(subtypes)]
                        seed = [config["seed"], seed_namespace, split_index, class_id, snr_index, i]
                        iq, metadata, reference = generate_sample(
                            name, subtype, snr, seed, config
                        )
                        metadata.update({"row": row, "class_id": class_id, "split": split,
                                         "source_id": "-".join(map(str, seed))})
                        x[row], y[row] = iq, class_id
                        stream.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                        if split == "train" and snr == 15 and i < len(subtypes):
                            np.savez(
                                previews / f"{name}_{subtype}.npz",
                                iq=iq, **reference,
                                metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
                            )
                        row += 1
                    print(f"{split}: {name} SNR={snr} generated {per_cell}", flush=True)
        x.flush()
        y.flush()
        total += n
        del x, y
    manifest.update({"status": "complete", "total_windows": total,
                     "iq_payload_bytes": total * 2 * config["dataset"]["window_samples"] * 4})
    write_json(output / "manifest.json", manifest)
    return manifest


def validate_dataset(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Dataset generation did not finish.")
    classes = tuple(manifest["classes"])
    if classes not in (CLASSES, CLASSES[:-1]):
        raise ValueError("Unrecognized dataset class order.")
    source_ids = set()
    measured_snr_errors = []
    split_reports = {}
    for split in SPLITS:
        x = np.load(folder / split / "iq.npy", mmap_mode="r")
        labels = np.load(folder / split / "labels.npy", mmap_mode="r")
        count = manifest["counts_per_class_per_snr"][split]
        expected = len(classes) * len(manifest["snr_db"]) * count
        if x.shape != (expected, 2, manifest["window_samples"]) or x.dtype != np.float32:
            raise AssertionError(f"Invalid IQ shape/dtype in {split}.")
        if labels.shape != (expected,):
            raise AssertionError("Invalid label shape.")
        cells, subtypes = {}, {}
        rows = 0
        with (folder / split / "metadata.jsonl").open(encoding="utf-8") as stream:
            for row, line in enumerate(stream):
                m = json.loads(line)
                if m["source_id"] in source_ids:
                    raise AssertionError("Repeated source within/across splits.")
                source_ids.add(m["source_id"])
                if m["row"] != row or m["split"] != split or labels[row] != m["class_id"]:
                    raise AssertionError("Label/metadata mismatch.")
                if classes[int(labels[row])] != m["class_name"]:
                    raise AssertionError("Wrong public class label.")
                if not np.all(np.isfinite(x[row])):
                    raise AssertionError("Nonfinite IQ.")
                power = float(np.mean(np.sum(x[row].astype(np.float64) ** 2, axis=0)))
                if abs(power - 1) > 2e-5:
                    raise AssertionError("Power normalization failed.")
                error = abs(m["measured_inband_snr_db"] - m["snr_db"])
                if error > 1e-5:
                    raise AssertionError("SNR scaling failed.")
                measured_snr_errors.append(error)
                key = (m["class_name"], m["snr_db"])
                cells[key] = cells.get(key, 0) + 1
                subkey = (*key, m["subtype"])
                subtypes[subkey] = subtypes.get(subkey, 0) + 1
                rows += 1
        if rows != expected:
            raise AssertionError("Metadata row count mismatch.")
        for name in classes:
            for snr in manifest["snr_db"]:
                if cells.get((name, snr), 0) != count:
                    raise AssertionError("Class/SNR cell is unbalanced.")
                if name in ("AM", "ASK"):
                    names = ("DSB_LC", "SSB") if name == "AM" else ("OOK", "BINARY_NONZERO")
                    if any(subtypes.get((name, snr, subtype), 0) != count // 2 for subtype in names):
                        raise AssertionError("AM/ASK subtypes are unbalanced.")
        split_reports[split] = {"windows": expected, "class_snr_cells": len(cells),
                                "unique_sources": rows}
    reference_errors = []
    for path in (folder / "preview_references").glob("*.npz"):
        with np.load(path, allow_pickle=False) as r:
            m = json.loads(str(r["metadata_json"]))
            clean = r["clean"][0] + 1j * r["clean"][1]
            noise = r["noise"][0] + 1j * r["noise"][1]
            fs, bw = m["output_sample_rate_hz"], m["nominal_bandwidth_hz"]
            snr = 10 * np.log10(band_power(clean, fs, bw) / band_power(noise, fs, bw))
            error = abs(float(snr) - m["snr_db"])
            if error > 0.002 or not np.allclose(r["iq"], r["clean"] + r["noise"], atol=3e-7):
                raise AssertionError("Saved reference components fail independent SNR audit.")
            reference_errors.append(error)
    report = {
        "passed": True, "splits": split_reports, "unique_sources": len(source_ids),
        "max_metadata_snr_error_db": max(measured_snr_errors),
        "reference_files_checked": len(reference_errors),
        "max_reference_snr_error_db": max(reference_errors, default=0),
        "checks": ["finite float32 IQ", "unit RMS", "class/SNR/subtype balance",
                   "independent source seeds", "label mapping", "reference component SNR"],
        "classification_accuracy": None,
    }
    return report
