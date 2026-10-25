"""Parallel dataset writer and bounded integrity checks for amc20-v1."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import yaml

from .config import SPLITS
from .dataset import write_json
from .preprocess import band_power
from .synthesis20 import CLASSES20, generate_sample20


_CONFIG = None


def load_config20(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if tuple(config["classes"]) != CLASSES20:
        raise ValueError("20-class order mismatch")
    if config["dataset"]["window_samples"] != 2048 or config["dataset"]["storage_dtype"] != "float32":
        raise ValueError("Dataset must use float32 [2,2048]")
    d = config["dataset"]
    if d["snr_db"] != [0, 5, 10, 15, 20, 25, 30]:
        raise ValueError("Unexpected SNR levels")
    for split in SPLITS:
        n = d["per_class_per_snr"][split]
        if not isinstance(n, int) or n < 1:
            raise ValueError("Invalid count")
        factor = d["ofdm_train_multiplier"] if split == "train" else 1
        if sum(g[split] for g in d["ofdm_groups"].values()) != factor*n:
            raise ValueError("OFDM allocation does not sum to split count")
    return config


def _init(config):
    global _CONFIG
    _CONFIG = config


def _chunk(job):
    split_id, class_id, snr_id, subtype, indices, first_row = job
    name = _CONFIG["classes"][class_id]
    snr = _CONFIG["dataset"]["snr_db"][snr_id]
    iq = np.empty((len(indices), 2, 2048), dtype=np.float32)
    lines, references = [], []
    split = SPLITS[split_id]
    for j, i in enumerate(indices):
        subs = _CONFIG["dataset"]["subtypes"].get(name, [name])
        selected = subtype if name == "OFDM" else subs[(i+snr_id+split_id) % len(subs)]
        seed = [_CONFIG["seed"], 20, split_id, class_id, snr_id, i]
        data, meta, ref = generate_sample20(name, selected, snr, seed, _CONFIG)
        meta.update(row=first_row+j, class_id=class_id, split=split,
                    source_id="-".join(map(str, seed)))
        iq[j] = data
        lines.append(json.dumps(meta, ensure_ascii=False, separators=(",", ":")))
        # Small auditable clean/noise references, only one per subtype at 15 dB.
        if split == "train" and snr == 15 and (j == 0 and (subtype is not None or i < len(subs))):
            if name == "OFDM" or i < len(subs):
                references.append((f"{name}_{selected}", data, meta, ref))
        if split == "train" and snr == 15 and name != "OFDM" and i < len(subs) and j != 0:
            references.append((f"{name}_{selected}", data, meta, ref))
    return first_row, class_id, iq, lines, references


def _jobs(config, split_id, chunk_size):
    split, row = SPLITS[split_id], 0
    d = config["dataset"]
    for class_id, name in enumerate(config["classes"]):
        for snr_id in range(len(d["snr_db"])):
            offset = 0
            groups = ([(key, value[split]) for key, value in d["ofdm_groups"].items()]
                      if name == "OFDM" else [(None, d["per_class_per_snr"][split])])
            for group, count in groups:
                for first in range(0, count, chunk_size):
                    size = min(chunk_size, count-first)
                    indices = list(range(offset+first, offset+first+size))
                    yield (split_id, class_id, snr_id, group, indices, row)
                    row += size
                offset += count


def expected_counts(config, split):
    d = config["dataset"]
    return {name: d["per_class_per_snr"][split]*(d["ofdm_train_multiplier"] if name == "OFDM" and split == "train" else 1)
            for name in config["classes"]}


def create_dataset20(config, output, workers=8, chunk_size=32, resume=False):
    output = Path(output)
    previous = None
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError(f"Refusing to overwrite nonempty dataset: {output}")
        previous = json.loads((output/"manifest.json").read_text(encoding="utf-8"))
        stored_config = yaml.safe_load((output/"config_resolved.yaml").read_text(encoding="utf-8"))
        if previous["status"] != "building" or previous["version"] != config["version"] or stored_config != config:
            raise ValueError("Resume requires the same config and an incomplete dataset")
        if previous["environment"]["chunk_size"] != chunk_size:
            raise ValueError("Resume must keep chunk boundaries unchanged")
    output.mkdir(parents=True, exist_ok=True)
    refs = output/"preview_references"
    refs.mkdir(exist_ok=resume)
    started = time.time()
    code_paths = [Path(__file__), Path(__file__).with_name("synthesis20.py"),
                  Path(__file__).with_name("preprocess.py"), Path(__file__).with_name("synthesis.py")]
    manifest = {
        "status": "building", "profile": "full", "format_version": 2,
        "version": config["version"], "classes": config["classes"],
        "snr_db": config["dataset"]["snr_db"], "window_samples": 2048,
        "dtype": "float32", "layout": "[window, IQ channel (I then Q), time]",
        "counts_per_class_per_snr": config["dataset"]["per_class_per_snr"],
        "actual_counts_per_class_per_snr": {s: expected_counts(config, s) for s in SPLITS},
        "ofdm_group_counts_per_snr": config["dataset"]["ofdm_groups"],
        "seed_namespace": 20, "seed": config["seed"],
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "workers": workers, "chunk_size": chunk_size},
        "code_sha256": {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},
        "scope": "Synthetic 20-label dataset only; no training, service changes, container build or publication.",
        "limitations": [
            "OFDM protocol-style groups are structural approximations, not standards-compliant protocol transmitters.",
            "Unspread independent random-payload BPSK/DBPSK and QPSK/DQPSK can be statistically indistinguishable. Labels retain transmitter encoding provenance, not guaranteed blind identifiability.",
            "DBPSK contains both unspread differential RRC and Barker-11 DSSS. This is not comprehensive DSSS/protocol coverage.",
            "2048 samples need not contain a complete OFDM symbol or cyclic-prefix pair.",
            "Unknown calibration is not included; no unknown class is trained or deployed here.",
        ],
    }
    if previous:
        manifest["resumed_from"] = {"started_utc": previous["started_utc"], "code_sha256": previous["code_sha256"],
                                    "reason": "Convert NumPy boolean CP metadata to a native bool; generated IQ algorithms unchanged."}
    write_json(output/"manifest.json", manifest)
    (output/"config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    totals = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(config,)) as pool:
        for split_id, split in enumerate(SPLITS):
            count_map = expected_counts(config, split)
            n = sum(count_map.values())*len(manifest["snr_db"])
            folder = output/split
            existing = folder.exists()
            folder.mkdir(exist_ok=resume)
            mode = "r+" if existing else "w+"
            x = np.lib.format.open_memmap(folder/"iq.npy", mode=mode, dtype=np.float32, shape=(n,2,2048))
            y = np.lib.format.open_memmap(folder/"labels.npy", mode=mode, dtype=np.int64, shape=(n,))
            done, last = 0, 0.
            if existing:
                if x.shape != (n,2,2048) or y.shape != (n,):
                    raise ValueError("Existing array shapes do not match")
                with (folder/"metadata.jsonl").open(encoding="utf-8") as f:
                    for line in f:
                        m = json.loads(line)
                        if m["row"] != done or m["split"] != split:
                            raise ValueError("Cannot resume malformed metadata")
                        done += 1
                print(json.dumps({"resume_split":split,"preserved_rows":done}),flush=True)
            jobs = list(_jobs(config, split_id, chunk_size))
            if done not in {j[5] for j in jobs} | {n}:
                raise ValueError("Resume position is not a completed chunk boundary")
            remaining = [j for j in jobs if j[5] >= done]
            with (folder/"metadata.jsonl").open("a" if existing else "w", encoding="utf-8", newline="\n") as f:
                for first, cid, data, lines, references in pool.map(_chunk, remaining, chunksize=1):
                    if first != done:
                        raise AssertionError("Unexpected row order")
                    stop = first+len(data)
                    x[first:stop], y[first:stop] = data, cid
                    f.write("\n".join(lines)+"\n")
                    for label, arr, meta, ref in references:
                        path = refs/f"{label}.npz"
                        if not path.exists():
                            np.savez(path, iq=arr, metadata_json=np.array(json.dumps(meta)), **ref)
                    done = stop
                    now = time.time()
                    if now-last > 15 or done == n:
                        progress = {"split": split, "completed": done, "total": n,
                                    "class": config["classes"][cid], "elapsed_s": round(now-started,1)}
                        print(json.dumps(progress), flush=True)
                        write_json(output/"progress.json", progress)
                        last = now
            x.flush(); y.flush()
            del x,y
            if done != n:
                raise AssertionError("Wrong generated count")
            totals[split] = n
    manifest.update(status="complete", splits=totals, total_windows=sum(totals.values()),
                    iq_payload_bytes=sum(totals.values())*16384,
                    generation_elapsed_s=round(time.time()-started,3),
                    completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output/"manifest.json", manifest)
    return manifest


def validate_dataset20(output):
    root = Path(output)
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    config = load_config20(root/"config_resolved.yaml")
    if manifest["status"] != "complete":
        raise AssertionError("Dataset is incomplete")
    ids, hashes = set(), set()
    summary, max_snr_error, max_power_error = {}, 0., 0.
    for split in SPLITS:
        x = np.load(root/split/"iq.npy", mmap_mode="r")
        y = np.load(root/split/"labels.npy", mmap_mode="r")
        expected = manifest["splits"][split]
        if x.shape != (expected,2,2048) or x.dtype != np.float32 or y.shape != (expected,) or y.dtype != np.int64:
            raise AssertionError("Invalid array shape/type")
        cells, groups, subtypes, channels, constellations, ffts, prefixes = (Counter() for _ in range(7))
        overlap_labels = Counter()
        rows = 0
        with (root/split/"metadata.jsonl").open(encoding="utf-8") as f:
            for row, line in enumerate(f):
                m = json.loads(line)
                if m["row"] != row or m["split"] != split or int(y[row]) != m["class_id"] or config["classes"][int(y[row])] != m["class_name"]:
                    raise AssertionError("Label/metadata mismatch")
                if m["source_id"] in ids or m["iq_blake2b128"] in hashes:
                    raise AssertionError("Repeated independent source or identical IQ")
                ids.add(m["source_id"]); hashes.add(m["iq_blake2b128"])
                iq = x[row]
                if not np.isfinite(iq).all():
                    raise AssertionError("Nonfinite IQ")
                digest = hashlib.blake2b(iq.tobytes(), digest_size=16).hexdigest()
                if digest != m["iq_blake2b128"]:
                    raise AssertionError("IQ checksum mismatch")
                power_error = abs(float(np.mean(np.sum(iq.astype(float)**2, axis=0)))-1)
                snr_error = abs(m["measured_inband_snr_db"]-m["snr_db"])
                max_power_error = max(max_power_error, power_error)
                max_snr_error = max(max_snr_error, snr_error)
                if power_error > 2e-5 or snr_error > 1e-6:
                    raise AssertionError("Normalization/SNR mismatch")
                name, snr = m["class_name"], m["snr_db"]
                cells[(name,snr)] += 1
                subtypes[(name,snr,m["subtype"])] += 1
                channels[m["channel"]["kind"]] += 1
                if m["statistically_overlapping_label_group"]:
                    overlap_labels[m["statistically_overlapping_label_group"]] += 1
                if name == "OFDM":
                    p = m["source_parameters"]
                    groups[(m["subtype"],snr)] += 1
                    constellations[p["subcarrier_modulation"]] += 1
                    ffts[p["fft_size"]] += 1
                    prefixes[p["cp_fraction"]] += 1
                    if not 0 < m["nominal_bandwidth_hz"] < m["input_sample_rate_hz"]:
                        raise AssertionError("Invalid OFDM frequency geometry")
                    if min(p["cp_samples_per_symbol"]) < 1 or not any(p["active_symbol_mask"]):
                        raise AssertionError("Invalid OFDM CP/activity")
                rows += 1
        if rows != expected:
            raise AssertionError("Wrong metadata count")
        for name, count in expected_counts(config, split).items():
            for snr in manifest["snr_db"]:
                if cells[(name,snr)] != count:
                    raise AssertionError("Wrong class/SNR count")
                if name in config["dataset"]["subtypes"]:
                    ns = config["dataset"]["subtypes"][name]
                    sizes = [subtypes[(name,snr,s)] for s in ns]
                    if max(sizes)-min(sizes) > 1:
                        raise AssertionError("Unbalanced subtype counts")
        for group, counts in config["dataset"]["ofdm_groups"].items():
            for snr in manifest["snr_db"]:
                if groups[(group,snr)] != counts[split]:
                    raise AssertionError("Wrong OFDM group count")
        summary[split] = {"windows": rows,
            "class_counts": {name: int(np.sum(y == i)) for i,name in enumerate(config["classes"])},
            "ofdm_group_counts": {g: sum(v for (group,_),v in groups.items() if group == g) for g in config["dataset"]["ofdm_groups"]},
            "ofdm_fft_counts": dict(ffts), "ofdm_cp_fraction_counts": dict(prefixes),
            "ofdm_constellation_counts": dict(constellations), "channel_counts": dict(channels),
            "overlapping_label_windows": dict(overlap_labels)}
    reference_errors = []
    for p in (root/"preview_references").glob("*.npz"):
        with np.load(p, allow_pickle=False) as r:
            m = json.loads(str(r["metadata_json"]))
            clean, noise = r["clean"][0]+1j*r["clean"][1], r["noise"][0]+1j*r["noise"][1]
            fs,bw = m["output_sample_rate_hz"],m["nominal_bandwidth_hz"]
            err = abs(float(10*np.log10(band_power(clean,fs,bw)/band_power(noise,fs,bw)))-m["snr_db"])
            if err > .002 or not np.allclose(r["iq"], r["clean"]+r["noise"], atol=4e-7):
                raise AssertionError("Saved clean/noise reference mismatch")
            reference_errors.append(err)
    return {"passed": True, "total_windows": len(ids), "unique_iq_hashes": len(hashes),
            "splits": summary, "max_snr_error_db": max_snr_error,
            "max_unit_power_error": max_power_error, "reference_files_checked": len(reference_errors),
            "max_reference_snr_error_db": max(reference_errors, default=0),
            "checks": ["exact class/SNR/group counts", "finite float32 [2,2048]", "unit power", "inband SNR",
                       "metadata/label/array correspondence", "source independence and exact IQ duplicates", "clean/noise references"],
            "classification_accuracy": None, "training_started": False}
