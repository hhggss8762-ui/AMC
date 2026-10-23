"""Paired one/five-window evaluation on frozen continuous-waveform records."""
import json
from pathlib import Path
import time
import numpy as np
import yaml
from .dataset import write_json
from .evaluation import summarize_predictions, plot_confusion_matrix
from .long_dataset import sha256_file
from .long_inference import LongIQClassifier, load_long_input


def _timing_stats(values):
    a = np.asarray(values, dtype=float)
    return {"count": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max())}


def evaluate_long_dataset(checkpoint, dataset, output, device="auto"):
    dataset, output, checkpoint = Path(dataset), Path(output), Path(checkpoint)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty evaluation: {output}")
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Dataset is incomplete.")
    wide = manifest["kind"] == "bandwidth_sweep_continuous_single_signal_raw_iq"
    resolved = yaml.safe_load((dataset / "config_resolved.yaml").read_text(encoding="utf-8"))
    settings, config = resolved["long_signal"], resolved["baseline"]
    rows = [json.loads(line) for line in (dataset / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != manifest["total_signals"]:
        raise ValueError("Metadata count does not match manifest.")
    checkpoint_sha = sha256_file(checkpoint)
    startup_at = time.perf_counter()
    classifier = LongIQClassifier(checkpoint, device, settings["score_threshold"])
    classifier._synchronize()
    startup_ms = (time.perf_counter() - startup_at) * 1000
    if classifier.classes != config["classes"] or classifier.settings != config["preprocess"]:
        raise ValueError("Checkpoint/data class or preprocessing mismatch.")
    classes = classifier.classes
    good_levels = config["acceptance"]["snr_db"]
    warmup_count = min(settings["timing_warmup_signals"], len(rows))
    for row in rows[:warmup_count]:
        raw, inputs = load_long_input(dataset / row["path"])
        for n in (1, settings["max_windows"]):
            classifier.predict(raw, **inputs, max_windows=n, batch_size=settings["batch_size"])
    output.mkdir(parents=True, exist_ok=True)
    labels, snrs, middle_predictions, fused_predictions = [], [], [], []
    middle_scores, fused_scores, all_window_predictions, all_window_scores = [], [], [], []
    consistency, disagreements = [], []
    timings = {"middle": {}, "five": {}, "disk_read": []}
    status_counts = {"classified": 0, "uncertain": 0}
    loop_at = time.perf_counter()
    with (output / "signal_results.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            before_io = time.perf_counter()
            raw, inputs = load_long_input(dataset / row["path"])
            io_ms = (time.perf_counter() - before_io) * 1000
            # Alternate call order so one mode is not always measured second.
            results = {}
            for mode in (("five", "middle") if index % 2 == 0 else ("middle", "five")):
                results[mode] = classifier.predict(
                    raw, **inputs, max_windows=settings["max_windows"] if mode == "five" else 1,
                    batch_size=settings["batch_size"],
                )
            five, middle = results["five"], results["middle"]
            if five["window_count"] != 5 or middle["window_count"] != 1:
                raise AssertionError("Expected a valid five-window and middle-only prediction.")
            labels.append(row["class_id"])
            snrs.append(row["snr_db"])
            middle_predictions.append(middle["class_id"])
            fused_predictions.append(five["class_id"])
            middle_scores.append([middle["class_scores"][c] for c in classes])
            fused_scores.append([five["class_scores"][c] for c in classes])
            all_window_predictions.append([p["class_id"] for p in five["windows"]])
            all_window_scores.append([[p["class_scores"][c] for c in classes] for p in five["windows"]])
            consistency.append(five["consistency"])
            disagreements.append(len({p["class_id"] for p in five["windows"]}) > 1)
            status_counts[five["status"]] += 1
            timings["disk_read"].append(io_ms)
            for mode in ("middle", "five"):
                for key, value in results[mode]["timing_ms"].items():
                    timings[mode].setdefault(key, []).append(value)
            stream.write(json.dumps({"source_id": row["source_id"], "row": index,
                                     "truth": {k: row[k] for k in ("class_id", "class_name", "subtype", "snr_db",
                                                                   "bandwidth_hz", "bandwidth_index") if k in row},
                                     "five_window": five, "middle_only": middle,
                                     "disk_read_ms": io_ms}, ensure_ascii=False) + "\n")
            if (index + 1) % 140 == 0 or index + 1 == len(rows):
                stream.flush()
                print(f"Evaluated {index + 1}/{len(rows)} continuous signals", flush=True)
    elapsed = time.perf_counter() - loop_at
    labels, snrs = np.asarray(labels), np.asarray(snrs)
    middle_predictions, fused_predictions = np.asarray(middle_predictions), np.asarray(fused_predictions)
    middle_scores, fused_scores = np.asarray(middle_scores), np.asarray(fused_scores)
    window_predictions, window_scores = np.asarray(all_window_predictions), np.asarray(all_window_scores)
    if not np.allclose(fused_scores, window_scores.mean(axis=1), atol=1e-8):
        raise AssertionError("Fusion is not the unweighted mean.")
    if not np.array_equal(fused_scores.argmax(axis=1), fused_predictions):
        raise AssertionError("Fusion class does not match argmax.")
    middle_metrics = summarize_predictions(labels, middle_predictions, snrs, classes, good_levels)
    fused_metrics = summarize_predictions(labels, fused_predictions, snrs, classes, good_levels)
    good = np.isin(snrs, good_levels)
    good_metrics = summarize_predictions(labels[good], fused_predictions[good], snrs[good], classes, good_levels)
    window_metrics = summarize_predictions(np.repeat(labels, 5), window_predictions.ravel(),
                                           np.repeat(snrs, 5), classes, good_levels)
    middle_correct, fused_correct = middle_predictions == labels, fused_predictions == labels
    improved, regressed = ~middle_correct & fused_correct, middle_correct & ~fused_correct
    if int(fused_correct.sum() - middle_correct.sum()) != int(improved.sum() - regressed.sum()):
        raise AssertionError("Paired comparison counts disagree.")
    if sha256_file(checkpoint) != checkpoint_sha:
        raise AssertionError("Checkpoint changed during evaluation.")
    metrics = {
        "profile": manifest["profile"], "unit": "independent continuous source signal",
        "count": len(rows), "good_snr_count": int(good.sum()), "selected_window_count": len(rows) * 5,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": classifier.window_classifier.checkpoint["epoch"],
        "checkpoint_modified": False, "retrained": False, "all_signals_counted": True,
        "dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
        "classes": classes, "middle_only": middle_metrics, "five_window": fused_metrics,
        "five_window_good_snr": good_metrics, "selected_windows_diagnostic": window_metrics,
        "paired_comparison": {
            "improved_signals": int(improved.sum()), "regressed_signals": int(regressed.sum()),
            "unchanged_correctness": int((middle_correct == fused_correct).sum()),
            "overall_accuracy_change_percentage_points": 100 * (fused_metrics["accuracy"] - middle_metrics["accuracy"]),
            "good_snr_accuracy_change_percentage_points": 100 * (fused_metrics["good_snr_accuracy"] - middle_metrics["good_snr_accuracy"]),
            "good_snr_improved": int((improved & good).sum()),
            "good_snr_regressed": int((regressed & good).sum()),
        },
        "window_consistency": {"mean": float(np.mean(consistency)),
                               "signals_with_any_disagreement": int(sum(disagreements)),
                               "all_five_agree_count": int(len(rows) - sum(disagreements))},
        "five_window_status_counts": status_counts,
        "timing_ms": {mode: {key: _timing_stats(values) for key, values in timings[mode].items()}
                      for mode in ("middle", "five")},
        "disk_read_ms": _timing_stats(timings["disk_read"]), "startup_ms": startup_ms,
        "timing_warmup_signals": warmup_count, "evaluation_loop_seconds": elapsed,
        "timing_note": "Warm paired calls, alternating order; GPU synchronized; disk read and startup separate. All signals counted in accuracy.",
        "good_snr_reference_target": config["acceptance"]["overall_top1_accuracy_target"],
        "good_snr_reference_target_reached": (fused_metrics["good_snr_accuracy"] >= config["acceptance"]["overall_top1_accuracy_target"]
                                              if manifest["profile"] == "full" else None),
        "scope": "same-family synthetic continuous single signals; no detector, receiver frames, or real-signal accuracy claim",
    }
    write_json(output / "metrics.json", metrics)
    sweep_arrays = {}
    if wide:
        bandwidths = np.array([r["bandwidth_hz"] for r in rows], dtype=np.float64)
        capture_ratios = np.array([r["source_parameters"]["acquisition_requested_rate_over_bandwidth"]
                                   for r in rows], dtype=np.float64)
        rate_errors = np.array([abs(r["preprocessing"]["output_sample_rate_hz"]
                                   / r["preprocessing"]["requested_output_sample_rate_hz"] - 1) for r in rows])
        grouped = {}
        for bandwidth in manifest["bandwidth_hz"]:
            mask = bandwidths == bandwidth
            by_ratio = {}
            for ratio in settings["acquisition_rate_over_bandwidth"]:
                selected = mask & (capture_ratios == ratio)
                by_ratio[str(ratio)] = {
                    "middle": summarize_predictions(labels[selected], middle_predictions[selected], snrs[selected], classes, good_levels),
                    "five": summarize_predictions(labels[selected], fused_predictions[selected], snrs[selected], classes, good_levels),
                }
            timing = {mode: {key: _timing_stats(np.asarray(values)[mask])
                             for key, values in timings[mode].items()} for mode in ("middle", "five")}
            grouped[str(bandwidth)] = {
                "bandwidth_hz": bandwidth,
                "middle": summarize_predictions(labels[mask], middle_predictions[mask], snrs[mask], classes, good_levels),
                "five": summarize_predictions(labels[mask], fused_predictions[mask], snrs[mask], classes, good_levels),
                "five_good_snr": summarize_predictions(labels[mask & good], fused_predictions[mask & good], snrs[mask & good], classes, good_levels),
                "by_acquisition_ratio": by_ratio, "timing_ms": timing,
                "output_rate_relative_error_max": float(rate_errors[mask].max()),
            }
        metrics.update({"dataset_kind": manifest["kind"], "bandwidth_hz": manifest["bandwidth_hz"],
                        "by_bandwidth": grouped,
                        "output_rate_relative_error_max": float(rate_errors.max()),
                        "resampling_max_denominator": config["preprocess"]["resampling_max_denominator"]})
        write_json(output / "metrics.json", metrics)
        sweep_arrays = {"bandwidth_hz": bandwidths, "acquisition_ratio": capture_ratios,
                        "output_rate_relative_error": rate_errors}
    np.savez(output / "predictions.npz", labels=labels, snrs=snrs,
             middle_predictions=middle_predictions, fused_predictions=fused_predictions,
             middle_scores=middle_scores, fused_scores=fused_scores,
             window_predictions=window_predictions, window_scores=window_scores, **sweep_arrays)
    plot_confusion_matrix(fused_metrics, output / "confusion_matrix.png")
    plot_confusion_matrix(good_metrics, output / "confusion_matrix_good_snr.png")
    return metrics
