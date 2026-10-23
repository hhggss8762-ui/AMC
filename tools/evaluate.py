import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from torch.utils.data import DataLoader
from amc.data_loader import IQWindowDataset
from amc.dataset import write_json
from amc.evaluation import evaluate_model, plot_confusion_matrix
from amc.model import load_classifier
from amc.training import resolve_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="data/preview_ofdm_v1")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive.")
    device = resolve_device(args.device)
    dataset = IQWindowDataset(args.dataset, args.split)
    model, payload = load_classifier(args.checkpoint, device, dataset.classes)
    if dataset.manifest["window_samples"] != model.window_samples:
        raise ValueError("Checkpoint/dataset window size mismatch.")
    data_config = yaml.safe_load((Path(args.dataset) / "config_resolved.yaml").read_text(encoding="utf-8"))
    if data_config["preprocess"] != payload["config"]["preprocess"]:
        raise ValueError("Checkpoint/dataset preprocessing mismatch.")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty evaluation: {output}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    # Full precision evaluation matches standalone inference.
    metrics, raw = evaluate_model(
        model, loader, device, dataset.classes, payload["config"]["acceptance"]["snr_db"]
    )
    metrics.update({
        "dataset_profile": dataset.manifest["profile"], "split": args.split,
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_epoch": payload["epoch"],
        "purpose": "preview_pipeline_check" if dataset.manifest["profile"] == "preview" else "formal_evaluation",
        "acceptance_evaluated": dataset.manifest["profile"] == "full" and args.split == "test",
    })
    metrics["acceptance_passed"] = (
        metrics["good_snr_accuracy"] >= payload["config"]["acceptance"]["overall_top1_accuracy_target"]
        if metrics["acceptance_evaluated"] and metrics["good_snr_accuracy"] is not None else None
    )
    output.mkdir(parents=True)
    write_json(output / "metrics.json", metrics)
    np.savez(output / "predictions.npz", **raw)
    plot_confusion_matrix(metrics, output / "confusion_matrix.png")
    print(json.dumps({k: metrics[k] for k in (
        "count", "accuracy", "good_snr_count", "good_snr_accuracy", "macro_f1",
        "dataset_profile", "purpose", "acceptance_evaluated", "acceptance_passed"
    )}, indent=2))


if __name__ == "__main__":
    main()
