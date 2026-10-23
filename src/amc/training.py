"""Fixed baseline training: train/validation only; test evaluation is a separate step."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from .data_loader import IQWindowDataset
from .dataset import write_json
from .evaluation import evaluate_model
from .model import build_model
from .device import resolve_device


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config, dataset_root, output, epochs=None, batch_size=None, device="auto"):
    config = deepcopy(config)
    t = config["training"]
    if epochs is not None:
        t["max_epochs"] = int(epochs)
    if batch_size is not None:
        t["batch_size"] = int(batch_size)
    if min(t["max_epochs"], t["batch_size"]) < 1:
        raise ValueError("Epochs/batch size must be positive.")
    if t["optimizer"] != "AdamW" or t["scheduler"] != "cosine":
        raise ValueError("Baseline expects AdamW and cosine scheduling.")
    if t["selection_metric"] != "validation_accuracy_good_snr":
        raise ValueError("Unsupported checkpoint selection metric.")
    device = resolve_device(device)
    classes = config["classes"]
    window = config["dataset"]["window_samples"]
    train_data = IQWindowDataset(dataset_root, "train", classes, window)
    valid_data = IQWindowDataset(dataset_root, "validation", classes, window)
    data_config = yaml.safe_load((Path(dataset_root) / "config_resolved.yaml").read_text(encoding="utf-8"))
    if data_config["preprocess"] != config["preprocess"]:
        raise ValueError("Dataset/training preprocessing configuration mismatch.")
    good_snrs = config["acceptance"]["snr_db"]
    if not np.isin(valid_data.snrs, good_snrs).any():
        raise ValueError("Validation set does not cover checkpoint-selection SNRs.")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator().manual_seed(seed)
    loader_args = {"batch_size": t["batch_size"], "num_workers": 0,
                   "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_args)
    valid_loader = DataLoader(valid_data, shuffle=False, **loader_args)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["max_epochs"])
    amp = bool(t["mixed_precision"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    manifest_path = Path(dataset_root) / "manifest.json"
    environment = {
        "python_executable": os.sys.executable, "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "mixed_precision": amp, "seed": seed, "num_workers": 0,
        "dataset_root": str(Path(dataset_root).resolve()),
        "dataset_profile": train_data.manifest["profile"],
        "dataset_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "train_windows": len(train_data), "validation_windows": len(valid_data),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in Path(__file__).parent.glob("*.py")},
        "test_used_for_model_selection": False,
    }
    write_json(output / "environment.json", environment)
    (output / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    history = []
    best_accuracy, best_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
    started = time.perf_counter()
    print(f"Device={device}, AMP={amp}, parameters={environment['parameter_count']:,}, "
          f"train={len(train_data)}, validation={len(valid_data)}", flush=True)
    with (output / "history.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, t["max_epochs"] + 1):
            epoch_started = time.perf_counter()
            model.train()
            total_loss, total_correct, seen = 0.0, 0, 0
            for x, y, _ in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss.")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach()) * len(y)
                total_correct += int((logits.detach().argmax(dim=1) == y).sum())
                seen += len(y)
            validation, _ = evaluate_model(
                model, valid_loader, device, classes, good_snrs, mixed_precision=amp
            )
            good_accuracy = validation["good_snr_accuracy"]
            good_loss = validation["good_snr_loss"]
            improved = (good_accuracy > best_accuracy or
                        (good_accuracy == best_accuracy and good_loss < best_loss))
            if improved:
                best_accuracy, best_loss, best_epoch, stale = good_accuracy, good_loss, epoch, 0
            else:
                stale += 1
            row = {
                "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": total_loss / seen, "train_accuracy": total_correct / seen,
                "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
                "validation_good_snr_accuracy": good_accuracy, "validation_good_snr_loss": good_loss,
                "best_epoch": best_epoch, "seconds": time.perf_counter() - epoch_started,
            }
            history.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            scheduler.step()
            payload = {
                "format_version": 1, "epoch": epoch, "classes": list(classes), "config": config,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                "torch_rng_state": torch.get_rng_state(), "loader_rng_state": generator.get_state(),
                "best_epoch": best_epoch, "best_validation_good_snr_accuracy": best_accuracy,
                "validation_metrics": validation, "environment": environment,
            }
            save_checkpoint(output / "last.pt", payload)
            if improved:
                save_checkpoint(output / "best.pt", payload)
            print(f"epoch {epoch:02d}/{t['max_epochs']} train_loss={row['train_loss']:.4f} "
                  f"train_acc={row['train_accuracy']:.1%} val_acc={row['validation_accuracy']:.1%} "
                  f"good_snr_val_acc={good_accuracy:.1%} time={row['seconds']:.2f}s", flush=True)
            if stale >= t["early_stopping_patience"]:
                print("Early stopping on validation metric.", flush=True)
                break
    summary = {
        "status": "complete", "epochs_completed": len(history), "best_epoch": best_epoch,
        "best_validation_good_snr_accuracy": best_accuracy,
        "first_train_loss": history[0]["train_loss"], "last_train_loss": history[-1]["train_loss"],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "dataset_profile": train_data.manifest["profile"],
        "run_kind": "preview_smoke" if train_data.manifest["profile"] == "preview" else "formal_training",
        "acceptance_evaluated": False, "test_used_for_model_selection": False,
        "best_checkpoint": str((output / "best.pt").resolve()),
        "environment": environment,
    }
    write_json(output / "summary.json", summary)
    return summary
