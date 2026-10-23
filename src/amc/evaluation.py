"""Top-1 metrics that count every window, including low-score predictions."""
import numpy as np
import torch


def summarize_predictions(labels, predictions, snrs, classes, good_snr_levels, losses=None):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    snrs = np.asarray(snrs, dtype=np.float64)
    count = len(labels)
    if count == 0 or predictions.shape != labels.shape or snrs.shape != labels.shape:
        raise ValueError("Invalid evaluation arrays.")
    n_classes = len(classes)
    if min(labels.min(), predictions.min()) < 0 or max(labels.max(), predictions.max()) >= n_classes:
        raise ValueError("Invalid class ID in predictions.")
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    correct = labels == predictions
    good = np.isin(snrs, np.asarray(good_snr_levels, dtype=float))
    per_snr = {}
    for snr in sorted(set(snrs.tolist())):
        mask = snrs == snr
        per_snr[f"{snr:g}"] = {"count": int(mask.sum()), "accuracy": float(correct[mask].mean())}
    per_class, f1s = {}, []
    for i, name in enumerate(classes):
        support = int(matrix[i].sum())
        predicted = int(matrix[:, i].sum())
        tp = int(matrix[i, i])
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
        if support:
            f1s.append(f1)
    result = {
        "count": count, "accuracy": float(correct.mean()),
        "good_snr_count": int(good.sum()),
        "good_snr_accuracy": float(correct[good].mean()) if good.any() else None,
        "good_snr_levels_db": list(good_snr_levels),
        "macro_f1": float(np.mean(f1s)), "per_snr": per_snr, "per_class": per_class,
        "classes": list(classes), "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axes": {"rows": "true", "columns": "predicted"},
        "all_samples_counted": True,
    }
    if losses is not None:
        losses = np.asarray(losses)
        if losses.shape != labels.shape:
            raise ValueError("Loss array does not align with labels.")
        result["loss"] = float(losses.mean())
        result["good_snr_loss"] = float(losses[good].mean()) if good.any() else None
    return result


@torch.inference_mode()
def evaluate_model(model, loader, device, classes, good_snr_levels, mixed_precision=False):
    model.eval()
    labels, predicted, snrs, losses, scores = [], [], [], [], []
    amp = mixed_precision and device.type == "cuda"
    for x, y, snr in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
            batch_loss = torch.nn.functional.cross_entropy(logits, y, reduction="none")
        probabilities = torch.softmax(logits.float(), dim=1)
        if not torch.isfinite(probabilities).all() or not torch.isfinite(batch_loss).all():
            raise RuntimeError("Nonfinite evaluation output.")
        labels.append(y.cpu().numpy())
        predicted.append(probabilities.argmax(dim=1).cpu().numpy())
        snrs.append(snr.numpy())
        losses.append(batch_loss.float().cpu().numpy())
        scores.append(probabilities.cpu().numpy())
    raw = {"labels": np.concatenate(labels), "predictions": np.concatenate(predicted),
           "snrs": np.concatenate(snrs), "scores": np.concatenate(scores)}
    metrics = summarize_predictions(raw["labels"], raw["predictions"], raw["snrs"],
                                    classes, good_snr_levels, np.concatenate(losses))
    return metrics, raw


def plot_confusion_matrix(metrics, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrix = np.asarray(metrics["confusion_matrix"])
    classes = metrics["classes"]
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Top-1 confusion matrix | accuracy={metrics['accuracy']:.1%}")
    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] > threshold else "black")
    fig.colorbar(image, ax=ax, label="Window count")
    fig.savefig(output, dpi=150)
    plt.close(fig)
