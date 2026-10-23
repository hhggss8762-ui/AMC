import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.run)
    rows = [json.loads(line) for line in (root / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(epochs, [r["train_loss"] for r in rows], marker="o", label="Train")
    axes[0].plot(epochs, [r["validation_loss"] for r in rows], marker="o", label="Validation")
    axes[0].set_ylabel("Cross-entropy loss")
    for key, label in (("train_accuracy", "Train"), ("validation_accuracy", "Validation"),
                       ("validation_good_snr_accuracy", "Validation: good SNR")):
        axes[1].plot(epochs, [100*r[key] for r in rows], marker="o", label=label)
    axes[1].set_ylabel("Top-1 accuracy (%)")
    axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle(f"{environment['dataset_profile']} run | train={environment['train_windows']}, "
                 f"validation={environment['validation_windows']} | test excluded from selection")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
