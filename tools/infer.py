import argparse
import json
from pathlib import Path
import numpy as np
from amc.dataset import write_json
from amc.inference import WindowClassifier


def main():
    parser = argparse.ArgumentParser(description="Classify preprocessed 2x2048 IQ windows.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="A float I/Q .npy array, shape [N,2,2048] or [2,2048].")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--score-threshold", type=float, default=0.6)
    args = parser.parse_args()
    windows = np.load(args.input, mmap_mode="r", allow_pickle=False)
    if windows.ndim == 2:
        windows = windows[None]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("Limit must be positive.")
        windows = windows[:args.limit]
    classifier = WindowClassifier(args.checkpoint, args.device, args.score_threshold)
    predictions = classifier.predict(windows, args.batch_size)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite inference output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {
        "input": str(Path(args.input).resolve()), "checkpoint": str(Path(args.checkpoint).resolve()),
        "classes": classifier.classes, "score_threshold": args.score_threshold,
        "checkpoint_dataset_profile": classifier.checkpoint["environment"]["dataset_profile"],
        "predictions": predictions,
        "note": "Softmax scores are not calibrated correctness probabilities; uncertain does not prove unknown modulation.",
    })
    print(json.dumps({"windows": len(predictions), "output": str(output.resolve()),
                      "classified": sum(p["status"] == "classified" for p in predictions),
                      "uncertain": sum(p["status"] == "uncertain" for p in predictions)}, indent=2))


if __name__ == "__main__":
    main()
