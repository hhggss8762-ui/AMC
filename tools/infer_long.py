import argparse
import json
from pathlib import Path
from amc.dataset import write_json
from amc.long_inference import LongIQClassifier, load_long_input


def main():
    parser = argparse.ArgumentParser(description="Classify one continuous raw IQ signal, using supplied acquisition parameters.")
    parser.add_argument("--checkpoint", default="models/amc_baseline_v1.pt")
    parser.add_argument("--input", required=True, help="NPZ: raw_iq and input_metadata_json, no true labels.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--score-threshold", type=float, default=0.6)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    raw, inputs = load_long_input(args.input)
    classifier = LongIQClassifier(args.checkpoint, args.device, args.score_threshold)
    result = classifier.predict(raw, **inputs)
    result.update({"input": str(Path(args.input).resolve()),
                   "checkpoint": str(Path(args.checkpoint).resolve()),
                   "note": "Scores are not calibrated probabilities; uncertain does not prove unknown modulation."})
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps({key: result.get(key) for key in (
        "class_name", "score", "status", "window_count", "consistency", "timing_ms")}, indent=2))


if __name__ == "__main__":
    main()
