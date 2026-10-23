import argparse
import json
from amc.long_evaluation import evaluate_long_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/amc_baseline_v1.pt")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    result = evaluate_long_dataset(args.checkpoint, args.dataset, args.output, args.device)
    print(json.dumps({"count": result["count"], "middle_accuracy": result["middle_only"]["accuracy"],
                      "five_accuracy": result["five_window"]["accuracy"],
                      "middle_good_snr_accuracy": result["middle_only"]["good_snr_accuracy"],
                      "five_good_snr_accuracy": result["five_window"]["good_snr_accuracy"],
                      "paired_comparison": result["paired_comparison"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
