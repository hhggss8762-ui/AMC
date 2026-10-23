import argparse
import json
from amc.config import load_config
from amc.long_dataset import create_long_dataset, load_long_settings


def main():
    parser = argparse.ArgumentParser(description="Generate independent continuous raw IQ, not stitched windows.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--settings", default="configs/long_signal_eval.yaml")
    parser.add_argument("--profile", choices=("preview", "full"), default="preview")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = create_long_dataset(load_config(args.config), load_long_settings(args.settings),
                                 args.output, args.profile)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
