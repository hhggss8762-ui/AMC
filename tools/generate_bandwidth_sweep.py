import argparse
import json
from amc.config import load_config
from amc.wideband_dataset import create_wideband_dataset, load_wideband_settings


def main():
    parser = argparse.ArgumentParser(description="100 kHz–20 MHz continuous IQ bandwidth sweep; frozen model.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--settings", default="configs/wideband_eval.yaml")
    parser.add_argument("--profile", choices=("preview", "full"), default="preview")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = create_wideband_dataset(load_config(args.config), load_wideband_settings(args.settings),
                                     args.output, args.profile)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
