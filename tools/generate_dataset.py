import argparse
import json
from pathlib import Path
from amc.config import load_config
from amc.dataset import create_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate a small preview by default.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--profile", choices=("preview", "full"), default="preview")
    parser.add_argument("--output", default="data/preview_ofdm_v1")
    args = parser.parse_args()
    if args.profile == "full" and args.output == "data/preview_ofdm_v1":
        parser.error("Full generation requires an explicit, separate --output path.")
    config = load_config(args.config)
    print(json.dumps(create_dataset(config, Path(args.output), args.profile),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
