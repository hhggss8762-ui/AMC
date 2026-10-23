import argparse
import json
from amc.config import load_config
from amc.training import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--dataset", default="data/preview_ofdm_v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    summary = train(load_config(args.config), args.dataset, args.output,
                    args.epochs, args.batch_size, args.device)
    print(json.dumps({k: v for k, v in summary.items() if k != "environment"}, indent=2))


if __name__ == "__main__":
    main()
