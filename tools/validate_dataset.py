import argparse
import json
from pathlib import Path
from amc.dataset import validate_dataset, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="data/preview_ofdm_v1")
    parser.add_argument("--report", default="reports/preview_ofdm_validation.json")
    args = parser.parse_args()
    report = validate_dataset(args.dataset)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
