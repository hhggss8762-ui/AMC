import argparse
import json
from pathlib import Path
from amc.dataset import write_json
from amc.long_dataset import validate_long_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--report", required=True)
    parser.add_argument("--prior-datasets", nargs="*", default=[
        "data/full_v1", "data/preview_ofdm_v1", "data/preview_v1", "data/long_signal_preview_v1",
        "data/long_signal_v1", "data/bandwidth_preview_v1"])
    args = parser.parse_args()
    priors = [p for p in args.prior_datasets if Path(p).resolve() != Path(args.dataset).resolve()]
    result = validate_long_dataset(args.dataset, priors)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
