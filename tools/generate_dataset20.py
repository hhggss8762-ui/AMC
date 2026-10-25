"""Generate and validate the separately versioned AMC20 dataset; no training."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import json
from pathlib import Path
from amc.dataset20 import load_config20, create_dataset20, validate_dataset20
from amc.dataset import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/amc20_v1.yaml")
    p.add_argument("--output", default="data/amc20_v1")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--resume", action="store_true", help="Continue only a matching incomplete dataset, preserving completed rows")
    args = p.parse_args()
    config = load_config20(args.config)
    if not args.validate_only:
        manifest = create_dataset20(config, args.output, args.workers, args.chunk_size, resume=args.resume)
        print(json.dumps({"generation_complete": True, "splits": manifest["splits"]}), flush=True)
    validation = validate_dataset20(args.output)
    write_json(Path(args.output)/"validation.json", validation)
    print(json.dumps({"validation_passed": validation["passed"], "total_windows": validation["total_windows"]}), flush=True)


if __name__ == "__main__":
    main()
