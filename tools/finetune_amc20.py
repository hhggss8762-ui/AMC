"""Run the agreed single AMC20 fine tune; retain best.pt only."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="4"

import argparse
from amc.config import load_config
from amc.finetuning20 import finetune20


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/amc20_v1.yaml")
    p.add_argument("--dataset",default="data/amc20_v1")
    p.add_argument("--source",default="models/amc_baseline_v1.pt")
    p.add_argument("--output",default="runs/amc20_v1")
    p.add_argument("--report",default="reports/amc20_v1_training")
    a=p.parse_args()
    finetune20(load_config(a.config),a.dataset,a.source,a.output,a.report)


if __name__=="__main__":
    main()
