#!/usr/bin/env python3
"""Train any registered forecasting model with the shared lifecycle."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting.runner import train_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a baseline or proposed model through one experiment runner."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Model config name, for example traisformer or vessel_direct_b3a.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one epoch on small train/validation slices; not a research result.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = train_experiment(
        args.config,
        args.run_dir,
        args.device,
        seed=args.seed,
        smoke=args.smoke,
    )
    print(f"Wrote training run to {run_dir}")


if __name__ == "__main__":
    main()
