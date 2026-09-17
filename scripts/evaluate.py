#!/usr/bin/env python3
"""Evaluate any registered forecasting model with shared metrics and artifacts."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting.runner import evaluate_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline or proposed model through one experiment runner."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Training run containing resolved_design.json and its checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Explicit state-dict path. Usually --run-dir is preferable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination; defaults to the supplied run directory.",
    )
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--confirm-frozen-design",
        action="store_true",
        help="Required for test evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or args.run_dir
    if output_dir is None:
        raise ValueError("Evaluation requires --output-dir when --run-dir is omitted.")
    output, metrics = evaluate_experiment(
        args.config,
        output_dir,
        args.split,
        args.device,
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        confirm_frozen_design=args.confirm_frozen_design,
    )
    print(f"Wrote {args.split} evaluation to {output_dir}")
    if output.probabilities.shape[1] > 1:
        print(
            "6 h probability-weighted FDE: "
            f"{metrics['probability_weighted_fde_nmi'][-1]:.3f} nmi"
        )
    else:
        print(f"6 h top-1 FDE: {metrics['top1_fde_nmi'][-1]:.3f} nmi")


if __name__ == "__main__":
    main()
