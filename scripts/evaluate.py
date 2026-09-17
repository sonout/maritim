#!/usr/bin/env python3
"""Evaluate any registered forecasting model with shared metrics and artifacts."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting.runner import evaluate_experiment


REPORT_HOURS = (1, 2, 3, 6)


def _format_value(value, decimals=3):
    value = float(value)
    if value != value:
        return "n/a"
    return f"{value:.{decimals}f}"


def _print_table(headers, rows, *, left_columns=()):
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in text_rows))
        for index, header in enumerate(headers)
    ]
    print(
        "  ".join(
            header.ljust(widths[index]) for index, header in enumerate(headers)
        )
    )
    print("  ".join("-" * width for width in widths))
    for row in text_rows:
        print(
            "  ".join(
                (
                    value.ljust(widths[index])
                    if index in left_columns
                    else value.rjust(widths[index])
                )
                for index, value in enumerate(row)
            )
        )


def print_metric_summary(metrics, mode_count):
    """Print the principal deployable, probabilistic, and oracle diagnostics."""
    seconds_to_index = {
        int(seconds): index for index, seconds in enumerate(metrics["horizon_seconds"])
    }
    selected = [
        (hours, seconds_to_index[hours * 3600])
        for hours in REPORT_HOURS
        if hours * 3600 in seconds_to_index
    ]
    if not selected:
        final_index = len(metrics["horizon_seconds"]) - 1
        final_hours = float(metrics["horizon_seconds"][final_index]) / 3600.0
        selected = [(final_hours, final_index)]

    rows = []
    for hours, index in selected:
        horizon = f"{hours:g} h"
        rows.append(
            (
                horizon,
                int(metrics["valid_counts"][index]),
                _format_value(metrics["top1_fde_nmi"][index]),
                _format_value(metrics["probability_weighted_fde_nmi"][index]),
                _format_value(metrics["top1_ade_cumulative_nmi"][index]),
                _format_value(metrics["weighted_energy_score_nmi"][index]),
                _format_value(metrics["weighted_pairwise_spread_nmi"][index]),
                _format_value(metrics["min_fde_at_k_nmi"][index]),
            )
        )

    print("\nMain forecast metrics (nautical miles)")
    _print_table(
        (
            "Horizon",
            "Valid",
            "Top-1 FDE",
            "Expected FDE",
            "Top-1 ADE",
            "Energy",
            "Spread",
            "minFDE@K*",
        ),
        rows,
    )
    primary = "Expected FDE" if mode_count > 1 else "Top-1 FDE"
    print(f"\nHeadline point metric: {primary} (K={mode_count}).")
    print("FDE, ADE, and energy are lower-is-better; spread measures diversity.")
    print("* minFDE@K uses the ground truth to choose a mode and is oracle-only.")
    if mode_count > 1:
        print(
            "Top-1 is meaningful only when the model assigns ranked mode "
            "probabilities."
        )

    runtime_rows = [
        (
            "Complete-trajectory energy",
            f"{_format_value(metrics['trajectory_energy_score_nmi'])} nmi",
        ),
        (
            "Clipped predicted points",
            f"{float(metrics['predicted_point_clipping_fraction']) * 100:.2f}%",
        ),
        ("Inference time", f"{float(metrics['inference_seconds']):.2f} s"),
        (
            "Throughput",
            f"{float(metrics['trajectories_per_second']):.2f} trajectories/s",
        ),
        ("Parameters", f"{int(metrics['parameter_count']):,}"),
    ]
    print("\nRun summary")
    _print_table(("Metric", "Value"), runtime_rows, left_columns=(0,))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a baseline or proposed model through one experiment runner."
        )
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
    print_metric_summary(metrics, output.probabilities.shape[1])


if __name__ == "__main__":
    main()
