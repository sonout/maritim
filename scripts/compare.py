#!/usr/bin/env python3
"""Paired comparison of two standardized prediction artifacts."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import haversine_nmi
from forecasting.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_a", type=Path)
    parser.add_argument("artifact_b", type=Path)
    parser.add_argument("--dataset", default="ct_dma")
    parser.add_argument(
        "--prediction-rule",
        choices=("top1", "expected"),
        default="top1",
        help=(
            "Candidate statistic to compare. Use 'expected' for the corrected "
            "TrAISformer mean-of-rollouts result."
        ),
    )
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args()


def _load(path):
    with np.load(path) as saved:
        required = {
            "predictions",
            "probabilities",
            "targets",
            "target_mask",
            "trajectory_ids",
            "vessel_ids",
        }
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"{path} is missing standardized fields: {sorted(missing)}")
        return {name: saved[name] for name in required}


def _forecast_errors(artifact, roi_min, roi_range, prediction_rule):
    predictions = artifact["predictions"]
    probabilities = artifact["probabilities"]
    if predictions.ndim != 4 or predictions.shape[0] != len(probabilities):
        raise ValueError("Predictions must use [examples, modes, steps, coordinates].")
    predictions_deg = predictions * roi_range + roi_min
    targets_deg = artifact["targets"] * roi_range + roi_min
    if prediction_rule == "top1":
        winners = probabilities.argmax(axis=1)
        selected = predictions_deg[np.arange(len(predictions)), winners]
        return haversine_nmi(targets_deg, selected)
    mode_errors = haversine_nmi(
        targets_deg[:, None, :, :], predictions_deg
    )
    return (mode_errors * probabilities[:, :, None]).sum(axis=1)


def main():
    args = parse_args()
    first, second = _load(args.artifact_a), _load(args.artifact_b)
    for field in ("trajectory_ids", "targets", "target_mask"):
        if not np.array_equal(first[field], second[field]):
            raise ValueError(f"Artifacts differ in {field}; paired comparison is invalid.")
    data = load_config(args.dataset, "dataset")
    roi_min = np.asarray([data.lat_min, data.lon_min])
    roi_range = np.asarray([data.lat_max - data.lat_min, data.lon_max - data.lon_min])
    error_a = _forecast_errors(first, roi_min, roi_range, args.prediction_rule)
    error_b = _forecast_errors(second, roi_min, roi_range, args.prediction_rule)
    cadence = int(data.cadence_seconds)
    rng = np.random.default_rng(args.bootstrap_seed)
    print(
        f"Paired {args.prediction_rule} FDE (A - B; negative favors A)"
    )
    for hours in (1, 2, 3, 6):
        step = hours * 3600 // cadence - 1
        valid = first["target_mask"][:, step].astype(bool)
        a, b = error_a[valid, step], error_b[valid, step]
        difference = a - b
        indices = rng.integers(0, len(difference), (args.bootstrap_resamples, len(difference)))
        ci = np.quantile(difference[indices].mean(axis=1), (0.025, 0.975))
        relative = difference.mean() / b.mean() * 100.0
        print(
            f"{hours} h: A={a.mean():.3f}, B={b.mean():.3f}, "
            f"delta={difference.mean():+.3f} nmi ({relative:+.1f}%), "
            f"CI=[{ci[0]:+.3f}, {ci[1]:+.3f}], A wins={(a < b).mean() * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
