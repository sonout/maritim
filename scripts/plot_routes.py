#!/usr/bin/env python3
"""Visual comparison of two aligned trajectory-forecast artifacts."""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import haversine_nmi
from forecasting.config import load_config


REQUIRED_FIELDS = {
    "predictions",
    "probabilities",
    "targets",
    "target_mask",
    "trajectory_ids",
    "vessel_ids",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Overlay the aligned routes predicted by two models and the true route."
        )
    )
    parser.add_argument("artifact_a", type=Path)
    parser.add_argument("artifact_b", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("route_comparison.png"),
        help="Output image (.png, .pdf, or .svg; default: route_comparison.png).",
    )
    parser.add_argument("--dataset", default="ct_dma")
    parser.add_argument("--label-a", help="Legend label for artifact A.")
    parser.add_argument("--label-b", help="Legend label for artifact B.")
    parser.add_argument(
        "--prediction-rule",
        choices=("top1", "expected"),
        default="expected",
        help=(
            "Route to emphasize: highest-probability mode, or the probability-weighted "
            "mean route (default: expected)."
        ),
    )
    parser.add_argument(
        "--selection",
        choices=(
            "representative",
            "largest-disagreement",
            "a-wins",
            "b-wins",
            "random",
        ),
        default="representative",
        help="How to choose examples when --trajectory-ids is not supplied.",
    )
    parser.add_argument("--num-examples", type=int, default=9)
    parser.add_argument(
        "--trajectory-ids",
        type=int,
        nargs="+",
        help="Plot these exact trajectory IDs, preserving the supplied order.",
    )
    parser.add_argument(
        "--horizon-hours",
        type=float,
        default=6.0,
        help="Maximum forecast horizon to draw and use for automatic selection.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Seed for random selection."
    )
    parser.add_argument(
        "--hide-modes",
        action="store_true",
        help="Do not draw individual candidate modes as faint lines.",
    )
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_artifact(path):
    with np.load(path) as saved:
        missing = REQUIRED_FIELDS - set(saved.files)
        if missing:
            raise ValueError(f"{path} is missing standardized fields: {sorted(missing)}")
        artifact = {name: saved[name] for name in REQUIRED_FIELDS}
    predictions = artifact["predictions"]
    probabilities = artifact["probabilities"]
    if predictions.ndim != 4 or predictions.shape[-1] != 2:
        raise ValueError(f"{path}: predictions must have shape [N, K, T, 2].")
    if probabilities.shape != predictions.shape[:2]:
        raise ValueError(f"{path}: probabilities do not align with predictions.")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError(f"{path}: probabilities must be non-negative and sum to one.")
    return artifact


def representative_routes(artifact, prediction_rule):
    """Return one emphasized route per example while retaining modes for context."""
    if prediction_rule == "top1":
        winner = artifact["probabilities"].argmax(axis=1)
        return artifact["predictions"][np.arange(len(winner)), winner]
    return np.einsum(
        "nk,nktc->ntc", artifact["probabilities"], artifact["predictions"]
    )


def _endpoint_errors(artifact, roi_min, roi_range, step, prediction_rule):
    target = artifact["targets"][:, step] * roi_range + roi_min
    modes = artifact["predictions"][:, :, step] * roi_range + roi_min
    mode_errors = haversine_nmi(target[:, None, :], modes)
    if prediction_rule == "top1":
        winner = artifact["probabilities"].argmax(axis=1)
        return mode_errors[np.arange(len(winner)), winner]
    return (mode_errors * artifact["probabilities"]).sum(axis=1)


def select_examples(
    first,
    second,
    roi_min,
    roi_range,
    step,
    count,
    strategy,
    seed,
    prediction_rule,
    trajectory_ids=None,
):
    valid = first["target_mask"][:, step].astype(bool)
    valid_indices = np.flatnonzero(valid)
    if trajectory_ids is not None:
        by_id = {int(value): index for index, value in enumerate(first["trajectory_ids"])}
        missing = [value for value in trajectory_ids if value not in by_id]
        if missing:
            raise ValueError(f"Unknown trajectory IDs: {missing}")
        selected = np.asarray(
            [by_id[value] for value in trajectory_ids], dtype=np.int64
        )
        incomplete = [
            int(first["trajectory_ids"][index]) for index in selected if not valid[index]
        ]
        if incomplete:
            raise ValueError(
                f"Trajectories do not reach the requested horizon: {incomplete}"
            )
        return selected
    if count <= 0:
        raise ValueError("--num-examples must be positive.")
    if not len(valid_indices):
        raise ValueError("No trajectories reach the requested horizon.")
    count = min(count, len(valid_indices))
    if strategy == "random":
        return np.random.default_rng(seed).choice(valid_indices, count, replace=False)

    error_a = _endpoint_errors(first, roi_min, roi_range, step, prediction_rule)
    error_b = _endpoint_errors(second, roi_min, roi_range, step, prediction_rule)
    delta = error_a - error_b
    ordered = valid_indices[np.argsort(delta[valid_indices], kind="stable")]
    if strategy == "a-wins":
        return ordered[:count]
    if strategy == "b-wins":
        return ordered[-count:][::-1]
    if strategy == "largest-disagreement":
        return valid_indices[
            np.argsort(np.abs(delta[valid_indices]), kind="stable")[-count:][::-1]
        ]
    positions = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return ordered[positions]


def _plot_model(ax, modes, route, color, label, show_modes):
    if show_modes and len(modes) > 1:
        for mode in modes:
            ax.plot(mode[:, 1], mode[:, 0], color=color, alpha=0.14, linewidth=0.8)
    ax.plot(route[:, 1], route[:, 0], color=color, linewidth=2.2, label=label)
    ax.scatter(route[-1, 1], route[-1, 0], color=color, s=18, zorder=4)


def main():
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as error:
        raise SystemExit(
            "Route plotting requires matplotlib. Install it with `python -m pip install "
            "matplotlib`."
        ) from error

    first = load_artifact(args.artifact_a)
    second = load_artifact(args.artifact_b)
    for field in ("trajectory_ids", "vessel_ids", "targets", "target_mask"):
        if not np.array_equal(first[field], second[field]):
            raise ValueError(f"Artifacts differ in {field}; route comparison is invalid.")
    if first["predictions"].shape[0] != second["predictions"].shape[0]:
        raise ValueError("Artifacts contain different numbers of examples.")
    if first["predictions"].shape[2] != second["predictions"].shape[2]:
        raise ValueError("Artifacts use different forecast horizons.")

    data = load_config(args.dataset, "dataset")
    roi_min = np.asarray([data.lat_min, data.lon_min], dtype=np.float64)
    roi_range = np.asarray(
        [data.lat_max - data.lat_min, data.lon_max - data.lon_min], dtype=np.float64
    )
    max_steps = first["predictions"].shape[2]
    requested_steps = int(args.horizon_hours * 3600 // int(data.cadence_seconds))
    if requested_steps < 1:
        raise ValueError("--horizon-hours must include at least one forecast step.")
    drawn_steps = min(requested_steps, max_steps)
    selection_step = drawn_steps - 1

    route_a = representative_routes(first, args.prediction_rule)
    route_b = representative_routes(second, args.prediction_rule)
    selected = select_examples(
        first,
        second,
        roi_min,
        roi_range,
        selection_step,
        args.num_examples,
        args.selection,
        args.seed,
        args.prediction_rule,
        args.trajectory_ids,
    )
    label_a = args.label_a or args.artifact_a.parent.name or "Model A"
    label_b = args.label_b or args.artifact_b.parent.name or "Model B"
    columns = max(1, min(args.columns, len(selected)))
    rows = math.ceil(len(selected) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(5.0 * columns, 4.4 * rows), squeeze=False
    )
    colors = ("#0072B2", "#D55E00")

    for ax, index in zip(axes.flat, selected):
        valid_steps = min(drawn_steps, int(first["target_mask"][index].sum()))
        actual = first["targets"][index, :valid_steps] * roi_range + roi_min
        modes_a = first["predictions"][index, :, :valid_steps] * roi_range + roi_min
        modes_b = second["predictions"][index, :, :valid_steps] * roi_range + roi_min
        selected_a = route_a[index, :valid_steps] * roi_range + roi_min
        selected_b = route_b[index, :valid_steps] * roi_range + roi_min

        ax.plot(
            actual[:, 1], actual[:, 0], color="#202020", linewidth=2.6, label="Actual"
        )
        ax.scatter(actual[-1, 1], actual[-1, 0], color="#202020", s=20, zorder=4)
        _plot_model(ax, modes_a, selected_a, colors[0], label_a, not args.hide_modes)
        _plot_model(ax, modes_b, selected_b, colors[1], label_b, not args.hide_modes)
        ax.scatter(
            actual[0, 1],
            actual[0, 0],
            marker="o",
            facecolor="white",
            edgecolor="#202020",
            linewidth=1.2,
            s=30,
            zorder=5,
        )
        trajectory_id = int(first["trajectory_ids"][index])
        vessel_id = int(first["vessel_ids"][index])
        ax.set_title(f"Trajectory {trajectory_id}  |  vessel {vessel_id}", fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, color="#dddddd", linewidth=0.6)
        mean_latitude = float(np.mean(actual[:, 0]))
        ax.set_aspect(1.0 / max(np.cos(np.deg2rad(mean_latitude)), 1e-6))
        ax.margins(0.10)

    for ax in axes.flat[len(selected):]:
        ax.set_visible(False)

    handles = [
        Line2D([0], [0], color="#202020", linewidth=2.6, label="Actual future"),
        Line2D([0], [0], color=colors[0], linewidth=2.2, label=label_a),
        Line2D([0], [0], color=colors[1], linewidth=2.2, label=label_b),
    ]
    title_rule = (
        "top-1 route" if args.prediction_rule == "top1" else "weighted mean route"
    )
    shown_hours = drawn_steps * int(data.cadence_seconds) / 3600
    fig.suptitle(
        f"Forecast route comparison at up to {shown_hours:g} h ({title_rule})",
        fontsize=14,
    )
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    plotted_ids = ", ".join(str(int(first["trajectory_ids"][i])) for i in selected)
    print(f"Wrote {len(selected)} route comparisons to {args.output}")
    print(f"Trajectory IDs: {plotted_ids}")
    if args.prediction_rule == "expected":
        print(
            "The emphasized expected route is the probability-weighted coordinate mean; "
            "faint lines are the individual modes/rollouts."
        )


if __name__ == "__main__":
    main()
