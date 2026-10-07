#!/usr/bin/env python3
"""Render the explicitly synthetic README illustration without AIS data."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("assets/forecasting_example.png")
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 11, "axes.titleweight": "bold"}
    )
    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=160)
    fig.patch.set_facecolor("#f5f8fb")
    ax.set_facecolor("#f5f8fb")

    observed_x = np.linspace(-12, 0, 18)
    observed_y = 0.12 * observed_x + 0.45 * np.sin(observed_x / 4)
    t = np.linspace(0, 1, 73)
    east = 42 * t
    future = 4 * t + 17 * t**2 - 5 * t**3
    direct = future - 7 * t**2
    conditioned = future + 7 * t**2

    # These curves are illustrative constructions, never trained predictions.
    for index in range(8):
        sampled = future + 12 * np.sin(index * 1.7) * t**1.6
        ax.plot(
            east,
            sampled,
            color="#8b6cb5",
            alpha=0.20,
            linewidth=1.3,
            label="Sampled futures · illustrative" if index == 0 else None,
        )
    ax.plot(east, direct, color="#2678b8", linewidth=2.8, label="B0 · illustrative")
    ax.plot(
        east, conditioned, color="#d37735", linewidth=2.8, label="B3A · illustrative"
    )
    ax.plot(
        east, future, color="#172e43", linewidth=3.2, label="Future path · synthetic"
    )
    ax.plot(
        observed_x,
        observed_y,
        color="#18958b",
        linewidth=3.8,
        marker="o",
        markersize=3,
        label="Observed path · synthetic",
    )
    ax.scatter(
        [0], [0], s=75, color="#172e43", edgecolors="white", linewidths=1.6, zorder=6
    )
    ax.annotate(
        "Forecast origin",
        (0, 0),
        xytext=(-9, 5),
        arrowprops={"arrowstyle": "->", "color": "#63788a"},
        fontsize=10,
        color="#63788a",
    )

    fig.text(
        0.09,
        0.94,
        "From observed motion to possible futures",
        fontsize=20,
        fontweight="bold",
        color="#172e43",
    )
    fig.text(
        0.09,
        0.895,
        "90-minute history  /  6-hour forecast  /  shared evaluation",
        fontsize=11,
        color="#63788a",
    )
    fig.text(
        0.09,
        0.045,
        "SYNTHETIC ILLUSTRATION  ·  Constructed curves, not AIS observations or model results",
        fontsize=9,
        color="#63788a",
    )
    ax.set_xlabel("Relative eastward position (nmi)", color="#63788a")
    ax.set_ylabel("Relative northward position (nmi)", color="#63788a")
    ax.set_xlim(-15, 46)
    ax.set_ylim(-6, 27)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#dce5ed", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#63788a", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.96, top=0.83, bottom=0.16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote synthetic illustration to {args.output}")


if __name__ == "__main__":
    main()
