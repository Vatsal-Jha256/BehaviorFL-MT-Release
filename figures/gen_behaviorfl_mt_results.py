#!/usr/bin/env python3
"""Plot destination and state scores from saved runs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import COLORS, save_figure


REPO = Path(__file__).resolve().parents[1]
_results_dir = Path(os.environ.get("BEHAVIORFL_RESULTS_DIR", "results"))
RESULTS = _results_dir if _results_dir.is_absolute() else REPO / _results_dir
OUTPUT = Path(__file__).resolve().parent / "behaviorfl_mt_results.pdf"
SEEDS = (7, 42, 123)
METHODS = {
    "destination_specialist": "Destination\nspecialist",
    "state_specialist": "State\nspecialist",
    "joint_no_tierdrop": "Joint",
    "joint_tierdrop": "Joint\n+ TierDrop",
    "behaviorfl_mt": "BehaviorFL-\nMT",
}


def load_run(method: str, seed: int) -> dict:
    path = RESULTS / f"{method}_seed{seed}.json"
    data = json.loads(path.read_text())
    metadata = data.get("dataset_metadata", {})
    if data.get("n_rounds") != 60 or metadata.get("schema") != "behaviorfl_multitask_v4":
        raise ValueError(f"Unexpected protocol in {path}")
    return data


def values(method: str, path: tuple[str, ...]) -> np.ndarray:
    out = []
    for seed in SEEDS:
        value = load_run(method, seed)
        for key in path:
            value = value[key]
        out.append(float(value))
    return np.asarray(out)


def mean_std(method: str, path: tuple[str, ...]) -> tuple[float, float]:
    data = values(method, path)
    return float(data.mean()), float(data.std(ddof=1))


def main() -> None:
    destination_methods = (
        "destination_specialist", "joint_no_tierdrop",
        "joint_tierdrop", "behaviorfl_mt",
    )
    destination_path = ("heldout_evaluation", "destination", "macro_f1")
    geometry_values = np.asarray([
        load_run(method, seed)["heldout_evaluation"]["destination_geometry_prior"]["macro_f1"]
        for method in destination_methods for seed in SEEDS
    ])
    if not np.allclose(geometry_values, geometry_values[0], rtol=0.0, atol=1e-12):
        raise ValueError("Deterministic geometry baseline differs across runs")
    geometry = float(geometry_values[0])

    state_methods = (
        "state_specialist", "joint_no_tierdrop",
        "joint_tierdrop", "behaviorfl_mt",
    )
    natural_path = ("heldout_evaluation", "state_natural_sensors", "auprc")
    gps_path = ("heldout_evaluation", "state_gps_only", "auprc")

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.72), gridspec_kw={"wspace": 0.38})
    fig.subplots_adjust(bottom=0.24, top=0.84)

    x = np.arange(len(destination_methods))
    for position, method in zip(x, destination_methods):
        per_seed = values(method, destination_path)
        axes[0].scatter(
            position + np.asarray([-0.055, 0.0, 0.055]), per_seed,
            s=14, facecolor="white", edgecolor=COLORS["blue"],
            linewidth=0.75, zorder=2,
        )
        mean, std = mean_std(method, destination_path)
        axes[0].errorbar(
            position, mean, yerr=std, fmt="s", markersize=4.8,
            color=COLORS["blue"], markeredgecolor="black", markeredgewidth=0.5,
            capsize=2.5, elinewidth=1.0, zorder=3,
        )
    axes[0].axhline(
        geometry, color=COLORS["red"], linestyle="--", linewidth=1.15,
        label="Fixed geometry prior",
    )
    axes[0].set_xticks(x, [METHODS[method] for method in destination_methods])
    axes[0].set_ylabel("Destination macro-F1")
    axes[0].set_title("Pooled held-out trips (all sources)", fontsize=8, pad=3)
    axes[0].set_ylim(0.25, 0.38)
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.55)
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].text(-0.13, 1.02, "(a)", transform=axes[0].transAxes, fontweight="bold")

    x = np.arange(len(state_methods))
    offsets = {"natural": -0.13, "gps": 0.13}
    for position, method in zip(x, state_methods):
        for name, path, color, marker in (
            ("natural", natural_path, COLORS["teal"], "o"),
            ("gps", gps_path, COLORS["gold"], "D"),
        ):
            per_seed = values(method, path)
            center = position + offsets[name]
            axes[1].scatter(
                center + np.asarray([-0.04, 0.0, 0.04]), per_seed,
                s=12, marker=marker, facecolor="white", edgecolor=color,
                linewidth=0.7, zorder=2,
            )
            mean, std = mean_std(method, path)
            axes[1].errorbar(
                center, mean, yerr=std, fmt=marker, markersize=4.5,
                color=color, markeredgecolor="black", markeredgewidth=0.45,
                capsize=2.3, elinewidth=1.0, zorder=3,
                label=("Natural sensors" if name == "natural" else "Forced GPS-only")
                if position == 0 else None,
            )
    axes[1].set_xticks(x, [METHODS[method] for method in state_methods])
    axes[1].set_ylabel("State AUPRC")
    axes[1].set_title(
        "Eligible held-out VED trips\n(same trips and checkpoint)",
        fontsize=8, pad=3,
    )
    axes[1].set_ylim(0.2, 0.8)
    axes[1].grid(axis="y", color="#DDDDDD", linewidth=0.55)
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].text(-0.13, 1.02, "(b)", transform=axes[1].transAxes, fontweight="bold")

    fig.text(
        0.5, 0.025,
        "Hollow markers: individual seeds 7/42/123; filled markers and bars: mean $\\pm$ sample SD ($n=3$).",
        ha="center", va="bottom", fontsize=7.2,
    )

    save_figure(fig, OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
