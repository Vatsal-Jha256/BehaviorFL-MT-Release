#!/usr/bin/env python3
"""Plot periodic held-out scores and state-client discovery."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import COLORS, panel_label, save_figure

REPO = Path(__file__).resolve().parents[1]
_results_dir = Path(os.environ.get("BEHAVIORFL_RESULTS_DIR", "results"))
RESULTS = _results_dir if _results_dir.is_absolute() else REPO / _results_dir
OUTPUT = Path(__file__).resolve().parent / "convergence_discovery.pdf"
SEEDS = (7, 42, 123)

CONVERGENCE_METHODS = {
    "behaviorfl_mt": ("BehaviorFL-MT", COLORS["blue"], "-", "o"),
    "oracle_task_aware": ("Oracle eligibility", COLORS["teal"], "--", "D"),
    "mas": ("MAS", COLORS["dark"], "-", "s"),
    "fedhca2": (r"FedHCA$^2$", COLORS["red"], "-", "^"),
}


def load(method: str, seed: int) -> dict:
    path = RESULTS / f"{method}_seed{seed}.json"
    data = json.loads(path.read_text())
    if data.get("n_rounds") != 60:
        raise ValueError(f"Unexpected round count in {path}")
    if data.get("dataset_metadata", {}).get("schema") != "behaviorfl_multitask_v4":
        raise ValueError(f"Unexpected dataset schema in {path}")
    return data


def mean_trajectory(method: str, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean +/- sample-SD trajectory of a periodic_heldout_trajectory series
    across seeds, aligned on the (identical, sparse) evaluated rounds."""
    per_seed_rounds = []
    per_seed_values = []
    for seed in SEEDS:
        series = load(method, seed)["periodic_heldout_trajectory"][key]
        if not series:
            raise ValueError(
                f"{method}_seed{seed}.json has no periodic_heldout_trajectory[{key}]; "
                "rerun with --eval-every-n-rounds > 0"
            )
        rounds, values = zip(*series)
        per_seed_rounds.append(tuple(rounds))
        per_seed_values.append(values)
    if len(set(per_seed_rounds)) != 1:
        raise ValueError(f"{method}: evaluated rounds differ across seeds")
    rounds = np.asarray(per_seed_rounds[0], dtype=float)
    values = np.asarray(per_seed_values, dtype=float)  # (n_seeds, n_rounds)
    return rounds, values.mean(axis=0), values.std(axis=0, ddof=1)


def discovery_trajectory() -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Real cumulative discovered state-supervised-client counts for
    behaviorfl_mt (the only method with a discovery mechanism)."""
    per_seed = []
    for seed in SEEDS:
        data = load("behaviorfl_mt", seed)
        series = data["known_state_clients"]
        if not series:
            raise ValueError("behaviorfl_mt result has no known_state_clients series")
        per_seed.append(series)
    per_seed = np.asarray(per_seed, dtype=float)  # (n_seeds, n_rounds)
    rounds = np.arange(1, per_seed.shape[1] + 1)

    # Total eligible = training clients with any state-supervised trip, read
    # directly from the frozen dataset used by this batch (not hardcoded).
    import torch
    dataset_path = REPO / "data" / "processed" / "behaviorfl_dataset.pt"
    d = torch.load(dataset_path, map_location="cpu", weights_only=False)
    splits = d["client_splits"]
    mask = d["state_supervision_mask"]
    n_eligible = sum(
        1 for cid, idx in splits.items()
        if mask[torch.as_tensor(idx, dtype=torch.long)].any()
    )
    return rounds, per_seed.mean(axis=0), per_seed.std(axis=0, ddof=1), n_eligible


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.15), gridspec_kw={"wspace": 0.3})

    for method, (label, color, style, marker) in CONVERGENCE_METHODS.items():
        rounds, mean, std = mean_trajectory(method, "state_auprc_natural")
        axes[0].plot(rounds, mean, label=label, color=color, linestyle=style, marker=marker,
                     markersize=3.2, markeredgecolor="black", markeredgewidth=0.5, zorder=3)
        axes[0].fill_between(rounds, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
    axes[0].set_xlabel("Communication round")
    axes[0].set_ylabel("Natural-sensor state AUPRC")
    axes[0].set_xlim(0, 61)
    axes[0].set_ylim(0.15, 0.72)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=6.5,
                   columnspacing=0.9, handlelength=1.8)
    panel_label(axes[0], "(a) Convergence")

    rounds, mean, std, n_eligible = discovery_trajectory()
    axes[1].plot(rounds, mean, color=COLORS["blue"], label="Discovered by BehaviorFL-MT", zorder=3)
    axes[1].fill_between(rounds, mean - std, mean + std, color=COLORS["blue"], alpha=0.18, linewidth=0)
    axes[1].axhline(y=n_eligible, color=COLORS["red"], linestyle="--", linewidth=1.2,
                    label=f"All eligible clients ({n_eligible})", zorder=2)
    axes[1].set_xlabel("Communication round")
    axes[1].set_ylabel("Cumulative state-supervised clients")
    axes[1].set_xlim(0, 61)
    axes[1].set_ylim(0, 205)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, fontsize=6.5,
                   handlelength=1.8)
    panel_label(axes[1], "(b) Online discovery")

    save_figure(fig, OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
