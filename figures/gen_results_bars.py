#!/usr/bin/env python3
"""Plot method comparisons and component ablations from saved runs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import BAR_LW, COLORS, panel_label, save_figure

REPO = Path(__file__).resolve().parents[1]
_results_dir = Path(os.environ.get("BEHAVIORFL_RESULTS_DIR", "results"))
RESULTS = _results_dir if _results_dir.is_absolute() else REPO / _results_dir
OUT_DIR = Path(__file__).resolve().parent
SEEDS = (7, 42, 123)
HELD = "heldout_evaluation"
NAT = (HELD, "state_natural_sensors", "auprc")
GPS = (HELD, "state_gps_only", "auprc")
HR5 = (HELD, "destination", "hr_at_5")
F1 = (HELD, "destination", "macro_f1")
# Positive rate of eligible held-out VED trips (3,174 trips): the AUPRC of a
# random scorer. Computed from data/processed/behaviorfl_dataset.pt.
STATE_POSITIVE_RATE = 0.277


def seed_values(method: str, path: tuple[str, ...]) -> np.ndarray:
    out = []
    for seed in SEEDS:
        data = json.loads((RESULTS / f"{method}_seed{seed}.json").read_text())
        if data.get("n_rounds") != 60:
            raise ValueError(f"Unexpected round count for {method} seed {seed}")
        for key in path:
            data = data[key]
        out.append(float(data))
    return np.asarray(out)


# Heads that a method does not train are left blank rather than plotted.
NO_STATE_HEAD = {"prior", "destination_specialist"}
NO_DEST_HEAD = {"state_specialist"}


def draw_bars(ax, rows, series, offsets, height, hatches, colors):
    """rows: method keys, drawn top to bottom; series: metric paths."""
    y = np.arange(len(rows), dtype=float)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for path, off, hatch, color in zip(series, offsets, hatches, colors):
        for yi, method in zip(y, rows):
            if path in (NAT, GPS) and method in NO_STATE_HEAD:
                continue
            if path in (HR5, F1) and method in NO_DEST_HEAD:
                continue
            if method == "prior":
                d = json.loads((RESULTS / f"destination_specialist_seed{SEEDS[0]}.json").read_text())
                v = d[HELD]["destination_geometry_prior"]["macro_f1" if path == F1 else "hr_at_5"]
                mean, sd, pts = v, 0.0, np.array([v])
            else:
                try:
                    vals = seed_values(method, path)
                except KeyError:
                    continue
                mean, sd, pts = vals.mean(), vals.std(ddof=1), vals
            ax.barh(yi - off, mean, height=height, color=color, edgecolor="black",
                    linewidth=BAR_LW, hatch=hatch, zorder=3)
            if sd > 0:
                ax.errorbar(mean, yi - off, xerr=sd, color="black", capsize=1.8,
                            elinewidth=0.8, capthick=0.8, zorder=4)
                ax.scatter(pts, np.full(len(pts), yi - off), s=5, color="black", zorder=5, linewidths=0)
    return y


def main_figure() -> None:
    labels = [
        ("Fixed geometry prior", "prior"),
        ("Destination specialist", "destination_specialist"),
        ("State specialist$^\\dagger$", "state_specialist"),
        ("FedAvg (Joint)", "joint_no_tierdrop"),
        ("MAS", "mas"),
        ("FedHCA$^2$", "fedhca2"),
        ("BehaviorFL-MT", "behaviorfl_mt"),
        ("BehaviorFL-MT (oracle)$^\\ddagger$", "oracle_task_aware"),
    ]
    rows = [m for _, m in labels]
    h = 0.34
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(5.5, 2.5), sharey=True,
                                   gridspec_kw={"wspace": 0.06})

    y = draw_bars(axa, rows, [NAT, GPS], [h / 2, -h / 2], h, ["", "//"],
                  [COLORS["blue"], COLORS["gold"]])
    axa.axvline(STATE_POSITIVE_RATE, color=COLORS["red"], linestyle="--", linewidth=1.1, zorder=2)
    axa.set_xlim(0, 0.8)
    axa.set_xticks([0, 0.2, 0.4, 0.6])
    axa.set_xlabel("State AUPRC (held-out VED)")
    axa.set_yticks(y)
    axa.set_yticklabels([l for l, _ in labels])
    axa.get_yticklabels()[rows.index("behaviorfl_mt")].set_fontweight("bold")
    panel_label(axa, "(a) Powertrain state")

    draw_bars(axb, rows, [HR5, F1], [h / 2, -h / 2], h, ["", "//"],
              [COLORS["teal"], COLORS["gray"]])
    axb.set_xlim(0, 1.0)
    axb.set_xlabel("Destination score (all held-out)")
    axb.tick_params(axis="y", length=0)
    panel_label(axb, "(b) Destination region")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles_a = [
        Patch(facecolor=COLORS["blue"], edgecolor="black", linewidth=BAR_LW, label="Natural sensors"),
        Patch(facecolor=COLORS["gold"], edgecolor="black", linewidth=BAR_LW, hatch="//", label="GPS-only"),
        Line2D([0], [0], color=COLORS["red"], linestyle="--", linewidth=1.1, label="Random scorer (0.277)"),
    ]
    handles_b = [
        Patch(facecolor=COLORS["teal"], edgecolor="black", linewidth=BAR_LW, label="HR@5"),
        Patch(facecolor=COLORS["gray"], edgecolor="black", linewidth=BAR_LW, hatch="//", label="Macro-F1"),
    ]
    axa.legend(handles=handles_a, loc="upper center", bbox_to_anchor=(0.5, -0.19),
                       ncol=3, fontsize=6.5, columnspacing=0.8, handlelength=1.4)
    axb.legend(handles=handles_b, loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2,
               fontsize=6.5, columnspacing=0.8, handlelength=1.4)
    save_figure(fig, OUT_DIR / "results_bars.pdf")
    plt.close(fig)


def ablation_figure() -> None:
    labels = [
        ("Full BehaviorFL-MT", "behaviorfl_mt"),
        ("$-$ discovery (uniform)", "joint_tierdrop"),
        ("$-$ block dropout", "behaviorfl_mt_no_dropout"),
        ("Zero-fill", "behaviorfl_mt_zero_fill"),
        ("Batch mean-fill", "behaviorfl_mt_mean_fill"),
        ("$-$ residual (bound 0)", "behaviorfl_mt_no_residual"),
    ]
    rows = [m for _, m in labels]
    h = 0.34
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(5.5, 2.15), sharey=True,
                                   gridspec_kw={"wspace": 0.06, "width_ratios": [1.15, 1]})
    y = draw_bars(axa, rows, [NAT, GPS], [h / 2, -h / 2], h, ["", "//"],
                  [COLORS["blue"], COLORS["gold"]])
    axa.set_xlim(0, 0.8)
    axa.set_xticks([0, 0.2, 0.4, 0.6])
    axa.set_xlabel("State AUPRC (held-out VED)")
    axa.set_yticks(y)
    axa.set_yticklabels([l for l, _ in labels])
    axa.get_yticklabels()[0].set_fontweight("bold")
    panel_label(axa, "(a) Powertrain state")

    draw_bars(axb, rows, [F1], [0.0], 0.5, [""], [COLORS["teal"]])
    axb.set_xlim(0, 0.5)
    axb.set_xlabel("Destination macro-F1")
    axb.tick_params(axis="y", length=0)
    panel_label(axb, "(b) Destination region")

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=COLORS["blue"], edgecolor="black", linewidth=BAR_LW, label="Natural sensors"),
        Patch(facecolor=COLORS["gold"], edgecolor="black", linewidth=BAR_LW, hatch="//", label="GPS-only"),
    ]
    axa.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
               fontsize=6.5, columnspacing=0.8, handlelength=1.4)
    save_figure(fig, OUT_DIR / "ablation_bars.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main_figure()
    ablation_figure()
