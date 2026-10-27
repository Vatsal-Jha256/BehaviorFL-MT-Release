"""Shared colors, fonts, and axes settings for result plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

COLORS = {
    "blue": "#2A7FD4",
    "gold": "#F5C518",
    "teal": "#1BA39C",
    "red": "#D9372B",
    "gray": "#B8BDC4",
    "dark": "#4D5560",
}

FRAME_LW = 1.2
BAR_LW = 0.9

plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "axes.titleweight": "bold",
    "axes.linewidth": FRAME_LW,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.axisbelow": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": "black",
    "legend.framealpha": 1.0,
    "legend.borderpad": 0.4,
    "patch.linewidth": BAR_LW,
    "lines.linewidth": 1.6,
    "grid.color": "0.85",
    "grid.linewidth": 0.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def panel_label(ax: Axes, text: str) -> None:
    ax.text(0.0, 1.03, text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, fontweight="bold")


def save_figure(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf")
    print(f"Saved {path}")
