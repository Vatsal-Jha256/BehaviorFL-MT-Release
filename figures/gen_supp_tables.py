#!/usr/bin/env python3
"""Build ablation and per-source result tables."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_results_dir = Path(os.environ.get("BEHAVIORFL_RESULTS_DIR", "results"))
RESULTS = _results_dir if _results_dir.is_absolute() else REPO / _results_dir
OUT = Path(__file__).resolve().parent
SEEDS = (7, 42, 123)
HELD = "heldout_evaluation"


def vals(method: str, *path: str) -> np.ndarray:
    out = []
    for seed in SEEDS:
        d = json.loads((RESULTS / f"{method}_seed{seed}.json").read_text())
        for k in path:
            d = d[k]
        out.append(float(d))
    return np.asarray(out)


def cell(method: str, *path: str) -> str:
    v = vals(method, *path)
    return f"{v.mean():.3f} $\\pm$ {v.std(ddof=1):.3f}"


ABLATION = [
    ("Full \\method", "behaviorfl_mt"),
    ("Uniform sampling (no discovery)", "joint_tierdrop"),
    ("No sensor-block dropout", "behaviorfl_mt_no_dropout"),
    ("Zero-fill", "behaviorfl_mt_zero_fill"),
    ("Batch mean-fill", "behaviorfl_mt_mean_fill"),
    ("No residual (bound 0)", "behaviorfl_mt_no_residual"),
    ("Oracle eligibility", "oracle_task_aware"),
]
rows = []
for label, m in ABLATION:
    rows.append(
        f"{label} & {cell(m, HELD, 'state_natural_sensors', 'auprc')} & "
        f"{cell(m, HELD, 'state_gps_only', 'auprc')} & "
        f"{cell(m, HELD, 'destination', 'macro_f1')} & "
        f"{cell(m, HELD, 'destination', 'hr_at_5')}"
    )
(OUT / "supp_ablation_rows.tex").write_text(" \\\\\n".join(rows) + "\n")

prior = json.loads((RESULTS / f"behaviorfl_mt_seed{SEEDS[0]}.json").read_text())[HELD]
src_rows = []
for src, name in (("geolife", "GeoLife"), ("tdrive", "T-Drive"), ("ved", "VED")):
    n = prior["destination_by_source"][src]["n"]
    p = prior["destination_geometry_prior_by_source"][src]["hr_at_5"]
    src_rows.append(
        f"{name} & {n:,} & {p:.3f} & "
        f"{cell('behaviorfl_mt', HELD, 'destination_by_source', src, 'hr_at_5')} & "
        f"{cell('behaviorfl_mt', HELD, 'destination_by_source', src, 'macro_f1')}"
    )
(OUT / "supp_source_rows.tex").write_text(" \\\\\n".join(src_rows) + "\n")
print("wrote supp_ablation_rows.tex, supp_source_rows.tex")
