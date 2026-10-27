#!/usr/bin/env python3
"""Build the main result table from three-seed runs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
_results_dir = Path(os.environ.get("BEHAVIORFL_RESULTS_DIR", "results"))
RESULTS = _results_dir if _results_dir.is_absolute() else REPO / _results_dir
OUTPUT = Path(__file__).resolve().parent / "main_results_rows.tex"
SEEDS = (7, 42, 123)


def load(method: str, seed: int) -> dict:
    path = RESULTS / f"{method}_seed{seed}.json"
    data = json.loads(path.read_text())
    if data.get("n_rounds") != 60:
        raise ValueError(f"Unexpected round count in {path}")
    if data.get("dataset_metadata", {}).get("schema") != "behaviorfl_multitask_v4":
        raise ValueError(f"Unexpected dataset schema in {path}")
    return data


def metric(method: str, *keys: str) -> str:
    vals = []
    for seed in SEEDS:
        value = load(method, seed)
        for key in keys:
            value = value[key]
        vals.append(float(value))
    array = np.asarray(vals)
    return f"{array.mean():.3f} $\\pm$ {array.std(ddof=1):.3f}"


def main() -> None:
    held = ("heldout_evaluation",)
    destination = held + ("destination", "macro_f1")
    hr5 = held + ("destination", "hr_at_5")
    natural = held + ("state_natural_sensors", "auprc")
    gps = held + ("state_gps_only", "auprc")
    geometry_rows = [
        load(method, seed)["heldout_evaluation"]["destination_geometry_prior"]
        for method in ("destination_specialist", "joint_no_tierdrop", "joint_tierdrop", "behaviorfl_mt")
        for seed in SEEDS
    ]
    geometry_values = np.asarray([
        (row["macro_f1"], row["hr_at_5"]) for row in geometry_rows
    ])
    if not np.allclose(geometry_values, geometry_values[0], rtol=0.0, atol=1e-12):
        raise ValueError("Deterministic geometry baseline differs across runs")
    geometry = geometry_rows[0]

    row_end = r" \\"
    rows = [
        f"Fixed geometry prior & {geometry['macro_f1']:.3f} & {geometry['hr_at_5']:.3f} & -- & --" + row_end,
        f"Destination specialist & {metric('destination_specialist', *destination)} & {metric('destination_specialist', *hr5)} & -- & --" + row_end,
        f"State specialist & -- & -- & {metric('state_specialist', *natural)} & {metric('state_specialist', *gps)}" + row_end,
        f"FedAvg (Joint) & {metric('joint_no_tierdrop', *destination)} & {metric('joint_no_tierdrop', *hr5)} & {metric('joint_no_tierdrop', *natural)} & {metric('joint_no_tierdrop', *gps)}" + row_end,
        f"MAS \\citep{{zhuang2023mas}} & {metric('mas', *destination)} & {metric('mas', *hr5)} & {metric('mas', *natural)} & {metric('mas', *gps)}" + row_end,
        f"FedHCA$^2$ \\citep{{lu2024fedhca}} & {metric('fedhca2', *destination)} & {metric('fedhca2', *hr5)} & {metric('fedhca2', *natural)} & {metric('fedhca2', *gps)}" + row_end,
        f"\\textbf{{BehaviorFL-MT}} & {metric('behaviorfl_mt', *destination)} & {metric('behaviorfl_mt', *hr5)} & {metric('behaviorfl_mt', *natural)} & {metric('behaviorfl_mt', *gps)}" + row_end,
        f"BehaviorFL-MT (oracle eligibility) & {metric('oracle_task_aware', *destination)} & {metric('oracle_task_aware', *hr5)} & {metric('oracle_task_aware', *natural)} & {metric('oracle_task_aware', *gps)}",
    ]
    OUTPUT.write_text("\n".join(rows) + "\n")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
