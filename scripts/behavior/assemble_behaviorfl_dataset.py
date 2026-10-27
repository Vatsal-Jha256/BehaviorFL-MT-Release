"""Build destination and VED powertrain-state data from three sources.

Inputs use the first 25% of each trip. Targets use the final GPS coordinate
and full-trip OBD summaries. Each client represents one vehicle or user.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.paths import RAW_DATA_DIR, check_raw_data
from src.data.behavior_extractors.geolife_extractor import GeoLifeExtractor
from src.data.behavior_extractors.tdrive_extractor import TDriveExtractor
from src.data.behavior_extractors.ved_extractor import ALL_FEATURES, VEDExtractor


MIN_SAMPLES_PER_CLIENT = 5
MIN_SAMPLES_PER_HELD_OUT_ENTRY = 2
HELD_OUT_FRACTION = 0.15
TDRIVE_SUBSAMPLE = 10357
RANDOM_SEED = 42
PREFIX_FRACTION = 0.25

# Fixed study windows.  A trip is eligible only when its start, observed-prefix
# endpoint, and final endpoint all lie inside the corresponding window.  This
# avoids silently collapsing out-of-region trips into edge destination cells.
REGION_BOUNDS = {
    "ved": (42.21, 42.33, -83.80, -83.64),
    "geolife": (39.75, 40.10, 116.20, 116.55),
    "tdrive": (39.75, 40.10, 116.20, 116.55),
}
GRID_SIZE = 4
DESTINATION_CLASSES = GRID_SIZE * GRID_SIZE
STATE_CLASSES = 2
INPUT_FEATURE_NAMES = ALL_FEATURES + [
    "start_lat_region", "start_lon_region", "start_time_sin", "start_time_cos",
    "prefix_delta_lat_region", "prefix_delta_lon_region",
    "prefix_displacement_region",
]


def load_or_extract(path: str, extractor_fn, desc: str, rebuild: bool = False) -> dict:
    """Load a schema-versioned cache or run its source extractor."""
    cache = ROOT / path
    if cache.exists() and not rebuild:
        print(f"[Assemble] Loading {desc}: {cache}")
        cached = torch.load(cache, map_location="cpu", weights_only=False)
        if "prefix_end_coords" in cached:
            return cached
        print(f"[Assemble] Refreshing stale {desc} cache (no prefix endpoint)")
    print(f"[Assemble] Extracting {desc}...")
    data = extractor_fn()
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache)
    print(f"[Assemble] Saved {desc}: {cache}")
    return data


def make_client_splits(
    vehicle_ids: np.ndarray,
    source_datasets: list,
    n_samples: int,
) -> tuple:
    """Create one client entry per ``(source, vehicle_id)`` vehicle."""
    groups: dict[str, list[int]] = {}
    for i in range(n_samples):
        key = f"{source_datasets[i]}_{int(vehicle_ids[i])}"
        groups.setdefault(key, []).append(i)

    primary, auxiliary, dropped = [], [], []
    for key, indices in groups.items():
        if len(indices) >= MIN_SAMPLES_PER_CLIENT:
            primary.append((key, indices))
        elif len(indices) >= MIN_SAMPLES_PER_HELD_OUT_ENTRY:
            auxiliary.append((key, indices))
        else:
            dropped.append((key, indices))

    client_splits, client_vehicle_key = {}, {}
    for client_id, (key, indices) in enumerate(primary + auxiliary):
        client_splits[client_id] = indices
        client_vehicle_key[client_id] = key
    primary_ids = list(range(len(primary)))
    auxiliary_ids = list(range(len(primary), len(primary) + len(auxiliary)))
    dropped_indices = sorted(i for _, indices in dropped for i in indices)
    return client_splits, client_vehicle_key, primary_ids, auxiliary_ids, dropped_indices


def _source_stratified_holdout(
    primary_ids: list[int], client_vehicle_key: dict[int, str]
) -> tuple[list[int], list[int]]:
    """Choose a fixed vehicle-disjoint holdout within each source."""
    rng = np.random.default_rng(RANDOM_SEED)
    train, held = [], []
    sources = sorted({client_vehicle_key[cid].split("_", 1)[0] for cid in primary_ids})
    for source in sources:
        ids = [cid for cid in primary_ids if client_vehicle_key[cid].startswith(source + "_")]
        perm = np.asarray(ids, dtype=np.int64)
        rng.shuffle(perm)
        n_hold = min(max(1, int(round(HELD_OUT_FRACTION * len(ids)))), max(1, len(ids) - 1))
        held.extend(int(x) for x in perm[:n_hold])
        train.extend(int(x) for x in perm[n_hold:])
    return sorted(train), sorted(held)


def _region_coordinates(coords: np.ndarray, sources: list[str]) -> np.ndarray:
    """Map latitude/longitude to fixed region-relative coordinates in [0, 1]."""
    out = np.zeros((len(coords), 2), dtype=np.float32)
    for i, (coord, source) in enumerate(zip(coords, sources)):
        lat0, lat1, lon0, lon1 = REGION_BOUNDS[source]
        out[i, 0] = np.clip((coord[0] - lat0) / (lat1 - lat0), 0.0, 1.0)
        out[i, 1] = np.clip((coord[1] - lon0) / (lon1 - lon0), 0.0, 1.0)
    return out


def _inside_study_region(
    start_coords: np.ndarray,
    prefix_end_coords: np.ndarray,
    end_coords: np.ndarray,
    sources: list[str],
) -> np.ndarray:
    """Return rows whose complete early/target geography is in scope."""
    keep = np.ones(len(sources), dtype=np.bool_)
    for source in sorted(set(sources)):
        rows = np.asarray([name == source for name in sources], dtype=np.bool_)
        lat0, lat1, lon0, lon1 = REGION_BOUNDS[source]
        source_keep = np.ones(int(rows.sum()), dtype=np.bool_)
        for coords in (start_coords[rows], prefix_end_coords[rows], end_coords[rows]):
            source_keep &= (
                (coords[:, 0] >= lat0)
                & (coords[:, 0] <= lat1)
                & (coords[:, 1] >= lon0)
                & (coords[:, 1] <= lon1)
            )
        keep[rows] = source_keep
    return keep


def _destination_labels(end_coords: np.ndarray, sources: list[str]) -> np.ndarray:
    rel = _region_coordinates(end_coords, sources)
    row = np.minimum((rel[:, 0] * GRID_SIZE).astype(np.int64), GRID_SIZE - 1)
    col = np.minimum((rel[:, 1] * GRID_SIZE).astype(np.int64), GRID_SIZE - 1)
    return row * GRID_SIZE + col


def _state_labels_from_training_vehicles(
    components: np.ndarray,
    component_mask: np.ndarray,
    supervised_train_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit the powertrain-state definition using training vehicles only.

    The score combines robustly standardized mean RPM, engine load, and MAF
    variability with weights 0.4/0.3/0.3.  The high-state cutoff is the 75th
    percentile among eligible training rows and is then frozen for held-out
    vehicles.
    """
    supervision = component_mask.min(axis=1) > 0
    train = np.asarray([i for i in supervised_train_indices if supervision[i]], dtype=np.int64)
    if len(train) == 0:
        raise RuntimeError("No OBD-supervised VED rows in the training-client partition")
    center = np.median(components[train], axis=0)
    q25 = np.quantile(components[train], 0.25, axis=0)
    q75 = np.quantile(components[train], 0.75, axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    z = (components - center) / scale
    score = z @ np.asarray([0.4, 0.3, 0.3], dtype=np.float32)
    threshold = float(np.quantile(score[train], 0.75))
    labels = (score >= threshold).astype(np.int64)
    labels[~supervision] = 0
    return labels, supervision.astype(np.bool_), {
        "components": ["mean_rpm", "mean_engine_load", "maf_variability_proxy"],
        "weights": [0.4, 0.3, 0.3],
        "center_train_median": center.tolist(),
        "scale_train_iqr": scale.tolist(),
        "threshold_train_q75": threshold,
        "fit_rows": int(len(train)),
    }


def assemble(
    include_geolife: bool = True,
    include_tdrive: bool = True,
    tdrive_subsample: int = TDRIVE_SUBSAMPLE,
    rebuild_extractors: bool = False,
    raw_data_dir: str | Path = RAW_DATA_DIR,
) -> dict:
    """Build ``data/processed/behaviorfl_dataset.pt`` from the three sources."""
    raw_root = Path(raw_data_dir).expanduser().resolve()
    caches = {"ved": ROOT / "data/processed/ved_behaviors_mtl_v2.pt"}
    if include_geolife:
        caches["geolife"] = ROOT / "data/processed/geolife_behaviors_mtl_v2.pt"
    if include_tdrive:
        caches["tdrive"] = ROOT / f"data/processed/tdrive_behaviors_mtl_v2_{tdrive_subsample}.pt"
    needed = tuple(source for source, cache in caches.items()
                   if rebuild_extractors or not cache.exists())
    if needed:
        check_raw_data(raw_root, needed)
    sources_data = [load_or_extract(
        "data/processed/ved_behaviors_mtl_v2.pt",
        lambda: VEDExtractor(str(raw_root / "ved")).extract_all(PREFIX_FRACTION),
        "VED early-trip behaviors", rebuild_extractors,
    )]
    if include_geolife:
        sources_data.append(load_or_extract(
            "data/processed/geolife_behaviors_mtl_v2.pt",
            lambda: GeoLifeExtractor(str(raw_root / "geolife")).extract_all(PREFIX_FRACTION),
            "GeoLife labeled car/taxi segments", rebuild_extractors,
        ))
    if include_tdrive:
        sources_data.append(load_or_extract(
            f"data/processed/tdrive_behaviors_mtl_v2_{tdrive_subsample}.pt",
            lambda: TDriveExtractor(str(raw_root / "tdrive")).extract_all(
                max_drivers=tdrive_subsample, prefix_fraction=PREFIX_FRACTION
            ),
            f"T-Drive segmented trajectories ({tdrive_subsample} drivers)", rebuild_extractors,
        ))

    def cat_tensor(key: str) -> torch.Tensor:
        return torch.cat([d[key] for d in sources_data], dim=0)

    behavior = cat_tensor("trip_features")
    behavior_mask = cat_tensor("tier_mask")
    vehicle_ids = cat_tensor("vehicle_ids")
    trip_ids = cat_tensor("trip_ids")
    start_times = cat_tensor("start_times")
    start_coords = cat_tensor("start_coords")
    prefix_end_coords = cat_tensor("prefix_end_coords")
    end_coords = cat_tensor("end_coords")
    state_components = cat_tensor("state_components_raw")
    state_component_mask = cat_tensor("state_components_mask")
    trip_keys = [key for d in sources_data for key in d["trip_keys"]]
    sources = [source for d in sources_data for source in d["source_dataset"]]

    pre_filter_source_counts = {
        name: sources.count(name) for name in sorted(set(sources))
    }
    # Require finite coordinates and complete containment in the fixed study
    # window.  In particular, do not convert out-of-window targets to edge-cell
    # labels through clipping.
    geo_valid = (
        torch.isfinite(start_coords).all(dim=1)
        & torch.isfinite(prefix_end_coords).all(dim=1)
        & torch.isfinite(end_coords).all(dim=1)
    )
    region_valid = torch.tensor(
        _inside_study_region(
            np.asarray(start_coords.tolist()),
            np.asarray(prefix_end_coords.tolist()),
            np.asarray(end_coords.tolist()),
            sources,
        ),
        dtype=torch.bool,
    )
    keep = torch.nonzero(geo_valid & region_valid, as_tuple=False).squeeze(1)
    behavior, behavior_mask = behavior[keep], behavior_mask[keep]
    vehicle_ids, trip_ids, start_times = vehicle_ids[keep], trip_ids[keep], start_times[keep]
    start_coords = start_coords[keep]
    prefix_end_coords = prefix_end_coords[keep]
    end_coords = end_coords[keep]
    state_components, state_component_mask = state_components[keep], state_component_mask[keep]
    keep_list = keep.tolist()
    trip_keys = [trip_keys[i] for i in keep_list]
    sources = [sources[i] for i in keep_list]
    n = len(behavior)

    start_rel = _region_coordinates(np.asarray(start_coords.tolist()), sources)
    prefix_end_rel = _region_coordinates(
        np.asarray(prefix_end_coords.tolist()), sources
    )
    prefix_delta = np.clip(prefix_end_rel - start_rel, -1.0, 1.0)
    prefix_displacement = np.clip(
        np.linalg.norm(prefix_delta, axis=1, keepdims=True) / np.sqrt(2.0),
        0.0,
        1.0,
    )
    seconds = np.asarray(start_times.tolist(), dtype=np.float64) % 86400.0
    phase = 2.0 * np.pi * seconds / 86400.0
    context = np.column_stack((
        start_rel,
        np.sin(phase),
        np.cos(phase),
        prefix_delta,
        prefix_displacement,
    )).astype(np.float32)
    trip_features = torch.cat((behavior, torch.tensor(context)), dim=1)
    tier_mask = torch.cat((
        behavior_mask, torch.ones((n, context.shape[1]), dtype=torch.float32)
    ), dim=1)
    destination_labels_np = _destination_labels(np.asarray(end_coords.tolist()), sources)

    client_splits, client_vehicle_key, primary_ids, auxiliary_ids, dropped_indices = (
        make_client_splits(np.asarray(vehicle_ids.tolist()), sources, n)
    )
    fl_client_ids, held_primary = _source_stratified_holdout(primary_ids, client_vehicle_key)
    held_out_client_ids = held_primary + auxiliary_ids
    fl_client_splits = {cid: client_splits[cid] for cid in fl_client_ids}
    supervised_train_indices = [i for cid in fl_client_ids for i in client_splits[cid]]
    state_labels_np, state_supervision_np, state_definition = _state_labels_from_training_vehicles(
        np.asarray(state_components.tolist(), dtype=np.float32),
        np.asarray(state_component_mask.tolist(), dtype=np.float32),
        supervised_train_indices,
    )

    destination_labels = torch.tensor(destination_labels_np, dtype=torch.long)
    state_labels = torch.tensor(state_labels_np, dtype=torch.long)
    state_supervision_mask = torch.tensor(state_supervision_np, dtype=torch.bool)
    source_counts = {name: sources.count(name) for name in sorted(set(sources))}

    for cid, indices in client_splits.items():
        owners = {f"{sources[i]}_{int(vehicle_ids[i])}" for i in indices}
        assert owners == {client_vehicle_key[cid]}

    dataset = {
        "trip_features": trip_features,
        "tier_mask": tier_mask,
        "destination_labels": destination_labels,
        "state_labels": state_labels,
        "state_supervision_mask": state_supervision_mask,
        "state_components_raw": state_components,
        "vehicle_ids": vehicle_ids,
        "trip_ids": trip_ids,
        "trip_keys": trip_keys,
        "start_times": start_times,
        "start_coords": start_coords,
        "end_coords": end_coords,
        "source_dataset": sources,
        "client_splits": fl_client_splits,
        "all_client_splits": client_splits,
        "fl_client_ids": fl_client_ids,
        "held_out_client_ids": held_out_client_ids,
        "client_vehicle_key": client_vehicle_key,
        "primary_client_ids": primary_ids,
        "auxiliary_client_ids": auxiliary_ids,
        "unassigned_trip_indices": dropped_indices,
        "num_destination_classes": DESTINATION_CLASSES,
        "num_state_classes": STATE_CLASSES,
        "feature_names": INPUT_FEATURE_NAMES,
        # Compatibility aliases keep old diagnostic utilities loadable while
        # the active model/client API uses destination/state terminology.
        "poi_labels": destination_labels,
        "music_labels": state_labels,
        "num_poi_categories": DESTINATION_CLASSES,
        "num_music_genres": STATE_CLASSES,
        "metadata": {
            "schema": "behaviorfl_multitask_v4",
            "n_trips": n,
            "n_input_features": int(trip_features.shape[1]),
            "n_fl_clients": len(fl_client_ids),
            "n_held_out_clients": len(held_out_client_ids),
            "sources": source_counts,
            "prefix_fraction": PREFIX_FRACTION,
            "destination_task": {
                "type": "region_relative_endpoint_grid",
                "grid_size": GRID_SIZE,
                "num_classes": DESTINATION_CLASSES,
                "bounds": REGION_BOUNDS,
                "study_region_filter": "start, prefix endpoint, and final endpoint all in bounds",
                "pre_filter_source_counts": pre_filter_source_counts,
                "excluded_source_counts": {
                    name: pre_filter_source_counts[name] - source_counts.get(name, 0)
                    for name in pre_filter_source_counts
                },
                "distribution": np.bincount(
                    destination_labels_np, minlength=DESTINATION_CLASSES
                ).tolist(),
            },
            "state_task": {
                **state_definition,
                "num_classes": STATE_CLASSES,
                "eligible_rows": int(state_supervision_np.sum()),
                "positive_rows": int(state_labels_np[state_supervision_np].sum()),
                "supervision": "VED rows with RPM, engine load, and MAF variability",
            },
            "split": {
                "vehicle_disjoint": True,
                "held_out_fraction_per_source": HELD_OUT_FRACTION,
                "seed": RANDOM_SEED,
            },
            "preprocessing": "fixed physical scaling; no held-out feature statistics",
        },
    }

    out = ROOT / "data/processed/behaviorfl_dataset.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, out)
    print(f"[Assemble] Wrote {out}")
    print(f"  rows={n:,}, inputs={trip_features.shape[1]}, FL clients={len(fl_client_ids):,}")
    print(f"  source rows={source_counts}")
    print(f"  state-supervised rows={int(state_supervision_np.sum()):,}")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-geolife", action="store_true")
    parser.add_argument("--no-tdrive", action="store_true")
    parser.add_argument("--tdrive-subsample", type=int, default=TDRIVE_SUBSAMPLE)
    parser.add_argument("--rebuild-extractors", action="store_true")
    parser.add_argument("--raw-data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--check-raw-data", action="store_true")
    args = parser.parse_args()
    if args.check_raw_data:
        sources = ("ved",) + (() if args.no_geolife else ("geolife",)) + (() if args.no_tdrive else ("tdrive",))
        try:
            counts = check_raw_data(args.raw_data_dir, sources)
        except FileNotFoundError as exc:
            parser.exit(2, str(exc) + "\n")
        for name, count in counts.items():
            print(f"{name}={count}")
        parser.exit()
    assemble(
        include_geolife=not args.no_geolife,
        include_tdrive=not args.no_tdrive,
        tdrive_subsample=args.tdrive_subsample,
        rebuild_extractors=args.rebuild_extractors,
        raw_data_dir=args.raw_data_dir,
    )
