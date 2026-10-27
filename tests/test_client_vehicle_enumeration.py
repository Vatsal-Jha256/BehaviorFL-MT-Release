"""Check that clients contain one vehicle and held-out splits are disjoint."""

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.behavior.assemble_behaviorfl_dataset import (
    MIN_SAMPLES_PER_CLIENT,
    MIN_SAMPLES_PER_HELD_OUT_ENTRY,
    make_client_splits,
)

DATASET_PATH = Path(__file__).resolve().parents[1] / "data/processed/behaviorfl_dataset.pt"


def test_make_client_splits_never_bundles_vehicles():
    # 3 primary vehicles (>=5 trips), 2 auxiliary (2-4 trips), 2 single-trip.
    sources, vehicles = [], []
    plan = [("ved", 1, 6), ("ved", 2, 5), ("geolife", 1, 7),
            ("ved", 3, 2), ("geolife", 2, 4),
            ("tdrive", 1, 1), ("tdrive", 2, 1)]
    for src, vid, n in plan:
        sources.extend([src] * n)
        vehicles.extend([vid] * n)
    vehicle_ids = np.array(vehicles, dtype=np.int64)

    splits, keys, primary_ids, aux_ids, dropped = make_client_splits(
        vehicle_ids, sources, len(sources)
    )

    assert primary_ids == [0, 1, 2]
    assert aux_ids == [3, 4]
    assert len(dropped) == 2

    for cid, indices in splits.items():
        owners = {f"{sources[i]}_{int(vehicle_ids[i])}" for i in indices}
        assert owners == {keys[cid]}, f"client {cid} bundles {owners}"
        assert len(indices) >= MIN_SAMPLES_PER_HELD_OUT_ENTRY

    for cid in primary_ids:
        assert len(splits[cid]) >= MIN_SAMPLES_PER_CLIENT
    for cid in aux_ids:
        assert len(splits[cid]) < MIN_SAMPLES_PER_CLIENT

    # No sample may be assigned twice or silently lost.
    assigned = [i for indices in splits.values() for i in indices]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) | set(dropped) == set(range(len(sources)))


def test_single_trip_vehicles_are_dropped_not_pooled():
    sources = ["ved"] * 5 + ["tdrive"] * 3
    vehicle_ids = np.array([1, 1, 1, 1, 1, 10, 11, 12], dtype=np.int64)

    splits, _, primary_ids, aux_ids, dropped = make_client_splits(
        vehicle_ids, sources, len(sources)
    )

    assert primary_ids == [0]
    assert aux_ids == []
    assert dropped == [5, 6, 7]
    assert len(splits) == 1


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="processed dataset not built")
def test_processed_artifact_has_one_vehicle_per_client():
    data = torch.load(DATASET_PATH, map_location="cpu", weights_only=False)
    # .tolist() rather than .numpy(): this assertion must still run on installs
    # where the torch/numpy ABI pairing is broken.
    vehicle_ids = data["vehicle_ids"].tolist()
    sources = data["source_dataset"]
    splits = data["all_client_splits"]
    keys = data["client_vehicle_key"]

    for cid, indices in splits.items():
        owners = {f"{sources[i]}_{int(vehicle_ids[i])}" for i in indices}
        assert len(owners) == 1, f"client {cid} bundles {len(owners)} vehicles"
        assert owners.pop() == keys[cid]
        assert len(indices) >= MIN_SAMPLES_PER_HELD_OUT_ENTRY

    fl_ids = set(data["fl_client_ids"])
    held_out_ids = set(data["held_out_client_ids"])
    assert fl_ids.isdisjoint(held_out_ids)
    assert fl_ids | held_out_ids == set(splits)
    assert data["client_splits"].keys() == fl_ids


@pytest.mark.skipif(not DATASET_PATH.exists(), reason="processed dataset not built")
def test_held_out_entries_support_disjoint_prefix_and_eval():
    from src.data.behavior_loader import BehaviorDataLoader

    loader = BehaviorDataLoader(str(DATASET_PATH), batch_size=32, use_all_clients=True)
    for cid in loader.held_out_client_ids:
        train_idx = loader._client_train_indices[cid]
        val_idx = loader._client_val_indices[cid]
        assert train_idx, f"held-out entry {cid} has no routing prefix"
        assert val_idx, f"held-out entry {cid} has no evaluation trip"
        assert not set(train_idx) & set(val_idx), (
            f"held-out entry {cid} evaluates on a trip that is also in its prefix pool"
        )
