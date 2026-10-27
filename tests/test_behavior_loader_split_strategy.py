from pathlib import Path

import torch


def _write_tiny_behavior_dataset(path: Path) -> None:
    """Create a minimal processed tensor with two clients and ordered trips."""
    n = 10
    data = {
        "trip_features": torch.arange(n * 15, dtype=torch.float32).reshape(n, 15),
        "tier_mask": torch.ones(n, 15, dtype=torch.float32),
        "poi_labels": torch.arange(n, dtype=torch.long) % 10,
        "music_labels": torch.arange(n, dtype=torch.long) % 10,
        "gps_coords": torch.zeros(n, 2, dtype=torch.float32),
        "source_dataset": ["synthetic"] * n,
        "client_splits": {
            0: [0, 1, 2, 3, 4],
            1: [5, 6, 7, 8, 9],
        },
        "fl_client_ids": [0, 1],
        "held_out_client_ids": [],
        "num_poi_categories": 10,
        "num_music_genres": 10,
        "feature_names": [f"f{i}" for i in range(15)],
        "metadata": {"synthetic": True},
    }
    torch.save(data, path)


def test_behavior_loader_leave_last_uses_suffix(tmp_path):
    from src.data.behavior_loader import BehaviorDataLoader

    dataset_path = tmp_path / "behaviorfl_dataset.pt"
    _write_tiny_behavior_dataset(dataset_path)

    loader = BehaviorDataLoader(
        str(dataset_path),
        train_ratio=0.6,
        random_seed=123,
        split_strategy="leave_last",
    )

    assert loader._client_train_indices[0] == [0, 1, 2]
    assert loader._client_val_indices[0] == [3, 4]
    assert loader._client_ordered_indices[0] == [0, 1, 2]
    assert loader._client_train_indices[1] == [5, 6, 7]
    assert loader._client_val_indices[1] == [8, 9]


def test_behavior_loader_random_strategy_is_deterministic(tmp_path):
    from src.data.behavior_loader import BehaviorDataLoader

    dataset_path = tmp_path / "behaviorfl_dataset.pt"
    _write_tiny_behavior_dataset(dataset_path)

    first = BehaviorDataLoader(str(dataset_path), train_ratio=0.6, random_seed=7)
    second = BehaviorDataLoader(str(dataset_path), train_ratio=0.6, random_seed=7)

    assert first.split_strategy == "random"
    assert first._client_train_indices == second._client_train_indices
    assert first._client_val_indices == second._client_val_indices
    assert set(first._client_train_indices[0]).isdisjoint(first._client_val_indices[0])


def test_behavior_loader_rejects_unknown_split_strategy(tmp_path):
    from src.data.behavior_loader import BehaviorDataLoader

    dataset_path = tmp_path / "behaviorfl_dataset.pt"
    _write_tiny_behavior_dataset(dataset_path)

    try:
        BehaviorDataLoader(str(dataset_path), split_strategy="chronological")
    except ValueError as exc:
        assert "split_strategy" in str(exc)
    else:
        raise AssertionError("unknown split_strategy should raise ValueError")
