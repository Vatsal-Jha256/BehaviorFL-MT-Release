"""PyTorch data access for the multi-source BehaviorFL tasks."""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


class BehaviorDataset(Dataset):
    """Early-trip inputs with destination and partially observed state labels."""

    def __init__(
        self,
        trip_features: torch.Tensor,
        tier_mask: torch.Tensor,
        destination_labels: torch.Tensor,
        state_labels: torch.Tensor,
        state_supervision_mask: torch.Tensor,
        source_dataset: Optional[list] = None,
        indices: Optional[list] = None,
    ):
        idx = (torch.tensor(indices, dtype=torch.long)
               if indices is not None else torch.arange(len(trip_features)))
        self.features = trip_features[idx]
        self.tier_mask = tier_mask[idx]
        self.destination_labels = destination_labels[idx]
        self.state_labels = state_labels[idx]
        self.state_supervision_mask = state_supervision_mask[idx]
        self.source_dataset = ([source_dataset[i] for i in idx.tolist()]
                               if source_dataset is not None else None)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "features": self.features[idx],
            "tier_mask": self.tier_mask[idx],
            "destination_label": self.destination_labels[idx],
            "state_label": self.state_labels[idx],
            "state_supervision_mask": self.state_supervision_mask[idx],
            # Legacy aliases are deliberately confined to the loader boundary.
            "poi_label": self.destination_labels[idx],
            "music_label": self.state_labels[idx],
        }
        if self.source_dataset is not None:
            item["source_dataset"] = self.source_dataset[idx]
        return item


class BehaviorDataLoader:
    """Load the assembled artifact and provide deterministic client partitions.

    ``split_seed`` fixes the evaluation examples for paired comparisons.
    ``random_seed`` changes only round/epoch shuffling across training seeds.
    """

    def __init__(
        self,
        dataset_path: str,
        train_ratio: float = 0.8,
        batch_size: int = 32,
        num_workers: int = 0,
        random_seed: int = 42,
        split_seed: int = 2026,
        use_all_clients: bool = False,
        split_strategy: str = "random",
    ):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}. Run "
                "scripts/behavior/assemble_behaviorfl_dataset.py first."
            )
        data = torch.load(dataset_path, map_location="cpu", weights_only=False)
        self.dataset_path = os.path.abspath(dataset_path)
        self.trip_features = data["trip_features"].float()
        self.tier_mask = data["tier_mask"].float()
        self.destination_labels = data.get("destination_labels", data["poi_labels"]).long()
        self.state_labels = data.get("state_labels", data["music_labels"]).long()
        self.state_supervision_mask = data.get(
            "state_supervision_mask",
            torch.tensor([s == "ved" for s in data.get("source_dataset", [])]),
        ).bool()
        if len(self.state_supervision_mask) != len(self.trip_features):
            self.state_supervision_mask = torch.ones(len(self.trip_features), dtype=torch.bool)
        if torch.isnan(self.trip_features).any() or torch.isinf(self.trip_features).any():
            raise ValueError(f"Non-finite input in {dataset_path}")

        self.source_dataset = data.get("source_dataset")
        self.client_splits = (data["all_client_splits"]
                              if use_all_clients and "all_client_splits" in data
                              else data["client_splits"])
        self.fl_client_ids = data.get("fl_client_ids", list(self.client_splits))
        self.held_out_client_ids = data.get("held_out_client_ids", [])
        self.num_destination_classes = int(data.get(
            "num_destination_classes", data.get("num_poi_categories", 10)
        ))
        self.num_state_classes = int(data.get(
            "num_state_classes", data.get("num_music_genres", 10)
        ))
        self.input_dim = int(self.trip_features.shape[1])
        self.feature_names = data.get("feature_names", [])
        self.metadata = data.get("metadata", {})
        self.train_ratio, self.batch_size = train_ratio, batch_size
        self.num_workers, self.random_seed, self.split_seed = num_workers, random_seed, split_seed
        if split_strategy not in {"random", "leave_last"}:
            raise ValueError(f"Unsupported split_strategy={split_strategy!r}")
        self.split_strategy = split_strategy

        self._client_train_indices, self._client_val_indices = {}, {}
        self._client_ordered_indices = {}
        for client_id, indices in self.client_splits.items():
            n_train = max(1, int(len(indices) * train_ratio))
            if split_strategy == "leave_last":
                train_idx, val_idx = list(indices[:n_train]), list(indices[n_train:])
            else:
                rng = torch.Generator().manual_seed(split_seed * 100_003 + int(client_id))
                perm = torch.randperm(len(indices), generator=rng).tolist()
                train_idx = [indices[i] for i in perm[:n_train]]
                val_idx = [indices[i] for i in perm[n_train:]]
            if not val_idx:
                val_idx = train_idx[-1:]
            self._client_train_indices[client_id] = train_idx
            self._client_val_indices[client_id] = val_idx
            train_set = set(train_idx)
            self._client_ordered_indices[client_id] = [i for i in indices if i in train_set]

        train_indices = sorted({i for xs in self._client_train_indices.values() for i in xs})
        train_idx = torch.tensor(train_indices, dtype=torch.long)
        dest_counts = torch.bincount(
            self.destination_labels[train_idx], minlength=self.num_destination_classes
        ).float()
        supervised = train_idx[self.state_supervision_mask[train_idx]]
        state_counts = torch.bincount(
            self.state_labels[supervised], minlength=self.num_state_classes
        ).float() if len(supervised) else torch.ones(self.num_state_classes)
        self.destination_class_weights = (
            dest_counts.sum() / (self.num_destination_classes * dest_counts.clamp(min=1))
        )
        self.state_class_weights = (
            state_counts.sum() / (self.num_state_classes * state_counts.clamp(min=1))
        )
        # Compatibility aliases for utilities not used by the workshop pipeline.
        self.poi_labels, self.music_labels = self.destination_labels, self.state_labels
        self.num_poi_categories, self.num_music_genres = (
            self.num_destination_classes, self.num_state_classes
        )
        self.poi_class_weights, self.music_class_weights = (
            self.destination_class_weights, self.state_class_weights
        )

    def get_client_loader(
        self,
        client_id: int,
        split: str = "train",
        batch_size: "int | None" = None,
        server_round: int = 0,
    ) -> DataLoader:
        if client_id not in self.client_splits:
            raise KeyError(f"Client {client_id} is not in this partition")
        indices = (self._client_train_indices[client_id]
                   if split == "train" else self._client_val_indices[client_id])
        dataset = BehaviorDataset(
            self.trip_features, self.tier_mask,
            self.destination_labels, self.state_labels, self.state_supervision_mask,
            self.source_dataset, indices,
        )
        bs = batch_size or self.batch_size
        shuffle = split == "train"
        generator = None
        if shuffle:
            generator = torch.Generator().manual_seed(
                self.random_seed * 100_003 + int(client_id) + 1 + server_round * 7919
            )
        return DataLoader(
            dataset, batch_size=bs, shuffle=shuffle, num_workers=self.num_workers,
            drop_last=(shuffle and len(dataset) >= bs), generator=generator,
        )

    @property
    def num_clients(self) -> int:
        return len(self.client_splits)

    def get_all_indices(self, split: str = "train") -> list:
        src = self._client_train_indices if split == "train" else self._client_val_indices
        return [i for indices in src.values() for i in indices]
