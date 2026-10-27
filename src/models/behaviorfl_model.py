"""Shared mask-aware encoder with destination and powertrain-state heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .behavior_encoder import BehaviorEncoderMLP


def _make_head(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class BehaviorFLModel(nn.Module):
    """Two-head model with an optional bounded destination residual.

    POI/music constructor names remain aliases for older callers.
    """

    def __init__(
        self,
        input_dim: int = 22,
        embedding_dim: int = 128,
        hidden_dims: "list | None" = None,
        num_destination_classes: int | None = None,
        num_state_classes: int | None = None,
        head_hidden_dim: int = 64,
        dropout: float = 0.1,
        use_uncertainty_weighting: bool = True,
        num_poi_classes: int | None = None,
        num_music_classes: int | None = None,
        use_contrastive: bool = False,
        use_geometry_prior: bool = True,
        geometry_prior_scale: float = 4.0,
        geometry_residual_bound: float = 0.1,
        fill_mode: str = "learned",
        **_legacy_kwargs,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128]
        self.num_destination_classes = int(
            num_destination_classes or num_poi_classes or 16
        )
        self.num_state_classes = int(num_state_classes or num_music_classes or 2)
        self.num_poi_classes = self.num_destination_classes
        self.num_music_classes = self.num_state_classes
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.use_uncertainty_weighting = use_uncertainty_weighting
        grid_size = int(round(math.sqrt(self.num_destination_classes)))
        self.use_geometry_prior = bool(
            use_geometry_prior
            and input_dim >= 21
            and grid_size * grid_size == self.num_destination_classes
        )
        self.destination_grid_size = grid_size
        self.geometry_prior_scale = float(geometry_prior_scale)
        self.geometry_residual_bound = float(geometry_residual_bound)
        # Compatibility attributes for older clients.
        self.use_contrastive = False
        self.aligner = None

        self.behavior_encoder = BehaviorEncoderMLP(
            input_dim=input_dim, embedding_dim=embedding_dim,
            hidden_dims=hidden_dims, dropout=dropout, fill_mode=fill_mode,
        )
        self.destination_head = _make_head(
            embedding_dim, head_hidden_dim, self.num_destination_classes, dropout
        )
        # With observed-prefix geometry available, start from the deterministic
        # kinematic predictor and learn only a federated residual.  This avoids
        # spending early partial-participation rounds relearning grid geometry.
        if self.use_geometry_prior:
            nn.init.zeros_(self.destination_head[-1].weight)
            nn.init.zeros_(self.destination_head[-1].bias)
        self.state_head = _make_head(
            embedding_dim, head_hidden_dim, self.num_state_classes, dropout
        )

    # Read-only aliases support archived journal utilities without registering
    # duplicate modules or duplicate state-dict keys.
    @property
    def poi_head(self):
        return self.destination_head

    @property
    def music_head(self):
        return self.state_head

    def forward(self, trip_features: torch.Tensor, tier_mask: "torch.Tensor | None" = None):
        if tier_mask is None:
            tier_mask = torch.ones_like(trip_features)
        embedding = self.behavior_encoder(trip_features, tier_mask)
        return self._destination_logits(embedding, trip_features), self.state_head(embedding)

    def _geometry_prior_logits(self, trip_features: torch.Tensor) -> torch.Tensor:
        """Distance-to-cell prior using only the observed first-quarter endpoint."""
        prefix_end = torch.stack((
            trip_features[:, 15] + trip_features[:, 19],
            trip_features[:, 16] + trip_features[:, 20],
        ), dim=1).clamp(0.0, 1.0)
        axis = (
            torch.arange(
                self.destination_grid_size,
                device=trip_features.device,
                dtype=trip_features.dtype,
            ) + 0.5
        ) / float(self.destination_grid_size)
        row, col = torch.meshgrid(axis, axis, indexing="ij")
        centers = torch.stack((row.reshape(-1), col.reshape(-1)), dim=1)
        squared_distance = (
            prefix_end[:, None, :] - centers[None, :, :]
        ).square().sum(dim=2)
        return -self.geometry_prior_scale * squared_distance

    def _destination_logits(
        self, embedding: torch.Tensor, trip_features: torch.Tensor
    ) -> torch.Tensor:
        residual = self.destination_head(embedding)
        if not self.use_geometry_prior:
            return residual
        bounded_residual = self.geometry_residual_bound * torch.tanh(residual)
        return self._geometry_prior_logits(trip_features) + bounded_residual

    def get_behavior_embedding(
        self, trip_features: torch.Tensor, tier_mask: "torch.Tensor | None" = None
    ) -> torch.Tensor:
        if tier_mask is None:
            tier_mask = torch.ones_like(trip_features)
        return self.behavior_encoder(trip_features, tier_mask)

    def get_behavior_centroid(
        self, trip_features: torch.Tensor, tier_mask: "torch.Tensor | None" = None
    ) -> torch.Tensor:
        if tier_mask is None:
            tier_mask = torch.ones_like(trip_features)
        return self.behavior_encoder.get_behavior_centroid(trip_features, tier_mask)

    @staticmethod
    def _masked_ce(logits, labels, mask=None, class_weights=None):
        if mask is None:
            return F.cross_entropy(logits, labels, weight=class_weights)
        active = mask.to(device=logits.device, dtype=torch.bool)
        if int(active.sum().item()) == 0:
            return None
        return F.cross_entropy(logits[active], labels[active], weight=class_weights)

    def compute_full_loss(
        self,
        trip_features: torch.Tensor,
        tier_mask: torch.Tensor,
        destination_labels: torch.Tensor,
        state_labels: torch.Tensor,
        destination_class_weights: "torch.Tensor | None" = None,
        state_class_weights: "torch.Tensor | None" = None,
        state_supervision_mask: "torch.Tensor | None" = None,
        destination_weight: float = 1.0,
        state_weight: float = 1.0,
        detach_state_gradient: bool = False,
        **legacy_kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """Compute task loss while omitting genuinely unobserved state labels.

        detach_state_gradient: MAS-style (Zhuang et al. 2023) merge-and-split
        baseline. When True, the state head still consumes the shared
        embedding but its gradient is blocked from flowing back into
        behavior_encoder, so past the split round the shared trunk is
        updated by the destination task only.
        """
        if destination_class_weights is None:
            destination_class_weights = legacy_kwargs.get("poi_class_weights")
        if state_class_weights is None:
            state_class_weights = legacy_kwargs.get("music_class_weights")
        state_supervision_mask = (
            state_supervision_mask
            if state_supervision_mask is not None
            else legacy_kwargs.get("music_supervision_mask")
        )
        if legacy_kwargs.get("music_weight") == 0.0:
            state_weight = 0.0

        embedding = self.get_behavior_embedding(trip_features, tier_mask)
        destination_logits = self._destination_logits(embedding, trip_features)
        state_embedding = embedding.detach() if detach_state_gradient else embedding
        state_logits = self.state_head(state_embedding)
        destination_loss = self._masked_ce(
            destination_logits, destination_labels, None,
            destination_class_weights.to(destination_logits.device)
            if destination_class_weights is not None else None,
        )
        state_loss = None
        if state_weight > 0:
            state_loss = self._masked_ce(
                state_logits, state_labels, state_supervision_mask,
                state_class_weights.to(state_logits.device)
                if state_class_weights is not None else None,
            )

        active = []
        if destination_weight > 0:
            active.append(destination_weight * destination_loss)
        if state_loss is not None:
            active.append(state_weight * state_loss)
        if not active:
            raise ValueError("Batch has no active supervised task")
        if self.use_uncertainty_weighting and len(active) > 1:
            try:
                from utils.uncertainty import compute_nilw_weights
            except ImportError:
                from src.utils.uncertainty import compute_nilw_weights
            weights = compute_nilw_weights(active)
            total = sum(w * loss for w, loss in zip(weights, active))
        else:
            total = sum(active)

        state_value = float(state_loss.detach().item()) if state_loss is not None else 0.0
        return total, {
            "destination_loss": float(destination_loss.detach().item()),
            "state_loss": state_value,
            "total_loss": float(total.detach().item()),
            # Legacy component names for archived log readers.
            "poi_loss": float(destination_loss.detach().item()),
            "music_loss": state_value,
            "contrastive_loss": 0.0,
        }

    def get_shared_parameters(self) -> list:
        return list(self.behavior_encoder.parameters())

    def get_task_specific_parameters(self) -> list:
        return list(self.destination_head.parameters()) + list(self.state_head.parameters())
