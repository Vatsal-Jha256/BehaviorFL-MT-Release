"""Mask-aware feature filling and MLP encoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BehaviorEncoderMLP(nn.Module):
    """Fill missing features, encode them with an MLP, and L2-normalize.

    The default encoder has hidden widths 64 and 128. Missing features use
    learned fill values, zeros, or observed batch means, depending on fill_mode.
    """

    def __init__(
        self,
        input_dim: int = 15,
        embedding_dim: int = 128,
        hidden_dims: "list | None" = None,
        dropout: float = 0.1,
        fill_mode: str = "learned",
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128]
        if fill_mode not in ("learned", "zero", "batch_mean"):
            raise ValueError(f"Unknown fill_mode: {fill_mode}")

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        # Learned fill is the default; zero and batch means are ablations.
        self.fill_mode = fill_mode

        # Predict fill values from the missing-feature pattern.
        self.tier_mask_proj = nn.Linear(input_dim, input_dim, bias=True)

        # MLP layers
        dims = [input_dim] + hidden_dims + [embedding_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)


    def forward(self, trip_features: torch.Tensor, tier_mask: torch.Tensor) -> torch.Tensor:
        """Encode behavior features into embedding space.

        Args:
            trip_features: [B, input_dim] normalized behavior/context features
            tier_mask: [B, input_dim] availability/confidence mask

        Returns:
            embedding: [B, embedding_dim] L2-normalized behavior embedding
        """
        # Imputation for missing features (see fill_mode above)
        # missing_mask: 1 where feature is unavailable
        missing_mask = 1.0 - tier_mask.clamp(0.0, 1.0)
        if self.fill_mode == "zero":
            imputed_fill = torch.zeros_like(trip_features)
        elif self.fill_mode == "batch_mean":
            observed = tier_mask.clamp(0.0, 1.0)
            observed_count = observed.sum(dim=0).clamp(min=1.0)
            observed_sum = (trip_features * observed).sum(dim=0)
            batch_mean = (observed_sum / observed_count).unsqueeze(0)  # [1, D]
            imputed_fill = batch_mean.expand_as(trip_features)
        else:
            # imputed_fill: learned "default" values for missing features
            imputed_fill = self.tier_mask_proj(missing_mask)  # [B, input_dim]
            imputed_fill = torch.tanh(imputed_fill)  # keep in reasonable range

        # Combine available features with learned fill
        x_masked = trip_features * tier_mask          # zero out missing
        x_filled = imputed_fill * missing_mask         # fill only where missing
        x_in = x_masked + x_filled                    # [B, input_dim]

        # Encode
        embedding = self.encoder(x_in)                # [B, embedding_dim]

        # Normalize the embedding
        embedding = F.normalize(embedding, p=2, dim=-1)
        return embedding

    def get_behavior_centroid(
        self, trip_features: torch.Tensor, tier_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute mean behavior embedding for a set of trips (cluster centroid).

        Args:
            trip_features: [N, input_dim] local trip features
            tier_mask: [N, input_dim] availability masks

        Returns:
            centroid: [embedding_dim] mean embedding (L2-normalized)
        """
        with torch.no_grad():
            embeddings = self.forward(trip_features, tier_mask)  # [N, D]
            centroid = embeddings.mean(dim=0)                     # [D]
            centroid = F.normalize(centroid, p=2, dim=-1)
        return centroid
