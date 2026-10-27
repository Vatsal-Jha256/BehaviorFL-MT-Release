"""
BehaviorFLClient: Flower NumPyClient for BehaviorFL.

Gradient conflict resolution uses PCGrad (Yu et al., NeurIPS 2020) via
apply_pcgrad_to_gradients() in src/utils/pcgrad.py.
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

import flwr as fl
from flwr.common import ParametersRecord, array_from_numpy

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.behaviorfl_model import BehaviorFLModel
from data.behavior_loader import BehaviorDataLoader
from utils.pcgrad import apply_pcgrad_to_gradients
from utils.metrics import compute_hr_ndcg, compute_classification_metrics

logger = logging.getLogger(__name__)


def _replace_dropout_with_identity(model: torch.nn.Module) -> torch.nn.Module:
    """Replace Dropout with Identity for Opacus vmap/hooks compatibility."""
    for name, child in list(model.named_children()):
        if isinstance(child, torch.nn.Dropout):
            setattr(model, name, torch.nn.Identity())
        else:
            _replace_dropout_with_identity(child)
    return model


class BehaviorFLClient(fl.client.NumPyClient):
    """Flower client for BehaviorFL.

    Args:
        client_id: Integer client ID
        model: BehaviorFLModel
        data_loader: BehaviorDataLoader
        config: Training config dict (local_epochs, batch_size, learning_rate,
                weight_decay, contrastive_weight, use_pcgrad, use_dp,
                dp_noise_multiplier, dp_max_grad_norm, fedprox_mu, local_only)
        device: torch.device (default cpu)
    """

    def __init__(
        self,
        client_id: int,
        model: BehaviorFLModel,
        data_loader: BehaviorDataLoader,
        config: Optional[dict] = None,
        device: Optional[torch.device] = None,
    ):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        self.config = config or {}
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        # Set by client_fn to the Flower Context object for this virtual client.
        # The Ray simulation backend reconstructs a fresh BehaviorFLClient (and
        # fresh model) on every single fit()/evaluate() call, so any state that
        # must survive across FL rounds (task heads kept local, or the entire
        # model for local-only) has to be threaded explicitly through
        # context.state, which Flower's RunInfoStore persists and restores
        # across calls for the same (run_id, node_id).
        self.context = None

        # NILW has no learnable parameters, so it is compatible with Opacus DP-SGD.
        # The previous mutual exclusion guard (use_dp + use_uncertainty_weighting)
        # was required for Kendall sigma parameters; it is not needed for NILW.


    # Flower NumPyClient interface


    def get_parameters(self, config: dict) -> list:  # noqa: ARG002
        """Return federated model parameters.

        Default: encoder + aligner only (task heads remain local).
        If config key 'federate_task_heads' is set in self.config,
        also includes poi_head and music_head parameters.

        For FedRouteFMTL: uses get_shared_state_dict() for BN buffer correctness.
        """
        model = self._unwrap_model(self.model)
        if hasattr(model, "get_shared_state_dict"):
            sd = model.get_shared_state_dict()
            params = [v.detach().cpu().numpy() for v in sd.values()]
        else:
            params = [p.detach().cpu().numpy() for p in model.get_shared_parameters()]
        if bool(self.config.get("federate_task_heads", False)):
            params += [p.detach().cpu().numpy() for p in model.get_task_specific_parameters()]
        return params

    def get_properties(self, config: dict) -> dict:  # noqa: ARG002
        """Cheap metadata query (no training): whether this client holds any
        state-supervised trip. Used by OracleTaskAwareFedAvg to learn the full
        eligible-client set before round 1, without a fit round."""
        client_train_indices = getattr(
            self.data_loader, "_client_train_indices", {}
        ).get(self.client_id, [])
        state_supervised = bool(
            self.data_loader.state_supervision_mask[
                torch.as_tensor(client_train_indices, dtype=torch.long)
            ].any()
        ) if len(client_train_indices) else False
        return {
            "client_id": float(self.client_id),
            "state_supervised_client": float(state_supervised),
        }

    def fit(self, parameters: list, config: dict) -> tuple:  # noqa: ARG002
        local_only = bool(self.config.get("local_only", False))
        if local_only:
            # No server communication at all -- the entire model (shared +
            # task-specific) must be persisted and restored by the client
            # itself, since a fresh model is otherwise reconstructed every call.
            # A missing context here means every "round" would silently retrain
            # from scratch (the exact bug this fix addresses) with no way to
            # detect it downstream, so this must fail loudly, not fall back.
            if self.context is None:
                raise RuntimeError(
                    f"Client {self.client_id}: local_only=True requires "
                    f"self.context to be set (by client_fn) for cross-round "
                    f"state persistence; got context=None."
                )
            persisted_full = self._load_persisted_params("local_only_full_model")
            if persisted_full is not None:
                self._load_full_model_params(persisted_full)
            # else: this client's first-ever fit() call this run -- keep the
            # freshly-initialized model as the round-1 starting point.
        else:
            self._set_parameters(parameters)
            # If task heads are federated, _set_parameters() above already
            # loaded the correctly server-aggregated head from `parameters`;
            # a locally-persisted pre-aggregation head must NOT overwrite it.
            if not bool(self.config.get("federate_task_heads", False)):
                persisted_heads = self._load_persisted_params("task_heads")
                if persisted_heads is not None:
                    self._load_task_head_params(persisted_heads)
                # else: first-ever fit() call for this client -- keep the
                # freshly-initialized task heads as the round-1 starting point.

        fedprox_mu = float(self.config.get("fedprox_mu", 0.0))
        global_params_np = parameters if fedprox_mu > 0 else None

        num_epochs = int(config.get("local_epochs", self.config.get("local_epochs", 5)))
        # Pass server_round so the DataLoader uses a round-specific shuffle seed,
        # giving different batch orderings each FL round.
        server_round = int(config.get("server_round", 0))
        train_metrics = self._train_local(
            num_epochs, global_params_np=global_params_np, server_round=server_round
        )

        emit_legacy = bool(self.config.get("emit_legacy_routing_metrics", True))
        embedding_dp_epsilon = float(self.config.get("embedding_dp_epsilon", 0.0))
        client_train_indices = getattr(
            self.data_loader, "_client_train_indices", {}
        ).get(self.client_id, [])
        state_supervised_examples = int(
            self.data_loader.state_supervision_mask[
                torch.as_tensor(client_train_indices, dtype=torch.long)
            ].sum()
        ) if len(client_train_indices) else 0
        metrics = {
            "train_loss": train_metrics.get("train_loss", 0.0),
            "client_id": float(self.client_id),
            "state_supervised_client": float(bool(
                self.data_loader.state_supervision_mask[
                    torch.as_tensor(
                        self.data_loader.client_splits[self.client_id],
                        dtype=torch.long,
                    )
                ].any()
            )),
            "state_supervised_examples": float(state_supervised_examples),
            "embedding_dp_epsilon": embedding_dp_epsilon,
        }
        if emit_legacy:
            centroid = self._compute_behavior_centroid()
            if embedding_dp_epsilon > 0.0:
                delta = 1e-5
                sigma_ldp = (
                    2.0 * np.sqrt(2.0 * np.log(1.25 / delta))
                    / embedding_dp_epsilon
                )
                centroid = centroid + np.random.normal(
                    0.0, sigma_ldp, centroid.shape
                ).astype(np.float32)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            metrics["behavior_centroid"] = centroid.astype(np.float32).tobytes()
        if "epsilon" in train_metrics:
            metrics["epsilon"] = train_metrics["epsilon"]
            metrics["dp_noise_multiplier"] = train_metrics.get("dp_noise_multiplier", 0.0)
            metrics["dp_n_steps"] = float(train_metrics.get("dp_n_steps", 0))
            metrics["dp_sample_rate"] = train_metrics.get("dp_sample_rate", 0.0)

        # Serialize task head parameters for server-side cold-start checkpointing.
        # The server averages these per cluster to produce a cluster-average task head
        # for new drivers who haven't fine-tuned locally yet.
        if emit_legacy:
            try:
                _underlying = self._unwrap_model(self.model)
                poi_flat = np.concatenate([p.detach().cpu().numpy().ravel()
                                           for p in _underlying.poi_head.parameters()])
                music_flat = np.concatenate([p.detach().cpu().numpy().ravel()
                                             for p in _underlying.music_head.parameters()])
                metrics["poi_head_bytes"] = poi_flat.astype(np.float32).tobytes()
                metrics["music_head_bytes"] = music_flat.astype(np.float32).tobytes()
            except Exception:
                pass  # legacy checkpoint metrics are best effort

        if local_only:
            self._save_persisted_params("local_only_full_model", self._get_full_model_params())
        elif not bool(self.config.get("federate_task_heads", False)):
            underlying = self._unwrap_model(self.model)
            self._save_persisted_params(
                "task_heads",
                [p.detach().cpu().numpy() for p in underlying.get_task_specific_parameters()],
            )
        # else: heads are federated -- get_parameters() below already returns
        # the trained head for server aggregation; nothing to persist locally.

        return self.get_parameters(config), train_metrics.get("n_samples", 0), metrics

    def evaluate(self, parameters: list, config: dict) -> tuple:  # noqa: ARG002
        local_only = bool(self.config.get("local_only", False))
        if local_only:
            if self.context is None:
                raise RuntimeError(
                    f"Client {self.client_id}: local_only=True requires "
                    f"self.context to be set (by client_fn) for cross-round "
                    f"state persistence; got context=None."
                )
            persisted_full = self._load_persisted_params("local_only_full_model")
            if persisted_full is not None:
                self._load_full_model_params(persisted_full)
            # else: evaluated before ever being fit (partial-participation edge
            # case) -- fall back to the freshly-initialized model.
        else:
            self._set_parameters(parameters)
            # See fit(): a federated head is already correct after
            # _set_parameters(); only overwrite from local state otherwise.
            if not bool(self.config.get("federate_task_heads", False)):
                persisted_heads = self._load_persisted_params("task_heads")
                if persisted_heads is not None:
                    self._load_task_head_params(persisted_heads)
                # else: same partial-participation fallback -- keep the fresh head.
        val_metrics = self._evaluate_local()
        # Stable identifier for deterministic server-side aggregation order
        # (client_proxy.cid is a per-session random node_id).
        val_metrics["client_id"] = float(self.client_id)
        return (
            val_metrics.get("loss", 0.0),
            val_metrics.get("n_samples", 1),
            val_metrics,
        )


    # Internal helpers


    def _unwrap_model(self, model) -> BehaviorFLModel:
        while hasattr(model, "_module"):
            model = model._module
        if hasattr(model, "base_model") and model.base_model is not None:
            model = model.base_model
        return model


    # Cross-round local state persistence (context.state).

    # The Ray simulation backend calls client_fn fresh for every single
    # fit()/evaluate() message (not once per client), so nothing stored as a
    # plain Python attribute on self.model survives between calls. Anything
    # that must persist locally across the 100 FL rounds of a run (task
    # heads kept off the server, or the full model for local-only clients)
    # must be saved into self.context.state, which Flower's RunInfoStore
    # explicitly persists and restores across calls for the same
    # (run_id, node_id) pair.


    def _load_persisted_params(self, key: str) -> Optional[list]:
        if self.context is None:
            return None
        rec = self.context.state.parameters_records.get(key)
        if rec is None:
            return None
        return [rec[f"p{i}"].numpy() for i in range(len(rec))]

    def _save_persisted_params(self, key: str, params_np: list) -> None:
        if self.context is None:
            return
        self.context.state.parameters_records[key] = ParametersRecord(
            {f"p{i}": array_from_numpy(np.asarray(a, dtype=np.float32))
             for i, a in enumerate(params_np)},
            keep_input=False,
        )

    def _load_task_head_params(self, params_np: list) -> None:
        underlying = self._unwrap_model(self.model)
        targets = list(underlying.get_task_specific_parameters())
        if len(params_np) != len(targets):
            raise ValueError(
                f"Client {self.client_id}: persisted task-head param count "
                f"({len(params_np)}) does not match model's task-head param "
                f"count ({len(targets)}) -- refusing to load a stale/mismatched "
                f"state rather than silently truncating via zip()."
            )
        for p, val in zip(targets, params_np):
            if tuple(p.shape) != tuple(val.shape):
                raise ValueError(
                    f"Client {self.client_id}: persisted task-head param shape "
                    f"{tuple(val.shape)} does not match model param shape {tuple(p.shape)}."
                )
            p.data = torch.tensor(val, dtype=p.dtype, device=self.device)

    def _get_full_model_params(self) -> list:
        underlying = self._unwrap_model(self.model)
        if hasattr(underlying, "get_shared_state_dict"):
            shared = [v.detach().cpu().numpy() for v in underlying.get_shared_state_dict().values()]
        else:
            shared = [p.detach().cpu().numpy() for p in underlying.get_shared_parameters()]
        heads = [p.detach().cpu().numpy() for p in underlying.get_task_specific_parameters()]
        return shared + heads

    def _load_full_model_params(self, params_np: list) -> None:
        underlying = self._unwrap_model(self.model)
        n_expected = len(self._get_full_model_params())
        if len(params_np) != n_expected:
            raise ValueError(
                f"Client {self.client_id}: persisted full-model param count "
                f"({len(params_np)}) does not match model's total param count "
                f"({n_expected}) -- refusing to load a stale/mismatched state "
                f"rather than silently truncating via zip()."
            )
        if hasattr(underlying, "get_shared_state_dict"):
            sd_keys = list(underlying.get_shared_state_dict().keys())
            n_shared = len(sd_keys)
            sd = {k: torch.tensor(v, dtype=torch.float32, device=self.device)
                  for k, v in zip(sd_keys, params_np[:n_shared])}
            underlying.load_shared_state_dict(sd)
        else:
            shared_params = underlying.get_shared_parameters()
            n_shared = len(shared_params)
            for p, val in zip(shared_params, params_np[:n_shared]):
                p.data = torch.tensor(val, dtype=p.dtype, device=self.device)
        for p, val in zip(underlying.get_task_specific_parameters(), params_np[n_shared:]):
            p.data = torch.tensor(val, dtype=p.dtype, device=self.device)

    def _set_parameters(self, parameters: list) -> None:
        underlying = self._unwrap_model(self.model)
        federate_heads = bool(self.config.get("federate_task_heads", False))
        if hasattr(underlying, "get_shared_state_dict"):
            sd_keys = list(underlying.get_shared_state_dict().keys())
            n_shared = len(sd_keys)
            if federate_heads:
                n_head = len(underlying.get_task_specific_parameters())
                if len(parameters) != n_shared + n_head:
                    raise ValueError(
                        f"Client {self.client_id}: param count mismatch with federated heads — "
                        f"expected {n_shared + n_head}, got {len(parameters)}."
                    )
                sd = {k: torch.tensor(v, dtype=torch.float32, device=self.device)
                      for k, v in zip(sd_keys, parameters[:n_shared])}
                underlying.load_shared_state_dict(sd)
                for p, val in zip(underlying.get_task_specific_parameters(), parameters[n_shared:]):
                    p.data = torch.tensor(val, dtype=p.dtype, device=self.device)
            else:
                if n_shared != len(parameters):
                    raise ValueError(
                        f"Client {self.client_id}: shared state-dict count mismatch — "
                        f"model has {n_shared}, server sent {len(parameters)}."
                    )
                sd = {k: torch.tensor(v, dtype=torch.float32, device=self.device)
                      for k, v in zip(sd_keys, parameters)}
                underlying.load_shared_state_dict(sd)
        else:
            shared_params = underlying.get_shared_parameters()
            n_shared = len(shared_params)
            if federate_heads:
                task_params = underlying.get_task_specific_parameters()
                if len(parameters) != n_shared + len(task_params):
                    raise ValueError(
                        f"Client {self.client_id}: param count mismatch with federated heads — "
                        f"expected {n_shared + len(task_params)}, got {len(parameters)}."
                    )
                for p, val in zip(shared_params, parameters[:n_shared]):
                    p.data = torch.tensor(val, dtype=p.dtype, device=self.device)
                for p, val in zip(task_params, parameters[n_shared:]):
                    p.data = torch.tensor(val, dtype=p.dtype, device=self.device)
            else:
                if n_shared != len(parameters):
                    raise ValueError(
                        f"Client {self.client_id}: shared param count mismatch — "
                        f"model has {n_shared}, server sent {len(parameters)}."
                    )
                for p, new_val in zip(shared_params, parameters):
                    p.data = torch.tensor(new_val, dtype=p.dtype, device=self.device)

    def _train_local(
        self,
        num_epochs: int,
        global_params_np: Optional[list] = None,
        server_round: int = 0,
    ) -> dict:
        """Local training with PCGrad + contrastive + optional DP + optional FedProx.

        Gradient conflict resolution: PCGrad (Yu et al., NeurIPS 2020).
        Implemented in src/utils/pcgrad.py as apply_pcgrad_to_gradients().
        """
        batch_size = int(self.config.get("batch_size", 32))
        lr = float(self.config.get("learning_rate", 0.001))
        wd = float(self.config.get("weight_decay", 1e-4))
        contrastive_weight = float(self.config.get("contrastive_weight", 0.1))
        use_pcgrad = bool(self.config.get("use_pcgrad", True))
        pcgrad_threshold = float(self.config.get("pcgrad_threshold", 0.0))
        use_dp = bool(self.config.get("use_dp", False))
        fedprox_mu = float(self.config.get("fedprox_mu", 0.0))
        force_tier1 = bool(self.config.get("force_tier1", False))
        poi_only = bool(self.config.get("poi_only", False))
        state_only = bool(self.config.get("state_only", False))
        sensor_block_dropout_p = float(self.config.get("sensor_block_dropout_p", 0.0))

        dp_noise_multiplier = float(self.config.get("dp_noise_multiplier", 1.1))
        dp_max_grad_norm = float(self.config.get("dp_max_grad_norm", 1.0))

        # Class weights for imbalanced POI labels
        poi_cw = getattr(self.data_loader, "poi_class_weights", None)
        music_cw = getattr(self.data_loader, "music_class_weights", None)
        if poi_cw is not None:
            poi_cw = poi_cw.to(self.device)
        if music_cw is not None:
            music_cw = music_cw.to(self.device)

        train_loader = self.data_loader.get_client_loader(
            self.client_id, split="train", batch_size=batch_size,
            server_round=server_round,
        )
        if len(train_loader.dataset) == 0:  # type: ignore[arg-type]
            return {"train_loss": 0.0, "n_samples": 0}

        self.model.train()

        privacy = None
        _pre_dp_model = None
        if use_dp:
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from federated.privacy import DifferentialPrivacy, PrivacyConfig
            privacy_cfg = PrivacyConfig(
                noise_multiplier=dp_noise_multiplier,
                max_grad_norm=dp_max_grad_norm,
            )
            privacy = DifferentialPrivacy(privacy_cfg)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)

        if use_dp and privacy is not None:
            try:
                import copy
                _pre_dp_model = self.model
                self.model = copy.deepcopy(self.model)
                self.model, optimizer, train_loader = privacy.attach_to_model(
                    self.model, optimizer, train_loader
                )
                self.model.train()
            except Exception as e:
                logger.warning(f"Client {self.client_id}: DP attach failed ({e}), no DP")
                if _pre_dp_model is not None:
                    self.model = _pre_dp_model
                    _pre_dp_model = None
                use_dp = False
                privacy = None

        # FedProx global reference — zip against get_shared_parameters() order,
        # NOT model.parameters() order (which differs: encoder, poi_head, music_head,
        # aligner vs shared = encoder, aligner only).
        global_param_tensors = None
        if fedprox_mu > 0 and global_params_np is not None:
            underlying_ref = self._unwrap_model(self.model)
            global_param_tensors = [
                torch.tensor(p_np, dtype=p.dtype, device=self.device)
                for p_np, p in zip(global_params_np,
                                   underlying_ref.get_shared_parameters())
            ]

        total_loss = 0.0
        n_samples = 0
        n_batches = 0
        n_dp_steps = 0

        underlying = self._unwrap_model(self.model)

        for _ in range(num_epochs):
            for batch in train_loader:
                features = batch["features"].to(self.device)
                tier_mask = batch.get("tier_mask", torch.ones_like(features)).to(self.device)
                if force_tier1:
                    # Zero out OBD-II features (10-14) to simulate Tier 1 GPS-only vehicle
                    tier_mask = tier_mask.clone()
                    tier_mask[:, 10:15] = 0.0
                    features = features.clone()
                    features[:, 10:15] = 0.0
                poi_labels = batch["poi_label"].to(self.device)
                music_labels = batch["music_label"].to(self.device)
                music_supervision_mask = batch.get("state_supervision_mask")
                if music_supervision_mask is not None:
                    music_supervision_mask = music_supervision_mask.to(
                        self.device, dtype=torch.bool
                    )
                if state_only and (
                    music_supervision_mask is not None
                    and int(music_supervision_mask.sum().item()) == 0
                ):
                    continue

                # TierDrop: one structured Bernoulli draw per equipped trip.
                # State supervision remains active, forcing the shared encoder
                # to transfer the real VED state task to GPS-only inference.
                if sensor_block_dropout_p > 0.0:
                    equipped = tier_mask[:, 10:15].sum(dim=1) > 0
                    drop = equipped & (
                        torch.rand(len(features), device=self.device) < sensor_block_dropout_p
                    )
                    if int(drop.sum().item()) > 0:
                        features = features.clone()
                        tier_mask = tier_mask.clone()
                        features[drop, 10:15] = 0.0
                        tier_mask[drop, 10:15] = 0.0

                B = len(features)
                if B < 2:
                    continue

                optimizer.zero_grad()

                if use_dp:
                    from opacus.grad_sample.grad_sample_module import GradSampleModule
                    if isinstance(self.model, GradSampleModule) and hasattr(self.model, "_module"):
                        # iterate_submodules is not a public Opacus API; use standard modules()
                        for m in self.model._module.modules():
                            if hasattr(m, "activations"):
                                m.activations = []
                    self.model.train()
                    if hasattr(self.model, "enable_hooks"):
                        self.model.enable_hooks()

                    for _p in self.model.parameters():
                        if _p.requires_grad and hasattr(_p, "_current_grad_sample"):
                            if _p._current_grad_sample is None or not isinstance(
                                _p._current_grad_sample, torch.Tensor
                            ):
                                delattr(_p, "_current_grad_sample")

                    # Pass tier_mask through the Opacus wrapper so BehaviorEncoderMLP
                    # receives it and runs learned imputation (tier_mask_proj).
                    # Pre-applying masking without passing tier_mask would set
                    # tier_mask=ones inside the encoder, zeroing out imputed_fill.
                    concatenated = self.model(features, tier_mask)
                    poi_logits = concatenated[:, :underlying.num_poi_classes]
                    music_logits = concatenated[:, underlying.num_poi_classes:]

                    # DP path: mean loss over genuinely supervised tasks only.
                    # Music CE is normalised by BATCH SIZE (B), not supervised count
                    # (M), so gradient scale stays consistent as batch composition
                    # varies between VED and non-VED samples.  Dividing by M would
                    # inflate music gradients when few VED samples are present and
                    # make the effective music learning rate depend on batch sampling.
                    B_dp = features.shape[0]
                    poi_loss_dp = F.cross_entropy(poi_logits, poi_labels, reduction="sum") / B_dp
                    active_losses = [] if state_only else [poi_loss_dp]
                    if poi_only:
                        pass  # POI-only ablation: music task fully disabled
                    elif music_supervision_mask is None:
                        active_losses.append(
                            F.cross_entropy(music_logits, music_labels, reduction="sum") / B_dp
                        )
                    elif int(music_supervision_mask.sum().item()) > 0:
                        active_losses.append(
                            F.cross_entropy(
                                music_logits[music_supervision_mask],
                                music_labels[music_supervision_mask],
                                reduction="sum",
                            ) / B_dp
                        )
                    loss = torch.stack(active_losses).mean()
                    loss.backward()
                    loss_val = loss.detach().item()

                    # FedProx proximal term is intentionally SKIPPED in the DP branch.
                    # Calling prox_loss.backward() inside the Opacus-wrapped context
                    # would route the prox gradients through Opacus per-sample clipping,
                    # which (a) adds noise to a non-private, deterministic regulariser
                    # and (b) corrupts the DP accounting by mixing non-data-dependent
                    # signal into the clipping step. DP + FedProx are mutually
                    # incompatible in this implementation.

                elif use_pcgrad and hasattr(underlying, "compute_full_loss"):
                    # PCGrad + NILW
                    # Two orthogonal mechanisms:
                    # NILW: scales task losses by normalized inverse-loss
                    # weights before PCGrad.
                    # Handles magnitude imbalance across tasks.
                    # PCGrad (Yu et al., 2020): projects per-task gradients to
                    # remove conflicting components. Handles direction
                    # conflict between tasks.
                    embedding = underlying.get_behavior_embedding(features, tier_mask)
                    poi_logits = (
                        underlying._destination_logits(embedding, features)
                        if hasattr(underlying, "_destination_logits")
                        else underlying.poi_head(embedding)
                    )
                    music_logits = underlying.music_head(embedding)

                    poi_loss = F.cross_entropy(poi_logits, poi_labels, weight=poi_cw)
                    if poi_only:
                        music_loss = torch.zeros((), device=self.device)
                    elif music_supervision_mask is None:
                        music_loss = F.cross_entropy(music_logits, music_labels, weight=music_cw)
                    elif int(music_supervision_mask.sum().item()) > 0:
                        music_loss = F.cross_entropy(
                            music_logits[music_supervision_mask],
                            music_labels[music_supervision_mask],
                            weight=music_cw,
                        )
                    else:
                        music_loss = torch.zeros((), device=self.device)

                    # Task 3: contrastive loss (if enabled)
                    contrastive_loss = None
                    if contrastive_weight > 0 and underlying.use_contrastive and underlying.aligner is not None:
                        raw_contrast = torch.zeros((), device=self.device)

                        b_proj = underlying.aligner.behavior_proj(embedding)
                        poi_proto = underlying.aligner.poi_proj(
                            underlying.aligner.poi_embeddings.weight
                        )
                        poi_contrast = underlying.aligner._prototype_contrastive_loss(
                            b_proj, poi_proto, poi_labels
                        )
                        if underlying.aligner.use_fedrcl:
                            poi_contrast = poi_contrast + (
                                underlying.aligner.fedrcl_penalty_weight
                                * underlying.aligner._fedrcl_divergence_penalty(
                                    b_proj, poi_labels
                                )
                            )
                        raw_contrast = raw_contrast + poi_contrast

                        if music_supervision_mask is None:
                            music_mask = torch.ones(len(music_labels), device=self.device, dtype=torch.bool)
                        else:
                            music_mask = music_supervision_mask
                        if int(music_mask.sum().item()) > 1:
                            emb_music = embedding[music_mask]
                            labels_music = music_labels[music_mask]
                            b_proj_music = underlying.aligner.behavior_proj(emb_music)
                            music_proto = underlying.aligner.music_proj(
                                underlying.aligner.music_embeddings.weight
                            )
                            music_contrast = underlying.aligner._prototype_contrastive_loss(
                                b_proj_music, music_proto, labels_music
                            )
                            if underlying.aligner.use_fedrcl:
                                music_contrast = music_contrast + (
                                    underlying.aligner.fedrcl_penalty_weight
                                    * underlying.aligner._fedrcl_divergence_penalty(
                                        b_proj_music, labels_music
                                    )
                                )
                            raw_contrast = raw_contrast + music_contrast

                        if underlying.use_uncertainty_weighting:
                            contrastive_loss = raw_contrast
                        else:
                            contrastive_loss = contrastive_weight * raw_contrast

                    # NILW: compute analytical inverse-loss weights and scale
                    # each task loss before handing to PCGrad.
                    # w_i = (1/L_i) / Σ_j(1/L_j)
                    # Guard: only include contrastive_loss in NILW when it is meaningfully
                    # non-zero (> 1e-8). A near-zero loss yields 1/tiny weight → NILW
                    # would assign ~100% weight to the nearly-solved task, collapsing all
                    # other task weights to zero. Matches the guard in compute_full_loss().

                    # Capture RAW losses before NILW scaling for monitoring purposes.
                    # The sum of NILW-weighted losses is a harmonic-mean proxy (~constant)
                    # and is not a useful convergence signal. Raw sum IS.
                    raw_poi_val = 0.0 if state_only else poi_loss.detach().item()
                    raw_music_val = music_loss.detach().item()
                    raw_contrast_val = contrastive_loss.detach().item() if contrastive_loss is not None else 0.0
                    state_task_active = (
                        not poi_only
                        and (
                            music_supervision_mask is None
                            or int(music_supervision_mask.sum().item()) > 0
                        )
                    )
                    if underlying.use_uncertainty_weighting:
                        try:
                            from utils.uncertainty import compute_nilw_weights
                        except ImportError:
                            from src.utils.uncertainty import compute_nilw_weights
                        _c_val = contrastive_loss.item() if contrastive_loss is not None else 0.0
                        # POI-only ablation: music is excluded from NILW inputs —
                        # a zero music loss would make 1/L_music blow up and collapse
                        # all other task weights to zero.
                        nilw_inputs = (
                            ([] if state_only else [poi_loss])
                            + ([music_loss] if state_task_active else [])
                            + ([contrastive_loss] if (contrastive_loss is not None and _c_val > 1e-8) else [])
                        )
                        w = compute_nilw_weights(nilw_inputs)
                        _wi = 0
                        if not state_only:
                            poi_loss = w[_wi] * poi_loss
                            _wi += 1
                        if state_task_active:
                            music_loss = w[_wi] * music_loss
                            _wi += 1
                        # Only apply the contrastive weight when contrastive was
                        # included in nilw_inputs (near-zero values are excluded
                        # to avoid IndexError / weight collapse).
                        if contrastive_loss is not None and _c_val > 1e-8:
                            contrastive_loss = w[_wi] * contrastive_loss

                    # Assemble task losses for PCGrad.
                    # FedProx proximal term (when active) is included here as an
                    # additional task so PCGrad can resolve conflicts between the
                    # proximal pull and task-specific gradients.  Adding prox AFTER
                    # PCGrad would bypass conflict resolution and corrupt the update.
                    task_losses = [] if state_only else [poi_loss]
                    if state_task_active:
                        task_losses.append(music_loss)
                    if contrastive_loss is not None:
                        task_losses.append(contrastive_loss)

                    prox_loss: "torch.Tensor | None" = None
                    if fedprox_mu > 0 and global_param_tensors is not None:
                        prox_terms = [
                            ((p - pg) ** 2).sum()
                            for p, pg in zip(
                                underlying.get_shared_parameters(),
                                global_param_tensors,
                            )
                        ]
                        if prox_terms:
                            prox_loss = (fedprox_mu / 2.0) * torch.stack(prox_terms).sum()
                            task_losses.append(prox_loss)

                    # Canonical PCGrad is applied only to one flattened shared-
                    # encoder gradient per active task.  Disjoint task heads are
                    # updated by their own ordinary gradients: including heads in
                    # the conflict dot product would let their norms distort the
                    # shared-task conflict test.
                    underlying.zero_grad()
                    param_list = list(underlying.parameters())
                    all_task_grads = []
                    for k, tl in enumerate(task_losses):
                        retain = (k < len(task_losses) - 1)
                        all_task_grads.append(list(torch.autograd.grad(
                            tl,
                            param_list,
                            retain_graph=retain,
                            create_graph=False,
                            allow_unused=True,
                        )))

                    shared_ids = {
                        id(p) for p in underlying.get_shared_parameters()
                    }
                    shared_indices = [
                        i for i, p in enumerate(param_list) if id(p) in shared_ids
                    ]
                    shared_vectors = []
                    for task_grads in all_task_grads:
                        shared_vectors.append(torch.cat([
                            (task_grads[i] if task_grads[i] is not None
                             else torch.zeros_like(param_list[i])).reshape(-1)
                            for i in shared_indices
                        ]))
                    projected = apply_pcgrad_to_gradients(
                        shared_vectors, threshold=pcgrad_threshold
                    )
                    combined_shared = torch.stack(projected).sum(dim=0)
                    offset = 0
                    for i in shared_indices:
                        p = param_list[i]
                        n_values = p.numel()
                        p.grad = combined_shared[offset:offset + n_values].reshape_as(p)
                        offset += n_values

                    # Non-shared parameters (the two disjoint heads) receive only
                    # the sum of their ordinary active-task gradients.
                    for i, p in enumerate(param_list):
                        if i in shared_indices:
                            continue
                        active_grads = [
                            task_grads[i] for task_grads in all_task_grads
                            if task_grads[i] is not None
                        ]
                        if active_grads:
                            p.grad = torch.stack(active_grads).sum(dim=0)

                    # Report raw (pre-NILW) sum — this actually reflects convergence.
                    # The sum of NILW-weighted losses is a harmonic-mean proxy (≈ constant)
                    # and would mask divergence or plateaus in the monitored training curve.
                    loss_val = raw_poi_val + raw_music_val + raw_contrast_val
                    if prox_loss is not None:
                        loss_val += prox_loss.detach().item()

                else:
                    # Standard training with full loss.
                    # MAS baseline (Zhuang et al. 2023): after mas_split_round,
                    # decouple the state task's gradient from the shared encoder
                    # ("split"), matching MAS's merge-then-split-per-task design
                    # adapted to a single joint model instead of separate FL jobs.
                    mas_split_round = self.config.get("mas_split_round")
                    detach_state_gradient = (
                        mas_split_round is not None
                        and server_round > int(mas_split_round)
                    )
                    loss, loss_components = underlying.compute_full_loss(
                        features, tier_mask, poi_labels, music_labels,
                        contrastive_weight=contrastive_weight,
                        poi_class_weights=poi_cw,
                        music_class_weights=music_cw,
                        music_supervision_mask=music_supervision_mask,
                        destination_weight=0.0 if state_only else 1.0,
                        music_weight=0.0 if poi_only else 1.0,
                        detach_state_gradient=detach_state_gradient,
                    )

                    if fedprox_mu > 0 and global_param_tensors is not None:
                        prox_terms = [
                            ((p - pg) ** 2).sum()
                            for p, pg in zip(
                                underlying.get_shared_parameters(),
                                global_param_tensors,
                            )
                        ]
                        if prox_terms:
                            loss = loss + (fedprox_mu / 2.0) * torch.stack(prox_terms).sum()

                    loss.backward()
                    # Log raw (pre-NILW) component sum for consistency with the
                    # PCGrad branch.  When use_uncertainty_weighting=True, `loss`
                    # is the NILW harmonic-mean proxy (~constant) and would mask
                    # convergence.  The raw sum of task losses reflects actual
                    # training progress and is comparable across all methods.
                    loss_val = (
                        (0.0 if state_only else loss_components["poi_loss"])
                        + loss_components["music_loss"]
                        + loss_components["contrastive_loss"]
                    )

                if not use_dp:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss_val
                n_samples += B
                n_batches += 1
                if use_dp:
                    n_dp_steps += 1

        result = {
            "train_loss": total_loss / max(n_batches, 1),
            "n_samples": n_samples,
        }

        if use_dp:
            epsilon_val = 0.0
            if privacy is not None:
                try:
                    spent = privacy.get_privacy_spent()
                    epsilon_val = float(spent.get("epsilon", 0.0))
                except Exception:
                    pass
            result["epsilon"] = epsilon_val
            result["delta"] = 1e-5
            result["dp_noise_multiplier"] = dp_noise_multiplier
            result["dp_n_steps"] = n_dp_steps
            n_train = len(train_loader.dataset)  # type: ignore[arg-type]
            # Clamp to [0, 1]: sample_rate > 1.0 is mathematically invalid for RDP
            # and will corrupt the Opacus accountant. This can happen when the client
            # has fewer training samples than batch_size.
            result["dp_sample_rate"] = float(min(1.0, batch_size / max(n_train, 1)))

        if use_dp and _pre_dp_model is not None:
            trained_underlying = self._unwrap_model(self.model)
            orig_underlying = self._unwrap_model(_pre_dp_model)
            for p_orig, p_trained in zip(orig_underlying.parameters(),
                                         trained_underlying.parameters()):
                p_orig.data.copy_(p_trained.data)
            self.model = _pre_dp_model

        return result

    def _evaluate_local(self) -> dict:
        """Evaluate on local validation split.

        Returns accuracy AND ranking metrics (HR@1/3/5, NDCG@3/5, MRR).
        HR@10 is intentionally omitted: with num_classes=10, HR@10=1.0 trivially
        for any model — it is a vacuous metric at this class count.
        """
        underlying = self._unwrap_model(self.model)
        underlying.eval()
        val_loader = self.data_loader.get_client_loader(
            self.client_id, split="val", batch_size=64
        )

        poi_total_loss = 0.0
        music_total_loss = 0.0
        poi_correct = 0
        music_correct = 0
        n_samples = 0
        music_n_samples = 0
        poi_ranked_all: list = []
        music_ranked_all: list = []
        poi_true_all: list = []
        music_true_all: list = []

        poi_cw = getattr(self.data_loader, "poi_class_weights", None)
        music_cw = getattr(self.data_loader, "music_class_weights", None)

        force_tier1 = bool(self.config.get("force_tier1", False))
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(self.device)
                tier_mask = batch.get("tier_mask", torch.ones_like(features)).to(self.device)
                if force_tier1:
                    tier_mask = tier_mask.clone()
                    tier_mask[:, 10:15] = 0.0
                    features = features.clone()
                    features[:, 10:15] = 0.0
                poi_labels = batch["poi_label"].to(self.device)
                music_labels = batch["music_label"].to(self.device)
                music_supervision_mask = batch.get("state_supervision_mask")
                if music_supervision_mask is not None:
                    music_supervision_mask = music_supervision_mask.to(
                        self.device, dtype=torch.bool
                    )

                poi_logits, music_logits = underlying(features, tier_mask)

                poi_loss = F.cross_entropy(
                    poi_logits, poi_labels,
                    weight=poi_cw.to(poi_logits.device) if poi_cw is not None else None,
                )
                if music_supervision_mask is None:
                    music_loss = F.cross_entropy(
                        music_logits, music_labels,
                        weight=music_cw.to(music_logits.device) if music_cw is not None else None,
                    )
                    active_music_idx = torch.arange(len(music_labels), device=self.device)
                elif int(music_supervision_mask.sum().item()) > 0:
                    music_loss = F.cross_entropy(
                        music_logits[music_supervision_mask], music_labels[music_supervision_mask],
                        weight=music_cw.to(music_logits.device) if music_cw is not None else None,
                    )
                    active_music_idx = torch.nonzero(music_supervision_mask, as_tuple=False).squeeze(1)
                else:
                    music_loss = torch.zeros((), device=self.device)
                    active_music_idx = torch.empty(0, dtype=torch.long, device=self.device)
                # Accumulate losses by task, each weighted by its own sample count.
                # POI loss is mean over len(features); music loss is mean over supervised
                # music samples only. Tracking separately avoids the old bug where music
                # was diluted when divided by total n_samples.
                poi_total_loss += poi_loss.item() * len(features)
                if len(active_music_idx) > 0:
                    music_total_loss += music_loss.item() * int(len(active_music_idx))

                poi_correct += int((poi_logits.argmax(1) == poi_labels).sum())
                # active_music_idx is always non-empty when music_supervision_mask is None
                # (it is set to torch.arange(len(music_labels))), so the elif branch is
                # unreachable and has been removed.
                if len(active_music_idx) > 0:
                    music_correct += int(
                        (music_logits[active_music_idx].argmax(1) == music_labels[active_music_idx]).sum()
                    )
                    music_n_samples += int(len(active_music_idx))
                n_samples += len(features)

                poi_ranked_all.extend(
                    poi_logits.argsort(dim=1, descending=True).cpu().tolist()
                )
                poi_true_all.extend(poi_labels.cpu().tolist())
                # Same rationale: active_music_idx is always non-empty when
                # music_supervision_mask is None, so a single branch suffices.
                if len(active_music_idx) > 0:
                    music_ranked_all.extend(
                        music_logits[active_music_idx].argsort(dim=1, descending=True).cpu().tolist()
                    )
                    music_true_all.extend(music_labels[active_music_idx].cpu().tolist())

        if n_samples == 0:
            return {
                "loss": 0.0, "n_samples": 0,
                "destination_accuracy": 0.0,
                "destination_hr@1": 0.0, "destination_hr@3": 0.0,
                "destination_ndcg@3": 0.0, "destination_ndcg@5": 0.0,
                "destination_mrr": 0.0,
                "state_accuracy": 0.0, "state_n_samples": 0,
            }

        poi_mean_loss = poi_total_loss / n_samples
        music_mean_loss = music_total_loss / music_n_samples if music_n_samples > 0 else 0.0
        # Report mean of per-task mean losses — each task normalised by its own sample count.
        # This avoids the old bug where music loss was diluted by total N when fewer samples
        # had music supervision.
        combined_loss = (poi_mean_loss + music_mean_loss) / (2 if music_n_samples > 0 else 1)

        poi_rec = compute_hr_ndcg(poi_ranked_all, poi_true_all, k_values=[1, 3, 5])
        music_rec = compute_hr_ndcg(music_ranked_all, music_true_all, k_values=[1, 3, 5]) if music_true_all else {
            "hr@1": 0.0, "hr@3": 0.0, "hr@5": 0.0,
            "ndcg@1": 0.0, "ndcg@3": 0.0, "ndcg@5": 0.0,
            "mrr": 0.0,
        }

        poi_preds = [r[0] for r in poi_ranked_all]
        music_preds = [r[0] for r in music_ranked_all]
        poi_classes = int(getattr(underlying, "num_destination_classes", 10))
        state_classes = int(getattr(underlying, "num_state_classes", 2))
        poi_cls = compute_classification_metrics(
            poi_preds, poi_true_all, num_classes=poi_classes
        )
        music_cls = (
            compute_classification_metrics(
                music_preds, music_true_all, num_classes=state_classes
            )
            if music_true_all
            else compute_classification_metrics([], [], num_classes=state_classes)
        )

        return {
            "loss": combined_loss,
            "n_samples": n_samples,
            "destination_accuracy": float(poi_correct / n_samples),
            "destination_hr@1": poi_rec["hr@1"],
            "destination_hr@3": poi_rec["hr@3"],
            "destination_hr@5": poi_rec["hr@5"],
            "destination_ndcg@3": poi_rec["ndcg@3"],
            "destination_ndcg@5": poi_rec["ndcg@5"],
            "destination_mrr": poi_rec["mrr"],
            "state_accuracy": (
                float(music_correct / music_n_samples) if music_n_samples > 0 else 0.0
            ),
            "destination_macro_f1": poi_cls["macro_f1"],
            "destination_balanced_acc": poi_cls["balanced_acc"],
            "state_macro_f1": music_cls["macro_f1"],
            "state_balanced_acc": music_cls["balanced_acc"],
            "state_n_samples": music_n_samples,
            **{f"destination_recall_class_{c}": poi_cls[f"recall_class_{c}"] for c in range(poi_classes)},
            **{f"state_recall_class_{c}": music_cls[f"recall_class_{c}"] for c in range(state_classes)},
            **{f"destination_tp_class_{c}": poi_cls[f"tp_class_{c}"] for c in range(poi_classes)},
            **{f"destination_fp_class_{c}": poi_cls[f"fp_class_{c}"] for c in range(poi_classes)},
            **{f"destination_support_class_{c}": poi_cls[f"support_class_{c}"] for c in range(poi_classes)},
            **{f"state_tp_class_{c}": music_cls[f"tp_class_{c}"] for c in range(state_classes)},
            **{f"state_fp_class_{c}": music_cls[f"fp_class_{c}"] for c in range(state_classes)},
            **{f"state_support_class_{c}": music_cls[f"support_class_{c}"] for c in range(state_classes)},
        }

    def _compute_behavior_centroid(self) -> np.ndarray:
        """Compute behavior embedding centroid over local training data.

        Uses a properly weighted mean (not mean-of-batch-means) to handle
        unequal final batch sizes.
        """
        underlying = self._unwrap_model(self.model)
        underlying.eval()
        train_loader = self.data_loader.get_client_loader(
            self.client_id, split="train", batch_size=64
        )

        dim = getattr(underlying, "embedding_dim",
                      getattr(underlying.behavior_encoder
                              if hasattr(underlying, "behavior_encoder") else underlying,
                              "embedding_dim", 128))

        total_emb = torch.zeros(dim, device=self.device)
        total_count = 0

        with torch.no_grad():
            for batch in train_loader:
                features = batch["features"].to(self.device)
                tier_mask = batch.get("tier_mask", torch.ones_like(features)).to(self.device)
                emb = underlying.get_behavior_embedding(features, tier_mask)  # [B, D]
                total_emb += emb.sum(dim=0)
                total_count += emb.shape[0]

        if total_count == 0:
            return np.zeros(dim, dtype=np.float32)

        centroid = F.normalize(total_emb / total_count, p=2, dim=-1)
        return centroid.cpu().numpy().astype(np.float32)
