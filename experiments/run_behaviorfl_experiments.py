"""Run destination/state federated experiments with Flower and PyTorch."""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")

import flwr as fl
from flwr.server import ServerConfig
from flwr.common import (
    Context, FitIns, GetPropertiesIns, parameters_to_ndarrays, ndarrays_to_parameters,
)

from data.behavior_loader import BehaviorDataLoader
from federated.behavior_cluster_strategy import BehaviorClusterStrategy
from flower.behavior_client import BehaviorFLClient
from models.behaviorfl_model import BehaviorFLModel
from utils.metrics import aggregate_client_eval_metrics


class CheckpointingFedAvg(fl.server.strategy.FedAvg):
    """Standard Flower FedAvg with the last aggregate retained for artifacts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_global_params_np = None
        # Optional per-round centralized evaluation hook.
        # Flower calls Strategy.evaluate(round, parameters) once per round
        # regardless of fraction_evaluate, so this runs independently of the
        # (disabled, for speed) per-client distributed evaluation below.
        self.centralized_eval_fn = None
        self.eval_every_n_rounds = 0

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is not None:
            self._last_global_params_np = parameters_to_ndarrays(parameters)
        return parameters, metrics

    def evaluate(self, server_round, parameters):
        if self.centralized_eval_fn is None or self.eval_every_n_rounds <= 0:
            return None
        if server_round != 1 and server_round % self.eval_every_n_rounds != 0:
            return None
        arrays = parameters_to_ndarrays(parameters)
        metrics = self.centralized_eval_fn(arrays)
        if not metrics:
            return None
        return 0.0, metrics


class TaskAwareFedAvg(CheckpointingFedAvg):
    """FedAvg with reserved slots for discovered state-supervised clients.

    Eligibility comes from successful fits. Remaining slots are sampled
    uniformly, keeping the total cohort size fixed.
    """

    def __init__(self, *args, state_clients_per_round=20,
                 experiment_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_clients_per_round = int(state_clients_per_round)
        self.experiment_seed = int(experiment_seed)
        self._known_state_proxy_ids = set()
        self._reserved_state_by_round = {}
        # Real per-client selection counts (dataset client_id -> times fit
        # succeeded), for the participation-fairness analysis. Keyed by the
        # client-reported "client_id" metric (BehaviorFLClient.fit()), not
        # Flower's internal cid, so it maps directly onto dataset client ids.
        self.selection_counts: dict = {}

    def configure_fit(self, server_round, parameters, client_manager):
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        fit_ins = FitIns(parameters, config)
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        if client_manager.num_available() < min_num_clients:
            return []

        available = list(client_manager.all().values())
        rng = np.random.default_rng(
            self.experiment_seed * 100_003 + int(server_round)
        )
        known_state = [
            client for client in available
            if client.cid in self._known_state_proxy_ids
        ]
        n_state = min(
            self.state_clients_per_round, len(known_state), sample_size
        )
        chosen_state = (
            [known_state[int(i)] for i in rng.choice(
                len(known_state), size=n_state, replace=False
            )]
            if n_state else []
        )
        chosen_ids = {client.cid for client in chosen_state}
        remaining = [client for client in available if client.cid not in chosen_ids]
        n_remaining = sample_size - len(chosen_state)
        chosen_general = (
            [remaining[int(i)] for i in rng.choice(
                len(remaining), size=n_remaining, replace=False
            )]
            if n_remaining else []
        )
        self._reserved_state_by_round[int(server_round)] = len(chosen_state)
        return [(client, fit_ins) for client in chosen_state + chosen_general]

    def aggregate_fit(self, server_round, results, failures):
        actual_state_clients = 0
        for client_proxy, fit_res in results:
            if float(fit_res.metrics.get("state_supervised_client", 0.0)) > 0.5:
                actual_state_clients += 1
                self._known_state_proxy_ids.add(client_proxy.cid)
            dataset_cid = fit_res.metrics.get("client_id")
            if dataset_cid is not None:
                self.selection_counts[dataset_cid] = self.selection_counts.get(dataset_cid, 0) + 1
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        metrics = dict(metrics or {})
        metrics["known_state_clients"] = len(self._known_state_proxy_ids)
        metrics["reserved_state_clients"] = self._reserved_state_by_round.get(
            int(server_round), 0
        )
        metrics["actual_state_clients"] = actual_state_clients
        return parameters, metrics

class OracleTaskAwareFedAvg(TaskAwareFedAvg):
    """Task-aware sampling with eligibility queried before round 1.

    This reference uses the same cohort size and state-client quota as the
    online sampler, but queries every client once instead of discovering
    eligibility through participation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._oracle_primed = False

    def configure_fit(self, server_round, parameters, client_manager):
        if not self._oracle_primed:
            for client in client_manager.all().values():
                res = client.get_properties(
                    GetPropertiesIns(config={}), timeout=None, group_id=None
                )
                if float(res.properties.get("state_supervised_client", 0.0)) > 0.5:
                    self._known_state_proxy_ids.add(client.cid)
            self._oracle_primed = True
        return super().configure_fit(server_round, parameters, client_manager)


class ConflictAversePersonalizedFedAvg(CheckpointingFedAvg):
    """FedHCA2-style baseline adapted to a shared destination/state model.

    Client deltas are weighted by cosine similarity to the consensus update,
    with a bonus for the majority state-supervision profile. This adaptation
    does not implement the published cross-attention or hyper-network.
    """

    def __init__(self, *args, conflict_temperature: float = 4.0,
                 profile_bonus: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.conflict_temperature = float(conflict_temperature)
        self.profile_bonus = float(profile_bonus)

    def aggregate_fit(self, server_round, results, failures):
        if not results or self._last_global_params_np is None:
            return super().aggregate_fit(server_round, results, failures)

        prior = [np.asarray(p) for p in self._last_global_params_np]
        client_arrays, n_examples, profiles = [], [], []
        for _client_proxy, fit_res in results:
            client_arrays.append(parameters_to_ndarrays(fit_res.parameters))
            n_examples.append(fit_res.num_examples)
            profiles.append(float(fit_res.metrics.get("state_supervised_client", 0.0)))

        deltas_flat = np.stack([
            np.concatenate([(a - p).reshape(-1) for a, p in zip(arrs, prior)])
            for arrs in client_arrays
        ])
        n_examples_arr = np.asarray(n_examples, dtype=np.float64)

        mean_direction = np.average(deltas_flat, axis=0, weights=n_examples_arr)
        mean_norm = np.linalg.norm(mean_direction) + 1e-12
        cos_to_mean = np.array([
            float(np.dot(d, mean_direction) / ((np.linalg.norm(d) + 1e-12) * mean_norm))
            for d in deltas_flat
        ])

        majority_profile = 1.0 if np.average(profiles, weights=n_examples_arr) >= 0.5 else 0.0
        profile_weight = np.array([
            1.0 + self.profile_bonus if p == majority_profile else 1.0
            for p in profiles
        ])

        conflict_weight = np.exp(self.conflict_temperature * cos_to_mean)
        combined = conflict_weight * profile_weight * n_examples_arr
        combined = combined / combined.sum()

        aggregated = [
            sum(w * arrs[i] for w, arrs in zip(combined, client_arrays))
            for i in range(len(prior))
        ]
        self._last_global_params_np = aggregated
        parameters = ndarrays_to_parameters(aggregated)

        metrics_list = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
        metrics = dict(self.fit_metrics_aggregation_fn(metrics_list)) if self.fit_metrics_aggregation_fn else {}
        metrics["mean_conflict_cosine"] = float(np.average(cos_to_mean, weights=n_examples_arr))
        return parameters, metrics


# Experiment Configs

MODES = {
    "smoke": {"n_rounds": 3, "n_clients_per_round": 3, "seeds": [42], "total_clients": 3,
              "ray_cpus": 6, "ray_obj_store_mb": 400, "cpus_per_client": 2},
    "baseline": {"n_rounds": 15, "n_clients_per_round": 30, "seeds": [42], "total_clients": None,
                 "ray_cpus": 16, "ray_obj_store_mb": 2000, "cpus_per_client": 2},
    "full": {"n_rounds": 60, "n_clients_per_round": 100, "seeds": [7, 42, 123], "total_clients": None,
             "ray_cpus": 16, "ray_obj_store_mb": 4000, "cpus_per_client": 2},
}

LEGACY_METHODS = {
    # Baselines (shared-encoder FL, not true FedAvg)
    "fedavg_global": {
        "strategy": "fedavg",
        "use_pcgrad": False, "contrastive_weight": 0.0, "use_uncertainty": False,
        "n_clusters": 1,
    },
    "fedprox": {
        "strategy": "fedavg",
        "use_pcgrad": False, "contrastive_weight": 0.0, "use_uncertainty": False,
        "fedprox_mu": 0.01, "n_clusters": 1,
    },
    "fedavg_finetune": {
        # "FedAvg + more local fine-tune epochs" baseline.
        # local_epochs=10 distinguishes this from fedavg_global (local_epochs=5 default).
        # do not set this back to 5 — at 5 it is identical to fedavg_global,
        # making the ablation table vacuous and the comparison non-informative.
        "strategy": "fedavg",
        "use_pcgrad": False, "contrastive_weight": 0.0, "use_uncertainty": False,
        "local_epochs": 10, "n_clusters": 1,
    },
    "local_only": {
        "strategy": "fedavg",
        "use_pcgrad": False, "contrastive_weight": 0.0, "use_uncertainty": False,
        "local_only": True, "n_clusters": 1,
    },
    # BehaviorFL variants
    "behaviorfl_full": {
        "strategy": "behavior_cluster",
        "use_pcgrad": True, "pcgrad_threshold": 0.1,
        "contrastive_weight": 0.1, "use_uncertainty": True,
        "n_clusters": 5,
    },
    "behaviorfl_no_clustering": {
        "strategy": "fedavg",
        "use_pcgrad": True, "pcgrad_threshold": 0.1,
        "contrastive_weight": 0.1, "use_uncertainty": True,
        "n_clusters": 1,
    },
    "behaviorfl_no_contrastive": {
        "strategy": "behavior_cluster",
        "use_pcgrad": True, "pcgrad_threshold": 0.1,
        "contrastive_weight": 0.0, "use_uncertainty": True,
        "n_clusters": 5,
    },
    # POI-only (λ_m=0) ablation — isolates cluster routing from the music
    # auxiliary confound. Mirrors behaviorfl_no_contrastive / fedavg_global
    # respectively, but with the music task fully disabled (no music CE loss,
    # music excluded from NILW and PCGrad).
    "behaviorfl_poi_only": {
        "strategy": "behavior_cluster",
        "use_pcgrad": True, "pcgrad_threshold": 0.1,
        "contrastive_weight": 0.0, "use_uncertainty": True,
        "n_clusters": 5,
        "poi_only": True,
    },
    "fedavg_poi_only": {
        "strategy": "fedavg",
        "use_pcgrad": False, "contrastive_weight": 0.0, "use_uncertainty": False,
        "n_clusters": 1,
        "poi_only": True,
    },
    # No-clustering control — Core-matched loss config, K=5 model packages,
    # hash-random frozen client-to-model assignment (only the routing signal
    # differs from behaviorfl_no_contrastive).
    "behaviorfl_random_routing": {
        "strategy": "random_routing",
        "use_pcgrad": True, "pcgrad_threshold": 0.1,
        "contrastive_weight": 0.0, "use_uncertainty": True,
        "n_clusters": 5,
    },
    "behaviorfl_no_pcgrad": {
        "strategy": "behavior_cluster",
        "use_pcgrad": False, "contrastive_weight": 0.1, "use_uncertainty": True,
        "n_clusters": 5,
    },
    "behaviorfl_no_uncertainty": {
        "strategy": "behavior_cluster",
        "use_pcgrad": True, "pcgrad_threshold": 0.1,
        "contrastive_weight": 0.1, "use_uncertainty": False,
        "n_clusters": 5,
    },
    # H4 Ablation: federated task heads
    # Compares cluster routing + LOCAL heads (default) vs FEDERATED heads.
    # Used to verify that the +4.5% gain over FedAvg is due to routing, not head locality.
    "behaviorfl_federated_heads": {
        "strategy": "behavior_cluster",
        "use_pcgrad": True, "pcgrad_threshold": 0.1,
        "contrastive_weight": 0.0, "use_uncertainty": True,
        "n_clusters": 5,
        "federate_task_heads": True,
    },
    # FedRouteFMTL — OUR prior architecture baseline
    # BatchNorm1d → Opacus-incompatible → NEVER enable use_dp=True for this method.
    "fedroute_fmtl": {
        "strategy": "fedavg",
        "model_type": "fedroute_fmtl",
        "use_pcgrad": False, "contrastive_weight": 0.0, "use_uncertainty": False,
        "n_clusters": 1,
    },
    # K-sensitivity sweep for behaviorfl_full
    # (K=1 covered by behaviorfl_no_clustering, K=5 by behaviorfl_full)
    "behaviorfl_k3": {
        "strategy": "behavior_cluster",
        "use_pcgrad": True, "pcgrad_threshold": 0.1,
        "contrastive_weight": 0.1, "use_uncertainty": True,
        "n_clusters": 3,
    },
    "behaviorfl_k7": {
        "strategy": "behavior_cluster",
        "use_pcgrad": True, "pcgrad_threshold": 0.1,
        "contrastive_weight": 0.1, "use_uncertainty": True,
        "n_clusters": 7,
    },
    # IFCA-style baseline — clustering signal is per-client val loss, not
    # behavior embedding.
    # Faithful enough for an ITSC short paper: clients self-route to the
    # lowest-loss cluster model each round. See ifca_strategy.py for details.
    "ifca": {
        "strategy": "ifca",
        "use_pcgrad": True, "pcgrad_threshold": 0.1,
        "contrastive_weight": 0.1, "use_uncertainty": True,
        "n_clusters": 5,
    },
}

# Focused multi-task matrix used by the current BehaviorFL paper.  The old
# journal menu above remains readable for provenance but is not executed.
METHODS = {
    "destination_specialist": {
        "strategy": "fedavg", "poi_only": True,
        "federate_task_heads": True, "use_pcgrad": False,
        "use_uncertainty": False, "sensor_block_dropout_p": 0.0,
    },
    "state_specialist": {
        "strategy": "fedavg", "state_only": True,
        "federate_task_heads": True, "use_pcgrad": False,
        "use_uncertainty": False, "sensor_block_dropout_p": 0.25,
        "min_clients_per_round": 20,
    },
    "joint_no_tierdrop": {
        "strategy": "fedavg", "federate_task_heads": True,
        "use_pcgrad": False, "use_uncertainty": False,
        "sensor_block_dropout_p": 0.0,
    },
    "joint_tierdrop": {
        "strategy": "fedavg", "federate_task_heads": True,
        "use_pcgrad": False, "use_uncertainty": False,
        "sensor_block_dropout_p": 0.25,
    },
    "behaviorfl_mt": {
        "strategy": "task_aware_fedavg", "federate_task_heads": True,
        "use_pcgrad": False, "use_uncertainty": False,
        "sensor_block_dropout_p": 0.25, "state_clients_per_round": 20,
    },
    # External baselines (adapted; see class/config docstrings for exactly
    # what was simplified relative to the published methods)
    "mas": {
        "strategy": "fedavg", "federate_task_heads": True,
        "use_pcgrad": False, "use_uncertainty": False,
        "sensor_block_dropout_p": 0.0, "mas_split_round": 10,
    },
    # Oracle reference: identical to behaviorfl_mt except eligibility for all
    # 188 state-supervised clients is known before round 1 (via a one-time
    # get_properties metadata query) instead of discovered incrementally.
    # Isolates how much of the discovery-aware gain comes from the online
    # discovery mechanism itself vs. simply knowing eligibility upfront.
    "oracle_task_aware": {
        "strategy": "oracle_task_aware", "federate_task_heads": True,
        "use_pcgrad": False, "use_uncertainty": False,
        "sensor_block_dropout_p": 0.25, "state_clients_per_round": 20,
    },
    "fedhca2": {
        "strategy": "hetero_conflict_averse", "federate_task_heads": True,
        "use_pcgrad": False, "use_uncertainty": False,
        "sensor_block_dropout_p": 0.0,
    },
    # Component-contribution ablation (each removes exactly one
    # BehaviorFL-MT component relative to behaviorfl_mt above; "no
    # discovery" is joint_tierdrop, already defined)
    "behaviorfl_mt_no_dropout": {
        "strategy": "task_aware_fedavg", "federate_task_heads": True,
        "use_pcgrad": False, "use_uncertainty": False,
        "sensor_block_dropout_p": 0.0, "state_clients_per_round": 20,
    },
    "behaviorfl_mt_no_residual": {
        "strategy": "task_aware_fedavg", "federate_task_heads": True,
        "use_pcgrad": False, "use_uncertainty": False,
        "sensor_block_dropout_p": 0.25, "state_clients_per_round": 20,
        "geometry_residual_bound": 0.0,
    },
    # Doubles as the "masks-only-removed" component-ablation arm and the
    # zero-fill comparator in the imputation-comparison paragraph.
    "behaviorfl_mt_zero_fill": {
        "strategy": "task_aware_fedavg", "federate_task_heads": True,
        "use_pcgrad": False, "use_uncertainty": False,
        "sensor_block_dropout_p": 0.25, "state_clients_per_round": 20,
        "fill_mode": "zero",
    },
    "behaviorfl_mt_mean_fill": {
        "strategy": "task_aware_fedavg", "federate_task_heads": True,
        "use_pcgrad": False, "use_uncertainty": False,
        "sensor_block_dropout_p": 0.25, "state_clients_per_round": 20,
        "fill_mode": "batch_mean",
    },
    # Geometry-residual sensitivity sweep (single-seed diagnostic)
    "behaviorfl_mt_bound05": {
        "strategy": "task_aware_fedavg", "federate_task_heads": True,
        "use_pcgrad": False, "use_uncertainty": False,
        "sensor_block_dropout_p": 0.25, "state_clients_per_round": 20,
        "geometry_residual_bound": 0.05,
    },
    "behaviorfl_mt_bound20": {
        "strategy": "task_aware_fedavg", "federate_task_heads": True,
        "use_pcgrad": False, "use_uncertainty": False,
        "sensor_block_dropout_p": 0.25, "state_clients_per_round": 20,
        "geometry_residual_bound": 0.2,
    },
    "behaviorfl_mt_scale2": {
        "strategy": "task_aware_fedavg", "federate_task_heads": True,
        "use_pcgrad": False, "use_uncertainty": False,
        "sensor_block_dropout_p": 0.25, "state_clients_per_round": 20,
        "geometry_prior_scale": 2.0,
    },
    "behaviorfl_mt_scale6": {
        "strategy": "task_aware_fedavg", "federate_task_heads": True,
        "use_pcgrad": False, "use_uncertainty": False,
        "sensor_block_dropout_p": 0.25, "state_clients_per_round": 20,
        "geometry_prior_scale": 6.0,
    },
    "behaviorfl_mt_pcgrad_nilw": {
        "strategy": "task_aware_fedavg", "federate_task_heads": True,
        "use_pcgrad": True, "pcgrad_threshold": 0.0,
        "use_uncertainty": True, "sensor_block_dropout_p": 0.25,
        "state_clients_per_round": 20,
    },
    "behaviorfl_mt_dp": {
        "strategy": "fedavg", "federate_task_heads": True,
        "use_pcgrad": False, "use_uncertainty": False,
        "sensor_block_dropout_p": 0.25, "use_dp": True,
        "dp_noise_multiplier": 1.1, "dp_max_grad_norm": 1.0,
    },
}

DATASET_PATH = str(REPO_ROOT / "data/processed/behaviorfl_dataset.pt")
RESULTS_DIR = REPO_ROOT / "results"


def _model_ablation_kwargs(method_cfg: dict) -> dict:
    """fill_mode / geometry_prior_scale / geometry_residual_bound overrides,
    only included when the method config sets them so every other method
    keeps BehaviorFLModel's defaults exactly as before."""
    kwargs = {}
    for key in ("fill_mode", "geometry_prior_scale", "geometry_residual_bound"):
        if key in method_cfg:
            kwargs[key] = method_cfg[key]
    return kwargs


def _weighted_average(metrics):
    """Aggregate evaluate metrics from clients into round-level metrics."""
    return aggregate_client_eval_metrics(metrics)


def _aggregate_fit_metrics(metrics):
    """Aggregate training diagnostics, retaining the worst privacy cost."""
    out = {}
    for key in ("train_loss", "dp_n_steps", "dp_sample_rate"):
        weight = sum(max(0, int(n)) for n, values in metrics if key in values)
        if weight > 0:
            out[key] = sum(
                max(0, int(n)) * float(values[key])
                for n, values in metrics if key in values
            ) / weight
    epsilons = [float(values["epsilon"]) for _, values in metrics if "epsilon" in values]
    deltas = [float(values["delta"]) for _, values in metrics if "delta" in values]
    if epsilons:
        out["epsilon_max"] = max(epsilons)
    if deltas:
        out["delta_max"] = max(deltas)
    for key in ("state_supervised_client", "state_supervised_examples"):
        out[key] = sum(float(values.get(key, 0.0)) for _, values in metrics)
    return out


def make_strategy(method_cfg: dict, n_clients_per_round: int, total_clients: int,
                  experiment_seed: int = 42) -> BehaviorClusterStrategy:
    """Build Flower strategy from method config."""
    strategy_name = method_cfg.get("strategy", "fedavg")
    n_clusters = method_cfg.get("n_clusters", 5)
    participation_fraction = float(method_cfg.get("participation_fraction", 1.0))
    round_evaluation = bool(method_cfg.get("round_evaluation", True))

    def _on_fit_config(server_round: int) -> dict:
        # Pass the round number to clients so they can vary the DataLoader
        # shuffle seed each round — preventing repeated identical batch orderings.
        return {"server_round": server_round}

    common = dict(
        fraction_fit=participation_fraction,
        fraction_evaluate=participation_fraction if round_evaluation else 0.0,
        min_fit_clients=n_clients_per_round,
        min_evaluate_clients=n_clients_per_round if round_evaluation else 0,
        min_available_clients=n_clients_per_round,
        on_fit_config_fn=_on_fit_config,
        fit_metrics_aggregation_fn=_aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=_weighted_average,
    )

    if strategy_name == "behavior_cluster":
        return BehaviorClusterStrategy(
            n_clusters=n_clusters,
            recluster_every=10,
            # Primary protocol: no server jitter. Nonzero jitter is reserved
            # for explicit robustness diagnostics because sigma=0.1 in 128-D
            # can dominate a unit-norm centroid before re-normalization.
            centroid_noise_sigma=0.0,
            experiment_seed=experiment_seed,
            clustering_signal="behavior",
            **common,
        )
    elif strategy_name == "random_routing":
        # No-clustering control: cluster assignment is hash-based random at
        # first sight of each client and then frozen — captures the
        # "K parallel models, no behavioral routing" comparator. Per-cluster
        # aggregation infrastructure identical to the behavior strategy, so
        # the only difference is the assignment signal.
        return BehaviorClusterStrategy(
            n_clusters=n_clusters,
            recluster_every=10,
            centroid_noise_sigma=0.0,
            experiment_seed=experiment_seed,
            clustering_signal="random",
            **common,
        )
    elif strategy_name == "ifca":
        # IFCA-style clustered FL baseline (Ghosh et al. 2020): clients are
        # clustered by per-client validation loss, re-clustered every 10 rounds.
        return BehaviorClusterStrategy(
            n_clusters=n_clusters,
            recluster_every=10,
            centroid_noise_sigma=0.0,
            experiment_seed=experiment_seed,
            clustering_signal="loss",
            **common,
        )
    elif strategy_name == "task_aware_fedavg":
        return TaskAwareFedAvg(
            state_clients_per_round=method_cfg.get(
                "state_clients_per_round", 20
            ),
            experiment_seed=experiment_seed,
            **common,
        )
    elif strategy_name == "oracle_task_aware":
        return OracleTaskAwareFedAvg(
            state_clients_per_round=method_cfg.get(
                "state_clients_per_round", 20
            ),
            experiment_seed=experiment_seed,
            **common,
        )
    elif strategy_name == "hetero_conflict_averse":
        return ConflictAversePersonalizedFedAvg(**common)
    elif strategy_name in ("fedavg", "fedprox", "local"):
        return CheckpointingFedAvg(**common)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def _save_legacy_model_checkpoint(
    strategy: BehaviorClusterStrategy,
    method_cfg: dict,
    results_dir: Path,
    method_name: str,
    seed: int,
    num_poi: int,
    num_music: int,
    contrastive_weight: float,
    use_uncertainty: bool,
) -> None:
    """Reconstruct model from strategy's saved params and checkpoint to disk."""
    if strategy._last_global_params_np is None:
        print(f"  WARNING: No global params stored in strategy — skipping checkpoint.")
        return

    model_type = method_cfg.get("model_type", "behaviorfl")
    if model_type == "fedroute_fmtl":
        from models.fedroute_fmtl import FedRouteFMTL
        model = FedRouteFMTL(input_dim=15, embedding_dim=128,
                              num_poi_classes=num_poi, num_music_classes=num_music)
    else:
        model = BehaviorFLModel(
            input_dim=15, embedding_dim=128,
            num_poi_classes=num_poi, num_music_classes=num_music,
            use_contrastive=(contrastive_weight > 0),
            use_uncertainty_weighting=use_uncertainty,
        )

    # Load shared params (encoder + aligner) into the model
    shared = model.get_shared_parameters()
    if len(shared) == len(strategy._last_global_params_np):
        for p, arr in zip(shared, strategy._last_global_params_np):
            p.data = torch.tensor(arr, dtype=p.dtype)
    else:
        print(f"  WARNING: shared param count mismatch ({len(shared)} vs "
              f"{len(strategy._last_global_params_np)}) — shared params not loaded.")

    # Load global-average task heads into the global model so that model_state_dict
    # is not saved with random weights.  The global average is a sample-weighted mean
    # over all cluster-average task heads.
    if strategy._cluster_poi_head_np and strategy._cluster_music_head_np:
        # Weight each cluster's head by the number of clients assigned to it.
        from collections import Counter
        cluster_sizes = Counter(strategy._cluster_assignments.values())
        total_assigned = sum(cluster_sizes.values()) or 1

        poi_global = None
        music_global = None
        for clu, poi_flat in strategy._cluster_poi_head_np.items():
            music_flat = strategy._cluster_music_head_np.get(clu)
            if music_flat is None:
                continue
            w = cluster_sizes.get(clu, 0) / total_assigned
            poi_global = poi_flat * w if poi_global is None else poi_global + poi_flat * w
            music_global = music_flat * w if music_global is None else music_global + music_flat * w

        if poi_global is not None and hasattr(model, "poi_head"):
            offset = 0
            for p in model.poi_head.parameters():
                n = p.numel()
                p.data = torch.tensor(poi_global[offset:offset + n],
                                      dtype=p.dtype).reshape(p.shape)
                offset += n
        if music_global is not None and hasattr(model, "music_head"):
            offset = 0
            for p in model.music_head.parameters():
                n = p.numel()
                p.data = torch.tensor(music_global[offset:offset + n],
                                      dtype=p.dtype).reshape(p.shape)
                offset += n

    # Build per-cluster full model state dicts for ColdStartProtocol.
    # Each cluster model has: shared params (from cluster aggregation) +
    # task heads (averaged across cluster members from fit metrics).
    cluster_state_dicts = {}
    cluster_params_np = {
        k: parameters_to_ndarrays(v)
        for k, v in strategy._cluster_model_params.items()
    }
    for cluster_id, params_np in cluster_params_np.items():
        if model_type == "fedroute_fmtl":
            from models.fedroute_fmtl import FedRouteFMTL
            cmodel = FedRouteFMTL(input_dim=15, embedding_dim=128,
                                  num_poi_classes=num_poi, num_music_classes=num_music)
        else:
            cmodel = BehaviorFLModel(
                input_dim=15, embedding_dim=128,
                num_poi_classes=num_poi, num_music_classes=num_music,
                use_contrastive=(contrastive_weight > 0),
                use_uncertainty_weighting=use_uncertainty,
            )
        # Load cluster-specific shared params
        c_shared = cmodel.get_shared_parameters()
        if len(c_shared) == len(params_np):
            for p, arr in zip(c_shared, params_np):
                p.data = torch.tensor(arr, dtype=p.dtype)
        # Load cluster-average task heads (from client fit metrics)
        poi_flat = strategy._cluster_poi_head_np.get(cluster_id)
        music_flat = strategy._cluster_music_head_np.get(cluster_id)
        if poi_flat is not None:
            offset = 0
            for p in cmodel.poi_head.parameters():
                n = p.numel()
                p.data = torch.tensor(poi_flat[offset:offset + n],
                                      dtype=p.dtype).reshape(p.shape)
                offset += n
        if music_flat is not None:
            offset = 0
            for p in cmodel.music_head.parameters():
                n = p.numel()
                p.data = torch.tensor(music_flat[offset:offset + n],
                                      dtype=p.dtype).reshape(p.shape)
                offset += n
        cluster_state_dicts[cluster_id] = cmodel.state_dict()

    pt_path = results_dir / f"{method_name}_seed{seed}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "cluster_model_params_np": cluster_params_np,
        "cluster_state_dicts": cluster_state_dicts,
        "cluster_centroids": (strategy._cluster_centroids.tolist()
                              if strategy._cluster_centroids is not None else None),
        "method": method_name,
        "seed": seed,
        "num_poi_classes": num_poi,
        "num_music_classes": num_music,
        # Save model config so downstream tools can reconstruct the correct architecture.
        "use_contrastive": (contrastive_weight > 0),
        "use_uncertainty_weighting": use_uncertainty,
    }, pt_path)
    print(f"  Checkpoint saved: {pt_path}")


def _save_model_checkpoint(
    strategy: BehaviorClusterStrategy,
    method_cfg: dict,
    results_dir: Path,
    method_name: str,
    seed: int,
    input_dim: int,
    num_destination: int,
    num_state: int,
    use_uncertainty: bool,
) -> None:
    """Save the single global multi-task model used by the focused protocol."""
    weights = strategy._last_global_params_np
    if weights is None:
        print("  WARNING: no aggregated parameters; checkpoint skipped")
        return
    model = BehaviorFLModel(
        input_dim=input_dim,
        embedding_dim=128,
        num_destination_classes=num_destination,
        num_state_classes=num_state,
        use_contrastive=False,
        use_uncertainty_weighting=use_uncertainty,
        **_model_ablation_kwargs(method_cfg),
    )
    targets = list(model.get_shared_parameters())
    if method_cfg.get("federate_task_heads", False):
        targets += list(model.get_task_specific_parameters())
    if len(targets) != len(weights):
        raise ValueError(
            f"checkpoint parameter mismatch: model={len(targets)}, aggregate={len(weights)}"
        )
    for parameter, value in zip(targets, weights):
        parameter.data.copy_(torch.as_tensor(value, dtype=parameter.dtype))
    path = results_dir / f"{method_name}_seed{seed}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "method": method_name,
        "method_config": dict(method_cfg),
        "seed": seed,
        "input_dim": input_dim,
        "num_destination_classes": num_destination,
        "num_state_classes": num_state,
        "frameworks": {"model": "PyTorch", "federation": "Flower", "dp": "Opacus"},
    }, path)
    print(f"  Checkpoint saved: {path}")


def _pooled_heldout_evaluation(checkpoint_path: Path, dataset_path: str) -> dict:
    """Evaluate one global model on entirely unseen vehicles.

    Predictions are pooled before AUROC/AUPRC computation; client-level AUROCs
    are never averaged.  State is evaluated with its natural prefix sensors and
    with the entire OBD block removed from the same examples.
    """
    from sklearn.metrics import (
        average_precision_score, balanced_accuracy_score, brier_score_loss,
        f1_score, roc_auc_score,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    # Ablation knobs (fill_mode / geometry_*) don't change parameter shapes,
    # so state_dict loading always succeeds even with the wrong values here —
    # they must be read back from the checkpoint's method_config or eval
    # silently diverges from what was actually trained.
    _ckpt_cfg = checkpoint.get("method_config", {}) or {}
    _model_kwargs = {}
    if "fill_mode" in _ckpt_cfg:
        _model_kwargs["fill_mode"] = _ckpt_cfg["fill_mode"]
    if "geometry_prior_scale" in _ckpt_cfg:
        _model_kwargs["geometry_prior_scale"] = _ckpt_cfg["geometry_prior_scale"]
    if "geometry_residual_bound" in _ckpt_cfg:
        _model_kwargs["geometry_residual_bound"] = _ckpt_cfg["geometry_residual_bound"]
    model = BehaviorFLModel(
        input_dim=int(checkpoint["input_dim"]),
        num_destination_classes=int(checkpoint["num_destination_classes"]),
        num_state_classes=int(checkpoint["num_state_classes"]),
        use_uncertainty_weighting=False,
        **_model_kwargs,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    held_ids = data["held_out_client_ids"]
    splits = data["all_client_splits"]
    indices = sorted(i for cid in held_ids for i in splits[cid])
    idx = torch.tensor(indices, dtype=torch.long)
    features = data["trip_features"][idx].float()
    masks = data["tier_mask"][idx].float()
    destination_true = data["destination_labels"][idx].long()
    state_true = data["state_labels"][idx].long()
    state_observed = data["state_supervision_mask"][idx].bool()
    sources = [data["source_dataset"][i] for i in indices]

    destination_logits, state_logits, state_gps_logits = [], [], []
    with torch.no_grad():
        for start in range(0, len(features), 1024):
            x = features[start:start + 1024]
            mask = masks[start:start + 1024]
            dlog, slog = model(x, mask)
            x_gps, mask_gps = x.clone(), mask.clone()
            x_gps[:, 10:15] = 0.0
            mask_gps[:, 10:15] = 0.0
            _, slog_gps = model(x_gps, mask_gps)
            destination_logits.append(dlog)
            state_logits.append(slog)
            state_gps_logits.append(slog_gps)
    destination_logits = torch.cat(destination_logits)
    state_logits = torch.cat(state_logits)
    state_gps_logits = torch.cat(state_gps_logits)

    # Choose operating thresholds without touching held-out vehicles.  These
    # examples are the deterministic validation portions of training clients
    # and were excluded from local optimization by BehaviorDataLoader.
    calibration_loader = BehaviorDataLoader(dataset_path, batch_size=1024)
    calibration_indices = sorted({
        i
        for cid in calibration_loader.fl_client_ids
        for i in calibration_loader._client_val_indices[cid]
        if bool(calibration_loader.state_supervision_mask[i])
    })
    calibration_idx = torch.tensor(calibration_indices, dtype=torch.long)
    calibration_features = data["trip_features"][calibration_idx].float()
    calibration_masks = data["tier_mask"][calibration_idx].float()
    calibration_true = data["state_labels"][calibration_idx].long().cpu().numpy()
    calibration_state_logits, calibration_gps_logits = [], []
    with torch.no_grad():
        for start in range(0, len(calibration_features), 1024):
            x = calibration_features[start:start + 1024]
            mask = calibration_masks[start:start + 1024]
            _, slog = model(x, mask)
            x_gps, mask_gps = x.clone(), mask.clone()
            x_gps[:, 10:15] = 0.0
            mask_gps[:, 10:15] = 0.0
            _, slog_gps = model(x_gps, mask_gps)
            calibration_state_logits.append(slog)
            calibration_gps_logits.append(slog_gps)
    calibration_state_logits = torch.cat(calibration_state_logits)
    calibration_gps_logits = torch.cat(calibration_gps_logits)

    ranked = destination_logits.argsort(dim=1, descending=True)
    geometry_ranked = model._geometry_prior_logits(features).argsort(
        dim=1, descending=True
    )

    def destination_metrics(
        select: torch.Tensor, order: torch.Tensor = ranked
    ) -> dict:
        y = destination_true[select]
        selected_order = order[select]
        pred = selected_order[:, 0]
        if len(y) == 0:
            return {"n": 0}
        return {
            "n": int(len(y)),
            "accuracy": float((pred == y).float().mean()),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "hr_at_1": float((selected_order[:, :1] == y[:, None]).any(dim=1).float().mean()),
            "hr_at_3": float((selected_order[:, :3] == y[:, None]).any(dim=1).float().mean()),
            "hr_at_5": float((selected_order[:, :5] == y[:, None]).any(dim=1).float().mean()),
        }

    def state_metrics(
        logits: torch.Tensor, calibration_logits: torch.Tensor
    ) -> dict:
        y = state_true[state_observed].cpu().numpy()
        prob = torch.softmax(logits[state_observed], dim=1)[:, 1].cpu().numpy()
        calibration_prob = torch.softmax(
            calibration_logits, dim=1
        )[:, 1].cpu().numpy()
        thresholds = np.linspace(0.01, 0.99, 197)
        calibration_scores = np.asarray([
            f1_score(
                calibration_true,
                (calibration_prob >= threshold).astype(np.int64),
                average="macro",
                zero_division=0,
            )
            for threshold in thresholds
        ])
        threshold = float(thresholds[int(calibration_scores.argmax())])
        pred = (prob >= threshold).astype(np.int64)
        pred_at_half = (prob >= 0.5).astype(np.int64)
        return {
            "n": int(len(y)),
            "auroc": float(roc_auc_score(y, prob)),
            "auprc": float(average_precision_score(y, prob)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "brier": float(brier_score_loss(y, prob)),
            "decision_threshold": threshold,
            "threshold_selection": "training-client validation macro-F1",
            "calibration_n": int(len(calibration_true)),
            "macro_f1_at_0_5": float(
                f1_score(y, pred_at_half, average="macro", zero_division=0)
            ),
        }

    source_array = np.asarray(sources)
    by_source = {
        source: destination_metrics(torch.tensor(source_array == source))
        for source in sorted(set(sources))
    }
    geometry_by_source = {
        source: destination_metrics(
            torch.tensor(source_array == source), geometry_ranked
        )
        for source in sorted(set(sources))
    }
    all_heldout = torch.ones(len(indices), dtype=torch.bool)
    return {
        "scope": "pooled predictions on source-stratified held-out vehicles",
        "n_held_out_vehicles": int(len(held_ids)),
        "destination": destination_metrics(all_heldout),
        "destination_by_source": by_source,
        "destination_geometry_prior": destination_metrics(
            all_heldout, geometry_ranked
        ),
        "destination_geometry_prior_by_source": geometry_by_source,
        "state_natural_sensors": state_metrics(
            state_logits, calibration_state_logits
        ),
        "state_gps_only": state_metrics(
            state_gps_logits, calibration_gps_logits
        ),
    }


def _prepare_periodic_eval_data(dataset_path: str) -> dict:
    """Load held-out tensors once per experiment for cheap per-round eval."""
    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    held_ids = data["held_out_client_ids"]
    splits = data["all_client_splits"]
    indices = sorted(i for cid in held_ids for i in splits[cid])
    idx = torch.tensor(indices, dtype=torch.long)
    return {
        "features": data["trip_features"][idx].float(),
        "masks": data["tier_mask"][idx].float(),
        "destination_true": data["destination_labels"][idx].long(),
        "state_true": data["state_labels"][idx].long(),
        "state_observed": data["state_supervision_mask"][idx].bool(),
    }


def _periodic_pooled_eval(model, held: dict) -> dict:
    """Threshold-free pooled held-out metrics for a real per-round trajectory.

    Deliberately mirrors _pooled_heldout_evaluation's data (same held-out
    vehicles) but skips its training-client threshold calibration: AUPRC and
    argmax-based destination metrics need no threshold, keeping this cheap
    enough to run every few rounds without perturbing the final evaluation
    path used for Table 3.
    """
    from sklearn.metrics import average_precision_score, f1_score

    model.eval()
    features, masks = held["features"], held["masks"]
    dest_logits, state_logits, state_gps_logits = [], [], []
    with torch.no_grad():
        for start in range(0, len(features), 2048):
            x = features[start:start + 2048]
            m = masks[start:start + 2048]
            dlog, slog = model(x, m)
            x_gps, m_gps = x.clone(), m.clone()
            x_gps[:, 10:15] = 0.0
            m_gps[:, 10:15] = 0.0
            _, slog_gps = model(x_gps, m_gps)
            dest_logits.append(dlog)
            state_logits.append(slog)
            state_gps_logits.append(slog_gps)
    dest_logits = torch.cat(dest_logits)
    state_logits = torch.cat(state_logits)
    state_gps_logits = torch.cat(state_gps_logits)

    dest_true = held["destination_true"]
    order = dest_logits.argsort(dim=1, descending=True)
    pred = order[:, 0]
    macro_f1 = float(f1_score(dest_true, pred, average="macro", zero_division=0))
    hr5 = float((order[:, :5] == dest_true[:, None]).any(dim=1).float().mean())

    obs = held["state_observed"]
    y = held["state_true"][obs].cpu().numpy()
    out = {
        "periodic_destination_macro_f1": macro_f1,
        "periodic_destination_hr5": hr5,
    }
    if len(set(y.tolist())) > 1:
        prob_nat = torch.softmax(state_logits[obs], dim=1)[:, 1].cpu().numpy()
        prob_gps = torch.softmax(state_gps_logits[obs], dim=1)[:, 1].cpu().numpy()
        out["periodic_state_auprc_natural"] = float(average_precision_score(y, prob_nat))
        out["periodic_state_auprc_gps"] = float(average_precision_score(y, prob_gps))
    return out


def run_experiment(
    method_name: str,
    method_cfg: dict,
    loader: BehaviorDataLoader,
    mode_cfg: dict,
    seed: int,
    results_dir: Path,
    participation_fraction: float = 0.02,
    eval_every_n_rounds: int = 0,
) -> dict:
    """Run a single experiment (one method, one seed)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    method_cfg = dict(method_cfg)
    candidate_client_ids = (
        list(loader.fl_client_ids) if loader.fl_client_ids
        else list(loader.client_splits)
    )
    if method_cfg.get("state_only", False):
        candidate_client_ids = [
            client_id for client_id in candidate_client_ids
            if bool(loader.state_supervision_mask[
                torch.as_tensor(loader.client_splits[client_id], dtype=torch.long)
            ].any())
        ]
        if not candidate_client_ids:
            raise RuntimeError("state_specialist has no state-supervised training clients")
    n_fl_clients = len(candidate_client_ids)
    total_clients = mode_cfg["total_clients"] or n_fl_clients
    total_clients = min(total_clients, n_fl_clients)
    n_clients_per_round = min(
        int(method_cfg.get("min_clients_per_round", mode_cfg["n_clients_per_round"])),
        total_clients,
    )
    n_rounds = mode_cfg["n_rounds"]
    method_cfg["participation_fraction"] = participation_fraction
    local_epochs = method_cfg.get("local_epochs", 2)

    print()
    print(f"  Method: {method_name}  |  Seed: {seed}  |  Rounds: {n_rounds}  |  Participation: {participation_fraction:.2f}")
    print()

    strategy = make_strategy(method_cfg, n_clients_per_round, total_clients,
                             experiment_seed=seed)

    num_poi = loader.num_destination_classes
    num_music = loader.num_state_classes
    input_dim = loader.input_dim
    dataset_path_abs = loader.dataset_path

    if eval_every_n_rounds > 0:
        _held_eval_data = _prepare_periodic_eval_data(dataset_path_abs)
        _federate_heads = bool(method_cfg.get("federate_task_heads", False))
        _use_uncertainty_for_eval = method_cfg.get("use_uncertainty", True)

        def _centralized_eval_fn(arrays):
            m = BehaviorFLModel(
                input_dim=input_dim, embedding_dim=128,
                num_destination_classes=num_poi, num_state_classes=num_music,
                use_contrastive=False,
                use_uncertainty_weighting=_use_uncertainty_for_eval,
                **_model_ablation_kwargs(method_cfg),
            )
            targets = list(m.get_shared_parameters())
            if _federate_heads:
                targets += list(m.get_task_specific_parameters())
            if len(targets) != len(arrays):
                return {}
            for p, v in zip(targets, arrays):
                p.data.copy_(torch.as_tensor(v, dtype=p.dtype))
            return _periodic_pooled_eval(m, _held_eval_data)

        strategy.centralized_eval_fn = _centralized_eval_fn
        strategy.eval_every_n_rounds = int(eval_every_n_rounds)
    use_pcgrad = method_cfg.get("use_pcgrad", False)
    pcgrad_threshold = method_cfg.get("pcgrad_threshold", 0.0)
    contrastive_weight = method_cfg.get("contrastive_weight", 0.0)
    use_uncertainty = method_cfg.get("use_uncertainty", True)
    model_type = method_cfg.get("model_type", "behaviorfl")

    # Deterministic initial global parameters. Without initial_parameters,
    # Flower requests them from ONE RANDOMLY SAMPLED client, whose per-client
    # seed then determines the starting global model — a per-session lottery
    # that breaks run-to-run reproducibility.
    torch.manual_seed(seed)
    if model_type == "fedroute_fmtl":
        from models.fedroute_fmtl import FedRouteFMTL
        _init_model = FedRouteFMTL(input_dim=15, embedding_dim=128,
                                   num_poi_classes=num_poi,
                                   num_music_classes=num_music)
    else:
        _init_model = BehaviorFLModel(input_dim=input_dim, embedding_dim=128,
                                      num_destination_classes=num_poi,
                                      num_state_classes=num_music,
                                      use_contrastive=False,
                                      use_uncertainty_weighting=use_uncertainty,
                                      **_model_ablation_kwargs(method_cfg))
    if hasattr(_init_model, "get_shared_state_dict"):
        _init_params = [v.detach().cpu().numpy()
                        for v in _init_model.get_shared_state_dict().values()]
    else:
        _init_params = [p.detach().cpu().numpy()
                        for p in _init_model.get_shared_parameters()]
    if method_cfg.get("federate_task_heads", False):
        _init_params += [p.detach().cpu().numpy()
                         for p in _init_model.get_task_specific_parameters()]
    strategy.initial_parameters = fl.common.ndarrays_to_parameters(_init_params)
    del _init_model

    fl_ids = candidate_client_ids[:total_clients]
    # Guard: virtual client count must not exceed unique dataset client IDs.
    # If total_clients > len(fl_ids), the modulo mapping in client_fn creates
    # collisions where two virtual clients share the same dataset client, causing
    # data duplication and biased aggregation.
    assert total_clients <= len(fl_ids), (
        f"total_clients={total_clients} exceeds available fl_ids={len(fl_ids)}. "
        "Reduce --total-clients or increase the dataset client count."
    )

    # Capture seed in closure for use inside Ray workers
    _seed = seed

    def client_fn(context: Context):
        # Flower ≥1.8: client_fn receives a Context object.
        # context.node_id is a RANDOM 64-bit integer
        # regenerated every Ray session — using it for client mapping or seeding
        # makes runs irreproducible and (via modulo collisions) maps multiple
        # virtual clients onto the same dataset client while skipping others.
        # context.node_config["partition-id"] is the stable 0..N-1 index and
        # must be used instead.
        partition_id = context.node_config.get("partition-id")
        if partition_id is None:
            raise RuntimeError(
                "partition-id missing from node_config — refusing to fall back "
                "to the irreproducible context.node_id mapping."
            )
        idx = int(partition_id)

        # SEED RAY WORKER
        # torch.manual_seed() in the main process does NOT propagate to Ray
        # workers. Each worker must seed explicitly. Use seed + partition index
        # so different clients have different but reproducible initializations.
        import random as _random
        import sys
        import numpy as _np
        import torch as _torch
        _client_seed = (_seed * 10_007 + idx) % (2 ** 32)
        _random.seed(_client_seed)
        _np.random.seed(_client_seed)
        _torch.manual_seed(_client_seed)
        _torch.cuda.manual_seed_all(_client_seed)


        sys.path.insert(0, str(SRC_DIR))
        from data.behavior_loader import BehaviorDataLoader
        from flower.behavior_client import BehaviorFLClient

        client_id = fl_ids[idx % len(fl_ids)]

        if model_type == "fedroute_fmtl":
            from models.fedroute_fmtl import FedRouteFMTL
            assert not method_cfg.get("use_dp", False), (
                "FedRouteFMTL uses BatchNorm1d which is Opacus-incompatible. "
                "Do not set use_dp=True for fedroute_fmtl configs."
            )
            model = FedRouteFMTL(
                input_dim=15, embedding_dim=128,
                num_poi_classes=num_poi, num_music_classes=num_music,
            )
        else:
            from models.behaviorfl_model import BehaviorFLModel
            model = BehaviorFLModel(
                input_dim=input_dim, embedding_dim=128,
                num_destination_classes=num_poi, num_state_classes=num_music,
                use_contrastive=False,
                use_uncertainty_weighting=use_uncertainty,
                **_model_ablation_kwargs(method_cfg),
            )
        # Pass experiment seed so data shuffle order varies across seeds.
        # Without this, BehaviorDataLoader defaults to random_seed=42 for every run,
        # meaning multi-seed experiments only vary model initialization (not data order).
        dl = BehaviorDataLoader(dataset_path_abs, batch_size=32, random_seed=_seed)
        cfg = {
            "local_epochs": local_epochs,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "use_pcgrad": use_pcgrad,
            "pcgrad_threshold": pcgrad_threshold,
            "contrastive_weight": contrastive_weight,
            "fedprox_mu": method_cfg.get("fedprox_mu", 0.0),
            "local_only": method_cfg.get("local_only", False),
            "federate_task_heads": method_cfg.get("federate_task_heads", False),
            "poi_only": method_cfg.get("poi_only", False),
            "state_only": method_cfg.get("state_only", False),
            "emit_legacy_routing_metrics": False,
            "sensor_block_dropout_p": method_cfg.get("sensor_block_dropout_p", 0.0),
            "use_dp": method_cfg.get("use_dp", False),
            "dp_noise_multiplier": method_cfg.get("dp_noise_multiplier", 1.1),
            "dp_max_grad_norm": method_cfg.get("dp_max_grad_norm", 1.0),
        }
        client_obj = BehaviorFLClient(client_id, model, dl, cfg)
        # Ray reconstructs this client fresh on every fit()/evaluate() call;
        # context.state is the only channel that survives across calls for
        # this virtual client, used to persist local-only task heads/models.
        client_obj.context = context
        return client_obj.to_client()

    t0 = time.time()
    ray_cpus = mode_cfg.get("ray_cpus", 6)
    ray_obj_mb = mode_cfg.get("ray_obj_store_mb", 400)
    cpus_per_client = mode_cfg.get("cpus_per_client", 2)
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=total_clients,
        config=ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        client_resources={"num_cpus": cpus_per_client, "num_gpus": 0.0},
        ray_init_args={
            "num_cpus": ray_cpus,
            "num_gpus": 0,
            "ignore_reinit_error": True,
            "object_store_memory": ray_obj_mb * 1_000_000,
        },
    )
    elapsed = time.time() - t0

    participation_fairness = None
    selection_counts = getattr(strategy, "selection_counts", None)
    if selection_counts:
        eligible_mask = loader.state_supervision_mask
        equipped_counts, nonequipped_counts = [], []
        for cid in fl_ids:
            idx = torch.as_tensor(loader.client_splits[cid], dtype=torch.long)
            is_equipped = bool(eligible_mask[idx].any())
            count = int(selection_counts.get(float(cid), 0))
            (equipped_counts if is_equipped else nonequipped_counts).append(count)

        def _entropy_fraction_of_uniform(counts: list) -> "float | None":
            counts_arr = np.asarray(counts, dtype=np.float64)
            total = counts_arr.sum()
            if total <= 0 or len(counts_arr) < 2:
                return None
            p = counts_arr / total
            p_nonzero = p[p > 0]
            entropy = float(-(p_nonzero * np.log(p_nonzero)).sum())
            max_entropy = float(np.log(len(counts_arr)))
            return entropy / max_entropy if max_entropy > 0 else None

        participation_fairness = {
            "n_equipped_clients": len(equipped_counts),
            "n_nonequipped_clients": len(nonequipped_counts),
            "equipped_selection_mean": float(np.mean(equipped_counts)) if equipped_counts else None,
            "equipped_selection_std": float(np.std(equipped_counts, ddof=1)) if len(equipped_counts) > 1 else None,
            "nonequipped_selection_mean": float(np.mean(nonequipped_counts)) if nonequipped_counts else None,
            "nonequipped_selection_std": float(np.std(nonequipped_counts, ddof=1)) if len(nonequipped_counts) > 1 else None,
            "nonequipped_selection_p5": float(np.percentile(nonequipped_counts, 5)) if nonequipped_counts else None,
            "nonequipped_selection_entropy_frac_of_uniform_max": _entropy_fraction_of_uniform(nonequipped_counts),
        }

    # Save model checkpoint
    _save_model_checkpoint(
        strategy=strategy,
        method_cfg=method_cfg,
        results_dir=results_dir,
        method_name=method_name,
        seed=seed,
        input_dim=input_dim,
        num_destination=num_poi,
        num_state=num_music,
        use_uncertainty=use_uncertainty,
    )
    checkpoint_path = results_dir / f"{method_name}_seed{seed}.pt"
    heldout_evaluation = _pooled_heldout_evaluation(
        checkpoint_path, loader.dataset_path
    )

    # Extract metrics
    fit_metrics = history.metrics_distributed_fit
    eval_metrics = history.metrics_distributed
    centralized_metrics = history.metrics_centralized
    losses = history.losses_distributed or fit_metrics.get("train_loss", [])

    def _extract_series(metrics_dict: dict, key: str) -> list:
        return [float(v) for _, v in metrics_dict.get(key, [])]

    def _extract_round_series(metrics_dict: dict, key: str) -> list:
        """Same as _extract_series but keeps the round index (sparse series)."""
        return [[int(r), float(v)] for r, v in metrics_dict.get(key, [])]

    result = {
        "method": method_name,
        "seed": seed,
        "n_rounds": n_rounds,
        "elapsed_sec": round(elapsed, 1),
        "participation_fraction": participation_fraction,
        "dataset_metadata": dict(loader.metadata),
        "dataset_n_rows": int(len(loader.trip_features)),
        "dataset_n_fl_clients": int(len(loader.fl_client_ids)),
        "eligible_training_clients": int(n_fl_clients),
        "method_config": dict(method_cfg),
        "frameworks": {"model": "PyTorch", "federation": "Flower", "dp": "Opacus"},
        "heldout_evaluation": heldout_evaluation,
        "periodic_heldout_trajectory": {
            "destination_macro_f1": _extract_round_series(centralized_metrics, "periodic_destination_macro_f1"),
            "destination_hr5": _extract_round_series(centralized_metrics, "periodic_destination_hr5"),
            "state_auprc_natural": _extract_round_series(centralized_metrics, "periodic_state_auprc_natural"),
            "state_auprc_gps": _extract_round_series(centralized_metrics, "periodic_state_auprc_gps"),
        },
        "participation_fairness": participation_fairness,
        "centroid_noise_sigma": float(method_cfg.get("centroid_noise_sigma", 0.0)),
        "losses": [(r, round(float(l), 4)) for r, l in losses],
        "final_loss": float(losses[-1][1]) if losses else None,
        "n_clusters_active": _extract_series(fit_metrics, "n_clusters_active"),
        "known_state_clients": _extract_series(fit_metrics, "known_state_clients"),
        "reserved_state_clients": _extract_series(
            fit_metrics, "reserved_state_clients"
        ),
        "actual_state_clients": _extract_series(
            fit_metrics, "actual_state_clients"
        ),
        "state_supervised_examples": _extract_series(
            fit_metrics, "state_supervised_examples"
        ),
        "epsilon_max": _extract_series(fit_metrics, "epsilon_max"),
        "delta_max": _extract_series(fit_metrics, "delta_max"),
        "destination_hr5": _extract_series(eval_metrics, "destination_hr@5"),
        "destination_hr3": _extract_series(eval_metrics, "destination_hr@3"),
        "destination_hr1": _extract_series(eval_metrics, "destination_hr@1"),
        "destination_mrr": _extract_series(eval_metrics, "destination_mrr"),
        "destination_ndcg3": _extract_series(eval_metrics, "destination_ndcg@3"),
        "destination_ndcg5": _extract_series(eval_metrics, "destination_ndcg@5"),
        "destination_macro_f1": _extract_series(eval_metrics, "destination_macro_f1"),
        "destination_balanced_acc": _extract_series(eval_metrics, "destination_balanced_acc"),
        "state_macro_f1": _extract_series(eval_metrics, "state_macro_f1"),
        "state_balanced_acc": _extract_series(eval_metrics, "state_balanced_acc"),
        "state_accuracy": _extract_series(eval_metrics, "state_accuracy"),
        "state_n_samples": _extract_series(eval_metrics, "state_n_samples"),
    }
    for class_idx in range(num_poi):
        result[f"destination_recall_class_{class_idx}"] = _extract_series(
            eval_metrics, f"destination_recall_class_{class_idx}"
        )
    for class_idx in range(num_music):
        result[f"state_recall_class_{class_idx}"] = _extract_series(
            eval_metrics, f"state_recall_class_{class_idx}"
        )

    # Compute final-round averages for quick summary
    final_scalar_keys = (
        "destination_hr5", "destination_hr3", "destination_hr1",
        "destination_mrr", "destination_ndcg3", "destination_ndcg5",
        "destination_macro_f1", "destination_balanced_acc",
        "state_macro_f1", "state_balanced_acc", "state_accuracy", "state_n_samples",
    )
    final_scalar_keys += tuple(f"destination_recall_class_{c}" for c in range(num_poi))
    final_scalar_keys += tuple(f"state_recall_class_{c}" for c in range(num_music))
    for key in final_scalar_keys:
        vals = result[key]
        result[f"final_{key}"] = float(vals[-1]) if vals else None

    fname = results_dir / f"{method_name}_seed{seed}.json"
    tmp_fname = fname.with_suffix(fname.suffix + ".tmp")
    with open(tmp_fname, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp_fname.replace(fname)

    final_hr5 = result.get("final_destination_hr5")
    if final_hr5 is None:
        final_hr5 = heldout_evaluation.get("destination", {}).get("hr_at_5")
    hr5_str = f"  Destination HR@5: {final_hr5:.4f}" if final_hr5 is not None else ""
    loss_str = (
        f"{result['final_loss']:.4f}"
        if result["final_loss"] is not None else "n/a"
    )
    print(f"  Final loss: {loss_str}{hr5_str}  |  Time: {elapsed:.1f}s")
    return result


def _load_existing_result(
    results_dir: Path,
    method_name: str,
    seed: int,
    expected_rounds: int,
    participation_fraction: float,
    expected_dataset_metadata: dict,
    eval_every_n_rounds: int = 0,
) -> dict | None:
    path = results_dir / f"{method_name}_seed{seed}.json"
    if not path.exists():
        return None
    try:
        with path.open() as f:
            result = json.load(f)
    except Exception:
        return None
    if result.get("method") != method_name:
        return None
    if int(result.get("seed", -1)) != int(seed):
        return None
    if int(result.get("n_rounds", -1)) != int(expected_rounds):
        return None
    existing_participation = float(result.get("participation_fraction", 1.0))
    if abs(existing_participation - float(participation_fraction)) > 1e-9:
        return None
    # JSON round-tripping converts dataset tuples (region bounds) to lists.
    # Compare the serialized representations used in result artifacts.
    expected_dataset_metadata = json.loads(json.dumps(expected_dataset_metadata))
    existing_metadata = result.get("dataset_metadata", {})
    identity_keys = ("schema", "n_trips", "n_fl_clients", "sources")
    if any(
        existing_metadata.get(key) != expected_dataset_metadata.get(key)
        for key in identity_keys
    ):
        return None
    if (
        existing_metadata.get("destination_task", {}).get("bounds")
        != expected_dataset_metadata.get("destination_task", {}).get("bounds")
    ):
        return None
    if eval_every_n_rounds > 0:
        expected = {0, 1, *range(eval_every_n_rounds, expected_rounds + 1, eval_every_n_rounds)}
        trajectory = result.get("periodic_heldout_trajectory", {})
        key = "state_auprc_natural" if method_name == "state_specialist" else "destination_hr5"
        observed = {int(row[0]) for row in trajectory.get(key, [])}
        if observed != expected:
            return None
    return result


def main():
    parser = argparse.ArgumentParser(description="BehaviorFL Experiment Runner")
    parser.add_argument("--mode", choices=["smoke", "baseline", "full"], default="smoke")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--dataset", default=DATASET_PATH)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--participation-fraction", type=float, default=0.02,
                        help="Fraction of available clients sampled per round. "
                             "The focused protocol defaults to 2%% per round.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip method/seed pairs whose JSON artifact already matches "
                             "method, seed, n_rounds, and participation_fraction.")
    parser.add_argument(
        "--skip-round-evaluation",
        action="store_true",
        help="Skip redundant sampled-client evaluation after each round. Final pooled "
             "held-out evaluation is always performed and saved.",
    )
    parser.add_argument(
        "--eval-every-n-rounds", type=int, default=0,
        help="If >0, run centralized pooled held-out "
             "evaluation (threshold-free AUPRC + destination HR@5/macro-F1) "
             "every N rounds (and at round 1), saved as periodic_heldout_trajectory. "
             "Independent of --skip-round-evaluation. 0 disables it (default).",
    )
    args = parser.parse_args()
    unknown = set(args.methods or ()) - METHODS.keys()
    if unknown:
        parser.error(f"Unknown methods: {sorted(unknown)}")
    if args.eval_every_n_rounds < 0:
        parser.error("--eval-every-n-rounds must be nonnegative")

    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    mode_cfg = MODES[args.mode]
    seeds = args.seeds or mode_cfg["seeds"]
    methods_to_run = args.methods or list(METHODS.keys())

    print(f"=== BehaviorFL Experiment Runner ===")
    print(f"  Mode: {args.mode}  |  Methods: {len(methods_to_run)}  |  Seeds: {seeds}")
    print(f"  Rounds: {mode_cfg['n_rounds']}  |  Min clients/round: {mode_cfg['n_clients_per_round']}"
          f"  |  Participation fraction: {args.participation_fraction:.2f}")
    print("  All focused methods use one Flower FedAvg model with both heads federated.")

    loader = BehaviorDataLoader(args.dataset, batch_size=32)
    print(f"  Dataset: {loader.num_clients} FL clients + {len(loader.held_out_client_ids)} held-out")
    print(f"  Destination classes: {loader.num_destination_classes}  |  "
          f"Vehicle-state classes: {loader.num_state_classes}")

    all_results = {}
    for method_name in methods_to_run:
        if method_name not in METHODS:
            print(f"  Unknown method '{method_name}', skipping")
            continue
        method_cfg = dict(METHODS[method_name])
        method_cfg["round_evaluation"] = not args.skip_round_evaluation
        for seed in seeds:
            key = f"{method_name}_s{seed}"
            if args.skip_existing:
                existing = _load_existing_result(
                    results_dir, method_name, seed, mode_cfg["n_rounds"],
                    args.participation_fraction, loader.metadata, args.eval_every_n_rounds,
                )
                if existing is not None:
                    print(f"  [skip-existing] {method_name} seed {seed}")
                    all_results[key] = existing
                    continue
            result = run_experiment(
                method_name, method_cfg, loader, mode_cfg, seed, results_dir,
                participation_fraction=args.participation_fraction,
                eval_every_n_rounds=args.eval_every_n_rounds,
            )
            all_results[key] = result

    combined_path = results_dir / f"all_results_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    tmp_combined_path = combined_path.with_suffix(combined_path.suffix + ".tmp")
    with open(tmp_combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp_combined_path.replace(combined_path)

    print(f"\n{'='*60}")
    print(f"  Results saved to {combined_path}")
    print(f"{'='*60}")

    print("\nFinal destination HR@5 summary (↑ better):")
    summary: dict = {}
    for key, result in all_results.items():
        method = result["method"]
        summary.setdefault(method, [])
        v = result.get("final_destination_hr5")
        if v is None:
            v = result.get("heldout_evaluation", {}).get(
                "destination", {}
            ).get("hr_at_5")
        if v is not None:
            summary[method].append(v)

    for method, vals in sorted(summary.items()):
        mean_v = np.mean(vals) if vals else float("nan")
        print(f"  {method:<35} HR@5={mean_v:.4f}")


if __name__ == "__main__":
    main()
