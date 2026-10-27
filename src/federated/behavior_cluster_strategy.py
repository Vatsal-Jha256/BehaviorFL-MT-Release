"""Flower strategy with centroid clustering and per-cluster FedAvg.

Clients report centroids, and the server periodically reclusters them with
K-means and routes each client to its cluster model. Optional server-side
centroid noise is a perturbation diagnostic, not local differential privacy: the
server receives raw centroids. The primary multi-task runs use FedAvg strategies
in the experiment runner instead.
"""

import hashlib
import numpy as np
from logging import WARNING
from typing import Optional

import flwr as fl
from flwr.common import (
    FitIns, EvaluateIns, FitRes, Parameters, Scalar, NDArrays,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from sklearn.cluster import KMeans


class BehaviorClusterStrategy(FedAvg):
    """FedAvg with behavior-clustered aggregation.

    Clients include their behavior centroid (128-dim embedding) in fit() metrics.
    Server clusters clients by centroid similarity and aggregates within clusters.
    Each client receives the model for its own cluster (not a global average).

    Args:
        n_clusters: Number of behavior clusters (default 5)
        recluster_every: Re-run K-Means every N rounds (default 10)
        centroid_noise_sigma: Gaussian noise std on centroids before clustering
            (default 0.0; nonzero values are diagnostic only).
            Applied after the server receives centroids; provides no local privacy.
        centroid_key: Key in fit metrics dict where centroid is stored
        All other args passed to FedAvg.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        recluster_every: int = 10,
        centroid_noise_sigma: float = 0.0,
        centroid_key: str = "behavior_centroid",
        experiment_seed: int = 42,
        clustering_signal: str = "behavior",
        kmeans_n_init: int = 10,
        **fedavg_kwargs,
    ):
        super().__init__(**fedavg_kwargs)
        self.n_clusters = n_clusters
        self.recluster_every = recluster_every
        self.centroid_noise_sigma = centroid_noise_sigma
        self.centroid_key = centroid_key
        self.experiment_seed = experiment_seed
        self.kmeans_n_init = kmeans_n_init
        # "behavior" = K-Means on driving-behavior centroids (default, our method)
        # "random"   = hash-based random initial assignment, frozen thereafter —
        # IFCA-style baseline: same per-cluster aggregation
        # infrastructure but no behavioral signal informs clustering
        # "loss"     = K-Means on scalar client train_loss; captures loss-based
        # clustering used by CFL / IFCA-style methods
        assert clustering_signal in ("behavior", "random", "loss")
        self.clustering_signal = clustering_signal

        self._round = 0
        self._cluster_assignments: dict = {}
        self._cluster_centroids: Optional[np.ndarray] = None  # [K, D]

        # Rolling centroid cache: {cid: np.ndarray [D]} — updated every round a
        # client submits a centroid.  Used to re-assign ALL known clients when
        # K-Means is re-run, so cluster IDs remain semantically consistent across
        # re-clustering events.
        self._client_centroid_cache: dict = {}

        # Per-cluster model parameters (Flower Parameters objects).
        self._cluster_model_params: dict = {}

        # Last global params as numpy arrays — saved after each aggregate_fit
        # so run_experiment() can reconstruct and checkpoint the model.
        self._last_global_params_np: "list | None" = None

        # Per-cluster averaged task head flat arrays (for cold-start checkpoint).
        # Shape: {cluster_id: np.ndarray flat}  — populated each round from
        # client fit metrics (poi_head_bytes / music_head_bytes).
        self._cluster_poi_head_np: dict = {}
        self._cluster_music_head_np: dict = {}

        # Server-side cumulative privacy budget tracker.
        self._privacy_trackers: dict = {}
        self._privacy_tracker_initialized = False


    # configure_fit / configure_evaluate — route cluster-specific params


    def configure_fit(self, server_round, parameters, client_manager):
        """Send each client the model for its assigned cluster.

        Clients not yet assigned (e.g. round 1) receive the global fallback
        (the `parameters` arg from the previous aggregate_fit call).
        """
        # Use FedAvg's sampling logic, then swap in per-cluster parameters.
        base = super().configure_fit(server_round, parameters, client_manager)
        result = []
        for client, fit_ins in base:
            cid = client.cid
            cluster_id = self._cluster_assignments.get(cid)
            if cluster_id is not None and cluster_id in self._cluster_model_params:
                client_params = self._cluster_model_params[cluster_id]
            else:
                client_params = parameters  # global fallback for unassigned clients
            result.append((client, FitIns(client_params, fit_ins.config)))
        return result

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Send each client the model for its assigned cluster for evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        base = super().configure_evaluate(server_round, parameters, client_manager)
        result = []
        for client, eval_ins in base:
            cid = client.cid
            cluster_id = self._cluster_assignments.get(cid)
            if cluster_id is not None and cluster_id in self._cluster_model_params:
                client_params = self._cluster_model_params[cluster_id]
            else:
                client_params = parameters
            result.append((client, EvaluateIns(client_params, eval_ins.config)))
        return result


    # aggregate_fit — per-cluster FedAvg, store cluster models


    def aggregate_evaluate(
        self,
        server_round: int,
        results: list,
        failures: list,
    ) -> tuple:
        # Sort by the client-reported stable client_id before delegating, so
        # the weighted loss/metric sums are order-independent across runs
        # (float summation is non-associative; Ray arrival order varies).
        results = sorted(
            results,
            key=lambda r: float(r[1].metrics.get("client_id", -1.0)),
        )
        return super().aggregate_evaluate(server_round, results, failures)

    def aggregate_fit(
        self,
        server_round: int,
        results: list,
        failures: list,
    ) -> tuple:
        """Override FedAvg.aggregate_fit to cluster clients and aggregate within clusters."""
        self._round = server_round

        # Deterministic aggregation order: Ray delivers results in scheduling
        # order, which varies across runs; float summation is non-associative,
        # so unsorted aggregation breaks bit-level reproducibility. Sort by the
        # client-reported dataset client_id (stable across sessions, unlike
        # the per-session random node_id in client_proxy.cid).
        results = sorted(
            results,
            key=lambda r: float(r[1].metrics.get("client_id", -1.0)),
        )

        # Extract behavior centroids, task head params, and DP metadata from metrics
        centroids = {}
        client_epsilons = []
        # Collect task head bytes per client — averaged per cluster for cold-start checkpoints
        client_task_heads: dict = {}  # {cid: {"poi": np.ndarray, "music": np.ndarray}}

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            poi_bytes = fit_res.metrics.get("poi_head_bytes")
            music_bytes = fit_res.metrics.get("music_head_bytes")
            if poi_bytes is not None and music_bytes is not None:
                client_task_heads[cid] = {
                    "poi": np.frombuffer(poi_bytes, dtype=np.float32).copy(),
                    "music": np.frombuffer(music_bytes, dtype=np.float32).copy(),
                }
            if "epsilon" in fit_res.metrics:
                client_epsilons.append(float(fit_res.metrics["epsilon"]))
            if "dp_n_steps" in fit_res.metrics:
                self._update_privacy_tracker(
                    client_id=cid,
                    noise_multiplier=float(fit_res.metrics.get("dp_noise_multiplier", 1.1)),
                    sample_rate=float(fit_res.metrics.get("dp_sample_rate", 0.01)),
                    steps=int(fit_res.metrics["dp_n_steps"]),
                )
            if self.centroid_key in fit_res.metrics:
                raw = fit_res.metrics[self.centroid_key]
                centroid = (np.frombuffer(raw, dtype=np.float32)
                            if isinstance(raw, bytes)
                            else np.array(raw, dtype=np.float32))
                if self.clustering_signal == "loss":
                    # Replace the behavior embedding with the client's train_loss
                    # (broadcast to D dims so KMeans + cosine routing still work).
                    train_loss = float(fit_res.metrics.get("train_loss", 0.0))
                    centroid = np.full(len(centroid), train_loss, dtype=np.float32)
                centroids[cid] = centroid
                # Always refresh the rolling cache so re-clustering uses
                # the most recent embedding for every client.
                self._client_centroid_cache[cid] = centroid.copy()

        # No-clustering control: hash each client into a cluster at first
        # sight, never recluster. Same per-cluster aggregation as behavior
        # clustering but no behavioral or loss signal — isolates whether ANY
        # clustering helps versus whether the BEHAVIOR signal in particular helps.

        # this must NOT depend on (a) client_proxy.cid
        # (a per-simulation-session random node_id — different every run) or
        # (b) Python's built-in hash() on a string (salted by PYTHONHASHSEED,
        # randomized per-process unless explicitly disabled). Both would make
        # this "random" assignment re-randomize on every invocation even at a
        # fixed experiment_seed. We instead hash the STABLE dataset client_id
        # (reported by the client in fit metrics) through hashlib, which is
        # unsalted and deterministic across processes and Python versions.
        if self.clustering_signal == "random":
            for client_proxy, fit_res in results:
                cid = client_proxy.cid
                if cid not in self._cluster_assignments:
                    stable_id = fit_res.metrics.get("client_id")
                    if stable_id is None:
                        raise RuntimeError(
                            "client_id missing from fit metrics — cannot "
                            "deterministically assign the no-clustering control."
                        )
                    key = f"{int(stable_id)}:{self.experiment_seed}".encode()
                    digest = hashlib.sha256(key).digest()
                    cluster_id = int.from_bytes(digest[:8], "big") % self.n_clusters
                    self._cluster_assignments[cid] = cluster_id
        # Re-cluster (K-Means on behavior centroids or scalar loss) if needed.
        # pass the full cache (not just current-round centroids) so
        # K-Means is fitted on all known clients and ALL assignments are updated
        # atomically.  Without this, stale clients keep assignments that reference
        # the OLD cluster-ID space after K-Means is re-run, routing them to the
        # wrong cluster model.
        elif centroids and (
            server_round % self.recluster_every == 1
            or not self._cluster_assignments
        ):
            all_known_centroids = {**self._client_centroid_cache}
            self._update_clusters(all_known_centroids)

        # Group results by cluster
        cluster_results: dict = {}
        n_unassigned = 0

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            if cid in self._cluster_assignments:
                cluster_id = self._cluster_assignments[cid]
            elif self._cluster_centroids is not None and cid in self._client_centroid_cache:
                # Client submitted a centroid but _update_clusters wasn't triggered
                # this round (recluster_every > 1).  Assign via nearest centroid so
                # first-ever clients are not left unassigned until next re-clustering.
                emb = self._client_centroid_cache[cid]
                sims = self._cluster_centroids @ emb  # [K]
                cluster_id = int(np.argmax(sims))
                self._cluster_assignments[cid] = cluster_id
            else:
                # No centroid available (round 1 serialization failure).
                # Do NOT silently dump into cluster 0 — use a dedicated "unassigned" bucket.
                cluster_id = -1
                n_unassigned += 1

            cluster_results.setdefault(cluster_id, []).append((client_proxy, fit_res))

        if n_unassigned > 0:
            print(f"[BehaviorClusterStrategy] WARNING: Round {server_round}: "
                  f"{n_unassigned} client(s) had no centroid — aggregated as global fallback")

        # Aggregate within each cluster separately
        aggregated_weights_per_cluster = {}
        for cluster_id, cluster_res in cluster_results.items():
            if not cluster_res:
                continue
            agg_params, agg_metrics = super().aggregate_fit(
                server_round, cluster_res, []
            )
            if agg_params is not None:
                aggregated_weights_per_cluster[cluster_id] = (
                    parameters_to_ndarrays(agg_params),
                    agg_metrics,
                )

        if not aggregated_weights_per_cluster:
            return None, {}

        # Store per-cluster params — configure_fit will route these to clients.
        # Cluster -1 (unassigned) contributes to the global fallback only.
        self._cluster_model_params = {}
        for cluster_id, (weights, _) in aggregated_weights_per_cluster.items():
            if cluster_id >= 0:
                self._cluster_model_params[cluster_id] = ndarrays_to_parameters(weights)

        # Weighted average task head parameters per cluster (for cold-start checkpoint).
        # Weighted by client sample count (fit_res.num_examples) — consistent with FedAvg.
        # New drivers get cluster-average task heads until they fine-tune locally.
        cluster_poi_buckets: dict = {}    # {clu: [(arr, weight)]}
        cluster_music_buckets: dict = {}  # {clu: [(arr, weight)]}
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            if cid not in client_task_heads:
                continue
            clu = self._cluster_assignments.get(cid)
            if clu is None or clu < 0:
                continue
            n_samples = max(fit_res.num_examples, 1)  # avoid zero-weight
            cluster_poi_buckets.setdefault(clu, []).append(
                (client_task_heads[cid]["poi"], n_samples)
            )
            cluster_music_buckets.setdefault(clu, []).append(
                (client_task_heads[cid]["music"], n_samples)
            )

        def _weighted_mean(arr_weight_list):
            arrs, weights = zip(*arr_weight_list)
            w = np.array(weights, dtype=np.float64)
            w /= w.sum()
            return sum(a * wi for a, wi in zip(arrs, w))

        self._cluster_poi_head_np = {
            clu: _weighted_mean(items)
            for clu, items in cluster_poi_buckets.items()
        }
        self._cluster_music_head_np = {
            clu: _weighted_mean(items)
            for clu, items in cluster_music_buckets.items()
        }

        # Compute global fallback: sample-weighted average across ALL clusters
        # (including -1 unassigned bucket).  This is used as (a) the server's
        # internal "global parameters" state, and (b) fallback params for clients
        # not yet assigned to any cluster.
        # Weight by total num_examples in each cluster — consistent with FedAvg's
        # sample-weighted aggregation (McMahan et al., 2017).  Weighting by client
        # count would bias toward clusters containing many small-dataset clients.
        cluster_total_samples: dict = {}
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            clu = self._cluster_assignments.get(cid, -1)
            cluster_total_samples[clu] = (
                cluster_total_samples.get(clu, 0) + max(fit_res.num_examples, 1)
            )
        total_samples = sum(cluster_total_samples.values())
        if total_samples == 0:
            return None, {}

        global_weights = None
        for cluster_id, (weights, _) in aggregated_weights_per_cluster.items():
            weight = cluster_total_samples.get(cluster_id, 0) / total_samples
            if weight == 0:
                continue
            if global_weights is None:
                global_weights = [w * weight for w in weights]
            else:
                global_weights = [gw + w * weight for gw, w in zip(global_weights, weights)]

        # Aggregate metrics across all clusters
        all_metrics = {}
        for cluster_id, (_, metrics) in aggregated_weights_per_cluster.items():
            for k, v in (metrics or {}).items():
                all_metrics.setdefault(k, []).append(v)
        avg_metrics = {
            k: float(np.mean(v)) for k, v in all_metrics.items()
            if v and isinstance(v[0], (int, float))
        }
        avg_metrics["n_clusters_active"] = float(len(self._cluster_model_params))
        if client_epsilons:
            avg_metrics["epsilon"] = float(max(client_epsilons))

        # Store last global params as numpy arrays for external checkpointing
        # (run_experiment() can reconstruct and save the model after simulation).
        self._last_global_params_np = global_weights

        # Return global weights as server state / fallback.
        # Per-cluster routing is handled in configure_fit / configure_evaluate.
        return ndarrays_to_parameters(global_weights), avg_metrics


    # Clustering helpers


    def _update_clusters(self, centroids: dict) -> None:
        """Run K-Means on noisy client centroids and re-assign ALL known clients.

        Args:
            centroids: dict {cid: np.ndarray [D]} — should be the full
                _client_centroid_cache so that K-Means is fitted on all known
                clients.  Callers MUST pass the cache, not just current-round
                centroids, to avoid the cluster-semantic-shift bug where stale
                assignments reference cluster IDs from a previous K-Means run.
        """
        cids = list(centroids.keys())
        centroid_array = np.stack([centroids[c] for c in cids])  # [N, D]

        # Server-side Gaussian noise on centroids before K-Means.
        # this is NOT LDP — the server receives raw centroids first.
        # The noise improves K-Means robustness to outliers; it does not provide
        # a privacy guarantee against a curious server.
        if self.centroid_noise_sigma > 0:
            rng_noise = np.random.default_rng(self.experiment_seed + self._round)
            noise = rng_noise.normal(0, self.centroid_noise_sigma,
                                     centroid_array.shape).astype(np.float32)
            noisy_centroids = centroid_array + noise
        else:
            noisy_centroids = centroid_array.copy()

        # Normalize (centroids are already L2-normalized from encoder)
        norms = np.linalg.norm(noisy_centroids, axis=1, keepdims=True)
        noisy_centroids /= (norms + 1e-8)

        # K-Means — seed is experiment_seed + round so that each re-clustering
        # event gets a different initialisation, preventing spurious cluster
        # stability when centroid_noise_sigma=0.
        k = min(self.n_clusters, len(cids))
        kmeans = KMeans(n_clusters=k, random_state=self.experiment_seed + self._round,
                        n_init=self.kmeans_n_init, init="k-means++")
        kmeans.fit(noisy_centroids)

        # L2-normalize cluster centers BEFORE re-assigning all clients so that
        # both K-Means-assigned and centroid-projected assignments use the same
        # normalized space.
        centers = kmeans.cluster_centers_.copy()
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        self._cluster_centroids = centers / (norms + 1e-8)  # [K, D]

        # Re-assign ALL known clients atomically using the new cluster centers.
        # This prevents the semantic-shift bug: after K-Means is re-run, old
        # integer IDs (0,1,2,...) refer to NEW centroids.  Keeping stale
        # assignments would route clients to the wrong cluster model.
        new_assignments = {}
        for cid, emb in centroids.items():
            sims = self._cluster_centroids @ emb  # [K] cosine similarities
            new_assignments[cid] = int(np.argmax(sims))
        self._cluster_assignments = new_assignments

        # Log distribution
        cluster_dist: dict = {}
        for c in new_assignments.values():
            cluster_dist[c] = cluster_dist.get(c, 0) + 1
        print(f"[BehaviorClusterStrategy] Round {self._round}: "
              f"Re-assigned {len(new_assignments)} clients into {k} clusters: {cluster_dist}")

    def get_cluster_assignment(self, client_cid: str) -> int:
        """Get cluster ID for a client (returns -1 if unassigned)."""
        return self._cluster_assignments.get(client_cid, -1)

    def _update_privacy_tracker(
        self,
        client_id: str,
        noise_multiplier: float,
        sample_rate: float,
        steps: int,
    ) -> None:
        """Update a per-client cumulative privacy tracker."""
        if not self._privacy_tracker_initialized:
            try:
                from federated.privacy import PrivacyBudgetTracker
                self._privacy_tracker_cls = PrivacyBudgetTracker
                self._privacy_tracker_initialized = True
            except Exception as exc:
                print(f"[BehaviorClusterStrategy] PrivacyBudgetTracker init failed: {exc}. "
                      "Cumulative epsilon will not be tracked.")
                self._privacy_tracker_initialized = True  # don't retry
                return

        tracker = self._privacy_trackers.get(client_id)
        if tracker is None:
            try:
                # initial_epsilon=float("inf"): the server tracker is for
                # monitoring cumulative spend only — it does not enforce a budget.
                # Using inf prevents get_remaining_budget() from misleadingly
                # reporting budget exhaustion before actual training completes.
                tracker = self._privacy_tracker_cls(initial_epsilon=float("inf"), delta=1e-5)
            except Exception as exc:
                print(f"[BehaviorClusterStrategy] PrivacyBudgetTracker creation failed: {exc}")
                return
            self._privacy_trackers[client_id] = tracker

        try:
            tracker.update(
                noise_multiplier=noise_multiplier,
                sample_rate=sample_rate,
                steps=steps,
            )
        except Exception as exc:
            print(f"[BehaviorClusterStrategy] PrivacyBudgetTracker.update failed: {exc}")

    def get_cumulative_epsilon(self, delta: float = 1e-5) -> float:
        """Return accumulated (ε, δ)-DP guarantee across all FL rounds.

        Returns 0.0 when DP is not in use or tracker failed to initialize.
        """
        if not self._privacy_trackers:
            return 0.0
        try:
            return float(max(
                tracker.get_total_consumed() for tracker in self._privacy_trackers.values()
            ))
        except Exception:
            return 0.0

    def get_cluster_info(self) -> dict:
        """Return current cluster state."""
        return {
            "assignments": dict(self._cluster_assignments),
            "n_clusters_active": len(self._cluster_model_params),
            "cluster_centroids": (
                self._cluster_centroids.tolist()
                if self._cluster_centroids is not None else None
            ),
        }
