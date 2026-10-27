"""
Evaluation metrics for BehaviorFL experiments.

Provides:
  - compute_hr_ndcg: HR@k, NDCG@k, and MRR for ranked category lists
  - compute_transfer_lift: cross-domain transfer improvement
  - compute_cluster_metrics: silhouette score and NMI for behavior clusters
  - aggregate_round_metrics: aggregate per-client metrics across a FL round
"""

import numpy as np


def compute_hr_ndcg(
    ranked_lists: "list[list[int]]",
    true_labels: "list[int]",
    k_values: "list[int] | None" = None,
) -> dict:
    """Compute HR@k, NDCG@k, and MRR for a batch of ranked category lists.

    Args:
        ranked_lists: List of N ranked lists, each of length >= max(k_values)
        true_labels: List of N ground-truth item IDs
        k_values: List of k cutoffs (default [5, 10])

    Returns:
        dict with keys like 'hr@5', 'ndcg@5', and 'mrr'
    """
    if k_values is None:
        k_values = [5, 10]
    if len(ranked_lists) != len(true_labels):
        raise ValueError(
            f"ranked_lists and true_labels must have equal length; "
            f"got {len(ranked_lists)} and {len(true_labels)}"
        )
    if not k_values or any(int(k) <= 0 for k in k_values):
        raise ValueError(f"k_values must contain positive cutoffs, got {k_values!r}")

    results = {}
    n = len(true_labels)
    if n == 0:
        return (
            {f"hr@{k}": 0.0 for k in k_values}
            | {f"ndcg@{k}": 0.0 for k in k_values}
            | {"mrr": 0.0}
        )

    reciprocal_ranks = []
    for ranked, true in zip(ranked_lists, true_labels):
        try:
            reciprocal_ranks.append(1.0 / float(ranked.index(true) + 1))
        except ValueError:
            reciprocal_ranks.append(0.0)
    results["mrr"] = float(np.mean(reciprocal_ranks))

    for k in k_values:
        hits = []
        ndcgs = []
        for ranked, true in zip(ranked_lists, true_labels):
            top_k = ranked[:k]
            hit = 1.0 if true in top_k else 0.0
            hits.append(hit)
            if true in top_k:
                rank = top_k.index(true) + 1  # 1-based
                ndcgs.append(1.0 / np.log2(rank + 1))
            else:
                ndcgs.append(0.0)
        results[f"hr@{k}"] = float(np.mean(hits))
        results[f"ndcg@{k}"] = float(np.mean(ndcgs))

    return results


def compute_classification_metrics(
    preds: "list[int]",
    labels: "list[int]",
    num_classes: int,
) -> dict:
    """Top-1 accuracy, balanced accuracy, macro-F1, and per-class recall.

    Classes absent from the evaluation labels are omitted from this client's
    own macro-F1/balanced-accuracy denominators; their recall fields are
    reported as 0.0. This per-client balanced_acc/macro_f1 is a local
    diagnostic only. Cross-client aggregation must not weighted-average these
    two fields directly (see `aggregate_client_eval_metrics`): a client with
    a tiny, single-class test set trivially reports balanced_acc=1.0 for that
    one class, which is not comparable to a client with broader class
    coverage. The raw `tp_class_c`/`fp_class_c`/`support_class_c` counts
    returned here let the aggregator pool counts across clients before
    dividing, producing a single dataset-level balanced accuracy consistent
    with the per-class recall table.
    """
    if not labels:
        empty = {"top1": 0.0, "balanced_acc": 0.0, "macro_f1": 0.0}
        empty.update({f"recall_class_{c}": 0.0 for c in range(num_classes)})
        empty.update({f"tp_class_{c}": 0.0 for c in range(num_classes)})
        empty.update({f"fp_class_{c}": 0.0 for c in range(num_classes)})
        empty.update({f"support_class_{c}": 0.0 for c in range(num_classes)})
        return empty

    preds_arr = np.asarray(preds, dtype=np.int64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    top1 = float((preds_arr == labels_arr).mean())

    per_class_recall = []
    per_class_f1 = []
    counts_by_class = {}
    for c in range(num_classes):
        gt_mask = labels_arr == c
        n_gt = int(gt_mask.sum())
        tp = int(((preds_arr == c) & gt_mask).sum())
        fp = int(((preds_arr == c) & ~gt_mask).sum())
        counts_by_class[f"tp_class_{c}"] = float(tp)
        counts_by_class[f"fp_class_{c}"] = float(fp)
        counts_by_class[f"support_class_{c}"] = float(n_gt)
        if n_gt == 0:
            continue
        recall = tp / max(n_gt, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    balanced_acc = float(np.mean(per_class_recall)) if per_class_recall else 0.0
    macro_f1 = float(np.mean(per_class_f1)) if per_class_f1 else 0.0
    recalls_by_class = {}
    for c in range(num_classes):
        n_gt = counts_by_class[f"support_class_{c}"]
        recalls_by_class[f"recall_class_{c}"] = (
            0.0 if n_gt == 0 else counts_by_class[f"tp_class_{c}"] / n_gt
        )

    return {
        "top1": top1,
        "balanced_acc": balanced_acc,
        "macro_f1": macro_f1,
        **recalls_by_class,
        **counts_by_class,
    }


_COUNT_KIND_PREFIXES = ("tp_class_", "fp_class_", "support_class_")
_TASK_PREFIXES = ("poi_", "music_", "destination_", "state_")


def aggregate_client_eval_metrics(metrics: "list[tuple[int, dict]]", num_classes: int = 10) -> dict:
    """Aggregate per-client Flower evaluate() metrics into one round-level dict.

    Per-sample rate metrics (HR@k, NDCG@k, MRR, loss, top-1/accuracy,
    n_clusters_active) are pooled by a sample-count-weighted average, which
    is the correct estimate for a per-sample metric and is equivalent to the
    pooled value over every held-out sample.

    Per-class recall, macro-F1, and balanced accuracy cannot be pooled the
    same way: recall for class c must be weighted by class c's own support,
    not by each client's total sample count. This first sums the raw
    tp/fp/support counts per class across clients, then derives recall/F1/
    balanced accuracy from the pooled counts, so the resulting balanced
    accuracy is consistent with the resulting per-class recall table.
    """
    totals: dict = {}
    weights: dict = {}
    count_sums: dict = {}

    for num_examples, m in metrics:
        for k, v in m.items():
            if not isinstance(v, (int, float)):
                continue
            if k in ("n_samples", "music_n_samples", "state_n_samples") or any(
                k.startswith(f"{p}{kind}") for p in _TASK_PREFIXES for kind in _COUNT_KIND_PREFIXES
            ):
                count_sums[k] = count_sums.get(k, 0.0) + float(v)
                continue
            metric_weight = num_examples
            if k.startswith("music_") and k != "music_n_samples":
                metric_weight = int(m.get("music_n_samples", 0))
            if k.startswith("state_") and k != "state_n_samples":
                metric_weight = int(m.get("state_n_samples", 0))
            totals[k] = totals.get(k, 0.0) + float(v) * metric_weight
            weights[k] = weights.get(k, 0) + metric_weight

    agg = {k: totals[k] / weights[k] for k in totals if weights[k] > 0}
    agg.update(count_sums)

    for prefix in _TASK_PREFIXES:
        observed_class_ids = []
        marker = f"{prefix}support_class_"
        for key in count_sums:
            if key.startswith(marker):
                try:
                    observed_class_ids.append(int(key[len(marker):]))
                except ValueError:
                    pass
        if not observed_class_ids:
            continue
        prefix_num_classes = max(observed_class_ids) + 1
        recalls, f1s = [], []
        for c in range(prefix_num_classes):
            tp = count_sums.get(f"{prefix}tp_class_{c}", 0.0)
            fp = count_sums.get(f"{prefix}fp_class_{c}", 0.0)
            support = count_sums.get(f"{prefix}support_class_{c}", 0.0)
            if support > 0:
                recall_c = tp / support
                precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                f1_c = (
                    0.0
                    if (precision_c + recall_c) == 0
                    else 2 * precision_c * recall_c / (precision_c + recall_c)
                )
                recalls.append(recall_c)
                f1s.append(f1_c)
            else:
                recall_c = 0.0
            agg[f"{prefix}recall_class_{c}"] = recall_c
        if any(
            f"{prefix}support_class_{c}" in count_sums
            for c in range(prefix_num_classes)
        ):
            agg[f"{prefix}balanced_acc"] = float(np.mean(recalls)) if recalls else 0.0
            agg[f"{prefix}macro_f1"] = float(np.mean(f1s)) if f1s else 0.0

    return agg


def compute_transfer_lift(
    single_domain_metrics: dict,
    cross_domain_metrics: dict,
    metric_key: str = "hr@10",
) -> dict:
    """Compute cross-domain transfer lift.

    Lift = (cross_domain - single_domain) / single_domain * 100%

    Args:
        single_domain_metrics: metrics when trained on one domain only
        cross_domain_metrics: metrics when trained with cross-domain alignment
        metric_key: which metric to compare (default 'hr@10')

    Returns:
        dict with lift values for each available metric
    """
    lifts = {}
    for key in single_domain_metrics:
        base = single_domain_metrics[key]
        cross = cross_domain_metrics.get(key, base)
        if base > 0:
            lifts[f"lift_{key}"] = float((cross - base) / base * 100.0)
        else:
            lifts[f"lift_{key}"] = 0.0
    return lifts


def compute_cluster_metrics(
    embeddings: np.ndarray,
    cluster_labels: "list[int] | np.ndarray",
) -> dict:
    """Compute clustering quality metrics for behavior embeddings.

    Args:
        embeddings: [N, D] behavior embeddings
        cluster_labels: [N] cluster assignments

    Returns:
        dict with 'silhouette_score', 'n_clusters', 'cluster_sizes'
    """
    from sklearn.metrics import silhouette_score, normalized_mutual_info_score

    labels = np.array(cluster_labels)
    n_clusters = len(np.unique(labels))

    result = {
        "n_clusters": int(n_clusters),
        "cluster_sizes": {
            int(c): int((labels == c).sum()) for c in np.unique(labels)
        },
    }

    if n_clusters >= 2 and len(embeddings) > n_clusters:
        try:
            result["silhouette_score"] = float(
                silhouette_score(embeddings, labels, metric="cosine")
            )
        except Exception:
            result["silhouette_score"] = 0.0
    else:
        result["silhouette_score"] = 0.0

    return result


def compute_cluster_nmi(
    true_labels: "list[int] | np.ndarray",
    pred_labels: "list[int] | np.ndarray",
) -> float:
    """Normalized Mutual Information between true and predicted cluster labels."""
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(true_labels, pred_labels))


def aggregate_round_metrics(
    per_client_metrics: "list[dict]",
    weights: "list[float] | None" = None,
) -> dict:
    """Weighted average of per-client metric dicts.

    Args:
        per_client_metrics: list of metric dicts from each client
        weights: per-client weights (e.g., n_samples); uniform if None

    Returns:
        aggregated metric dict
    """
    if not per_client_metrics:
        return {}

    if weights is None:
        weights = [1.0] * len(per_client_metrics)

    total_weight = sum(weights)
    if total_weight == 0:
        return {}

    agg: dict = {}
    all_keys = set()
    for m in per_client_metrics:
        all_keys.update(m.keys())

    for key in all_keys:
        vals = []
        wts = []
        for m, w in zip(per_client_metrics, weights):
            if key in m and isinstance(m[key], (int, float)):
                vals.append(float(m[key]))
                wts.append(w)
        if vals:
            agg[key] = float(np.average(vals, weights=wts))

    return agg


def format_metrics_table(results: dict, title: str = "") -> str:
    """Format a nested results dict as a readable table string.

    Args:
        results: {method_name: {metric: value}} or {metric: value}
        title: Optional table title

    Returns:
        Formatted string
    """
    lines = []
    if title:
        lines.append(f"\n{'='*60}")
        lines.append(f"  {title}")
        lines.append(f"{'='*60}")

    # Detect if nested (method -> metrics) or flat (metrics)
    first_val = next(iter(results.values())) if results else None
    if isinstance(first_val, dict):
        # Collect all metric keys
        all_keys = sorted({k for v in results.values() for k in v})
        # Header
        col_w = 14
        header = f"{'Method':<24}" + "".join(f"{k:>{col_w}}" for k in all_keys)
        lines.append(header)
        lines.append("-" * len(header))
        for method, metrics in results.items():
            row = f"{method:<24}"
            for k in all_keys:
                v = metrics.get(k, float("nan"))
                row += f"{v:>{col_w}.4f}" if isinstance(v, float) else f"{str(v):>{col_w}}"
            lines.append(row)
    else:
        # Flat dict
        for k, v in results.items():
            lines.append(f"  {k:<30} {v:.4f}" if isinstance(v, float) else f"  {k:<30} {v}")

    return "\n".join(lines)
