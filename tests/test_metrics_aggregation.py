import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.metrics import compute_classification_metrics, aggregate_client_eval_metrics


def test_compute_classification_metrics_hand_computed():
    # labels: [0, 0, 1, 1, 1], preds: [0, 1, 1, 1, 2]
    # class 0: support=2, tp=1 -> recall=0.5
    # class 1: support=3, tp=2 -> recall=2/3
    # class 2: support=0 -> excluded from balanced_acc, recall reported 0.0
    preds = [0, 1, 1, 1, 2]
    labels = [0, 0, 1, 1, 1]
    m = compute_classification_metrics(preds, labels, num_classes=3)
    assert m["recall_class_0"] == 0.5
    assert abs(m["recall_class_1"] - 2 / 3) < 1e-9
    assert m["recall_class_2"] == 0.0
    assert abs(m["balanced_acc"] - (0.5 + 2 / 3) / 2) < 1e-9
    assert m["support_class_0"] == 2
    assert m["support_class_1"] == 3
    assert m["support_class_2"] == 0
    assert m["tp_class_0"] == 1
    assert m["tp_class_1"] == 2


def _client_metrics(preds, labels, num_classes=3):
    m = compute_classification_metrics(preds, labels, num_classes=num_classes)
    return {
        "poi_hr@5": 1.0 if preds[0] == labels[0] else 0.0,
        "poi_balanced_acc": m["balanced_acc"],
        "poi_macro_f1": m["macro_f1"],
        **{f"poi_recall_class_{c}": m[f"recall_class_{c}"] for c in range(num_classes)},
        **{f"poi_tp_class_{c}": m[f"tp_class_{c}"] for c in range(num_classes)},
        **{f"poi_fp_class_{c}": m[f"fp_class_{c}"] for c in range(num_classes)},
        **{f"poi_support_class_{c}": m[f"support_class_{c}"] for c in range(num_classes)},
    }


def test_aggregate_reconciles_balanced_acc_with_pooled_recall():
    # Two tiny clients, each with a single-class test set (the realistic
    # BehaviorFL case: ~2-3 held-out trips per client). Client A only ever
    # sees class 0 and gets it right every time (locally "perfect balanced
    # accuracy" for the one class it has). Client B only sees class 1 and
    # gets it wrong every time. A naive weighted average of each client's
    # own balanced_acc would report ~0.5 (average of 1.0 and 0.0), which is
    # not the true dataset-level balanced accuracy across classes 0 and 1.
    client_a = (2, _client_metrics(preds=[0, 0], labels=[0, 0]))
    client_b = (2, _client_metrics(preds=[2, 2], labels=[1, 1]))

    agg = aggregate_client_eval_metrics([client_a, client_b], num_classes=3)

    # Pooled: class 0 recall = 2/2 = 1.0, class 1 recall = 0/2 = 0.0, class 2 absent.
    assert agg["poi_recall_class_0"] == 1.0
    assert agg["poi_recall_class_1"] == 0.0
    assert agg["poi_recall_class_2"] == 0.0
    assert abs(agg["poi_balanced_acc"] - 0.5) < 1e-9

    # balanced_acc must equal the mean of the reported per-class recall
    # columns restricted to globally-present classes (0 and 1 here) -- this
    # is exactly the reconciliation that was previously broken.
    present_recalls = [agg["poi_recall_class_0"], agg["poi_recall_class_1"]]
    assert abs(agg["poi_balanced_acc"] - sum(present_recalls) / len(present_recalls)) < 1e-9


def test_aggregate_pools_hr_by_sample_count():
    # HR@5-style rate metrics must be pooled by sample count (equivalent to
    # a single trip-level pooled accuracy), unlike the per-class metrics.
    client_a = (3, {"poi_hr@5": 1.0})
    client_b = (1, {"poi_hr@5": 0.0})
    agg = aggregate_client_eval_metrics([client_a, client_b], num_classes=1)
    # (3*1.0 + 1*0.0) / 4 = 0.75
    assert abs(agg["poi_hr@5"] - 0.75) < 1e-9


def test_aggregate_pools_macro_f1_from_pooled_counts():
    # Two-class case exercising precision (via false positives), not just
    # recall, to catch bugs specific to macro-F1 pooling.
    # Client A: labels=[0,0,1], preds=[0,1,1]
    # class 0: support=2, tp=1, fp=0
    # class 1: support=1, tp=1, fp=1
    # Client B: labels=[1,1], preds=[0,1]
    # class 0: support=0, tp=0, fp=1
    # class 1: support=2, tp=1, fp=0
    client_a = (3, _client_metrics(preds=[0, 1, 1], labels=[0, 0, 1], num_classes=2))
    client_b = (2, _client_metrics(preds=[0, 1], labels=[1, 1], num_classes=2))

    agg = aggregate_client_eval_metrics([client_a, client_b], num_classes=2)

    # Pooled class 0: tp=1, fp=0+1=1, support=2 -> recall=0.5, precision=0.5, f1=0.5
    # Pooled class 1: tp=1+1=2, fp=1+0=1, support=1+2=3 -> recall=2/3, precision=2/3, f1=2/3
    expected_macro_f1 = (0.5 + 2 / 3) / 2
    assert abs(agg["poi_macro_f1"] - expected_macro_f1) < 1e-9
    assert abs(agg["poi_recall_class_0"] - 0.5) < 1e-9
    assert abs(agg["poi_recall_class_1"] - 2 / 3) < 1e-9


def test_aggregate_weights_music_by_music_n_samples_not_total_examples():
    # Music-prefixed rate keys must be weighted by music_n_samples (the
    # count of music-supervised samples), not by the client's total sample
    # count, since not every trip carries music supervision.
    client_a = (10, {"music_hr@5": 1.0, "music_n_samples": 2})
    client_b = (10, {"music_hr@5": 0.0, "music_n_samples": 8})
    agg = aggregate_client_eval_metrics([client_a, client_b], num_classes=1)
    # Weighted by music_n_samples: (2*1.0 + 8*0.0) / 10 = 0.2
    # (weighting by total num_examples would instead give (10*1.0+10*0.0)/20=0.5)
    assert abs(agg["music_hr@5"] - 0.2) < 1e-9


def test_aggregate_does_not_invent_absent_legacy_task_metrics():
    agg = aggregate_client_eval_metrics([
        (4, {
            "destination_tp_class_0": 3,
            "destination_fp_class_0": 1,
            "destination_support_class_0": 4,
        })
    ], num_classes=16)
    assert "destination_balanced_acc" in agg
    assert not any(key.startswith("poi_recall_class_") for key in agg)
    assert not any(key.startswith("music_recall_class_") for key in agg)
