"""Unit tests for Network Intrusion Detection classification metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import compute_nids_metrics, optimize_decision_threshold


def test_perfect_nids_metrics():
    """Verify metrics computation on a perfect classification run."""
    y_true = [0, 0, 1, 1]
    y_probs = [0.05, 0.10, 0.90, 0.95]

    m = compute_nids_metrics(y_true=y_true, y_pred_probs=y_probs, threshold=0.5)

    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["attack_recall"] == 1.0
    assert m["attack_precision"] == 1.0
    assert m["attack_f1"] == 1.0
    assert m["benign_recall"] == 1.0
    assert m["benign_precision"] == 1.0
    assert m["false_positive_rate"] == 0.0
    assert m["false_negative_rate"] == 0.0
    assert m["roc_auc"] == 1.0

    cm = m["confusion_matrix"]
    assert cm["tn"] == 2
    assert cm["fp"] == 0
    assert cm["fn"] == 0
    assert cm["tp"] == 2


def test_known_imperfect_nids_metrics():
    """Verify explicit calculation of detection rate, false alarm rate, and confusion matrix."""
    # 4 benign (0), 4 attacks (1)
    # Predictions:
    # 3 benign predicted benign (TN=3), 1 benign predicted attack (FP=1) -> FPR = 1/4 = 0.25
    # 3 attacks predicted attack (TP=3), 1 attack predicted benign (FN=1) -> Attack Recall = 3/4 = 0.75, FNR = 0.25
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_probs = [0.1, 0.2, 0.3, 0.8, 0.2, 0.7, 0.8, 0.9]

    m = compute_nids_metrics(y_true=y_true, y_pred_probs=y_probs, threshold=0.5)

    assert m["confusion_matrix"]["tn"] == 3
    assert m["confusion_matrix"]["fp"] == 1
    assert m["confusion_matrix"]["fn"] == 1
    assert m["confusion_matrix"]["tp"] == 3

    assert m["attack_recall"] == 0.75
    assert m["benign_recall"] == 0.75
    assert m["false_positive_rate"] == 0.25
    assert m["false_negative_rate"] == 0.25
    assert m["accuracy"] == 0.75
    assert m["balanced_accuracy"] == 0.75


def test_mismatched_length_raises():
    """Verify ValueError is raised if y_true and y_pred_probs lengths do not match."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_nids_metrics(y_true=[0, 1], y_pred_probs=[0.5])


def test_optimize_decision_threshold_youden_j():
    """Verify threshold optimizer identifies cutoff maximizing Recall - FPR."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Clear separation around 0.45
    y_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9])

    tau = optimize_decision_threshold(y_true, y_probs, metric="youden_j")
    assert 0.40 <= tau <= 0.50

    # With tau around 0.45, TPR = 1.0, FPR = 0.0 -> Youden's J = 1.0
    m = compute_nids_metrics(y_true, y_probs, threshold=tau)
    assert m["attack_recall"] == 1.0
    assert m["false_positive_rate"] == 0.0


def test_optimize_decision_threshold_constrained_fpr():
    """Verify threshold optimizer enforces maximum allowable false positive rate."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Benign probabilities are slightly elevated
    y_probs = np.array([0.05, 0.10, 0.15, 0.25, 0.30, 0.50, 0.70, 0.85])

    # Allow max FPR 0.0 -> threshold must be > 0.25
    tau = optimize_decision_threshold(y_true, y_probs, metric="constrained_fpr", max_allowed_fpr=0.0)
    m = compute_nids_metrics(y_true, y_probs, threshold=tau)
    assert m["false_positive_rate"] == 0.0

