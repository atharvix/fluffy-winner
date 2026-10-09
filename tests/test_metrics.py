"""Tests for classification metrics computation."""

import numpy as np
import pytest

from src.evaluation.metrics import compute_classification_metrics


def test_perfect_classification_metrics():
    """Verify metrics calculation for perfect predictions."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_probs = np.array([0.05, 0.10, 0.90, 0.85, 0.95])

    metrics = compute_classification_metrics(y_true, y_probs, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"]["true_malignant_pred_malignant"] == 2
    assert metrics["confusion_matrix"]["true_malignant_pred_benign"] == 0
    assert metrics["confusion_matrix"]["true_benign_pred_malignant"] == 0
    assert metrics["confusion_matrix"]["true_benign_pred_benign"] == 3


def test_known_imperfect_metrics():
    """Verify metrics calculation with known false positives and false negatives."""
    # Class 0: Malignant, Class 1: Benign
    y_true = np.array([0, 0, 1, 1])
    # Predictions: 0 -> predicted 0 (TP0), 0 -> predicted 1 (FP1), 1 -> predicted 0 (FN1), 1 -> predicted 1 (TP1)
    y_probs = np.array([0.1, 0.8, 0.2, 0.9])

    metrics = compute_classification_metrics(y_true, y_probs, threshold=0.5)

    # Predictions: [0, 1, 0, 1]
    # Accuracy: 2 / 4 = 0.5
    assert metrics["accuracy"] == 0.5
    # Class 1 (Benign) Precision: 1 TP / (1 TP + 1 FP) = 0.5
    assert metrics["precision"] == 0.5
    # Class 1 (Benign) Recall: 1 TP / (1 TP + 1 FN) = 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5

    cm = metrics["confusion_matrix"]
    assert cm["true_malignant_pred_malignant"] == 1
    assert cm["true_malignant_pred_benign"] == 1
    assert cm["true_benign_pred_malignant"] == 1
    assert cm["true_benign_pred_benign"] == 1


def test_mismatched_length_raises():
    """Verify that mismatched array lengths raise ValueError."""
    y_true = [0, 1, 0]
    y_probs = [0.1, 0.9]

    with pytest.raises(ValueError):
        compute_classification_metrics(y_true, y_probs)
