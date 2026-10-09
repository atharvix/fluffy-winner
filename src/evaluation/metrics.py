"""Evaluation metrics calculation for binary classification with clinical metric audit.

Explicit target conventions:
    0 -> malignant (clinical condition of interest)
    1 -> benign (non-cancerous)

Positive-class conventions:
- In clinical screening: Malignant (0) is the condition of interest.
  - Malignant Sensitivity (Recall): Proportion of actual malignant cases correctly identified.
  - Malignant Specificity: Proportion of actual benign cases correctly identified.
- In standard scikit-learn binary metrics with labels {0, 1}, class 1 (Benign) is the positive label:
  - Benign Precision: Proportion of predicted benign cases that are actually benign.
  - Benign Recall: Proportion of actual benign cases correctly identified.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred_probs: Union[np.ndarray, List[float]],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for binary predictions.

    Args:
        y_true: Ground truth binary labels (0 or 1), 1D array or list.
        y_pred_probs: Predicted probabilities for class 1 (benign), shape (N,).
        threshold: Decision threshold to convert probabilities to class predictions (default 0.5).

    Returns:
        Dictionary containing overall, malignant-specific, and benign-specific metrics.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_probs_arr = np.asarray(y_pred_probs).ravel()

    if len(y_true_arr) != len(y_probs_arr):
        raise ValueError(
            f"Shape mismatch: y_true has {len(y_true_arr)} items, y_pred_probs has {len(y_probs_arr)} items."
        )

    y_pred = (y_probs_arr >= threshold).astype(int)

    acc = float(accuracy_score(y_true_arr, y_pred))

    # Scikit-learn default (pos_label=1, Benign)
    prec_benign = float(precision_score(y_true_arr, y_pred, pos_label=1, zero_division=0))
    rec_benign = float(recall_score(y_true_arr, y_pred, pos_label=1, zero_division=0))
    f1_benign = float(f1_score(y_true_arr, y_pred, pos_label=1, zero_division=0))

    # Malignant (pos_label=0)
    prec_malignant = float(precision_score(y_true_arr, y_pred, pos_label=0, zero_division=0))
    rec_malignant = float(recall_score(y_true_arr, y_pred, pos_label=0, zero_division=0))
    f1_malignant = float(f1_score(y_true_arr, y_pred, pos_label=0, zero_division=0))

    # ROC-AUC is applicable only if both classes are present in y_true
    roc_auc: Optional[float] = None
    unique_classes = np.unique(y_true_arr)
    if len(unique_classes) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true_arr, y_probs_arr))
        except Exception:
            roc_auc = None

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    # For labels [0 (Malignant), 1 (Benign)]:
    # cm[0, 0] = True Malignant, Pred Malignant (TP_mal / TN_ben)
    # cm[0, 1] = True Malignant, Pred Benign    (FN_mal / FP_ben)
    # cm[1, 0] = True Benign,    Pred Malignant (FP_mal / FN_ben)
    # cm[1, 1] = True Benign,    Pred Benign    (TN_mal / TP_ben)
    tp_mal = int(cm[0, 0])
    fn_mal = int(cm[0, 1])
    fp_mal = int(cm[1, 0])
    tn_mal = int(cm[1, 1])

    # Sensitivity (Recall) & Specificity for Malignant (class 0)
    mal_total = tp_mal + fn_mal
    ben_total = tn_mal + fp_mal

    mal_sensitivity = float(tp_mal / mal_total) if mal_total > 0 else 0.0
    mal_specificity = float(tn_mal / ben_total) if ben_total > 0 else 0.0

    cm_dict = {
        "raw_matrix": cm.tolist(),
        "true_malignant_pred_malignant": tp_mal,
        "true_malignant_pred_benign": fn_mal,
        "true_benign_pred_malignant": fp_mal,
        "true_benign_pred_benign": tn_mal,
    }

    return {
        "accuracy": acc,
        "precision": prec_benign,
        "recall": rec_benign,
        "f1": f1_benign,
        "roc_auc": roc_auc,
        "malignant_sensitivity": mal_sensitivity,
        "malignant_specificity": mal_specificity,
        "malignant_metrics": {
            "sensitivity": mal_sensitivity,
            "specificity": mal_specificity,
            "precision": prec_malignant,
            "recall": rec_malignant,
            "f1": f1_malignant,
        },
        "benign_metrics": {
            "precision": prec_benign,
            "recall": rec_benign,
            "specificity": mal_sensitivity,
            "f1": f1_benign,
        },
        "confusion_matrix": cm_dict,
        "decision_threshold": float(threshold),
        "sample_count": int(len(y_true_arr)),
    }
