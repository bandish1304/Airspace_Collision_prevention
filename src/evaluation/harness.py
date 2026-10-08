"""Reusable evaluation harness for binary collision-risk models.

The harness is designed to provide a single, consistent scoring definition for the
Week 3 imbalance study and for later GNN/TFT comparisons.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# A missed conflict is treated as five times worse than a false alarm. The
# ratio is the project's, set in the Week 5 plan; it is a parameter so a
# sensitivity analysis can vary it.
DEFAULT_FALSE_NEGATIVE_COST = 5.0


# This function turns model probabilities into a single, consistent set of
# classification metrics for the imbalance study and later model comparisons.
# It computes both threshold-based metrics (precision, recall, F1) and ranking
# metrics (AUROC, average precision), which are needed for safety-critical risk modeling.
def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    false_negative_cost: float = DEFAULT_FALSE_NEGATIVE_COST,
) -> dict[str, float | int | dict[str, Any]]:
    """Return the core metrics used across the project comparison harness."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("y_true and y_prob must be 1D arrays.")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have equal length.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in the range [0, 1].")

    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    actual_positive = tp + fn
    actual_negative = tn + fp

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auroc = roc_auc_score(y_true, y_prob)
    average_precision = average_precision_score(y_true, y_prob)

    false_negative_rate = float(fn / actual_positive) if actual_positive else 0.0
    false_positive_rate = float(fp / actual_negative) if actual_negative else 0.0

    # Cost of the errors, with each false alarm costing 1. The normalised form
    # divides by the cost of never predicting risk at all, so 0 is perfect and
    # 1 is no better than ignoring the problem -- comparable across test sets
    # of different sizes. It can exceed 1 when false alarms pile up.
    cost = false_negative_cost * fn + fp
    normalised_cost = (
        float(cost / (false_negative_cost * actual_positive)) if actual_positive else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(y_true)) if len(y_true) else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "average_precision": float(average_precision),
        "false_negative_rate": float(false_negative_rate),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_cost": float(false_negative_cost),
        "misclassification_cost": float(cost),
        "normalised_cost": normalised_cost,
        # The counts again as plain numbers: the dictionary below is kept for
        # existing readers, but MLflow only logs scalars and silently skipped it.
        "true_positives": int(tp),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


# This function extracts the precision-recall curve values so the model can be
# assessed across multiple decision thresholds, which is especially important when
# the positive class is rare and false negatives are costly.
def precision_recall_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return precision and recall arrays for the PR curve."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("y_true and y_prob must be 1D arrays.")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have equal length.")

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    return {"precision": precision, "recall": recall, "thresholds": thresholds}


# This is the main reusable evaluation entry point. It groups all metrics and PR
# curve information into one dictionary so every model in the project uses exactly
# the same scoring setup during the imbalance study and later comparison phases.
def build_evaluation_harness(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    false_negative_cost: float = DEFAULT_FALSE_NEGATIVE_COST,
) -> dict[str, Any]:
    """Bundle classification metrics and PR-curve data into one consistent report."""
    metrics = compute_classification_metrics(
        y_true, y_prob, threshold=threshold, false_negative_cost=false_negative_cost
    )
    pr_curve = precision_recall_curve_data(y_true, y_prob)
    metrics["pr_curve"] = pr_curve
    return metrics
