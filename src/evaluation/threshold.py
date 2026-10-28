"""Validation-only threshold selection shared by the baseline and GraphSAGE."""

from __future__ import annotations

import numpy as np


def fit_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Choose the validation threshold with lowest 5:1 normalized error cost."""
    if y_true.ndim != 1 or y_prob.ndim != 1 or len(y_true) != len(y_prob):
        raise ValueError("Validation labels and probabilities must be equal-length 1D arrays.")
    positives = int(y_true.sum())
    if positives == 0 or positives == len(y_true):
        raise ValueError("Validation labels must contain both classes to tune a threshold.")

    order = np.argsort(-y_prob, kind="stable")
    sorted_probabilities = y_prob[order]
    sorted_positives = y_true[order].astype(np.int64)
    cumulative_positives = np.cumsum(sorted_positives)
    group_ends = np.r_[
        np.flatnonzero(sorted_probabilities[:-1] != sorted_probabilities[1:]),
        len(sorted_probabilities) - 1,
    ]
    predicted_positive_count = group_ends + 1
    true_positives = cumulative_positives[group_ends]
    false_negatives = positives - true_positives
    false_positives = predicted_positive_count - true_positives
    normalized_cost = (5 * false_negatives + false_positives) / (5 * positives)
    best = int(np.argmin(normalized_cost))
    return float(sorted_probabilities[group_ends[best]])
