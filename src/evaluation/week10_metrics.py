"""Shared-harness metrics for the tuned GNN and TFT on the held-out test session.

Decision thresholds are chosen on validation data only (lowest 5:1 normalised
error cost, the project's rule) and then applied unchanged to the test
predictions. Threshold-free metrics (AUROC, average precision) do not use them.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.evaluation.harness import compute_classification_metrics
from src.evaluation.threshold import fit_threshold
from src.evaluation.week10_test_set import (
    ALIGNED_PREDICTIONS,
    GNN_PREDICTIONS,
    PROJECT_ROOT,
    TFT_PREDICTIONS,
)
from src.evaluation.week10_validation_scores import (
    ALIGNED_VALIDATION_PREDICTIONS,
    GNN_VALIDATION_PREDICTIONS,
    TFT_VALIDATION_PREDICTIONS,
)

METRICS_PATH = PROJECT_ROOT / "experiments" / "eda" / "week10_shared_harness_metrics.csv"
THRESHOLDS_PATH = PROJECT_ROOT / "experiments" / "eda" / "week10_thresholds.json"
METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "auroc",
    "average_precision",
    "false_negative_rate",
    "false_positive_rate",
    "normalised_cost",
    "true_positives",
    "false_negatives",
    "false_positives",
    "true_negatives",
]


def evaluate(
    evaluation_set: str,
    model: str,
    threshold_rule: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """Run the shared harness once and label the row."""
    metrics = compute_classification_metrics(labels, probabilities, threshold=threshold)
    row: dict[str, object] = {
        "evaluation_set": evaluation_set,
        "model": model,
        "threshold_rule": threshold_rule,
        "threshold": float(threshold),
        "pairs": int(len(labels)),
        "positives": int(np.sum(labels)),
    }
    row.update({column: metrics[column] for column in METRIC_COLUMNS})
    return row


def main() -> None:
    aligned_validation = pd.read_csv(ALIGNED_VALIDATION_PREDICTIONS)
    aligned_test = pd.read_csv(ALIGNED_PREDICTIONS)
    gnn_validation = pd.read_csv(GNN_VALIDATION_PREDICTIONS)
    gnn_test = pd.read_csv(GNN_PREDICTIONS)
    tft_validation = pd.read_csv(TFT_VALIDATION_PREDICTIONS)
    tft_test = pd.read_csv(TFT_PREDICTIONS)

    # (set name, model, validation frame/label/score columns, test frame/label/score columns)
    plans = [
        (
            "shared",
            "GraphSAGE (tuned)",
            (aligned_validation, "future_conflict", "gnn_probability"),
            (aligned_test, "future_conflict", "gnn_probability"),
        ),
        (
            "shared",
            "TFT (tuned)",
            (aligned_validation, "future_conflict", "tft_probability"),
            (aligned_test, "future_conflict", "tft_probability"),
        ),
        (
            "context_gnn_full_test_pairs",
            "GraphSAGE (tuned)",
            (gnn_validation, "future_conflict", "probability"),
            (gnn_test, "future_conflict", "probability"),
        ),
        (
            "context_tft_full_test_windows",
            "TFT (tuned)",
            (tft_validation, "tft_y_future", "probability"),
            (tft_test, "tft_y_future", "probability"),
        ),
    ]

    rows = []
    thresholds: dict[str, dict[str, object]] = {}
    for evaluation_set, model, validation, test in plans:
        validation_frame, label_column, score_column = validation
        test_frame, test_label_column, test_score_column = test
        threshold = fit_threshold(
            validation_frame[label_column].to_numpy(dtype=int),
            validation_frame[score_column].to_numpy(dtype=float),
        )
        thresholds[f"{evaluation_set}:{model}"] = {
            "threshold": threshold,
            "chosen_on": "validation",
            "validation_pairs": int(len(validation_frame)),
            "validation_positives": int(validation_frame[label_column].sum()),
        }
        labels = test_frame[test_label_column].to_numpy(dtype=int)
        scores = test_frame[test_score_column].to_numpy(dtype=float)
        rows.append(
            evaluate(
                evaluation_set, model, "validation_5to1_cost", labels, scores, threshold
            )
        )
        rows.append(evaluate(evaluation_set, model, "fixed_0.5", labels, scores, 0.5))

    result = pd.DataFrame(rows)
    result.to_csv(METRICS_PATH, index=False)
    THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(
            result.drop(columns=["true_negatives", "false_positive_rate"])
            .round(4)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
