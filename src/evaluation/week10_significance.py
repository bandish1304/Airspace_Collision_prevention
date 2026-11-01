"""Significance tests for the GNN-vs-TFT metric differences on the shared test pairs.

Rows are repeated observations of the same aircraft pair, so they are not
independent. The primary test is a paired cluster bootstrap: whole aircraft
pairs are resampled, and both models are scored on every resample with the
thresholds chosen on validation data. McNemar's exact test on the row-level
errors is reported as a secondary check; it ignores the clustering and is
therefore more optimistic.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.week10_test_set import ALIGNED_PREDICTIONS, PROJECT_ROOT
from src.evaluation.week10_metrics import THRESHOLDS_PATH

SIGNIFICANCE_PATH = PROJECT_ROOT / "experiments" / "eda" / "week10_significance_tests.csv"
SECONDARY_PATH = PROJECT_ROOT / "experiments" / "eda" / "week10_mcnemar_tests.json"
BOOTSTRAP_RESAMPLES = 2_000
CONFIDENCE = 0.95
SEED = 42
METRICS = [
    "precision",
    "recall",
    "f1",
    "false_negative_rate",
    "auroc",
    "average_precision",
]


def weighted_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """Harness metrics where each row counts ``weights`` times (bootstrap multiplicity)."""
    true_positive = float(np.sum(weights * (predictions == 1) * (labels == 1)))
    false_positive = float(np.sum(weights * (predictions == 1) * (labels == 0)))
    false_negative = float(np.sum(weights * (predictions == 0) * (labels == 1)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_negative_rate": 1.0 - recall,
        "auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
        "average_precision": float(
            average_precision_score(labels, scores, sample_weight=weights)
        ),
    }


def bootstrap_p_value(differences: np.ndarray) -> float:
    """Two-sided percentile-bootstrap p-value for a zero difference."""
    at_or_below = int(np.sum(differences <= 0))
    at_or_above = int(np.sum(differences >= 0))
    p_value = 2 * (min(at_or_below, at_or_above) + 1) / (len(differences) + 1)
    return float(min(1.0, p_value))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment for several simultaneous comparisons."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_cluster_bootstrap(
    labels: np.ndarray,
    gnn_scores: np.ndarray,
    tft_scores: np.ndarray,
    gnn_threshold: float,
    tft_threshold: float,
    clusters: np.ndarray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
    progress: Callable[[int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Return per-resample metric differences (TFT minus GNN) for each metric."""
    cluster_ids, cluster_index = np.unique(clusters, return_inverse=True)
    rng = np.random.default_rng(seed)
    gnn_predictions = (gnn_scores >= gnn_threshold).astype(int)
    tft_predictions = (tft_scores >= tft_threshold).astype(int)
    differences = {metric: np.empty(resamples) for metric in METRICS}
    for draw in range(resamples):
        counts = np.bincount(
            rng.integers(0, len(cluster_ids), len(cluster_ids)),
            minlength=len(cluster_ids),
        )
        weights = counts[cluster_index].astype(float)
        if not np.any((labels == 1) & (weights > 0)):
            raise ValueError("A bootstrap resample contained no positives.")
        gnn = weighted_metrics(labels, gnn_scores, gnn_predictions, weights)
        tft = weighted_metrics(labels, tft_scores, tft_predictions, weights)
        for metric in METRICS:
            differences[metric][draw] = tft[metric] - gnn[metric]
        if progress is not None and (draw + 1) % 250 == 0:
            progress(draw + 1)
    return differences


def mcnemar_exact(first_correct: np.ndarray, second_correct: np.ndarray) -> dict[str, float | int]:
    """Exact McNemar test on paired correct/incorrect outcomes."""
    first_only = int(np.sum(first_correct & ~second_correct))
    second_only = int(np.sum(~first_correct & second_correct))
    discordant = first_only + second_only
    p_value = (
        float(binomtest(first_only, discordant, 0.5).pvalue) if discordant else 1.0
    )
    return {
        "gnn_only_correct": first_only,
        "tft_only_correct": second_only,
        "p_value": p_value,
    }


def main() -> None:
    aligned = pd.read_csv(ALIGNED_PREDICTIONS)
    thresholds = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    gnn_threshold = float(thresholds["shared:GraphSAGE (tuned)"]["threshold"])
    tft_threshold = float(thresholds["shared:TFT (tuned)"]["threshold"])

    labels = aligned["future_conflict"].to_numpy(dtype=int)
    gnn_scores = aligned["gnn_probability"].to_numpy(dtype=float)
    tft_scores = aligned["tft_probability"].to_numpy(dtype=float)
    clusters = (
        aligned["icao24_a"].astype(str) + "|" + aligned["icao24_b"].astype(str)
    ).to_numpy()

    ones = np.ones(len(labels))
    gnn_point = weighted_metrics(labels, gnn_scores, (gnn_scores >= gnn_threshold).astype(int), ones)
    tft_point = weighted_metrics(labels, tft_scores, (tft_scores >= tft_threshold).astype(int), ones)

    print(
        f"{len(labels):,} pairs in {len(np.unique(clusters)):,} aircraft-pair clusters; "
        f"{BOOTSTRAP_RESAMPLES:,} resamples",
        flush=True,
    )
    differences = paired_cluster_bootstrap(
        labels,
        gnn_scores,
        tft_scores,
        gnn_threshold,
        tft_threshold,
        clusters,
        progress=lambda done: print(f"  {done:,} resamples", flush=True),
    )

    alpha = (1 - CONFIDENCE) / 2
    rows = []
    for metric in METRICS:
        draws = differences[metric]
        rows.append(
            {
                "metric": metric,
                "gnn": gnn_point[metric],
                "tft": tft_point[metric],
                "difference_tft_minus_gnn": tft_point[metric] - gnn_point[metric],
                "ci_low": float(np.quantile(draws, alpha)),
                "ci_high": float(np.quantile(draws, 1 - alpha)),
                "p_value": bootstrap_p_value(draws),
            }
        )
    adjusted = holm_adjust([row["p_value"] for row in rows])
    for row, holm_p in zip(rows, adjusted, strict=True):
        row["p_value_holm"] = holm_p
        row["clusters"] = int(len(np.unique(clusters)))
        row["resamples"] = BOOTSTRAP_RESAMPLES
    result = pd.DataFrame(rows)
    result.to_csv(SIGNIFICANCE_PATH, index=False)

    gnn_flag = gnn_scores >= gnn_threshold
    tft_flag = tft_scores >= tft_threshold
    positive = labels == 1
    secondary = {
        "note": "Row-level McNemar ignores repeated observations of the same pair; treat as optimistic.",
        "detected_conflicts_among_positives": mcnemar_exact(
            gnn_flag[positive], tft_flag[positive]
        ),
        "correct_rejections_among_negatives": mcnemar_exact(
            ~gnn_flag[~positive], ~tft_flag[~positive]
        ),
    }
    SECONDARY_PATH.write_text(json.dumps(secondary, indent=2), encoding="utf-8")

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(result.drop(columns=["clusters", "resamples"]).round(4).to_string(index=False))
    print(json.dumps(secondary, indent=2))


if __name__ == "__main__":
    main()
