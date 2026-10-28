"""Train the logistic regression baseline on the future-conflict label."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.future_label_splits import iter_labeled_split
from src.evaluation.threshold import fit_threshold
from src.graph.sparse_graph import CANDIDATE_RADIUS_NM
from src.features.model_inputs import MODEL_FEATURES, TIME_TO_CPA_CAP_SECONDS, feature_names
from src.labeling.future_conflicts import (
    DEFAULT_FORECAST_HORIZON_SECONDS,
    prepare_future_model_inputs,
)

PREDICTIONS_DIR = Path("data/processed/future_predictions")
# Main baseline uses all six features; the variant drops lateral distance and
# vertical separation, which define the label thresholds.
VARIANTS = {"": True, "_no_distance": False}


def load_split(
    split: str, features: list[str] = MODEL_FEATURES
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Eligible pairs of one split: known label, candidate edge (within the radius).

    This is the single definition of the evaluation set; GraphSAGE must score
    exactly these pairs. Also returns the pair identifiers (snapshot and aircraft).
    """
    parts = []
    for chunk in iter_labeled_split(split):
        candidates = chunk[chunk["lateral_distance_nm"] <= CANDIDATE_RADIUS_NM]
        if not candidates.empty:
            chunk_features, target, ids = prepare_future_model_inputs(candidates)
            parts.append(chunk_features.join(ids).assign(future_conflict=target))
    data = pd.concat(parts, ignore_index=True)
    ids = data[["snapshot_time", "icao24_a", "icao24_b"]]
    return data[features], data["future_conflict"], ids


def save_predictions(model, split: str, features: list[str], suffix: str) -> Path:
    """Write one predicted probability per eligible pair of the split."""
    X, y, ids = load_split(split, features)
    out = ids.assign(future_conflict=y, probability=model.predict_proba(X)[:, 1])
    path = PREDICTIONS_DIR / f"logreg{suffix}_{split}_predictions.csv"
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def evaluate(split: str, threshold: float, suffix: str) -> dict:
    """Metrics of the saved predictions at the validation-chosen threshold."""
    d = pd.read_csv(PREDICTIONS_DIR / f"logreg{suffix}_{split}_predictions.csv")
    y, p = d["future_conflict"], d["probability"]
    flagged = (p >= threshold).astype(int)
    return {
        "pairs": len(d),
        "positives": int(y.sum()),
        "flagged": int(flagged.sum()),
        "precision": precision_score(y, flagged, zero_division=0),
        "recall": recall_score(y, flagged),
        "f1": f1_score(y, flagged),
        "average_precision": average_precision_score(y, p),
    }


def run(suffix: str, include_label_features: bool) -> None:
    features = feature_names(include_label_features)
    X_train, y_train, _ = load_split("train", features)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42))
    model.fit(X_train, y_train)
    model_path = Path(f"models/future_logistic_regression{suffix}.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[{suffix or 'main'}] trained on {len(X_train)} pairs, {int(y_train.sum())} positive; saved {model_path}")
    for split in ("validation", "test"):
        save_predictions(model, split, features, suffix)

    validation = pd.read_csv(PREDICTIONS_DIR / f"logreg{suffix}_validation_predictions.csv")
    threshold = fit_threshold(validation["future_conflict"].to_numpy(), validation["probability"].to_numpy())
    (PREDICTIONS_DIR / f"logreg{suffix}_threshold.json").write_text(
        json.dumps({"threshold": threshold, "chosen_on": "validation"})
    )
    print(f"[{suffix or 'main'}] threshold (validation only): {threshold:.6f}")
    print(f"Saved: {write_run_record(model, threshold, len(X_train), features, suffix)}")


def main() -> None:
    for suffix, include_label_features in VARIANTS.items():
        run(suffix, include_label_features)


def write_run_record(model, threshold: float, train_rows: int, features: list[str], suffix: str) -> Path:
    """Save the feature list, preprocessing, settings, prediction timing and results."""
    X_test, _, ids = load_split("test", features)
    start = time.perf_counter()
    model.predict_proba(X_test)
    seconds = time.perf_counter() - start
    snapshots = ids["snapshot_time"].nunique()
    scaler, classifier = model[0], model[-1]
    record = {
        "features": features,
        "preprocessing": {
            "scaler": "StandardScaler, fitted on training pairs only",
            "mean": dict(zip(features, scaler.mean_.tolist())),
            "scale": dict(zip(features, scaler.scale_.tolist())),
            "time_to_cpa_cap_s": TIME_TO_CPA_CAP_SECONDS,
            "eligible_pairs": f"known label and lateral distance <= {CANDIDATE_RADIUS_NM} NM",
        },
        "model": {**classifier.get_params(), "train_pairs": train_rows},
        "coefficients": dict(zip(features, classifier.coef_[0].tolist())),
        "intercept": float(classifier.intercept_[0]),
        "prediction_time": {
            "features_taken_at": "snapshot_time t (current state only)",
            "predicts": f"conflict onset within {DEFAULT_FORECAST_HORIZON_SECONDS:g} s after t",
            "test_snapshots": int(snapshots),
            "test_pairs": len(X_test),
            "test_scoring_seconds": seconds,
            "test_scoring_ms_per_snapshot": 1000 * seconds / snapshots,
        },
        "threshold": threshold,
        "threshold_chosen_on": "validation",
        "validation_results": evaluate("validation", threshold, suffix),
        "test_results": evaluate("test", threshold, suffix),
    }
    path = PREDICTIONS_DIR / f"logreg{suffix}_run_record.json"
    path.write_text(json.dumps(record, indent=2))
    return path


if __name__ == "__main__":
    main()
