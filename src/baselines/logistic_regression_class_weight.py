"""Train the Week 3 logistic regression baseline with class weighting."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.baselines.data import DEFAULT_PAIRS_PATH, load_dataset, split_dataset
from src.evaluation.harness import build_evaluation_harness

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = DEFAULT_PAIRS_PATH
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "week3_logistic_regression_class_weight.joblib"

# This function trains logistic regression with balanced class weights. The
# minority class receives more influence during fitting without changing the
# evaluation rows or the shared evaluation procedure.
def train_class_weight_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    random_state: int = 42,
) -> tuple[Pipeline, dict]:

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_eval)[:, 1]
    metrics = build_evaluation_harness(y_eval.to_numpy(), y_prob, threshold=0.5)
    return pipeline, metrics


# This function saves the trained classifier for the comparison and the
# later MLflow logging step.
def save_model(model: Pipeline, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the labeled pair dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained class-weighted model.",
    )
    args = parser.parse_args()

    X, y, times = load_dataset(args.data_path)
    parts = split_dataset(X, y, times)
    model, metrics = train_class_weight_model(*parts["train"], *parts["validation"])
    save_model(model, args.output_path)

    print(f"Dataset rows: {len(X)}")
    print(f"Positive rate: {float(y.mean()):.4f}")
    print(f"Model saved to: {args.output_path}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"False negative rate: {metrics['false_negative_rate']:.4f}")


if __name__ == "__main__":
    main()