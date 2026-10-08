"""Train the Week 3 imbalance strategies on the backfill and log them to MLflow.

Each strategy is trained under two splits. The temporal split is the project
split every model now uses. The random split is how Week 3 originally divided
its rows; it is run only to measure how much repeated encounters straddling a
random cut inflate the scores.

Models are scored on the validation part. The test part is left untouched for
the Week 10 comparison, since choosing the best strategy here is itself a
decision that would otherwise be made by looking at the test set.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.baselines.data import (
    DEFAULT_PAIRS_PATH,
    SPLIT_METHODS,
    load_dataset,
    split_dataset,
)
from src.baselines.logistic_regression_baseline import (
    save_model as save_baseline_model,
    train_baseline_model,
)
from src.baselines.logistic_regression_class_weight import (
    save_model as save_class_weight_model,
    train_class_weight_model,
)
from src.baselines.logistic_regression_focal_loss import (
    save_model as save_focal_model,
    train_focal_loss_model,
)
from src.baselines.logistic_regression_smote import (
    save_model as save_smote_model,
    train_smote_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = DEFAULT_PAIRS_PATH
DEFAULT_TRACKING_URI = (PROJECT_ROOT / "mlruns").resolve().as_uri()
# A fresh experiment, so the original single-snapshot Week 3 runs stay intact
# as history rather than being mixed in with runs on different data.
DEFAULT_EXPERIMENT_NAME = "step6_split_comparison"
DEFAULT_COMPARISON_PATH = PROJECT_ROOT / "experiments" / "eda" / "step6_split_comparison.csv"
MODEL_DIR = PROJECT_ROOT / "models"

# Columns written to the comparison table, in reading order.
COMPARISON_METRICS = (
    "normalised_cost", "recall", "precision", "f1", "average_precision", "auroc",
    "false_negative_rate", "false_positive_rate",
    "true_positives", "false_negatives", "false_positives", "true_negatives",
)


# This function writes the PR-curve arrays from the shared evaluation harness to a
# CSV artifact so every MLflow run keeps the exact threshold data behind its plot.
def write_pr_curve_csv(pr_curve: dict[str, np.ndarray], output_path: Path) -> None:
    thresholds = np.asarray(pr_curve["thresholds"])
    precision = np.asarray(pr_curve["precision"])
    recall = np.asarray(pr_curve["recall"])

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["precision", "recall", "threshold"])
        for index in range(len(precision)):
            threshold = thresholds[index] if index < len(thresholds) else ""
            writer.writerow([precision[index], recall[index], threshold])


# This function creates the precision-recall visualization used as an MLflow
# artifact for one imbalance strategy.
def write_pr_curve_plot(
    pr_curve: dict[str, np.ndarray],
    average_precision: float,
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(
        pr_curve["recall"],
        pr_curve["precision"],
        label=f"Average precision = {average_precision:.4f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Week 3 precision-recall curve")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# This function records one model's parameters, metrics, model artifact, and PR
# curve artifacts in a single MLflow run.
def log_run(
    strategy: str,
    split: str,
    model: object,
    metrics: dict,
    save_model_function: object,
    output_path: Path,
    artifact_directory: Path,
    sizes: dict[str, int],
) -> None:
    mlflow.start_run(run_name=f"{strategy}-{split}")
    try:
        mlflow.set_tags({
            "week": "3", "study": "imbalance", "strategy": strategy,
            "split": split, "evaluated_on": "validation",
            "data": "states_socal_backfill_1790375809",
        })
        mlflow.log_params({"strategy": strategy, "split": split,
                           "threshold": metrics["threshold"], **sizes})

        scalar_metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and key != "threshold"
        }
        mlflow.log_metrics({key: float(value) for key, value in scalar_metrics.items()})

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_model_function(model, output_path)
        mlflow.log_artifact(str(output_path), artifact_path="model")

        stem = f"{strategy}_{split}_precision_recall_curve"
        curve_csv = artifact_directory / f"{stem}.csv"
        curve_plot = artifact_directory / f"{stem}.png"
        write_pr_curve_csv(metrics["pr_curve"], curve_csv)
        write_pr_curve_plot(metrics["pr_curve"], metrics["average_precision"], curve_plot)
        mlflow.log_artifact(str(curve_csv), artifact_path="precision_recall_curve")
        mlflow.log_artifact(str(curve_plot), artifact_path="precision_recall_curve")
    finally:
        mlflow.end_run()


# name, train function, save function, model file, trains on numpy arrays
STRATEGIES = [
    ("baseline", train_baseline_model, save_baseline_model,
     "week3_logistic_regression_baseline.joblib", False),
    ("smote", train_smote_model, save_smote_model,
     "week3_logistic_regression_smote.joblib", False),
    ("class_weight", train_class_weight_model, save_class_weight_model,
     "week3_logistic_regression_class_weight.joblib", False),
    ("focal_loss", train_focal_loss_model, save_focal_model,
     "week3_logistic_regression_focal_loss.pt", True),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        help="MLflow tracking URI. Defaults to the local mlruns folder.",
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--splits", nargs="+", choices=SPLIT_METHODS,
                        default=list(SPLIT_METHODS))
    parser.add_argument("--comparison-path", type=Path, default=DEFAULT_COMPARISON_PATH)
    args = parser.parse_args()

    if args.tracking_uri.startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    X, y, times = load_dataset(args.data_path)
    rows = []

    with TemporaryDirectory() as temporary_directory:
        artifact_directory = Path(temporary_directory)
        for split in args.splits:
            parts = split_dataset(X, y, times, method=split)
            (X_train, y_train), (X_eval, y_eval) = parts["train"], parts["validation"]
            sizes = {"train_rows": len(y_train), "train_positives": int(y_train.sum()),
                     "eval_rows": len(y_eval), "eval_positives": int(y_eval.sum())}

            for strategy, train, save, model_file, uses_arrays in STRATEGIES:
                print(f"  training {strategy:<13} on the {split} split ...", flush=True)
                if uses_arrays:
                    model, metrics = train(X_train.to_numpy(), y_train.to_numpy(),
                                           X_eval.to_numpy(), y_eval.to_numpy())
                else:
                    model, metrics = train(X_train, y_train, X_eval, y_eval)

                # The temporal models are the ones carried forward, so they
                # replace the Week 3 files; random-split models are only evidence.
                directory = MODEL_DIR if split == "temporal" else MODEL_DIR / "random_split"
                log_run(strategy, split, model, metrics, save, directory / model_file,
                        artifact_directory, sizes)
                rows.append({"split": split, "strategy": strategy, **sizes,
                             **{name: metrics[name] for name in COMPARISON_METRICS}})

    comparison = pd.DataFrame(rows)
    args.comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.comparison_path, index=False)

    print()
    print(comparison[["split", "strategy", "normalised_cost", "recall", "precision",
                      "f1", "average_precision", "auroc"]].round(4).to_string(index=False))
    print()
    print(f"MLflow experiment: {args.experiment_name}")
    print(f"Comparison table : {args.comparison_path.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()