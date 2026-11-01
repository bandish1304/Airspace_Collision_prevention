"""Log Week 8 TFT trials and the final Week 9 model to MLflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRIALS_PATH = PROJECT_ROOT / "experiments" / "eda" / "week8_optuna_trials.csv"
BEST_TRIAL_PATH = PROJECT_ROOT / "experiments" / "eda" / "week8_optuna_best.json"
MODEL_DIR = PROJECT_ROOT / "models" / "week9_tft"
CONFIG_PATH = MODEL_DIR / "week9_tft_training_config.json"
CHECKPOINT_PATH = MODEL_DIR / "week9_tft_best.ckpt"
DEFAULT_TRACKING_URI = (PROJECT_ROOT / "mlruns").resolve().as_uri()
DEFAULT_EXPERIMENT_NAME = "week9_tft_final_training"


def build_validation_history(log_dir: Path) -> pd.DataFrame:
    """Combine resumed Lightning logs into one validation row per epoch."""
    history_rows: list[dict[str, float | int | str]] = []
    for metrics_path in sorted(log_dir.glob("version_*/metrics.csv")):
        version = int(metrics_path.parent.name.removeprefix("version_"))
        metrics = pd.read_csv(metrics_path)
        if not {"epoch", "val_loss"}.issubset(metrics.columns):
            raise ValueError(f"Lightning metrics file lacks required columns: {metrics_path}")
        validation = metrics.loc[metrics["val_loss"].notna(), ["epoch", "step", "val_loss"]]
        for row in validation.itertuples(index=False):
            history_rows.append(
                {
                    "epoch": int(row.epoch),
                    "step": int(row.step),
                    "val_loss": float(row.val_loss),
                    "lightning_log_version": version,
                }
            )
    if not history_rows:
        raise FileNotFoundError(f"No validation metrics found under {log_dir}.")
    history = pd.DataFrame(history_rows)
    return (
        history.sort_values(["epoch", "lightning_log_version"])
        .drop_duplicates("epoch", keep="last")
        .sort_values("epoch")
        .reset_index(drop=True)
    )


def log_week9_run(
    *,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> str:
    """Log the Optuna study and selected TFT checkpoint in one MLflow run."""
    required_paths = [TRIALS_PATH, BEST_TRIAL_PATH, CONFIG_PATH, CHECKPOINT_PATH]
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Required Week 8/9 artifacts are missing: "
            + ", ".join(str(path) for path in missing_paths)
        )

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    best_trial = json.loads(BEST_TRIAL_PATH.read_text(encoding="utf-8"))
    if config["checkpoint"] != CHECKPOINT_PATH.name:
        raise ValueError("Training configuration does not point to the best checkpoint.")
    if config["test_data_used"]:
        raise ValueError("Refusing to log: the Week 9 training config says test data was used.")

    trials = pd.read_csv(TRIALS_PATH)
    if trials.empty:
        raise ValueError("Optuna trial table is empty.")
    history = build_validation_history(MODEL_DIR / "logs")
    model = TemporalFusionTransformer.load_from_checkpoint(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(model, TemporalFusionTransformer):
        raise TypeError("Best checkpoint did not load as a TemporalFusionTransformer.")

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="week9-tft-final-training") as run:
        mlflow.set_tags(
            {
                "week": "9",
                "step": "3",
                "model_family": "TemporalFusionTransformer",
                "training_status": "early_stopped",
                "test_data_used": "false",
                "selected_by": "Week 8 validation PR-AUC",
                "imbalance_strategy": "SMOTE training data only",
            }
        )
        mlflow.log_params(
            {
                "attention_head_size": int(model.hparams.attention_head_size),
                "hidden_size": int(model.hparams.hidden_size),
                "dropout": float(model.hparams.dropout),
                "learning_rate": float(model.hparams.learning_rate),
                "hidden_continuous_size": int(model.hparams.hidden_continuous_size),
                "encoder_length": int(config["encoder_length"]),
                "batch_size": int(config["batch_size"]),
                "validation_batch_size": int(config["validation_batch_size"]),
                "max_epochs": int(config["max_epochs"]),
                "epochs_completed": int(config["epochs_completed"]),
                "early_stopping_patience": int(config["early_stopping_patience"]),
                "early_stopping_wait_count": int(config["early_stopping_wait_count"]),
                "seed": int(config["seed"]),
                "optuna_trials": int(best_trial["n_trials"]),
                "optuna_best_trial": int(best_trial["best_trial"]),
                "train_windows_after_smote": int(
                    config["training_data"]["train_windows"]
                ),
                "validation_windows": int(
                    config["training_data"]["validation_windows"]
                ),
                "train_positive_windows_after_smote": int(
                    config["training_data"]["train_positive"]
                ),
                "validation_positive_windows": int(
                    config["training_data"]["validation_positive"]
                ),
                "test_data_used": False,
            }
        )
        mlflow.log_metrics(
            {
                "optuna_best_validation_pr_auc": float(
                    best_trial["validation_pr_auc"]
                ),
                "tft_best_validation_loss": float(
                    config["best_checkpoint_score"]
                ),
                "tft_final_validation_loss": float(config["final_validation_loss"]),
                "tft_best_epoch": int(config["best_epoch"]),
                "tft_epochs_completed": int(config["epochs_completed"]),
                "tft_learning_rate_reduced": float(
                    config["learning_rate_reduced"]
                ),
            }
        )
        for row in trials.itertuples(index=False):
            values = row._asdict()
            number = int(values["number"])
            if pd.notna(values.get("validation_pr_auc")):
                mlflow.log_metric(
                    f"optuna_trial_{number:02d}_validation_pr_auc",
                    float(values["validation_pr_auc"]),
                )
            if pd.notna(values.get("user_attrs_val_loss")):
                mlflow.log_metric(
                    f"optuna_trial_{number:02d}_validation_loss",
                    float(values["user_attrs_val_loss"]),
                )

        mlflow.log_artifact(str(TRIALS_PATH), artifact_path="optuna")
        mlflow.log_artifact(str(BEST_TRIAL_PATH), artifact_path="optuna")
        mlflow.log_artifact(str(CONFIG_PATH), artifact_path="training")
        mlflow.log_artifact(str(CHECKPOINT_PATH), artifact_path="model")
        mlflow.log_artifact(str(MODEL_DIR / "last.ckpt"), artifact_path="training")

        history_path = MODEL_DIR / "week9_tft_validation_history.csv"
        history.to_csv(history_path, index=False)
        mlflow.log_artifact(str(history_path), artifact_path="training")

        for metrics_path in sorted((MODEL_DIR / "logs").glob("version_*/metrics.csv")):
            mlflow.log_artifact(
                str(metrics_path),
                artifact_path=f"lightning_logs/{metrics_path.parent.name}",
            )

        mlflow.pytorch.log_model(
            model,
            artifact_path="final_tft_model",
            code_paths=[str(PROJECT_ROOT / "src")],
            serialization_format="pickle",
        )
        run_id = run.info.run_id

    print(
        f"Logged Week 9 TFT training to MLflow: experiment={experiment_name!r}, "
        f"run_id={run_id}, tracking_uri={tracking_uri}",
        flush=True,
    )
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    args = parser.parse_args()
    log_week9_run(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
