"""Tune the Week 7 TFT with Optuna for Week 8, step 1.

Hyperparameters are selected with validation data only. The test split is never
loaded into a dataset here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.metrics import average_precision_score

from src.features.sequence_splits import (
    DEFAULT_SESSION_NAMES,
    SESSION_DIR,
    build_sequence_splits,
    load_and_prepare_sessions,
)
from src.features.tft_imbalance import smote_training_windows
from src.features.tft_model import build_tft_model
from src.features.tft_smoke import sample_indices
from src.features.tft_timeseries import (
    build_timeseries_dataset,
    collect_resampled_windows,
    collect_windows,
    validation_dataset,
)

OUTPUT_DIR = Path("experiments/eda")
TRIALS_PATH = OUTPUT_DIR / "week8_optuna_trials.csv"
BEST_PATH = OUTPUT_DIR / "week8_optuna_best.json"

SEARCH_SPACE = {
    "attention_head_size": [1, 2, 4],
    "hidden_size": [8, 16, 32, 64],
    "dropout": (0.05, 0.4),
    "learning_rate": (1e-4, 1e-2),
}
SEED = 42


def suggest_hyperparameters(trial: optuna.Trial) -> dict[str, Any]:
    """Sample the four hyperparameters required by the Week 8 roadmap."""
    return {
        "attention_head_size": trial.suggest_categorical(
            "attention_head_size", SEARCH_SPACE["attention_head_size"]
        ),
        "hidden_size": trial.suggest_categorical(
            "hidden_size", SEARCH_SPACE["hidden_size"]
        ),
        "dropout": trial.suggest_float("dropout", *SEARCH_SPACE["dropout"]),
        "learning_rate": trial.suggest_float(
            "learning_rate", *SEARCH_SPACE["learning_rate"], log=True
        ),
    }


def validation_average_precision(model, loader) -> float:
    """Average precision (PR-AUC) of predicted risk on validation windows."""
    predictions = model.predict(loader, mode="prediction", return_y=True)
    scores = predictions.output.detach().cpu().numpy().reshape(-1)
    targets = predictions.y[0].detach().cpu().numpy().reshape(-1)
    return float(average_precision_score(targets, scores))


def prepare_tuning_data(
    max_train_samples: int,
    max_validation_samples: int,
    batch_size: int,
):
    """Build SMOTE-resampled training and untouched validation loaders once."""
    pairs_by_session = load_and_prepare_sessions(
        [SESSION_DIR / name for name in DEFAULT_SESSION_NAMES]
    )
    split_datasets, _ = build_sequence_splits(pairs_by_session)
    feature_names = split_datasets["train"].datasets[0].windows.feature_names

    smote_training, _ = smote_training_windows(split_datasets["train"])
    train_labels = smote_training.tensors[1].cpu().numpy().astype(np.int8)
    train_indices = sample_indices(train_labels, max_train_samples, stratified=True)
    train_x, train_y = collect_resampled_windows(smote_training, train_indices)

    validation_labels = np.concatenate(
        [leaf.windows.y_future for leaf in split_datasets["validation"].datasets]
    )
    validation_indices = sample_indices(
        validation_labels, max_validation_samples, stratified=False, random_state=43
    )
    validation_x, validation_y = collect_windows(
        split_datasets["validation"], feature_names, validation_indices
    )
    if len(np.unique(validation_y)) < 2:
        raise ValueError(
            "Sampled validation windows need both classes; increase "
            "--max-validation-samples."
        )

    training_data = build_timeseries_dataset(train_x, train_y, feature_names)
    validation_data = validation_dataset(
        training_data, validation_x, validation_y, feature_names
    )
    training_loader = training_data.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    validation_loader = validation_data.to_dataloader(
        train=False, batch_size=batch_size * 4, num_workers=0
    )
    summary = {
        "train_windows": int(len(train_x)),
        "train_positive": int(train_y.sum()),
        "validation_windows": int(len(validation_x)),
        "validation_positive": int(validation_y.sum()),
    }
    return training_data, training_loader, validation_loader, summary


def run_tuning(
    n_trials: int = 12,
    max_epochs: int = 3,
    max_train_samples: int = 8000,
    max_validation_samples: int = 20000,
    batch_size: int = 64,
) -> optuna.Study:
    """Run an Optuna study maximising validation PR-AUC."""
    if n_trials < 1 or max_epochs < 1:
        raise ValueError("n_trials and max_epochs must be at least 1.")

    training_data, training_loader, validation_loader, data_summary = (
        prepare_tuning_data(max_train_samples, max_validation_samples, batch_size)
    )
    print(f"Tuning data: {data_summary}")

    def objective(trial: optuna.Trial) -> float:
        params = suggest_hyperparameters(trial)
        seed_everything(SEED, workers=True)
        model = build_tft_model(
            training_data,
            hidden_size=params["hidden_size"],
            attention_head_size=params["attention_head_size"],
            dropout=params["dropout"],
            hidden_continuous_size=max(4, params["hidden_size"] // 2),
            learning_rate=params["learning_rate"],
        )
        trainer = Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=max_epochs,
            num_sanity_val_steps=0,
            callbacks=[EarlyStopping(monitor="val_loss", patience=2, mode="min")],
            gradient_clip_val=0.1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
        )
        trainer.fit(
            model,
            train_dataloaders=training_loader,
            val_dataloaders=validation_loader,
        )
        trial.set_user_attr("val_loss", float(trainer.callback_metrics["val_loss"]))
        return validation_average_precision(model, validation_loader)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name="week8_tft_tuning",
    )
    study.optimize(objective, n_trials=n_trials)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials = trials.rename(columns={"value": "validation_pr_auc"})
    trials.to_csv(TRIALS_PATH, index=False)
    BEST_PATH.write_text(
        json.dumps(
            {
                "best_trial": study.best_trial.number,
                "validation_pr_auc": study.best_value,
                "best_params": study.best_params,
                "selection_metric": "validation PR-AUC (average precision)",
                "data": data_summary,
                "n_trials": n_trials,
                "max_epochs": max_epochs,
                "torch_version": torch.__version__,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Best validation PR-AUC={study.best_value:.4f}; params={study.best_params}")
    return study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--max-epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=8000)
    parser.add_argument("--max-validation-samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    run_tuning(
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
