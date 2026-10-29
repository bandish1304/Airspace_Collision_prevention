"""Run a short real-data TFT training smoke test for Week 7, step 7."""

from __future__ import annotations

import argparse

import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping

from src.features.sequence_splits import (
    DEFAULT_SESSION_NAMES,
    SESSION_DIR,
    build_sequence_splits,
    load_and_prepare_sessions,
)
from src.features.tft_imbalance import smote_training_windows
from src.features.tft_model import build_tft_model
from src.features.tft_timeseries import (
    build_timeseries_dataset,
    collect_resampled_windows,
    collect_windows,
    validation_dataset,
)

DEFAULT_MAX_TRAIN_SAMPLES = 512
DEFAULT_MAX_VALIDATION_SAMPLES = 256
DEFAULT_BATCH_SIZE = 32


def sample_indices(
    labels: np.ndarray,
    max_samples: int,
    *,
    stratified: bool,
    random_state: int = 42,
) -> np.ndarray:
    """Select a deterministic bounded smoke-test sample."""
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1.")
    rng = np.random.default_rng(random_state)
    if len(labels) <= max_samples:
        return np.arange(len(labels), dtype=np.int64)
    if not stratified:
        return np.sort(rng.choice(len(labels), size=max_samples, replace=False))

    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
        raise ValueError("Stratified training sample requires both target classes.")
    positive_count = min(len(positive), max(1, max_samples // 2))
    negative_count = min(len(negative), max_samples - positive_count)
    selected = np.concatenate(
        [
            rng.choice(positive, size=positive_count, replace=False),
            rng.choice(negative, size=negative_count, replace=False),
        ]
    )
    return np.sort(selected)


def run_smoke_test(
    max_train_samples: int = DEFAULT_MAX_TRAIN_SAMPLES,
    max_validation_samples: int = DEFAULT_MAX_VALIDATION_SAMPLES,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Train one short epoch on SMOTE training windows and real validation data."""
    if max_train_samples < 2 or max_validation_samples < 1 or batch_size < 1:
        raise ValueError("Smoke-test sample sizes and batch_size must be positive.")

    pairs_by_session = load_and_prepare_sessions(
        [SESSION_DIR / name for name in DEFAULT_SESSION_NAMES]
    )
    split_datasets, _ = build_sequence_splits(pairs_by_session)

    feature_names = split_datasets["train"].datasets[0].windows.feature_names
    smote_training, smote_metadata = smote_training_windows(split_datasets["train"])

    train_labels = smote_training.tensors[1].cpu().numpy().astype(np.int8)
    train_indices = sample_indices(
        train_labels, max_train_samples, stratified=True
    )
    train_x, train_y = collect_resampled_windows(
        smote_training, train_indices
    )

    validation_labels = np.concatenate(
        [leaf.windows.y_future for leaf in split_datasets["validation"].datasets]
    )
    validation_indices = sample_indices(
        validation_labels,
        max_validation_samples,
        stratified=False,
        random_state=43,
    )
    validation_x, validation_y = collect_windows(
        split_datasets["validation"], feature_names, validation_indices
    )
    if len(np.unique(validation_y)) < 2:
        raise ValueError(
            "The sampled validation windows do not contain both classes. "
            "Increase --max-validation-samples."
        )

    training_data = build_timeseries_dataset(train_x, train_y, feature_names)
    validation_data = validation_dataset(
        training_data, validation_x, validation_y, feature_names
    )
    training_loader = training_data.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    validation_loader = validation_data.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )

    model = build_tft_model(
        training_data,
        hidden_size=8,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=4,
        learning_rate=1e-3,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=5,
        num_sanity_val_steps=0,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=1, mode="min")
        ],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
    )
    trainer.fit(model, train_dataloaders=training_loader, val_dataloaders=validation_loader)

    train_batches = min(len(training_loader), 10)
    validation_batches = min(len(validation_loader), 5)
    if not trainer.callback_metrics.get("train_loss"):
        raise RuntimeError("TFT smoke test completed without recording training loss.")
    if not trainer.callback_metrics.get("val_loss"):
        raise RuntimeError("TFT smoke test completed without recording validation loss.")
    print(
        "TFT smoke test passed: "
        f"SMOTE training windows={smote_metadata['resampled_windows']:,}; "
        f"sampled train={len(train_x):,} ({int(train_y.sum())} positive); "
        f"sampled validation={len(validation_x):,} "
        f"({int(validation_y.sum())} positive); "
        f"train_batches={train_batches}; validation_batches={validation_batches}; "
        f"train_loss={float(trainer.callback_metrics['train_loss']):.6f}; "
        f"val_loss={float(trainer.callback_metrics['val_loss']):.6f}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-train-samples", type=int, default=DEFAULT_MAX_TRAIN_SAMPLES)
    parser.add_argument(
        "--max-validation-samples",
        type=int,
        default=DEFAULT_MAX_VALIDATION_SAMPLES,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    run_smoke_test(
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
