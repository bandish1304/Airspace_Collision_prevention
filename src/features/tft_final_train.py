"""Train the selected Week 8 TFT configuration to convergence for Week 9."""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from torch import Tensor
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader, Dataset

from src.evaluation.splits import DEFAULT_OUTPUT_PATH, load_split
from src.features.build_pipeline import build_pair_features
from src.features.model_inputs import feature_names, prepare_model_inputs
from src.features.sequence_splits import (
    DEFAULT_SESSION_NAMES,
    SESSION_DIR,
)
from src.features.sequences import build_windows
from src.features.tft_dataset import TFTWindowDataset
from src.features.tft_imbalance import smote_training_windows
from src.features.tft_model import build_tft_model
from src.features.tft_smoke import sample_indices
from src.features.tft_timeseries import (
    build_timeseries_dataset,
    build_timeseries_frame,
    collect_windows,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BEST_CONFIG_PATH = PROJECT_ROOT / "experiments" / "eda" / "week8_optuna_best.json"
OUTPUT_DIR = PROJECT_ROOT / "models" / "week9_tft"
ENCODER_LENGTH = 30
TEMPLATE_TRAIN_SAMPLES = 8_000
DEFAULT_BATCH_SIZE = 128
DEFAULT_VALIDATION_BATCH_SIZE = 512
DEFAULT_MAX_EPOCHS = 100
DEFAULT_EARLY_STOPPING_PATIENCE = 8
SEED = 42


class TFTArrayDataset(Dataset[tuple[Tensor, Tensor]]):
    """Expose in-memory windows without expanding them into dataframe rows."""

    def __init__(self, windows: Tensor, targets: Tensor) -> None:
        if windows.ndim != 3 or targets.ndim != 1:
            raise ValueError("Expected windows (N, T, F) and scalar targets (N,).")
        if len(windows) != len(targets):
            raise ValueError("Windows and targets must have the same number of rows.")
        if not len(windows):
            raise ValueError("At least one window is required.")
        self.windows = windows
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.windows[index], self.targets[index]


class TFTArrayCollator:
    """Create the PyTorch Forecasting TFT batch structure from window tensors."""

    def __init__(self, training_dataset) -> None:
        self.training_dataset = training_dataset
        self.reals = list(training_dataset.reals)
        self.feature_names = list(training_dataset.time_varying_unknown_reals)
        self.feature_indices = {
            name: index for index, name in enumerate(self.feature_names)
        }
        self.encoder_length = int(training_dataset.max_encoder_length)
        if self.encoder_length != ENCODER_LENGTH:
            raise ValueError(
                f"Expected encoder length {ENCODER_LENGTH}; got {self.encoder_length}."
            )
        required_reals = {"time_idx", "relative_time_idx", *self.feature_names}
        if not required_reals.issubset(self.reals):
            raise ValueError("Training dataset real-variable schema is incomplete.")

    def __call__(self, samples: Sequence[tuple[Tensor, Tensor]]):
        windows = torch.stack([sample[0] for sample in samples]).cpu().numpy()
        targets = torch.stack([sample[1] for sample in samples]).to(torch.float32)
        batch_size = len(samples)
        encoder_cont = np.empty(
            (batch_size, self.encoder_length, len(self.reals)), dtype=np.float32
        )
        decoder_cont = np.empty((batch_size, 1, len(self.reals)), dtype=np.float32)

        for column, name in enumerate(self.reals):
            if name == "time_idx":
                encoder_values = np.broadcast_to(
                    np.arange(self.encoder_length, dtype=np.float32),
                    (batch_size, self.encoder_length),
                )
                decoder_values = np.full((batch_size, 1), self.encoder_length)
            elif name == "relative_time_idx":
                encoder_values = np.broadcast_to(
                    np.arange(-self.encoder_length, 0, dtype=np.float32)
                    / self.encoder_length,
                    (batch_size, self.encoder_length),
                )
                decoder_values = np.zeros((batch_size, 1), dtype=np.float32)
            else:
                encoder_values = windows[:, :, self.feature_indices[name]]
                decoder_values = np.zeros((batch_size, 1), dtype=np.float32)

            encoder_cont[:, :, column] = np.asarray(
                self.training_dataset.transform_values(name, encoder_values),
                dtype=np.float32,
            ).reshape(encoder_values.shape)
            decoder_cont[:, :, column] = np.asarray(
                self.training_dataset.transform_values(name, decoder_values),
                dtype=np.float32,
            ).reshape(decoder_values.shape)

        decoder_target = targets.reshape(batch_size, 1)
        x = {
            "encoder_cat": torch.empty(
                (batch_size, self.encoder_length, 0), dtype=torch.long
            ),
            "encoder_cont": torch.from_numpy(encoder_cont),
            "encoder_lengths": torch.full(
                (batch_size,), self.encoder_length, dtype=torch.long
            ),
            "encoder_target": torch.zeros(
                (batch_size, self.encoder_length), dtype=torch.float32
            ),
            "decoder_cat": torch.empty((batch_size, 1, 0), dtype=torch.long),
            "decoder_cont": torch.from_numpy(decoder_cont),
            "decoder_lengths": torch.ones(batch_size, dtype=torch.long),
            "decoder_target": decoder_target,
            "decoder_time_idx": torch.full(
                (batch_size, 1), self.encoder_length, dtype=torch.long
            ),
            "groups": torch.arange(batch_size, dtype=torch.long).unsqueeze(1).expand(
                batch_size, len(self.training_dataset.group_ids)
            ),
            "target_scale": torch.tensor(
                [0.0, 1.0], dtype=torch.float32
            ).expand(batch_size, 2),
        }
        return x, (decoder_target, None)


def build_train_validation_windows():
    """Load only the three training sessions and the validation session."""
    train_pieces: list[TFTWindowDataset] = []
    validation_pieces: list[TFTWindowDataset] = []
    train_names = DEFAULT_SESSION_NAMES[:3]
    validation_name = DEFAULT_SESSION_NAMES[3]
    required_columns = [
        "snapshot_time",
        "icao24_a",
        "icao24_b",
        *feature_names(),
        "collision_risk",
    ]

    for session_name, expected_split in [
        *((name, "train") for name in train_names),
        (validation_name, "validation"),
    ]:
        session_path = SESSION_DIR / session_name
        if not session_path.is_file():
            raise FileNotFoundError(f"Session data file not found: {session_path}")
        print(f"Preparing {expected_split} windows from {session_name}", flush=True)
        states = pd.read_csv(session_path)
        aircraft_codes = {
            identifier: code
            for code, identifier in enumerate(states["icao24"].dropna().unique())
        }
        unique_times = states["snapshot_time"].drop_duplicates()
        frame_parts = load_split(unique_times, DEFAULT_OUTPUT_PATH)
        part_by_time = dict(zip(unique_times.to_numpy(), frame_parts.to_numpy()))
        pair_chunks = []
        for snapshot_time, snapshot_states in states.groupby(
            "snapshot_time", sort=True
        ):
            if part_by_time.get(snapshot_time) != expected_split:
                continue
            pairs = prepare_model_inputs(build_pair_features(snapshot_states))
            if pairs.empty:
                continue
            pairs["icao24_a"] = pairs["icao24_a"].map(aircraft_codes).astype(np.int32)
            pairs["icao24_b"] = pairs["icao24_b"].map(aircraft_codes).astype(np.int32)
            selected = pairs[required_columns].copy()
            selected["snapshot_time"] = selected["snapshot_time"].astype(np.int64)
            selected[["icao24_a", "icao24_b"]] = selected[
                ["icao24_a", "icao24_b"]
            ].astype(np.int32)
            selected[feature_names()] = selected[feature_names()].astype(np.float32)
            selected["collision_risk"] = selected["collision_risk"].astype(np.int8)
            pair_chunks.append(selected)
        del states, aircraft_codes, frame_parts, part_by_time
        if not pair_chunks:
            raise ValueError(
                f"No pair observations for {expected_split} in session {session_name}."
            )
        selected = pd.concat(pair_chunks, ignore_index=True, copy=False)
        del pair_chunks
        windows = build_windows(selected, session=session_name)
        if not len(windows):
            raise ValueError(
                f"No windows for {expected_split} in session {session_name}."
            )
        dataset = TFTWindowDataset(windows)
        if expected_split == "train":
            train_pieces.append(dataset)
        else:
            validation_pieces.append(dataset)
        print(
            f"  pair_observations={len(selected):,} windows={len(windows):,} "
            f"positive_windows={int(windows.y_future.sum()):,}",
            flush=True,
        )
        del selected, windows, dataset
        gc.collect()

    return train_pieces, validation_pieces


def build_full_training_data():
    """Resample training windows and materialize the untouched validation set."""
    train_pieces, validation_pieces = build_train_validation_windows()
    feature_names = train_pieces[0].windows.feature_names
    train_dataset = torch.utils.data.ConcatDataset(train_pieces)
    validation_dataset = torch.utils.data.ConcatDataset(validation_pieces)

    smote_training, smote_metadata = smote_training_windows(train_dataset)
    train_windows, train_targets = smote_training.tensors
    del train_dataset, train_pieces
    gc.collect()
    validation_targets = np.concatenate(
        [leaf.windows.y_future for leaf in validation_pieces]
    ).astype(np.float32, copy=False)
    validation_windows, validation_targets = collect_windows(
        validation_dataset,
        feature_names,
        np.arange(len(validation_dataset), dtype=np.int64),
    )
    del validation_dataset, validation_pieces
    del smote_training
    gc.collect()

    if len(np.unique(validation_targets)) != 2:
        raise ValueError("Full validation data must contain both target classes.")
    if not torch.isfinite(train_windows).all() or not np.isfinite(validation_windows).all():
        raise ValueError("Training and validation windows must contain finite values.")
    summary = {
        "train_windows": int(len(train_targets)),
        "train_positive": int(train_targets.sum()),
        "validation_windows": int(len(validation_targets)),
        "validation_positive": int(validation_targets.sum()),
        "smote": smote_metadata,
    }
    return (
        train_windows,
        train_targets,
        torch.from_numpy(validation_windows),
        torch.from_numpy(validation_targets),
        feature_names,
        summary,
    )


def build_template_from_checkpoint(
    checkpoint_path: Path,
    sample_windows: Tensor,
    sample_targets: Tensor,
    feature_names: list[str],
) -> TimeSeriesDataSet:
    """Recreate the exact training transforms stored in a TFT checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        dataset_parameters = checkpoint["hyper_parameters"]["dataset_parameters"]
    except KeyError as exc:
        raise ValueError(
            f"Checkpoint does not contain PyTorch Forecasting dataset parameters: "
            f"{checkpoint_path}"
        ) from exc
    return TimeSeriesDataSet.from_parameters(
        dataset_parameters,
        build_timeseries_frame(
            sample_windows.cpu().numpy(),
            sample_targets.cpu().numpy(),
            feature_names,
        ),
        predict=False,
        stop_randomization=True,
    )


def checkpoint_completed_epochs(checkpoint_path: Path) -> int:
    """Return the number of completed epochs recorded in a Lightning checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"Checkpoint has an invalid epoch value: {checkpoint_path}")
    return epoch + 1


def run_final_training(
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    validation_batch_size: int = DEFAULT_VALIDATION_BATCH_SIZE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    resume_from: Path | None = None,
) -> None:
    """Fit the selected TFT, optionally resuming for a number of more epochs."""
    if min(batch_size, validation_batch_size, max_epochs, early_stopping_patience) < 1:
        raise ValueError("Batch sizes, max_epochs, and patience must be positive.")
    if not BEST_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Week 8 best-parameter file not found: {BEST_CONFIG_PATH}")
    if resume_from is not None and not resume_from.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
    if resume_from is None and OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Training output already exists: {OUTPUT_DIR}. Move it or choose a "
            "different output directory before starting another run."
        )

    best_config = json.loads(BEST_CONFIG_PATH.read_text(encoding="utf-8"))
    params = best_config["best_params"]
    seed_everything(SEED, workers=True)
    (
        train_windows,
        train_targets,
        validation_windows,
        validation_targets,
        feature_names,
        data_summary,
    ) = build_full_training_data()

    template_indices = sample_indices(
        train_targets.cpu().numpy().astype(np.int8),
        min(TEMPLATE_TRAIN_SAMPLES, len(train_targets)),
        stratified=True,
    )
    if resume_from is None:
        training_template = build_timeseries_dataset(
            train_windows[template_indices].cpu().numpy(),
            train_targets[template_indices].cpu().numpy(),
            feature_names,
        )
    else:
        training_template = build_template_from_checkpoint(
            resume_from,
            train_windows[template_indices],
            train_targets[template_indices],
            feature_names,
        )
    collator = TFTArrayCollator(training_template)
    training_loader = DataLoader(
        TFTArrayDataset(train_windows, train_targets),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        TFTArrayDataset(validation_windows, validation_targets),
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    model = build_tft_model(
        training_template,
        hidden_size=int(params["hidden_size"]),
        attention_head_size=int(params["attention_head_size"]),
        dropout=float(params["dropout"]),
        hidden_continuous_size=max(4, int(params["hidden_size"]) // 2),
        learning_rate=float(params["learning_rate"]),
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename="week9_tft_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    effective_max_epochs = max_epochs
    if resume_from is not None:
        effective_max_epochs += checkpoint_completed_epochs(resume_from)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=early_stopping_patience,
        min_delta=1e-4,
        verbose=True,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=effective_max_epochs,
        callbacks=[
            early_stopping,
            checkpoint,
            LearningRateMonitor(logging_interval="epoch"),
        ],
        logger=CSVLogger(save_dir=str(OUTPUT_DIR), name="logs"),
        gradient_clip_val=0.1,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        enable_model_summary=False,
        deterministic=True,
    )
    trainer.fit(
        model,
        train_dataloaders=training_loader,
        val_dataloaders=validation_loader,
        ckpt_path=str(resume_from) if resume_from is not None else None,
        weights_only=False if resume_from is not None else None,
    )
    if checkpoint.best_model_path == "":
        raise RuntimeError("Training ended without saving a best validation checkpoint.")
    if checkpoint.last_model_path == "":
        raise RuntimeError("Training ended without saving the latest checkpoint.")

    completed_epochs = (
        early_stopping.stopped_epoch + 1
        if early_stopping.stopped_epoch > 0
        else checkpoint_completed_epochs(Path(checkpoint.last_model_path))
    )
    final_checkpoint = torch.load(
        checkpoint.last_model_path,
        map_location="cpu",
        weights_only=False,
    )
    final_learning_rate = float(
        final_checkpoint["optimizer_states"][0]["param_groups"][0]["lr"]
    )

    config = {
        "selected_week8_config": best_config,
        "training_data": data_summary,
        "feature_names": feature_names,
        "encoder_length": ENCODER_LENGTH,
        "batch_size": batch_size,
        "validation_batch_size": validation_batch_size,
        "max_epochs": effective_max_epochs,
        "additional_epochs_requested": max_epochs if resume_from is not None else None,
        "epochs_completed": completed_epochs,
        "best_epoch": checkpoint_completed_epochs(
            Path(checkpoint.best_model_path)
        )
        - 1,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_monitor": "val_loss",
        "early_stopping_wait_count": early_stopping.wait_count,
        "early_stopping_triggered": early_stopping.stopped_epoch > 0,
        "learning_rate_scheduler": "ReduceLROnPlateau (patience 4)",
        "scheduler_monitor": "val_loss",
        "learning_rate_initial": float(params["learning_rate"]),
        "learning_rate_final": final_learning_rate,
        "learning_rate_reduced": final_learning_rate < float(params["learning_rate"]),
        "seed": SEED,
        "checkpoint": Path(checkpoint.best_model_path).name,
        "best_checkpoint_score": float(checkpoint.best_model_score),
        "final_validation_loss": float(trainer.callback_metrics["val_loss"]),
        "test_data_used": False,
        "resumed_from": str(resume_from) if resume_from is not None else None,
    }
    (OUTPUT_DIR / "week9_tft_training_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Training finished after {trainer.current_epoch + 1} epoch(s); "
        f"best val_loss={float(checkpoint.best_model_score):.6f}; "
        f"checkpoint={checkpoint.best_model_path}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=DEFAULT_VALIDATION_BATCH_SIZE,
    )
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume a saved Lightning checkpoint for --max-epochs additional epochs.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
    )
    args = parser.parse_args()
    run_final_training(
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
