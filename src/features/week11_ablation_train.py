"""Retrain the Week 11 TFT ablation models with shared data and training settings."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from src.features.tft_final_train import (
    ENCODER_LENGTH,
    PROJECT_ROOT,
    SEED,
    TEMPLATE_TRAIN_SAMPLES,
    TFTArrayCollator,
    TFTArrayDataset,
    build_full_training_data,
    checkpoint_completed_epochs,
)
from src.features.tft_model import build_tft_model
from src.features.tft_smoke import sample_indices
from src.features.tft_timeseries import (
    SHORT_ENCODER_LENGTH,
    build_timeseries_dataset,
    shorten_encoder_windows,
)

TRAINING_CONFIG_PATH = PROJECT_ROOT / "models" / "week9_tft" / (
    "week9_tft_training_config.json"
)
OUTPUT_DIR = PROJECT_ROOT / "models" / "week11_ablations"


class EpochProgressCallback(Callback):
    """Persist completed-epoch progress so an interrupted run can be resumed."""

    def __init__(
        self,
        path: Path,
        variant_name: str,
        max_epochs: int,
        pause_after_epoch: bool,
    ) -> None:
        self.path = path
        self.variant_name = variant_name
        self.max_epochs = max_epochs
        self.pause_after_epoch = pause_after_epoch

    def on_validation_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        if trainer.sanity_checking:
            return
        validation_loss = trainer.callback_metrics.get("val_loss")
        payload: dict[str, object] = {
            "variant": self.variant_name,
            "completed_epochs": min(trainer.current_epoch + 1, self.max_epochs),
            "max_epochs": self.max_epochs,
            "validation_loss": (
                float(validation_loss.detach().cpu())
                if validation_loss is not None
                else None
            ),
            "status": "training",
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(self.path)
        if self.pause_after_epoch and trainer.current_epoch + 1 < self.max_epochs:
            trainer.should_stop = True

@dataclass(frozen=True)
class AblationVariant:
    name: str
    encoder_length: int
    disable_learned_variable_selection: bool = False


VARIANTS = (
    AblationVariant("full_tft", encoder_length=ENCODER_LENGTH),
    AblationVariant(
        "no_learned_variable_selection",
        encoder_length=ENCODER_LENGTH,
        disable_learned_variable_selection=True,
    ),
    AblationVariant("encoder_15", encoder_length=SHORT_ENCODER_LENGTH),
)


def load_training_settings(
    config_path: Path = TRAINING_CONFIG_PATH,
) -> dict[str, int]:
    """Reuse the completed Week 9 run's epoch and batch-size budget."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Week 9 training config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = {
        "max_epochs": int(config["epochs_completed"]),
        "batch_size": int(config["batch_size"]),
        "validation_batch_size": int(config["validation_batch_size"]),
        "early_stopping_patience": int(config["early_stopping_patience"]),
    }
    if min(settings.values()) < 1:
        raise ValueError(f"Week 9 training settings must be positive: {settings}")
    return settings


def prepare_variant_windows(
    variant: AblationVariant,
    train_windows: torch.Tensor,
    validation_windows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the variant's encoder length identically to train and validation."""
    expected_length = ENCODER_LENGTH
    if train_windows.ndim != 3 or validation_windows.ndim != 3:
        raise ValueError("Train and validation windows must have shape (N, T, F).")
    if train_windows.shape[1] != expected_length:
        raise ValueError(
            f"Expected {expected_length}-frame training windows; "
            f"got {train_windows.shape[1]}."
        )
    if validation_windows.shape[1] != expected_length:
        raise ValueError(
            f"Expected {expected_length}-frame validation windows; "
            f"got {validation_windows.shape[1]}."
        )
    if variant.encoder_length > expected_length:
        raise ValueError("Variant encoder length cannot exceed the baseline length.")

    if variant.encoder_length == expected_length:
        return train_windows, validation_windows
    train_array = shorten_encoder_windows(
        train_windows.cpu().numpy(), variant.encoder_length
    )
    validation_array = shorten_encoder_windows(
        validation_windows.cpu().numpy(), variant.encoder_length
    )
    return torch.from_numpy(train_array), torch.from_numpy(validation_array)


def train_variant(
    variant: AblationVariant,
    *,
    train_windows: torch.Tensor,
    train_targets: torch.Tensor,
    validation_windows: torch.Tensor,
    validation_targets: torch.Tensor,
    feature_names: list[str],
    data_summary: dict[str, object],
    best_config: dict[str, object],
    training_settings: dict[str, int],
    output_root: Path = OUTPUT_DIR,
    resume: bool = False,
    pause_after_epoch: bool = False,
) -> dict[str, object]:
    """Train and save one TFT variant without changing its data split."""
    output_path = output_root / variant.name
    progress_path = output_path / "progress.json"
    checkpoint_path = output_path / "last.ckpt"
    resume_checkpoint: Path | None = None
    if output_path.exists() and any(output_path.iterdir()) and not resume:
        raise FileExistsError(
            f"Output already exists: {output_path}. Choose a new output root "
            "or pass --resume to continue its saved checkpoint."
        )
    if resume and output_path.exists() and any(output_path.iterdir()):
        if (output_path / "training_config.json").is_file():
            raise FileExistsError(
                f"{variant.name} is already complete at {output_path}; "
                "completed variants are not retrained on resume."
            )
        if checkpoint_path.is_file():
            resume_checkpoint = checkpoint_path
        else:
            print(
                f"No epoch checkpoint exists for {variant.name}; restarting this "
                "variant from epoch 1.",
                flush=True,
            )

    variant_train, variant_validation = prepare_variant_windows(
        variant, train_windows, validation_windows
    )
    targets = train_targets.cpu().numpy().astype(np.int8)
    template_indices = sample_indices(
        targets,
        min(TEMPLATE_TRAIN_SAMPLES, len(train_targets)),
        stratified=True,
    )
    training_template = build_timeseries_dataset(
        variant_train[template_indices].cpu().numpy(),
        train_targets[template_indices].cpu().numpy(),
        feature_names,
    )
    collator = TFTArrayCollator(training_template)
    training_loader = DataLoader(
        TFTArrayDataset(variant_train, train_targets),
        batch_size=training_settings["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        TFTArrayDataset(variant_validation, validation_targets),
        batch_size=training_settings["validation_batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    params = best_config["best_params"]
    seed_everything(SEED, workers=True)
    model = build_tft_model(
        training_template,
        hidden_size=int(params["hidden_size"]),
        attention_head_size=int(params["attention_head_size"]),
        dropout=float(params["dropout"]),
        hidden_continuous_size=max(4, int(params["hidden_size"]) // 2),
        learning_rate=float(params["learning_rate"]),
        disable_learned_variable_selection=(
            variant.disable_learned_variable_selection
        ),
    )

    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=output_path,
        filename=f"{variant.name}_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    progress_callback = EpochProgressCallback(
        progress_path,
        variant.name,
        training_settings["max_epochs"],
        pause_after_epoch,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=training_settings["early_stopping_patience"],
        min_delta=1e-4,
        verbose=True,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=training_settings["max_epochs"],
        callbacks=[
            early_stopping,
            checkpoint,
            LearningRateMonitor(logging_interval="epoch"),
            progress_callback,
        ],
        logger=CSVLogger(save_dir=str(output_path), name="logs"),
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
        ckpt_path=str(resume_checkpoint) if resume_checkpoint is not None else None,
        weights_only=False if resume_checkpoint is not None else None,
    )
    if not checkpoint.best_model_path or not checkpoint.last_model_path:
        raise RuntimeError(f"Training did not save checkpoints for {variant.name}.")

    progress_data = json.loads(progress_path.read_text(encoding="utf-8"))
    completed_epochs = int(progress_data["completed_epochs"])
    is_complete = (
        completed_epochs >= training_settings["max_epochs"]
        or early_stopping.stopped_epoch > 0
    )
    final_validation_loss = trainer.callback_metrics.get("val_loss")
    if final_validation_loss is None:
        raise RuntimeError(
            f"Training did not record validation loss for {variant.name}."
        )
    result: dict[str, object] = {
        "variant": variant.name,
        "encoder_length": variant.encoder_length,
        "disable_learned_variable_selection": (
            variant.disable_learned_variable_selection
        ),
        "selected_week8_config": best_config,
        "training_data": data_summary,
        "feature_names": feature_names,
        "training_settings": training_settings,
        "seed": SEED,
        "epochs_completed": completed_epochs,
        "best_epoch": checkpoint_completed_epochs(Path(checkpoint.best_model_path))
        - 1,
        "best_val_loss": float(checkpoint.best_model_score),
        "final_val_loss": float(final_validation_loss.detach().cpu()),
        "checkpoint": Path(checkpoint.best_model_path).name,
        "test_data_used": False,
    }
    progress = {
        "variant": variant.name,
        "completed_epochs": completed_epochs,
        "max_epochs": training_settings["max_epochs"],
        "validation_loss": result["final_val_loss"],
        "status": "completed" if is_complete else "paused",
    }
    progress_path.write_text(
        json.dumps(progress, indent=2) + "\n", encoding="utf-8"
    )
    result["status"] = progress["status"]
    if is_complete:
        (output_path / "training_config.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    run_status = "Finished" if is_complete else "Paused"
    print(
        f"{run_status} {variant.name} after {completed_epochs} epoch(s): "
        f"best val_loss={result['best_val_loss']:.6f}; "
        f"checkpoint={checkpoint.best_model_path}",
        flush=True,
    )
    del model, trainer, training_loader, validation_loader, collator
    gc.collect()
    return result


def run_ablation_training(
    *,
    variants: tuple[AblationVariant, ...] = VARIANTS,
    output_root: Path = OUTPUT_DIR,
    max_epochs: int | None = None,
    resume: bool = False,
    pause_after_epoch: bool = False,
) -> list[dict[str, object]]:
    """Retrain the baseline and implementable ablations on shared splits."""
    if not variants:
        raise ValueError("At least one ablation variant must be selected.")
    if (output_root / "training_summary.json").exists() and not resume:
        raise FileExistsError(
            f"Training summary already exists: {output_root / 'training_summary.json'}. "
            "Choose a new output root or pass --resume."
        )
    if not TRAINING_CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Week 9 training settings not found: {TRAINING_CONFIG_PATH}"
        )
    settings = load_training_settings()
    if max_epochs is not None:
        if max_epochs < 1:
            raise ValueError("max_epochs must be at least 1.")
        settings["max_epochs"] = max_epochs

    for variant in variants:
        output_path = output_root / variant.name
        if output_path.exists() and any(output_path.iterdir()) and not resume:
            raise FileExistsError(
                f"Output already exists: {output_path}. Choose a new output root "
                "or pass --resume."
            )

    best_config_path = PROJECT_ROOT / "experiments" / "eda" / "week8_optuna_best.json"
    if not best_config_path.is_file():
        raise FileNotFoundError(f"Week 8 best-parameter file not found: {best_config_path}")
    best_config = json.loads(best_config_path.read_text(encoding="utf-8"))
    (
        train_windows,
        train_targets,
        validation_windows,
        validation_targets,
        feature_names,
        data_summary,
    ) = build_full_training_data()

    results = []
    for variant in variants:
        output_path = output_root / variant.name
        completed_config = output_path / "training_config.json"
        if resume and completed_config.is_file():
            print(f"Skipping completed Week 11 variant: {variant.name}", flush=True)
            results.append(json.loads(completed_config.read_text(encoding="utf-8")))
            continue
        print(f"Training Week 11 variant: {variant.name}", flush=True)
        results.append(
            train_variant(
                variant,
                train_windows=train_windows,
                train_targets=train_targets,
                validation_windows=validation_windows,
                validation_targets=validation_targets,
                feature_names=feature_names,
                data_summary=data_summary,
                best_config=best_config,
                training_settings=settings,
                output_root=output_root,
                resume=resume,
                pause_after_epoch=pause_after_epoch,
            )
        )
        if pause_after_epoch:
            if results[-1]["status"] == "paused":
                print(
                    f"Paused {variant.name} after epoch "
                    f"{results[-1]['epochs_completed']}; resume to continue.",
                    flush=True,
                )
            break
    if not pause_after_epoch and all(
        result["status"] == "completed" for result in results
    ):
        (output_root / "training_summary.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override the Week 9 epoch budget; use only for smoke runs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_DIR,
        help="New output directory; existing variant runs are never overwritten.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume incomplete variants from their last checkpoint and skip completed ones.",
    )
    parser.add_argument(
        "--pause-after-epoch",
        action="store_true",
        help="Save a resumable checkpoint and stop after the next completed epoch.",
    )
    args = parser.parse_args()
    run_ablation_training(
        output_root=args.output_root,
        max_epochs=args.max_epochs,
        resume=args.resume,
        pause_after_epoch=args.pause_after_epoch,
    )


if __name__ == "__main__":
    main()
