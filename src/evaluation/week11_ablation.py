"""Evaluate the trained Week 11 TFT ablations on validation and held-out test data."""

from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.evaluation.harness import build_evaluation_harness
from src.evaluation.threshold import fit_threshold
from src.evaluation.week10_significance import (
    BOOTSTRAP_RESAMPLES,
    CONFIDENCE,
    METRICS as BOOTSTRAP_METRICS,
    paired_cluster_bootstrap,
)
from src.evaluation.week10_test_set import (
    PROJECT_ROOT,
    build_session_windows,
)
from src.features.sequence_splits import DEFAULT_SESSION_NAMES
from src.features.sequences import WindowSet
from src.features.tft_final_train import (
    TFTArrayCollator,
    build_template_from_checkpoint,
)
from src.features.tft_model import TFTWithoutLearnedVariableSelection
from src.features.tft_smoke import sample_indices
from src.features.tft_timeseries import shorten_encoder_windows

MODEL_ROOT = PROJECT_ROOT / "models" / "week11_ablations"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "week11_ablations"
EDA_DIR = PROJECT_ROOT / "experiments" / "eda"
VALIDATION_SESSION = DEFAULT_SESSION_NAMES[3]
TEST_SESSION = DEFAULT_SESSION_NAMES[-1]
TEMPLATE_SAMPLES = 2_000
SCORING_BATCH_SIZE = 1_024
PAIR_KEY = ["snapshot_time", "icao24_a", "icao24_b"]
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


@dataclass(frozen=True)
class AblationVariant:
    name: str
    encoder_length: int
    checkpoint: Path
    disable_learned_variable_selection: bool = False


VARIANTS = (
    AblationVariant(
        "full_tft",
        30,
        MODEL_ROOT / "full_tft" / "full_tft_best.ckpt",
    ),
    AblationVariant(
        "no_learned_variable_selection",
        30,
        MODEL_ROOT / "no_learned_variable_selection"
        / "no_learned_variable_selection_best.ckpt",
        disable_learned_variable_selection=True,
    ),
    AblationVariant(
        "encoder_15",
        15,
        MODEL_ROOT / "encoder_15" / "encoder_15_best.ckpt",
    ),
)


class AblationWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """Gather encoder windows at the variant's trained sequence length."""

    def __init__(self, windows: WindowSet, encoder_length: int) -> None:
        if not 1 <= encoder_length <= windows.window_frames:
            raise ValueError(
                f"Encoder length must be in [1, {windows.window_frames}]."
            )
        self.windows = windows
        self.encoder_length = encoder_length

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        window = self.windows.gather(np.array([index]))[0]
        window = shorten_encoder_windows(window[np.newaxis, ...], self.encoder_length)[0]
        target = float(self.windows.y_future[index])
        return torch.from_numpy(window), torch.tensor(target, dtype=torch.float32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction_paths(split: str, variant: AblationVariant) -> tuple[Path, Path]:
    csv_path = OUTPUT_DIR / f"{split}_{variant.name}_predictions.csv"
    return csv_path, csv_path.with_suffix(".metadata.json")


def _validate_completed_variants() -> None:
    for variant in VARIANTS:
        config_path = MODEL_ROOT / variant.name / "training_config.json"
        if not variant.checkpoint.is_file() or not config_path.is_file():
            raise FileNotFoundError(
                f"Completed model artifacts are missing for {variant.name}."
            )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("status") != "completed":
            raise ValueError(f"{variant.name} is not marked completed.")
        if int(config.get("encoder_length", -1)) != variant.encoder_length:
            raise ValueError(f"Unexpected encoder length in {config_path}.")
        expected_disable = variant.disable_learned_variable_selection
        if bool(config.get("disable_learned_variable_selection")) != expected_disable:
            raise ValueError(f"Unexpected model variant recorded in {config_path}.")


def _read_cached_predictions(
    split: str,
    session_name: str,
    variant: AblationVariant,
    expected: pd.DataFrame,
) -> pd.DataFrame | None:
    csv_path, metadata_path = _prediction_paths(split, variant)
    if not csv_path.exists() and not metadata_path.exists():
        return None
    if not csv_path.is_file() or not metadata_path.is_file():
        raise ValueError(
            f"Incomplete saved predictions for {split}/{variant.name}; "
            "both CSV and metadata must exist."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "split": split,
        "session": session_name,
        "variant": variant.name,
        "encoder_length": variant.encoder_length,
        "checkpoint_sha256": _sha256(variant.checkpoint),
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError(
            f"Saved {split}/{variant.name} predictions do not match the current "
            "session or checkpoint; refusing to reuse or silently rescore them."
        )
    saved = pd.read_csv(csv_path)
    expected_keys = expected[PAIR_KEY + ["future_conflict"]].reset_index(drop=True)
    saved_keys = saved[PAIR_KEY + ["future_conflict"]].reset_index(drop=True)
    if not saved_keys.equals(expected_keys):
        raise ValueError(f"Saved {split}/{variant.name} rows do not match the split.")
    probabilities = saved["probability"].to_numpy(dtype=float)
    if len(saved) != int(metadata.get("rows", -1)) or not np.isfinite(probabilities).all():
        raise ValueError(f"Saved {split}/{variant.name} scores are invalid.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"Saved {split}/{variant.name} probabilities are out of range.")
    return saved


def _save_predictions(
    split: str,
    session_name: str,
    variant: AblationVariant,
    predictions: pd.DataFrame,
) -> None:
    csv_path, metadata_path = _prediction_paths(split, variant)
    metadata = {
        "split": split,
        "session": session_name,
        "variant": variant.name,
        "encoder_length": variant.encoder_length,
        "checkpoint_sha256": _sha256(variant.checkpoint),
        "rows": len(predictions),
        "test_scored_once": split == "test",
    }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_temp = csv_path.with_suffix(".tmp")
    metadata_temp = metadata_path.with_suffix(".tmp")
    predictions.to_csv(csv_temp, index=False)
    metadata_temp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    csv_temp.replace(csv_path)
    metadata_temp.replace(metadata_path)


def _score_windows(windows: WindowSet, variant: AblationVariant) -> np.ndarray:
    targets = windows.y_future.astype(np.int8)
    sample = sample_indices(targets, TEMPLATE_SAMPLES, stratified=True)
    sample_windows = shorten_encoder_windows(
        windows.gather(sample), variant.encoder_length
    )
    template = build_template_from_checkpoint(
        variant.checkpoint,
        torch.from_numpy(sample_windows),
        torch.from_numpy(targets[sample].astype(np.float32)),
        windows.feature_names,
    )

    loader = DataLoader(
        AblationWindowDataset(windows, variant.encoder_length),
        batch_size=SCORING_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=TFTArrayCollator(template),
    )
    model_class = (
        TFTWithoutLearnedVariableSelection
        if variant.disable_learned_variable_selection
        else TemporalFusionTransformer
    )
    model = model_class.load_from_checkpoint(
        variant.checkpoint, map_location="cpu", weights_only=False
    )
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for batch_number, (x, _) in enumerate(loader, start=1):
            prediction = model.loss.to_prediction(model(x)["prediction"])
            probabilities.append(prediction.reshape(-1).cpu().numpy())
            if batch_number % 25 == 0:
                print(
                    f"  {variant.name}: scored "
                    f"{min(batch_number * SCORING_BATCH_SIZE, len(windows)):,}/"
                    f"{len(windows):,} windows",
                    flush=True,
                )
    scores = np.concatenate(probabilities)
    if len(scores) != len(windows) or not np.isfinite(scores).all():
        raise ValueError(f"{variant.name} produced missing or non-finite scores.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError(f"{variant.name} produced scores outside [0, 1].")
    return scores


def score_split(split: str, session_name: str) -> dict[str, pd.DataFrame]:
    """Score or reuse all completed variants on one shared chronological split."""
    windows, metadata = build_session_windows(session_name, split)
    expected = metadata.copy()
    expected["future_conflict"] = expected.pop("tft_y_future").astype(int)
    predictions: dict[str, pd.DataFrame] = {}
    pending = []
    for variant in VARIANTS:
        cached = _read_cached_predictions(split, session_name, variant, expected)
        if cached is None:
            pending.append(variant)
        else:
            predictions[variant.name] = cached

    for variant in pending:
        print(
            f"Scoring {len(windows):,} {split} windows with {variant.name} "
            f"(encoder {variant.encoder_length})...",
            flush=True,
        )
        result = expected.copy()
        result["probability"] = _score_windows(windows, variant)
        _save_predictions(split, session_name, variant, result)
        predictions[variant.name] = result

    if len(predictions) != len(VARIANTS):
        raise RuntimeError(f"Not all Week 11 variants were scored on {split}.")
    del windows, metadata, expected
    gc.collect()
    return predictions


def _metric_row(
    split: str,
    variant: AblationVariant,
    predictions: pd.DataFrame,
    threshold_rule: str,
    threshold: float,
) -> dict[str, object]:
    labels = predictions["future_conflict"].to_numpy(dtype=int)
    scores = predictions["probability"].to_numpy(dtype=float)
    metrics = build_evaluation_harness(labels, scores, threshold=threshold)
    row: dict[str, object] = {
        "evaluation_set": split,
        "model": variant.name,
        "threshold_rule": threshold_rule,
        "threshold": threshold,
        "pairs": len(predictions),
        "positives": int(labels.sum()),
    }
    row.update({name: metrics[name] for name in METRIC_COLUMNS})
    return row


def _assert_aligned(predictions: dict[str, pd.DataFrame], split: str) -> None:
    reference = predictions[VARIANTS[0].name]
    reference_rows = reference[PAIR_KEY + ["future_conflict"]].reset_index(drop=True)
    for variant in VARIANTS[1:]:
        current = predictions[variant.name][PAIR_KEY + ["future_conflict"]].reset_index(
            drop=True
        )
        if not current.equals(reference_rows):
            raise ValueError(f"Week 11 variant predictions are not aligned on {split}.")


def evaluate() -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, object],
]:
    """Score each model, select thresholds on validation, and compare on test."""
    _validate_completed_variants()
    validation = score_split("validation", VALIDATION_SESSION)
    _assert_aligned(validation, "validation")
    test = score_split("test", TEST_SESSION)
    _assert_aligned(test, "test")

    thresholds: dict[str, object] = {}
    rows = []
    for variant in VARIANTS:
        validation_predictions = validation[variant.name]
        labels = validation_predictions["future_conflict"].to_numpy(dtype=int)
        scores = validation_predictions["probability"].to_numpy(dtype=float)
        threshold = fit_threshold(labels, scores)
        thresholds[variant.name] = {
            "threshold": threshold,
            "chosen_on": "validation",
            "validation_windows": len(validation_predictions),
            "validation_positives": int(labels.sum()),
        }
        rows.append(
            _metric_row(
                "validation",
                variant,
                validation_predictions,
                "validation_5to1_cost",
                threshold,
            )
        )
        rows.append(
            _metric_row(
                "test",
                variant,
                test[variant.name],
                "validation_5to1_cost",
                threshold,
            )
        )
        rows.append(
            _metric_row("test", variant, test[variant.name], "fixed_0.5", 0.5)
        )

    metrics = pd.DataFrame(rows)
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(EDA_DIR / "week11_ablation_metrics.csv", index=False)
    (EDA_DIR / "week11_ablation_thresholds.json").write_text(
        json.dumps(thresholds, indent=2) + "\n", encoding="utf-8"
    )
    return metrics, validation, test, thresholds


def write_comparison_intervals(
    test: dict[str, pd.DataFrame], thresholds: dict[str, object]
) -> pd.DataFrame:
    """Bootstrap paired metric differences by aircraft pair for each ablation."""
    _assert_aligned(test, "test")
    baseline = test["full_tft"]
    labels = baseline["future_conflict"].to_numpy(dtype=int)
    clusters = (
        baseline["icao24_a"].astype(str) + "|" + baseline["icao24_b"].astype(str)
    ).to_numpy()
    baseline_scores = baseline["probability"].to_numpy(dtype=float)
    baseline_threshold = float(thresholds["full_tft"]["threshold"])
    baseline_point = build_evaluation_harness(
        labels, baseline_scores, threshold=baseline_threshold
    )
    rows = []
    for variant in VARIANTS[1:]:
        candidate = test[variant.name]
        differences = paired_cluster_bootstrap(
            labels,
            baseline_scores,
            candidate["probability"].to_numpy(dtype=float),
            baseline_threshold,
            float(thresholds[variant.name]["threshold"]),
            clusters,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=42,
            progress=lambda done, name=variant.name: print(
                f"  {name} bootstrap: {done:,}/{BOOTSTRAP_RESAMPLES:,}", flush=True
            ),
        )
        candidate_scores = candidate["probability"].to_numpy(dtype=float)
        candidate_point = build_evaluation_harness(
            labels,
            candidate_scores,
            threshold=float(thresholds[variant.name]["threshold"]),
        )
        for metric in BOOTSTRAP_METRICS:
            draws = differences[metric]
            rows.append(
                {
                    "comparison": f"{variant.name} - full_tft",
                    "metric": metric,
                    "difference": float(candidate_point[metric] - baseline_point[metric]),
                    "ci_low": float(np.quantile(draws, (1 - CONFIDENCE) / 2)),
                    "ci_high": float(np.quantile(draws, 1 - (1 - CONFIDENCE) / 2)),
                    "confidence": CONFIDENCE,
                    "clusters": int(len(np.unique(clusters))),
                    "resamples": BOOTSTRAP_RESAMPLES,
                    "thresholds_chosen_on": "validation",
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(EDA_DIR / "week11_ablation_comparison_ci.csv", index=False)
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, _, test_predictions, thresholds = evaluate()
    intervals = write_comparison_intervals(test_predictions, thresholds)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(metrics.loc[metrics["evaluation_set"] == "test"].round(4).to_string(index=False))
        print(intervals.round(4).to_string(index=False))
    print(f"Saved Week 11 metrics: {EDA_DIR / 'week11_ablation_metrics.csv'}")


if __name__ == "__main__":
    main()
