"""Score the tuned TFT once on the held-out test session and align it with the GNN.

The tuned GraphSAGE model already scored its test candidate pairs once; those
saved predictions are reused unchanged. This module scores every TFT test
window once, saves the result, and keeps only the pair observations that both
models scored, using the GNN's future-conflict label as the shared target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.evaluation.splits import DEFAULT_OUTPUT_PATH, load_split
from src.features.build_pipeline import build_pair_features
from src.features.model_inputs import feature_names, prepare_model_inputs
from src.features.sequence_splits import DEFAULT_SESSION_NAMES, SESSION_DIR
from src.features.sequences import WindowSet, build_windows
from src.features.tft_final_train import (
    TFTArrayCollator,
    build_template_from_checkpoint,
)
from src.features.tft_smoke import sample_indices

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_SESSION = DEFAULT_SESSION_NAMES[-1]
TFT_CHECKPOINT = PROJECT_ROOT / "models" / "week9_tft" / "week9_tft_best.ckpt"
TFT_CONFIG = PROJECT_ROOT / "models" / "week9_tft" / "week9_tft_training_config.json"
GNN_PREDICTIONS = (
    PROJECT_ROOT / "data" / "processed" / "graphsage_tuning" / "test_predictions.csv"
)
GNN_CONFIG = PROJECT_ROOT / "data" / "processed" / "graphsage_tuning" / "best_config.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "week10_comparison"
TFT_PREDICTIONS = OUTPUT_DIR / "tft_test_predictions.csv"
ALIGNED_PREDICTIONS = OUTPUT_DIR / "aligned_test_predictions.csv"
SUMMARY_PATH = PROJECT_ROOT / "experiments" / "eda" / "week10_test_alignment.json"
PAIR_KEY = ["snapshot_time", "icao24_a", "icao24_b"]
SCORING_BATCH_SIZE = 1024
TEMPLATE_SAMPLES = 2_000


class WindowSetDataset(Dataset[tuple[Tensor, Tensor]]):
    """Gather one window at a time so the full window tensor is never materialised."""

    def __init__(self, windows: WindowSet) -> None:
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        window = self.windows.gather(np.array([index]))[0]
        target = float(self.windows.y_future[index])
        return torch.from_numpy(window), torch.tensor(target, dtype=torch.float32)


def normalize_pair_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Lower-case aircraft IDs and order each pair so both models share one key."""
    result = frame.copy()
    first = result["icao24_a"].astype(str).str.lower()
    second = result["icao24_b"].astype(str).str.lower()
    result["icao24_a"] = np.where(first <= second, first, second)
    result["icao24_b"] = np.where(first <= second, second, first)
    if result.duplicated(PAIR_KEY).any():
        raise ValueError("Predictions contain duplicate snapshot/pair rows.")
    return result


def align_predictions(
    gnn: pd.DataFrame, tft: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Keep pair observations scored by both models and summarise the overlap."""
    required_gnn = {*PAIR_KEY, "future_conflict", "probability"}
    required_tft = {*PAIR_KEY, "tft_y_future", "tft_y_now", "probability"}
    if not required_gnn.issubset(gnn.columns):
        raise ValueError(f"GNN predictions need columns {sorted(required_gnn)}.")
    if not required_tft.issubset(tft.columns):
        raise ValueError(f"TFT predictions need columns {sorted(required_tft)}.")

    gnn = normalize_pair_ids(gnn).rename(columns={"probability": "gnn_probability"})
    tft = normalize_pair_ids(tft).rename(columns={"probability": "tft_probability"})
    aligned = gnn.merge(tft, on=PAIR_KEY, how="inner", validate="one_to_one")
    aligned = aligned.sort_values(PAIR_KEY).reset_index(drop=True)

    label = aligned["future_conflict"].astype(int)
    tft_target = aligned["tft_y_future"].astype(int)
    summary: dict[str, object] = {
        "shared_label": "future_conflict (new conflict within 60 s, GNN definition)",
        "gnn_pairs": int(len(gnn)),
        "gnn_positive_pairs": int(gnn["future_conflict"].sum()),
        "gnn_snapshots": int(gnn["snapshot_time"].nunique()),
        "tft_windows": int(len(tft)),
        "tft_positive_windows": int(tft["tft_y_future"].sum()),
        "aligned_pairs": int(len(aligned)),
        "aligned_positive_pairs": int(label.sum()),
        "aligned_snapshots": int(aligned["snapshot_time"].nunique()),
        "gnn_pair_coverage": float(len(aligned) / len(gnn)),
        "gnn_positives_without_tft_window": int(
            gnn["future_conflict"].sum() - label.sum()
        ),
        "tft_windows_outside_gnn_pairs": int(len(tft) - len(aligned)),
        "tft_already_risky_among_aligned": int(aligned["tft_y_now"].sum()),
        "label_agreement": {
            "both_positive": int(((label == 1) & (tft_target == 1)).sum()),
            "future_conflict_only": int(((label == 1) & (tft_target == 0)).sum()),
            "tft_target_only": int(((label == 0) & (tft_target == 1)).sum()),
            "both_negative": int(((label == 0) & (tft_target == 0)).sum()),
        },
    }
    return aligned, summary


def load_pair_observations(
    session_name: str = TEST_SESSION, part: str = "test"
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build model-ready pair observations for one session of a single split."""
    session_path = SESSION_DIR / session_name
    if not session_path.is_file():
        raise FileNotFoundError(f"Session data not found: {session_path}")
    states = pd.read_csv(session_path)
    states["icao24"] = states["icao24"].str.lower()
    icao_by_code = np.array(sorted(states["icao24"].dropna().unique()))
    code_by_icao = {icao: code for code, icao in enumerate(icao_by_code)}

    unique_times = states["snapshot_time"].drop_duplicates()
    parts = load_split(unique_times, DEFAULT_OUTPUT_PATH)
    if set(parts.unique()) != {part}:
        raise ValueError(f"{session_name} is not assigned entirely to the {part} split.")

    columns = [
        "snapshot_time",
        "icao24_a",
        "icao24_b",
        *feature_names(),
        "collision_risk",
    ]
    chunks = []
    for _, snapshot_states in states.groupby("snapshot_time", sort=True):
        pairs = prepare_model_inputs(build_pair_features(snapshot_states))
        if pairs.empty:
            continue
        pairs["icao24_a"] = pairs["icao24_a"].str.lower().map(code_by_icao)
        pairs["icao24_b"] = pairs["icao24_b"].str.lower().map(code_by_icao)
        chunks.append(pairs[columns].copy())
    if not chunks:
        raise ValueError(f"{session_name} produced no pair observations.")
    observations = pd.concat(chunks, ignore_index=True)
    observations[["icao24_a", "icao24_b"]] = observations[
        ["icao24_a", "icao24_b"]
    ].astype(np.int32)
    observations["snapshot_time"] = observations["snapshot_time"].astype(np.int64)
    observations["collision_risk"] = observations["collision_risk"].astype(np.int8)
    return observations, icao_by_code


def build_session_windows(
    session_name: str = TEST_SESSION, part: str = "test"
) -> tuple[WindowSet, pd.DataFrame]:
    """Build windows and recover each window's aircraft pair and end time."""
    observations, icao_by_code = load_pair_observations(session_name, part)
    windows = build_windows(observations, session=session_name)
    if not len(windows):
        raise ValueError(f"No windows were produced for {session_name}.")

    ordered = observations.assign(
        aircraft_a=np.minimum(observations["icao24_a"], observations["icao24_b"]),
        aircraft_b=np.maximum(observations["icao24_a"], observations["icao24_b"]),
    ).sort_values(["aircraft_a", "aircraft_b", "snapshot_time"]).reset_index(drop=True)
    ends = ordered.iloc[windows.ends]
    if not np.array_equal(
        ends[windows.feature_names].to_numpy(dtype=np.float32),
        windows.features[windows.ends],
    ) or not np.array_equal(ends["snapshot_time"].to_numpy(dtype=float), windows.end_time):
        raise ValueError("Recovered window identities do not match the window features.")

    metadata = pd.DataFrame(
        {
            "snapshot_time": ends["snapshot_time"].to_numpy(dtype=np.int64),
            "icao24_a": icao_by_code[ends["aircraft_a"].to_numpy()],
            "icao24_b": icao_by_code[ends["aircraft_b"].to_numpy()],
            "tft_y_future": windows.y_future.astype(int),
            "tft_y_now": windows.y_now.astype(int),
        }
    )
    return windows, metadata


def score_windows(windows: WindowSet, checkpoint_path: Path) -> np.ndarray:
    """Return the saved TFT's risk probability for every window."""
    targets = windows.y_future.astype(np.int8)
    sample = sample_indices(targets, TEMPLATE_SAMPLES, stratified=True)
    template = build_template_from_checkpoint(
        checkpoint_path,
        torch.from_numpy(windows.gather(sample)),
        torch.from_numpy(targets[sample].astype(np.float32)),
        windows.feature_names,
    )
    loader = DataLoader(
        WindowSetDataset(windows),
        batch_size=SCORING_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=TFTArrayCollator(template),
    )
    model = TemporalFusionTransformer.load_from_checkpoint(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for batch_number, (x, _) in enumerate(loader, start=1):
            prediction = model.loss.to_prediction(model(x)["prediction"])
            probabilities.append(prediction.reshape(-1).cpu().numpy())
            if batch_number % 25 == 0:
                print(f"  scored {batch_number * SCORING_BATCH_SIZE:,} windows", flush=True)
    scores = np.concatenate(probabilities)
    if len(scores) != len(windows) or not np.isfinite(scores).all():
        raise ValueError("TFT scores are missing or non-finite.")
    return scores


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    for path in (TFT_CHECKPOINT, TFT_CONFIG, GNN_PREDICTIONS, GNN_CONFIG):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if TFT_PREDICTIONS.is_file():
        print(f"Reusing saved one-time TFT test scores: {TFT_PREDICTIONS}")
        tft = pd.read_csv(TFT_PREDICTIONS)
    else:
        print("Building TFT test windows...", flush=True)
        windows, tft = build_session_windows()
        print(f"Scoring {len(windows):,} test windows once...", flush=True)
        tft["probability"] = score_windows(windows, TFT_CHECKPOINT)
        tft.to_csv(TFT_PREDICTIONS, index=False)
        print(f"Saved TFT test scores: {TFT_PREDICTIONS}")

    gnn = pd.read_csv(GNN_PREDICTIONS)
    aligned, summary = align_predictions(gnn, tft)
    aligned.to_csv(ALIGNED_PREDICTIONS, index=False)

    gnn_config = json.loads(GNN_CONFIG.read_text(encoding="utf-8"))
    tft_config = json.loads(TFT_CONFIG.read_text(encoding="utf-8"))
    summary.update(
        {
            "test_session": TEST_SESSION,
            "tft_checkpoint": str(TFT_CHECKPOINT.relative_to(PROJECT_ROOT)),
            "tft_checkpoint_sha256": sha256(TFT_CHECKPOINT),
            "tft_best_epoch": tft_config["best_epoch"],
            "gnn_predictions": str(GNN_PREDICTIONS.relative_to(PROJECT_ROOT)),
            "gnn_predictions_sha256": sha256(GNN_PREDICTIONS),
            "gnn_best_epoch": gnn_config["best_epoch"],
            "gnn_hidden_channels": gnn_config["hidden_channels"],
            "gnn_learning_rate": gnn_config["learning_rate"],
            "aligned_predictions": str(ALIGNED_PREDICTIONS.relative_to(PROJECT_ROOT)),
            "each_model_scored_test_once": True,
        }
    )
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
