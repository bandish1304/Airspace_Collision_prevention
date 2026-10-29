"""Build temporally separated TFT window datasets from prepared pair sessions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset

from src.evaluation.splits import DEFAULT_OUTPUT_PATH, load_split
from src.features.build_pipeline import build_pair_features
from src.features.model_inputs import prepare_model_inputs
from src.features.sequences import (
    DEFAULT_HORIZON_FRAMES,
    DEFAULT_MAX_GAP_SECONDS,
    DEFAULT_WINDOW_FRAMES,
    WindowSet,
    build_windows,
)
from src.features.tft_imbalance import smote_training_windows
from src.features.tft_dataset import TFTWindowDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SESSION_DIR = PROJECT_ROOT / "data" / "raw" / "sessions"
DEFAULT_SESSION_NAMES = (
    "states_socal_backfill_1790375809.csv",
    "states_socal_session_1791380397_06-39AM.csv",
    "states_socal_session_1791393504_10-18AM.csv",
    "states_socal_session_1791408650_02-30PM.csv",
    "states_socal_session_1791427160_07-39PM.csv",
)
DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT / "experiments" / "eda" / "week7_sequence_split_summary.csv"
)
SPLITS = ("train", "validation", "test")


def build_sequence_splits(
    pairs_by_session: Mapping[str, pd.DataFrame],
    assignment_path: Path = DEFAULT_OUTPUT_PATH,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    horizon_frames: int = DEFAULT_HORIZON_FRAMES,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
) -> tuple[dict[str, ConcatDataset], pd.DataFrame]:
    """Build DataLoader-ready datasets without windows crossing split boundaries.

    ``pairs_by_session`` must contain model-ready, labelled pair observations
    for each independent collection session. The saved frame assignment is
    applied before windows are built, so both encoder frames and future target
    frames belong to the same temporal split.
    """
    if not pairs_by_session:
        raise ValueError("At least one prepared pair session is required.")

    datasets: dict[str, list[TFTWindowDataset]] = {split: [] for split in SPLITS}
    summary: dict[str, dict[str, int | set[str]]] = {
        split: {
            "sessions": set(),
            "pair_observations": 0,
            "windows": 0,
            "positive_windows": 0,
            "negative_windows": 0,
            "unique_pairs": 0,
            "first_snapshot_time": None,
            "last_snapshot_time": None,
        }
        for split in SPLITS
    }

    for session, pairs in pairs_by_session.items():
        required = {"snapshot_time", "icao24_a", "icao24_b", "collision_risk"}
        missing = required.difference(pairs.columns)
        if missing:
            raise ValueError(
                f"{session} is missing required columns: {sorted(missing)}."
            )
        assigned = pairs.copy()
        assigned["split"] = load_split(assigned["snapshot_time"], assignment_path).to_numpy()

        for split in SPLITS:
            selected = assigned.loc[assigned["split"].eq(split)].drop(columns="split")
            if selected.empty:
                continue
            windows = build_windows(
                selected,
                session=session,
                window_frames=window_frames,
                horizon_frames=horizon_frames,
                max_gap_seconds=max_gap_seconds,
            )
            if not len(windows):
                continue

            datasets[split].append(TFTWindowDataset(windows))
            counts = summary[split]
            counts["sessions"].add(session)
            counts["pair_observations"] += len(selected)
            counts["windows"] += len(windows)
            counts["positive_windows"] += int(windows.y_future.sum())
            counts["negative_windows"] += int(len(windows) - windows.y_future.sum())
            counts["unique_pairs"] += int(np.unique(windows.pair_id).size)
            first_time = float(selected["snapshot_time"].min())
            last_time = float(selected["snapshot_time"].max())
            counts["first_snapshot_time"] = (
                first_time
                if counts["first_snapshot_time"] is None
                else min(float(counts["first_snapshot_time"]), first_time)
            )
            counts["last_snapshot_time"] = (
                last_time
                if counts["last_snapshot_time"] is None
                else max(float(counts["last_snapshot_time"]), last_time)
            )

    empty_splits = [split for split, pieces in datasets.items() if not pieces]
    if empty_splits:
        raise ValueError(
            "No sequence windows were produced for required split(s): "
            f"{', '.join(empty_splits)}. Check the temporal assignment and "
            "available session coverage."
        )

    result = {
        split: ConcatDataset(pieces)
        for split, pieces in datasets.items()
    }
    summary_rows = []
    for split in SPLITS:
        counts = summary[split]
        windows = int(counts["windows"])
        summary_rows.append(
            {
                "split": split,
                "sessions": len(counts["sessions"]),
                "session_names": ";".join(sorted(counts["sessions"])),
                "pair_observations": counts["pair_observations"],
                "windows": windows,
                "positive_windows": counts["positive_windows"],
                "negative_windows": counts["negative_windows"],
                "positive_rate": (
                    int(counts["positive_windows"]) / windows if windows else 0.0
                ),
                "unique_pairs_per_session_sum": counts["unique_pairs"],
                "first_snapshot_time": counts["first_snapshot_time"],
                "last_snapshot_time": counts["last_snapshot_time"],
                "window_frames": window_frames,
                "horizon_frames": horizon_frames,
                "max_gap_seconds": max_gap_seconds,
            }
        )
    return result, pd.DataFrame(summary_rows)


def load_and_prepare_sessions(session_paths: list[Path]) -> dict[str, pd.DataFrame]:
    """Load real ADS-B sessions and produce labelled pair observations."""
    prepared = {}
    for session_path in session_paths:
        if not session_path.is_file():
            raise FileNotFoundError(f"Session data file not found: {session_path}")
        print(f"Preparing pair observations: {session_path.name}", flush=True)
        states = pd.read_csv(session_path)
        pairs = build_pair_features(states)
        prepared[session_path.name] = prepare_model_inputs(pairs)
        print(
            f"  states={len(states):,} pair_observations={len(prepared[session_path.name]):,}",
            flush=True,
        )
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions",
        nargs="+",
        type=Path,
        default=[SESSION_DIR / name for name in DEFAULT_SESSION_NAMES],
        help="Raw ADS-B session CSVs. Defaults to the five sessions in the project split.",
    )
    parser.add_argument("--assignment", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--window-frames", type=int, default=DEFAULT_WINDOW_FRAMES)
    parser.add_argument("--horizon-frames", type=int, default=DEFAULT_HORIZON_FRAMES)
    parser.add_argument(
        "--max-gap-seconds", type=float, default=DEFAULT_MAX_GAP_SECONDS
    )
    args = parser.parse_args()

    pairs_by_session = load_and_prepare_sessions(args.sessions)
    datasets, summary = build_sequence_splits(
        pairs_by_session,
        assignment_path=args.assignment,
        window_frames=args.window_frames,
        horizon_frames=args.horizon_frames,
        max_gap_seconds=args.max_gap_seconds,
    )
    original_train_samples = len(datasets["train"])
    datasets["train"], smote_metadata = smote_training_windows(datasets["train"])
    summary["dataset_samples"] = [len(datasets[split]) for split in SPLITS]
    for key, value in smote_metadata.items():
        summary.loc[summary["split"].eq("train"), f"smote_{key}"] = value
    summary.loc[summary["split"].eq("train"), "original_train_windows"] = (
        original_train_samples
    )
    summary.loc[summary["split"].ne("train"), "original_train_windows"] = np.nan

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))
    print(
        "SMOTE applied to training only: "
        f"{original_train_samples:,} -> {len(datasets['train']):,} windows; "
        f"validation={len(datasets['validation']):,}, "
        f"test={len(datasets['test']):,} unchanged."
    )
    print(f"Saved split summary: {args.output}")


if __name__ == "__main__":
    main()
