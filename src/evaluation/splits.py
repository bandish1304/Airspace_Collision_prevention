"""Temporal train/validation/test split shared by every model in the project.

Rows are assigned by snapshot time, never at random. A random row split puts
frames of the same encounter on both sides -- the model is scored on aircraft
it saw ten seconds earlier -- and step 3 showed a grouped split cannot fix that
either, because one connected component holds half the positive class.

Train comes first and test last, so every model learns from the past and is
judged on the future, which is how it would be used. Frames within
``gap_seconds`` after each cut are discarded so encounters running across a
boundary do not leak into the next part. Safe pairs persist across frames as
much as risky ones, which is why the gap is in time rather than in encounters.

With one continuous collection window the window is cut by time. Once several
sessions exist -- separate poll runs, hours or days apart -- whole sessions are
assigned to each part instead, so no session is split down the middle.

Usage:
    python -m src.evaluation.splits data/raw/sessions/<frames>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "eda" / "step6_split_assignment.csv"

DEFAULT_TRAIN_FRACTION = 0.6
DEFAULT_VALIDATION_FRACTION = 0.2

# Twice the median encounter length. Chosen in step 6 after measuring the
# trade-off: it cut encounters spanning two parts from 136 to 49 at a cost of
# 6.8% of positive observations.
DEFAULT_GAP_SECONDS = 120

# Frames further apart than this belong to different collection sessions.
# Frames inside one session are 5-30 s apart; poll sessions are hours apart.
SESSION_BREAK_SECONDS = 600

PARTS = ("train", "validation", "test")


# Step 6: label each snapshot time as train, validation, test or gap.
def temporal_split(
    snapshot_times: pd.Series,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
) -> pd.Series:
    """Return a part label for every row, aligned to the input index."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Fractions must each be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave room for test.")
    if gap_seconds < 0:
        raise ValueError("gap_seconds must be zero or more.")

    frames = np.sort(pd.unique(snapshot_times.dropna()))
    sessions = _session_of(frames)

    if sessions.max() + 1 >= len(PARTS):
        part_of_frame = _split_by_session(
            frames, sessions, train_fraction, validation_fraction
        )
    else:
        part_of_frame = _split_by_time(
            frames, train_fraction, validation_fraction, gap_seconds
        )

    lookup = pd.Series(part_of_frame, index=frames)
    return snapshot_times.map(lookup)


# Step 6: look up each row's part in the saved project split.
def load_split(
    snapshot_times: pd.Series, assignment_path: Path = DEFAULT_OUTPUT_PATH
) -> pd.Series:
    """Return the saved part for every row, aligned to the input index.

    Models read the saved assignment rather than recomputing it, so the GNN,
    the TFT and the baselines are guaranteed the same train, validation and
    test frames. A frame missing from the file is an error, not a silent drop:
    it means new data arrived and the split needs regenerating.
    """
    assignment = pd.read_csv(assignment_path).set_index("snapshot_time")["part"]
    parts = snapshot_times.map(assignment)
    missing = parts.isna()
    if missing.any():
        raise ValueError(
            f"{snapshot_times[missing].nunique()} frame(s) are not in {assignment_path.name}. "
            "Regenerate the split with: python -m src.evaluation.splits <frames csv>"
        )
    return parts


def _session_of(frames: np.ndarray) -> np.ndarray:
    breaks = np.diff(frames) > SESSION_BREAK_SECONDS
    return np.concatenate([[0], np.cumsum(breaks)])


def _split_by_time(
    frames: np.ndarray,
    train_fraction: float,
    validation_fraction: float,
    gap_seconds: float,
) -> np.ndarray:
    start, span = frames[0], frames[-1] - frames[0]
    train_end = start + train_fraction * span
    validation_end = start + (train_fraction + validation_fraction) * span

    return np.select(
        [
            frames < train_end,
            (frames >= train_end + gap_seconds) & (frames < validation_end),
            frames >= validation_end + gap_seconds,
        ],
        PARTS,
        default="gap",
    )


def _split_by_session(
    frames: np.ndarray,
    sessions: np.ndarray,
    train_fraction: float,
    validation_fraction: float,
) -> np.ndarray:
    # Sessions are hours apart already, so no gap is needed between them.
    # Each session goes to the part its midpoint falls in, measured by frame
    # count so a long session weighs more than a short one.
    session_ids, frame_counts = np.unique(sessions, return_counts=True)
    midpoint = (np.cumsum(frame_counts) - frame_counts / 2) / frame_counts.sum()
    part_of_session = np.select(
        [
            midpoint < train_fraction,
            midpoint < train_fraction + validation_fraction,
        ],
        PARTS[:2],
        default=PARTS[2],
    )

    # With few sessions a part can end up empty; the last sessions always go
    # to test and the one before to validation, so all three exist.
    part_of_session[-1] = "test"
    if "validation" not in part_of_session:
        part_of_session[-2] = "validation"

    return part_of_session[np.searchsorted(session_ids, sessions)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames_paths", type=Path, nargs="+",
                        help="One or more collected frame CSVs.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--gap-seconds", type=float, default=DEFAULT_GAP_SECONDS)
    args = parser.parse_args()

    times = pd.concat(
        pd.read_csv(path, usecols=["snapshot_time"])["snapshot_time"]
        for path in args.frames_paths
    ).drop_duplicates().sort_values().reset_index(drop=True)
    parts = temporal_split(times, gap_seconds=args.gap_seconds)

    assignment = pd.DataFrame({"snapshot_time": times, "part": parts})
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(args.output_path, index=False)

    start = times.min()
    sessions = _session_of(times.to_numpy()).max() + 1
    print(f"Frames   : {len(times)} in {sessions} session(s)")
    # Report contiguous runs, so the two separate gaps are not printed as one
    # long stretch from the first gap's start to the second gap's end.
    run = (assignment["part"] != assignment["part"].shift()).cumsum()
    for _, block in assignment.groupby(run, sort=True):
        part = block["part"].iloc[0]
        print(f"{part:<10}: {len(block):>4} frames, "
              f"minute {(block['snapshot_time'].min() - start) / 60:5.1f} "
              f"to {(block['snapshot_time'].max() - start) / 60:5.1f}")
    print(f"Wrote    : {args.output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
