"""Count how many sliding windows each session can supply for sequence models.

Week 6, step 2 design check. A window is a run of consecutive observations of
one aircraft pair. A run ends when the pair is unseen for more than
``max_gap_seconds``. For each window length the script reports how many windows
exist, how many end on a risky observation, and how many are followed by a
risky observation within the next ``horizon_frames`` frames of the same run.
Whole sessions are counted separately, so nothing crosses a session boundary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_pipeline import build_pair_features
from src.features.model_inputs import prepare_model_inputs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SESSION_DIR = PROJECT_ROOT / "data" / "raw" / "sessions"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "eda" / "week6_window_feasibility.csv"
DEFAULT_SESSIONS = [
    "states_socal_backfill_1790375809.csv",
    "states_socal_session_1791380397_06-39AM.csv",
    "states_socal_session_1791393504_10-18AM.csv",
    "states_socal_session_1791408650_02-30PM.csv",
    "states_socal_session_1791427160_07-39PM.csv",
]
DEFAULT_WINDOW_LENGTHS = (6, 12, 30, 60)
DEFAULT_MAX_GAP_SECONDS = 30.0
DEFAULT_HORIZON_FRAMES = 6


def count_windows(
    pairs: pd.DataFrame,
    session_name: str,
    window_lengths: tuple[int, ...] = DEFAULT_WINDOW_LENGTHS,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    horizon_frames: int = DEFAULT_HORIZON_FRAMES,
) -> pd.DataFrame:
    """Return window counts per window length for one session's labelled pairs."""
    if horizon_frames < 1:
        raise ValueError("horizon_frames must be at least 1.")
    if max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be greater than zero.")
    if not window_lengths or min(window_lengths) < 1:
        raise ValueError("window_lengths must be positive integers.")

    obs = pairs[["snapshot_time", "icao24_a", "icao24_b", "collision_risk"]].copy()
    obs["aircraft_a"] = np.minimum(obs["icao24_a"], obs["icao24_b"])
    obs["aircraft_b"] = np.maximum(obs["icao24_a"], obs["icao24_b"])
    obs["risk"] = obs["collision_risk"].astype(int)
    obs = obs.sort_values(["aircraft_a", "aircraft_b", "snapshot_time"]).reset_index(
        drop=True
    )

    pair_id = obs.groupby(["aircraft_a", "aircraft_b"], sort=False).ngroup().to_numpy()
    times = obs["snapshot_time"].to_numpy(dtype=float)
    risk = obs["risk"].to_numpy()

    new_run = np.ones(len(obs), dtype=bool)
    new_run[1:] = (pair_id[1:] != pair_id[:-1]) | (np.diff(times) > max_gap_seconds)
    run_id = np.cumsum(new_run) - 1
    run_start = np.flatnonzero(new_run)
    run_length = np.diff(np.append(run_start, len(obs)))
    position = np.arange(len(obs)) - run_start[run_id]
    remaining = run_length[run_id] - position - 1

    cumulative = np.concatenate([[0], np.cumsum(risk)])
    ahead_sum = np.zeros(len(obs), dtype=int)
    has_future = remaining >= horizon_frames
    index = np.arange(len(obs))
    ahead_sum[has_future] = (
        cumulative[index[has_future] + horizon_frames + 1]
        - cumulative[index[has_future] + 1]
    )

    rows = []
    for length in window_lengths:
        valid = position >= length - 1
        now_positive = valid & (risk == 1)
        ahead_valid = valid & has_future
        ahead_positive = ahead_valid & (ahead_sum > 0)
        rows.append(
            {
                "session": session_name,
                "window_frames": length,
                "windows": int(valid.sum()),
                "pairs_with_windows": int(len(np.unique(pair_id[valid]))),
                "windows_ending_risky": int(now_positive.sum()),
                "pairs_with_risky_end": int(len(np.unique(pair_id[now_positive]))),
                "windows_with_future_data": int(ahead_valid.sum()),
                "windows_risky_within_horizon": int(ahead_positive.sum()),
                "pairs_risky_within_horizon": int(
                    len(np.unique(pair_id[ahead_positive]))
                ),
                "horizon_frames": horizon_frames,
                "max_gap_seconds": max_gap_seconds,
            }
        )
    result = pd.DataFrame(rows)
    result["risky_end_rate_pct"] = (
        100 * result["windows_ending_risky"] / result["windows"].where(result["windows"] > 0)
    )
    return result


def analyse_session(session_path: Path, **kwargs) -> pd.DataFrame:
    """Build and label pairs for one raw session, then count its windows."""
    states = pd.read_csv(session_path)
    labelled = prepare_model_inputs(build_pair_features(states))
    return count_windows(labelled, session_path.name, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-gap-seconds", type=float, default=DEFAULT_MAX_GAP_SECONDS)
    parser.add_argument("--horizon-frames", type=int, default=DEFAULT_HORIZON_FRAMES)
    parser.add_argument(
        "--window-lengths", type=int, nargs="+", default=list(DEFAULT_WINDOW_LENGTHS)
    )
    args = parser.parse_args()

    frames = []
    for name in DEFAULT_SESSIONS:
        path = SESSION_DIR / name
        print(f"Counting windows in {name}", flush=True)
        frames.append(
            analyse_session(
                path,
                window_lengths=tuple(args.window_lengths),
                max_gap_seconds=args.max_gap_seconds,
                horizon_frames=args.horizon_frames,
            )
        )
    result = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
