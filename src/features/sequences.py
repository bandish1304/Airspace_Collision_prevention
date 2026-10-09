"""Sliding-window trajectory sequences for sequence models (Week 6, step 2).

A window is ``window_frames`` consecutive observations of one aircraft pair.
Observations are consecutive when the gap between them is at most
``max_gap_seconds``; a larger gap starts a new run, so no window spans a gap,
a pair change or a session boundary. Pairs seen for fewer than
``window_frames`` consecutive frames produce no windows (they are dropped, not
padded).

Each window carries two targets:

* ``y_future``: 1 if the pair is risky in any of the next ``horizon_frames``
  frames of the same run. Windows without that many frames ahead are dropped.
* ``y_now``: 1 if the last frame of the window is already risky. A model given
  the label-defining features can read this off directly, so ``onset_only``
  removes those windows and leaves the harder "will it become risky" task.

Windows are stored as end indices into one feature array, so overlapping
windows cost nothing until ``WindowSet.gather`` materialises a batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.model_inputs import feature_names

DEFAULT_WINDOW_FRAMES = 30
DEFAULT_HORIZON_FRAMES = 6
DEFAULT_MAX_GAP_SECONDS = 30.0


@dataclass
class WindowSet:
    """Windows of one session: features plus per-window targets and metadata."""

    features: np.ndarray  # (n_observations, n_features), float32, sorted by pair then time
    feature_names: list[str]
    window_frames: int
    ends: np.ndarray  # (n_windows,) index of each window's last observation
    y_future: np.ndarray
    y_now: np.ndarray
    pair_id: np.ndarray
    end_time: np.ndarray
    session: str

    def __len__(self) -> int:
        return len(self.ends)

    def gather(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Return windows as an array of shape (n, window_frames, n_features)."""
        chosen = self.ends if indices is None else self.ends[np.asarray(indices)]
        offsets = np.arange(-self.window_frames + 1, 1)
        return self.features[chosen[:, None] + offsets]

    def subset(self, mask: np.ndarray) -> "WindowSet":
        """Keep the windows selected by a boolean mask or index array."""
        return WindowSet(
            features=self.features,
            feature_names=self.feature_names,
            window_frames=self.window_frames,
            ends=self.ends[mask],
            y_future=self.y_future[mask],
            y_now=self.y_now[mask],
            pair_id=self.pair_id[mask],
            end_time=self.end_time[mask],
            session=self.session,
        )


def build_windows(
    pairs: pd.DataFrame,
    session: str = "",
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    horizon_frames: int = DEFAULT_HORIZON_FRAMES,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    include_label_features: bool = True,
    onset_only: bool = False,
) -> WindowSet:
    """Build windows from the output of ``prepare_model_inputs`` for one session."""
    if window_frames < 1 or horizon_frames < 1:
        raise ValueError("window_frames and horizon_frames must be at least 1.")
    if max_gap_seconds <= 0:
        raise ValueError("max_gap_seconds must be greater than zero.")

    names = feature_names(include_label_features)
    missing = [c for c in names + ["snapshot_time", "icao24_a", "icao24_b", "collision_risk"]
               if c not in pairs.columns]
    if missing:
        raise ValueError(f"pairs is missing columns: {missing}")

    obs = pairs.copy()
    obs["aircraft_a"] = np.minimum(obs["icao24_a"], obs["icao24_b"])
    obs["aircraft_b"] = np.maximum(obs["icao24_a"], obs["icao24_b"])
    obs = obs.sort_values(["aircraft_a", "aircraft_b", "snapshot_time"]).reset_index(
        drop=True
    )

    pair_id = obs.groupby(["aircraft_a", "aircraft_b"], sort=False).ngroup().to_numpy()
    times = obs["snapshot_time"].to_numpy(dtype=float)
    risk = obs["collision_risk"].to_numpy(dtype=int)

    new_run = np.ones(len(obs), dtype=bool)
    if len(obs) > 1:
        new_run[1:] = (pair_id[1:] != pair_id[:-1]) | (np.diff(times) > max_gap_seconds)
    run_id = np.cumsum(new_run) - 1
    run_start = np.flatnonzero(new_run)
    run_length = np.diff(np.append(run_start, len(obs)))
    position = np.arange(len(obs)) - run_start[run_id]
    remaining = run_length[run_id] - position - 1

    cumulative = np.concatenate([[0], np.cumsum(risk)])
    index = np.arange(len(obs))
    keep = (position >= window_frames - 1) & (remaining >= horizon_frames)
    ends = index[keep]
    y_future = (
        cumulative[ends + horizon_frames + 1] - cumulative[ends + 1] > 0
    ).astype(np.int8)
    y_now = risk[ends].astype(np.int8)

    windows = WindowSet(
        features=obs[names].to_numpy(dtype=np.float32),
        feature_names=names,
        window_frames=window_frames,
        ends=ends,
        y_future=y_future,
        y_now=y_now,
        pair_id=pair_id[ends],
        end_time=times[ends],
        session=session,
    )
    return windows.subset(y_now == 0) if onset_only else windows
