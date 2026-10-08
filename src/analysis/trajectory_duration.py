"""Censoring-aware observed duration of aircraft tracks in a collection window.

The result is the time each airborne aircraft was observed in one continuous
track segment, not the aircraft's true flight duration. A segment present in
the first frame is left-censored; a segment is right-censored when the data
ends before a 120-second report gap could establish its end. Left-censored
segments are excluded from the Kaplan-Meier estimate because their true
segment start is unknown; right-censored segments with an observed start are
included.

Usage:
    python -m src.analysis.trajectory_duration
    python -m src.analysis.trajectory_duration data/raw/sessions/<frames>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.build_pipeline import DEFAULT_MAX_POSITION_AGE_SECONDS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAMES_PATH = (
    PROJECT_ROOT / "data" / "raw" / "sessions"
    / "states_socal_backfill_1790375809.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "eda"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "experiments" / "figures"
DEFAULT_GAP_SECONDS = 120.0
REQUIRED_COLUMNS = {
    "icao24", "snapshot_time", "time_position", "latitude", "longitude", "on_ground",
}


def build_trajectory_segments(
    states: pd.DataFrame,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
    max_position_age_seconds: float = DEFAULT_MAX_POSITION_AGE_SECONDS,
) -> pd.DataFrame:
    """Summarize continuous observed track segments and collection-edge censoring."""
    if gap_seconds < 0:
        raise ValueError("gap_seconds must be zero or greater.")
    if max_position_age_seconds < 0:
        raise ValueError("max_position_age_seconds must be zero or greater.")

    missing = REQUIRED_COLUMNS.difference(states.columns)
    if missing:
        raise ValueError(f"State data is missing required columns: {', '.join(sorted(missing))}.")
    if states.empty:
        return _empty_segments()

    window_start = pd.to_numeric(states["snapshot_time"], errors="coerce").min()
    window_end = pd.to_numeric(states["snapshot_time"], errors="coerce").max()
    if pd.isna(window_start) or pd.isna(window_end):
        raise ValueError("State data must contain at least one valid snapshot_time.")

    tracked = states.copy()
    for column in ("snapshot_time", "time_position", "latitude", "longitude"):
        tracked[column] = pd.to_numeric(tracked[column], errors="coerce")
    tracked["position_age_s"] = tracked["snapshot_time"] - tracked["time_position"]
    valid = (
        tracked["icao24"].notna()
        & tracked["snapshot_time"].notna()
        & tracked["time_position"].notna()
        & tracked["latitude"].notna()
        & tracked["longitude"].notna()
        & tracked["on_ground"].eq(False)
        & tracked["position_age_s"].between(0, max_position_age_seconds)
    )
    tracked = tracked.loc[valid, ["icao24", "snapshot_time", "latitude", "longitude"]]
    tracked = tracked.drop_duplicates(["icao24", "snapshot_time"], keep="last")
    if tracked.empty:
        return _empty_segments()

    tracked = tracked.sort_values(["icao24", "snapshot_time"]).reset_index(drop=True)
    time_gap = tracked.groupby("icao24", sort=False)["snapshot_time"].diff()
    new_segment = tracked["icao24"].ne(tracked["icao24"].shift()) | time_gap.gt(gap_seconds)
    tracked["segment_number"] = new_segment.cumsum()

    segments = tracked.groupby(["icao24", "segment_number"], sort=False).agg(
        start_time=("snapshot_time", "min"),
        end_time=("snapshot_time", "max"),
        observations=("snapshot_time", "size"),
        start_latitude=("latitude", "first"),
        start_longitude=("longitude", "first"),
        end_latitude=("latitude", "last"),
        end_longitude=("longitude", "last"),
    ).reset_index()
    segments["observed_duration_seconds"] = segments["end_time"] - segments["start_time"]
    segments["left_censored"] = segments["start_time"].eq(window_start)
    segments["right_censored"] = segments["end_time"] + gap_seconds >= window_end
    segments["survival_duration_seconds"] = np.where(
        segments["right_censored"],
        window_end - segments["start_time"],
        segments["observed_duration_seconds"],
    )
    segments["censoring_status"] = np.select(
        [
            segments["left_censored"] & segments["right_censored"],
            segments["left_censored"],
            segments["right_censored"],
        ],
        ["both", "left", "right"],
        default="complete",
    )
    segments["event_observed"] = ~segments["right_censored"]
    segments["window_start"] = window_start
    segments["window_end"] = window_end
    segments["gap_seconds"] = gap_seconds
    return segments


def kaplan_meier(segments: pd.DataFrame) -> pd.DataFrame:
    """Estimate track-segment survival, excluding segments with unknown starts."""
    eligible = segments.loc[~segments["left_censored"]]
    if eligible.empty:
        return pd.DataFrame(columns=["time_seconds", "at_risk", "events", "censored", "survival"])

    durations = eligible["survival_duration_seconds"].to_numpy(dtype=float)
    events = eligible["event_observed"].to_numpy(dtype=bool)
    survival = 1.0
    rows = []
    for time_seconds in np.sort(np.unique(durations)):
        at_risk = int((durations >= time_seconds).sum())
        events_at_time = int(((durations == time_seconds) & events).sum())
        censored_at_time = int(((durations == time_seconds) & ~events).sum())
        if at_risk:
            survival *= 1 - events_at_time / at_risk
        rows.append(
            {
                "time_seconds": time_seconds,
                "at_risk": at_risk,
                "events": events_at_time,
                "censored": censored_at_time,
                "survival": survival,
            }
        )
    return pd.DataFrame(rows)


def summarize_segments(segments: pd.DataFrame, survival: pd.DataFrame) -> pd.DataFrame:
    """Return duration and censoring counts, with a censoring-aware median."""
    if segments.empty:
        return pd.DataFrame(
            columns=["censoring_status", "segments", "median_observed_duration_seconds",
                     "q25_observed_duration_seconds", "q75_observed_duration_seconds"]
        )

    summary = segments.groupby("censoring_status", as_index=False).agg(
        segments=("icao24", "size"),
        median_observed_duration_seconds=("observed_duration_seconds", "median"),
        q25_observed_duration_seconds=("observed_duration_seconds", lambda v: v.quantile(0.25)),
        q75_observed_duration_seconds=("observed_duration_seconds", lambda v: v.quantile(0.75)),
    )
    summary["censoring_status"] = pd.Categorical(
        summary["censoring_status"], ["complete", "left", "right", "both"], ordered=True
    )
    summary = summary.sort_values("censoring_status").reset_index(drop=True)
    km_median = (
        float(survival.loc[survival["survival"] <= 0.5, "time_seconds"].iloc[0])
        if not survival.empty and (survival["survival"] <= 0.5).any()
        else np.nan
    )
    summary["kaplan_meier_median_seconds_excluding_left_censored"] = km_median
    summary["segments_in_kaplan_meier"] = int((~segments["left_censored"]).sum())
    summary["left_censored_excluded_from_kaplan_meier"] = int(segments["left_censored"].sum())
    return summary


def plot_survival(survival: pd.DataFrame, output_path: Path) -> None:
    """Save the Kaplan-Meier curve for observed track-segment durations."""
    figure, axis = plt.subplots(figsize=(8, 5))
    if survival.empty:
        axis.text(0.5, 0.5, "No segments with an observed start", ha="center", va="center",
                  transform=axis.transAxes)
    else:
        axis.step(
            np.r_[0, survival["time_seconds"].to_numpy()],
            np.r_[1, survival["survival"].to_numpy()],
            where="post",
            color="#3d74b8",
            linewidth=2,
        )
    axis.set(
        xlabel="Observed track-segment duration (seconds)",
        ylabel="Estimated share still being observed",
        ylim=(0, 1.02),
        title="Censoring-aware observed aircraft track duration",
    )
    axis.grid(True, color="#e2e8f0", linewidth=0.7)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _empty_segments() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "icao24", "segment_number", "start_time", "end_time", "observations",
            "start_latitude", "start_longitude", "end_latitude", "end_longitude",
            "observed_duration_seconds", "survival_duration_seconds",
            "left_censored", "right_censored",
            "censoring_status", "event_observed", "window_start", "window_end",
            "gap_seconds",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames_path", nargs="?", type=Path, default=DEFAULT_FRAMES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--gap-seconds", type=float, default=DEFAULT_GAP_SECONDS)
    args = parser.parse_args()

    columns = sorted(REQUIRED_COLUMNS)
    states = pd.read_csv(args.frames_path, usecols=columns)
    segments = build_trajectory_segments(states, gap_seconds=args.gap_seconds)
    survival = kaplan_meier(segments)
    summary = summarize_segments(segments, survival)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    segments.to_csv(args.output_dir / "step9_trajectory_durations.csv", index=False)
    summary.to_csv(args.output_dir / "step9_trajectory_duration_summary.csv", index=False)
    survival.to_csv(args.output_dir / "step9_trajectory_duration_kaplan_meier.csv", index=False)
    plot_survival(survival, args.figure_dir / "step9_trajectory_duration_survival.png")

    print(f"Input frames     : {args.frames_path}")
    print(f"Track segments   : {len(segments):,}")
    print(f"Observed aircraft: {segments['icao24'].nunique():,}")
    print("Censoring status :")
    print(segments["censoring_status"].value_counts().sort_index().to_string())
    median = summary["kaplan_meier_median_seconds_excluding_left_censored"].dropna()
    if not median.empty:
        print(f"Kaplan-Meier median observed duration: {median.iloc[0]:.0f} s "
              "(left-censored segments excluded)")
    else:
        print("Kaplan-Meier median not reached in the observed window.")
    print(f"Wrote outputs to : {args.output_dir}")


if __name__ == "__main__":
    main()
