"""Compare collision-rule encounter rates across sampled times of day.

Each session is processed independently with the existing pair-feature builder
and label rule. Outputs include one session-level summary and five-minute
within-session rates; the samples are descriptive, not independent days.

Usage:
    python -m src.analysis.time_of_day
    python -m src.analysis.time_of_day data/raw/sessions/<session>.csv [...]
"""

from __future__ import annotations

import argparse
import re
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.build_pipeline import build_pair_features
from src.features.model_inputs import prepare_model_inputs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSIONS_DIR = PROJECT_ROOT / "data" / "raw" / "sessions"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "eda"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "experiments" / "figures"
DEFAULT_WINDOW_MINUTES = 5
DEFAULT_ENCOUNTER_GAP_SECONDS = 60
LOCAL_TIMEZONE = "America/Los_Angeles"

REQUIRED_PAIR_COLUMNS = {
    "snapshot_time",
    "icao24_a",
    "icao24_b",
    "collision_risk",
}
REQUIRED_STATE_COLUMNS = {"snapshot_time"}
TIMED_SESSION_NAME = re.compile(r"^states_socal_session_.*_\d{2}-\d{2}[AP]M\.csv$")


def summarise_session_pairs(
    pairs: pd.DataFrame,
    session_start: float,
    session_end: float,
    session_name: str,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
    encounter_gap_seconds: int = DEFAULT_ENCOUNTER_GAP_SECONDS,
    frame_count: int | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Return session rates and five-minute rates from labelled pair rows."""
    missing = REQUIRED_PAIR_COLUMNS.difference(pairs.columns)
    if missing:
        raise ValueError(
            f"Pair data is missing required columns: {', '.join(sorted(missing))}."
        )
    if window_minutes <= 0:
        raise ValueError("window_minutes must be greater than zero.")
    if encounter_gap_seconds < 0:
        raise ValueError("encounter_gap_seconds must be zero or greater.")
    if pairs.empty:
        raise ValueError(f"No model-ready pair observations in {session_name}.")
    if not np.isfinite([session_start, session_end]).all() or session_end <= session_start:
        raise ValueError(f"Session {session_name} must span a positive duration.")

    session_start_local = pd.Timestamp(
        session_start, unit="s", tz="UTC"
    ).tz_convert(LOCAL_TIMEZONE)
    duration_hours = (session_end - session_start) / 3600

    observations = pairs[["snapshot_time", "icao24_a", "icao24_b", "collision_risk"]].copy()
    observations["snapshot_time"] = pd.to_numeric(
        observations["snapshot_time"], errors="coerce"
    )
    observations = observations.dropna(subset=["snapshot_time", "collision_risk"])
    observations["collision_risk"] = observations["collision_risk"].astype(int)
    observations["aircraft_a"] = np.minimum(
        observations["icao24_a"], observations["icao24_b"]
    )
    observations["aircraft_b"] = np.maximum(
        observations["icao24_a"], observations["icao24_b"]
    )
    if observations.empty:
        raise ValueError(f"No valid labelled pair observations in {session_name}.")

    aircraft_rows = pd.concat(
        [
            observations[["snapshot_time", "icao24_a", "collision_risk"]].rename(
                columns={"icao24_a": "aircraft"}
            ),
            observations[["snapshot_time", "icao24_b", "collision_risk"]].rename(
                columns={"icao24_b": "aircraft"}
            ),
        ],
        ignore_index=True,
    )
    aircraft_moments = aircraft_rows.groupby(
        ["aircraft", "snapshot_time"], as_index=False
    )["collision_risk"].max()

    positive_pairs = observations.loc[
        observations["collision_risk"].eq(1),
        ["snapshot_time", "aircraft_a", "aircraft_b"],
    ].sort_values(["aircraft_a", "aircraft_b", "snapshot_time"])
    if positive_pairs.empty:
        encounter_count = 0
    else:
        pair_changed = (
            positive_pairs["aircraft_a"].ne(positive_pairs["aircraft_a"].shift())
            | positive_pairs["aircraft_b"].ne(positive_pairs["aircraft_b"].shift())
        )
        resumed_after_gap = (
            positive_pairs.groupby(["aircraft_a", "aircraft_b"])["snapshot_time"].diff()
            > encounter_gap_seconds
        )
        encounter_count = int((pair_changed | resumed_after_gap).sum())

    risky_observations = int(observations["collision_risk"].sum())
    risky_aircraft_moments = int(aircraft_moments["collision_risk"].sum())
    summary: dict[str, object] = {
        "session": session_name,
        "local_date": session_start_local.date().isoformat(),
        "session_start_local": session_start_local.strftime("%Y-%m-%d %I:%M %p %Z"),
        "session_start_time": session_start_local.strftime("%I:%M %p"),
        "session_start_epoch": session_start,
        "session_start_minutes_local": (
            session_start_local.hour * 60 + session_start_local.minute
        ),
        "duration_minutes": round((session_end - session_start) / 60, 1),
        "frames": (
            int(frame_count)
            if frame_count is not None
            else int(observations["snapshot_time"].nunique())
        ),
        "pair_observations": len(observations),
        "risky_pair_observations": risky_observations,
        "pair_observation_rate_pct": 100 * risky_observations / len(observations),
        "aircraft_moments": len(aircraft_moments),
        "risky_aircraft_moments": risky_aircraft_moments,
        "aircraft_moment_rate_pct": (
            100 * risky_aircraft_moments / len(aircraft_moments)
        ),
        "positive_pair_encounters": encounter_count,
        "positive_encounters_per_hour": encounter_count / duration_hours,
        "encounter_gap_seconds": encounter_gap_seconds,
    }

    window_seconds = window_minutes * 60
    observations["window"] = (
        (observations["snapshot_time"] - session_start) // window_seconds
    ).astype(int)
    pair_rates = observations.groupby("window")["collision_risk"].agg(
        pair_observations="size",
        risky_pair_observations="sum",
    )
    aircraft_moments["window"] = (
        (aircraft_moments["snapshot_time"] - session_start) // window_seconds
    ).astype(int)
    aircraft_rates = aircraft_moments.groupby("window")["collision_risk"].agg(
        aircraft_moments="size",
        risky_aircraft_moments="sum",
    )

    windows = pair_rates.join(aircraft_rates, how="outer").reset_index()
    windows.insert(0, "session", session_name)
    windows.insert(1, "session_start_local", summary["session_start_local"])
    windows["window_start_min"] = windows["window"] * window_minutes
    windows["window_start_local"] = [
        (session_start_local + timedelta(minutes=int(offset))).strftime(
            "%Y-%m-%d %I:%M %p %Z"
        )
        for offset in windows["window_start_min"]
    ]
    windows["minutes_covered"] = np.minimum(
        window_minutes,
        np.maximum(
            0,
            (session_end - session_start) / 60 - windows["window_start_min"],
        ),
    ).round(1)
    windows["pair_observation_rate_pct"] = (
        100 * windows["risky_pair_observations"] / windows["pair_observations"]
    )
    windows["aircraft_moment_rate_pct"] = (
        100 * windows["risky_aircraft_moments"] / windows["aircraft_moments"]
    )
    return summary, windows


def analyse_session(
    session_path: Path,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Build, label, and summarize pairs for one raw OpenSky session."""
    states = pd.read_csv(session_path)
    missing = REQUIRED_STATE_COLUMNS.difference(states.columns)
    if missing:
        raise ValueError(
            f"{session_path} is missing required columns: {', '.join(sorted(missing))}."
        )
    snapshot_times = pd.to_numeric(states["snapshot_time"], errors="coerce").dropna()
    if snapshot_times.empty:
        raise ValueError(f"{session_path} contains no valid snapshot timestamps.")

    pairs = build_pair_features(states)
    labelled_pairs = prepare_model_inputs(pairs)
    return summarise_session_pairs(
        labelled_pairs,
        float(snapshot_times.min()),
        float(snapshot_times.max()),
        session_path.name,
        window_minutes=window_minutes,
        frame_count=int(states["snapshot_time"].nunique()),
    )


def plot_time_of_day(
    sessions: pd.DataFrame,
    windows: pd.DataFrame,
    figure_path: Path,
) -> None:
    """Plot session-level rates and five-minute within-session pair rates."""
    ordered = sessions.sort_values("session_start_epoch")
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(len(ordered))
    axes[0].bar(
        x,
        ordered["pair_observation_rate_pct"],
        color="#4477AA",
    )
    axes[0].set_xticks(x, ordered["session_start_time"])
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_title("Risky pair-observation rate")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        x,
        ordered["aircraft_moment_rate_pct"],
        color="#EE6677",
    )
    axes[1].set_xticks(x, ordered["session_start_time"])
    axes[1].set_ylabel("Rate (%)")
    axes[1].set_title("Risky aircraft-moment rate")
    axes[1].grid(axis="y", alpha=0.25)

    for row in ordered.itertuples():
        selected = windows.loc[windows["session"] == row.session].sort_values(
            "window_start_min"
        )
        axes[2].plot(
            selected["window_start_min"],
            selected["pair_observation_rate_pct"],
            marker="o",
            label=row.session_start_time,
        )
    axes[2].set_xlabel("Minutes since session start")
    axes[2].set_ylabel("Risky pair-observation rate (%)")
    axes[2].set_title("Five-minute rates within each session")
    axes[2].legend(title="Session start", frameon=False)
    axes[2].grid(alpha=0.25)

    figure.suptitle("Step 10 — descriptive time-of-day comparison (one date)")
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _default_session_paths() -> list[Path]:
    return sorted(
        path
        for path in DEFAULT_SESSIONS_DIR.glob("states_socal_session_*_*.csv")
        if TIMED_SESSION_NAME.fullmatch(path.name)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sessions",
        nargs="*",
        type=Path,
        help="Timed raw session CSVs; defaults to all time-suffixed sessions.",
    )
    parser.add_argument("--window-minutes", type=int, default=DEFAULT_WINDOW_MINUTES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()

    if args.window_minutes <= 0:
        parser.error("--window-minutes must be greater than zero.")
    session_paths = args.sessions or _default_session_paths()
    if not session_paths:
        parser.error("No timed session files found; pass one or more session CSV paths.")

    summaries = []
    window_tables = []
    for session_path in session_paths:
        summary, windows = analyse_session(session_path, args.window_minutes)
        summaries.append(summary)
        window_tables.append(windows)

    session_table = pd.DataFrame(summaries).sort_values("session_start_epoch")
    window_table = pd.concat(window_tables, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_output = args.output_dir / "step10_time_of_day_sessions.csv"
    window_output = args.output_dir / "step10_time_of_day_windows.csv"
    session_table.to_csv(session_output, index=False)
    window_table.to_csv(window_output, index=False)
    figure_path = args.figure_dir / "step10_time_of_day_rates.png"
    plot_time_of_day(session_table, window_table, figure_path)

    print("Step 10 — descriptive time-of-day comparison")
    for row in session_table.itertuples():
        print(
            f"  {row.session_start_time}: "
            f"pair-observation rate {row.pair_observation_rate_pct:.3f}% "
            f"({row.risky_pair_observations:,}/{row.pair_observations:,}); "
            f"aircraft-moment rate {row.aircraft_moment_rate_pct:.2f}%; "
            f"{row.positive_pair_encounters:,} positive pair-encounters "
            f"({row.positive_encounters_per_hour:.1f}/hour)"
        )
    print(f"Wrote: {session_output}")
    print(f"Wrote: {window_output}")
    print(f"Wrote: {figure_path}")
    print("Interpretation is limited to these samples from one collection date.")


if __name__ == "__main__":
    main()
