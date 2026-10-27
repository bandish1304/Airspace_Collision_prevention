"""Label prediction-time aircraft pairs using later observed states."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.clean_states import DEFAULT_OUTPUT_DIR as CLEANED_STATES_DIR
from src.data.clean_states import clean_state_observations
from src.features.build_pipeline import DEFAULT_MAX_DISTANCE_NM, build_pair_features
from src.features.model_inputs import TIME_TO_CPA_CAP_SECONDS, feature_names
from src.labeling.label_pairs import label_collision_risk

DEFAULT_FORECAST_HORIZON_SECONDS = 60.0
DEFAULT_MAX_TRACKING_GAP_SECONDS = 30.0
DEFAULT_LATERAL_SEPARATION_NM = 5.0
DEFAULT_VERTICAL_SEPARATION_FT = 1_000.0
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "future_labels"
DEFAULT_SESSION_NAMES = (
    "states_socal_backfill_1790375809.csv",
    "states_socal_session_1791380397_06-39AM.csv",
    "states_socal_session_1791393504_10-18AM.csv",
    "states_socal_session_1791408650_02-30PM.csv",
    "states_socal_session_1791427160_07-39PM.csv",
)
FUTURE_LABEL_COLUMN = "future_conflict"
LABEL_STATUS_COLUMN = "future_conflict_status"
METADATA_COLUMNS = ["snapshot_time", "icao24_a", "icao24_b", LABEL_STATUS_COLUMN]


def iter_future_conflict_labels(
    states: pd.DataFrame,
    *,
    forecast_horizon_seconds: float = DEFAULT_FORECAST_HORIZON_SECONDS,
    max_tracking_gap_seconds: float = DEFAULT_MAX_TRACKING_GAP_SECONDS,
    max_distance_nm: float = DEFAULT_MAX_DISTANCE_NM,
    lateral_separation_nm: float = DEFAULT_LATERAL_SEPARATION_NM,
    vertical_separation_ft: float = DEFAULT_VERTICAL_SEPARATION_FT,
) -> Iterator[pd.DataFrame]:
    """Yield one snapshot's current pair features and future-onset labels.

    Future observations are inspected only to assign the target and status.
    They are never copied into the yielded feature rows. A new future position
    report is required to establish an onset, and a negative requires paired
    aircraft observations spanning the horizon with no excessive gap.
    """
    if forecast_horizon_seconds <= 0:
        raise ValueError("forecast_horizon_seconds must be greater than zero.")
    if max_tracking_gap_seconds <= 0:
        raise ValueError("max_tracking_gap_seconds must be greater than zero.")

    cleaned = clean_state_observations(states)
    if cleaned.empty:
        return

    frames = [
        frame
        for _, frame in cleaned.groupby("snapshot_time", sort=True)
    ]
    times = np.array(
        [float(frame["snapshot_time"].iloc[0]) for frame in frames],
        dtype=np.float64,
    )
    aircraft_by_time = [
        set(frame["icao24"].astype(str))
        for frame in frames
    ]
    positive_pair_states_by_time: list[
        dict[tuple[str, str], set[tuple[float, float]]]
    ] = []

    for frame in frames:
        pair_features = build_pair_features(frame, max_distance_nm=max_distance_nm)
        labels = label_collision_risk(
            pair_features,
            lateral_separation_nm=lateral_separation_nm,
            vertical_separation_ft=vertical_separation_ft,
        )
        positive_states: dict[
            tuple[str, str], set[tuple[float, float]]
        ] = {}
        positive_rows = labels.loc[
            labels["collision_risk"].eq(1),
            [
                "snapshot_time",
                "icao24_a",
                "icao24_b",
                "position_age_a_s",
                "position_age_b_s",
            ],
        ]
        for row in positive_rows.itertuples(index=False):
            pair_key, report_times = _pair_state_key(
                row.icao24_a,
                row.icao24_b,
                float(row.snapshot_time - row.position_age_a_s),
                float(row.snapshot_time - row.position_age_b_s),
            )
            positive_states.setdefault(pair_key, set()).add(report_times)
        positive_pair_states_by_time.append(positive_states)

    for frame_index, (snapshot_time, frame) in enumerate(zip(times, frames, strict=True)):
        current_pairs = build_pair_features(frame, max_distance_nm=max_distance_nm)
        if current_pairs.empty:
            continue

        current_labels = label_collision_risk(
            current_pairs,
            lateral_separation_nm=lateral_separation_nm,
            vertical_separation_ft=vertical_separation_ft,
        )["collision_risk"]
        horizon_end = snapshot_time + forecast_horizon_seconds
        coverage_end = int(np.searchsorted(times, horizon_end, side="left"))
        current_report_time = {
            str(row.icao24): float(row.time_position)
            for row in frame[["icao24", "time_position"]].itertuples(index=False)
        }
        labels: list[object] = []
        statuses: list[str] = []

        for pair_position, pair in enumerate(
            current_pairs[["icao24_a", "icao24_b"]].itertuples(index=False, name=None)
        ):
            pair_key = _pair_key(*pair)
            current_state_key = _pair_state_key(
                pair[0],
                pair[1],
                current_report_time[str(pair[0])],
                current_report_time[str(pair[1])],
            )
            current_label = current_labels.iloc[pair_position]
            if pd.isna(current_label):
                labels.append(pd.NA)
                statuses.append("unknown_current_separation")
                continue
            if int(current_label) == 1:
                labels.append(pd.NA)
                statuses.append("ongoing_excluded")
                continue

            observed_conflict = False
            last_joint_observation = snapshot_time
            tracking_gap = False
            coverage_indices = range(
                frame_index + 1,
                min(coverage_end + 1, len(frames)),
            )
            for future_index in coverage_indices:
                future_time = times[future_index]
                aircraft = aircraft_by_time[future_index]
                if pair[0] not in aircraft or pair[1] not in aircraft:
                    continue

                if future_time - last_joint_observation > max_tracking_gap_seconds:
                    tracking_gap = True
                last_joint_observation = future_time

                has_new_position_state = any(
                    report_time_a > current_state_key[1][0]
                    or report_time_b > current_state_key[1][1]
                    for report_time_a, report_time_b in
                    positive_pair_states_by_time[future_index].get(pair_key, set())
                )
                if future_time <= horizon_end and has_new_position_state:
                    observed_conflict = True
                    break

            if observed_conflict:
                labels.append(1)
                statuses.append("positive")
            elif (
                coverage_end < len(times)
                and last_joint_observation >= horizon_end
                and not tracking_gap
            ):
                labels.append(0)
                statuses.append("negative")
            else:
                labels.append(pd.NA)
                statuses.append("unknown_incomplete_followup")

        current_pairs[FUTURE_LABEL_COLUMN] = pd.array(labels, dtype="Int8")
        current_pairs[LABEL_STATUS_COLUMN] = statuses
        yield current_pairs


def _pair_state_key(
    first: object,
    second: object,
    first_time: float,
    second_time: float,
) -> tuple[tuple[str, str], tuple[float, float]]:
    first_id, second_id = str(first), str(second)
    if first_id <= second_id:
        return (first_id, second_id), (first_time, second_time)
    return (second_id, first_id), (second_time, first_time)


def _pair_key(first: object, second: object) -> tuple[str, str]:
    return _pair_state_key(first, second, 0.0, 0.0)[0]


def prepare_future_model_inputs(
    labeled_pairs: pd.DataFrame,
    *,
    include_label_features: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Separate prediction-time features, known targets, and non-feature metadata."""
    features = feature_names(include_label_features)
    required = [
        *[
            column
            for column in features
            if column not in {"log_time_to_cpa", "has_cpa"}
        ],
        "time_to_cpa_s",
        FUTURE_LABEL_COLUMN,
        LABEL_STATUS_COLUMN,
        *METADATA_COLUMNS[:-1],
    ]
    missing = [column for column in required if column not in labeled_pairs.columns]
    if missing:
        raise ValueError(f"Labeled pair data is missing required columns: {missing}.")

    known = (
        labeled_pairs[FUTURE_LABEL_COLUMN].notna()
        & labeled_pairs[LABEL_STATUS_COLUMN].isin(["positive", "negative"])
    )
    selected = labeled_pairs.loc[known]
    time_to_cpa = pd.to_numeric(selected["time_to_cpa_s"], errors="coerce").to_numpy(
        dtype=float
    )
    x = selected.copy()
    x["has_cpa"] = np.isfinite(time_to_cpa).astype(float)
    capped = np.minimum(
        np.nan_to_num(time_to_cpa, posinf=TIME_TO_CPA_CAP_SECONDS),
        TIME_TO_CPA_CAP_SECONDS,
    )
    x["log_time_to_cpa"] = np.log1p(capped)
    x = x[features]
    finite = np.isfinite(x.to_numpy(dtype=float)).all(axis=1)
    x = x.iloc[np.flatnonzero(finite)].copy()
    y = selected[FUTURE_LABEL_COLUMN].astype(int).iloc[np.flatnonzero(finite)].copy()
    metadata = selected[METADATA_COLUMNS].iloc[np.flatnonzero(finite)].copy()
    return x, y, metadata


def label_session_file(
    input_path: Path,
    output_path: Path,
    *,
    forecast_horizon_seconds: float = DEFAULT_FORECAST_HORIZON_SECONDS,
    max_tracking_gap_seconds: float = DEFAULT_MAX_TRACKING_GAP_SECONDS,
    max_distance_nm: float = DEFAULT_MAX_DISTANCE_NM,
    lateral_separation_nm: float = DEFAULT_LATERAL_SEPARATION_NM,
    vertical_separation_ft: float = DEFAULT_VERTICAL_SEPARATION_FT,
) -> dict[str, int | str | float]:
    """Label one raw/cleaned session and stream the result to a CSV."""
    states = pd.read_csv(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    counts = {
        "positive": 0,
        "negative": 0,
        "ongoing_excluded": 0,
        "unknown_current_separation": 0,
        "unknown_incomplete_followup": 0,
    }
    rows_written = 0
    wrote_header = False
    for chunk in iter_future_conflict_labels(
        states,
        forecast_horizon_seconds=forecast_horizon_seconds,
        max_tracking_gap_seconds=max_tracking_gap_seconds,
        max_distance_nm=max_distance_nm,
        lateral_separation_nm=lateral_separation_nm,
        vertical_separation_ft=vertical_separation_ft,
    ):
        chunk.to_csv(output_path, mode="a", index=False, header=not wrote_header)
        wrote_header = True
        rows_written += len(chunk)
        chunk_counts = chunk[LABEL_STATUS_COLUMN].value_counts()
        for status, count in chunk_counts.items():
            counts[str(status)] += int(count)

    if not wrote_header:
        output_path.write_text("", encoding="utf-8")
    return {
        "input_file": input_path.name,
        "output_file": str(output_path),
        "rows": rows_written,
        **counts,
        "forecast_horizon_seconds": forecast_horizon_seconds,
        "max_tracking_gap_seconds": max_tracking_gap_seconds,
        "max_distance_nm": max_distance_nm,
        "lateral_separation_nm": lateral_separation_nm,
        "vertical_separation_ft": vertical_separation_ft,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[CLEANED_STATES_DIR / name for name in DEFAULT_SESSION_NAMES],
        help="Cleaned session CSVs; each session is labeled independently.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--forecast-horizon-seconds",
        type=float,
        default=DEFAULT_FORECAST_HORIZON_SECONDS,
    )
    parser.add_argument(
        "--max-tracking-gap-seconds",
        type=float,
        default=DEFAULT_MAX_TRACKING_GAP_SECONDS,
    )
    parser.add_argument("--max-distance-nm", type=float, default=DEFAULT_MAX_DISTANCE_NM)
    parser.add_argument(
        "--lateral-separation-nm",
        type=float,
        default=DEFAULT_LATERAL_SEPARATION_NM,
    )
    parser.add_argument(
        "--vertical-separation-ft",
        type=float,
        default=DEFAULT_VERTICAL_SEPARATION_FT,
    )
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "future_label_audit.csv",
    )
    args = parser.parse_args()

    audits = []
    for input_path in args.inputs:
        output_path = args.output_dir / f"{input_path.stem}_future_labels.csv"
        audits.append(
            label_session_file(
                input_path,
                output_path,
                forecast_horizon_seconds=args.forecast_horizon_seconds,
                max_tracking_gap_seconds=args.max_tracking_gap_seconds,
                max_distance_nm=args.max_distance_nm,
                lateral_separation_nm=args.lateral_separation_nm,
                vertical_separation_ft=args.vertical_separation_ft,
            )
        )
    audit = pd.DataFrame(audits)
    args.audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.audit_path, index=False)
    print(audit.to_string(index=False))
    print(f"Wrote future-label audit: {args.audit_path}")


if __name__ == "__main__":
    main()
