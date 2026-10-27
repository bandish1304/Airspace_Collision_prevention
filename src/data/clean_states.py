"""Clean OpenSky state observations for aircraft-pair processing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MAX_POSITION_AGE_SECONDS = 15.0
REQUIRED_COLUMNS = {
    "icao24",
    "latitude",
    "longitude",
    "on_ground",
    "snapshot_time",
    "time_position",
    "velocity",
    "true_track",
}
NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
    "snapshot_time",
    "time_position",
    "velocity",
    "true_track",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATHS = (
    PROJECT_ROOT / "data" / "raw" / "sessions" / "states_socal_backfill_1790375809.csv",
    PROJECT_ROOT / "data" / "raw" / "sessions" / "states_socal_session_1791380397_06-39AM.csv",
    PROJECT_ROOT / "data" / "raw" / "sessions" / "states_socal_session_1791393504_10-18AM.csv",
    PROJECT_ROOT / "data" / "raw" / "sessions" / "states_socal_session_1791408650_02-30PM.csv",
    PROJECT_ROOT / "data" / "raw" / "sessions" / "states_socal_session_1791427160_07-39PM.csv",
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "cleaned_states"
DEFAULT_AUDIT_PATH = PROJECT_ROOT / "data" / "processed" / "state_cleaning_audit.csv"


def clean_state_observations(
    states: pd.DataFrame,
    max_position_age_seconds: float = DEFAULT_MAX_POSITION_AGE_SECONDS,
) -> pd.DataFrame:
    """Return usable airborne observations, retaining one row per aircraft/frame.

    Repeated position reports across different snapshots are retained because
    each snapshot is a distinct prediction time; they are flagged for later
    trajectory-label handling rather than mistaken for new position updates.
    """
    if max_position_age_seconds < 0:
        raise ValueError("max_position_age_seconds must be greater than or equal to zero.")

    missing_columns = REQUIRED_COLUMNS.difference(states.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"State data is missing required columns: {missing}.")

    cleaned = states.copy()
    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["icao24"] = cleaned["icao24"].astype("string").str.strip().str.lower()
    cleaned["on_ground"] = _parse_on_ground(cleaned["on_ground"])
    cleaned["position_age_s"] = cleaned["snapshot_time"] - cleaned["time_position"]

    required = ["icao24", *NUMERIC_COLUMNS, "on_ground"]
    valid = cleaned[required].notna().all(axis=1)
    finite = np.isfinite(cleaned[NUMERIC_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    valid &= finite
    valid &= cleaned["icao24"].ne("")
    valid &= cleaned["icao24"].str.fullmatch(r"[0-9a-f]{6}", na=False)
    valid &= cleaned["latitude"].between(-90, 90)
    valid &= cleaned["longitude"].between(-180, 180)
    valid &= cleaned["velocity"].ge(0)
    valid &= cleaned["true_track"].between(0, 360)
    valid &= ~cleaned["on_ground"]
    valid &= cleaned["position_age_s"].between(0, max_position_age_seconds)

    cleaned = cleaned.loc[valid].copy()
    cleaned = cleaned.drop_duplicates(
        subset=["icao24", "snapshot_time"],
        keep="last",
    )
    cleaned["repeated_position_report"] = cleaned.duplicated(
        subset=["icao24", "time_position"],
        keep="first",
    )
    return cleaned.sort_values(
        ["icao24", "snapshot_time"],
        kind="stable",
    ).reset_index(drop=True)


def _parse_on_ground(values: pd.Series) -> pd.Series:
    """Normalize OpenSky boolean values while preserving unrecognized values as missing."""
    normalized = values.astype("string").str.strip().str.lower()
    return normalized.map(
        {
            "true": True,
            "1": True,
            "1.0": True,
            "false": False,
            "0": False,
            "0.0": False,
        }
    ).astype("boolean")


def _audit_counts(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    max_position_age_seconds: float,
) -> dict[str, int | str]:
    """Summarize row counts without conflating overlapping quality conditions."""
    numeric = raw.copy()
    for column in NUMERIC_COLUMNS:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    age = numeric["snapshot_time"] - numeric["time_position"]
    coordinates_invalid = (
        ~numeric["latitude"].between(-90, 90)
        | ~numeric["longitude"].between(-180, 180)
    )
    kinematics_invalid = (
        numeric["velocity"].lt(0)
        | ~numeric["true_track"].between(0, 360)
    )
    raw_ids = raw["icao24"].astype("string").str.strip().str.lower()
    invalid_ids = raw_ids.notna() & raw_ids.ne("") & ~raw_ids.str.fullmatch(
        r"[0-9a-f]{6}",
        na=False,
    )
    parsed_ground = _parse_on_ground(raw["on_ground"])
    required_values = numeric[NUMERIC_COLUMNS].isna().any(axis=1)
    required_values |= raw_ids.isna() | raw_ids.eq("")
    required_values |= invalid_ids | parsed_ground.isna()
    return {
        "input_file": "",
        "input_rows": len(raw),
        "cleaned_rows": len(cleaned),
        "removed_rows": len(raw) - len(cleaned),
        "missing_or_unparseable_required_rows": int(required_values.sum()),
        "invalid_icao24_rows": int(invalid_ids.sum()),
        "unrecognized_ground_status_rows": int(parsed_ground.isna().sum()),
        "invalid_coordinate_rows": int(coordinates_invalid.fillna(False).sum()),
        "invalid_kinematic_rows": int(kinematics_invalid.fillna(False).sum()),
        "ground_rows": int(_parse_on_ground(raw["on_ground"]).eq(True).sum()),
        "stale_or_future_position_rows": int(
            (~age.between(0, max_position_age_seconds)).fillna(False).sum()
        ),
        "duplicate_aircraft_snapshot_rows": int(
            raw.assign(icao24=raw_ids).duplicated(
                subset=["icao24", "snapshot_time"],
                keep="last",
            ).sum()
        ),
        "repeated_position_report_rows_retained": int(
            cleaned["repeated_position_report"].sum()
        ),
        "missing_altitude_rows_retained": int(
            cleaned.reindex(columns=["baro_altitude", "geo_altitude"])
            .apply(pd.to_numeric, errors="coerce")
            .isna()
            .all(axis=1)
            .sum()
        ),
    }


def clean_session_file(
    input_path: Path,
    output_dir: Path,
    max_position_age_seconds: float = DEFAULT_MAX_POSITION_AGE_SECONDS,
) -> dict[str, int | str]:
    """Clean one session CSV and write it without modifying the raw source."""
    raw = pd.read_csv(input_path)
    cleaned = clean_state_observations(
        raw,
        max_position_age_seconds=max_position_age_seconds,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name
    cleaned.to_csv(output_path, index=False)
    audit = _audit_counts(raw, cleaned, max_position_age_seconds)
    audit["input_file"] = input_path.name
    audit["output_file"] = str(output_path)
    audit["max_position_age_seconds"] = max_position_age_seconds
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_INPUT_PATHS),
        help="Raw OpenSky session CSVs (default: the five modeling sessions).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for cleaned copies; raw files are never modified.",
    )
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=DEFAULT_AUDIT_PATH,
        help="CSV path for per-session cleaning counts.",
    )
    parser.add_argument(
        "--max-position-age-seconds",
        type=float,
        default=DEFAULT_MAX_POSITION_AGE_SECONDS,
        help="Maximum accepted position age (default: 15 seconds).",
    )
    args = parser.parse_args()

    audits = [
        clean_session_file(
            input_path,
            args.output_dir,
            max_position_age_seconds=args.max_position_age_seconds,
        )
        for input_path in args.inputs
    ]
    audit_table = pd.DataFrame(audits)
    args.audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_table.to_csv(args.audit_path, index=False)
    print(audit_table.to_string(index=False))
    print(f"Wrote cleaning audit: {args.audit_path}")


if __name__ == "__main__":
    main()
