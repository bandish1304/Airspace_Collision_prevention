"""Build pair-level kinematic features from an OpenSky snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .kinematics import (
    bearing_difference,
    closing_speed,
    haversine_distance,
    time_to_cpa,
    vertical_separation,
)
from src.data.clean_states import clean_state_observations

METRES_PER_NAUTICAL_MILE = 1_852.0
DEFAULT_MAX_DISTANCE_NM = 50.0
DEFAULT_MAX_POSITION_AGE_SECONDS = 15.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


# Week 2, step 6: build features for all unique aircraft pairs within range.
def build_pair_features(
    states: pd.DataFrame,
    max_distance_nm: float = DEFAULT_MAX_DISTANCE_NM,
    max_position_age_seconds: float = DEFAULT_MAX_POSITION_AGE_SECONDS,
) -> pd.DataFrame:
    """Return one feature row per usable aircraft pair within the distance limit.

    Pairs are formed **within** a snapshot and never across snapshots, so a
    frame sequence from `poll_opensky` or `backfill_opensky` yields one block
    of pairs per frame instead of one collapsed block for the whole file.
    """
    if max_distance_nm <= 0:
        raise ValueError("max_distance_nm must be greater than zero.")
    if max_position_age_seconds < 0:
        raise ValueError("max_position_age_seconds must be greater than or equal to zero.")

    usable_states = clean_state_observations(
        states,
        max_position_age_seconds=max_position_age_seconds,
    )
    maximum_distance_metres = max_distance_nm * METRES_PER_NAUTICAL_MILE
    if usable_states.empty:
        return pd.DataFrame(columns=_pair_feature_columns())

    snapshot_pairs = [
        _pairs_for_snapshot(snapshot_states, maximum_distance_metres)
        for _, snapshot_states in usable_states.groupby("snapshot_time", sort=True)
    ]
    snapshot_pairs = [frame for frame in snapshot_pairs if not frame.empty]
    if not snapshot_pairs:
        return pd.DataFrame(columns=_pair_feature_columns())

    return pd.concat(snapshot_pairs, ignore_index=True)[_pair_feature_columns()]


# Week 2, step 6: build every in-range pair for one snapshot, as whole arrays.
def _pairs_for_snapshot(
    snapshot_states: pd.DataFrame, maximum_distance_metres: float
) -> pd.DataFrame:
    """Return the pair rows for a single frame.

    The Week 2 kinematic functions already broadcast over NumPy arrays, so each
    is called once per frame rather than once per pair. On a 340-aircraft frame
    that replaces roughly 57,000 scalar calls with five array calls.
    """
    snapshot_states = snapshot_states.drop_duplicates(subset="icao24", keep="last")
    aircraft_count = len(snapshot_states)
    if aircraft_count < 2:
        return pd.DataFrame(columns=_pair_feature_columns())

    latitude = snapshot_states["latitude"].to_numpy(dtype=float)
    longitude = snapshot_states["longitude"].to_numpy(dtype=float)
    velocity = snapshot_states["velocity"].to_numpy(dtype=float)
    true_track = snapshot_states["true_track"].to_numpy(dtype=float)
    altitude = _select_altitudes(snapshot_states)
    icao24 = snapshot_states["icao24"].to_numpy()
    position_age = snapshot_states["position_age_s"].to_numpy(dtype=float)
    snapshot_time = snapshot_states["snapshot_time"].to_numpy()

    # Upper-triangle indices name each unordered pair exactly once, in the same
    # order itertools.combinations produced.
    index_a, index_b = np.triu_indices(aircraft_count, k=1)

    lateral_distance_metres = haversine_distance(
        latitude[index_a], longitude[index_a], latitude[index_b], longitude[index_b]
    )

    # Drop out-of-range pairs before computing anything else; across a bounding
    # box this wide most pairs fail the distance test.
    within_range = lateral_distance_metres <= maximum_distance_metres
    if not within_range.any():
        return pd.DataFrame(columns=_pair_feature_columns())
    index_a = index_a[within_range]
    index_b = index_b[within_range]
    lateral_distance_metres = lateral_distance_metres[within_range]

    return pd.DataFrame(
        {
            "snapshot_time": snapshot_time[index_a],
            "icao24_a": icao24[index_a],
            "icao24_b": icao24[index_b],
            "position_age_a_s": position_age[index_a],
            "position_age_b_s": position_age[index_b],
            "latitude_a": latitude[index_a],
            "longitude_a": longitude[index_a],
            "altitude_a_m": altitude[index_a],
            "velocity_a_mps": velocity[index_a],
            "true_track_a_deg": true_track[index_a],
            "latitude_b": latitude[index_b],
            "longitude_b": longitude[index_b],
            "altitude_b_m": altitude[index_b],
            "velocity_b_mps": velocity[index_b],
            "true_track_b_deg": true_track[index_b],
            "lateral_distance_m": lateral_distance_metres,
            "lateral_distance_nm": lateral_distance_metres / METRES_PER_NAUTICAL_MILE,
            "closing_speed_mps": closing_speed(
                latitude[index_a],
                longitude[index_a],
                velocity[index_a],
                true_track[index_a],
                latitude[index_b],
                longitude[index_b],
                velocity[index_b],
                true_track[index_b],
            ),
            "bearing_difference_deg": bearing_difference(
                true_track[index_a], true_track[index_b]
            ),
            "vertical_separation_m": vertical_separation(
                altitude[index_a], altitude[index_b]
            ),
            "time_to_cpa_s": time_to_cpa(
                latitude[index_a],
                longitude[index_a],
                velocity[index_a],
                true_track[index_a],
                latitude[index_b],
                longitude[index_b],
                velocity[index_b],
                true_track[index_b],
            ),
        }
    )


# Week 2, step 6: remove aircraft without fresh, airborne kinematic data.
# Week 2, steps 4 and 6: prefer barometric altitude and fall back to geo altitude.
def _select_altitudes(snapshot_states: pd.DataFrame) -> np.ndarray:
    """Return one altitude per aircraft in metres, NaN where neither is reported."""
    missing = pd.Series(np.nan, index=snapshot_states.index, dtype=float)
    baro_altitude = (
        pd.to_numeric(snapshot_states["baro_altitude"], errors="coerce")
        if "baro_altitude" in snapshot_states
        else missing
    )
    geo_altitude = (
        pd.to_numeric(snapshot_states["geo_altitude"], errors="coerce")
        if "geo_altitude" in snapshot_states
        else missing
    )
    return baro_altitude.fillna(geo_altitude).to_numpy(dtype=float)


# Week 2, step 6: keep an explicit, stable schema for empty and populated outputs.
def _pair_feature_columns() -> list[str]:
    return [
        "snapshot_time",
        "icao24_a",
        "icao24_b",
        "position_age_a_s",
        "position_age_b_s",
        "latitude_a",
        "longitude_a",
        "altitude_a_m",
        "velocity_a_mps",
        "true_track_a_deg",
        "latitude_b",
        "longitude_b",
        "altitude_b_m",
        "velocity_b_mps",
        "true_track_b_deg",
        "lateral_distance_m",
        "lateral_distance_nm",
        "closing_speed_mps",
        "bearing_difference_deg",
        "vertical_separation_m",
        "time_to_cpa_s",
    ]


# Week 2, step 6: run the pair-feature pipeline on a saved OpenSky CSV.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="Path to an OpenSky snapshot CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the pair-level CSV (default: data/processed/).",
    )
    parser.add_argument(
        "--max-distance-nm",
        type=float,
        default=DEFAULT_MAX_DISTANCE_NM,
        help="Maximum lateral separation to retain (default: 50).",
    )
    parser.add_argument(
        "--max-position-age-seconds",
        type=float,
        default=DEFAULT_MAX_POSITION_AGE_SECONDS,
        help="Maximum allowed position age (default: 15).",
    )
    args = parser.parse_args()

    pairs = build_pair_features(
        pd.read_csv(args.input_path),
        max_distance_nm=args.max_distance_nm,
        max_position_age_seconds=args.max_position_age_seconds,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.input_path.stem}_pairs.csv"
    pairs.to_csv(output_path, index=False)

    print(f"Input aircraft  : {len(pd.read_csv(args.input_path))}")
    print(f"Nearby pairs    : {len(pairs)}")
    print(f"Wrote           : {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()