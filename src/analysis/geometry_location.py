"""Step 4: encounter taxonomy, geography, airport proximity, and altitude bands.

Answers the professor's location and trajectory-geometry questions:
"are risky cases associated with crossing paths, convergence, overtaking,
head-on movement?" and "where do risky events occur geographically?".

Classification happens per *encounter*, not per pair-observation -- otherwise
a single long-running formation-style encounter (see step3_single_encounter.png)
would dominate the taxonomy counts the same way it dominates the raw positive
count. Each encounter is summarised by its median bearing difference and
closing speed rather than a single frame, since both fluctuate frame to frame.

Usage:
    python -m src.analysis.geometry_location data/processed/<file>_pairs.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.encounters import (
    DEFAULT_ENCOUNTER_GAP_SECONDS,
    DEFAULT_LATERAL_SEPARATION_NM,
    DEFAULT_VERTICAL_SEPARATION_FT,
    METRES_PER_FOOT,
    READ_CHUNK_ROWS,
    group_by_pair_over_time,
    load_positive_pairs,
)
from src.features.kinematics import haversine_distance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "eda"

METRES_PER_NAUTICAL_MILE = 1_852.0

# Band edges in feet. 3,000 ft is a common pattern-altitude ceiling near towered
# airports; 18,000 ft is the floor of Class A airspace, where operations shift
# from mixed VFR/IFR to IFR-only.
ALTITUDE_BAND_EDGES_FT = [-float("inf"), 3_000.0, 10_000.0, 18_000.0, float("inf")]
ALTITUDE_BAND_LABELS = [
    "ground/pattern (<3,000 ft)",
    "low (3,000-10,000 ft)",
    "mid (10,000-18,000 ft)",
    "high (>=18,000 ft)",
]

# Approximate reference points for the airports the SoCal bounding box was
# chosen to cover (see fetch_opensky.SOCAL_BBOX). Precise to airport-proximity
# purposes, not to runway-threshold precision.
AIRPORTS = {
    "LAX": (33.9425, -118.4081),
    "SAN": (32.7338, -117.1933),
    "ONT": (34.0560, -117.6012),
    "BUR": (34.2007, -118.3590),
    "SNA": (33.6757, -117.8682),
    "LGB": (33.8177, -118.1516),
}

# Boundaries between geometry categories. Judgement calls, not physical
# constants -- chosen so each category has a clear physical reading rather
# than to fit the data.
HEAD_ON_BEARING_DEG = 150.0
OVERTAKE_BEARING_DEG = 30.0
FORMATION_MIN_DURATION_S = 300.0
FORMATION_MAX_BEARING_DEG = 20.0
FORMATION_MAX_CLOSING_SPEED_MPS = 5.0


# Step 4: one row per encounter, carrying the geometry and location needed to classify it.
def summarise_encounter_geometry(
    positives: pd.DataFrame, encounter_id: pd.Series
) -> pd.DataFrame:
    """Return one row per encounter with median geometry and mean position.

    ``aircraft_a``/``aircraft_b`` are normalised to (min, max) order, matching
    the convention ``encounters.group_by_pair_over_time`` uses internally --
    the raw ``icao24_a``/``icao24_b`` columns are not ordered consistently,
    since which aircraft lands in which column depends on iteration order
    within a snapshot, not on identity.
    """
    working = positives.copy()
    working["encounter_id"] = encounter_id
    working["aircraft_a"] = np.minimum(working["icao24_a"], working["icao24_b"])
    working["aircraft_b"] = np.maximum(working["icao24_a"], working["icao24_b"])
    working["midpoint_latitude"] = (working["latitude_a"] + working["latitude_b"]) / 2
    working["midpoint_longitude"] = (working["longitude_a"] + working["longitude_b"]) / 2

    summary = working.groupby("encounter_id").agg(
        aircraft_a=("aircraft_a", "first"),
        aircraft_b=("aircraft_b", "first"),
        observations=("snapshot_time", "size"),
        start_time=("snapshot_time", "min"),
        end_time=("snapshot_time", "max"),
        median_bearing_difference_deg=("bearing_difference_deg", "median"),
        median_closing_speed_mps=("closing_speed_mps", "median"),
        median_lateral_distance_nm=("lateral_distance_nm", "median"),
        median_altitude_a_m=("altitude_a_m", "median"),
        median_altitude_b_m=("altitude_b_m", "median"),
        mean_latitude=("midpoint_latitude", "mean"),
        mean_longitude=("midpoint_longitude", "mean"),
    ).reset_index()
    summary["duration_s"] = summary["end_time"] - summary["start_time"]
    return summary


# Step 4: classify each encounter's geometry from its median bearing and closing speed.
def classify_geometry(encounters: pd.DataFrame) -> pd.Series:
    """Return a geometry category per encounter.

    Precedence matters: a sustained, near-parallel, near-zero-closing-speed
    encounter is labelled formation even though it would otherwise also pass
    the overtaking test, because duration and near-zero net closure are the
    stronger signal that this is not a developing conflict.
    """
    bearing = encounters["median_bearing_difference_deg"]
    closing = encounters["median_closing_speed_mps"]
    duration = encounters["duration_s"]

    category = pd.Series("crossing", index=encounters.index)
    category[closing <= 0] = "separating"
    category[(closing > 0) & (bearing <= OVERTAKE_BEARING_DEG)] = "overtaking"
    category[(closing > 0) & (bearing >= HEAD_ON_BEARING_DEG)] = "head-on"

    formation = (
        (duration >= FORMATION_MIN_DURATION_S)
        & (bearing <= FORMATION_MAX_BEARING_DEG)
        & (closing.abs() <= FORMATION_MAX_CLOSING_SPEED_MPS)
    )
    category[formation] = "parallel / formation"
    return category


# Step 4: attach each encounter's nearest airport and distance to it.
def nearest_airport(latitude: pd.Series, longitude: pd.Series) -> pd.DataFrame:
    """Return the nearest airport code and distance in nm, per row.

    Six airports and a few thousand encounters is small enough to just compute
    every encounter-to-airport distance and take the minimum, rather than
    building a spatial index.
    """
    distances_nm = {
        code: haversine_distance(latitude, longitude, airport_lat, airport_lon)
        / METRES_PER_NAUTICAL_MILE
        for code, (airport_lat, airport_lon) in AIRPORTS.items()
    }
    distance_table = pd.DataFrame(distances_nm, index=latitude.index)
    return pd.DataFrame(
        {
            "nearest_airport": distance_table.idxmin(axis=1),
            "airport_distance_nm": distance_table.min(axis=1),
        }
    )


# Step 4: risk rate per altitude band, over every pair -- not just the positives.
def altitude_band_risk_rates(
    pairs_path: Path,
    lateral_separation_nm: float = DEFAULT_LATERAL_SEPARATION_NM,
    vertical_separation_ft: float = DEFAULT_VERTICAL_SEPARATION_FT,
) -> pd.DataFrame:
    """Return total pairs, risky pairs, and risk rate for each altitude band.

    Risk rate needs a denominator of *all* pairs, so this reads the whole file
    in chunks rather than the positives-only loader used elsewhere. Each pair
    is assigned to a band by its mean altitude, since a pair close enough to
    breach vertical separation is close enough that averaging the two rarely
    changes the band.
    """
    vertical_limit_m = vertical_separation_ft * METRES_PER_FOOT
    totals = pd.Series(0, index=ALTITUDE_BAND_LABELS, dtype="int64")
    positives = pd.Series(0, index=ALTITUDE_BAND_LABELS, dtype="int64")

    columns = [
        "altitude_a_m", "altitude_b_m", "lateral_distance_nm", "vertical_separation_m",
    ]
    for chunk in pd.read_csv(pairs_path, usecols=columns, chunksize=READ_CHUNK_ROWS):
        mean_altitude_ft = (chunk["altitude_a_m"] + chunk["altitude_b_m"]) / 2 / METRES_PER_FOOT
        band = pd.cut(mean_altitude_ft, ALTITUDE_BAND_EDGES_FT, labels=ALTITUDE_BAND_LABELS)
        breach = (chunk["lateral_distance_nm"] <= lateral_separation_nm) & (
            chunk["vertical_separation_m"] <= vertical_limit_m
        )
        totals = totals.add(band.value_counts(), fill_value=0)
        positives = positives.add(band[breach].value_counts(), fill_value=0)

    summary = pd.DataFrame({"total_pairs": totals, "risky_pairs": positives})
    summary = summary.reindex(ALTITUDE_BAND_LABELS)
    summary["risk_rate_pct"] = (
        summary["risky_pairs"] / summary["total_pairs"].replace(0, pd.NA) * 100
    )
    return summary.reset_index(names="altitude_band")


# Step 4: pick one representative encounter per geometry category, for trajectory plots.
def select_representative_encounters(
    encounters: pd.DataFrame, categories: tuple[str, ...]
) -> pd.DataFrame:
    """Return the most-observed encounter in each requested category.

    More observations means a longer, clearer path to plot; the single busiest
    encounter in each category is picked rather than a random one so the
    figure does not land on a two-frame blip.
    """
    rows = []
    for category in categories:
        in_category = encounters[encounters["geometry"] == category]
        if in_category.empty:
            continue
        rows.append(in_category.sort_values("observations", ascending=False).iloc[0])
    return pd.DataFrame(rows)


# Step 4: find a genuine near-miss -- converged close but never breached both limits.
def find_safe_example_pair(
    pairs_path: Path,
    lateral_separation_nm: float = DEFAULT_LATERAL_SEPARATION_NM,
    vertical_separation_ft: float = DEFAULT_VERTICAL_SEPARATION_FT,
    near_miss_max_nm: float = 12.0,
    min_observations: int = 10,
) -> tuple[str, str] | None:
    """Return the (aircraft_a, aircraft_b) pair with the closest safe approach.

    A fair visual contrast against the risky examples needs a pair that
    actually came close, not an arbitrary distant one, so every pair in the
    file is scanned for its closest approach and observation count.
    """
    vertical_limit_m = vertical_separation_ft * METRES_PER_FOOT
    columns = [
        "snapshot_time", "icao24_a", "icao24_b", "lateral_distance_nm", "vertical_separation_m",
    ]
    minima_per_chunk = []
    observation_counts = pd.Series(
        dtype="int64",
        index=pd.MultiIndex.from_arrays([[], []], names=["aircraft_a", "aircraft_b"]),
    )
    ever_breached: set[tuple[str, str]] = set()

    for chunk in pd.read_csv(pairs_path, usecols=columns, chunksize=READ_CHUNK_ROWS):
        chunk = chunk.assign(
            aircraft_a=np.minimum(chunk["icao24_a"], chunk["icao24_b"]),
            aircraft_b=np.maximum(chunk["icao24_a"], chunk["icao24_b"]),
        )
        breach = (chunk["lateral_distance_nm"] <= lateral_separation_nm) & (
            chunk["vertical_separation_m"] <= vertical_limit_m
        )
        ever_breached.update(
            chunk.loc[breach, ["aircraft_a", "aircraft_b"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        observation_counts = observation_counts.add(
            chunk.groupby(["aircraft_a", "aircraft_b"]).size(), fill_value=0
        )
        closest_index = chunk.groupby(["aircraft_a", "aircraft_b"])["lateral_distance_nm"].idxmin()
        minima_per_chunk.append(
            chunk.loc[closest_index, ["aircraft_a", "aircraft_b", "lateral_distance_nm"]]
        )

    combined = pd.concat(minima_per_chunk, ignore_index=True)
    closest = combined.loc[
        combined.groupby(["aircraft_a", "aircraft_b"])["lateral_distance_nm"].idxmin()
    ].set_index(["aircraft_a", "aircraft_b"])
    closest["observations"] = observation_counts

    candidates = closest[
        (closest["lateral_distance_nm"] > lateral_separation_nm)
        & (closest["lateral_distance_nm"] <= near_miss_max_nm)
        & (closest["observations"] >= min_observations)
    ]
    candidates = candidates.loc[~candidates.index.isin(ever_breached)]
    if candidates.empty:
        return None
    return candidates.sort_values("lateral_distance_nm").index[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs_path", type=Path, help="Path to a pair-feature CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--encounter-gap-seconds", type=int, default=DEFAULT_ENCOUNTER_GAP_SECONDS
    )
    args = parser.parse_args()

    positives, examined = load_positive_pairs(args.pairs_path)
    if positives.empty:
        print("No positive pair-observations found.")
        return

    encounter_id = group_by_pair_over_time(positives, args.encounter_gap_seconds)
    encounters = summarise_encounter_geometry(positives, encounter_id)
    encounters["geometry"] = classify_geometry(encounters)
    encounters = pd.concat(
        [encounters, nearest_airport(encounters["mean_latitude"], encounters["mean_longitude"])],
        axis=1,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "step4_encounter_geometry.csv"
    encounters.to_csv(output_path, index=False)

    print(f"Pair-observations examined : {examined:,}")
    print(f"Distinct encounters        : {len(encounters):,}")
    print("\nGeometry category counts:")
    print(encounters["geometry"].value_counts().to_string())

    print("\nNearest-airport counts:")
    print(encounters["nearest_airport"].value_counts().to_string())
    within_5nm = (encounters["airport_distance_nm"] <= 5).mean() * 100
    within_10nm = (encounters["airport_distance_nm"] <= 10).mean() * 100
    print(f"\nMedian distance to nearest airport : {encounters['airport_distance_nm'].median():.1f} nm")
    print(f"Within 5 nm of an airport           : {within_5nm:.1f}%")
    print(f"Within 10 nm of an airport          : {within_10nm:.1f}%")
    print(f"\nWrote {output_path.relative_to(PROJECT_ROOT)}")

    band_summary = altitude_band_risk_rates(args.pairs_path)
    band_output_path = args.output_dir / "step4_altitude_band_risk_rates.csv"
    band_summary.to_csv(band_output_path, index=False)
    print("\nRisk rate by altitude band (all pairs, not just positives):")
    print(band_summary.to_string(index=False))
    print(f"\nWrote {band_output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
