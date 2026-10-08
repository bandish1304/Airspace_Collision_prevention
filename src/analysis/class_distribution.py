"""Step 8: how rare risk is, counted at several levels.

Answers the progress feedback's request to "go beyond the overall 0.66% and
examine risk by aircraft, flight/trajectory, time window, geographic region,
and other meaningful groupings".

The positive rate is not a property of the data alone; it depends on what is
counted as one case. Six levels are measured here, each with an explicit unit:

    pair-observation  one pair at one moment, risky if it breaks the rule then
    visit             one pair for one continuous stretch within range, risky
                      if it breaks the rule at any point during it
    aircraft pair     one pair over the whole window, risky if it ever does
    aircraft-moment   one aircraft at one moment, risky if it is in at least
                      one risky pair then
    trajectory        one aircraft for one continuous stretch of observation,
                      risky if it is ever in a risky pair
    aircraft          one aircraft over the whole window, same test

Then by time window and by map cell, where the denominator is all pairs in
that window or cell. Step 4 mapped positives alone, which shows where traffic
is as much as where risk is; a rate per cell separates the two.

Aircraft are counted only when they appear in at least one pair, i.e. within
50 nm of another aircraft. A few isolated aircraft are therefore left out.

Usage:
    python -m src.analysis.class_distribution
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.data import DEFAULT_PAIRS_PATH
from src.features.model_inputs import prepare_model_inputs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "eda"

# A pair or aircraft unseen for longer than this starts a new visit or
# trajectory. Frames are 10 s apart; the stale-position filter can drop an
# aircraft for a frame or two, which should not split it.
VISIT_GAP_SECONDS = 60
TRAJECTORY_GAP_SECONDS = 120

WINDOW_MINUTES = 5

# Map cells of a quarter degree, roughly 15 nm across at this latitude. Cells
# with few observations give unstable rates and are reported but not mapped.
CELL_DEGREES = 0.25
MIN_CELL_OBSERVATIONS = 2_000

PAIR_COLUMNS = [
    "snapshot_time", "icao24_a", "icao24_b",
    "latitude_a", "longitude_a", "latitude_b", "longitude_b",
    "lateral_distance_nm", "vertical_separation_m", "closing_speed_mps",
    "bearing_difference_deg", "time_to_cpa_s",
]

LEVEL_ORDER = [
    "pair-observation", "visit", "aircraft pair",
    "aircraft-moment", "trajectory", "aircraft",
]


# Step 8: labelled pairs with an unordered pair key and a map position.
def load_pairs(pairs_path: Path = DEFAULT_PAIRS_PATH) -> pd.DataFrame:
    pairs = prepare_model_inputs(pd.read_csv(pairs_path, usecols=PAIR_COLUMNS))
    return pd.DataFrame(
        {
            "snapshot_time": pairs["snapshot_time"].to_numpy(),
            "aircraft_a": np.minimum(pairs["icao24_a"], pairs["icao24_b"]).to_numpy(),
            "aircraft_b": np.maximum(pairs["icao24_a"], pairs["icao24_b"]).to_numpy(),
            "latitude": ((pairs["latitude_a"] + pairs["latitude_b"]) / 2).to_numpy(),
            "longitude": ((pairs["longitude_a"] + pairs["longitude_b"]) / 2).to_numpy(),
            "risky": pairs["collision_risk"].to_numpy(),
        }
    )


def _runs(keys: pd.DataFrame, times: pd.Series, gap_seconds: float) -> pd.Series:
    """Number contiguous runs of the same key, breaking where time jumps by more than the gap."""
    changed = (keys != keys.shift()).any(axis=1)
    return (changed | (times.diff() > gap_seconds)).cumsum()


# Step 8: one row per aircraft per frame, risky if any of its pairs is.
def aircraft_moments(pairs: pd.DataFrame) -> pd.DataFrame:
    both_ends = pd.concat(
        [
            pairs[["snapshot_time", "aircraft_a", "risky"]].rename(columns={"aircraft_a": "aircraft"}),
            pairs[["snapshot_time", "aircraft_b", "risky"]].rename(columns={"aircraft_b": "aircraft"}),
        ],
        ignore_index=True,
    )
    return both_ends.groupby(["aircraft", "snapshot_time"], as_index=False)["risky"].max()


# Step 8: the positive rate at each level of counting.
def rates_by_level(pairs: pd.DataFrame, moments: pd.DataFrame) -> pd.DataFrame:
    ordered = pairs.sort_values(["aircraft_a", "aircraft_b", "snapshot_time"])
    visit = _runs(ordered[["aircraft_a", "aircraft_b"]], ordered["snapshot_time"],
                  VISIT_GAP_SECONDS)
    visits = ordered.groupby(visit)["risky"].max()
    pair_level = pairs.groupby(["aircraft_a", "aircraft_b"])["risky"].max()

    tracked = moments.sort_values(["aircraft", "snapshot_time"])
    trajectory = _runs(tracked[["aircraft"]], tracked["snapshot_time"], TRAJECTORY_GAP_SECONDS)
    trajectories = tracked.groupby(trajectory)["risky"].max()
    aircraft_level = moments.groupby("aircraft")["risky"].max()

    counts = {
        "pair-observation": pairs["risky"],
        "visit": visits,
        "aircraft pair": pair_level,
        "aircraft-moment": moments["risky"],
        "trajectory": trajectories,
        "aircraft": aircraft_level,
    }
    return pd.DataFrame(
        [
            {"level": level, "risky": int(values.sum()), "total": len(values),
             "rate_pct": 100 * values.mean()}
            for level, values in counts.items()
        ]
    )


# Step 8: rate per time window, over pairs and over aircraft.
def rates_by_window(pairs: pd.DataFrame, moments: pd.DataFrame) -> pd.DataFrame:
    start = pairs["snapshot_time"].min()
    window_seconds = WINDOW_MINUTES * 60

    def window_of(times: pd.Series) -> pd.Series:
        return (times - start) // window_seconds

    pair_rates = pairs.groupby(window_of(pairs["snapshot_time"]))["risky"].agg(["sum", "size"])
    # An aircraft counts once per window, as risky if it was risky at any moment in it.
    per_aircraft = moments.assign(window=window_of(moments["snapshot_time"])).groupby(
        ["window", "aircraft"])["risky"].max()
    aircraft_rates = per_aircraft.groupby("window").agg(["sum", "size"])

    table = pd.DataFrame(
        {
            "window_start_min": pair_rates.index * WINDOW_MINUTES,
            "pair_observations": pair_rates["size"].to_numpy(),
            "risky_observations": pair_rates["sum"].to_numpy(),
            "observation_rate_pct": (100 * pair_rates["sum"] / pair_rates["size"]).to_numpy(),
            "aircraft": aircraft_rates["size"].to_numpy(),
            "risky_aircraft": aircraft_rates["sum"].to_numpy(),
            "aircraft_rate_pct": (100 * aircraft_rates["sum"] / aircraft_rates["size"]).to_numpy(),
        }
    )
    # A window cut short by the end of the collection is flagged, not hidden.
    table["minutes_covered"] = np.minimum(
        WINDOW_MINUTES,
        (pairs["snapshot_time"].max() - start) / 60 - table["window_start_min"],
    ).round(1)
    return table


# Step 8: rate per map cell, using every pair in the cell as the denominator.
def rates_by_region(pairs: pd.DataFrame) -> pd.DataFrame:
    cell_lat = np.floor(pairs["latitude"] / CELL_DEGREES) * CELL_DEGREES
    cell_lon = np.floor(pairs["longitude"] / CELL_DEGREES) * CELL_DEGREES
    grouped = pairs.assign(cell_lat=cell_lat, cell_lon=cell_lon).groupby(["cell_lat", "cell_lon"])

    table = grouped.agg(
        observations=("risky", "size"),
        risky_observations=("risky", "sum"),
    ).reset_index()
    distinct = grouped.apply(
        lambda cell: cell.loc[cell["risky"] == 1, ["aircraft_a", "aircraft_b"]]
        .drop_duplicates().shape[0],
        include_groups=False,
    )
    table["risky_aircraft_pairs"] = distinct.to_numpy()
    table["rate_pct"] = 100 * table["risky_observations"] / table["observations"]
    table["mapped"] = table["observations"] >= MIN_CELL_OBSERVATIONS
    return table


# Step 8: how much of its observed time each aircraft spends in a risky pair.
def aircraft_involvement(pairs: pd.DataFrame, moments: pd.DataFrame) -> pd.DataFrame:
    observed = moments.groupby("aircraft").agg(
        frames_observed=("risky", "size"), risky_frames=("risky", "sum")
    )
    risky_pairs = pairs.loc[pairs["risky"] == 1, ["aircraft_a", "aircraft_b"]].drop_duplicates()
    partners = pd.concat([risky_pairs["aircraft_a"], risky_pairs["aircraft_b"]]).value_counts()
    observed["risky_partners"] = partners.reindex(observed.index, fill_value=0)
    observed["share_of_time_risky_pct"] = 100 * observed["risky_frames"] / observed["frames_observed"]
    return observed.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    pairs = load_pairs(args.pairs_path)
    moments = aircraft_moments(pairs)

    outputs = {
        "step8_rates_by_level.csv": rates_by_level(pairs, moments),
        "step8_rates_by_window.csv": rates_by_window(pairs, moments),
        "step8_rates_by_region.csv": rates_by_region(pairs),
        "step8_aircraft_involvement.csv": aircraft_involvement(pairs, moments),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        table.to_csv(args.output_dir / name, index=False)

    levels = outputs["step8_rates_by_level.csv"]
    print("Positive rate at each level of counting:")
    for row in levels.itertuples():
        print(f"  {row.level:<17} {row.risky:>9,} of {row.total:>11,}   {row.rate_pct:6.2f}%")

    regions = outputs["step8_rates_by_region.csv"]
    mapped = regions.loc[regions["mapped"]]
    print(f"\nMap cells: {len(regions)} occupied, {len(mapped)} with at least "
          f"{MIN_CELL_OBSERVATIONS:,} observations")
    print(f"Cell rates range from {mapped['rate_pct'].min():.2f}% to {mapped['rate_pct'].max():.2f}%")
    print(f"\nWrote four tables to {args.output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
