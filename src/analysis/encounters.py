"""Collapse repeated positive pair-observations into distinct encounters.

A pair that breaches separation does not do so once. It breaches on every frame
for as long as the situation lasts, so a twenty-minute encounter contributes a
hundred rows that are near-duplicates of each other. Counting those rows as
independent examples overstates how much positive data there is and leaks
across any random train/test split.

Two groupings are computed here because they answer different questions:

    pair over time   same two aircraft, contiguous in time. This is what the
                     Week 4 progress feedback asked about.
    components       also joins aircraft that are close at the same instant.
                     Prone to transitive chaining -- A near B, B near C -- so
                     it merges far more aggressively and its largest group is
                     not a physical encounter.

Usage:
    python -m src.analysis.encounters data/processed/<file>_pairs.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "eda"

DEFAULT_LATERAL_SEPARATION_NM = 5.0
DEFAULT_VERTICAL_SEPARATION_FT = 1_000.0
METRES_PER_FOOT = 0.3048

# Separating for longer than this and coming back counts as a second encounter.
# The choice is a judgement call; gap_sensitivity() exists to show how much it
# actually matters, which is not much.
DEFAULT_ENCOUNTER_GAP_SECONDS = 60

# Pair files run to several million rows, so they are read in pieces and
# filtered down to positives before anything is held in memory.
READ_CHUNK_ROWS = 1_000_000

PAIR_COLUMNS = [
    "snapshot_time",
    "icao24_a",
    "icao24_b",
    "latitude_a",
    "longitude_a",
    "latitude_b",
    "longitude_b",
    "lateral_distance_nm",
    "vertical_separation_m",
    "closing_speed_mps",
    "bearing_difference_deg",
    "time_to_cpa_s",
    "altitude_a_m",
    "altitude_b_m",
]


def load_positive_pairs(
    pairs_path: Path,
    lateral_separation_nm: float = DEFAULT_LATERAL_SEPARATION_NM,
    vertical_separation_ft: float = DEFAULT_VERTICAL_SEPARATION_FT,
) -> tuple[pd.DataFrame, int]:
    """Return the breaching pair-observations, and how many were examined."""
    vertical_limit_m = vertical_separation_ft * METRES_PER_FOOT
    kept = []
    examined = 0

    for chunk in pd.read_csv(pairs_path, usecols=PAIR_COLUMNS, chunksize=READ_CHUNK_ROWS):
        examined += len(chunk)
        breach = (chunk["lateral_distance_nm"] <= lateral_separation_nm) & (
            chunk["vertical_separation_m"] <= vertical_limit_m
        )
        if breach.any():
            kept.append(chunk.loc[breach])

    if not kept:
        return pd.DataFrame(columns=PAIR_COLUMNS), examined
    return pd.concat(kept, ignore_index=True), examined


def _unordered_pair(positives: pd.DataFrame) -> pd.DataFrame:
    """Put each pair in a fixed order so A-B and B-A group together."""
    return pd.DataFrame(
        {
            "aircraft_a": np.minimum(positives["icao24_a"], positives["icao24_b"]),
            "aircraft_b": np.maximum(positives["icao24_a"], positives["icao24_b"]),
            "snapshot_time": positives["snapshot_time"],
        }
    )


def group_by_pair_over_time(
    positives: pd.DataFrame,
    gap_seconds: float = DEFAULT_ENCOUNTER_GAP_SECONDS,
) -> pd.Series:
    """Label each observation with an encounter id, aligned to the input index."""
    pairs = _unordered_pair(positives).sort_values(
        ["aircraft_a", "aircraft_b", "snapshot_time"]
    )
    changed_pair = (pairs["aircraft_a"] != pairs["aircraft_a"].shift()) | (
        pairs["aircraft_b"] != pairs["aircraft_b"].shift()
    )
    resumed_after_gap = pairs["snapshot_time"].diff() > gap_seconds
    encounter_id = (changed_pair | resumed_after_gap).cumsum()
    return encounter_id.reindex(positives.index)


def group_by_components(positives: pd.DataFrame) -> pd.Series:
    """Label each observation with a spatio-temporal component id.

    Nodes are (aircraft, frame). An edge joins the two aircraft of a breaching
    pair within a frame, and joins one aircraft to itself across consecutive
    frames. Connected components of that graph are the situations.
    """
    frames = np.sort(positives["snapshot_time"].unique())
    frame_index = {time: index for index, time in enumerate(frames)}
    aircraft = pd.unique(
        pd.concat([positives["icao24_a"], positives["icao24_b"]])
    )
    aircraft_index = {code: index for index, code in enumerate(aircraft)}

    frame_of = positives["snapshot_time"].map(frame_index).to_numpy()
    first_of = positives["icao24_a"].map(aircraft_index).to_numpy()
    second_of = positives["icao24_b"].map(aircraft_index).to_numpy()

    def node(craft: np.ndarray, frame: np.ndarray) -> np.ndarray:
        return craft * len(frames) + frame

    source = [node(first_of, frame_of)]
    target = [node(second_of, frame_of)]

    appearances = pd.DataFrame(
        {
            "craft": np.concatenate([first_of, second_of]),
            "frame": np.concatenate([frame_of, frame_of]),
        }
    ).drop_duplicates().sort_values(["craft", "frame"])
    consecutive = (appearances["craft"].diff() == 0) & (appearances["frame"].diff() == 1)
    source.append(
        node(appearances["craft"][consecutive].to_numpy(),
             appearances["frame"][consecutive].to_numpy() - 1)
    )
    target.append(
        node(appearances["craft"][consecutive].to_numpy(),
             appearances["frame"][consecutive].to_numpy())
    )

    source = np.concatenate(source)
    target = np.concatenate(target)
    node_count = len(aircraft) * len(frames)
    graph = coo_matrix(
        (np.ones(len(source)), (source, target)), shape=(node_count, node_count)
    )
    _, component_of_node = connected_components(graph, directed=False)
    return pd.Series(component_of_node[node(first_of, frame_of)], index=positives.index)


def summarise_encounters(
    positives: pd.DataFrame, encounter_id: pd.Series
) -> pd.DataFrame:
    """One row per encounter, with its aircraft, extent and observation count."""
    pairs = _unordered_pair(positives)
    pairs["encounter_id"] = encounter_id
    summary = pairs.groupby("encounter_id").agg(
        aircraft_a=("aircraft_a", "first"),
        aircraft_b=("aircraft_b", "first"),
        observations=("snapshot_time", "size"),
        start_time=("snapshot_time", "min"),
        end_time=("snapshot_time", "max"),
    )
    summary["duration_s"] = summary["end_time"] - summary["start_time"]
    return summary.reset_index()


def gap_sensitivity(
    positives: pd.DataFrame,
    thresholds: tuple[int, ...] = (10, 20, 30, 60, 90, 120, 300, 600),
) -> pd.DataFrame:
    """Show how the encounter count responds to the gap threshold.

    The headline inflation figure is only meaningful if it survives a different
    choice of threshold, so this sweep belongs with the result rather than in a
    footnote.
    """
    rows = []
    for threshold in thresholds:
        count = group_by_pair_over_time(positives, gap_seconds=threshold).nunique()
        rows.append(
            {
                "gap_seconds": threshold,
                "encounters": count,
                "inflation": round(len(positives) / count, 2),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs_path", type=Path, help="Path to a pair-feature CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--lateral-separation-nm", type=float, default=DEFAULT_LATERAL_SEPARATION_NM
    )
    parser.add_argument(
        "--vertical-separation-ft", type=float, default=DEFAULT_VERTICAL_SEPARATION_FT
    )
    parser.add_argument(
        "--encounter-gap-seconds", type=int, default=DEFAULT_ENCOUNTER_GAP_SECONDS
    )
    args = parser.parse_args()

    positives, examined = load_positive_pairs(
        args.pairs_path,
        lateral_separation_nm=args.lateral_separation_nm,
        vertical_separation_ft=args.vertical_separation_ft,
    )
    if positives.empty:
        print("No positive pair-observations found.")
        return

    encounter_id = group_by_pair_over_time(positives, args.encounter_gap_seconds)
    component_id = group_by_components(positives)
    encounters = summarise_encounters(positives, encounter_id)
    sensitivity = gap_sensitivity(positives)

    component_sizes = component_id.value_counts()
    unordered = _unordered_pair(positives)
    distinct_pairs = unordered.groupby(["aircraft_a", "aircraft_b"]).ngroups

    args.output_dir.mkdir(parents=True, exist_ok=True)
    encounters.to_csv(args.output_dir / "step3_encounters.csv", index=False)
    sensitivity.to_csv(args.output_dir / "step3_gap_sensitivity.csv", index=False)
    assignment = positives[["snapshot_time", "icao24_a", "icao24_b"]].copy()
    assignment["encounter_id"] = encounter_id
    assignment["situation_id"] = component_id
    assignment.to_csv(args.output_dir / "step3_situation_assignment.csv", index=False)

    print(f"Pair-observations examined : {examined:,}")
    print(f"Positive pair-observations : {len(positives):,} "
          f"({100 * len(positives) / examined:.3f}%)")
    print(f"Distinct aircraft pairs    : {distinct_pairs:,}")
    print(f"Encounters (pair + time)   : {len(encounters):,}")
    print(f"Inflation                  : {len(positives) / len(encounters):.1f}x")
    print(f"Components (chained)       : {component_sizes.size:,}")
    print(f"Largest component holds    : {component_sizes.iloc[0]:,} "
          f"({100 * component_sizes.iloc[0] / len(positives):.1f}% of positives)")
    print(f"Converging                 : {100 * (positives['closing_speed_mps'] > 0).mean():.1f}%")
    print(f"Below 10,000 ft            : "
          f"{100 * (positives['altitude_a_m'] / METRES_PER_FOOT < 10_000).mean():.1f}%")
    print(f"Wrote                      : {args.output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
