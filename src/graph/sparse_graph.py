"""Sparse aircraft-interaction graph for one prediction-time snapshot."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.neighbors import BallTree

from src.data.clean_states import clean_state_observations
from src.features.build_pipeline import METRES_PER_NAUTICAL_MILE
from src.features.kinematics import (
    EARTH_RADIUS_METRES,
    bearing_difference,
    closing_speed,
    haversine_distance,
    time_to_cpa,
    vertical_separation,
)
from src.features.model_inputs import TIME_TO_CPA_CAP_SECONDS

# Candidate-edge rule: connect aircraft within this lateral distance. Chosen from
# the labeled sessions, where 99.91% of future conflicts start within 15 NM.
CANDIDATE_RADIUS_NM = 15.0

# Edge convention: each candidate pair is one undirected edge, stored once with
# the lower node index first. Message passing reads it in both directions.

NODE_COLUMNS = [
    "icao24",
    "latitude",
    "longitude",
    "altitude_m",
    "velocity_mps",
    "track_sin",
    "track_cos",
]


def build_nodes(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Return one node per usable aircraft in a single snapshot.

    Node features use only that snapshot's reported state. Row position in the
    returned table is the node index. Altitude is NaN when neither barometric
    nor geometric altitude is reported.
    """
    if snapshot["snapshot_time"].nunique() > 1:
        raise ValueError("snapshot must contain exactly one snapshot_time.")

    aircraft = clean_state_observations(snapshot)
    altitude = pd.to_numeric(aircraft.get("baro_altitude"), errors="coerce")
    altitude = altitude.fillna(pd.to_numeric(aircraft.get("geo_altitude"), errors="coerce"))
    track = np.radians(aircraft["true_track"].to_numpy(dtype=float))
    return pd.DataFrame(
        {
            "icao24": aircraft["icao24"].astype(str),
            "latitude": aircraft["latitude"],
            "longitude": aircraft["longitude"],
            "altitude_m": altitude,
            "velocity_mps": aircraft["velocity"],
            "track_sin": np.sin(track),
            "track_cos": np.cos(track),
        }
    )[NODE_COLUMNS].reset_index(drop=True)


def build_candidate_edges(
    nodes: pd.DataFrame, radius_nm: float = CANDIDATE_RADIUS_NM
) -> tuple[np.ndarray, np.ndarray]:
    """Return undirected candidate edges (index_a < index_b) within radius_nm.

    A BallTree on great-circle distance finds each aircraft's neighbours directly,
    so the all-pairs distance matrix is never built.
    """
    coordinates = np.radians(nodes[["latitude", "longitude"]].to_numpy(dtype=float))
    if len(coordinates) < 2:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    radius = radius_nm * METRES_PER_NAUTICAL_MILE / EARTH_RADIUS_METRES
    neighbours = BallTree(coordinates, metric="haversine").query_radius(coordinates, r=radius)
    index_a = np.repeat(np.arange(len(neighbours)), [len(n) for n in neighbours])
    index_b = np.concatenate(neighbours)
    keep = index_a < index_b
    return index_a[keep], index_b[keep]


def build_graph(snapshot: pd.DataFrame, radius_nm: float = CANDIDATE_RADIUS_NM) -> dict:
    """Build the sparse graph for one snapshot and record its size.

    Returns nodes, undirected edges (index_a, index_b), edge features, and counts:
    aircraft, candidate edges, and average degree (2 * edges / aircraft).
    """
    nodes = build_nodes(snapshot)
    index_a, index_b = build_candidate_edges(nodes, radius_nm)
    aircraft, edges = len(nodes), len(index_a)
    return {
        "nodes": nodes,
        "index_a": index_a,
        "index_b": index_b,
        "edge_features": build_edge_features(nodes, index_a, index_b),
        "aircraft": aircraft,
        "candidate_edges": edges,
        "average_degree": 2 * edges / aircraft if aircraft else 0.0,
    }


def snapshot_graph_counts(states: pd.DataFrame) -> pd.DataFrame:
    """Return aircraft, candidate-edge and average-degree counts per snapshot."""
    rows = []
    for snapshot_time, snapshot in states.groupby("snapshot_time"):
        graph = build_graph(snapshot)
        rows.append(
            {
                "snapshot_time": snapshot_time,
                "aircraft": graph["aircraft"],
                "all_possible_pairs": graph["aircraft"] * (graph["aircraft"] - 1) // 2,
                "candidate_edges": graph["candidate_edges"],
                "average_degree": graph["average_degree"],
            }
        )
    return pd.DataFrame(rows)


EDGE_FEATURE_COLUMNS = [
    "lateral_distance_nm",
    "vertical_separation_m",
    "closing_speed_mps",
    "bearing_difference_deg",
    "log_time_to_cpa",
    "has_cpa",
]


def build_edge_features(
    nodes: pd.DataFrame, index_a: np.ndarray, index_b: np.ndarray
) -> pd.DataFrame:
    """Return one feature row per candidate edge (node index_a[i], index_b[i]).

    Features are computed only from the two nodes' current state, using the same
    formulas and transforms as the pairwise baseline.
    """
    lat = nodes["latitude"].to_numpy(dtype=float)
    lon = nodes["longitude"].to_numpy(dtype=float)
    speed = nodes["velocity_mps"].to_numpy(dtype=float)
    alt = nodes["altitude_m"].to_numpy(dtype=float)
    track = np.degrees(np.arctan2(nodes["track_sin"], nodes["track_cos"])) % 360.0
    a, b = np.asarray(index_a), np.asarray(index_b)

    cpa = time_to_cpa(lat[a], lon[a], speed[a], track[a], lat[b], lon[b], speed[b], track[b])
    capped = np.minimum(np.nan_to_num(cpa, posinf=TIME_TO_CPA_CAP_SECONDS), TIME_TO_CPA_CAP_SECONDS)
    distance_m = haversine_distance(lat[a], lon[a], lat[b], lon[b])
    return pd.DataFrame(
        {
            "lateral_distance_nm": distance_m / METRES_PER_NAUTICAL_MILE,
            "vertical_separation_m": vertical_separation(alt[a], alt[b]),
            "closing_speed_mps": closing_speed(
                lat[a], lon[a], speed[a], track[a], lat[b], lon[b], speed[b], track[b]
            ),
            "bearing_difference_deg": bearing_difference(track[a], track[b]),
            "log_time_to_cpa": np.log1p(capped),
            "has_cpa": np.isfinite(cpa).astype(float),
        }
    )[EDGE_FEATURE_COLUMNS]


def main() -> None:
    from pathlib import Path

    out_dir = Path("data/processed/graph_stats")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path("data/processed/cleaned_states").glob("*.csv")):
        counts = snapshot_graph_counts(pd.read_csv(path, low_memory=False))
        counts.to_csv(out_dir / f"{path.stem}_graph_counts.csv", index=False)
        print(path.name, len(counts), "snapshots")


if __name__ == "__main__":
    main()
