"""Convert a time-window aircraft snapshot into a graph representation."""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from src.features.kinematics import (
    bearing_difference,
    closing_speed,
    haversine_distance,
    time_to_cpa,
    vertical_separation,
)

REQUIRED_EDGE_ATTRIBUTE_KEYS = {
    "lateral_distance_m",
    "lateral_distance_nm",
    "closing_speed_mps",
    "bearing_difference_deg",
    "vertical_separation_m",
    "time_to_cpa_s",
}

METRES_PER_NAUTICAL_MILE = 1_852.0


# Week 4, step 1: turn one snapshot of aircraft into an undirected graph where
# each aircraft is a node and each relevant nearby pair becomes an edge.
def build_snapshot_graph(
    snapshot: pd.DataFrame,
    max_distance_nm: float = 50.0,
) -> dict[str, object]:
    """Return a graph dictionary for a single airspace snapshot.

    The graph is structured to match the later GNN workflow: aircraft are nodes,
    and edges connect aircraft pairs that are close enough to be considered
    interaction candidates. Edge attributes store the engineered kinematic
    features computed during Week 2.
    """
    if max_distance_nm <= 0:
        raise ValueError("max_distance_nm must be greater than zero.")

    required_columns = {
        "icao24",
        "latitude",
        "longitude",
        "on_ground",
        "snapshot_time",
        "time_position",
        "velocity",
        "true_track",
    }
    missing_columns = required_columns.difference(snapshot.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Snapshot is missing required columns: {missing}.")

    usable_snapshot = snapshot.copy()
    for column in ["latitude", "longitude", "velocity", "true_track", "snapshot_time"]:
        usable_snapshot[column] = pd.to_numeric(usable_snapshot[column], errors="coerce")

    usable_snapshot = usable_snapshot.loc[
        (~usable_snapshot["on_ground"]) & usable_snapshot["icao24"].notna()
    ].dropna(subset=["icao24", "latitude", "longitude", "velocity", "true_track"])

    if usable_snapshot.empty:
        return {
            "snapshot_time": None,
            "num_nodes": 0,
            "num_edges": 0,
            "node_index": {},
            "node_features": [],
            "edge_index": [],
            "edges": [],
        }

    unique_icao24 = list(dict.fromkeys(usable_snapshot["icao24"].tolist()))
    node_index = {icao: index for index, icao in enumerate(unique_icao24)}
    node_features: list[dict[str, float | str]] = []

    for icao in unique_icao24:
        aircraft = usable_snapshot.loc[usable_snapshot["icao24"] == icao].iloc[0]
        altitude = _select_altitude(aircraft)
        node_features.append(
            {
                "icao24": str(icao),
                "latitude": float(aircraft["latitude"]),
                "longitude": float(aircraft["longitude"]),
                "altitude_m": float(altitude),
                "velocity_mps": float(aircraft["velocity"]),
                "true_track_deg": float(aircraft["true_track"]),
            }
        )

    edge_index: set[tuple[int, int]] = set()
    edges: list[dict[str, object]] = []
    maximum_distance_metres = max_distance_nm * METRES_PER_NAUTICAL_MILE

    for aircraft_a, aircraft_b in combinations(usable_snapshot.itertuples(index=False), 2):
        if aircraft_a.icao24 == aircraft_b.icao24:
            continue

        lateral_distance_metres = haversine_distance(
            aircraft_a.latitude,
            aircraft_a.longitude,
            aircraft_b.latitude,
            aircraft_b.longitude,
        )
        if lateral_distance_metres > maximum_distance_metres:
            continue

        altitude_a = _select_altitude(pd.Series(aircraft_a._asdict()))
        altitude_b = _select_altitude(pd.Series(aircraft_b._asdict()))
        source_index = node_index[aircraft_a.icao24]
        target_index = node_index[aircraft_b.icao24]
        edge_key = tuple(sorted((source_index, target_index)))
        if edge_key in edge_index:
            continue

        edge_index.add(edge_key)
        edges.append(
            {
                "source": str(aircraft_a.icao24),
                "target": str(aircraft_b.icao24),
                "source_index": source_index,
                "target_index": target_index,
                "attributes": {
                    "lateral_distance_m": float(lateral_distance_metres),
                    "lateral_distance_nm": float(lateral_distance_metres / METRES_PER_NAUTICAL_MILE),
                    "closing_speed_mps": float(
                        closing_speed(
                            aircraft_a.latitude,
                            aircraft_a.longitude,
                            aircraft_a.velocity,
                            aircraft_a.true_track,
                            aircraft_b.latitude,
                            aircraft_b.longitude,
                            aircraft_b.velocity,
                            aircraft_b.true_track,
                        )
                    ),
                    "bearing_difference_deg": float(
                        bearing_difference(aircraft_a.true_track, aircraft_b.true_track)
                    ),
                    "vertical_separation_m": float(
                        vertical_separation(altitude_a, altitude_b)
                    ),
                    "time_to_cpa_s": float(
                        time_to_cpa(
                            aircraft_a.latitude,
                            aircraft_a.longitude,
                            aircraft_a.velocity,
                            aircraft_a.true_track,
                            aircraft_b.latitude,
                            aircraft_b.longitude,
                            aircraft_b.velocity,
                            aircraft_b.true_track,
                        )
                    ),
                },
            }
        )

    return {
        "snapshot_time": int(usable_snapshot["snapshot_time"].iloc[0]),
        "num_nodes": len(unique_icao24),
        "num_edges": len(edges),
        "node_index": node_index,
        "node_features": node_features,
        "edge_index": sorted(edge_index),
        "edges": edges,
    }


# Week 2, steps 4 and 6: prefer barometric altitude and fall back to geo altitude.
def validate_snapshot_graph(
    snapshot: pd.DataFrame,
    max_distance_nm: float = 50.0,
) -> dict[str, float | int | bool | list[dict[str, object]]]:
    """Check that one snapshot produces a valid graph and edge attributes.

    This function is used for Week 4, step 3: validating that the graph
    structure is coherent on a real time window before building the longer
    sequence dataset used by the GNN.
    """
    graph = build_snapshot_graph(snapshot, max_distance_nm=max_distance_nm)

    has_expected_node_count = graph["num_nodes"] == len(graph["node_index"])
    has_expected_edge_count = graph["num_edges"] == len(graph["edges"])
    all_edges_have_attributes = all(
        REQUIRED_EDGE_ATTRIBUTE_KEYS.issubset(set(edge["attributes"].keys()))
        for edge in graph["edges"]
    )

    return {
        "snapshot_time": graph["snapshot_time"],
        "num_nodes": graph["num_nodes"],
        "num_edges": graph["num_edges"],
        "density": 0.0 if graph["num_nodes"] <= 1 else graph["num_edges"] / max(1, graph["num_nodes"] * (graph["num_nodes"] - 1) / 2),
        "is_valid": has_expected_node_count and has_expected_edge_count and all_edges_have_attributes,
        "edge_attributes_present": all_edges_have_attributes,
        "edges": graph["edges"],
    }


# Week 2, steps 4 and 6: prefer barometric altitude and fall back to geo altitude.
def _select_altitude(aircraft: pd.Series) -> float:
    baro_altitude = pd.to_numeric(aircraft.get("baro_altitude"), errors="coerce")
    if pd.notna(baro_altitude):
        return float(baro_altitude)

    geo_altitude = pd.to_numeric(aircraft.get("geo_altitude"), errors="coerce")
    if pd.notna(geo_altitude):
        return float(geo_altitude)

    return float("nan")


if __name__ == "__main__":
    sample_path = "data/raw/sessions/states_socal_backfill_1790375809.csv"
    sample_snapshot = pd.read_csv(sample_path)
    sample_snapshot = sample_snapshot.loc[sample_snapshot["snapshot_time"] == int(sample_snapshot["snapshot_time"].iloc[0])].copy()
    validation = validate_snapshot_graph(sample_snapshot, max_distance_nm=50.0)
    print(f"snapshot_time={validation['snapshot_time']}")
    print(f"num_nodes={validation['num_nodes']}")
    print(f"num_edges={validation['num_edges']}")
    print(f"is_valid={validation['is_valid']}")
    print(f"edge_attribute_keys={sorted(REQUIRED_EDGE_ATTRIBUTE_KEYS)}")
