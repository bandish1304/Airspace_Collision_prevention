import unittest

import numpy as np
import pandas as pd
from src.data.clean_states import clean_state_observations
from src.graph.sparse_graph import (
    EDGE_FEATURE_COLUMNS,
    NODE_COLUMNS,
    build_edge_features,
    build_candidate_edges,
    build_nodes,
    snapshot_graph_counts,
)


def make_snapshot(**overrides) -> pd.DataFrame:
    row = {
        "icao24": "abc123",
        "snapshot_time": 100,
        "time_position": 100,
        "latitude": 33.0,
        "longitude": -118.0,
        "velocity": 70.0,
        "true_track": 90.0,
        "on_ground": False,
        "baro_altitude": None,
        "geo_altitude": 1200.0,
    }
    other = {**row, "icao24": "def456", "baro_altitude": 1000.0, "true_track": 0.0}
    return pd.DataFrame([{**row, **overrides}, other])


class BuildNodesTests(unittest.TestCase):
    def test_one_node_per_aircraft_with_prediction_time_features(self):
        nodes = build_nodes(make_snapshot())

        self.assertEqual(list(nodes.columns), NODE_COLUMNS)
        self.assertEqual(nodes["icao24"].tolist(), ["abc123", "def456"])
        self.assertEqual(nodes["altitude_m"].tolist(), [1200.0, 1000.0])
        self.assertAlmostEqual(nodes.loc[0, "track_sin"], 1.0)
        self.assertAlmostEqual(nodes.loc[1, "track_cos"], 1.0)

    def test_grounded_aircraft_are_not_nodes(self):
        nodes = build_nodes(make_snapshot(on_ground=True))

        self.assertEqual(nodes["icao24"].tolist(), ["def456"])

    def test_rejects_multiple_snapshots(self):
        snapshot = make_snapshot()
        snapshot.loc[1, "snapshot_time"] = 110

        with self.assertRaises(ValueError):
            build_nodes(snapshot)


class BuildEdgeFeaturesTests(unittest.TestCase):
    def test_edge_features_match_pairwise_pipeline(self):
        from src.features.build_pipeline import _pairs_for_snapshot
        from src.features.model_inputs import prepare_model_inputs

        snapshot = make_snapshot(on_ground=False)
        snapshot["latitude"] = [33.0, 33.05]
        snapshot["longitude"] = [-118.0, -118.02]
        nodes = build_nodes(snapshot)
        edges = build_edge_features(nodes, [0], [1])

        pairs = _pairs_for_snapshot(
            clean_state_observations(snapshot), 50 * 1852.0
        )
        expected = prepare_model_inputs(pairs)[EDGE_FEATURE_COLUMNS].reset_index(drop=True)

        self.assertEqual(list(edges.columns), EDGE_FEATURE_COLUMNS)
        pd.testing.assert_frame_equal(edges, expected, check_dtype=False)


class CandidateEdgeTests(unittest.TestCase):
    def test_matches_brute_force_and_is_undirected(self):
        from src.features.kinematics import haversine_distance

        rng = np.random.default_rng(0)
        nodes = pd.DataFrame(
            {
                "latitude": rng.uniform(32.0, 35.0, 300),
                "longitude": rng.uniform(-120.0, -116.0, 300),
            }
        )
        index_a, index_b = build_candidate_edges(nodes)

        all_a, all_b = np.triu_indices(len(nodes), k=1)
        distance_nm = haversine_distance(
            nodes["latitude"].to_numpy()[all_a],
            nodes["longitude"].to_numpy()[all_a],
            nodes["latitude"].to_numpy()[all_b],
            nodes["longitude"].to_numpy()[all_b],
        ) / 1852.0
        expected = set(zip(all_a[distance_nm <= 15.0], all_b[distance_nm <= 15.0]))

        self.assertGreater(len(expected), 0)
        self.assertTrue((index_a < index_b).all())
        self.assertEqual(set(zip(index_a, index_b)), expected)

    def test_fewer_than_two_nodes_gives_no_edges(self):
        nodes = pd.DataFrame({"latitude": [33.0], "longitude": [-118.0]})
        self.assertEqual(len(build_candidate_edges(nodes)[0]), 0)


class GraphCountsTests(unittest.TestCase):
    def test_counts_per_snapshot(self):
        near = make_snapshot()
        near["latitude"] = [33.0, 33.05]
        far = make_snapshot(snapshot_time=200, time_position=200)
        far["snapshot_time"] = 200
        far["time_position"] = 200
        far["latitude"] = [33.0, 36.0]
        counts = snapshot_graph_counts(pd.concat([near, far], ignore_index=True))

        self.assertEqual(counts["aircraft"].tolist(), [2, 2])
        self.assertEqual(counts["candidate_edges"].tolist(), [1, 0])
        self.assertEqual(counts["average_degree"].tolist(), [1.0, 0.0])


class PredictionTimeOnlyTests(unittest.TestCase):
    def test_graph_ignores_label_columns_and_rejects_other_times(self):
        snapshot = make_snapshot()
        with_labels = snapshot.assign(future_conflict=1, future_conflict_status="positive")

        pd.testing.assert_frame_equal(build_nodes(snapshot), build_nodes(with_labels))
        forbidden = {"future_conflict", "future_conflict_status"}
        self.assertFalse(forbidden & set(build_nodes(with_labels).columns))
        self.assertFalse(forbidden & set(EDGE_FEATURE_COLUMNS))

        later = make_snapshot(snapshot_time=130, time_position=130)
        with self.assertRaises(ValueError):
            build_nodes(pd.concat([snapshot, later], ignore_index=True))


if __name__ == "__main__":
    unittest.main()
