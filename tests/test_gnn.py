import math
import unittest

import pandas as pd
import torch
from torch_geometric.data import Batch

from src.gnn import (
    FeatureScaler,
    GraphSAGEEdgeClassifier,
    pair_feature_names,
    pairs_to_graphs,
    sparse_snapshot_to_graph,
)
from src.gnn.future_tuning import verify_candidate_pair_alignment
from src.features.model_inputs import TIME_TO_CPA_CAP_SECONDS
from src.gnn.data import NODE_FEATURES


def _pair(time, a, b, lateral_nm, vertical_m, time_to_cpa, altitude_b=1_000.0):
    return {
        "snapshot_time": time,
        "icao24_a": a,
        "icao24_b": b,
        "latitude_a": 33.8,
        "longitude_a": -118.2,
        "altitude_a_m": 1_000.0,
        "velocity_a_mps": 70.0,
        "true_track_a_deg": 90.0,
        "latitude_b": 33.9,
        "longitude_b": -118.1,
        "altitude_b_m": altitude_b,
        "velocity_b_mps": 75.0,
        "true_track_b_deg": 270.0,
        "lateral_distance_nm": lateral_nm,
        "vertical_separation_m": vertical_m,
        "closing_speed_mps": 40.0,
        "bearing_difference_deg": 180.0,
        "time_to_cpa_s": time_to_cpa,
    }


def _sample_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Frame 100: one breaching pair, one far apart and diverging.
            _pair(100, "a1", "a2", 2.0, 100.0, 120.0),
            _pair(100, "a1", "a3", 30.0, 2_000.0, math.inf),
            # Frame 110: a breaching pair, and one with a missing altitude.
            _pair(110, "a1", "a2", 3.0, 200.0, 90.0),
            _pair(110, "a1", "a4", 4.0, float("nan"), 60.0, altitude_b=float("nan")),
        ]
    )


class TestPairsToGraphs(unittest.TestCase):
    def test_one_labelled_graph_per_snapshot(self) -> None:
        graphs = pairs_to_graphs(_sample_pairs())

        self.assertEqual([graph.snapshot_time for graph in graphs], [100, 110])
        self.assertEqual(graphs[0].num_nodes, 3)
        self.assertEqual(graphs[0].edge_label.tolist(), [1.0, 0.0])
        # Message passing needs both directions of every pair.
        self.assertEqual(graphs[0].edge_index.size(1), 2 * graphs[0].edge_label.numel())

    def test_pair_without_altitude_is_dropped(self) -> None:
        second = pairs_to_graphs(_sample_pairs())[1]

        self.assertEqual(second.edge_label.numel(), 1)
        self.assertEqual(second.num_nodes, 2)
        self.assertTrue(torch.isfinite(second.x).all())

    def test_never_converging_pair_is_kept_capped_and_flagged(self) -> None:
        first = pairs_to_graphs(_sample_pairs())[0]
        names = pair_feature_names()
        diverging = first.pair_attr[1]

        self.assertEqual(diverging[names.index("has_cpa")].item(), 0.0)
        self.assertAlmostEqual(
            diverging[names.index("log_time_to_cpa")].item(),
            math.log1p(TIME_TO_CPA_CAP_SECONDS),
            places=4,
        )

    def test_label_defining_features_can_be_left_out(self) -> None:
        graphs = pairs_to_graphs(_sample_pairs(), include_label_features=False)
        names = pair_feature_names(include_label_features=False)

        self.assertNotIn("lateral_distance_nm", names)
        self.assertNotIn("vertical_separation_m", names)
        self.assertEqual(graphs[0].pair_attr.size(1), len(names))


class TestGraphSAGEEdgeClassifier(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.graphs = pairs_to_graphs(_sample_pairs())
        self.model = GraphSAGEEdgeClassifier(
            node_features=len(NODE_FEATURES),
            pair_features=len(pair_feature_names()),
            hidden_channels=16,
        )

    def test_scores_every_pair_regardless_of_endpoint_order(self) -> None:
        self.model.eval()
        graph = self.graphs[0]
        swapped = graph.clone()
        swapped.edge_label_index = graph.edge_label_index.flip(0)

        with torch.no_grad():
            scores = self.model(graph)
            swapped_scores = self.model(swapped)

        self.assertEqual(scores.shape, (graph.edge_label.numel(),))
        self.assertTrue(torch.allclose(scores, swapped_scores, atol=1e-6))

    def test_batch_scores_match_individual_graphs(self) -> None:
        # Batching offsets node indices; if edge_label_index were not shifted
        # with them, pairs would be scored against the wrong aircraft.
        self.model.eval()
        with torch.no_grad():
            separate = torch.cat([self.model(graph) for graph in self.graphs])
            batched = self.model(Batch.from_data_list(self.graphs))

        self.assertTrue(torch.allclose(separate, batched, atol=1e-6))

    def test_gradients_reach_the_graph_layers(self) -> None:
        graph = self.graphs[0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            self.model(graph), graph.edge_label
        )
        loss.backward()

        self.assertIsNotNone(self.model.convs[0].lin_l.weight.grad)
        self.assertGreater(self.model.convs[0].lin_l.weight.grad.abs().sum().item(), 0)


class TestSparseSnapshotToGraph(unittest.TestCase):
    def test_graph_uses_sparse_candidate_edges_and_prediction_time_features(self):
        snapshot = pd.DataFrame(
            [
                {
                    "icao24": "abc123",
                    "snapshot_time": 100,
                    "time_position": 100,
                    "latitude": 33.0,
                    "longitude": -118.0,
                    "velocity": 70.0,
                    "true_track": 90.0,
                    "on_ground": False,
                    "baro_altitude": 1000.0,
                    "geo_altitude": 1000.0,
                },
                {
                    "icao24": "def456",
                    "snapshot_time": 100,
                    "time_position": 100,
                    "latitude": 33.05,
                    "longitude": -118.02,
                    "velocity": 75.0,
                    "true_track": 270.0,
                    "on_ground": False,
                    "baro_altitude": 1200.0,
                    "geo_altitude": 1200.0,
                },
            ]
        )

        graph = sparse_snapshot_to_graph(snapshot)

        self.assertEqual(graph.snapshot_time, 100)
        self.assertEqual(tuple(graph.x.shape), (2, len(NODE_FEATURES)))
        self.assertEqual(tuple(graph.edge_index.shape), (2, 2))
        self.assertEqual(tuple(graph.edge_label_index.shape), (2, 1))
        self.assertEqual(tuple(graph.pair_attr.shape), (1, len(pair_feature_names())))
        self.assertTrue(torch.isfinite(graph.x).all())
        self.assertTrue(torch.isfinite(graph.pair_attr).all())
        self.assertEqual(graph.edge_label_index.tolist(), [[0], [1]])
        self.assertEqual(
            {tuple(edge) for edge in graph.edge_index.t().tolist()},
            {(0, 1), (1, 0)},
        )
        self.assertFalse(hasattr(graph, "edge_label"))

    def test_future_labels_align_with_classifier_probabilities(self):
        snapshot = pd.DataFrame(
            [
                {
                    "icao24": icao24,
                    "snapshot_time": 100,
                    "time_position": 100,
                    "latitude": latitude,
                    "longitude": -118.0,
                    "velocity": 70.0,
                    "true_track": track,
                    "on_ground": False,
                    "baro_altitude": altitude,
                    "geo_altitude": altitude,
                }
                for icao24, latitude, track, altitude in (
                    ("abc123", 33.0, 90.0, 1000.0),
                    ("def456", 33.05, 270.0, 1200.0),
                    ("fed123", 33.1, 180.0, 1400.0),
                )
            ]
        )
        labels = pd.DataFrame(
            {
                "snapshot_time": [100],
                "icao24_a": ["abc123"],
                "icao24_b": ["def456"],
                "future_conflict": [1],
                "future_conflict_status": ["positive"],
            }
        )
        graph = sparse_snapshot_to_graph(snapshot, labels)
        model = GraphSAGEEdgeClassifier(
            node_features=len(NODE_FEATURES),
            pair_features=len(pair_feature_names()),
            hidden_channels=8,
        )
        model.eval()

        with torch.no_grad():
            probabilities = torch.sigmoid(model(graph))

        self.assertEqual(graph.edge_label.tolist(), [1.0])
        self.assertEqual(graph.edge_label_index.size(1), 1)
        self.assertEqual(graph.edge_index.size(1), 6)
        self.assertEqual(probabilities.shape, graph.edge_label.shape)
        self.assertTrue(torch.all((probabilities >= 0) & (probabilities <= 1)))


class TestCandidatePairAlignment(unittest.TestCase):
    def test_pair_alignment_ignores_row_order_and_endpoint_order(self):
        baseline = pd.DataFrame(
            {
                "snapshot_time": [100, 100],
                "icao24_a": ["abc123", "def456"],
                "icao24_b": ["def456", "fed123"],
                "future_conflict": [1, 0],
                "probability": [0.8, 0.1],
            }
        )
        graphsage = pd.DataFrame(
            {
                "snapshot_time": [100, 100],
                "icao24_a": ["fed123", "def456"],
                "icao24_b": ["def456", "abc123"],
                "future_conflict": [0, 1],
                "probability": [0.2, 0.7],
            }
        )

        verify_candidate_pair_alignment(graphsage, baseline)

    def test_pair_alignment_rejects_missing_candidate(self):
        baseline = pd.DataFrame(
            {
                "snapshot_time": [100],
                "icao24_a": ["abc123"],
                "icao24_b": ["def456"],
                "future_conflict": [1],
            }
        )

        with self.assertRaisesRegex(ValueError, "do not match"):
            verify_candidate_pair_alignment(baseline.iloc[0:0], baseline)


class TestGraphSAGEEmbeddings(unittest.TestCase):
    def test_encode_returns_neighbor_contextualized_embedding_per_aircraft(self):
        torch.manual_seed(0)
        snapshot = pd.DataFrame(
            [
                {
                    "icao24": icao24,
                    "snapshot_time": 100,
                    "time_position": 100,
                    "latitude": latitude,
                    "longitude": -118.0,
                    "velocity": 70.0,
                    "true_track": 90.0,
                    "on_ground": False,
                    "baro_altitude": altitude,
                    "geo_altitude": altitude,
                }
                for icao24, latitude, altitude in (
                    ("abc123", 33.0, 1000.0),
                    ("def456", 33.05, 1200.0),
                    ("fed123", 34.0, 1500.0),
                )
            ]
        )
        graph = sparse_snapshot_to_graph(snapshot)
        model = GraphSAGEEdgeClassifier(
            node_features=len(NODE_FEATURES),
            pair_features=len(pair_feature_names()),
            hidden_channels=8,
        )
        model.eval()

        with torch.no_grad():
            contextual = model.encode(graph.x, graph.edge_index)
            without_edges = model.encode(graph.x, graph.edge_index.new_empty((2, 0)))

        self.assertEqual(tuple(contextual.shape), (3, 8))
        self.assertTrue(torch.isfinite(contextual).all())
        self.assertFalse(torch.allclose(contextual, without_edges))


class TestCandidatePairRepresentations(unittest.TestCase):
    def test_pair_representation_combines_both_embeddings_and_pair_features(self):
        graph = sparse_snapshot_to_graph(
            pd.DataFrame(
                [
                    {
                        "icao24": icao24,
                        "snapshot_time": 100,
                        "time_position": 100,
                        "latitude": latitude,
                        "longitude": -118.0,
                        "velocity": 70.0,
                        "true_track": 90.0,
                        "on_ground": False,
                        "baro_altitude": altitude,
                        "geo_altitude": altitude,
                    }
                    for icao24, latitude, altitude in (
                        ("abc123", 33.0, 1000.0),
                        ("def456", 33.05, 1200.0),
                    )
                ]
            )
        )
        hidden_channels = 8
        model = GraphSAGEEdgeClassifier(
            node_features=len(NODE_FEATURES),
            pair_features=len(pair_feature_names()),
            hidden_channels=hidden_channels,
        )
        model.eval()

        with torch.no_grad():
            representation = model.pair_representations(graph)

        self.assertEqual(
            tuple(representation.shape),
            (graph.edge_label_index.size(1), 3 * hidden_channels + graph.pair_attr.size(1)),
        )
        self.assertTrue(torch.isfinite(representation).all())
        self.assertTrue(torch.equal(representation[:, -graph.pair_attr.size(1):], graph.pair_attr))


class TestFeatureScaler(unittest.TestCase):
    def test_training_features_are_centred(self) -> None:
        graphs = pairs_to_graphs(_sample_pairs())
        scaled = FeatureScaler().fit(graphs).transform(graphs)
        nodes = torch.cat([graph.x for graph in scaled])

        self.assertTrue(torch.allclose(nodes.mean(0), torch.zeros(nodes.size(1)), atol=1e-5))

    def test_fit_accepts_a_one_pass_graph_iterator(self) -> None:
        graphs = pairs_to_graphs(_sample_pairs())
        from_list = FeatureScaler().fit(graphs)
        from_iterator = FeatureScaler().fit(iter(graphs))

        self.assertTrue(torch.allclose(from_list.node_mean, from_iterator.node_mean))
        self.assertTrue(torch.allclose(from_list.node_std, from_iterator.node_std))
        self.assertTrue(torch.allclose(from_list.pair_mean, from_iterator.pair_mean))
        self.assertTrue(torch.allclose(from_list.pair_std, from_iterator.pair_std))

    def test_transform_before_fit_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            FeatureScaler().transform(pairs_to_graphs(_sample_pairs()))


if __name__ == "__main__":
    unittest.main()
