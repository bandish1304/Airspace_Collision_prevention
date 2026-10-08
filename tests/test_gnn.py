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
)
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
