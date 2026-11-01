import unittest

import pandas as pd

from src.evaluation.week10_test_set import align_predictions, normalize_pair_ids


class Week10AlignmentTests(unittest.TestCase):
    def test_pair_order_and_case_do_not_change_the_key(self) -> None:
        frame = pd.DataFrame(
            {"snapshot_time": [1], "icao24_a": ["BBB111"], "icao24_b": ["aaa222"]}
        )

        result = normalize_pair_ids(frame)

        self.assertEqual(result.loc[0, "icao24_a"], "aaa222")
        self.assertEqual(result.loc[0, "icao24_b"], "bbb111")

    def test_alignment_keeps_only_pairs_scored_by_both_models(self) -> None:
        gnn = pd.DataFrame(
            {
                "snapshot_time": [1, 1, 2],
                "icao24_a": ["a", "a", "b"],
                "icao24_b": ["b", "c", "c"],
                "future_conflict": [1, 0, 0],
                "probability": [0.9, 0.1, 0.2],
            }
        )
        tft = pd.DataFrame(
            {
                "snapshot_time": [1, 1, 3],
                "icao24_a": ["b", "d", "a"],
                "icao24_b": ["a", "a", "b"],
                "tft_y_future": [0, 1, 0],
                "tft_y_now": [0, 0, 0],
                "probability": [0.4, 0.5, 0.6],
            }
        )

        aligned, summary = align_predictions(gnn, tft)

        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned.loc[0, "gnn_probability"], 0.9)
        self.assertEqual(aligned.loc[0, "tft_probability"], 0.4)
        self.assertEqual(summary["aligned_positive_pairs"], 1)
        self.assertEqual(summary["gnn_positives_without_tft_window"], 0)
        self.assertEqual(summary["label_agreement"]["future_conflict_only"], 1)

    def test_duplicate_pairs_are_rejected(self) -> None:
        frame = pd.DataFrame(
            {
                "snapshot_time": [1, 1],
                "icao24_a": ["a", "b"],
                "icao24_b": ["b", "a"],
            }
        )

        with self.assertRaises(ValueError):
            normalize_pair_ids(frame)


if __name__ == "__main__":
    unittest.main()
