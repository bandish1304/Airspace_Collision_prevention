import unittest

import pandas as pd

from src.analysis.window_feasibility import count_windows


def _pairs(times, risks, a="A", b="B"):
    return pd.DataFrame(
        {
            "snapshot_time": times,
            "icao24_a": a,
            "icao24_b": b,
            "collision_risk": risks,
        }
    )


class CountWindowsTests(unittest.TestCase):
    def test_windows_counted_within_one_run(self):
        pairs = _pairs([0, 10, 20, 30, 40], [0, 0, 0, 0, 1])
        row = count_windows(pairs, "s", window_lengths=(3,), horizon_frames=1).iloc[0]
        self.assertEqual(row["windows"], 3)
        self.assertEqual(row["windows_ending_risky"], 1)
        self.assertEqual(row["windows_with_future_data"], 2)
        self.assertEqual(row["windows_risky_within_horizon"], 1)

    def test_gap_splits_runs_so_short_runs_give_no_windows(self):
        pairs = _pairs([0, 10, 20, 100, 110, 120], [0] * 6)
        row = count_windows(
            pairs, "s", window_lengths=(4,), max_gap_seconds=30
        ).iloc[0]
        self.assertEqual(row["windows"], 0)

    def test_pair_order_does_not_split_a_pair(self):
        first = _pairs([0, 10], [0, 0], a="A", b="B")
        second = _pairs([20, 30], [0, 0], a="B", b="A")
        row = count_windows(
            pd.concat([first, second]), "s", window_lengths=(4,), max_gap_seconds=30
        ).iloc[0]
        self.assertEqual(row["windows"], 1)
        self.assertEqual(row["pairs_with_windows"], 1)

    def test_invalid_arguments_raise(self):
        pairs = _pairs([0, 10], [0, 0])
        with self.assertRaises(ValueError):
            count_windows(pairs, "s", horizon_frames=0)
        with self.assertRaises(ValueError):
            count_windows(pairs, "s", window_lengths=(0,))


if __name__ == "__main__":
    unittest.main()
