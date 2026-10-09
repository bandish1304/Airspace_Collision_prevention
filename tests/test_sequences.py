import unittest

import numpy as np
import pandas as pd

from src.features.model_inputs import MODEL_FEATURES
from src.features.sequences import build_windows


def make_pairs(times, risks, a="aaa", b="bbb"):
    n = len(times)
    data = {name: np.arange(n, dtype=float) for name in MODEL_FEATURES}
    data.update(
        snapshot_time=list(times),
        icao24_a=[a] * n,
        icao24_b=[b] * n,
        collision_risk=list(risks),
    )
    return pd.DataFrame(data)


class BuildWindowsTests(unittest.TestCase):
    def test_shapes_and_targets(self):
        pairs = make_pairs(range(0, 100, 10), [0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        w = build_windows(pairs, window_frames=3, horizon_frames=2)
        # ends at positions 2..7 (need 2 frames ahead)
        self.assertEqual(w.ends.tolist(), [2, 3, 4, 5, 6, 7])
        self.assertEqual(w.y_future.tolist(), [0, 0, 1, 1, 0, 0])
        self.assertEqual(w.y_now.tolist(), [0, 0, 0, 0, 1, 0])
        self.assertEqual(w.gather().shape, (6, 3, len(MODEL_FEATURES)))

    def test_window_content_is_contiguous_and_ordered(self):
        pairs = make_pairs(range(0, 60, 10), [0] * 6)
        w = build_windows(pairs, window_frames=3, horizon_frames=1)
        first = w.gather([0])[0, :, 0]
        self.assertEqual(first.tolist(), [0.0, 1.0, 2.0])

    def test_gap_breaks_windows(self):
        pairs = make_pairs([0, 10, 20, 200, 210, 220], [0] * 6)
        w = build_windows(pairs, window_frames=2, horizon_frames=1, max_gap_seconds=30)
        # windows may not cross the 20 -> 200 gap
        self.assertEqual(w.ends.tolist(), [1, 4])

    def test_pairs_do_not_mix_and_order_is_invariant(self):
        one = make_pairs(range(0, 50, 10), [0] * 5, "aaa", "bbb")
        two = make_pairs(range(0, 50, 10), [0] * 5, "ccc", "ddd")
        w = build_windows(pd.concat([one, two]), window_frames=2, horizon_frames=1)
        swapped = pd.concat([one, two]).rename(
            columns={"icao24_a": "icao24_b", "icao24_b": "icao24_a"}
        )
        w2 = build_windows(swapped, window_frames=2, horizon_frames=1)
        self.assertEqual(len(w), len(w2))
        self.assertEqual(len(np.unique(w.pair_id)), 2)

    def test_onset_only_drops_already_risky(self):
        pairs = make_pairs(range(0, 80, 10), [0, 0, 1, 1, 1, 0, 0, 0])
        w = build_windows(pairs, window_frames=2, horizon_frames=1, onset_only=True)
        self.assertTrue((w.y_now == 0).all())

    def test_drop_label_features(self):
        pairs = make_pairs(range(0, 50, 10), [0] * 5)
        w = build_windows(pairs, window_frames=2, horizon_frames=1,
                          include_label_features=False)
        self.assertNotIn("lateral_distance_nm", w.feature_names)
        self.assertEqual(w.gather().shape[2], len(MODEL_FEATURES) - 2)

    def test_invalid_and_missing(self):
        pairs = make_pairs(range(0, 50, 10), [0] * 5)
        with self.assertRaises(ValueError):
            build_windows(pairs, window_frames=0)
        with self.assertRaises(ValueError):
            build_windows(pairs.drop(columns=["closing_speed_mps"]))

    def test_short_history_gives_no_windows(self):
        pairs = make_pairs(range(0, 30, 10), [0] * 3)
        self.assertEqual(len(build_windows(pairs, window_frames=5, horizon_frames=1)), 0)


if __name__ == "__main__":
    unittest.main()
