import unittest

import numpy as np

from src.evaluation.week11_ablation import AblationWindowDataset
from src.features.sequences import WindowSet


class AblationWindowDatasetTests(unittest.TestCase):
    def test_short_encoder_uses_most_recent_frames_and_preserves_target(self):
        windows = WindowSet(
            features=np.arange(6, dtype=np.float32).reshape(6, 1),
            feature_names=["feature"],
            window_frames=3,
            ends=np.array([2]),
            y_future=np.array([1], dtype=np.int8),
            y_now=np.array([0], dtype=np.int8),
            pair_id=np.array([0]),
            end_time=np.array([2]),
            session="test",
        )

        features, target = AblationWindowDataset(windows, encoder_length=2)[0]

        np.testing.assert_array_equal(features.numpy().ravel(), [1.0, 2.0])
        self.assertEqual(float(target), 1.0)

    def test_encoder_length_cannot_exceed_window_length(self):
        windows = WindowSet(
            features=np.arange(3, dtype=np.float32).reshape(3, 1),
            feature_names=["feature"],
            window_frames=3,
            ends=np.array([2]),
            y_future=np.array([0], dtype=np.int8),
            y_now=np.array([0], dtype=np.int8),
            pair_id=np.array([0]),
            end_time=np.array([2]),
            session="test",
        )

        with self.assertRaises(ValueError):
            AblationWindowDataset(windows, encoder_length=4)


if __name__ == "__main__":
    unittest.main()
