import unittest

import numpy as np
from pytorch_forecasting import TimeSeriesDataSet

from src.features.tft_timeseries import (
    SHORT_ENCODER_LENGTH,
    build_timeseries_dataset,
    shorten_encoder_windows,
    validation_dataset,
)


class TFTTimeSeriesTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.features = ["distance", "closing_speed"]
        self.windows = rng.normal(size=(8, 6, 2)).astype(np.float32)
        self.labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)

    def test_builds_one_step_binary_forecasting_dataset(self):
        dataset = build_timeseries_dataset(
            self.windows, self.labels, self.features
        )

        self.assertIsInstance(dataset, TimeSeriesDataSet)
        self.assertEqual(len(dataset), len(self.windows))
        x, y = next(iter(dataset.to_dataloader(train=False, batch_size=4)))
        self.assertEqual(tuple(x["encoder_cont"].shape), (4, 6, 4))
        self.assertEqual(tuple(y[0].shape), (4, 1))
        self.assertEqual(sorted(y[0].squeeze(-1).tolist()), [0.0, 0.0, 1.0, 1.0])

    def test_validation_reuses_training_encoders(self):
        training = build_timeseries_dataset(
            self.windows, self.labels, self.features
        )
        validation = validation_dataset(
            training, self.windows[:4], self.labels[:4], self.features
        )

        self.assertIsInstance(validation, TimeSeriesDataSet)
        self.assertEqual(len(validation), 4)
        x, y = next(iter(validation.to_dataloader(train=False, batch_size=4)))
        self.assertEqual(tuple(x["encoder_cont"].shape), (4, 6, 4))
        self.assertEqual(sorted(y[0].squeeze(-1).tolist()), [0.0, 0.0, 1.0, 1.0])

    def test_rejects_mismatched_labels(self):
        with self.assertRaisesRegex(ValueError, "one scalar target"):
            build_timeseries_dataset(
                self.windows, self.labels[:2], self.features
            )

    def test_shortens_windows_to_the_most_recent_encoder_frames(self):
        shortened = shorten_encoder_windows(self.windows, 3)

        self.assertEqual(shortened.shape, (8, 3, 2))
        np.testing.assert_array_equal(shortened, self.windows[:, -3:, :])

    def test_week11_short_encoder_is_half_the_baseline_length(self):
        windows = np.arange(2 * 30 * 2, dtype=np.float32).reshape(2, 30, 2)

        self.assertEqual(SHORT_ENCODER_LENGTH, 15)
        shortened = shorten_encoder_windows(windows, SHORT_ENCODER_LENGTH)
        self.assertEqual(shortened.shape, (2, 15, 2))
        np.testing.assert_array_equal(shortened, windows[:, -15:, :])

    def test_rejects_invalid_shortened_encoder_length(self):
        with self.assertRaisesRegex(ValueError, "at least 1"):
            shorten_encoder_windows(self.windows, 0)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            shorten_encoder_windows(self.windows, 7)


if __name__ == "__main__":
    unittest.main()
