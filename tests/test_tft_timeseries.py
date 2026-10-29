import unittest

import numpy as np
from pytorch_forecasting import TimeSeriesDataSet

from src.features.tft_timeseries import (
    build_timeseries_dataset,
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


if __name__ == "__main__":
    unittest.main()
