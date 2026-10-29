import unittest

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet

from src.features.tft_model import BinaryCrossEntropyWithLogits, build_tft_model


class TFTModelTests(unittest.TestCase):
    @staticmethod
    def make_training_dataset():
        rows = []
        for series_id in range(4):
            for time_idx in range(12):
                rows.append(
                    {
                        "series_id": f"pair-{series_id}",
                        "time_idx": time_idx,
                        "feature_a": float(time_idx),
                        "feature_b": float(series_id),
                        "target": float((series_id + time_idx) % 2),
                    }
                )
        frame = pd.DataFrame(rows)
        return TimeSeriesDataSet(
            frame,
            time_idx="time_idx",
            target="target",
            group_ids=["series_id"],
            min_encoder_length=6,
            max_encoder_length=6,
            min_prediction_length=1,
            max_prediction_length=1,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["feature_a", "feature_b", "target"],
            target_normalizer=None,
            add_relative_time_idx=True,
        )

    def test_builds_binary_tft_and_runs_forward_with_probabilities(self):
        dataset = self.make_training_dataset()
        model = build_tft_model(
            dataset,
            hidden_size=8,
            attention_head_size=2,
            hidden_continuous_size=4,
        )
        batch = next(iter(dataset.to_dataloader(train=True, batch_size=4)))
        output = model(batch[0])

        self.assertIsInstance(model, torch.nn.Module)
        self.assertIsInstance(model.loss, BinaryCrossEntropyWithLogits)
        self.assertEqual(model.hparams.output_size, 1)
        self.assertEqual(tuple(output["prediction"].shape), (4, 1, 1))
        self.assertTrue(torch.isfinite(output["prediction"]).all())
        probabilities = model.loss.to_prediction(output["prediction"])
        self.assertTrue(torch.all((probabilities >= 0) & (probabilities <= 1)))

        loss = model.loss.loss(output["prediction"], batch[1][0]).mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_rejects_invalid_hyperparameters(self):
        dataset = self.make_training_dataset()
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            build_tft_model(dataset, hidden_size=0)
        with self.assertRaisesRegex(ValueError, "dropout"):
            build_tft_model(dataset, dropout=1.0)


if __name__ == "__main__":
    unittest.main()
