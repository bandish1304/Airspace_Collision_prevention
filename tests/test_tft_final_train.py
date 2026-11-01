import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.features.tft_final_train import (
    TFTArrayCollator,
    TFTArrayDataset,
    build_template_from_checkpoint,
    checkpoint_completed_epochs,
)
from src.features.tft_model import build_tft_model
from src.features.tft_timeseries import build_timeseries_dataset


class TFTFinalTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = [
            "lateral_distance_nm",
            "vertical_separation_m",
            "closing_speed_mps",
            "bearing_difference_deg",
            "log_time_to_cpa",
            "has_cpa",
        ]
        rng = np.random.default_rng(42)
        self.windows = rng.normal(size=(4, 30, len(self.features))).astype(np.float32)
        self.labels = np.array([0, 1, 0, 1], dtype=np.float32)
        self.training_dataset = build_timeseries_dataset(
            self.windows, self.labels, self.features
        )

    def test_collated_batches_match_forecasting_dataset_schema(self) -> None:
        reference_x, reference_y = next(
            iter(self.training_dataset.to_dataloader(train=False, batch_size=2))
        )
        samples = [
            (torch.from_numpy(self.windows[index]), torch.tensor(self.labels[index]))
            for index in range(2)
        ]
        actual_x, actual_y = TFTArrayCollator(self.training_dataset)(samples)

        self.assertEqual(set(actual_x), set(reference_x))
        for key in reference_x:
            self.assertEqual(actual_x[key].shape, reference_x[key].shape, key)
            torch.testing.assert_close(actual_x[key], reference_x[key])
        torch.testing.assert_close(actual_y[0], reference_y[0])
        self.assertIsNone(actual_y[1])

    def test_array_batch_runs_through_binary_tft(self) -> None:
        dataset = TFTArrayDataset(
            torch.from_numpy(self.windows), torch.from_numpy(self.labels)
        )
        batch_x, batch_y = TFTArrayCollator(self.training_dataset)(
            [dataset[0], dataset[1]]
        )
        model = build_tft_model(
            self.training_dataset,
            hidden_size=8,
            attention_head_size=2,
            dropout=0.1,
            hidden_continuous_size=4,
        )
        prediction = model(batch_x)["prediction"]
        loss = model.loss.loss(prediction, batch_y[0]).mean()

        self.assertEqual(tuple(prediction.shape), (2, 1, 1))
        self.assertTrue(torch.isfinite(loss))

    def test_checkpoint_template_reuses_training_transforms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "resume.ckpt"
            torch.save(
                {
                    "epoch": 4,
                    "hyper_parameters": {
                        "dataset_parameters": self.training_dataset.get_parameters()
                    },
                },
                checkpoint_path,
            )
            restored_dataset = build_template_from_checkpoint(
                checkpoint_path,
                torch.from_numpy(self.windows[:2]),
                torch.from_numpy(self.labels[:2]),
                self.features,
            )
            samples = [
                (torch.from_numpy(self.windows[index]), torch.tensor(self.labels[index]))
                for index in range(2)
            ]
            expected_x, _ = TFTArrayCollator(self.training_dataset)(samples)
            actual_x, _ = TFTArrayCollator(restored_dataset)(samples)
            for key in expected_x:
                torch.testing.assert_close(actual_x[key], expected_x[key])
            self.assertEqual(checkpoint_completed_epochs(checkpoint_path), 5)


if __name__ == "__main__":
    unittest.main()
