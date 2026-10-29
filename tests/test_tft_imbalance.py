import unittest

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, TensorDataset

from src.features.model_inputs import MODEL_FEATURES
from src.features.sequences import build_windows
from src.features.tft_dataset import TFTWindowDataset
from src.features.tft_imbalance import smote_training_windows


def make_dataset(risks, offset=0):
    rows = []
    for index, risk in enumerate(risks):
        row = {name: float(offset + index) for name in MODEL_FEATURES}
        row.update(
            snapshot_time=index * 10,
            icao24_a="abc",
            icao24_b="def",
            collision_risk=risk,
        )
        rows.append(row)
    windows = build_windows(
        pd.DataFrame(rows),
        window_frames=2,
        horizon_frames=1,
        include_label_features=False,
    )
    return TFTWindowDataset(windows)


class TFTSMOTETests(unittest.TestCase):
    def test_smote_balances_only_supplied_training_windows(self):
        training = ConcatDataset(
            [
                make_dataset([0, 0, 0, 1, 0, 0, 1, 0, 0, 0]),
                make_dataset([0, 0, 1, 0, 0, 0, 0, 1, 0, 0], offset=100),
            ]
        )
        validation = make_dataset([0, 1, 0, 0, 0, 0])
        test = make_dataset([0, 0, 0, 1, 0, 0])
        validation_before = [validation[index] for index in range(len(validation))]
        test_before = [test[index] for index in range(len(test))]
        train_labels_before = np.concatenate(
            [dataset.windows.y_future for dataset in training.datasets]
        )

        resampled, metadata = smote_training_windows(
            training,
            negative_multiplier=3,
            target_ratio=0.5,
            random_state=7,
        )

        features, labels = resampled.tensors
        self.assertIsInstance(resampled, TensorDataset)
        self.assertEqual(tuple(features.shape[1:]), (2, len(MODEL_FEATURES) - 2))
        self.assertEqual(int(labels.sum()), metadata["resampled_positive_windows"])
        self.assertEqual(
            metadata["resampled_positive_windows"],
            int(np.ceil(0.5 * metadata["sampled_negative_windows"])),
        )
        self.assertEqual(
            metadata["original_positive_windows"], int(train_labels_before.sum())
        )
        self.assertLessEqual(
            metadata["sampled_negative_windows"],
            3 * metadata["original_positive_windows"],
        )
        self.assertEqual(len(validation), len(validation_before))
        self.assertEqual(len(test), len(test_before))
        for index, (features_before, label_before) in enumerate(validation_before):
            features_after, label_after = validation[index]
            self.assertTrue(torch.equal(features_before, features_after))
            self.assertTrue(torch.equal(label_before, label_after))
        for index, (features_before, label_before) in enumerate(test_before):
            features_after, label_after = test[index]
            self.assertTrue(torch.equal(features_before, features_after))
            self.assertTrue(torch.equal(label_before, label_after))

    def test_fails_when_training_data_has_too_few_positives(self):
        training = make_dataset([0, 0, 0, 0, 1, 0, 0, 0])

        with self.assertRaisesRegex(ValueError, "at least two positive"):
            smote_training_windows(training)

    def test_rejects_non_tft_datasets(self):
        with self.assertRaisesRegex(TypeError, "TFTWindowDataset"):
            smote_training_windows(TensorDataset(torch.zeros(3, 2), torch.zeros(3)))


if __name__ == "__main__":
    unittest.main()
