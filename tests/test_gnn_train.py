import unittest

import numpy as np
import torch

from src.gnn.train import (
    combine_smote_losses,
    fit_threshold,
    generate_smote_positives,
)


class TestThresholdSelection(unittest.TestCase):
    def test_chooses_validation_threshold_with_minimum_weighted_cost(self) -> None:
        labels = np.array([1, 1, 0, 0])
        probabilities = np.array([0.9, 0.4, 0.35, 0.1])

        threshold = fit_threshold(labels, probabilities)

        self.assertEqual(threshold, 0.4)

    def test_requires_both_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "both classes"):
            fit_threshold(np.array([1, 1]), np.array([0.9, 0.8]))


class TestSmoteRepresentations(unittest.TestCase):
    def test_generates_only_requested_synthetic_positive_rows(self) -> None:
        positives = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        negatives = np.arange(60, dtype=np.float32).reshape(30, 2)

        synthetic = generate_smote_positives(
            positives,
            negatives,
            target_ratio=0.5,
            random_state=7,
        )

        self.assertEqual(synthetic.shape, (12, 2))
        self.assertTrue(np.isfinite(synthetic).all())

    def test_rejects_unusable_class_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two positive"):
            generate_smote_positives(
                np.array([[1.0, 1.0]]),
                np.array([[0.0, 0.0], [2.0, 2.0]]),
            )


class TestSmoteLossWeighting(unittest.TestCase):
    def test_combines_real_and_synthetic_losses_by_example_count(self) -> None:
        result = combine_smote_losses(
            original_loss=torch.tensor(2.0),
            original_count=900,
            synthetic_loss=torch.tensor(4.0),
            synthetic_count=100,
        )

        self.assertAlmostEqual(result.item(), 2.2)

    def test_rejects_empty_loss_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be non-empty"):
            combine_smote_losses(
                torch.tensor(2.0),
                0,
                torch.tensor(4.0),
                10,
            )


if __name__ == "__main__":
    unittest.main()
