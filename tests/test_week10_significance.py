import unittest

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.evaluation.week10_significance import (
    bootstrap_p_value,
    holm_adjust,
    mcnemar_exact,
    paired_cluster_bootstrap,
    weighted_metrics,
)


class SignificanceTests(unittest.TestCase):
    def test_unit_weights_match_standard_metrics(self) -> None:
        labels = np.array([1, 0, 1, 0, 0, 1])
        scores = np.array([0.9, 0.2, 0.6, 0.7, 0.1, 0.4])
        predictions = (scores >= 0.5).astype(int)

        result = weighted_metrics(labels, scores, predictions, np.ones(6))

        self.assertAlmostEqual(result["f1"], f1_score(labels, predictions))
        self.assertAlmostEqual(result["auroc"], roc_auc_score(labels, scores))
        self.assertAlmostEqual(
            result["average_precision"], average_precision_score(labels, scores)
        )
        self.assertAlmostEqual(result["false_negative_rate"], 1 / 3)

    def test_bootstrap_detects_a_clearly_better_model(self) -> None:
        rng = np.random.default_rng(0)
        clusters = np.repeat(np.arange(200), 5)
        labels = np.tile([1, 0, 0, 0, 0], 200)
        good = np.where(labels == 1, 0.9, 0.1) + rng.normal(0, 0.02, len(labels))
        poor = rng.uniform(0, 1, len(labels))

        differences = paired_cluster_bootstrap(
            labels, poor, good, 0.5, 0.5, clusters, resamples=200, seed=1
        )

        self.assertGreater(differences["auroc"].min(), 0)
        self.assertLess(bootstrap_p_value(differences["auroc"]), 0.05)

    def test_p_value_is_large_when_differences_straddle_zero(self) -> None:
        self.assertGreater(bootstrap_p_value(np.linspace(-1, 1, 1001)), 0.9)

    def test_holm_adjustment_is_monotone_and_capped(self) -> None:
        adjusted = holm_adjust([0.01, 0.04, 0.03])

        self.assertEqual(adjusted, [0.03, 0.06, 0.06])
        self.assertTrue(all(value <= 1 for value in adjusted))

    def test_mcnemar_counts_only_discordant_pairs(self) -> None:
        first = np.array([True, True, False, False, True])
        second = np.array([True, False, True, True, True])

        result = mcnemar_exact(first, second)

        self.assertEqual(result["gnn_only_correct"], 1)
        self.assertEqual(result["tft_only_correct"], 2)


if __name__ == "__main__":
    unittest.main()
