import unittest

import numpy as np

from src.evaluation.harness import build_evaluation_harness

# Four positives, six negatives; at a 0.5 threshold this gives tp=2, fn=2, fp=1.
Y_TRUE = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
Y_PROB = np.array([0.9, 0.8, 0.4, 0.3, 0.7, 0.2, 0.1, 0.1, 0.05, 0.05])


class TestCostMetric(unittest.TestCase):
    def test_missed_conflicts_cost_five_times_a_false_alarm(self) -> None:
        metrics = build_evaluation_harness(Y_TRUE, Y_PROB)
        self.assertEqual(metrics["misclassification_cost"], 5 * 2 + 1)
        self.assertAlmostEqual(metrics["normalised_cost"], 11 / 20)

    def test_never_predicting_risk_scores_exactly_one(self) -> None:
        metrics = build_evaluation_harness(Y_TRUE, np.zeros(len(Y_TRUE)))
        self.assertEqual(metrics["normalised_cost"], 1.0)

    def test_cost_ratio_can_be_varied(self) -> None:
        metrics = build_evaluation_harness(Y_TRUE, Y_PROB, false_negative_cost=10)
        self.assertEqual(metrics["misclassification_cost"], 10 * 2 + 1)


class TestConfusionCounts(unittest.TestCase):
    def test_counts_are_plain_numbers_mlflow_can_log(self) -> None:
        metrics = build_evaluation_harness(Y_TRUE, Y_PROB)
        counts = [metrics[key] for key in
                  ("true_positives", "false_negatives", "false_positives", "true_negatives")]
        self.assertEqual(counts, [2, 2, 1, 5])
        self.assertTrue(all(isinstance(count, int) for count in counts))


if __name__ == "__main__":
    unittest.main()
