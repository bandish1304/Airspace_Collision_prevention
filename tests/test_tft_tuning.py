import unittest

import numpy as np
import optuna

from src.features.tft_tuning import SEARCH_SPACE, suggest_hyperparameters


class TFTTuningTests(unittest.TestCase):
    def test_search_space_covers_required_hyperparameters(self) -> None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
        for _ in range(30):
            trial = study.ask()
            params = suggest_hyperparameters(trial)
            self.assertEqual(
                set(params),
                {"attention_head_size", "hidden_size", "dropout", "learning_rate"},
            )
            self.assertIn(params["attention_head_size"], SEARCH_SPACE["attention_head_size"])
            self.assertIn(params["hidden_size"], SEARCH_SPACE["hidden_size"])
            self.assertTrue(0.05 <= params["dropout"] <= 0.4)
            self.assertTrue(1e-4 <= params["learning_rate"] <= 1e-2)
            self.assertEqual(params["hidden_size"] % params["attention_head_size"], 0)
            study.tell(trial, float(np.random.default_rng(0).random()))


if __name__ == "__main__":
    unittest.main()
