import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.features.tft_timeseries import SHORT_ENCODER_LENGTH
from src.features.week11_ablation_train import (
    ENCODER_LENGTH,
    EpochProgressCallback,
    VARIANTS,
    load_training_settings,
    prepare_variant_windows,
)


class Week11AblationTrainingTests(unittest.TestCase):
    def test_variant_definitions_share_baseline_length_and_use_15_frame_ablation(self):
        variants = {variant.name: variant for variant in VARIANTS}

        self.assertEqual(variants["full_tft"].encoder_length, ENCODER_LENGTH)
        self.assertEqual(
            variants["no_learned_variable_selection"].encoder_length,
            ENCODER_LENGTH,
        )
        self.assertTrue(
            variants["no_learned_variable_selection"].disable_learned_variable_selection
        )
        self.assertEqual(
            variants["encoder_15"].encoder_length, SHORT_ENCODER_LENGTH
        )

    def test_short_encoder_is_applied_to_train_and_validation(self):
        train = torch.arange(3 * 30 * 2).reshape(3, 30, 2)
        validation = torch.arange(2 * 30 * 2).reshape(2, 30, 2)
        variant = next(item for item in VARIANTS if item.name == "encoder_15")

        train_short, validation_short = prepare_variant_windows(
            variant, train, validation
        )

        self.assertEqual(tuple(train_short.shape), (3, 15, 2))
        self.assertEqual(tuple(validation_short.shape), (2, 15, 2))
        torch.testing.assert_close(train_short, train[:, -15:, :])
        torch.testing.assert_close(validation_short, validation[:, -15:, :])

    def test_training_settings_reuse_week9_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "week9.json"
            path.write_text(
                json.dumps(
                    {
                        "epochs_completed": 18,
                        "batch_size": 128,
                        "validation_batch_size": 512,
                        "early_stopping_patience": 8,
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                load_training_settings(path),
                {
                    "max_epochs": 18,
                    "batch_size": 128,
                    "validation_batch_size": 512,
                    "early_stopping_patience": 8,
                },
            )

    def test_epoch_progress_is_persisted_after_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            callback = EpochProgressCallback(
                path, "baseline", max_epochs=18, pause_after_epoch=True
            )
            trainer = SimpleNamespace(
                sanity_checking=False,
                current_epoch=2,
                callback_metrics={"val_loss": torch.tensor(0.25)},
                should_stop=False,
            )

            callback.on_validation_end(trainer, torch.nn.Linear(1, 1))
            progress = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(progress["completed_epochs"], 3)
            self.assertEqual(progress["max_epochs"], 18)
            self.assertEqual(progress["status"], "training")
            self.assertAlmostEqual(progress["validation_loss"], 0.25)
            self.assertTrue(trainer.should_stop)


if __name__ == "__main__":
    unittest.main()
