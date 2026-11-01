import tempfile
import unittest
from pathlib import Path

from src.features.log_tft_mlflow import build_validation_history


class TFTMLflowTests(unittest.TestCase):
    def test_validation_history_combines_resumed_logs_once_per_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_dir = Path(temporary_directory)
            version_zero = log_dir / "version_0"
            version_two = log_dir / "version_2"
            version_zero.mkdir()
            version_two.mkdir()
            (version_zero / "metrics.csv").write_text(
                "epoch,step,val_loss\n0,10,0.5\n1,20,0.4\n2,30,0.3\n",
                encoding="utf-8",
            )
            (version_two / "metrics.csv").write_text(
                "epoch,step,val_loss\n1,40,0.35\n2,50,0.25\n",
                encoding="utf-8",
            )

            history = build_validation_history(log_dir)

            self.assertEqual(history["epoch"].tolist(), [0, 1, 2])
            self.assertEqual(history["val_loss"].tolist(), [0.5, 0.35, 0.25])
            self.assertEqual(
                history["lightning_log_version"].tolist(), [0, 2, 2]
            )


if __name__ == "__main__":
    unittest.main()
