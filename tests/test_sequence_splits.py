import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.model_inputs import MODEL_FEATURES
from src.features.sequence_splits import build_sequence_splits


def make_pairs(times, risks):
    rows = []
    for index, (time, risk) in enumerate(zip(times, risks)):
        row = {name: float(index) for name in MODEL_FEATURES}
        row.update(
            snapshot_time=time,
            icao24_a="abc",
            icao24_b="def",
            collision_risk=risk,
        )
        rows.append(row)
    return pd.DataFrame(rows)


class SequenceSplitTests(unittest.TestCase):
    def test_builds_datasets_from_temporal_assignment_without_crossing_splits(self):
        times = list(range(0, 300, 10))
        assignment = pd.DataFrame(
            {
                "snapshot_time": times,
                "part": [
                    "train" if time < 100 else
                    "validation" if time < 200 else "test"
                    for time in times
                ],
            }
        )
        pairs = make_pairs(times, [0, 0, 0, 0, 1] * 6)

        with tempfile.TemporaryDirectory() as temp_dir:
            assignment_path = Path(temp_dir) / "assignment.csv"
            assignment.to_csv(assignment_path, index=False)
            datasets, summary = build_sequence_splits(
                {"sample.csv": pairs},
                assignment_path=assignment_path,
                window_frames=3,
                horizon_frames=2,
            )

        self.assertEqual(set(datasets), {"train", "validation", "test"})
        self.assertEqual(
            [len(datasets[split]) for split in ("train", "validation", "test")],
            [6, 6, 6],
        )
        self.assertEqual(summary["split"].tolist(), ["train", "validation", "test"])
        self.assertEqual(summary["windows"].tolist(), [6, 6, 6])
        self.assertTrue(np.all(summary["positive_windows"].to_numpy() >= 0))
        self.assertLess(
            summary.loc[0, "last_snapshot_time"],
            summary.loc[1, "first_snapshot_time"],
        )
        self.assertLess(
            summary.loc[1, "last_snapshot_time"],
            summary.loc[2, "first_snapshot_time"],
        )

    def test_fails_when_a_split_has_no_windows(self):
        times = list(range(0, 120, 10))
        assignment = pd.DataFrame(
            {
                "snapshot_time": times,
                "part": ["train"] * 4 + ["validation"] * 4 + ["test"] * 4,
            }
        )
        pairs = make_pairs(times, [0] * len(times))

        with tempfile.TemporaryDirectory() as temp_dir:
            assignment_path = Path(temp_dir) / "assignment.csv"
            assignment.to_csv(assignment_path, index=False)
            with self.assertRaisesRegex(ValueError, "test"):
                build_sequence_splits(
                    {"sample.csv": pairs},
                    assignment_path=assignment_path,
                    window_frames=3,
                    horizon_frames=2,
                )


if __name__ == "__main__":
    unittest.main()
