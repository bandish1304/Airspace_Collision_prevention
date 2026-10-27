import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.future_label_splits import (
    build_future_label_split_manifest,
    fit_future_feature_scaler,
    iter_labeled_split,
)
from src.features.model_inputs import feature_names


class FutureLabelSplitTests(unittest.TestCase):
    def test_manifest_assigns_whole_sessions_and_loader_excludes_unknowns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignment_path = root / "assignment.csv"
            audit_path = root / "audit.csv"
            manifest_path = root / "manifest.csv"
            files = {
                "train": root / "train.csv",
                "validation": root / "validation.csv",
                "test": root / "test.csv",
            }
            rows_by_split = {
                "train": [
                    (10, 1, "positive"),
                    (10, 0, "negative"),
                    (10, None, "unknown_incomplete_followup"),
                ],
                "validation": [(20, 0, "negative")],
                "test": [(30, 1, "positive")],
            }
            assignment_rows = []
            audit_rows = []
            for split, path in files.items():
                rows = rows_by_split[split]
                pd.DataFrame(
                    rows,
                    columns=["snapshot_time", "future_conflict", "future_conflict_status"],
                ).assign(
                    icao24_a="a00001",
                    icao24_b="b00002",
                    lateral_distance_nm={"train": 2.0, "validation": 100.0, "test": 200.0}[split],
                    vertical_separation_m=100.0,
                    closing_speed_mps=10.0,
                    bearing_difference_deg=90.0,
                    time_to_cpa_s=10.0,
                ).to_csv(path, index=False)
                assignment_rows.append(
                    {"snapshot_time": rows[0][0], "part": split}
                )
                status_counts = pd.Series([row[2] for row in rows]).value_counts()
                audit_rows.append(
                    {
                        "output_file": str(path),
                        "rows": len(rows),
                        "positive": int(status_counts.get("positive", 0)),
                        "negative": int(status_counts.get("negative", 0)),
                        "ongoing_excluded": 0,
                        "unknown_current_separation": 0,
                        "unknown_incomplete_followup": int(
                            status_counts.get("unknown_incomplete_followup", 0)
                        ),
                    }
                )
            pd.DataFrame(assignment_rows).to_csv(assignment_path, index=False)
            pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

            manifest = build_future_label_split_manifest(
                label_audit_path=audit_path,
                assignment_path=assignment_path,
                manifest_path=manifest_path,
                chunksize=2,
            )

            self.assertEqual(set(manifest["split"]), {"train", "validation", "test"})
            train_chunks = list(
                iter_labeled_split("train", manifest_path=manifest_path, chunksize=2)
            )
            train_rows = pd.concat(train_chunks, ignore_index=True)
            self.assertEqual(len(train_rows), 2)
            self.assertNotIn("unknown_incomplete_followup", set(train_rows["future_conflict_status"]))
            scaler = fit_future_feature_scaler(manifest_path, chunksize=2)
            distance_index = feature_names().index("lateral_distance_nm")
            self.assertEqual(scaler.mean_[distance_index], 2.0)
            scaled_validation = scaler.transform(
                pd.DataFrame(
                    {
                        feature: [100.0]
                        for feature in feature_names()
                    }
                )
            )
            self.assertGreater(scaled_validation[0, distance_index], 1.0)

    def test_manifest_rejects_session_spanning_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label_path = root / "mixed.csv"
            pd.DataFrame(
                {
                    "snapshot_time": [1, 2],
                    "future_conflict": [0, 0],
                    "future_conflict_status": ["negative", "negative"],
                }
            ).to_csv(label_path, index=False)
            pd.DataFrame(
                {"snapshot_time": [1, 2], "part": ["train", "test"]}
            ).to_csv(root / "assignment.csv", index=False)
            pd.DataFrame(
                {
                    "output_file": [str(label_path)],
                    "rows": [2],
                    "positive": [0],
                    "negative": [2],
                    "ongoing_excluded": [0],
                    "unknown_current_separation": [0],
                    "unknown_incomplete_followup": [0],
                }
            ).to_csv(root / "audit.csv", index=False)

            with self.assertRaisesRegex(ValueError, "spans multiple split parts"):
                build_future_label_split_manifest(
                    label_audit_path=root / "audit.csv",
                    assignment_path=root / "assignment.csv",
                    manifest_path=root / "manifest.csv",
                    chunksize=1,
                )


if __name__ == "__main__":
    unittest.main()
