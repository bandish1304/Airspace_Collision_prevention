import unittest

import pandas as pd

from src.labeling.future_conflicts import (
    FUTURE_LABEL_COLUMN,
    LABEL_STATUS_COLUMN,
    iter_future_conflict_labels,
    prepare_future_model_inputs,
)


def make_states(
    *,
    conflict_at: int | None = None,
    stop_after: int | None = None,
    initially_conflicting: bool = False,
) -> pd.DataFrame:
    rows = []
    for timestamp in range(0, 81, 10):
        if stop_after is not None and timestamp > stop_after:
            continue
        separation_nm = 4.0 if initially_conflicting else 10.0
        if conflict_at is not None and timestamp >= conflict_at:
            separation_nm = 4.0
        for icao24, latitude in (
            ("a00001", 33.0),
            ("b00002", 33.0 + separation_nm / 60.0),
        ):
            rows.append(
                {
                    "icao24": icao24,
                    "snapshot_time": timestamp,
                    "time_position": timestamp,
                    "latitude": latitude,
                    "longitude": -118.0,
                    "baro_altitude": 10_000.0,
                    "geo_altitude": None,
                    "velocity": 70.0,
                    "true_track": 90.0,
                    "on_ground": False,
                }
            )
    return pd.DataFrame(rows)


def label_at_zero(states: pd.DataFrame) -> pd.Series:
    chunks = list(iter_future_conflict_labels(states))
    return next(chunk.iloc[0] for chunk in chunks if chunk["snapshot_time"].iloc[0] == 0)


class FutureConflictLabelTests(unittest.TestCase):
    def test_future_entry_is_positive_and_features_are_prediction_time_only(self):
        row = label_at_zero(make_states(conflict_at=30))

        self.assertEqual(int(row[FUTURE_LABEL_COLUMN]), 1)
        self.assertEqual(row[LABEL_STATUS_COLUMN], "positive")
        self.assertAlmostEqual(row["lateral_distance_nm"], 10.0, delta=0.1)
        self.assertNotIn("future_lateral_distance_nm", row.index)

    def test_full_followup_without_entry_is_negative(self):
        row = label_at_zero(make_states())

        self.assertEqual(int(row[FUTURE_LABEL_COLUMN]), 0)
        self.assertEqual(row[LABEL_STATUS_COLUMN], "negative")

    def test_incomplete_followup_is_unknown_not_negative(self):
        row = label_at_zero(make_states(stop_after=40))

        self.assertTrue(pd.isna(row[FUTURE_LABEL_COLUMN]))
        self.assertEqual(row[LABEL_STATUS_COLUMN], "unknown_incomplete_followup")

    def test_pair_already_in_conflict_is_excluded(self):
        row = label_at_zero(make_states(initially_conflicting=True))

        self.assertTrue(pd.isna(row[FUTURE_LABEL_COLUMN]))
        self.assertEqual(row[LABEL_STATUS_COLUMN], "ongoing_excluded")

    def test_entry_after_horizon_does_not_count(self):
        row = label_at_zero(make_states(conflict_at=70))

        self.assertEqual(int(row[FUTURE_LABEL_COLUMN]), 0)
        self.assertEqual(row[LABEL_STATUS_COLUMN], "negative")

    def test_repeated_position_reports_do_not_create_new_conflict_onset(self):
        states = make_states()
        repeated_position = states["snapshot_time"].eq(10)
        states.loc[repeated_position, "time_position"] = 0
        second_aircraft_at_10 = repeated_position & states["icao24"].eq("b00002")
        states.loc[second_aircraft_at_10, "latitude"] = 33.0 + 4.0 / 60.0

        row = label_at_zero(states)

        self.assertEqual(int(row[FUTURE_LABEL_COLUMN]), 0)
        self.assertEqual(row[LABEL_STATUS_COLUMN], "negative")

    def test_future_targets_and_status_are_not_model_features(self):
        chunks = list(iter_future_conflict_labels(make_states(conflict_at=30)))
        labeled = pd.concat(chunks, ignore_index=True)
        features, target, metadata = prepare_future_model_inputs(labeled)

        self.assertNotIn(FUTURE_LABEL_COLUMN, features.columns)
        self.assertNotIn(LABEL_STATUS_COLUMN, features.columns)
        self.assertNotIn(FUTURE_LABEL_COLUMN, metadata.columns)
        self.assertNotIn(LABEL_STATUS_COLUMN, features.columns)
        self.assertEqual(features.index.tolist(), target.index.tolist())
        self.assertEqual(target.isin([0, 1]).all(), True)


if __name__ == "__main__":
    unittest.main()
