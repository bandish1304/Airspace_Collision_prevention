import unittest

import pandas as pd

from src.analysis.trajectory_duration import (
    build_trajectory_segments,
    kaplan_meier,
    summarize_segments,
)


def _states(rows: list[tuple[str, int, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "icao24": aircraft,
                "snapshot_time": timestamp,
                "time_position": timestamp,
                "latitude": 34.0,
                "longitude": -118.0,
                "on_ground": on_ground,
            }
            for aircraft, timestamp, on_ground in rows
        ]
    )


class TestTrajectoryDuration(unittest.TestCase):
    def test_splits_after_gap_and_marks_collection_edge_censoring(self) -> None:
        states = _states(
            [
                ("left", 100, False),
                ("left", 110, False),
                ("left", 140, False),
                ("right", 110, False),
                ("right", 140, False),
                ("right", 160, False),
                ("complete", 110, False),
                ("complete", 120, False),
                ("ground", 100, True),
                ("ground", 140, True),
                ("ground", 170, True),
            ]
        )
        segments = build_trajectory_segments(states, gap_seconds=20)

        left = segments.loc[segments["icao24"] == "left"].sort_values("start_time")
        self.assertEqual(len(left), 2)
        self.assertEqual(left.iloc[0]["censoring_status"], "left")
        self.assertEqual(left.iloc[1]["censoring_status"], "complete")

        right = segments.loc[segments["icao24"] == "right"].iloc[-1]
        self.assertEqual(right["censoring_status"], "right")
        self.assertFalse(right["event_observed"])
        self.assertEqual(right["survival_duration_seconds"], 30)

        complete = segments.loc[segments["icao24"] == "complete"].iloc[0]
        self.assertEqual(complete["observed_duration_seconds"], 10)
        self.assertEqual(complete["censoring_status"], "complete")
        self.assertNotIn("ground", set(segments["icao24"]))

    def test_censored_survival_excludes_left_censored_segments(self) -> None:
        states = _states(
            [
                ("left", 100, False),
                ("left", 110, False),
                ("complete", 110, False),
                ("complete", 120, False),
                ("right", 110, False),
                ("right", 140, False),
                ("ground", 150, True),
            ]
        )
        segments = build_trajectory_segments(states, gap_seconds=20)
        survival = kaplan_meier(segments)
        summary = summarize_segments(segments, survival)

        self.assertEqual(int(summary["segments_in_kaplan_meier"].iloc[0]), 3)
        self.assertEqual(int(summary["left_censored_excluded_from_kaplan_meier"].iloc[0]), 1)
        self.assertEqual(int(survival.iloc[-1]["censored"]), 1)
        self.assertAlmostEqual(float(survival.iloc[-1]["survival"]), 1 / 3)

    def test_rejects_missing_required_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            build_trajectory_segments(pd.DataFrame({"icao24": ["x"]}))


if __name__ == "__main__":
    unittest.main()
