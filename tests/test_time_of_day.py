import unittest

import pandas as pd

from src.analysis.time_of_day import summarise_session_pairs


class TestTimeOfDay(unittest.TestCase):
    def test_summarises_rates_encounters_and_five_minute_windows(self) -> None:
        pairs = pd.DataFrame(
            [
                (100, "a", "b", 1),
                (110, "a", "b", 1),
                (410, "a", "b", 0),
            ],
            columns=["snapshot_time", "icao24_a", "icao24_b", "collision_risk"],
        )

        summary, windows = summarise_session_pairs(
            pairs,
            session_start=100,
            session_end=700,
            session_name="session.csv",
            window_minutes=5,
        )

        self.assertEqual(summary["pair_observations"], 3)
        self.assertEqual(summary["risky_pair_observations"], 2)
        self.assertAlmostEqual(summary["pair_observation_rate_pct"], 200 / 3)
        self.assertEqual(summary["aircraft_moments"], 6)
        self.assertEqual(summary["risky_aircraft_moments"], 4)
        self.assertEqual(summary["positive_pair_encounters"], 1)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows.iloc[0]["risky_pair_observations"], 2)
        self.assertEqual(windows.iloc[1]["risky_pair_observations"], 0)

    def test_separated_positive_stretches_count_as_distinct_encounters(self) -> None:
        pairs = pd.DataFrame(
            [
                (100, "a", "b", 1),
                (110, "b", "a", 1),
                (171, "a", "b", 1),
            ],
            columns=["snapshot_time", "icao24_a", "icao24_b", "collision_risk"],
        )

        summary, _ = summarise_session_pairs(
            pairs,
            session_start=100,
            session_end=200,
            session_name="session.csv",
        )

        self.assertEqual(summary["positive_pair_encounters"], 2)

    def test_rejects_missing_columns_and_invalid_duration(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            summarise_session_pairs(
                pd.DataFrame({"snapshot_time": [100]}),
                100,
                200,
                "session.csv",
            )

        pairs = pd.DataFrame(
            [(100, "a", "b", 0)],
            columns=["snapshot_time", "icao24_a", "icao24_b", "collision_risk"],
        )
        with self.assertRaisesRegex(ValueError, "positive duration"):
            summarise_session_pairs(pairs, 100, 100, "session.csv")


if __name__ == "__main__":
    unittest.main()
