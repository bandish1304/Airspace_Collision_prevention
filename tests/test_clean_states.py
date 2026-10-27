import unittest

import pandas as pd

from src.data.clean_states import clean_state_observations


def make_states():
    return pd.DataFrame(
        [
            {
                "icao24": " ABC123 ",
                "snapshot_time": 100,
                "time_position": 100,
                "latitude": 33.0,
                "longitude": -118.0,
                "velocity": 70.0,
                "true_track": 90.0,
                "on_ground": False,
                "baro_altitude": 1000.0,
                "geo_altitude": None,
            },
            {
                "icao24": "abc123",
                "snapshot_time": 100,
                "time_position": 99,
                "latitude": 33.1,
                "longitude": -118.1,
                "velocity": 71.0,
                "true_track": 91.0,
                "on_ground": False,
                "baro_altitude": 1001.0,
                "geo_altitude": None,
            },
            {
                "icao24": "def456",
                "snapshot_time": 100,
                "time_position": 100,
                "latitude": 34.0,
                "longitude": -118.0,
                "velocity": 70.0,
                "true_track": 90.0,
                "on_ground": "False",
                "baro_altitude": None,
                "geo_altitude": None,
            },
            {
                "icao24": "ghi789",
                "snapshot_time": 100,
                "time_position": 100,
                "latitude": 91.0,
                "longitude": -118.0,
                "velocity": 70.0,
                "true_track": 90.0,
                "on_ground": False,
                "baro_altitude": 1000.0,
                "geo_altitude": None,
            },
            {
                "icao24": "jkl012",
                "snapshot_time": 100,
                "time_position": 70,
                "latitude": 33.0,
                "longitude": -118.0,
                "velocity": 70.0,
                "true_track": 90.0,
                "on_ground": False,
                "baro_altitude": 1000.0,
                "geo_altitude": None,
            },
        ]
    )


class CleanStateObservationsTests(unittest.TestCase):
    def test_filters_invalid_ground_stale_and_duplicate_snapshot_rows(self):
        cleaned = clean_state_observations(make_states())

        self.assertEqual(cleaned["icao24"].tolist(), ["abc123", "def456"])
        self.assertEqual(cleaned.iloc[0]["latitude"], 33.1)

    def test_retains_missing_altitude_and_marks_repeated_reports(self):
        first = make_states().iloc[[0]].copy()
        first["baro_altitude"] = None
        second = first.copy()
        second["snapshot_time"] = 110
        repeated = pd.concat([first, second], ignore_index=True)

        cleaned = clean_state_observations(repeated)

        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned["repeated_position_report"].tolist(), [False, True])
        self.assertTrue(cleaned["baro_altitude"].isna().all())

    def test_rejects_negative_position_age_limit(self):
        with self.assertRaises(ValueError):
            clean_state_observations(make_states(), max_position_age_seconds=-1)

    def test_sorts_by_aircraft_then_snapshot_time(self):
        states = pd.concat(
            [
                make_states().iloc[[0]].assign(snapshot_time=120, time_position=120),
                make_states().iloc[[0]].assign(
                    icao24="def456",
                    snapshot_time=100,
                    time_position=100,
                ),
                make_states().iloc[[0]].assign(snapshot_time=110, time_position=110),
            ],
            ignore_index=True,
        )

        cleaned = clean_state_observations(states)

        self.assertEqual(cleaned["icao24"].tolist(), ["abc123", "abc123", "def456"])
        self.assertEqual(cleaned["snapshot_time"].tolist(), [110, 120, 100])


if __name__ == "__main__":
    unittest.main()
