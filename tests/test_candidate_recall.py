import unittest

import pandas as pd

from src.graph.candidate_recall import add_episode_ids, positives_in_candidates


def states(lat_b: float) -> pd.DataFrame:
    base = {
        "snapshot_time": 100,
        "time_position": 100,
        "longitude": -118.0,
        "velocity": 70.0,
        "true_track": 90.0,
        "on_ground": False,
        "baro_altitude": 1000.0,
        "geo_altitude": 1000.0,
    }
    return pd.DataFrame(
        [{**base, "icao24": "aaaaaa", "latitude": 33.0}, {**base, "icao24": "bbbbbb", "latitude": lat_b}]
    )


def positive_pair(times=(100,)) -> pd.DataFrame:
    return pd.DataFrame({"snapshot_time": list(times), "icao24_a": "aaaaaa", "icao24_b": "bbbbbb"})


class CandidateRecallTests(unittest.TestCase):
    def test_close_pair_is_kept_and_far_pair_is_missed(self):
        self.assertTrue(positives_in_candidates(states(33.05), positive_pair())["in_candidates"].all())
        self.assertFalse(positives_in_candidates(states(33.5), positive_pair())["in_candidates"].any())

    def test_episodes_split_on_gaps_over_30_seconds(self):
        episodes = add_episode_ids(positive_pair((100, 110, 200)))

        self.assertEqual(episodes["episode"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
