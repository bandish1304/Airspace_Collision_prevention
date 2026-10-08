import unittest

import numpy as np
import pandas as pd

from src.evaluation.splits import temporal_split


def _window(start: int = 1_790_000_000, frames: int = 349, step: int = 10) -> pd.Series:
    return pd.Series(start + step * np.arange(frames))


class TestSingleWindow(unittest.TestCase):
    def setUp(self) -> None:
        self.times = _window()
        self.parts = temporal_split(self.times, gap_seconds=120)

    def _times_in(self, part: str) -> pd.Series:
        return self.times[self.parts == part]

    def test_past_trains_and_future_tests(self) -> None:
        self.assertLess(self._times_in("train").max(), self._times_in("validation").min())
        self.assertLess(self._times_in("validation").max(), self._times_in("test").min())

    def test_gap_separates_each_part(self) -> None:
        self.assertGreaterEqual(
            self._times_in("validation").min() - self._times_in("train").max(), 120
        )
        self.assertGreaterEqual(
            self._times_in("test").min() - self._times_in("validation").max(), 120
        )

    def test_parts_are_roughly_sixty_twenty_twenty(self) -> None:
        span = self.times.max() - self.times.min()
        train_span = self._times_in("train").max() - self.times.min()
        self.assertAlmostEqual(train_span / span, 0.6, delta=0.02)

    def test_repeated_times_get_the_same_label(self) -> None:
        # A pair table has thousands of rows per frame; all must land together.
        rows = pd.concat([self.times, self.times, self.times], ignore_index=True)
        parts = temporal_split(rows, gap_seconds=120)
        self.assertEqual(
            pd.DataFrame({"t": rows, "p": parts}).groupby("t")["p"].nunique().max(), 1
        )


class TestSeveralSessions(unittest.TestCase):
    def test_whole_sessions_are_never_split(self) -> None:
        # Five 30-minute poll sessions, four hours apart.
        starts = 1_790_000_000 + 4 * 3600 * np.arange(5)
        times = pd.concat(
            [_window(start=int(s), frames=180) for s in starts], ignore_index=True
        )
        parts = temporal_split(times)
        session = (times - times.min()) // (4 * 3600)

        per_session = pd.DataFrame({"session": session, "part": parts})
        self.assertEqual(per_session.groupby("session")["part"].nunique().max(), 1)
        self.assertEqual(set(parts), {"train", "validation", "test"})
        self.assertNotIn("gap", set(parts))

    def test_three_sessions_give_one_part_each(self) -> None:
        starts = 1_790_000_000 + 4 * 3600 * np.arange(3)
        times = pd.concat(
            [_window(start=int(s), frames=180) for s in starts], ignore_index=True
        )
        parts = temporal_split(times)
        self.assertEqual(list(pd.unique(parts)), ["train", "validation", "test"])


class TestArguments(unittest.TestCase):
    def test_fractions_must_leave_room_for_test(self) -> None:
        with self.assertRaises(ValueError):
            temporal_split(_window(), train_fraction=0.7, validation_fraction=0.3)

    def test_negative_gap_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            temporal_split(_window(), gap_seconds=-1)


if __name__ == "__main__":
    unittest.main()
