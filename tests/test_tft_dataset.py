import unittest

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.features.model_inputs import MODEL_FEATURES
from src.features.sequences import build_windows
from src.features.tft_dataset import TFTWindowDataset


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


class TFTWindowDatasetTests(unittest.TestCase):
    def test_item_contains_encoder_window_and_future_target(self):
        pairs = make_pairs(
            range(0, 70, 10),
            [0, 0, 0, 0, 1, 0, 0],
        )
        windows = build_windows(pairs, window_frames=3, horizon_frames=2)
        dataset = TFTWindowDataset(windows)

        encoder, target = dataset[0]

        self.assertEqual(tuple(encoder.shape), (3, len(MODEL_FEATURES)))
        self.assertEqual(encoder.dtype, torch.float32)
        self.assertTrue(torch.equal(encoder[:, 0], torch.tensor([0.0, 1.0, 2.0])))
        self.assertEqual(target.dtype, torch.float32)
        self.assertEqual(target.item(), 1.0)

    def test_dataloader_batches_encoder_sequences_and_targets(self):
        pairs = make_pairs(range(0, 80, 10), [0, 0, 0, 0, 1, 0, 0, 0])
        windows = build_windows(pairs, window_frames=3, horizon_frames=2)
        batch_encoder, batch_target = next(
            iter(DataLoader(TFTWindowDataset(windows), batch_size=2))
        )

        self.assertEqual(
            tuple(batch_encoder.shape), (2, 3, len(MODEL_FEATURES))
        )
        self.assertEqual(tuple(batch_target.shape), (2,))
        self.assertEqual(batch_encoder.dtype, torch.float32)
        self.assertEqual(batch_target.tolist(), [1.0, 1.0])

    def test_empty_window_set_is_a_valid_empty_dataset(self):
        pairs = make_pairs([0, 10, 20], [0, 0, 0])
        windows = build_windows(pairs, window_frames=5, horizon_frames=1)

        self.assertEqual(len(TFTWindowDataset(windows)), 0)

    def test_out_of_range_index_raises(self):
        pairs = make_pairs(range(0, 50, 10), [0, 0, 0, 0, 0])
        dataset = TFTWindowDataset(
            build_windows(pairs, window_frames=2, horizon_frames=1)
        )

        with self.assertRaises(IndexError):
            dataset[len(dataset)]


if __name__ == "__main__":
    unittest.main()
