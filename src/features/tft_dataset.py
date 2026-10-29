"""PyTorch dataset for pairwise trajectory windows and future-risk labels."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.features.sequences import WindowSet


class TFTWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """Expose fixed-length encoder windows and binary future-risk targets.

    Each item is ``(encoder, target)`` where ``encoder`` has shape
    ``(window_frames, n_features)`` and ``target`` is the future-risk label
    for that window. A DataLoader batches encoders as
    ``(batch, window_frames, n_features)`` for a sequence model.
    """

    def __init__(self, windows: WindowSet) -> None:
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError("TFT window index out of range.")

        encoder = self.windows.gather(np.asarray([index], dtype=np.intp))[0]
        target = self.windows.y_future[index]
        return (
            torch.from_numpy(encoder),
            torch.tensor(target, dtype=torch.float32),
        )
