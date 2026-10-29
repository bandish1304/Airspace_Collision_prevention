"""Training-only SMOTE for fixed-length trajectory windows."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from torch.utils.data import ConcatDataset, Dataset, TensorDataset

from src.features.tft_dataset import TFTWindowDataset

SMOTE_NEGATIVE_MULTIPLIER = 10
SMOTE_TARGET_RATIO = 0.5
SMOTE_RANDOM_STATE = 42


def smote_training_windows(
    training_dataset: Dataset,
    *,
    negative_multiplier: int = SMOTE_NEGATIVE_MULTIPLIER,
    target_ratio: float = SMOTE_TARGET_RATIO,
    random_state: int = SMOTE_RANDOM_STATE,
) -> tuple[TensorDataset, dict[str, int | float]]:
    """SMOTE-resample flattened training windows and return sequence tensors.

    At most ``negative_multiplier`` negative windows are sampled per positive
    window. SMOTE then raises the positive count to ``target_ratio`` times the
    sampled-negative count. Only the supplied training dataset is read.
    """
    if negative_multiplier < 1:
        raise ValueError("negative_multiplier must be at least 1.")
    if not 0 < target_ratio <= 1:
        raise ValueError("target_ratio must be in the range (0, 1].")

    leaves = list(_window_datasets(training_dataset))
    if not leaves:
        raise ValueError("Training dataset contains no TFTWindowDataset items.")

    labels = np.concatenate([leaf.windows.y_future for leaf in leaves]).astype(
        np.int8, copy=False
    )
    positive_indices = np.flatnonzero(labels == 1)
    negative_indices = np.flatnonzero(labels == 0)
    if len(positive_indices) < 2:
        raise ValueError("SMOTE requires at least two positive training windows.")
    if len(negative_indices) == 0:
        raise ValueError("SMOTE requires negative training windows.")

    sampled_negative_count = min(
        len(negative_indices), negative_multiplier * len(positive_indices)
    )
    desired_positive_count = int(np.ceil(target_ratio * sampled_negative_count))
    if desired_positive_count <= len(positive_indices):
        raise ValueError(
            "The SMOTE target ratio does not request additional positive windows."
        )

    rng = np.random.default_rng(random_state)
    sampled_negatives = rng.choice(
        negative_indices, size=sampled_negative_count, replace=False
    )
    selected_indices = np.sort(np.concatenate([positive_indices, sampled_negatives]))
    selected_x = _gather_windows(leaves, selected_indices)
    selected_y = labels[selected_indices]

    sequence_shape = selected_x.shape[1:]
    flattened_x = selected_x.reshape(len(selected_x), -1)
    smote = SMOTE(
        sampling_strategy={1: desired_positive_count},
        k_neighbors=min(5, len(positive_indices) - 1),
        random_state=random_state,
    )
    resampled_x, resampled_y = smote.fit_resample(flattened_x, selected_y)
    resampled_x = np.asarray(resampled_x, dtype=np.float32).reshape(
        -1, *sequence_shape
    )
    resampled_y = np.asarray(resampled_y, dtype=np.float32)

    if int(resampled_y.sum()) != desired_positive_count:
        raise RuntimeError("SMOTE did not generate the requested positive count.")

    dataset = TensorDataset(
        torch.from_numpy(resampled_x),
        torch.from_numpy(resampled_y),
    )
    metadata: dict[str, int | float] = {
        "original_positive_windows": len(positive_indices),
        "original_negative_windows": len(negative_indices),
        "sampled_negative_windows": sampled_negative_count,
        "synthetic_positive_windows": desired_positive_count - len(positive_indices),
        "resampled_positive_windows": desired_positive_count,
        "resampled_negative_windows": sampled_negative_count,
        "resampled_windows": len(dataset),
        "positive_to_negative_ratio": desired_positive_count / sampled_negative_count,
    }
    return dataset, metadata


def _window_datasets(dataset: Dataset) -> Iterator[TFTWindowDataset]:
    if isinstance(dataset, TFTWindowDataset):
        yield dataset
    elif isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            yield from _window_datasets(child)
    else:
        raise TypeError(
            "SMOTE training input must be a TFTWindowDataset or a ConcatDataset "
            "containing TFTWindowDataset instances."
        )


def _gather_windows(
    leaves: list[TFTWindowDataset], selected_indices: np.ndarray
) -> np.ndarray:
    gathered: list[np.ndarray] = []
    offset = 0
    for leaf in leaves:
        stop = offset + len(leaf)
        local_indices = selected_indices[
            (selected_indices >= offset) & (selected_indices < stop)
        ] - offset
        if len(local_indices):
            gathered.append(leaf.windows.gather(local_indices))
        offset = stop
    if not gathered:
        raise RuntimeError("No selected training windows could be materialized.")
    return np.concatenate(gathered, axis=0)
