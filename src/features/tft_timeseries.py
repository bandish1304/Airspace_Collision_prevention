"""Convert fixed-length trajectory windows to PyTorch Forecasting datasets."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import ConcatDataset, Dataset, TensorDataset

from src.features.tft_dataset import TFTWindowDataset

TARGET_COLUMN = "collision_risk_target"
GROUP_COLUMN = "window_id"
TIME_COLUMN = "time_idx"


def collect_windows(
    dataset: Dataset,
    feature_names: list[str],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize selected windows from original or SMOTE-resampled data."""
    if not feature_names:
        raise ValueError("feature_names must contain at least one feature.")
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional.")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    offset = 0
    for leaf in _window_leaves(dataset):
        stop = offset + len(leaf)
        local_indices = indices[(indices >= offset) & (indices < stop)] - offset
        if len(local_indices):
            xs.append(leaf.windows.gather(local_indices))
            ys.append(leaf.windows.y_future[local_indices].astype(np.float32))
        offset = stop

    if xs:
        return np.concatenate(xs), np.concatenate(ys)
    raise ValueError("No requested windows were found in the dataset.")


def collect_resampled_windows(
    dataset: TensorDataset,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize selected windows from the SMOTE TensorDataset."""
    if len(dataset.tensors) != 2:
        raise ValueError("Resampled data must contain feature and label tensors.")
    features, labels = dataset.tensors
    selected = torch.as_tensor(indices, dtype=torch.long)
    return (
        features[selected].detach().cpu().numpy(),
        labels[selected].detach().cpu().numpy().astype(np.float32),
    )


def build_timeseries_dataset(
    windows: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
) -> TimeSeriesDataSet:
    """Create one encoder-plus-target sequence per window for a scalar TFT."""
    frame = _window_frame(windows, labels, feature_names)
    encoder_length = windows.shape[1]
    return TimeSeriesDataSet(
        frame,
        time_idx=TIME_COLUMN,
        target=TARGET_COLUMN,
        group_ids=[GROUP_COLUMN],
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=1,
        max_prediction_length=1,
        time_varying_known_reals=[TIME_COLUMN],
        time_varying_unknown_reals=feature_names,
        target_normalizer=None,
        add_relative_time_idx=True,
        add_encoder_length=False,
        allow_missing_timesteps=False,
    )


def _window_frame(
    windows: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    if windows.ndim != 3:
        raise ValueError("windows must have shape (samples, time, features).")
    if windows.shape[0] == 0:
        raise ValueError("At least one window is required.")
    if windows.shape[2] != len(feature_names):
        raise ValueError("Window feature count does not match feature_names.")
    if labels.shape != (windows.shape[0],):
        raise ValueError("labels must have one scalar target per window.")
    if not np.isfinite(windows).all():
        raise ValueError("Window features must all be finite.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Binary TFT targets must be 0 or 1.")

    encoder_length = windows.shape[1]
    sequence_length = encoder_length + 1
    frame = pd.DataFrame(
        {
            GROUP_COLUMN: np.repeat(
                [f"window_{index}" for index in range(len(windows))],
                sequence_length,
            ),
            TIME_COLUMN: np.tile(
                np.arange(sequence_length, dtype=np.int64), len(windows)
            ),
        }
    )
    for feature_index, name in enumerate(feature_names):
        values = np.zeros((len(windows), sequence_length), dtype=np.float32)
        values[:, :encoder_length] = windows[:, :, feature_index]
        frame[name] = values.reshape(-1)

    target_values = np.zeros((len(windows), sequence_length), dtype=np.float32)
    target_values[:, -1] = labels
    frame[TARGET_COLUMN] = target_values.reshape(-1)
    return frame


def validation_dataset(
    training_dataset: TimeSeriesDataSet,
    windows: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
) -> TimeSeriesDataSet:
    """Create a validation dataset using training-fitted encoders and scalers."""
    return TimeSeriesDataSet.from_dataset(
        training_dataset,
        _window_frame(windows, labels, feature_names),
        predict=False,
        stop_randomization=True,
    )


def _window_leaves(dataset: Dataset) -> Iterator[TFTWindowDataset]:
    if isinstance(dataset, TFTWindowDataset):
        yield dataset
    elif isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            yield from _window_leaves(child)
