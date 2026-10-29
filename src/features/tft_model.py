"""PyTorch Forecasting model configuration for pairwise trajectory sequences."""

from __future__ import annotations

from typing import Any

import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MultiHorizonMetric


class BinaryCrossEntropyWithLogits(MultiHorizonMetric):
    """Binary classification loss for scalar TFT logits."""

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = y_pred.squeeze(-1)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            target.to(dtype=logits.dtype),
            reduction="none",
        )

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(y_pred.squeeze(-1))

    def to_quantiles(
        self, y_pred: torch.Tensor, quantiles: list[float] | None = None
    ) -> torch.Tensor:
        return self.to_prediction(y_pred).unsqueeze(-1)

    def rescale_parameters(
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder: Any,
    ) -> torch.Tensor:
        return parameters


def build_tft_model(
    training_dataset: TimeSeriesDataSet,
    *,
    hidden_size: int = 16,
    attention_head_size: int = 4,
    dropout: float = 0.1,
    hidden_continuous_size: int = 8,
    learning_rate: float = 1e-3,
) -> TemporalFusionTransformer:
    """Create a binary collision-risk TFT from a training TimeSeriesDataSet."""
    if hidden_size < 1:
        raise ValueError("hidden_size must be at least 1.")
    if attention_head_size < 1:
        raise ValueError("attention_head_size must be at least 1.")
    if hidden_continuous_size < 1:
        raise ValueError("hidden_continuous_size must be at least 1.")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in the range [0, 1).")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero.")

    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=1,
        loss=BinaryCrossEntropyWithLogits(),
        optimizer="adam",
        reduce_on_plateau_patience=4,
    )
