"""PyTorch Forecasting model configuration for pairwise trajectory sequences."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import torch
from torch import Tensor, nn
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MultiHorizonMetric


class _VariableSelectionNetworkLike(Protocol):
    input_sizes: Mapping[str, int]
    single_variable_grns: nn.ModuleDict
    prescalers: nn.ModuleDict


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


class EqualWeightVariableSelection(nn.Module):
    """Encode each variable independently and combine them without learned weights."""

    def __init__(
        self,
        input_sizes: Mapping[str, int],
        single_variable_grns: nn.ModuleDict,
        prescalers: nn.ModuleDict,
    ) -> None:
        super().__init__()
        self.input_names = tuple(input_sizes)
        self.single_variable_grns = single_variable_grns
        self.prescalers = prescalers

    def forward(
        self,
        x: dict[str, Tensor],
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not self.input_names:
            if context is None:
                raise ValueError("A context tensor is required when no variables exist.")
            outputs = torch.zeros_like(context)
            weights = outputs.new_zeros((*outputs.shape[:-1], 1, 0))
            return outputs, weights

        encoded_variables = []
        for name in self.input_names:
            variable = x[name]
            if name in self.prescalers:
                variable = self.prescalers[name](variable)
            encoded_variables.append(self.single_variable_grns[name](variable))

        stacked = torch.stack(encoded_variables, dim=-1)
        outputs = stacked.mean(dim=-1)
        weights = outputs.new_full(
            (*outputs.shape[:-1], 1, len(self.input_names)),
            1.0 / len(self.input_names),
        )
        return outputs, weights


class TFTWithoutLearnedVariableSelection(TemporalFusionTransformer):
    """TFT whose encoder/decoder variables are combined with fixed equal weights."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.encoder_variable_selection = self._without_learned_weights(
            self.encoder_variable_selection
        )
        self.decoder_variable_selection = self._without_learned_weights(
            self.decoder_variable_selection
        )
        if self.hparams.static_categoricals or self.hparams.static_reals:
            self.static_variable_selection = self._without_learned_weights(
                self.static_variable_selection
            )

    @staticmethod
    def _without_learned_weights(
        variable_selection: _VariableSelectionNetworkLike,
    ) -> EqualWeightVariableSelection:
        return EqualWeightVariableSelection(
            input_sizes=variable_selection.input_sizes,
            single_variable_grns=variable_selection.single_variable_grns,
            prescalers=variable_selection.prescalers,
        )


def build_tft_model(
    training_dataset: TimeSeriesDataSet,
    *,
    hidden_size: int = 16,
    attention_head_size: int = 4,
    dropout: float = 0.1,
    hidden_continuous_size: int = 8,
    learning_rate: float = 1e-3,
    disable_learned_variable_selection: bool = False,
) -> TemporalFusionTransformer:
    """Create a binary collision-risk TFT from a training TimeSeriesDataSet.

    When requested, the ablation retains per-variable transformations but
    replaces learned variable-selection weights with a fixed equal-weight mean.
    """
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

    model_class = (
        TFTWithoutLearnedVariableSelection
        if disable_learned_variable_selection
        else TemporalFusionTransformer
    )
    return model_class.from_dataset(
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
