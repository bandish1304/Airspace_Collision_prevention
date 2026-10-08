"""GraphSAGE baseline for edge-level collision-risk classification.

Each aircraft first gathers information from the aircraft around it, through a
stack of GraphSAGE layers. Each pair is then scored from the two aircraft it
joins together with the pair's own kinematic features.

SAGEConv does not take edge features during message passing, so the pair
features would otherwise be lost. Feeding them straight into the classifier
head keeps the pairwise signal -- the strongest one in this data -- in front
of the decision.
"""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


# Week 5, step 1: GraphSAGE encoder with a symmetric pair classifier.
class GraphSAGEEdgeClassifier(nn.Module):
    """Score every pair in a snapshot graph with a collision-risk logit.

    Outputs logits rather than probabilities so training can use
    ``BCEWithLogitsLoss`` with a positive-class weight for the imbalance.
    """

    def __init__(
        self,
        node_features: int,
        pair_features: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        aggr: str = "mean",
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.dropout = dropout
        self.convs = nn.ModuleList()
        in_channels = node_features
        for _ in range(num_layers):
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
            in_channels = hidden_channels

        # Three views of the two endpoints plus the pair's own features.
        self.head = nn.Sequential(
            nn.Linear(3 * hidden_channels + pair_features, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x

    def pair_representations(self, graph: Data) -> torch.Tensor:
        """Return symmetric endpoint embeddings concatenated with pair features."""
        embedding = self.encode(graph.x, graph.edge_index)
        first = embedding[graph.edge_label_index[0]]
        second = embedding[graph.edge_label_index[1]]

        # Which aircraft is "a" and which is "b" is an accident of row order,
        # so the pair representation must not depend on it. Concatenating the
        # two embeddings would; sum, absolute difference and product do not.
        return torch.cat(
            [first + second, (first - second).abs(), first * second, graph.pair_attr],
            dim=-1,
        )

    def classify_pair_representations(self, pair: torch.Tensor) -> torch.Tensor:
        """Return one collision-risk logit for every pair representation."""
        return self.head(pair).squeeze(-1)

    def forward(self, graph: Data) -> torch.Tensor:
        return self.classify_pair_representations(self.pair_representations(graph))
