"""Turn pair-feature tables into PyTorch Geometric graphs for the GNN baseline.

The graphs are built from the same pair table the Week 3 baselines use, not
from the Week 4 dictionary graphs. That table already applies the
stale-position filter construction.py skips, and it is vectorised, so the GNN
and the logistic regressions see exactly the same aircraft, edges and labels.
Without that the Week 10 comparison would be measuring two different datasets.

One graph per snapshot:

    x                 one row per aircraft in the frame
    edge_index        every pair within range, both directions, for message passing
    edge_label_index  every pair once, for classification
    pair_attr         kinematic features of each pair, aligned to edge_label_index
    edge_label        collision-risk label of each pair
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# Labelling, the time-to-CPA treatment and row filtering all live in
# model_inputs, shared with the baselines, so both models see the same rows.
from src.features.model_inputs import feature_names, prepare_model_inputs

NODE_FEATURES = [
    "latitude",
    "longitude",
    "altitude_m",
    "velocity_mps",
    "track_sin",
    "track_cos",
]

PAIR_COLUMNS = [
    "snapshot_time",
    "icao24_a",
    "icao24_b",
    "latitude_a",
    "longitude_a",
    "altitude_a_m",
    "velocity_a_mps",
    "true_track_a_deg",
    "latitude_b",
    "longitude_b",
    "altitude_b_m",
    "velocity_b_mps",
    "true_track_b_deg",
    "lateral_distance_nm",
    "vertical_separation_m",
    "closing_speed_mps",
    "bearing_difference_deg",
    "time_to_cpa_s",
]


# Week 5, step 1: read only the columns the graphs need from a pair CSV.
def load_pairs(pairs_path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Load a pair-feature CSV. Full backfill files run to several million rows."""
    return pd.read_csv(pairs_path, usecols=PAIR_COLUMNS, nrows=nrows)


# Week 5, step 1: build one labelled graph per snapshot.
def pairs_to_graphs(
    pairs: pd.DataFrame,
    include_label_features: bool = True,
    lateral_separation_nm: float = 5.0,
    vertical_separation_ft: float = 1_000.0,
) -> list[Data]:
    """Return one PyG ``Data`` object per snapshot in the pair table.

    Pairs whose separation cannot be computed are dropped, since they cannot
    be labelled. Pairs that never reach a closest approach are kept, with
    their infinite time-to-CPA capped and flagged; see model_inputs.
    """
    prepared = prepare_model_inputs(
        pairs,
        lateral_separation_nm=lateral_separation_nm,
        vertical_separation_ft=vertical_separation_ft,
    )
    pair_features = pair_feature_names(include_label_features)

    graphs = []
    for snapshot_time, frame in prepared.groupby("snapshot_time", sort=True):
        graphs.append(_frame_to_graph(frame, snapshot_time, pair_features))
    return graphs


# Week 5, step 1: name the pair features a graph will carry.
def pair_feature_names(include_label_features: bool = True) -> list[str]:
    return feature_names(include_label_features)


def _frame_to_graph(
    frame: pd.DataFrame, snapshot_time: int, pair_features: list[str]
) -> Data:
    aircraft = pd.concat(
        [
            _endpoint_frame(frame, "a"),
            _endpoint_frame(frame, "b"),
        ],
        ignore_index=True,
    ).drop_duplicates(subset="icao24", keep="first").reset_index(drop=True)

    node_of = pd.Index(aircraft["icao24"])
    source = node_of.get_indexer(frame["icao24_a"])
    target = node_of.get_indexer(frame["icao24_b"])

    track = np.radians(aircraft["true_track_deg"].to_numpy(dtype=float))
    # Heading is circular: 359 and 1 degree are neighbours, which a raw angle
    # would place at opposite ends of the scale.
    node_values = np.column_stack(
        [
            aircraft["latitude"],
            aircraft["longitude"],
            aircraft["altitude_m"],
            aircraft["velocity_mps"],
            np.sin(track),
            np.cos(track),
        ]
    )

    label_index = torch.tensor(np.vstack([source, target]), dtype=torch.long)
    return Data(
        x=torch.tensor(node_values, dtype=torch.float),
        # Message passing has to run both ways along each pair; classification
        # only needs each pair once.
        edge_index=torch.cat([label_index, label_index.flip(0)], dim=1),
        edge_label_index=label_index,
        pair_attr=torch.tensor(frame[pair_features].to_numpy(dtype=float),
                               dtype=torch.float),
        edge_label=torch.tensor(frame["collision_risk"].to_numpy(dtype=float),
                                dtype=torch.float),
        snapshot_time=int(snapshot_time),
        num_nodes=len(aircraft),
    )


def _endpoint_frame(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "icao24": frame[f"icao24_{side}"].to_numpy(),
            "latitude": frame[f"latitude_{side}"].to_numpy(),
            "longitude": frame[f"longitude_{side}"].to_numpy(),
            "altitude_m": frame[f"altitude_{side}_m"].to_numpy(),
            "velocity_mps": frame[f"velocity_{side}_mps"].to_numpy(),
            "true_track_deg": frame[f"true_track_{side}_deg"].to_numpy(),
        }
    )


# Week 5, step 1: standardise features using training graphs only.
class FeatureScaler:
    """Standardise node and pair features to zero mean and unit variance.

    Fit on the training graphs alone. Fitting on everything would let the
    validation and test distributions leak into the inputs the model trains on.
    """

    def __init__(self) -> None:
        self.node_mean: torch.Tensor | None = None
        self.node_std: torch.Tensor | None = None
        self.pair_mean: torch.Tensor | None = None
        self.pair_std: torch.Tensor | None = None

    def fit(self, graphs: Iterable[Data]) -> FeatureScaler:
        # Statistics in float64: longitude sits near -118, where float32 can
        # only resolve about 8e-6 degrees, enough to leave a visible offset
        # once a narrow spread is divided out.
        node_stats = _RunningStats()
        pair_stats = _RunningStats()
        for graph in graphs:
            node_stats.update(graph.x)
            pair_stats.update(graph.pair_attr)
        self.node_mean, self.node_std = node_stats.finish()
        self.pair_mean, self.pair_std = pair_stats.finish()
        return self

    def transform(self, graphs: list[Data]) -> list[Data]:
        if self.node_mean is None:
            raise RuntimeError("FeatureScaler must be fit before transform.")
        scaled = []
        for graph in graphs:
            graph = graph.clone()
            graph.x = ((graph.x.double() - self.node_mean) / self.node_std).float()
            graph.pair_attr = (
                (graph.pair_attr.double() - self.pair_mean) / self.pair_std
            ).float()
            scaled.append(graph)
        return scaled


class _RunningStats:
    """Merge per-graph means and variances without retaining every graph."""

    def __init__(self) -> None:
        self.count = 0
        self.mean: torch.Tensor | None = None
        self.sum_squared_deviations: torch.Tensor | None = None

    def update(self, values: torch.Tensor) -> None:
        if values.ndim != 2 or values.size(0) == 0:
            return
        batch = values.double()
        batch_count = batch.size(0)
        batch_mean = batch.mean(0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(0)
        if self.mean is None:
            self.count = batch_count
            self.mean = batch_mean
            self.sum_squared_deviations = batch_m2
            return

        assert self.sum_squared_deviations is not None
        combined_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.sum_squared_deviations += (
            batch_m2 + delta.square() * self.count * batch_count / combined_count
        )
        self.mean += delta * batch_count / combined_count
        self.count = combined_count

    def finish(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mean is None or self.sum_squared_deviations is None:
            raise ValueError("Cannot fit feature scaling without graph observations.")
        if self.count > 1:
            variance = self.sum_squared_deviations / (self.count - 1)
            std = torch.sqrt(variance)
        else:
            std = torch.ones_like(self.mean)
        return self.mean, torch.where(std > 0, std, torch.ones_like(std))
