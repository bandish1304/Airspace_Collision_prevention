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
from src.features.model_inputs import MODEL_FEATURES, feature_names, prepare_model_inputs
from src.graph.sparse_graph import (
    build_candidate_edges,
    build_edge_features,
    build_nodes,
)

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


def sparse_snapshot_to_graph(
    snapshot: pd.DataFrame, future_labels: pd.DataFrame | None = None
) -> Data:
    """Convert one snapshot to a sparse PyG graph, optionally with future labels."""
    nodes = build_nodes(snapshot)
    node_values = nodes[NODE_FEATURES].to_numpy(dtype=float)

    # Aircraft without usable altitude cannot provide finite node or pair features.
    usable = np.isfinite(node_values).all(axis=1)
    nodes = nodes.loc[usable].reset_index(drop=True)
    node_values = nodes[NODE_FEATURES].to_numpy(dtype=np.float32)

    index_a, index_b = build_candidate_edges(nodes)
    if len(index_a):
        pair_features = build_edge_features(nodes, index_a, index_b)
        finite_pairs = np.isfinite(pair_features[MODEL_FEATURES].to_numpy(dtype=float)).all(axis=1)
        index_a, index_b = index_a[finite_pairs], index_b[finite_pairs]
        pair_values = pair_features.loc[finite_pairs, MODEL_FEATURES].to_numpy(dtype=np.float32)
    else:
        pair_values = np.empty((0, len(MODEL_FEATURES)), dtype=np.float32)

    message_a, message_b = index_a.copy(), index_b.copy()
    edge_labels = None
    if future_labels is not None:
        required = {
            "snapshot_time",
            "icao24_a",
            "icao24_b",
            "future_conflict",
            "future_conflict_status",
        }
        missing = required.difference(future_labels.columns)
        if missing:
            raise ValueError(f"Future labels are missing required columns: {sorted(missing)}.")
        if future_labels["snapshot_time"].nunique() > 1 or (
            not future_labels.empty
            and int(future_labels["snapshot_time"].iloc[0]) != int(snapshot["snapshot_time"].iloc[0])
        ):
            raise ValueError("Future labels must belong to the snapshot being converted.")

        known = future_labels.loc[
            future_labels["future_conflict_status"].isin(["positive", "negative"])
            & future_labels["future_conflict"].notna()
        ]
        label_by_pair: dict[tuple[str, str], float] = {}
        for row in known.itertuples(index=False):
            pair = tuple(sorted((str(row.icao24_a).lower(), str(row.icao24_b).lower())))
            if pair in label_by_pair:
                raise ValueError(f"Duplicate future label for candidate pair: {pair}.")
            label_by_pair[pair] = float(row.future_conflict)

        aircraft_ids = nodes["icao24"].to_numpy()
        selected = [
            i
            for i, (a, b) in enumerate(zip(index_a, index_b))
            if tuple(sorted((aircraft_ids[a], aircraft_ids[b]))) in label_by_pair
        ]
        selected_array = np.asarray(selected, dtype=int)
        edge_labels = np.asarray(
            [
                label_by_pair[
                    tuple(sorted((aircraft_ids[index_a[i]], aircraft_ids[index_b[i]])))
                ]
                for i in selected
            ],
            dtype=np.float32,
        )
        index_a, index_b = index_a[selected_array], index_b[selected_array]
        pair_values = pair_values[selected_array]

    edge_label_index = torch.tensor(
        np.vstack([index_a, index_b]), dtype=torch.long
    ).reshape(2, -1)
    message_edge_index = torch.tensor(
        np.vstack([message_a, message_b]), dtype=torch.long
    ).reshape(2, -1)
    aircraft_ids = nodes["icao24"].to_numpy()
    pair_ids = [
        (aircraft_ids[a], aircraft_ids[b]) for a, b in zip(index_a, index_b)
    ]
    graph = Data(
        x=torch.tensor(node_values, dtype=torch.float),
        edge_index=torch.cat([message_edge_index, message_edge_index.flip(0)], dim=1),
        edge_label_index=edge_label_index,
        pair_attr=torch.tensor(pair_values, dtype=torch.float),
        pair_ids=pair_ids,
        snapshot_time=int(snapshot["snapshot_time"].iloc[0]),
        num_nodes=len(nodes),
    )
    if edge_labels is not None:
        graph.edge_label = torch.tensor(edge_labels, dtype=torch.float)
    return graph


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
