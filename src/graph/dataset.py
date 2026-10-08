"""Dataset utilities for sequencing graph snapshots across time windows."""

from __future__ import annotations

from typing import Iterator

import pandas as pd

from src.graph.construction import build_snapshot_graph


# Week 4, step 4: organize many aircraft snapshots into a sequence of graphs so
# the GNN can consume temporal windows of airspace interactions.
class GraphSnapshotDataset:
    """Return one graph object per snapshot time for the GNN training pipeline."""

    def __init__(
        self,
        states: pd.DataFrame,
        max_distance_nm: float = 50.0,
    ) -> None:
        if states.empty:
            raise ValueError("states must contain at least one row.")

        self.states = states.copy().reset_index(drop=True)
        self.max_distance_nm = max_distance_nm
        self.snapshot_times = sorted(self.states["snapshot_time"].dropna().unique().tolist())

    def __len__(self) -> int:
        return len(self.snapshot_times)

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0 or index >= len(self.snapshot_times):
            raise IndexError("Graph snapshot index out of range.")

        snapshot_time = self.snapshot_times[index]
        snapshot = self.states.loc[self.states["snapshot_time"] == snapshot_time].copy()
        graph = build_snapshot_graph(snapshot, max_distance_nm=self.max_distance_nm)
        return {
            "snapshot_time": int(snapshot_time),
            "graph": graph,
            "num_nodes": graph["num_nodes"],
            "num_edges": graph["num_edges"],
        }

    def __iter__(self) -> Iterator[dict[str, object]]:
        for index in range(len(self)):
            yield self[index]


if __name__ == "__main__":
    sample = pd.read_csv("data/raw/sessions/states_socal_backfill_1790375809.csv")
    dataset = GraphSnapshotDataset(sample, max_distance_nm=50.0)
    first = dataset[0]
    print(f"num_snapshots={len(dataset)}")
    print(f"first_snapshot_time={first['snapshot_time']}")
    print(f"first_graph_nodes={first['graph']['num_nodes']}")
    print(f"first_graph_edges={first['graph']['num_edges']}")
