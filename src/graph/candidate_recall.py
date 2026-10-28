"""Candidate-generation recall: share of future conflicts kept as graph edges."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.graph.sparse_graph import CANDIDATE_RADIUS_NM, build_candidate_edges, build_nodes

EPISODE_GAP_S = 30


def positives_in_candidates(
    states: pd.DataFrame, positives: pd.DataFrame, radius_nm: float = CANDIDATE_RADIUS_NM
) -> pd.DataFrame:
    """Return the positive pair rows with an `in_candidates` flag.

    For each snapshot the graph is built from `states` exactly as at prediction
    time, then each positive pair is checked against that snapshot's edges.
    """
    positives = positives.copy()
    positives["in_candidates"] = False
    snapshots = states.groupby("snapshot_time")
    for snapshot_time, rows in positives.groupby("snapshot_time"):
        nodes = build_nodes(snapshots.get_group(snapshot_time))
        index_a, index_b = build_candidate_edges(nodes, radius_nm)
        ids = nodes["icao24"].to_numpy()
        edges = set(zip(ids[index_a], ids[index_b])) | set(zip(ids[index_b], ids[index_a]))
        found = [(a, b) in edges for a, b in zip(rows["icao24_a"].astype(str), rows["icao24_b"].astype(str))]
        positives.loc[rows.index, "in_candidates"] = found
    return positives


def add_episode_ids(positives: pd.DataFrame) -> pd.DataFrame:
    """Group each pair's positive rows into episodes (gap above 30 s starts a new one)."""
    positives = positives.sort_values(["icao24_a", "icao24_b", "snapshot_time"]).copy()
    pair = positives["icao24_a"].astype(str) + "|" + positives["icao24_b"].astype(str)
    new_episode = (pair != pair.shift()) | (positives["snapshot_time"].diff() > EPISODE_GAP_S)
    positives["episode"] = new_episode.cumsum()
    return positives


def main() -> None:
    manifest = pd.read_csv("data/processed/future_labels/future_label_split_manifest.csv")
    results = []
    for row in manifest.itertuples():
        states = pd.read_csv(Path("data/processed/cleaned_states") / row.session_file.replace("_future_labels", ""), low_memory=False)
        labels = pd.read_csv(row.label_file, usecols=["snapshot_time", "icao24_a", "icao24_b", "future_conflict_status"])
        positives = labels[labels["future_conflict_status"] == "positive"].reset_index(drop=True)
        checked = add_episode_ids(positives_in_candidates(states, positives))
        episodes = checked.groupby("episode")["in_candidates"].agg(["any", "first"])
        results.append(
            {
                "split": row.split,
                "session_file": row.session_file,
                "positive_rows": len(checked),
                "rows_in_candidates": int(checked["in_candidates"].sum()),
                "episodes": len(episodes),
                "episodes_any_row_in_candidates": int(episodes["any"].sum()),
                "episodes_first_row_in_candidates": int(episodes["first"].sum()),
            }
        )
        print(results[-1])
    pd.DataFrame(results).to_csv("data/processed/graph_stats/candidate_recall.csv", index=False)


if __name__ == "__main__":
    main()
