"""Score the validation session with the saved tuned GNN and TFT.

These scores exist only to choose each model's decision threshold on validation
data before the held-out test predictions are evaluated.
"""

from __future__ import annotations

import pandas as pd
import torch

from src.evaluation.week10_test_set import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    TFT_CHECKPOINT,
    align_predictions,
    build_session_windows,
    score_windows,
)
from src.features.sequence_splits import DEFAULT_SESSION_NAMES
from src.gnn.data import FeatureScaler, NODE_FEATURES
from src.gnn.future_tuning import load_graphs
from src.gnn.model import GraphSAGEEdgeClassifier

VALIDATION_SESSION = DEFAULT_SESSION_NAMES[3]
GNN_MODEL = PROJECT_ROOT / "models" / "future_graphsage.pt"
GNN_VALIDATION_PREDICTIONS = OUTPUT_DIR / "gnn_validation_predictions.csv"
TFT_VALIDATION_PREDICTIONS = OUTPUT_DIR / "tft_validation_predictions.csv"
ALIGNED_VALIDATION_PREDICTIONS = OUTPUT_DIR / "aligned_validation_predictions.csv"


def score_gnn_validation() -> pd.DataFrame:
    """Score validation candidate pairs with the saved best-epoch GraphSAGE."""
    saved = torch.load(GNN_MODEL, map_location="cpu", weights_only=False)
    config = saved["config"]
    scaler = FeatureScaler()
    scaler.node_mean = saved["scaler"]["node_mean"]
    scaler.node_std = saved["scaler"]["node_std"]
    scaler.pair_mean = saved["scaler"]["pair_mean"]
    scaler.pair_std = saved["scaler"]["pair_std"]

    graphs = scaler.transform(load_graphs("validation"))
    model = GraphSAGEEdgeClassifier(
        node_features=len(NODE_FEATURES),
        pair_features=graphs[0].pair_attr.size(1),
        hidden_channels=config["hidden_channels"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        aggr=config["aggr"],
    )
    model.load_state_dict(saved["model_state_dict"])
    model.eval()

    rows = []
    with torch.no_grad():
        for graph in graphs:
            probabilities = torch.sigmoid(model(graph)).numpy()
            if len(probabilities) != len(graph.pair_ids):
                raise ValueError("Validation probabilities and pair IDs do not align.")
            for (icao24_a, icao24_b), label, probability in zip(
                graph.pair_ids, graph.edge_label.numpy(), probabilities, strict=True
            ):
                rows.append(
                    {
                        "snapshot_time": graph.snapshot_time,
                        "icao24_a": icao24_a,
                        "icao24_b": icao24_b,
                        "future_conflict": int(label),
                        "probability": float(probability),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Scoring validation pairs with the saved GNN...", flush=True)
    gnn = score_gnn_validation()
    gnn.to_csv(GNN_VALIDATION_PREDICTIONS, index=False)
    print(f"  GNN validation pairs: {len(gnn):,} (positives {int(gnn.future_conflict.sum()):,})")

    if TFT_VALIDATION_PREDICTIONS.is_file():
        print(f"Reusing saved TFT validation scores: {TFT_VALIDATION_PREDICTIONS}")
        tft = pd.read_csv(TFT_VALIDATION_PREDICTIONS)
    else:
        print("Building TFT validation windows...", flush=True)
        windows, tft = build_session_windows(VALIDATION_SESSION, "validation")
        print(f"Scoring {len(windows):,} validation windows...", flush=True)
        tft["probability"] = score_windows(windows, TFT_CHECKPOINT)
        tft.to_csv(TFT_VALIDATION_PREDICTIONS, index=False)

    aligned, summary = align_predictions(gnn, tft)
    aligned.to_csv(ALIGNED_VALIDATION_PREDICTIONS, index=False)
    print({key: value for key, value in summary.items() if key != "label_agreement"})


if __name__ == "__main__":
    main()
