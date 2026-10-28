"""Tune GraphSAGE settings on the future-conflict validation split only."""

from __future__ import annotations

import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from src.evaluation.future_label_splits import DEFAULT_MANIFEST_PATH, PROJECT_ROOT
from src.features.model_inputs import MODEL_FEATURES
from src.gnn.data import FeatureScaler, NODE_FEATURES, sparse_snapshot_to_graph
from src.gnn.model import GraphSAGEEdgeClassifier

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "graphsage_tuning"
CONFIGS = [
    {"hidden_channels": hidden, "learning_rate": rate}
    for hidden in (32, 64)
    for rate in (1e-3, 5e-4)
]
EPOCHS = 5
SEED = 42
MODEL_PATH = PROJECT_ROOT / "models" / "future_graphsage.pt"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
LOGREG_TEST_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "future_predictions"
    / "logreg_test_predictions.csv"
)


def verify_candidate_pair_alignment(graphsage: pd.DataFrame, logistic_regression: pd.DataFrame) -> None:
    """Require identical candidate-pair identifiers and labels for fair comparison."""
    columns = ["snapshot_time", "icao24_a", "icao24_b", "future_conflict"]
    missing = set(columns).difference(graphsage.columns) | set(columns).difference(
        logistic_regression.columns
    )
    if missing:
        raise ValueError(f"Prediction data is missing alignment columns: {sorted(missing)}.")

    def normalized(data: pd.DataFrame) -> pd.DataFrame:
        result = data[columns].copy()
        first = result[["icao24_a", "icao24_b"]].min(axis=1).str.lower()
        second = result[["icao24_a", "icao24_b"]].max(axis=1).str.lower()
        result["icao24_a"], result["icao24_b"] = first, second
        if result.duplicated(columns[:3]).any():
            raise ValueError("Prediction data contains duplicate candidate pairs.")
        return result.sort_values(columns[:3]).reset_index(drop=True)

    graph_pairs = normalized(graphsage)
    baseline_pairs = normalized(logistic_regression)
    if not graph_pairs.equals(baseline_pairs):
        raise ValueError(
            "GraphSAGE predictions do not match the logistic-regression candidate pairs and labels."
        )


def load_graphs(split: str) -> list:
    """Load sparse graphs with known future labels from one manifest split."""
    manifest = pd.read_csv(DEFAULT_MANIFEST_PATH)
    selected = manifest.loc[manifest["split"].eq(split)]
    if selected.empty:
        raise ValueError(f"No labeled sessions are assigned to {split}.")

    graphs = []
    label_columns = [
        "snapshot_time",
        "icao24_a",
        "icao24_b",
        "lateral_distance_nm",
        "future_conflict",
        "future_conflict_status",
    ]
    for row in selected.itertuples(index=False):
        state_name = row.session_file.replace("_future_labels.csv", ".csv")
        states = pd.read_csv(
            PROJECT_ROOT / "data" / "processed" / "cleaned_states" / state_name
        )
        state_groups = states.groupby("snapshot_time", sort=False)
        label_path = PROJECT_ROOT / row.label_file
        pending = pd.DataFrame()
        for chunk in pd.read_csv(label_path, usecols=label_columns, chunksize=500_000):
            chunk = chunk.loc[
                (chunk["lateral_distance_nm"] <= 15.0)
                & chunk["future_conflict_status"].isin(["positive", "negative"])
            ]
            combined = pd.concat([pending, chunk], ignore_index=True)
            if combined.empty:
                pending = combined
                continue
            last_time = combined["snapshot_time"].iloc[-1]
            complete = combined.loc[combined["snapshot_time"] != last_time]
            pending = combined.loc[combined["snapshot_time"] == last_time].copy()
            for timestamp, labels in complete.groupby("snapshot_time", sort=False):
                if timestamp in state_groups.groups:
                    graphs.append(
                        sparse_snapshot_to_graph(
                            state_groups.get_group(timestamp), labels
                        )
                    )
        if not pending.empty:
            timestamp, labels = pending["snapshot_time"].iloc[0], pending
            if timestamp in state_groups.groups:
                graphs.append(
                    sparse_snapshot_to_graph(state_groups.get_group(timestamp), labels)
                )
    if not graphs:
        raise ValueError(f"No labeled candidate graphs were found for {split}.")
    return graphs


def fit_and_score(
    train_graphs: list,
    validation_graphs: list,
    config: dict[str, float | int],
) -> tuple[float, int]:
    """Train one setting and return its best validation AP and epoch."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    scaler = FeatureScaler().fit(train_graphs)
    train = scaler.transform(train_graphs)
    validation = scaler.transform(validation_graphs)
    model = GraphSAGEEdgeClassifier(
        node_features=len(NODE_FEATURES),
        pair_features=train[0].pair_attr.size(1),
        hidden_channels=int(config["hidden_channels"]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    positives = sum(int(graph.edge_label.sum()) for graph in train)
    negatives = sum(graph.edge_label.numel() - int(graph.edge_label.sum()) for graph in train)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, dtype=torch.float)
    )

    best_ap, best_epoch = -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for index in torch.randperm(len(train)).tolist():
            graph = train[index]
            if graph.edge_label.numel() == 0:
                continue
            optimizer.zero_grad()
            loss = loss_fn(model(graph), graph.edge_label)
            loss.backward()
            optimizer.step()

        model.eval()
        labels, probabilities = [], []
        with torch.no_grad():
            for graph in validation:
                if graph.edge_label.numel():
                    labels.append(graph.edge_label.numpy())
                    probabilities.append(torch.sigmoid(model(graph)).numpy())
        ap = float(average_precision_score(np.concatenate(labels), np.concatenate(probabilities)))
        if ap > best_ap:
            best_ap, best_epoch = ap, epoch
    return best_ap, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Train the selected configuration and save its model and test predictions.",
    )
    args = parser.parse_args()
    if args.finalize:
        finalize_best()
        return

    print("Loading training and validation candidate graphs...")
    train_graphs = load_graphs("train")
    validation_graphs = load_graphs("validation")
    results = []
    for config in CONFIGS:
        ap, epoch = fit_and_score(train_graphs, validation_graphs, config)
        results.append({**config, "epochs": epoch, "validation_average_precision": ap})
        print(f"{config}: validation AP={ap:.5f} at epoch {epoch}")

    results.sort(key=lambda result: result["validation_average_precision"], reverse=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)
    (OUTPUT_DIR / "best_config.json").write_text(json.dumps(results[0], indent=2))
    print(f"Best validation-only configuration: {results[0]}")


def finalize_best() -> None:
    """Train the selected settings, keep the best validation epoch, and score test."""
    config_path = OUTPUT_DIR / "best_config.json"
    if not config_path.is_file():
        raise FileNotFoundError("Run tuning first to create best_config.json.")
    selected = json.loads(config_path.read_text())
    config = {
        "hidden_channels": int(selected["hidden_channels"]),
        "learning_rate": float(selected["learning_rate"]),
        "epochs": int(selected["epochs"]),
        "num_layers": 2,
        "dropout": 0.2,
        "aggr": "mean",
        "seed": SEED,
    }

    print("Loading train, validation and test graphs for final training...")
    train_graphs = load_graphs("train")
    validation_graphs = load_graphs("validation")
    test_graphs = load_graphs("test")
    scaler = FeatureScaler().fit(train_graphs)
    train = scaler.transform(train_graphs)
    validation = scaler.transform(validation_graphs)
    test = scaler.transform(test_graphs)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = GraphSAGEEdgeClassifier(
        node_features=len(NODE_FEATURES),
        pair_features=train[0].pair_attr.size(1),
        hidden_channels=config["hidden_channels"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        aggr=config["aggr"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    positives = sum(int(graph.edge_label.sum()) for graph in train)
    negatives = sum(graph.edge_label.numel() - int(graph.edge_label.sum()) for graph in train)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, dtype=torch.float)
    )
    best_ap, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        for index in torch.randperm(len(train)).tolist():
            graph = train[index]
            if not graph.edge_label.numel():
                continue
            optimizer.zero_grad()
            loss = loss_fn(model(graph), graph.edge_label)
            loss.backward()
            optimizer.step()

        model.eval()
        labels, probabilities = [], []
        with torch.no_grad():
            for graph in validation:
                if graph.edge_label.numel():
                    labels.append(graph.edge_label.numpy())
                    probabilities.append(torch.sigmoid(model(graph)).numpy())
        validation_ap = float(
            average_precision_score(np.concatenate(labels), np.concatenate(probabilities))
        )
        print(f"Final training epoch {epoch}: validation AP={validation_ap:.5f}")
        if validation_ap > best_ap:
            best_ap, best_epoch = validation_ap, epoch
            best_state = {key: value.clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    rows = []
    with torch.no_grad():
        for graph in test:
            probabilities = torch.sigmoid(model(graph)).numpy()
            if len(probabilities) != len(graph.pair_ids) or len(probabilities) != graph.edge_label.numel():
                raise ValueError("Test probabilities, labels and pair IDs do not align.")
            for (icao24_a, icao24_b), label, probability in zip(
                graph.pair_ids,
                graph.edge_label.numpy(),
                probabilities,
                strict=True,
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
    predictions = pd.DataFrame(rows)
    if not LOGREG_TEST_PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Generate the Phase 5 baseline test predictions first: "
            f"{LOGREG_TEST_PREDICTIONS_PATH}"
        )
    logistic_regression = pd.read_csv(LOGREG_TEST_PREDICTIONS_PATH)
    verify_candidate_pair_alignment(predictions, logistic_regression)
    print(f"Candidate-pair alignment verified: {len(predictions):,} identical test pairs.")
    predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)
    scaler_state = {
        "node_mean": scaler.node_mean.cpu(),
        "node_std": scaler.node_std.cpu(),
        "pair_mean": scaler.pair_mean.cpu(),
        "pair_std": scaler.pair_std.cpu(),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "best_epoch": best_epoch,
            "best_validation_average_precision": best_ap,
            "node_features": NODE_FEATURES,
            "pair_features": MODEL_FEATURES,
            "scaler": scaler_state,
        },
        MODEL_PATH,
    )
    config.update(
        {
            "best_epoch": best_epoch,
            "best_validation_average_precision": best_ap,
            "node_features": NODE_FEATURES,
            "pair_features": [
                "lateral_distance_nm",
                "vertical_separation_m",
                "closing_speed_mps",
                "bearing_difference_deg",
                "log_time_to_cpa",
                "has_cpa",
            ],
            "training_graphs": len(train),
            "validation_graphs": len(validation),
            "test_graphs": len(test),
            "test_pairs": len(predictions),
            "test_positive_pairs": int(predictions["future_conflict"].sum()),
            "test_predictions": str(TEST_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
            "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        }
    )
    (OUTPUT_DIR / "best_config.json").write_text(json.dumps(config, indent=2))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved test predictions: {TEST_PREDICTIONS_PATH} ({len(predictions):,} pairs)")


if __name__ == "__main__":
    main()
