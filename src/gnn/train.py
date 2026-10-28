"""Train and evaluate GraphSAGE on the shared multi-session split.

Raw state files are processed one frame at a time, keeping pair tables and
graphs out of memory. The saved split assignment is required; unknown frames
cause an error rather than being silently omitted.

Usage:
    python -m src.gnn.train
    python -m src.gnn.train --epochs 10
"""

from __future__ import annotations

import argparse
import copy
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from src.evaluation.harness import compute_classification_metrics
from src.evaluation.threshold import fit_threshold
from src.evaluation.splits import DEFAULT_OUTPUT_PATH
from src.features.build_pipeline import build_pair_features
from src.gnn.data import (
    NODE_FEATURES,
    FeatureScaler,
    pair_feature_names,
    pairs_to_graphs,
)
from src.gnn.model import GraphSAGEEdgeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_PATHS = (
    PROJECT_ROOT / "data" / "raw" / "sessions" / "states_socal_backfill_1790375809.csv",
    PROJECT_ROOT / "data" / "raw" / "sessions" / "states_socal_session_1791380397_06-39AM.csv",
    PROJECT_ROOT / "data" / "raw" / "sessions" / "states_socal_session_1791393504_10-18AM.csv",
    PROJECT_ROOT / "data" / "raw" / "sessions" / "states_socal_session_1791408650_02-30PM.csv",
    PROJECT_ROOT / "data" / "raw" / "sessions" / "states_socal_session_1791427160_07-39PM.csv",
)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "week5_graphsage_smote.pt"
DEFAULT_BASELINE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "week5_logistic_regression_shared_split.joblib"
)
DEFAULT_METRICS_PATH = (
    PROJECT_ROOT / "experiments" / "eda" / "step11_graphsage_smote_metrics.csv"
)
DEFAULT_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "eda"
    / "step11_graphsage_smote_test_predictions.csv"
)
RANDOM_SEED = 42
DEFAULT_EPOCHS = 5
DEFAULT_PATIENCE = 2
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_HIDDEN_CHANNELS = 32
SMOTE_NEGATIVE_MULTIPLIER = 10
SMOTE_TARGET_RATIO = 0.5
SMOTE_BATCH_SIZE = 256


def load_frame_assignments(assignment_path: Path) -> dict[int, str]:
    """Read the saved frame-to-part mapping and reject duplicate timestamps."""
    assignment = pd.read_csv(assignment_path, usecols=["snapshot_time", "part"])
    if assignment["snapshot_time"].duplicated().any():
        raise ValueError(f"{assignment_path} contains duplicate frame assignments.")
    if not assignment["part"].isin(["train", "validation", "test"]).all():
        raise ValueError(f"{assignment_path} contains an invalid split part.")
    return dict(
        zip(
            assignment["snapshot_time"].astype(int),
            assignment["part"],
            strict=True,
        )
    )


def iter_graphs(
    state_paths: tuple[Path, ...],
    assignments: dict[int, str],
    part: str,
    include_label_features: bool = True,
) -> Iterator[tuple[int, Data]]:
    """Yield one valid labelled frame graph at a time for a selected split."""
    if part not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split part: {part!r}.")
    for state_path in state_paths:
        if not state_path.is_file():
            raise FileNotFoundError(f"Required session file does not exist: {state_path}")
        states = pd.read_csv(state_path)
        if "snapshot_time" not in states:
            raise ValueError(f"{state_path} has no snapshot_time column.")
        states["snapshot_time"] = pd.to_numeric(states["snapshot_time"], errors="coerce")
        states = states.dropna(subset=["snapshot_time"])
        file_times = set(states["snapshot_time"].astype(int).unique())
        missing = file_times.difference(assignments)
        if missing:
            raise ValueError(
                f"{state_path.name} has {len(missing)} frame(s) missing from "
                f"the split assignment. Regenerate it with src.evaluation.splits."
            )

        selected_times = sorted(
            timestamp for timestamp in file_times if assignments[timestamp] == part
        )
        grouped = states.groupby("snapshot_time", sort=False)
        for timestamp in selected_times:
            frame_states = grouped.get_group(timestamp)
            pairs = build_pair_features(frame_states)
            graphs = pairs_to_graphs(
                pairs,
                include_label_features=include_label_features,
            )
            if graphs:
                yield timestamp, graphs[0]


def score_part(
    model: GraphSAGEEdgeClassifier,
    scaler: FeatureScaler,
    graphs: Iterator[tuple[int, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Score frame graphs and return labels, probabilities, times, and mean loss."""
    model.eval()
    all_labels = []
    all_probabilities = []
    all_times = []
    losses = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for snapshot_time, graph in graphs:
            scaled_graph = scaler.transform([graph])[0].to(device)
            logits = model(scaled_graph)
            labels = scaled_graph.edge_label
            probabilities = torch.sigmoid(logits)
            all_labels.append(labels.cpu().numpy().astype(np.int8))
            all_probabilities.append(probabilities.cpu().numpy())
            all_times.append(np.full(labels.numel(), snapshot_time, dtype=np.int64))
            losses.append(
                torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels
                ).item()
            )
    if not all_labels:
        raise ValueError("No labelled pair graphs were available for evaluation.")
    return (
        np.concatenate(all_labels),
        np.concatenate(all_probabilities),
        np.concatenate(all_times),
        float(np.mean(losses)),
    )


def collect_split(
    state_paths: tuple[Path, ...],
    assignments: dict[int, str],
    part: str,
    include_label_features: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect pair features for the small linear baseline, one frame at a time."""
    feature_chunks = []
    label_chunks = []
    time_chunks = []
    for snapshot_time, graph in iter_graphs(
        state_paths,
        assignments,
        part,
        include_label_features=include_label_features,
    ):
        feature_chunks.append(graph.pair_attr.numpy())
        label_chunks.append(graph.edge_label.numpy().astype(np.int8))
        time_chunks.append(
            np.full(graph.edge_label.numel(), snapshot_time, dtype=np.int64)
        )
    if not feature_chunks:
        raise ValueError(f"No labelled pair graphs were available for {part}.")
    return (
        np.concatenate(feature_chunks),
        np.concatenate(label_chunks),
        np.concatenate(time_chunks),
    )


def generate_smote_positives(
    positive_representations: np.ndarray,
    sampled_negatives: np.ndarray,
    target_ratio: float = SMOTE_TARGET_RATIO,
    random_state: int = RANDOM_SEED,
) -> np.ndarray:
    """SMOTE oversample positive learned pair representations only."""
    if positive_representations.ndim != 2 or sampled_negatives.ndim != 2:
        raise ValueError("SMOTE inputs must be two-dimensional feature arrays.")
    if positive_representations.shape[1] != sampled_negatives.shape[1]:
        raise ValueError("SMOTE inputs must have the same feature count.")
    if len(positive_representations) < 2:
        raise ValueError("SMOTE needs at least two positive pair representations.")
    if len(sampled_negatives) == 0:
        raise ValueError("SMOTE needs sampled negative pair representations.")

    features = np.concatenate([positive_representations, sampled_negatives])
    labels = np.concatenate(
        [
            np.ones(len(positive_representations), dtype=np.int8),
            np.zeros(len(sampled_negatives), dtype=np.int8),
        ]
    )
    desired_positives = int(np.ceil(target_ratio * len(sampled_negatives)))
    if desired_positives <= len(positive_representations):
        raise ValueError("SMOTE target ratio must request more positives than provided.")
    smote = SMOTE(
        sampling_strategy={1: desired_positives},
        k_neighbors=min(5, len(positive_representations) - 1),
        random_state=random_state,
    )
    resampled_features, resampled_labels = smote.fit_resample(features, labels)
    synthetic_start = len(features)
    synthetic = resampled_features[synthetic_start:][
        resampled_labels[synthetic_start:] == 1
    ]
    if len(synthetic) != desired_positives - len(positive_representations):
        raise RuntimeError("SMOTE did not generate the requested positive sample count.")
    return synthetic.astype(np.float32, copy=False)


def combine_smote_losses(
    original_loss: torch.Tensor,
    original_count: int,
    synthetic_loss: torch.Tensor,
    synthetic_count: int,
) -> torch.Tensor:
    """Weight real and synthetic edge losses by their represented row counts."""
    if original_count < 1 or synthetic_count < 1:
        raise ValueError("Original and synthetic loss batches must be non-empty.")
    return (
        original_loss * original_count + synthetic_loss * synthetic_count
    ) / (original_count + synthetic_count)


def extract_smote_training_examples(
    model: GraphSAGEEdgeClassifier,
    scaler: FeatureScaler,
    graphs: Iterator[tuple[int, Data]],
    negative_sample_limit: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect all positives and a uniform reservoir sample of negative embeddings."""
    if negative_sample_limit < 1:
        raise ValueError("negative_sample_limit must be positive.")
    model.eval()
    generator = np.random.default_rng(random_state)
    positive_chunks = []
    negative_reservoir: np.ndarray | None = None
    seen_negatives = 0
    device = next(model.parameters()).device

    with torch.no_grad():
        for _, graph in graphs:
            scaled_graph = scaler.transform([graph])[0].to(device)
            representations = model.pair_representations(scaled_graph).cpu().numpy()
            labels = graph.edge_label.cpu().numpy().astype(bool)
            if labels.any():
                positive_chunks.append(representations[labels].copy())
            negatives = representations[~labels]
            if not len(negatives):
                continue
            if negative_reservoir is None:
                negative_reservoir = np.empty(
                    (negative_sample_limit, representations.shape[1]),
                    dtype=np.float32,
                )
            if seen_negatives < negative_sample_limit:
                take = min(
                    negative_sample_limit - seen_negatives,
                    len(negatives),
                )
                negative_reservoir[seen_negatives:seen_negatives + take] = negatives[:take]
                negatives = negatives[take:]
                seen_negatives += take
            if len(negatives):
                candidate_indices = np.arange(
                    seen_negatives,
                    seen_negatives + len(negatives),
                    dtype=np.int64,
                )
                replacement_slots = generator.integers(
                    0,
                    candidate_indices + 1,
                )
                for local_index in np.flatnonzero(
                    replacement_slots < negative_sample_limit
                ):
                    negative_reservoir[replacement_slots[local_index]] = negatives[
                        local_index
                    ]
                seen_negatives += len(negatives)

    if not positive_chunks or negative_reservoir is None:
        raise ValueError("Training split must contain positive and negative pair examples.")
    positive = np.concatenate(positive_chunks).astype(np.float32, copy=False)
    sampled_negatives = negative_reservoir[:min(negative_sample_limit, seen_negatives)]
    if len(sampled_negatives) <= len(positive):
        raise ValueError("Too few negative embeddings to generate SMOTE examples.")
    return positive, sampled_negatives.copy()


def train(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit GraphSAGE, tune its threshold on validation, and evaluate test once."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_paths = tuple(args.sessions)
    assignments = load_frame_assignments(args.assignment_path)

    class_counts = np.zeros(2, dtype=np.int64)
    training_graph_count = 0

    def scaler_training_graphs() -> Iterator[Data]:
        nonlocal training_graph_count
        for _, graph in iter_graphs(
            state_paths,
            assignments,
            "train",
            include_label_features=not args.exclude_label_features,
        ):
            class_counts[:] += np.bincount(
                graph.edge_label.numpy().astype(int), minlength=2
            )
            training_graph_count += 1
            yield graph

    scaler = FeatureScaler().fit(scaler_training_graphs())
    if not class_counts[0] or not class_counts[1]:
        raise ValueError("Training split must contain both positive and negative labels.")

    model = GraphSAGEEdgeClassifier(
        node_features=len(NODE_FEATURES),
        pair_features=len(pair_feature_names(not args.exclude_label_features)),
        hidden_channels=args.hidden_channels,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = torch.nn.BCEWithLogitsLoss()

    best_ap = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        positive_representations, sampled_negatives = extract_smote_training_examples(
            model,
            scaler,
            iter_graphs(
                state_paths,
                assignments,
                "train",
                include_label_features=not args.exclude_label_features,
            ),
            negative_sample_limit=SMOTE_NEGATIVE_MULTIPLIER * int(class_counts[1]),
            random_state=RANDOM_SEED + epoch,
        )
        synthetic_positives = generate_smote_positives(
            positive_representations,
            sampled_negatives,
            random_state=RANDOM_SEED + epoch,
        )
        synthetic_tensor = torch.from_numpy(synthetic_positives).to(device)
        synthetic_order = torch.randperm(len(synthetic_tensor), device=device)
        synthetic_offset = 0

        model.train()
        losses = []
        graphs_seen = 0
        for _, graph in iter_graphs(
            state_paths,
            assignments,
            "train",
            include_label_features=not args.exclude_label_features,
        ):
            graph = scaler.transform([graph])[0].to(device)
            optimizer.zero_grad()
            original_loss = loss_function(model(graph), graph.edge_label)
            if synthetic_offset + SMOTE_BATCH_SIZE > len(synthetic_order):
                synthetic_order = torch.randperm(len(synthetic_tensor), device=device)
                synthetic_offset = 0
            synthetic_indices = synthetic_order[
                synthetic_offset:synthetic_offset + SMOTE_BATCH_SIZE
            ]
            synthetic_offset += SMOTE_BATCH_SIZE
            synthetic_logits = model.classify_pair_representations(
                synthetic_tensor[synthetic_indices]
            )
            synthetic_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                synthetic_logits, torch.ones_like(synthetic_logits)
            )
            loss = combine_smote_losses(
                original_loss,
                graph.edge_label.numel(),
                synthetic_loss,
                synthetic_logits.numel(),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(loss.item())
            graphs_seen += 1
        if graphs_seen == 0:
            raise ValueError("No training graphs were generated from the assigned frames.")

        validation = score_part(
            model,
            scaler,
            iter_graphs(
                state_paths,
                assignments,
                "validation",
                include_label_features=not args.exclude_label_features,
            ),
        )
        validation_labels, validation_probabilities, _, validation_loss = validation
        validation_ap = float(
            average_precision_score(validation_labels, validation_probabilities)
        )
        epoch_row = {
            "epoch": epoch,
            "training_graphs": graphs_seen,
            "training_pairs": int(class_counts.sum()),
            "training_positive_pairs": int(class_counts[1]),
            "synthetic_positive_pairs": len(synthetic_positives),
            "effective_training_positive_ratio_pct": 100 * (
                class_counts[1] + len(synthetic_positives)
            ) / (class_counts.sum() + len(synthetic_positives)),
            "training_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
            "validation_average_precision": validation_ap,
        }
        history.append(epoch_row)
        print(
            f"Epoch {epoch}/{args.epochs}: train loss={epoch_row['training_loss']:.4f}, "
            f"validation loss={validation_loss:.4f}, validation AP={validation_ap:.4f}"
        )

        if validation_ap > best_ap:
            best_ap = validation_ap
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a best validation checkpoint.")
    model.load_state_dict(best_state)

    validation_labels, validation_probabilities, _, _ = score_part(
        model,
        scaler,
        iter_graphs(
            state_paths,
            assignments,
            "validation",
            include_label_features=not args.exclude_label_features,
        ),
    )
    threshold = fit_threshold(validation_labels, validation_probabilities)
    test_labels, test_probabilities, test_times, test_loss = score_part(
        model,
        scaler,
        iter_graphs(
            state_paths,
            assignments,
            "test",
            include_label_features=not args.exclude_label_features,
        ),
    )

    records = []
    for split_name, labels, probabilities in (
        ("validation", validation_labels, validation_probabilities),
        ("test", test_labels, test_probabilities),
    ):
        for threshold_name, threshold_value in (
            ("fixed_0.5", 0.5),
            ("validation_tuned", threshold),
        ):
            metrics = compute_classification_metrics(
                labels,
                probabilities,
                threshold=threshold_value,
            )
            records.append(
                {
                    "model": "GraphSAGE_SMOTE",
                    "split": split_name,
                    "threshold_selection": threshold_name,
                    "threshold": threshold_value,
                    "rows": len(labels),
                    "positive_rows": int(labels.sum()),
                    "prevalence_pct": 100 * float(labels.mean()),
                    "average_precision": metrics["average_precision"],
                    "auroc": metrics["auroc"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "normalised_cost": metrics["normalised_cost"],
                    "false_positives": metrics["false_positives"],
                    "false_negatives": metrics["false_negatives"],
                    "validation_best_average_precision": best_ap,
                    "test_mean_frame_loss": test_loss if split_name == "test" else np.nan,
                    "imbalance_strategy": "latent_representation_smote",
                    "smote_target_ratio_on_negative_reservoir": SMOTE_TARGET_RATIO,
                    "smote_negative_sample_multiplier": SMOTE_NEGATIVE_MULTIPLIER,
                }
            )

    baseline_train_x, baseline_train_y, _ = collect_split(
        state_paths,
        assignments,
        "train",
        include_label_features=not args.exclude_label_features,
    )
    baseline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5_000, random_state=RANDOM_SEED)),
        ]
    )
    baseline.fit(baseline_train_x, baseline_train_y)
    del baseline_train_x, baseline_train_y

    baseline_validation_x, baseline_validation_y, _ = collect_split(
        state_paths,
        assignments,
        "validation",
        include_label_features=not args.exclude_label_features,
    )
    baseline_validation_probabilities = baseline.predict_proba(baseline_validation_x)[:, 1]
    baseline_threshold = fit_threshold(
        baseline_validation_y, baseline_validation_probabilities
    )
    baseline_test_x, baseline_test_y, baseline_test_times = collect_split(
        state_paths,
        assignments,
        "test",
        include_label_features=not args.exclude_label_features,
    )
    baseline_test_probabilities = baseline.predict_proba(baseline_test_x)[:, 1]
    del baseline_test_x

    for split_name, labels, probabilities in (
        ("validation", baseline_validation_y, baseline_validation_probabilities),
        ("test", baseline_test_y, baseline_test_probabilities),
    ):
        for threshold_name, threshold_value in (
            ("fixed_0.5", 0.5),
            ("validation_tuned", baseline_threshold),
        ):
            metrics = compute_classification_metrics(
                labels,
                probabilities,
                threshold=threshold_value,
            )
            records.append(
                {
                    "model": "LogisticRegression",
                    "split": split_name,
                    "threshold_selection": threshold_name,
                    "threshold": threshold_value,
                    "rows": len(labels),
                    "positive_rows": int(labels.sum()),
                    "prevalence_pct": 100 * float(labels.mean()),
                    "average_precision": metrics["average_precision"],
                    "auroc": metrics["auroc"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "normalised_cost": metrics["normalised_cost"],
                    "false_positives": metrics["false_positives"],
                    "false_negatives": metrics["false_negatives"],
                    "validation_best_average_precision": float(
                        average_precision_score(
                            baseline_validation_y,
                            baseline_validation_probabilities,
                        )
                    ),
                    "test_mean_frame_loss": np.nan,
                }
            )

    prediction_table = pd.DataFrame(
        {
            "snapshot_time": test_times,
            "collision_risk": test_labels,
            "risk_probability": test_probabilities,
            "smote_graphsage_prediction_at_0_5": (
                test_probabilities >= 0.5
            ).astype(np.int8),
            "smote_graphsage_prediction_at_validation_threshold": (
                test_probabilities >= threshold
            ).astype(np.int8),
        }
    )
    if not np.array_equal(test_times, baseline_test_times) or not np.array_equal(
        test_labels, baseline_test_y
    ):
        raise ValueError("GraphSAGE and linear baseline test rows do not align.")
    prediction_table["baseline_risk_probability"] = baseline_test_probabilities
    prediction_table["baseline_prediction_at_0_5"] = (
        baseline_test_probabilities >= 0.5
    ).astype(np.int8)
    prediction_table["baseline_prediction_at_validation_threshold"] = (
        baseline_test_probabilities >= baseline_threshold
    ).astype(np.int8)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "node_features": NODE_FEATURES,
            "pair_features": pair_feature_names(not args.exclude_label_features),
            "node_mean": scaler.node_mean,
            "node_std": scaler.node_std,
            "pair_mean": scaler.pair_mean,
            "pair_std": scaler.pair_std,
            "validation_best_average_precision": best_ap,
            "validation_threshold": threshold,
            "seed": RANDOM_SEED,
            "hidden_channels": args.hidden_channels,
            "excluded_label_defining_features": args.exclude_label_features,
            "imbalance_strategy": "latent_representation_smote",
            "smote_target_ratio_on_negative_reservoir": SMOTE_TARGET_RATIO,
            "smote_negative_sample_multiplier": SMOTE_NEGATIVE_MULTIPLIER,
        },
        args.model_path,
    )
    args.baseline_model_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(baseline, args.baseline_model_path)
    metrics_table = pd.DataFrame(records)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(args.metrics_path, index=False)
    args.predictions_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(args.predictions_path, index=False)
    pd.DataFrame(history).to_csv(
        args.metrics_path.with_name("step11_graphsage_smote_training.csv"),
        index=False,
    )
    print(f"Training graphs: {training_graph_count:,}; training labels: {class_counts.tolist()}")
    print(f"Validation-selected threshold: {threshold:.6f}")
    print(f"Saved model: {args.model_path}")
    print(f"Saved matched LogisticRegression baseline: {args.baseline_model_path}")
    print(f"Saved metrics: {args.metrics_path}")
    print(f"Saved test predictions: {args.predictions_path}")
    return metrics_table, prediction_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions",
        nargs="+",
        type=Path,
        default=list(DEFAULT_STATE_PATHS),
        help="Raw session CSVs present in the shared saved split.",
    )
    parser.add_argument("--assignment-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--baseline-model-path", type=Path, default=DEFAULT_BASELINE_MODEL_PATH
    )
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument(
        "--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--hidden-channels", type=int, default=DEFAULT_HIDDEN_CHANNELS)
    parser.add_argument(
        "--exclude-label-features",
        action="store_true",
        help="Train an ablation without lateral/vertical label-defining features.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be positive.")
    if args.learning_rate <= 0 or args.hidden_channels < 1:
        parser.error("--learning-rate and --hidden-channels must be positive.")
    train(args)


if __name__ == "__main__":
    main()
