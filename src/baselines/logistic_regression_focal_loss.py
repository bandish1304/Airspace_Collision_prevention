"""Train the Week 3 logistic regression baseline with focal loss."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.baselines.data import DEFAULT_PAIRS_PATH, load_dataset, split_dataset
from src.evaluation.harness import build_evaluation_harness

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = DEFAULT_PAIRS_PATH
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "week3_logistic_regression_focal_loss.pt"

# This function defines the focal loss used for the rare positive class. The model
# is asked to pay more attention to hard, minority-class examples instead of simply
# optimizing ordinary cross-entropy on the majority class.
def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction="none"
    )
    p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    modulating_factor = (1.0 - p_t).pow(gamma)
    return (alpha_factor * modulating_factor * ce_loss).mean()


# This function trains a single logistic model in PyTorch with the focal loss
# objective. This matches the Week 3 requirement to compare a focal-loss version
# against the plain and SMOTE baselines using the same evaluation harness.
def train_focal_loss_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    random_state: int = 42,
    learning_rate: float = 0.01,
    epochs: int = 400,
) -> tuple[torch.nn.Module, dict]:
    torch.manual_seed(random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    model = torch.nn.Sequential(
        torch.nn.Linear(X_train_scaled.shape[1], 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_eval_t = torch.tensor(X_eval_scaled, dtype=torch.float32)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = focal_loss(logits, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        eval_logits = model(X_eval_t)
        y_prob = torch.sigmoid(eval_logits).numpy().ravel()

    metrics = build_evaluation_harness(y_eval, y_prob, threshold=0.5)
    return model, metrics


# This function saves the trained model so it can be compared against the plain and
# SMOTE baselines and later logged to MLflow with the Week 3 experiment results.
def save_model(model: torch.nn.Module, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the labeled pair dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained focal-loss model.",
    )
    args = parser.parse_args()

    features, labels, times = load_dataset(args.data_path)
    parts = split_dataset(features, labels, times)
    (X_train, y_train), (X_eval, y_eval) = parts["train"], parts["validation"]
    model, metrics = train_focal_loss_model(
        X_train.to_numpy(), y_train.to_numpy(), X_eval.to_numpy(), y_eval.to_numpy()
    )
    save_model(model, args.output_path)

    print(f"Dataset rows: {len(features)}")
    print(f"Positive rate: {float(labels.mean()):.4f}")
    print(f"Model saved to: {args.output_path}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"False negative rate: {metrics['false_negative_rate']:.4f}")


if __name__ == "__main__":
    main()
