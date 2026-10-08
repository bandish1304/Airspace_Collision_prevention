"""Load the backfill pair table for the logistic-regression baselines.

Every baseline reads its data through here. The four Week 3 scripts used to
each carry a copy of their own loader, pointed at the single Week 1 snapshot,
and each deleted the pairs that never reach a closest approach. They now share
the preparation in src.features.model_inputs with the GNN, so all five models
train on the same rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation.splits import (
    DEFAULT_TRAIN_FRACTION,
    PARTS,
    load_split,
)
from src.features.model_inputs import feature_names, prepare_model_inputs

SPLIT_METHODS = ("temporal", "random")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "states_socal_backfill_1790375809_pairs.csv"
)

# Only what the features and the split need. Aircraft identifiers are left out:
# at four million rows the two string columns alone would cost more memory
# than everything else combined.
BASELINE_COLUMNS = [
    "snapshot_time",
    "lateral_distance_nm",
    "vertical_separation_m",
    "closing_speed_mps",
    "bearing_difference_deg",
    "time_to_cpa_s",
]


# Step 6: features, labels and frame times for the baseline models.
def load_dataset(
    pairs_path: Path = DEFAULT_PAIRS_PATH,
    include_label_features: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return the feature matrix, the labels, and each row's snapshot time.

    The snapshot time is what the temporal split needs to put each row in
    train, validation or test.
    """
    pairs = pd.read_csv(pairs_path, usecols=BASELINE_COLUMNS)
    prepared = prepare_model_inputs(pairs).reset_index(drop=True)
    features = prepared[feature_names(include_label_features)]
    return features, prepared["collision_risk"], prepared["snapshot_time"]


# Step 6: cut the data into train, validation and test.
def split_dataset(
    features: pd.DataFrame,
    labels: pd.Series,
    snapshot_time: pd.Series,
    method: str = "temporal",
    random_state: int = 42,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Return ``{part: (features, labels)}`` for train, validation and test.

    ``temporal`` is the project split. ``random`` shuffles rows into the same
    60/20/20 proportions, stratified on the label, which is how Week 3 split;
    it is kept only to measure how much that flatters the scores.
    """
    if method == "temporal":
        part = load_split(snapshot_time)
    elif method == "random":
        part = _random_parts(labels, random_state)
    else:
        raise ValueError(f"method must be one of {SPLIT_METHODS}, not {method!r}.")
    return {name: (features[part == name], labels[part == name]) for name in PARTS}


def _random_parts(labels: pd.Series, random_state: int) -> pd.Series:
    rows = np.arange(len(labels))
    train, rest = train_test_split(
        rows, train_size=DEFAULT_TRAIN_FRACTION, random_state=random_state, stratify=labels
    )
    # The remaining 40% halves into validation and test.
    validation, test = train_test_split(
        rest, test_size=0.5, random_state=random_state, stratify=labels.iloc[rest]
    )
    part = np.empty(len(labels), dtype=object)
    part[train], part[validation], part[test] = PARTS
    return pd.Series(part, index=labels.index)
