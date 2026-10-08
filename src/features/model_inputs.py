"""Model-ready pair features, shared by every model in the project.

The GNN and the logistic-regression baselines both start here, so they train
on the same rows with the same features and the same labels. Before this they
each had their own loader, and the two disagreed: the baselines deleted every
pair whose closest approach lies in the past, which removed about half the
positive class, while the GNN kept them.

Preparation does three things:

    label        the Week 2 separation rule, applied on load
    time to CPA  capped at an hour and log-scaled, with a flag for pairs that
                 never reach a closest approach, instead of being dropped
    filter       rows that cannot be labelled, or carry a missing value, go
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.labeling.label_pairs import label_collision_risk

# Beyond an hour, time to closest approach says nothing useful about risk;
# 99% of finite values in the backfill fall under 3,514 s.
TIME_TO_CPA_CAP_SECONDS = 3_600.0

MODEL_FEATURES = [
    "lateral_distance_nm",
    "vertical_separation_m",
    "closing_speed_mps",
    "bearing_difference_deg",
    "log_time_to_cpa",
    "has_cpa",
]

# The label is computed from these two, so a model given them can learn the
# rule back instead of anything about risk. Kept by default to match the plan;
# whether to drop them is the open labelling question.
LABEL_DEFINING_FEATURES = ("lateral_distance_nm", "vertical_separation_m")

# Raw columns a pair table must provide for the features above.
REQUIRED_PAIR_COLUMNS = [
    "snapshot_time",
    "icao24_a",
    "icao24_b",
    "lateral_distance_nm",
    "vertical_separation_m",
    "closing_speed_mps",
    "bearing_difference_deg",
    "time_to_cpa_s",
]

# Per-aircraft columns the GNN also reads. Checked for missing values when
# present, so a pair kept by one model is never dropped by another.
ENDPOINT_COLUMNS = [
    "latitude_a", "longitude_a", "altitude_a_m", "velocity_a_mps", "true_track_a_deg",
    "latitude_b", "longitude_b", "altitude_b_m", "velocity_b_mps", "true_track_b_deg",
]


# Step 6: name the features a model will be given.
def feature_names(include_label_features: bool = True) -> list[str]:
    if include_label_features:
        return list(MODEL_FEATURES)
    return [name for name in MODEL_FEATURES if name not in LABEL_DEFINING_FEATURES]


# Step 6: label pairs and turn raw kinematics into model-ready features.
def prepare_model_inputs(
    pairs: pd.DataFrame,
    lateral_separation_nm: float = 5.0,
    vertical_separation_ft: float = 1_000.0,
) -> pd.DataFrame:
    """Return labelled pairs with model features, dropping unusable rows."""
    labelled = label_collision_risk(
        pairs,
        lateral_separation_nm=lateral_separation_nm,
        vertical_separation_ft=vertical_separation_ft,
    )
    labelled = labelled.loc[labelled["collision_risk"].notna()].copy()
    labelled["collision_risk"] = labelled["collision_risk"].astype(int)

    time_to_cpa = labelled["time_to_cpa_s"].to_numpy(dtype=float)
    labelled["has_cpa"] = np.isfinite(time_to_cpa).astype(float)
    capped = np.minimum(
        np.nan_to_num(time_to_cpa, posinf=TIME_TO_CPA_CAP_SECONDS),
        TIME_TO_CPA_CAP_SECONDS,
    )
    # Most closest approaches arrive within minutes and a few take an hour, so
    # on the raw scale the long tail would dominate.
    labelled["log_time_to_cpa"] = np.log1p(capped)

    checked = MODEL_FEATURES + [c for c in ENDPOINT_COLUMNS if c in labelled.columns]
    finite = np.isfinite(labelled[checked].to_numpy(dtype=float)).all(axis=1)
    return labelled.loc[finite]
