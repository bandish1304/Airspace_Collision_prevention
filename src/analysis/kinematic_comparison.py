"""Step 7: compare kinematic features between risky and safe aircraft pairs.

Answers the progress feedback's request to "compare lateral distance, vertical
separation, closing speed, bearing difference, time to closest point of
approach, and other important features between safe and risky cases".

Three groups rather than two. Comparing risky pairs with every safe pair is
too easy: most safe pairs are 20-50 nm apart, so almost any feature differs.
The near-miss group -- inside 10 nm and 2,000 ft, twice both limits, without
breaching -- is matched on proximity and turns out to be the same kind of
traffic, so differences against it are about how the aircraft move rather
than which aircraft they are.

Lateral and vertical separation are listed but are not findings: the label is
defined from them, so risky pairs sit below the limits by construction.

Each comparison is made twice: over rows, and over aircraft pairs (one median
per pair). Step 3 showed a single encounter can contribute a hundred rows, so
the row view weights long encounters heavily; the pair view counts each once.

Usage:
    python -m src.analysis.kinematic_comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.baselines.data import DEFAULT_PAIRS_PATH
from src.features.model_inputs import prepare_model_inputs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "eda"

METRES_PER_FOOT = 0.3048
METRES_PER_SECOND_PER_KNOT = 0.514444

# Twice each separation limit: close enough to have nearly breached on both
# counts, so the comparison is between pairs in similar situations.
NEAR_MISS_LATERAL_NM = 10.0
NEAR_MISS_VERTICAL_FT = 2_000.0

GROUP_ORDER = ["risky", "near-miss", "other safe"]

# column -> (label, defines the risk label)
FEATURES = {
    "lateral_distance_nm": ("Lateral separation (nm)", True),
    "vertical_separation_ft": ("Vertical separation (ft)", True),
    "closing_speed_mps": ("Closing speed (m/s)", False),
    "bearing_difference_deg": ("Bearing difference (deg)", False),
    "time_to_cpa_min": ("Time to closest approach (min)", False),
    "converging": ("Converging", False),
    "altitude_ft": ("Mean altitude (ft)", False),
    "speed_kt": ("Mean ground speed (kt)", False),
}

PAIR_COLUMNS = [
    "snapshot_time", "icao24_a", "icao24_b",
    "lateral_distance_nm", "vertical_separation_m", "closing_speed_mps",
    "bearing_difference_deg", "time_to_cpa_s",
    "altitude_a_m", "altitude_b_m", "velocity_a_mps", "velocity_b_mps",
]


# Step 7: load the pair table and tag every row with its comparison group.
def load_groups(pairs_path: Path = DEFAULT_PAIRS_PATH) -> pd.DataFrame:
    pairs = prepare_model_inputs(pd.read_csv(pairs_path, usecols=PAIR_COLUMNS))

    frame = pd.DataFrame(
        {
            "snapshot_time": pairs["snapshot_time"],
            "aircraft_a": np.minimum(pairs["icao24_a"], pairs["icao24_b"]),
            "aircraft_b": np.maximum(pairs["icao24_a"], pairs["icao24_b"]),
            "lateral_distance_nm": pairs["lateral_distance_nm"],
            "vertical_separation_ft": pairs["vertical_separation_m"] / METRES_PER_FOOT,
            "closing_speed_mps": pairs["closing_speed_mps"],
            "bearing_difference_deg": pairs["bearing_difference_deg"],
            # Only converging pairs have a closest approach ahead of them; the
            # rest are kept but carry no time, and are counted in "converging".
            "time_to_cpa_min": pairs["time_to_cpa_s"].where(pairs["has_cpa"] == 1) / 60,
            "converging": pairs["has_cpa"],
            "altitude_ft": (pairs["altitude_a_m"] + pairs["altitude_b_m"]) / 2 / METRES_PER_FOOT,
            "speed_kt": (pairs["velocity_a_mps"] + pairs["velocity_b_mps"]) / 2
            / METRES_PER_SECOND_PER_KNOT,
            "log_time_to_cpa": pairs["log_time_to_cpa"],
        }
    )

    risky = pairs["collision_risk"] == 1
    near_miss = (
        ~risky
        & (frame["lateral_distance_nm"] <= NEAR_MISS_LATERAL_NM)
        & (frame["vertical_separation_ft"] <= NEAR_MISS_VERTICAL_FT)
    )
    frame["group"] = np.select([risky, near_miss], GROUP_ORDER[:2], default=GROUP_ORDER[2])
    return frame.reset_index(drop=True)


# Step 7: one median per aircraft pair and group, so every pair counts once.
def per_pair(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["group", "aircraft_a", "aircraft_b"])[list(FEATURES)]
        .median()
        .reset_index()
    )


# Step 7: medians, quartiles and counts per group and feature.
def summarise(frame: pd.DataFrame, unit: str) -> pd.DataFrame:
    rows = []
    for group in GROUP_ORDER:
        members = frame.loc[frame["group"] == group]
        for feature, (label, defines_label) in FEATURES.items():
            values = members[feature].dropna()
            rows.append(
                {
                    "unit": unit,
                    "group": group,
                    "feature": feature,
                    "label_defining": defines_label,
                    "count": len(values),
                    "median": values.median(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                    "mean": values.mean(),
                }
            )
    return pd.DataFrame(rows)


# Step 7: how well each feature alone separates risky pairs from the others.
def separability(frame: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Single-feature AUROC: 0.5 means indistinguishable, 1 or 0 means fully separated.

    Above 0.5 means risky pairs tend to have the larger value. Time to closest
    approach uses the capped log form, so pairs with no closest approach rank
    as the longest rather than being dropped.
    """
    scored = {**FEATURES}
    scored.pop("time_to_cpa_min")
    columns = ["log_time_to_cpa", *scored]

    rows = []
    risky = frame.loc[frame["group"] == "risky"]
    for comparison in GROUP_ORDER[1:]:
        other = frame.loc[frame["group"] == comparison]
        both = pd.concat([risky, other])
        is_risky = (both["group"] == "risky").to_numpy()
        for feature in columns:
            values = both[feature].to_numpy(dtype=float)
            keep = np.isfinite(values)
            rows.append(
                {
                    "unit": unit,
                    "comparison": f"risky vs {comparison}",
                    "feature": "time_to_cpa_min" if feature == "log_time_to_cpa" else feature,
                    "label_defining": FEATURES.get(feature, ("", False))[1],
                    "auroc": roc_auc_score(is_risky[keep], values[keep]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = load_groups(args.pairs_path)
    pairs = per_pair(rows)
    pairs["log_time_to_cpa"] = np.log1p(
        np.minimum(pairs["time_to_cpa_min"].fillna(60.0) * 60, 3_600.0)
    )

    summary = pd.concat([summarise(rows, "row"), summarise(pairs, "pair")])
    separation = pd.concat([separability(rows, "row"), separability(pairs, "pair")])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "step7_kinematic_summary.csv", index=False)
    separation.to_csv(args.output_dir / "step7_feature_separability.csv", index=False)

    print("Group sizes (rows / aircraft pairs):")
    for group in GROUP_ORDER:
        print(f"  {group:<11} {int((rows['group'] == group).sum()):>10,} / "
              f"{int((pairs['group'] == group).sum()):>7,}")

    print("\nSingle-feature AUROC, risky vs near-miss (0.5 = indistinguishable):")
    view = separation.loc[separation["comparison"] == "risky vs near-miss"].pivot(
        index="feature", columns="unit", values="auroc"
    )
    print(view.reindex(list(FEATURES)).round(3).to_string())
    print(f"\nWrote {args.output_dir.relative_to(PROJECT_ROOT)}"
          "/step7_kinematic_summary.csv and step7_feature_separability.csv")


if __name__ == "__main__":
    main()
