"""Compare the current proximity label against convergence/CPA candidates.

This is decision-support analysis only. It writes counts for alternate label
definitions and never replaces the project's current collision_risk label.

Usage:
    python -m src.analysis.labeling_sensitivity data/processed/<file>_pairs.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.encounters import (
    DEFAULT_LATERAL_SEPARATION_NM,
    DEFAULT_VERTICAL_SEPARATION_FT,
    METRES_PER_FOOT,
    READ_CHUNK_ROWS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "eda"
CPA_HORIZONS_SECONDS = (300, 600, 900, 1_800)


def label_rule_comparison(
    pairs_path: Path,
    lateral_separation_nm: float = DEFAULT_LATERAL_SEPARATION_NM,
    vertical_separation_ft: float = DEFAULT_VERTICAL_SEPARATION_FT,
) -> pd.DataFrame:
    """Return positive counts/rates for the current and candidate label rules."""
    vertical_limit_m = vertical_separation_ft * METRES_PER_FOOT
    total_pairs = 0
    positive_counts = {
        "current separation rule": 0,
        "current rule + converging": 0,
        **{f"converging + CPA <= {seconds // 60} min": 0 for seconds in CPA_HORIZONS_SECONDS},
    }
    columns = [
        "lateral_distance_nm",
        "vertical_separation_m",
        "closing_speed_mps",
        "time_to_cpa_s",
    ]

    for chunk in pd.read_csv(pairs_path, usecols=columns, chunksize=READ_CHUNK_ROWS):
        total_pairs += len(chunk)
        current = (chunk["lateral_distance_nm"] <= lateral_separation_nm) & (
            chunk["vertical_separation_m"] <= vertical_limit_m
        )
        converging = current & (chunk["closing_speed_mps"] > 0)
        positive_counts["current separation rule"] += int(current.sum())
        positive_counts["current rule + converging"] += int(converging.sum())
        for seconds in CPA_HORIZONS_SECONDS:
            name = f"converging + CPA <= {seconds // 60} min"
            positive_counts[name] += int(
                (converging & (chunk["time_to_cpa_s"] <= seconds)).sum()
            )

    comparison = pd.DataFrame(
        {
            "rule": list(positive_counts),
            "positive_pair_observations": list(positive_counts.values()),
        }
    )
    comparison["positive_rate_pct"] = (
        comparison["positive_pair_observations"] / total_pairs * 100
    )
    comparison["removed_vs_current"] = (
        positive_counts["current separation rule"]
        - comparison["positive_pair_observations"]
    )
    comparison["retained_vs_current_pct"] = (
        comparison["positive_pair_observations"]
        / positive_counts["current separation rule"]
        * 100
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs_path", type=Path, help="Path to a pair-feature CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    comparison = label_rule_comparison(args.pairs_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "step5_label_rule_comparison.csv"
    comparison.to_csv(output_path, index=False)
    print(comparison.to_string(index=False))
    print(f"\nWrote {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()