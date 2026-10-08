"""Figure for the Step 5 label-rule sensitivity comparison.

Usage:
    python -m src.analysis.labeling_sensitivity_figures \
        experiments/eda/step5_label_rule_comparison.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "experiments" / "figures"

INK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"
AXIS = "#cbd5e1"
CURRENT = "#3d74b8"
CANDIDATE = "#b45309"
SURFACE = "#ffffff"


def plot_label_rule_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    """Show how much of the current positive class each candidate retains."""
    figure, axes = plt.subplots(figsize=(11, 5.6))
    figure.patch.set_facecolor(SURFACE)
    positions = range(len(comparison))
    colors = [CURRENT] + [CANDIDATE] * (len(comparison) - 1)
    axes.bar(positions, comparison["positive_pair_observations"], color=colors, width=0.66)
    axes.set_xticks(positions, comparison["rule"], rotation=17, ha="right", fontsize=9.5)
    axes.set_ylabel("Rule-labeled positive pair-observations", fontsize=10, color=INK)
    axes.set_title(
        "Convergence removes nearly half of proximity-only positives; CPA horizon changes little after that",
        fontsize=12, color=INK, loc="left", fontweight="bold",
    )
    for position, row in enumerate(comparison.itertuples()):
        axes.annotate(
            f"{row.positive_pair_observations:,}\n({row.retained_vs_current_pct:.1f}%)",
            (position, row.positive_pair_observations), xytext=(0, 6),
            textcoords="offset points", ha="center", fontsize=9, color=INK,
            fontweight="bold",
        )
    axes.grid(axis="y", color=GRID, linewidth=0.7)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
    axes.tick_params(colors=MUTED)
    axes.set_facecolor(SURFACE)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_path", type=Path)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()

    comparison = pd.read_csv(args.comparison_path)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.figure_dir / "step5_label_rule_comparison.png"
    plot_label_rule_comparison(comparison, output_path)
    print(f"Wrote {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()