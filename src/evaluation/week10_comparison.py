"""Comparison table and figure for the GNN vs TFT test results (Week 10, step 4).

Reads the saved shared-test predictions, harness metrics and significance
results; nothing is retrained or rescored.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.harness import precision_recall_curve_data
from src.evaluation.week10_metrics import METRICS_PATH
from src.evaluation.week10_significance import SIGNIFICANCE_PATH
from src.evaluation.week10_test_set import ALIGNED_PREDICTIONS, PROJECT_ROOT

TABLE_CSV = PROJECT_ROOT / "experiments" / "eda" / "week10_comparison_table.csv"
TABLE_MARKDOWN = PROJECT_ROOT / "experiments" / "eda" / "week10_comparison_table.md"
FIGURE_PATH = PROJECT_ROOT / "experiments" / "figures" / "week10_gnn_vs_tft.png"

INK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"
AXIS = "#cbd5e1"
GNN_COLOR = "#b45309"
TFT_COLOR = "#3d74b8"
SURFACE = "#ffffff"
LABELS = {
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "false_negative_rate": "False negative rate",
    "auroc": "AUROC",
    "average_precision": "Average precision",
}
DISPLAY_ORDER = [
    "precision",
    "recall",
    "f1",
    "auroc",
    "average_precision",
    "false_negative_rate",
]


def _style(axes: plt.Axes) -> None:
    axes.grid(True, color=GRID, linewidth=0.7)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
    axes.tick_params(labelsize=9, colors=MUTED)
    axes.set_facecolor(SURFACE)


def format_p_value(value: float) -> str:
    """Bootstrap p-values are floored by the resample count, so show a bound."""
    return f"\u2264{value:.3f}" if value <= 0.0061 else f"{value:.3f}"


def build_table(significance: pd.DataFrame) -> pd.DataFrame:
    """One row per metric: both models, the difference, its CI and adjusted p."""
    indexed = significance.set_index("metric").loc[DISPLAY_ORDER]
    return pd.DataFrame(
        {
            "Metric": [LABELS[metric] for metric in indexed.index],
            "GraphSAGE": [f"{value:.3f}" for value in indexed["gnn"]],
            "TFT": [f"{value:.3f}" for value in indexed["tft"]],
            "Difference (TFT - GNN)": [
                f"{value:+.3f}" for value in indexed["difference_tft_minus_gnn"]
            ],
            "95% CI": [
                f"[{low:.3f}, {high:.3f}]"
                for low, high in zip(indexed["ci_low"], indexed["ci_high"], strict=True)
            ],
            "Holm-adjusted p": [format_p_value(p) for p in indexed["p_value_holm"]],
        }
    )


def to_markdown(table: pd.DataFrame) -> str:
    header = "| " + " | ".join(table.columns) + " |"
    divider = "|" + "|".join(["---"] + ["---:"] * (len(table.columns) - 1)) + "|"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False)
    ]
    return "\n".join([header, divider, *body]) + "\n"


def plot_comparison(
    aligned: pd.DataFrame,
    operating_points: pd.DataFrame,
    significance: pd.DataFrame,
) -> None:
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1, 1.05]}
    )
    figure.patch.set_facecolor(SURFACE)

    labels = aligned["future_conflict"].to_numpy(dtype=int)
    prevalence = labels.mean()
    for column, model, color, name in [
        ("gnn_probability", "GraphSAGE (tuned)", GNN_COLOR, "GraphSAGE"),
        ("tft_probability", "TFT (tuned)", TFT_COLOR, "TFT"),
    ]:
        curve = precision_recall_curve_data(labels, aligned[column].to_numpy())
        row = operating_points.loc[
            (operating_points["model"] == model)
            & (operating_points["threshold_rule"] == "validation_5to1_cost")
        ].iloc[0]
        left.plot(
            curve["recall"],
            curve["precision"],
            color=color,
            linewidth=2,
            label=f"{name} (AP {row['average_precision']:.2f})",
        )
        left.scatter(
            [row["recall"]],
            [row["precision"]],
            color=color,
            edgecolor=INK,
            s=70,
            zorder=3,
        )
    left.axhline(prevalence, color=MUTED, linestyle=":", linewidth=1.2)
    left.text(0.02, prevalence + 0.015, f"no-skill {prevalence:.3f}", color=MUTED, fontsize=8)
    left.set_xlim(0, 1.0)
    left.set_ylim(0, 1.02)
    left.set_xlabel("Recall", color=INK, fontsize=10)
    left.set_ylabel("Precision", color=INK, fontsize=10)
    left.set_title(
        "Precision-recall on the shared test pairs\n(dots: validation-chosen operating points)",
        color=INK,
        fontsize=10.5,
        loc="left",
    )
    left.legend(frameon=False, fontsize=9, loc="upper right")
    _style(left)

    ordered = significance.set_index("metric").loc[DISPLAY_ORDER]
    positions = np.arange(len(ordered))[::-1]
    differences = ordered["difference_tft_minus_gnn"].to_numpy()
    lower = differences - ordered["ci_low"].to_numpy()
    upper = ordered["ci_high"].to_numpy() - differences
    right.errorbar(
        differences,
        positions,
        xerr=[lower, upper],
        fmt="o",
        color=TFT_COLOR,
        ecolor=TFT_COLOR,
        elinewidth=2,
        capsize=4,
        markersize=7,
    )
    right.axvline(0, color=INK, linewidth=1)
    right.set_yticks(positions)
    right.set_yticklabels(
        [
            LABELS[metric] + (" (lower is better)" if metric == "false_negative_rate" else "")
            for metric in ordered.index
        ],
        fontsize=9,
        color=INK,
    )
    right.set_xlabel("TFT minus GraphSAGE (95% cluster-bootstrap CI)", color=INK, fontsize=10)
    right.set_title(
        "Metric differences, resampled by aircraft pair",
        color=INK,
        fontsize=10.5,
        loc="left",
    )
    _style(right)
    right.grid(False, axis="y")

    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=160, facecolor=SURFACE)
    plt.close(figure)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    aligned = pd.read_csv(ALIGNED_PREDICTIONS)
    significance = pd.read_csv(SIGNIFICANCE_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    shared = metrics.loc[metrics["evaluation_set"] == "shared"]

    table = build_table(significance)
    table.to_csv(TABLE_CSV, index=False)
    TABLE_MARKDOWN.write_text(to_markdown(table), encoding="utf-8")
    plot_comparison(aligned, shared, significance)
    print(to_markdown(table))
    print(f"Saved figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
