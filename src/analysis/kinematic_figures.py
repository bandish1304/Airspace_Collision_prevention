"""Figures for the step 7 kinematic comparison.

Three plots, matching the plot types the progress feedback asked for:

    distributions  each feature's shape per group, one median per aircraft pair
    box plots      the same features compactly, side by side
    scatter        the label's own boundary, and the two motion features that
                   ought to separate risky pairs if anything does

Distributions and box plots use one value per aircraft pair so a long
encounter counts once; the scatter uses sampled rows so the hard edge of the
labelling rule stays visible.

Usage:
    python -m src.analysis.kinematic_figures
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.kinematic_comparison import (
    GROUP_ORDER,
    NEAR_MISS_LATERAL_NM,
    NEAR_MISS_VERTICAL_FT,
    load_groups,
    per_pair,
)
from src.baselines.data import DEFAULT_PAIRS_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "experiments" / "figures"
SEPARABILITY_PATH = PROJECT_ROOT / "experiments" / "eda" / "step7_feature_separability.csv"

INK, MUTED, GRID, AXIS, SURFACE = "#0f172a", "#64748b", "#e2e8f0", "#cbd5e1", "#ffffff"

# Checked for colour-blind separation (OKLab distance >= 16 under protanopia,
# deuteranopia and tritanopia). Line style repeats the identity for print.
STYLE = {
    "risky": {"color": "#b45309", "linestyle": "-", "linewidth": 2.2},
    "near-miss": {"color": "#3d74b8", "linestyle": "--", "linewidth": 2.0},
    "other safe": {"color": "#94a3b8", "linestyle": ":", "linewidth": 2.0},
}

# feature -> (axis label, display range, log x)
PANELS = {
    "lateral_distance_nm": ("Lateral separation (nm), defines the label", (0, 50), False),
    "vertical_separation_ft": ("Vertical separation (ft), defines the label", (0, 20_000), False),
    "closing_speed_mps": ("Closing speed (m/s)", (-300, 300), False),
    "bearing_difference_deg": ("Bearing difference (deg)", (0, 180), False),
    "time_to_cpa_min": ("Time to closest approach (min, converging only)", (0.001, 60), True),
    "altitude_ft": ("Mean altitude (ft)", (0, 40_000), False),
    "speed_kt": ("Mean ground speed (kt)", (0, 500), False),
}

# The two features the risk label is computed from. They are drawn because the
# feedback asked for them by name, on a shaded background so their perfect
# separation is not read as a finding: risky pairs sit below the limits by
# construction.
LABEL_DEFINING_PANELS = {"lateral_distance_nm", "vertical_separation_ft"}
LABEL_DEFINING_SHADE = "#eef2f6"

# Panels whose display range leaves out a visible share of a group.
TRIMMED_PANELS = {"closing_speed_mps": "closing speeds beyond +/-300 m/s",
                  "vertical_separation_ft": "vertical separations above 20,000 ft"}

SCATTER_SAMPLE = 6_000


def _style(axes: plt.Axes) -> None:
    axes.grid(True, color=GRID, linewidth=0.7)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
    axes.tick_params(labelsize=9, colors=MUTED)
    axes.set_facecolor(SURFACE)


def _auroc_note(separability: pd.DataFrame, feature: str) -> str:
    near = separability.query(
        "unit == 'pair' and comparison == 'risky vs near-miss' and feature == @feature"
    )["auroc"]
    return f"AUROC vs near-miss {near.iloc[0]:.2f}" if len(near) else ""


def plot_distributions(pairs: pd.DataFrame, separability: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(19.5, 8.6))
    figure.patch.set_facecolor(SURFACE)

    for axis, (feature, (label, (low, high), log_x)) in zip(axes.flat, PANELS.items()):
        if log_x:
            bins = np.logspace(np.log10(low), np.log10(high), 40)
            axis.set_xscale("log")
        else:
            bins = np.linspace(low, high, 40)
        for group in GROUP_ORDER:
            values = pairs.loc[pairs["group"] == group, feature].dropna()
            # Values outside the range are left out, not clipped: clipping piles
            # them into the end bins, where they read as a real peak.
            values = values[(values >= low) & (values <= high)]
            # Share of the group per bin rather than density: on the log axis
            # the narrow left-hand bins would otherwise be inflated by their width.
            axis.hist(values, bins=bins, weights=np.full(len(values), 1 / max(len(values), 1)),
                      histtype="step", **STYLE[group])
        axis.set_xlim(low, high)
        # Headroom so the tallest peak clears the AUROC note in the corner.
        axis.set_ylim(0, axis.get_ylim()[1] * 1.18)
        axis.set_xlabel(label, fontsize=10, color=INK)
        axis.set_yticks([])
        axis.annotate(_auroc_note(separability, feature), (0.98, 0.94),
                      xycoords="axes fraction", ha="right", fontsize=9.5,
                      color=INK, fontweight="bold")
        _style(axis)
        if feature in LABEL_DEFINING_PANELS:
            axis.set_facecolor(LABEL_DEFINING_SHADE)

    # The share converging is a single proportion per group, so it gets bars.
    axis = axes.flat[-1]
    shares = [pairs.loc[pairs["group"] == group, "converging"].mean() * 100
              for group in GROUP_ORDER]
    bars = axis.bar(GROUP_ORDER, shares, color=[STYLE[g]["color"] for g in GROUP_ORDER],
                    width=0.6)
    for bar, share in zip(bars, shares):
        axis.annotate(f"{share:.0f}%", (bar.get_x() + bar.get_width() / 2, share),
                      xytext=(0, 4), textcoords="offset points", ha="center",
                      fontsize=10, color=INK, fontweight="bold")
    axis.set_ylim(0, 100)
    axis.set_ylabel("Pairs converging (%)", fontsize=10, color=INK)
    axis.annotate(_auroc_note(separability, "converging"), (0.98, 0.94),
                  xycoords="axes fraction", ha="right", fontsize=9.5,
                  color=INK, fontweight="bold")
    _style(axis)

    handles = [plt.Line2D([], [], **STYLE[group]) for group in GROUP_ORDER]
    figure.legend(handles, GROUP_ORDER, loc="upper right", ncol=3, frameon=False,
                  fontsize=10, bbox_to_anchor=(0.995, 1.0))
    figure.suptitle(
        "Risky pairs move like near-misses; both are lower and slower than other traffic",
        fontsize=13, color=INK, x=0.006, ha="left", y=1.0, fontweight="bold")
    figure.tight_layout(rect=(0, 0.09, 1, 0.97))
    trimmed = []
    for feature, description in TRIMMED_PANELS.items():
        low, high = PANELS[feature][1]
        shares = ", ".join(
            f"{group} {100 * (~pairs.loc[pairs['group'] == group, feature].between(low, high)).mean():.1f}%"
            for group in GROUP_ORDER
        )
        trimmed.append(f"{description} ({shares})")
    figure.text(
        0.006, 0.005,
        "One median per aircraft pair, so long encounters count once. Each curve shows its group's share per bin, "
        "so groups of very different sizes can be compared by shape. AUROC is for that feature alone,\nrisky "
        "against near-miss: 0.5 means indistinguishable. Shaded panels are the two features the risk label is "
        "defined from, so risky pairs fall below their limits by construction.\nNot drawn: "
        + "; ".join(trimmed) + ".",
        fontsize=9.5, color=MUTED, ha="left", va="bottom")
    figure.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)


def plot_boxplots(pairs: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, len(PANELS), figsize=(22, 4.8))
    figure.patch.set_facecolor(SURFACE)

    for axis, (feature, (label, (low, high), log_y)) in zip(axes, PANELS.items()):
        data = [pairs.loc[pairs["group"] == group, feature].dropna() for group in GROUP_ORDER]
        boxes = axis.boxplot(data, widths=0.6, showfliers=False, patch_artist=True,
                             medianprops={"color": INK, "linewidth": 1.8})
        for patch, group in zip(boxes["boxes"], GROUP_ORDER):
            patch.set_facecolor(STYLE[group]["color"])
            patch.set_alpha(0.75)
            patch.set_edgecolor(STYLE[group]["color"])
        if log_y:
            axis.set_yscale("log")
        axis.set_xticks(range(1, len(GROUP_ORDER) + 1), GROUP_ORDER, fontsize=9,
                        rotation=20)
        axis.set_title(textwrap.fill(label, 26), fontsize=10, color=INK, loc="left")
        _style(axis)
        if feature in LABEL_DEFINING_PANELS:
            axis.set_facecolor(LABEL_DEFINING_SHADE)

    figure.suptitle("The same features as box plots", fontsize=13, color=INK,
                    x=0.006, ha="left", y=1.02, fontweight="bold")
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.text(
        0.006, 0.005,
        "Boxes span the middle half of aircraft pairs; whiskers reach 1.5 times that span. Outliers are "
        "not drawn -- with 78,000 safe pairs they would cover the plot. Shaded panels are the two features "
        "the risk label is defined from.",
        fontsize=9.5, color=MUTED, ha="left", va="bottom")
    figure.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)


def plot_scatter(rows: pd.DataFrame, path: Path, seed: int = 0) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 5.8))
    figure.patch.set_facecolor(SURFACE)
    markers = {"risky": "o", "near-miss": "s", "other safe": "^"}

    # Left: the rule itself. Other safe pairs are drawn only near the box,
    # since the full range would push the interesting corner out of view.
    near_view = rows.loc[(rows["lateral_distance_nm"] <= 12)
                         & (rows["vertical_separation_ft"] <= 2_500)]
    for group in reversed(GROUP_ORDER):
        members = near_view.loc[near_view["group"] == group]
        members = members.sample(min(SCATTER_SAMPLE, len(members)), random_state=seed)
        left.scatter(members["lateral_distance_nm"], members["vertical_separation_ft"],
                     s=6, alpha=0.35, color=STYLE[group]["color"],
                     marker=markers[group], label=group, linewidths=0)
    for width, height, text in [(5, 1_000, "risk rule"),
                                (NEAR_MISS_LATERAL_NM, NEAR_MISS_VERTICAL_FT, "near-miss band")]:
        left.plot([0, width, width], [height, height, 0], color=INK, linewidth=1.3)
        left.annotate(text, (width, height), xytext=(4, 4), textcoords="offset points",
                      fontsize=9, color=INK, fontweight="bold")
    left.set_xlim(0, 12)
    left.set_ylim(0, 2_500)
    left.set_xlabel("Lateral separation (nm)", fontsize=10, color=INK)
    left.set_ylabel("Vertical separation (ft)", fontsize=10, color=INK)
    left.set_title("The label is a box drawn on these two axes", fontsize=12,
                   color=INK, loc="left", fontweight="bold")
    _style(left)

    # Right: the motion that should mark a real conflict.
    converging = rows.loc[(rows["converging"] == 1) & rows["group"].isin(GROUP_ORDER[:2])]
    for group in reversed(GROUP_ORDER[:2]):
        members = converging.loc[converging["group"] == group]
        members = members.sample(min(SCATTER_SAMPLE, len(members)), random_state=seed)
        members = members.loc[members["closing_speed_mps"].between(0, 300)
                              & members["time_to_cpa_min"].between(0.001, 60)]
        right.scatter(members["closing_speed_mps"], members["time_to_cpa_min"],
                      s=6, alpha=0.35,
                      color=STYLE[group]["color"], marker=markers[group], linewidths=0)
    right.set_yscale("log")
    right.set_xlim(0, 300)
    right.set_ylim(0.001, 60)
    right.set_xlabel("Closing speed (m/s)", fontsize=10, color=INK)
    right.set_ylabel("Time to closest approach (min)", fontsize=10, color=INK)
    right.set_title("Converging pairs: risky ones reach closest approach sooner",
                    fontsize=12, color=INK, loc="left", fontweight="bold")
    _style(right)

    handles = [plt.Line2D([], [], marker=markers[g], linestyle="", markersize=7,
                          color=STYLE[g]["color"]) for g in GROUP_ORDER]
    figure.legend(handles, GROUP_ORDER, loc="upper right", ncol=3, frameon=False,
                  fontsize=10, bbox_to_anchor=(0.995, 1.03))
    figure.tight_layout(rect=(0, 0.11, 1, 1))
    figure.text(
        0.006, 0.005,
        f"Up to {SCATTER_SAMPLE:,} sampled rows per group. Left: risky pairs fill the rule's box by definition, "
        "so these two axes separate the groups perfectly and say nothing about risk.\nRight: at the same "
        "closing speed, risky pairs sit below the near-misses. Time to closest approach is roughly distance "
        "over closing speed, and risky pairs are closer by definition,\nso this separation is largely the "
        "label's distance limit reappearing rather than a separate sign of risk.",
        fontsize=9.5, color=MUTED, ha="left", va="bottom")
    figure.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()

    if not SEPARABILITY_PATH.exists():
        raise SystemExit("Run python -m src.analysis.kinematic_comparison first; "
                         "the figures annotate its AUROC values.")
    separability = pd.read_csv(SEPARABILITY_PATH)

    rows = load_groups(args.pairs_path)
    pairs = per_pair(rows)

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "step7_feature_distributions.png": lambda p: plot_distributions(pairs, separability, p),
        "step7_feature_boxplots.png": lambda p: plot_boxplots(pairs, p),
        "step7_feature_scatter.png": lambda p: plot_scatter(rows, p),
    }
    for name, draw in outputs.items():
        draw(args.figure_dir / name)
        print(f"Wrote {name}")


if __name__ == "__main__":
    main()
