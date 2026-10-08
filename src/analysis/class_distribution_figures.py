"""Figures for the step 8 class distribution.

    levels    the positive rate under six ways of counting a case
    regions   traffic volume and positive rate per map cell, side by side,
              so busy cells are not mistaken for risky ones
    time      the rate per five-minute window, and how much of each aircraft's
              observed time is spent in a risky pair

Usage:
    python -m src.analysis.class_distribution_figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.patches import Rectangle

from src.analysis.class_distribution import CELL_DEGREES, LEVEL_ORDER, MIN_CELL_OBSERVATIONS
from src.analysis.geometry_location import AIRPORTS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDA_DIR = PROJECT_ROOT / "experiments" / "eda"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "experiments" / "figures"

INK, MUTED, GRID, AXIS, SURFACE = "#0f172a", "#64748b", "#e2e8f0", "#cbd5e1", "#ffffff"
ACCENT, ALERT = "#3d74b8", "#b45309"

# Single-hue sequential ramps, light to dark: blue for how much traffic, the
# alert hue for how much of it is risky.
TRAFFIC_RAMP = LinearSegmentedColormap.from_list("traffic", ["#dbe7f5", "#7aa9db", "#1b4477"])
RATE_RAMP = LinearSegmentedColormap.from_list("rate", ["#fbeee0", "#e0a160", "#7c3a06"])

UNIT_NOTE = {
    "pair-observation": "one pair, one moment",
    "visit": "one pair, one continuous stretch",
    "aircraft pair": "one pair, whole hour",
    "aircraft-moment": "one aircraft, one moment",
    "trajectory": "one aircraft, one continuous stretch",
    "aircraft": "one aircraft, whole hour",
}


def _style(axes: plt.Axes) -> None:
    axes.grid(True, color=GRID, linewidth=0.7)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
    axes.tick_params(labelsize=9, colors=MUTED)
    axes.set_facecolor(SURFACE)


def plot_levels(levels: pd.DataFrame, path: Path) -> None:
    levels = levels.set_index("level").reindex(LEVEL_ORDER)
    figure, axis = plt.subplots(figsize=(11.5, 5.2))
    figure.patch.set_facecolor(SURFACE)

    positions = np.arange(len(levels))[::-1]
    axis.barh(positions, levels["rate_pct"], height=0.62, color=ALERT)
    for position, (level, row) in zip(positions, levels.iterrows()):
        # reindex turns the counts into floats; cast back so they print as whole numbers.
        axis.annotate(f"{row['rate_pct']:.2f}%   ({int(row['risky']):,} of {int(row['total']):,})",
                      (row["rate_pct"], position), xytext=(6, 0), textcoords="offset points",
                      va="center", fontsize=10, color=INK, fontweight="bold")
    axis.set_yticks(positions, [f"{level}\n{UNIT_NOTE[level]}" for level in levels.index],
                    fontsize=9.5)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Positive rate (%)", fontsize=10, color=INK)
    _style(axis)
    axis.grid(False, axis="y")

    figure.suptitle("One hour of traffic, six ways to count a case: risk is 0.6% of "
                    "observations but touches 79% of aircraft",
                    fontsize=12.5, color=INK, x=0.006, ha="left", y=1.0, fontweight="bold")
    figure.tight_layout(rect=(0, 0.08, 1, 0.97))
    figure.text(
        0.006, 0.005,
        "Each bar uses a different unit. A case is risky if it breaks the current rule at least once; "
        "a stretch ends after 60 s (pair) or 120 s (aircraft) unseen.\nAircraft are counted only "
        "when within 50 nm of another aircraft.",
        fontsize=9.5, color=MUTED, ha="left", va="bottom")
    figure.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)


def _draw_cells(axis: plt.Axes, cells: pd.DataFrame, column: str, ramp, norm) -> None:
    for row in cells.itertuples():
        axis.add_patch(Rectangle((row.cell_lon, row.cell_lat), CELL_DEGREES, CELL_DEGREES,
                                 facecolor=ramp(norm(getattr(row, column))),
                                 edgecolor=SURFACE, linewidth=0.8))


def plot_regions(regions: pd.DataFrame, path: Path) -> None:
    mapped = regions.loc[regions["mapped"]]
    sparse = regions.loc[~regions["mapped"]]

    figure, axes = plt.subplots(1, 2, figsize=(15, 6.6))
    figure.patch.set_facecolor(SURFACE)
    panels = [
        (axes[0], "observations", TRAFFIC_RAMP,
         LogNorm(mapped["observations"].min(), mapped["observations"].max()),
         "Traffic: pair-observations per cell", "Pair-observations (log scale)"),
        (axes[1], "rate_pct", RATE_RAMP, Normalize(0, mapped["rate_pct"].max()),
         "Risk: positive rate per cell", "Positive rate (%)"),
    ]
    for axis, column, ramp, norm, title, bar_label in panels:
        _draw_cells(axis, mapped, column, ramp, norm)
        # Too few observations for a stable rate: outlined, not coloured.
        for row in sparse.itertuples():
            axis.add_patch(Rectangle((row.cell_lon, row.cell_lat), CELL_DEGREES, CELL_DEGREES,
                                     facecolor="none", edgecolor=AXIS, linewidth=0.8,
                                     hatch="///"))
        for code, (latitude, longitude) in AIRPORTS.items():
            axis.plot(longitude, latitude, marker="^", ms=7, color=INK,
                      markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=5)
            axis.annotate(code, (longitude, latitude), xytext=(5, 4), textcoords="offset points",
                          fontsize=8.5, color=INK, fontweight="bold", zorder=6)
        axis.set_xlim(regions["cell_lon"].min() - 0.05,
                      regions["cell_lon"].max() + CELL_DEGREES + 0.05)
        axis.set_ylim(regions["cell_lat"].min() - 0.05,
                      regions["cell_lat"].max() + CELL_DEGREES + 0.05)
        axis.set_aspect(1 / np.cos(np.radians(34.0)))
        axis.set_xlabel("Longitude (deg)", fontsize=10, color=INK)
        axis.set_ylabel("Latitude (deg)", fontsize=10, color=INK)
        axis.set_title(title, fontsize=12, color=INK, loc="left", fontweight="bold")
        _style(axis)
        colour_bar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=ramp), ax=axis,
                                     fraction=0.035, pad=0.02)
        colour_bar.set_label(bar_label, fontsize=9.5, color=INK)
        colour_bar.ax.tick_params(labelsize=8.5, colors=MUTED)
        colour_bar.outline.set_visible(False)

    figure.suptitle("The busiest cells are not the riskiest", fontsize=13, color=INK,
                    x=0.006, ha="left", y=1.0, fontweight="bold")
    figure.tight_layout(rect=(0, 0.07, 1, 0.97))
    figure.text(
        0.006, 0.005,
        f"Quarter-degree cells, about 15 nm across, placed by the midpoint of each pair. Hatched cells "
        f"have fewer than {MIN_CELL_OBSERVATIONS:,} observations and no rate is shown.\nRates count "
        "repeated observations of the same encounter; triangles mark the six commercial airports only.",
        fontsize=9.5, color=MUTED, ha="left", va="bottom")
    figure.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)


def plot_time_and_aircraft(windows: pd.DataFrame, involvement: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    figure.patch.set_facecolor(SURFACE)

    # The two window rates differ a hundredfold, so they get separate panels
    # rather than two y-axes on one.
    partial = windows["minutes_covered"] < windows["minutes_covered"].max()
    for axis, column, title, ylabel in [
        (axes[0], "observation_rate_pct", "Per pair-observation", "Positive rate (%)"),
        (axes[1], "aircraft_rate_pct", "Per aircraft", "Aircraft in a risky pair (%)"),
    ]:
        colours = np.where(partial, AXIS, ALERT)
        axis.bar(windows["window_start_min"] + 2.5, windows[column], width=4.2, color=colours)
        axis.set_xlabel("Minutes into the collection", fontsize=10, color=INK)
        axis.set_ylabel(ylabel, fontsize=10, color=INK)
        axis.set_title(f"{title}, by five-minute window", fontsize=11.5, color=INK,
                       loc="left", fontweight="bold")
        axis.set_ylim(0, windows[column].max() * 1.25)
        _style(axis)
    last = windows.iloc[-1]
    axes[1].annotate(f"covers only {last['minutes_covered']:g} min",
                     (last["window_start_min"] + 2.5, last["aircraft_rate_pct"]),
                     xytext=(last["window_start_min"] + 2.5, windows["aircraft_rate_pct"].max() * 1.15),
                     ha="right", fontsize=8.5, color=MUTED,
                     arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})

    axis = axes[2]
    share = involvement["share_of_time_risky_pct"]
    axis.hist(share, bins=np.linspace(0, 100, 21), color=ALERT, edgecolor=SURFACE, linewidth=0.6)
    median = share.median()
    axis.axvline(median, color=INK, linestyle="--", linewidth=1.6)
    axis.annotate(f"median {median:.0f}%", (median, axis.get_ylim()[1] * 0.9), xytext=(6, 0),
                  textcoords="offset points", fontsize=9.5, color=INK, fontweight="bold")
    axis.set_xlabel("Share of the aircraft's observed time in a risky pair (%)",
                    fontsize=10, color=INK)
    axis.set_ylabel("Aircraft", fontsize=10, color=INK)
    axis.set_title("Per aircraft, over its observed time", fontsize=11.5, color=INK,
                   loc="left", fontweight="bold")
    _style(axis)

    figure.suptitle("Risk is steady across the hour and spread across most aircraft",
                    fontsize=13, color=INK, x=0.006, ha="left", y=1.02, fontweight="bold")
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    figure.text(
        0.006, 0.005,
        "Left two panels share the same windows; an aircraft counts as risky in a window if it was in a "
        "risky pair at any moment in it. Grey bar: a window cut short by the end of collection.\nRight: "
        f"{int((share > 50).sum())} of {len(share)} aircraft were in a risky pair for more than half the "
        "time they were observed.",
        fontsize=9.5, color=MUTED, ha="left", va="bottom")
    figure.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()

    tables = {name: EDA_DIR / f"step8_{name}.csv"
              for name in ("rates_by_level", "rates_by_region", "rates_by_window",
                           "aircraft_involvement")}
    missing = [path.name for path in tables.values() if not path.exists()]
    if missing:
        raise SystemExit(f"Run python -m src.analysis.class_distribution first; missing {missing}")

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    plot_levels(pd.read_csv(tables["rates_by_level"]),
                args.figure_dir / "step8_rates_by_level.png")
    plot_regions(pd.read_csv(tables["rates_by_region"]),
                 args.figure_dir / "step8_rates_by_region.png")
    plot_time_and_aircraft(pd.read_csv(tables["rates_by_window"]),
                           pd.read_csv(tables["aircraft_involvement"]),
                           args.figure_dir / "step8_rates_over_time_and_aircraft.png")
    for name in ("step8_rates_by_level.png", "step8_rates_by_region.png",
                 "step8_rates_over_time_and_aircraft.png"):
        print(f"Wrote {name}")


if __name__ == "__main__":
    main()
