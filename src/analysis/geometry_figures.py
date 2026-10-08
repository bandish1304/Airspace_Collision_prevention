"""Figures for the step 4 geometry/location analysis.

Two panels, one point each: where risky encounters sit relative to the six
SoCal airports, and how concentrated they are close to those airports rather
than spread evenly across the bounding box.

Usage:
    python -m src.analysis.geometry_figures data/processed/<file>_pairs.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.encounters import (
    DEFAULT_ENCOUNTER_GAP_SECONDS,
    DEFAULT_LATERAL_SEPARATION_NM,
    DEFAULT_VERTICAL_SEPARATION_FT,
    METRES_PER_FOOT,
    READ_CHUNK_ROWS,
    group_by_pair_over_time,
    load_positive_pairs,
)
from src.analysis.geometry_location import (
    AIRPORTS,
    altitude_band_risk_rates,
    classify_geometry,
    find_safe_example_pair,
    nearest_airport,
    select_representative_encounters,
    summarise_encounter_geometry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "experiments" / "figures"

# Seconds of context to show before/after an encounter, and either side of a
# safe pair's closest approach, without pulling in the rest of their session.
TRAJECTORY_CONTEXT_PADDING_S = 60
SAFE_EXAMPLE_WINDOW_S = 300

INK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"
AXIS = "#cbd5e1"
ACCENT = "#3d74b8"
ALERT = "#b45309"
SURFACE = "#ffffff"

# One color per geometry category, in the same order every legend uses.
CATEGORY_COLORS = {
    "separating": "#94a3b8",
    "crossing": "#3d74b8",
    "head-on": "#b91c1c",
    "overtaking": "#b45309",
    "parallel / formation": "#166534",
}
CATEGORY_ORDER = list(CATEGORY_COLORS)


def _style(axes: plt.Axes) -> None:
    axes.grid(True, color=GRID, linewidth=0.7)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
    axes.tick_params(labelsize=9, colors=MUTED)
    axes.set_facecolor(SURFACE)


def plot_geography(encounters: pd.DataFrame, output_path: Path) -> None:
    """Encounter locations by geometry category, and distance to nearest airport."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.6))
    figure.patch.set_facecolor(SURFACE)

    for category in CATEGORY_ORDER:
        rows = encounters[encounters["geometry"] == category]
        if rows.empty:
            continue
        left.scatter(
            rows["mean_longitude"], rows["mean_latitude"],
            s=18, alpha=0.55, color=CATEGORY_COLORS[category],
            label=f"{category} ({len(rows)})", linewidths=0,
        )
    for code, (airport_lat, airport_lon) in AIRPORTS.items():
        left.scatter(airport_lon, airport_lat, marker="^", s=110, color=INK,
                     edgecolors=SURFACE, linewidths=1.2, zorder=5)
        left.annotate(code, (airport_lon, airport_lat), xytext=(6, 4),
                      textcoords="offset points", fontsize=9.5, color=INK,
                      fontweight="bold")
    left.set_xlabel("Longitude", fontsize=10, color=INK)
    left.set_ylabel("Latitude", fontsize=10, color=INK)
    left.set_title("Risky encounters cluster around airports, not evenly",
                   fontsize=12, color=INK, loc="left", fontweight="bold")
    left.legend(loc="lower left", fontsize=8.5, frameon=False, markerscale=1.4)
    _style(left)

    distance = encounters["airport_distance_nm"].clip(upper=40)
    right.hist(distance, bins=np.arange(0, 40.5, 2), color=ACCENT,
              edgecolor=SURFACE, linewidth=0.5)
    median = encounters["airport_distance_nm"].median()
    right.axvline(median, color=ALERT, linewidth=2, linestyle="--")
    right.annotate(f"median {median:.1f} nm", (median, right.get_ylim()[1] * 0.9),
                   xytext=(8, 0), textcoords="offset points", fontsize=9.5,
                   color=ALERT, fontweight="bold")
    right.set_xlabel("Distance to nearest airport (nm)", fontsize=10, color=INK)
    right.set_ylabel("Number of encounters", fontsize=10, color=INK)
    right.set_title("Most risky encounters sit close to an airport",
                    fontsize=12, color=INK, loc="left", fontweight="bold")
    within_10 = (encounters["airport_distance_nm"] <= 10).mean() * 100
    right.annotate(f"{within_10:.0f}% within 10 nm\nof an airport",
                   (0.62, 0.72), xycoords="axes fraction", fontsize=9.5, color=MUTED)
    _style(right)

    figure.suptitle(
        f"{len(encounters):,} distinct risky encounters — where they happen",
        fontsize=13, color=INK, x=0.006, ha="left", y=1.02, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)


def plot_altitude_bands(band_summary: pd.DataFrame, output_path: Path) -> None:
    """Risk rate per altitude band, log-scaled since it spans two orders of magnitude."""
    figure, axes = plt.subplots(figsize=(9, 5.2))
    figure.patch.set_facecolor(SURFACE)

    positions = np.arange(len(band_summary))
    axes.bar(positions, band_summary["risk_rate_pct"], color=ACCENT, width=0.6,
             edgecolor=SURFACE, linewidth=0.5)
    axes.set_yscale("log")
    axes.set_ylim(top=band_summary["risk_rate_pct"].max() * 4)
    axes.set_xticks(positions, band_summary["altitude_band"], fontsize=9.5)
    axes.set_ylabel("Risk rate (%, log scale)", fontsize=10, color=INK)
    axes.set_title("Risk rate falls roughly 150x from ground level to high altitude",
                   fontsize=12.5, color=INK, loc="left", fontweight="bold")
    for position, row in zip(positions, band_summary.itertuples()):
        axes.annotate(
            f"{row.risk_rate_pct:.3f}%\n({row.total_pairs:,} pairs)",
            (position, row.risk_rate_pct), xytext=(0, 6), textcoords="offset points",
            ha="center", fontsize=9, color=INK, fontweight="bold")
    _style(axes)

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)


# Step 4: every observation of one aircraft pair, including lat/lon, for a trajectory trace.
def _trace_pair_positions(pairs_path: Path, aircraft_a: str, aircraft_b: str) -> pd.DataFrame:
    columns = [
        "snapshot_time", "icao24_a", "icao24_b",
        "latitude_a", "longitude_a", "latitude_b", "longitude_b",
        "lateral_distance_nm", "vertical_separation_m",
    ]
    found = []
    for chunk in pd.read_csv(pairs_path, usecols=columns, chunksize=READ_CHUNK_ROWS):
        low = np.minimum(chunk["icao24_a"], chunk["icao24_b"])
        high = np.maximum(chunk["icao24_a"], chunk["icao24_b"])
        found.append(chunk.loc[(low == aircraft_a) & (high == aircraft_b)])
    return pd.concat(found).sort_values("snapshot_time")


def plot_trajectories(examples: list[dict], output_path: Path) -> None:
    """Small multiples: one lat/lon path per representative example.

    Aircraft are labelled by their role in this pair (icao24_a / icao24_b in
    the trace, not a fixed identity), since which aircraft is "a" or "b" is
    arbitrary per pair and carries no meaning across panels.
    """
    columns = 3
    rows = -(-len(examples) // columns)
    figure, axes_grid = plt.subplots(rows, columns, figsize=(14, 4.6 * rows))
    figure.patch.set_facecolor(SURFACE)
    axes_list = np.atleast_1d(axes_grid).flatten()

    for axes, example in zip(axes_list, examples):
        track = example["track"]
        mean_lat = track[["latitude_a", "latitude_b"]].to_numpy().mean()
        axes.plot(track["longitude_a"], track["latitude_a"], color=ACCENT,
                  linewidth=2, marker="o", markersize=3, label="aircraft A")
        axes.plot(track["longitude_b"], track["latitude_b"], color=ALERT,
                  linewidth=2, marker="o", markersize=3, label="aircraft B")
        axes.scatter(track["longitude_a"].iloc[0], track["latitude_a"].iloc[0],
                     marker="^", s=110, color=ACCENT, zorder=5, edgecolors=INK,
                     linewidths=0.8, label="start")
        axes.scatter(track["longitude_b"].iloc[0], track["latitude_b"].iloc[0],
                     marker="^", s=110, color=ALERT, zorder=5, edgecolors=INK,
                     linewidths=0.8)
        axes.scatter(track["longitude_a"].iloc[-1], track["latitude_a"].iloc[-1],
                     marker="s", s=100, color=SURFACE, zorder=6, edgecolors=ACCENT,
                     linewidths=2.2, label="end")
        axes.scatter(track["longitude_b"].iloc[-1], track["latitude_b"].iloc[-1],
                     marker="s", s=100, color=SURFACE, zorder=6, edgecolors=ALERT,
                     linewidths=2.2)
        # Degrees lat/lon are not equal distance; approximate true shape at this latitude.
        axes.set_aspect(1 / max(np.cos(np.radians(mean_lat)), 0.2))
        axes.set_title(example["title"], fontsize=11, color=INK, loc="left", fontweight="bold")
        axes.tick_params(labelsize=8, colors=MUTED)
        axes.set_facecolor(SURFACE)

    for axes in axes_list[len(examples):]:
        axes.axis("off")
    handles, labels = axes_list[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper right", fontsize=9, frameon=False,
                  bbox_to_anchor=(0.99, 1.02))

    figure.suptitle(
        "Representative risky vs. safe trajectories (\u25b2 start, \u25a0 end)",
        fontsize=13, color=INK, x=0.006, ha="left", y=1.01, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs_path", type=Path, help="Path to a pair-feature CSV.")
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument(
        "--encounter-gap-seconds", type=int, default=DEFAULT_ENCOUNTER_GAP_SECONDS
    )
    args = parser.parse_args()

    positives, _ = load_positive_pairs(args.pairs_path)
    if positives.empty:
        print("No positive pair-observations found.")
        return

    encounter_id = group_by_pair_over_time(positives, args.encounter_gap_seconds)
    encounters = summarise_encounter_geometry(positives, encounter_id)
    encounters["geometry"] = classify_geometry(encounters)
    encounters = pd.concat(
        [encounters, nearest_airport(encounters["mean_latitude"], encounters["mean_longitude"])],
        axis=1,
    )

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.figure_dir / "step4_geography.png"
    plot_geography(encounters, output_path)
    print(f"Wrote {output_path.name}")

    band_summary = altitude_band_risk_rates(args.pairs_path)
    band_output_path = args.figure_dir / "step4_altitude_bands.png"
    plot_altitude_bands(band_summary, band_output_path)
    print(f"Wrote {band_output_path.name}")

    trajectory_categories = ("head-on", "crossing", "overtaking", "parallel / formation")
    representatives = select_representative_encounters(encounters, trajectory_categories)
    examples = []
    for row in representatives.itertuples():
        # A little padding either side of the encounter shows the approach and
        # departure, without pulling in the pair's whole unrelated session history.
        full_track = _trace_pair_positions(args.pairs_path, row.aircraft_a, row.aircraft_b)
        window = full_track["snapshot_time"].between(
            row.start_time - TRAJECTORY_CONTEXT_PADDING_S,
            row.end_time + TRAJECTORY_CONTEXT_PADDING_S,
        )
        track = full_track.loc[window]
        closest_nm = track["lateral_distance_nm"].min()
        examples.append({
            "track": track,
            "title": f"{row.geometry} — closest {closest_nm:.1f} nm (risky)",
        })

    safe_pair = find_safe_example_pair(args.pairs_path)
    if safe_pair is not None:
        full_track = _trace_pair_positions(args.pairs_path, *safe_pair)
        # No encounter window exists for a pair that never breached, so centre
        # the window on the moment of closest approach instead.
        closest_time = full_track.loc[full_track["lateral_distance_nm"].idxmin(), "snapshot_time"]
        window = full_track["snapshot_time"].between(
            closest_time - SAFE_EXAMPLE_WINDOW_S, closest_time + SAFE_EXAMPLE_WINDOW_S
        )
        track = full_track.loc[window]
        closest_nm = track["lateral_distance_nm"].min()
        examples.append({
            "track": track,
            "title": f"near-miss, never breached — closest {closest_nm:.1f} nm (safe)",
        })

    trajectory_output_path = args.figure_dir / "step4_trajectories.png"
    plot_trajectories(examples, trajectory_output_path)
    print(f"Wrote {trajectory_output_path.name}")


if __name__ == "__main__":
    main()
