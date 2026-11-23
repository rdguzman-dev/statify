"""Chart 2: how valence is spread across the catalogue.

A single histogram with the bars tinted by mood band, the share of songs in
each band printed above the tallest bar of that band, and mean/median lines.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from . import cli, theme as theme_mod
from .valence_stats import ROOT, band_index, band_table, load_tracks

DEFAULT_OUT = ROOT / "charts" / "valence_distribution.png"
BINS = 40


def plot(path: Path, out: Path, tm: theme_mod.Theme) -> None:
    tracks = load_tracks(path)
    values = tracks["valence"].to_numpy()
    bands = band_table(tracks)
    colors = tm.ramp("bands")

    counts, edges = np.histogram(values, bins=BINS, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2
    bar_colors = [colors[i] for i in band_index(centers)]

    fig, ax = plt.subplots(figsize=tm.figure_size("distribution", (11.0, 6.5)))
    ax.bar(
        centers,
        counts,
        width=(edges[1] - edges[0]) * 0.98,
        color=bar_colors,
        zorder=2,
    )

    # Share of songs in each band, above that band's tallest bar.
    band_ids = band_index(centers)
    for idx, row in enumerate(bands.iter_rows(named=True)):
        in_band = counts[band_ids == idx]
        top = in_band.max() if in_band.size else 0
        ax.annotate(
            f"{row['share']:.0%}",
            xy=(centers[band_ids == idx][in_band.argmax()], top),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=tm.size("annotation"),
            color=tm.color("text"),
            fontweight="bold",
        )
        ax.annotate(
            row["label"],
            xy=(centers[band_ids == idx][in_band.argmax()], top),
            xytext=(0, 22),
            textcoords="offset points",
            ha="center",
            fontsize=tm.size("caption"),
            color=tm.color("muted_text"),
        )

    mean = float(np.mean(values))
    median = float(np.median(values))
    ax.axvline(mean, color=tm.color("reference"), lw=1.6, zorder=3)
    ax.annotate(
        f"mean {mean:.2f} / median {median:.2f}",
        xy=(mean, 0.94),
        xycoords=("data", "axes fraction"),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=tm.size("annotation"),
        color=tm.color("reference"),
    )

    below = float((values < 0.5).mean())
    tm.title(
        ax,
        f"Mood is spread across the whole scale, with {below:.0%} of songs "
        f"on the sadder half",
    )
    ax.set_xlabel("valence  (0 = sad, 1 = happy)", fontsize=tm.size("label"))
    ax.set_ylabel("songs", fontsize=tm.size("label"))
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="x", visible=False)

    tm.caption(
        fig,
        "Each bar is a 0.025-wide slice of the valence scale; colours mark "
        "the five bands used in the other charts.",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.18)
    tm.save(fig, out)


def main() -> None:
    args = cli.base_parser(__doc__ or "", DEFAULT_OUT).parse_args()
    tm = theme_mod.load(args.theme).apply()
    plot(args.data, args.out, tm)


if __name__ == "__main__":
    main()
