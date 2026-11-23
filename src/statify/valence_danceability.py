"""Chart 4: valence against danceability.

Danceability is the audio feature that moves most with valence, so this shows
the binned mean danceability at each level of valence with a 95% confidence
band and prints the correlation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from . import cli, theme as theme_mod
from .valence_stats import ROOT, load_tracks

DEFAULT_OUT = ROOT / "charts" / "valence_danceability.png"
BINS = 12


def binned(tracks: pl.DataFrame) -> pl.DataFrame:
    edges = np.linspace(0.0, 1.0, BINS + 1)
    idx = np.clip(
        np.digitize(tracks["valence"].to_numpy(), edges[1:-1], right=False),
        0,
        BINS - 1,
    )
    framed = tracks.with_columns(pl.Series("bin_idx", idx, dtype=pl.Int64))
    return (
        framed.group_by("bin_idx")
        .agg(
            pl.len().alias("n"),
            pl.col("danceability").mean().alias("mean"),
            pl.col("danceability").std().alias("sd"),
        )
        .with_columns(
            (1.96 * pl.col("sd") / pl.col("n").sqrt()).alias("ci"),
            ((pl.col("bin_idx") + 0.5) / BINS).alias("center"),
        )
        .sort("bin_idx")
    )


def plot(path: Path, out: Path, tm: theme_mod.Theme) -> None:
    tracks = load_tracks(path)
    curve = binned(tracks)
    r = float(tracks.select(pl.corr("valence", "danceability")).item())
    overall = float(tracks["danceability"].mean())

    fig, ax = plt.subplots(figsize=tm.figure_size("danceability", (11.0, 6.5)))
    x = curve["center"].to_numpy()
    mean = curve["mean"].to_numpy()
    ci = curve["ci"].to_numpy()

    ax.fill_between(
        x, mean - ci, mean + ci, color=tm.color("accent_soft"), alpha=0.55, zorder=2
    )
    ax.plot(x, mean, color=tm.color("accent"), marker="o", ms=5, lw=2.2, zorder=3)
    ax.axhline(
        overall, color=tm.color("reference"), ls=":", lw=1.4, zorder=1
    )
    ax.annotate(
        f"every song {overall:.2f}",
        xy=(1.0, overall),
        xytext=(0, 6),
        textcoords="offset points",
        ha="right",
        fontsize=tm.size("caption"),
        color=tm.color("reference"),
    )
    ax.annotate(
        f"r = {r:+.2f}",
        xy=(0.03, 0.86),
        xycoords="axes fraction",
        fontsize=tm.size("title") + 4,
        color=tm.color("accent"),
        fontweight="bold",
    )
    ax.annotate(
        "correlation between valence and danceability",
        xy=(0.03, 0.80),
        xycoords="axes fraction",
        fontsize=tm.size("annotation"),
        color=tm.color("muted_text"),
    )

    tm.title(ax, "Happier songs are more danceable, and the trend is steady")
    ax.set_xlabel("valence  (0 = sad, 1 = happy)", fontsize=tm.size("label"))
    ax.set_ylabel("mean danceability", fontsize=tm.size("label"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="x", visible=False)

    n_per_bin = int(round(curve["n"].mean()))
    tm.caption(
        fig,
        f"Each point averages one of {BINS} equal-width valence slices "
        f"(~{n_per_bin:,} songs each); shading is the 95% confidence band. "
        f"The danceability axis starts at 0.",
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.87, bottom=0.19)
    tm.save(fig, out)


def main() -> None:
    args = cli.base_parser(__doc__ or "", DEFAULT_OUT).parse_args()
    tm = theme_mod.load(args.theme).apply()
    plot(args.data, args.out, tm)


if __name__ == "__main__":
    main()
