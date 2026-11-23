"""Chart 1: the happiest and saddest genres.

Two lollipop panels, one for the lowest mean valence genres and one for the
highest, with every genre's value labelled. The reference line marks the
mean valence of the whole catalogue.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from . import cli, theme as theme_mod
from .valence_stats import ROOT, genre_table, load_track_genres, load_tracks

DEFAULT_OUT = ROOT / "charts" / "valence_genre_rank.png"


def draw_panel(ax, rows, overall: float, tm: theme_mod.Theme) -> None:
    y = np.arange(rows.height)
    means = rows["mean_valence"].to_numpy()
    ax.hlines(y, 0, means, color=tm.color("accent_soft"), lw=2.2, zorder=2)
    ax.plot(means, y, "o", ms=7, color=tm.color("accent"), zorder=3)
    for yi, mean in zip(y, means):
        ax.annotate(
            f"{mean:.2f}",
            (mean, yi),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=tm.size("tick"),
            color=tm.color("text"),
        )
    ax.set_yticks(y)
    ax.set_yticklabels(rows["track_genre"].to_list(), fontsize=tm.size("tick"))
    ax.set_ylim(-0.8, rows.height - 0.2)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.axvline(overall, color=tm.color("reference"), ls=":", lw=1.4, zorder=1)
    ax.annotate(
        f"all tracks {overall:.2f}",
        xy=(overall, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=tm.size("annotation"),
        color=tm.color("reference"),
    )
    ax.grid(axis="y", visible=False)


def plot(path: Path, out: Path, tm: theme_mod.Theme, show: int = 12) -> None:
    tracks = load_tracks(path)
    genres = genre_table(load_track_genres(path))
    overall = float(tracks["valence"].mean())

    low = genres.head(show)
    high = genres.tail(show).reverse()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=tm.figure_size("genre_rank", (12.0, 7.5)),
        sharex=True,
        gridspec_kw={"wspace": 0.35},
    )
    draw_panel(axes[0], low, overall, tm)
    draw_panel(axes[1], high, overall, tm)
    axes[0].set_title(
        "Saddest genres", loc="left", fontsize=tm.size("subtitle"),
        color=tm.color("muted_text"), pad=26,
    )
    axes[1].set_title(
        "Happiest genres", loc="left", fontsize=tm.size("subtitle"),
        color=tm.color("muted_text"), pad=26,
    )

    tm.heading(
        fig,
        f"Genre predicts mood far better than it predicts popularity: "
        f"mean valence runs {genres['mean_valence'].min():.2f} to "
        f"{genres['mean_valence'].max():.2f}",
    )
    fig.text(
        0.5,
        0.095,
        "mean valence  (0 = sad, 1 = happy)",
        ha="center",
        fontsize=tm.size("label"),
        color=tm.color("text"),
    )
    tm.caption(
        fig,
        f"Average of ~1,000 unique tracks per genre; {genres.height - 2 * show} "
        f"of the {genres.height} genres sit between these two groups.",
    )
    fig.subplots_adjust(left=0.14, right=0.985, top=0.83, bottom=0.17, wspace=0.45)
    tm.fit_top(fig)
    tm.save(fig, out)


def main() -> None:
    parser = cli.base_parser(__doc__ or "", DEFAULT_OUT)
    parser.add_argument("--show", type=int, default=12, help="genres per panel")
    args = parser.parse_args()
    tm = theme_mod.load(args.theme).apply()
    plot(args.data, args.out, tm, show=args.show)


if __name__ == "__main__":
    main()
