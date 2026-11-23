"""Chart 5: does a song's mood predict its popularity?

Mean popularity per valence band with 95% confidence intervals. The optional
second panel repeats the trick for major vs minor key, the other thing people
expect to matter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from . import cli, theme as theme_mod
from .valence_stats import ROOT, band_table, load_tracks, mode_split

DEFAULT_OUT = ROOT / "charts" / "valence_popularity.png"
POPULARITY_LIMIT = 42.0


def plot(path: Path, out: Path, tm: theme_mod.Theme, with_mode: bool = False) -> None:
    tracks = load_tracks(path)
    bands = band_table(tracks)
    colors = tm.ramp("bands")

    panels = 2 if with_mode else 1
    fig, axes = plt.subplots(
        1,
        panels,
        figsize=tm.figure_size("popularity", (11.0, 6.0)),
        sharey=False,
        gridspec_kw={"wspace": 0.3, "width_ratios": [3, 1.3][:panels]},
    )
    ax = np.atleast_1d(axes)[0]

    x = np.arange(bands.height)
    means = bands["mean_popularity"].to_numpy()
    cis = bands["ci_popularity"].to_numpy()
    ax.bar(
        x,
        means,
        color=colors,
        width=0.72,
        yerr=cis,
        capsize=4,
        error_kw={"ecolor": tm.color("text"), "lw": 1.1},
        zorder=2,
    )
    for xi, mean in zip(x, means):
        ax.annotate(
            f"{mean:.1f}",
            (xi, mean),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=tm.size("annotation"),
            color=tm.color("text"),
            fontweight="bold",
        )

    spread = float(means.max() - means.min())
    top = float((means + cis).max())
    ax.annotate(
        "",
        xy=(bands.height - 0.64, top + 3.3),
        xytext=(-0.36, top + 3.3),
        arrowprops={
            "arrowstyle": "<->",
            "color": tm.color("reference"),
            "lw": 1.2,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )
    ax.annotate(
        f"{spread:.1f} points separate the happiest and saddest bands",
        xy=(bands.height / 2 - 0.5, top + 3.9),
        ha="center",
        fontsize=tm.size("annotation"),
        color=tm.color("reference"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{row['band']}\n{row['label']}" for row in bands.iter_rows(named=True)],
        fontsize=tm.size("tick"),
    )
    ax.set_xlabel("valence band", fontsize=tm.size("label"))
    ax.set_ylabel("mean popularity (0-100)", fontsize=tm.size("label"))
    ax.set_ylim(0, POPULARITY_LIMIT)
    ax.grid(axis="x", visible=False)
    tm.title(
        ax,
        f"Mood barely predicts popularity: {spread:.1f} points out of 100",
        size=tm.size("title") - 2,
    )

    if with_mode:
        ax2 = np.atleast_1d(axes)[1]
        modes = mode_split(tracks)
        mx = np.arange(modes.height)
        mmeans = modes["mean_valence"].to_numpy()
        mcis = modes["ci_valence"].to_numpy()
        ax2.bar(
            mx,
            mmeans,
            color=[tm.color("accent"), tm.color("accent_soft")],
            width=0.6,
            yerr=mcis,
            capsize=4,
            error_kw={"ecolor": tm.color("text"), "lw": 1.1},
            zorder=2,
        )
        for xi, mean in zip(mx, mmeans):
            ax2.annotate(
                f"{mean:.2f}",
                (xi, mean),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=tm.size("annotation"),
                color=tm.color("text"),
                fontweight="bold",
            )
        gap = float(abs(mmeans[0] - mmeans[-1]))
        ax2.annotate(
            f"{gap:.3f} apart",
            xy=(0.5, 0.93),
            xycoords="axes fraction",
            ha="center",
            fontsize=tm.size("annotation"),
            color=tm.color("text"),
        )
        ax2.set_xticks(mx)
        ax2.set_xticklabels(modes["key_mode"].to_list(), fontsize=tm.size("tick"))
        ax2.set_xlabel("key", fontsize=tm.size("label"))
        ax2.set_ylabel("mean valence", fontsize=tm.size("label"))
        ax2.set_ylim(0, 1)
        ax2.grid(axis="x", visible=False)
        tm.title(ax2, "Major key isn't happier", size=tm.size("title") - 2)

    tm.caption(
        fig,
        "Bars show mean popularity with 95% confidence intervals; the five "
        "valence bands hold 12,000 to 22,000 songs each.",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.855, bottom=0.22)
    tm.save(fig, out)


def main() -> None:
    parser = cli.base_parser(__doc__ or "", DEFAULT_OUT)
    parser.add_argument(
        "--with-mode",
        action="store_true",
        help="add a second panel comparing major and minor keys",
    )
    args = parser.parse_args()
    tm = theme_mod.load(args.theme).apply()
    plot(args.data, args.out, tm, with_mode=args.with_mode)


if __name__ == "__main__":
    main()
