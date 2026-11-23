"""Chart 3: the mood map.

Valence against energy, split at both medians into four named quadrants.
Hexbins show where songs cluster and each quadrant label carries its share of
the catalogue in the quadrant's own colour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from . import cli, theme as theme_mod
from .valence_stats import load_tracks, mood_quadrants

DEFAULT_OUT = Path(__file__).resolve().with_suffix(".png")

# (valence side, energy side) -> (label, colour index in colors.quadrants)
QUADRANTS = {
    (0, 0): ("calm & sad", 2),
    (1, 0): ("calm & happy", 3),
    (0, 1): ("intense & sad", 0),
    (1, 1): ("intense & happy", 1),
}


def plot(path: Path, out: Path, tm: theme_mod.Theme) -> None:
    tracks = load_tracks(path)
    valence = tracks["valence"].to_numpy()
    energy = tracks["energy"].to_numpy()
    v_cut = float(np.median(valence))
    e_cut = float(np.median(energy))
    quadrants = mood_quadrants(tracks, v_cut, e_cut)
    colors = tm.ramp("quadrants")

    fig, ax = plt.subplots(figsize=tm.figure_size("mood_map", (9.5, 8.5)))
    ax.hexbin(
        valence,
        energy,
        gridsize=34,
        mincnt=1,
        cmap=tm.colormap(),
        linewidths=0,
        zorder=2,
    )
    ax.axvline(v_cut, color=tm.color("text"), lw=1.2, ls="--", zorder=3)
    ax.axhline(e_cut, color=tm.color("text"), lw=1.2, ls="--", zorder=3)

    for (v_side, e_side), (name, color_idx) in QUADRANTS.items():
        share = quadrants.filter(
            (quadrants["mood"] == ("happy" if v_side else "sad"))
            & (quadrants["drive"] == ("intense" if e_side else "calm"))
        )["share"][0]
        chip = colors[color_idx]
        ax.text(
            0.96 if v_side else 0.04,
            0.95 if e_side else 0.05,
            f"{name}  {share:.0%}",
            transform=ax.transAxes,
            ha="right" if v_side else "left",
            va="top" if e_side else "bottom",
            fontsize=tm.size("subtitle"),
            color=tm.text_on(chip),
            fontweight="bold",
            zorder=4,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": chip,
                "edgecolor": "none",
            },
        )

    tm.title(ax, "Songs split almost evenly across the four moods")
    ax.set_xlabel("valence  (0 = sad, 1 = happy)", fontsize=tm.size("label"))
    ax.set_ylabel("energy  (0 = calm, 1 = intense)", fontsize=tm.size("label"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(False)

    tm.caption(
        fig,
        f"Quadrants split at median valence ({v_cut:.2f}) and median energy "
        f"({e_cut:.2f}); darker cells hold more songs. Shares are of the "
        f"whole catalogue.",
    )
    fig.subplots_adjust(left=0.10, right=0.985, top=0.93, bottom=0.16)
    tm.save(fig, out)


def main() -> None:
    args = cli.base_parser(__doc__ or "", DEFAULT_OUT).parse_args()
    tm = theme_mod.load(args.theme).apply()
    plot(args.data, args.out, tm)


if __name__ == "__main__":
    main()
