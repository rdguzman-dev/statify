"""Look and feel for the valence charts, loaded from a JSON file.

Every chart script accepts `--theme path/to/theme.json`. The file only needs
the keys you want to change; everything else falls back to
`themes/default.json`. So a restyle is usually a few hex codes and font sizes
in one file, and every chart picks them up.

Keys are addressed with dots, e.g. `colors.accent`, `font.title`,
`layout.figure_sizes.mood_map`.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

THEMES_DIR = Path(__file__).resolve().parent / "themes"
DEFAULT_THEME_PATH = THEMES_DIR / "default.json"


def _merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve(path: Path | str | None) -> Path:
    """Accept a file path, or a bare name looked up in themes/."""
    if path is None:
        return DEFAULT_THEME_PATH
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    named = THEMES_DIR / f"{candidate.stem or candidate.name}.json"
    if named.is_file():
        return named
    raise FileNotFoundError(f"no theme file at {candidate} or {named}")


@dataclass(frozen=True)
class Theme:
    data: dict[str, Any]
    source: Path

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def color(self, name: str) -> str:
        return str(self.get(f"colors.{name}", "#333333"))

    def ramp(self, name: str) -> list[str]:
        return list(self.get(f"colors.{name}", []))

    def colormap(self, name: str = "cmap"):
        """A matplotlib colormap name, or one built from a colour list."""
        value = self.get(f"colors.{name}", "Blues")
        if isinstance(value, list):
            from matplotlib.colors import LinearSegmentedColormap

            return LinearSegmentedColormap.from_list(name, value)
        return value

    def text_on(self, background: str) -> str:
        """Dark or white text, whichever stays readable on `background`."""
        r, g, b = (int(background[i : i + 2], 16) / 255 for i in (1, 3, 5))
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "#FFFFFF" if luma < 0.55 else self.color("text")

    def size(self, name: str, default: float = 10.0) -> float:
        return float(self.get(f"font.{name}", default))

    @property
    def dpi(self) -> float:
        return float(self.get("layout.dpi", 200))

    def figure_size(self, chart: str, fallback: tuple[float, float]) -> tuple[float, float]:
        size = self.get(f"layout.figure_sizes.{chart}", fallback)
        return (float(size[0]), float(size[1]))

    def rc(self) -> dict[str, Any]:
        spines = bool(self.get("layout.spines", False))
        return {
            "figure.facecolor": self.color("background"),
            "savefig.facecolor": self.color("background"),
            "savefig.pad_inches": 0.2,
            "axes.facecolor": self.color("axes_background"),
            "font.family": self.get("font.family", ["DejaVu Sans"]),
            "font.size": self.size("label", 11),
            "text.color": self.color("text"),
            "axes.labelcolor": self.color("text"),
            "axes.titlecolor": self.color("text"),
            "axes.edgecolor": self.color("grid"),
            "axes.linewidth": 1.0,
            "axes.grid": bool(self.get("layout.grid", True)),
            "grid.color": self.color("grid"),
            "grid.linewidth": 0.9,
            "grid.alpha": 1.0,
            "axes.spines.top": spines,
            "axes.spines.right": spines,
            "xtick.color": self.color("muted_text"),
            "ytick.color": self.color("muted_text"),
            "xtick.labelsize": self.size("tick"),
            "ytick.labelsize": self.size("tick"),
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.fontsize": self.size("legend"),
            "legend.frameon": False,
        }

    def apply(self) -> "Theme":
        style = "whitegrid" if self.get("layout.grid", True) else "white"
        sns.set_theme(style=style, rc=self.rc())
        return self

    def title(self, ax, text: str, *, size: float | None = None) -> None:
        ax.set_title(
            text,
            loc=self.get("layout.title_align", "left"),
            fontsize=self.size("title", 16) if size is None else size,
            fontweight=self.get("font.title_weight", "bold"),
            color=self.color("text"),
            pad=12,
            wrap=True,
        )

    def heading(self, fig, text: str, y: float = 0.985) -> None:
        align = self.get("layout.title_align", "left")
        fig.text(
            0.01 if align == "left" else 0.5,
            y,
            text,
            ha="left" if align == "left" else "center",
            va="top",
            fontsize=self.size("title", 16),
            fontweight=self.get("font.title_weight", "bold"),
            color=self.color("text"),
            wrap=True,
        )

    def caption(self, fig, notes: str | None = None, width: int = 96) -> None:
        """Footnote plus credit, wrapped and centred along the bottom edge."""
        lines: list[str] = []
        if notes:
            lines.extend(textwrap.wrap(notes, width=width))
        credit = self.get("credit", "")
        if credit:
            lines.append(credit)
        for offset, line in enumerate(reversed(lines)):
            fig.text(
                0.5,
                0.008 + offset * 0.032,
                line,
                ha="center",
                va="bottom",
                fontsize=self.size("caption", 9),
                color=self.color("muted_text"),
            )

    def fit_top(self, fig, extra: float = 6.0) -> None:
        """Pull the axes down so a wrapped figure heading never overlaps them.

        Call after the figure-level heading and the subplot margins are set.
        Only headings in the top quarter of the figure count, so captions at
        the bottom cannot collapse the plot area.
        """
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        height = float(fig.bbox.height)
        bottoms = []
        for text in fig.texts:
            if not text.get_text().strip():
                continue
            box = text.get_window_extent(renderer)
            if box.y0 / height > 0.75:
                bottoms.append(box.y0)
        if not bottoms:
            return
        reserve = 0.0
        for ax in fig.axes:
            if ax.get_title():
                title = ax.title.get_window_extent(renderer)
                reserve = max(reserve, title.height + 14 * fig.dpi / 72)
        top = (min(bottoms) - reserve - extra * fig.dpi / 72) / height
        fig.subplots_adjust(top=min(fig.subplotpars.top, top))

    def save(self, fig, out: Path) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out,
            dpi=self.dpi,
            transparent=bool(self.get("layout.transparent", False)),
        )
        plt.close(fig)
        print(f"wrote {out}")
        return out


def load(path: Path | str | None = None) -> Theme:
    source = resolve(path)
    with DEFAULT_THEME_PATH.open() as handle:
        data = json.load(handle)
    if source != DEFAULT_THEME_PATH:
        with source.open() as handle:
            data = _merge(data, json.load(handle))
    return Theme(data=data, source=source)
