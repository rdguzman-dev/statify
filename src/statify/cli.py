"""Small argparse helper so all five chart scripts take the same flags."""

from __future__ import annotations

import argparse
from pathlib import Path

from .valence_stats import DEFAULT_DATA


def base_parser(description: str, default_out: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="dataset CSV")
    parser.add_argument("--out", type=Path, default=default_out, help="output PNG")
    parser.add_argument(
        "--theme",
        type=Path,
        default=None,
        help="theme JSON; bare names like 'warm' resolve inside themes/",
    )
    return parser
