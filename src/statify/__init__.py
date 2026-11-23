"""Statify: exploratory data analysis of the Spotify Tracks Dataset."""

from __future__ import annotations


def main() -> None:
    """Run the statistics report used by every valence chart."""
    from .valence_stats import main as run

    run()
