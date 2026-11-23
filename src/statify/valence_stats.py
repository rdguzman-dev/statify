"""Statistics behind the valence charts, plus a small reporting CLI.

Polars does the aggregation so every chart and the printed report agree on
the same numbers. Two frames come out of the CSV:

- `load_tracks` keeps one row per song (84k of the 114k rows are repeats of a
  track listed under a second genre).
- `load_track_genres` keeps one row per (song, genre) pair, for anything that
  compares genres.

Run `python -m statify.valence_stats` (or the `valence-stats` command) to
print every highlight used in the infographic and dump them to JSON/CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "dataset.csv"
DEFAULT_OUT_DIR = ROOT / "stats"

BAND_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
BAND_LABELS = ("0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")
BAND_TITLES = ("very sad", "sad", "mid", "happy", "very happy")

# Audio features that are reasonable to compare against valence.
FEATURES = (
    "danceability",
    "energy",
    "loudness",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness",
    "tempo",
    "popularity",
)


def load_tracks(path: Path | str = DEFAULT_DATA) -> pl.DataFrame:
    return pl.read_csv(path).unique(subset=["track_id"], keep="first")


def load_track_genres(path: Path | str = DEFAULT_DATA) -> pl.DataFrame:
    return pl.read_csv(path).unique(subset=["track_id", "track_genre"], keep="first")


def band_index(values) -> np.ndarray:
    """0-4 index of the valence band each value falls in (upper edge wins)."""
    arr = np.asarray(values, dtype=float)
    idx = np.searchsorted(BAND_EDGES, arr, side="right") - 1
    return np.clip(idx, 0, len(BAND_LABELS) - 1)


def with_band(df: pl.DataFrame) -> pl.DataFrame:
    idx = band_index(df["valence"].to_numpy())
    return df.with_columns(
        pl.Series("band_idx", idx, dtype=pl.Int64),
        pl.Series("band", [BAND_LABELS[i] for i in idx], dtype=pl.String),
    )


def summary(tracks: pl.DataFrame) -> dict:
    v = tracks["valence"]
    return {
        "n_tracks": tracks.height,
        "mean": round(float(v.mean()), 3),
        "median": round(float(v.median()), 3),
        "sd": round(float(v.std()), 3),
        "min": round(float(v.min()), 3),
        "q1": round(float(v.quantile(0.25)), 3),
        "q3": round(float(v.quantile(0.75)), 3),
        "max": round(float(v.max()), 3),
        "share_above_half": round(float((v > 0.5).mean()), 3),
    }


def band_table(tracks: pl.DataFrame) -> pl.DataFrame:
    framed = with_band(tracks)
    counts = (
        framed.group_by("band_idx")
        .agg(
            pl.len().alias("n"),
            pl.col("popularity").mean().alias("mean_popularity"),
            pl.col("popularity").std().alias("sd_popularity"),
            pl.col("danceability").mean().alias("mean_danceability"),
            pl.col("energy").mean().alias("mean_energy"),
        )
        .sort("band_idx")
    )
    labels = pl.DataFrame(
        {
            "band_idx": list(range(len(BAND_LABELS))),
            "band": list(BAND_LABELS),
            "label": list(BAND_TITLES),
        },
        schema={"band_idx": pl.Int64, "band": pl.String, "label": pl.String},
    )
    return (
        counts.join(labels, on="band_idx")
        .with_columns(
            (pl.col("n") / framed.height).alias("share"),
            (1.96 * pl.col("sd_popularity") / pl.col("n").sqrt()).alias("ci_popularity"),
        )
        .select(
            "band_idx",
            "band",
            "label",
            "n",
            "share",
            "mean_popularity",
            "ci_popularity",
            "mean_danceability",
            "mean_energy",
        )
    )


def genre_table(track_genres: pl.DataFrame) -> pl.DataFrame:
    """One row per genre, sorted from lowest to highest mean valence."""
    return (
        track_genres.group_by("track_genre")
        .agg(
            pl.len().alias("n"),
            pl.col("valence").mean().alias("mean_valence"),
            pl.col("valence").median().alias("median_valence"),
            pl.col("valence").std().alias("sd_valence"),
            pl.col("energy").mean().alias("mean_energy"),
            pl.col("popularity").mean().alias("mean_popularity"),
            pl.corr("valence", "popularity").alias("r_popularity"),
        )
        .sort("mean_valence")
    )


def variance_share(track_genres: pl.DataFrame) -> dict:
    """How much of the spread in valence is between genres vs within them."""
    overall_mean = float(track_genres["valence"].mean())
    stats = track_genres.group_by("track_genre").agg(
        pl.len().alias("n"), pl.col("valence").mean().alias("m")
    )
    between = float((((stats["m"] - overall_mean) ** 2) * stats["n"]).sum())
    total = float((((track_genres["valence"] - overall_mean) ** 2)).sum())
    return {
        "between_genre_share": round(between / total, 3),
        "within_genre_sd_median": round(
            float(
                track_genres.group_by("track_genre")
                .agg(pl.col("valence").std())["valence"].median()
            ),
            3,
        ),
        "overall_sd": round(float(track_genres["valence"].std()), 3),
    }


def correlations(tracks: pl.DataFrame) -> pl.DataFrame:
    r = tracks.select(
        [pl.corr("valence", feature).alias(feature) for feature in FEATURES]
    ).transpose(include_header=True, header_name="feature", column_names=["r"])
    return r.with_columns(pl.col("r").abs().alias("abs_r")).sort("abs_r", descending=True)


def mood_quadrants(tracks: pl.DataFrame, valence_cut: float | None = None,
                   energy_cut: float | None = None) -> pl.DataFrame:
    v_cut = float(tracks["valence"].median()) if valence_cut is None else valence_cut
    e_cut = float(tracks["energy"].median()) if energy_cut is None else energy_cut
    framed = tracks.with_columns(
        pl.when(pl.col("valence") >= v_cut)
        .then(pl.lit("happy"))
        .otherwise(pl.lit("sad"))
        .alias("mood"),
        pl.when(pl.col("energy") >= e_cut)
        .then(pl.lit("intense"))
        .otherwise(pl.lit("calm"))
        .alias("drive"),
    )
    return (
        framed.group_by(["mood", "drive"])
        .agg(
            pl.len().alias("n"),
            pl.col("danceability").mean().alias("mean_danceability"),
            pl.col("acousticness").mean().alias("mean_acousticness"),
            pl.col("popularity").mean().alias("mean_popularity"),
        )
        .with_columns((pl.col("n") / framed.height).alias("share"))
        .with_columns(
            (pl.col("mood") + "-" + pl.col("drive")).alias("quadrant")
        )
        .select("quadrant", "mood", "drive", "n", "share",
                "mean_danceability", "mean_acousticness", "mean_popularity")
        .sort("n", descending=True)
    ).with_columns(
        pl.lit(v_cut).alias("valence_cut"),
        pl.lit(e_cut).alias("energy_cut"),
    )


def mode_split(tracks: pl.DataFrame) -> pl.DataFrame:
    return (
        tracks.group_by("mode")
        .agg(
            pl.len().alias("n"),
            pl.col("valence").mean().alias("mean_valence"),
            pl.col("valence").std().alias("sd_valence"),
        )
        .with_columns(
            pl.when(pl.col("mode") == 1)
            .then(pl.lit("major"))
            .otherwise(pl.lit("minor"))
            .alias("key_mode"),
            (1.96 * pl.col("sd_valence") / pl.col("n").sqrt()).alias("ci_valence"),
        )
        .select("key_mode", "n", "mean_valence", "ci_valence")
        .sort("key_mode")
    )


def popular_split(tracks: pl.DataFrame, cut: int = 70) -> dict:
    hi = tracks.filter(pl.col("popularity") >= cut)["valence"]
    lo = tracks.filter(pl.col("popularity") < cut)["valence"]
    return {
        "cut": cut,
        "n_above": hi.len(),
        "mean_above": round(float(hi.mean()), 3),
        "mean_below": round(float(lo.mean()), 3),
    }


def within_genre_r(track_genres: pl.DataFrame) -> dict:
    centered = track_genres.with_columns(
        (pl.col("valence") - pl.col("valence").mean().over("track_genre")).alias("v"),
        (pl.col("energy") - pl.col("energy").mean().over("track_genre")).alias("e"),
        (pl.col("popularity") - pl.col("popularity").mean().over("track_genre")).alias("p"),
    )
    return {
        "valence_popularity": round(float(centered.select(pl.corr("v", "p")).item()), 3),
        "valence_energy": round(float(centered.select(pl.corr("v", "e")).item()), 3),
    }


def report(path: Path | str = DEFAULT_DATA) -> dict:
    tracks = load_tracks(path)
    pairs = load_track_genres(path)
    genres = genre_table(pairs)
    bands = band_table(tracks)
    corr = correlations(tracks)
    share = variance_share(pairs)

    out = {
        "summary": summary(tracks),
        "bands": bands.to_dicts(),
        "correlations": corr.to_dicts(),
        "variance_share": share,
        "within_genre_r": within_genre_r(pairs),
        "mood_quadrants": mood_quadrants(tracks).to_dicts(),
        "mode": mode_split(tracks).to_dicts(),
        "popularity_split": popular_split(tracks),
        "genre_lowest": genres.head(8).select("track_genre", "n", "mean_valence").to_dicts(),
        "genre_highest": genres.tail(8).select("track_genre", "n", "mean_valence").to_dicts(),
        "genre_spread": round(
            float(genres["mean_valence"].max() - genres["mean_valence"].min()), 3
        ),
    }
    _print(out)
    return out


def _print(out: dict) -> None:
    s = out["summary"]
    print(f"unique tracks: {s['n_tracks']:,}")
    print(
        f"valence mean {s['mean']:.3f}, median {s['median']:.3f}, sd {s['sd']:.3f}, "
        f"IQR {s['q1']:.3f}-{s['q3']:.3f}, range {s['min']:.3f}-{s['max']:.3f}"
    )
    print(f"share of songs above 0.5: {s['share_above_half']:.1%}")
    print("\nby band:")
    for row in out["bands"]:
        print(
            f"  {row['band']:<10} {row['label']:<10} {row['share']:>6.1%}  "
            f"n={row['n']:>6,}  mean popularity {row['mean_popularity']:.2f}"
        )
    print("\ncorrelation with valence:")
    for row in out["correlations"]:
        print(f"  {row['feature']:<17} r = {row['r']:+.3f}")
    print(
        f"\nbetween-genre share of valence variance: "
        f"{out['variance_share']['between_genre_share']:.1%}"
    )
    print(
        f"median within-genre sd: {out['variance_share']['within_genre_sd_median']} "
        f"(overall sd {out['variance_share']['overall_sd']})"
    )
    print(
        "within-genre r: valence-popularity "
        f"{out['within_genre_r']['valence_popularity']:+.3f}, valence-energy "
        f"{out['within_genre_r']['valence_energy']:+.3f}"
    )
    print("\nmood quadrants:")
    for row in out["mood_quadrants"]:
        print(f"  {row['quadrant']:<16} {row['share']:>6.1%}  n={row['n']:>6,}")
    print("\nmajor vs minor:")
    for row in out["mode"]:
        print(f"  {row['key_mode']:<6} n={row['n']:>6,}  mean valence {row['mean_valence']:.3f}")
    p = out["popularity_split"]
    print(
        f"\npopularity >= {p['cut']}: {p['n_above']:,} songs, mean valence "
        f"{p['mean_above']:.3f} vs {p['mean_below']:.3f} for the rest"
    )
    print("\nlowest valence genres:")
    for row in out["genre_lowest"]:
        print(f"  {row['track_genre']:<16} {row['mean_valence']:.3f}  n={row['n']}")
    print("highest valence genres:")
    for row in out["genre_highest"]:
        print(f"  {row['track_genre']:<16} {row['mean_valence']:.3f}  n={row['n']}")
    print(f"\ngenre mean spread: {out['genre_spread']:.3f} on a 0-1 scale")


def write(out_dir: Path, path: Path | str = DEFAULT_DATA) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = report(path)
    (out_dir / "valence_stats.json").write_text(json.dumps(out, indent=2) + "\n")
    tracks = load_tracks(path)
    band_table(tracks).write_csv(out_dir / "valence_bands.csv")
    genre_table(load_track_genres(path)).write_csv(out_dir / "valence_genres.csv")
    correlations(tracks).write_csv(out_dir / "valence_correlations.csv")
    print(f"\nwrote {out_dir}/valence_stats.json plus three CSV tables")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    write(args.out_dir, args.data)


if __name__ == "__main__":
    main()
