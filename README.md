# Statify

Exploratory data analysis of the [Spotify Tracks Dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
## Data

`data/dataset.csv`, 114,000 rows and 114 genres. Each genre contributes about
1,000 tracks, and a track that fits two genres appears twice, so per-song
statistics use `track_id` deduplication (`89,741` unique songs) while
per-genre statistics use `(track_id, track_genre)` pairs.

## Running the charts

```bash
uv run valence-stats                          # print and save every number
uv run valence-genre                          # chart 1
uv run valence-distribution                   # chart 2
uv run valence-mood-map                       # chart 3
uv run valence-danceability                   # chart 4
uv run valence-popularity --with-mode         # chart 5
```

Without `uv`, `.venv/bin/python -m statify.valence_genre` works the same.
Every script also takes `--data`, `--out` and `--theme`.

| Script | Output | Question it answers |
| --- | --- | --- |
| `valence_genre_rank.py` | `valence_genre_rank.png` | Which genres are happiest and saddest? |
| `valence_distribution.py` | `valence_distribution.png` | How are songs spread across the mood scale? |
| `valence_mood_map.py` | `valence_mood_map.png` | How do mood and energy combine? |
| `valence_danceability.py` | `valence_danceability.png` | Which audio feature tracks mood best? |
| `valence_popularity.py` | `valence_popularity.png` | Does mood predict popularity? |
| `valence_stats.py` | `stats/*.json`, `stats/*.csv` | All of the numbers above, checked in one place |

## Restyling

All five charts read their colours, fonts, spacing and figure sizes from
`src/statify/themes/default.json`. Keys fall back to the default, so
we can use a two-line override instead of copying the whole file:

### Example

```json
{
  "colors": { "accent": "#B5179E", "accent_soft": "#F1C0E8" },
  "font": { "family": ["Avenir Next", "DejaVu Sans"], "title": 18 }
}
```

Save that as `src/statify/themes/sakura.json` and run
`uv run valence-genre --theme sakura`, or pass any path.

### Configuration

What lives where:

- `colors.cmap` sets the density colours on the mood map. It accepts a
  matplotlib colormap name like `"Blues"` or a list of colours from light to
  dark, for example `["#FFF8F0", "#B5179E"]`.
- `colors.accent` and `colors.accent_soft` drive the main series and its
  confidence band, `colors.bands` is the sad-to-happy ramp used by the
  distribution and popularity bars, `colors.quadrants` colours the mood-map
  chips, and `colors.neutral`, `colors.highlight` and `colors.reference` cover
  secondary marks and guide lines.
- `font.*` covers family and size for title, subtitle, labels, ticks,
  annotations, captions and legends.
- `layout.dpi` sets export resolution, `layout.grid` and `layout.spines` turn
  grid lines and plot borders on or off, `layout.title_align` switches titles
  between left and centre.
- `layout.figure_sizes.<chart>` sets the width and height per chart, where
  `<chart>` is `genre_rank`, `distribution`, `mood_map`, `danceability` or
  `popularity`.
- `credit` is the source line printed under every chart.

Titles wrap automatically at the figure edge, so raising `font.title` a few
points is safe.

Chip and label text picks black or white automatically based on the background
colour (`Theme.text_on`), so swapping the palette does not create unreadable
labels.
