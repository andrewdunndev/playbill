# Playbill — Video Program Assembler

## Project Overview

TOML-driven video program assembler. Reads `program.toml`, renders each
segment to a temp file in `output/segments/`, then concatenates them with
fade-to-black transitions via ffmpeg into a single output video.

Designed for event programs, screening nights, or any sequenced video
playlist that needs title cards, intros, and transitions between clips.

## Project Structure

```
playbill/
  assemble.py         # Main assembler (reads program.toml)
  program.toml        # Declarative segment list
  fonts/              # Bundled EB Garamond (OFL licensed)
  input/              # User-provided assets (gitignored)
    intros/           # Presenter/intro video clips
    episodes/         # Full episodes or video files
    scenes/           # Short clips or highlights
    music/            # Background music for title cards
    photos/           # Photos for credits scroll
    friends/          # Friend video/photo quotes for credits
  output/             # Generated artifacts (gitignored)
    segments/         # Temp per-segment renders
  Makefile            # Build targets
  pyproject.toml      # Python packaging + ruff config
  requirements.txt    # Python dependencies (numpy, Pillow)
```

## Build Commands

```bash
make venv             # Create virtual environment
make render           # Full assembly
make preview          # Preview mode (~15s per segment)
make list             # List segments and exit
make segment N=3      # Render only segment 3
make clean            # Remove output/
make clean-all        # Remove output/ + .venv/
```

## Segment Types

| Type | Description |
|------|-------------|
| `title` | Text card — white on black, configurable lines, fade in/out. Optional background music via `[output].title_music`. |
| `intro` | Portrait video (left half) + title/subtitle/description text (right half). |
| `episode` | Pass-through video file, scaled/padded to output resolution. Supports `trim_end` and `outro_duration`. |
| `scene` | Same as episode — semantic distinction for organization. |
| `quote` | Photo or video (left) + name/quote text (right). |
| `credits` | Scrolling photo wall + friend quotes (video/photo) + background music with audio ducking. |

## Dependencies

- Python 3.11+
- Pillow (PIL) — image/text rendering
- NumPy — array operations for frame rendering
- ffmpeg 7+ — video encoding, audio mixing
- ffprobe — duration detection (bundled with ffmpeg)

## Key Files

- `assemble.py` — single-file assembler. All rendering functions, ffmpeg
  orchestration, and CLI handling. ~1450 lines.
- `program.toml` — declarative segment list. Each `[[segment]]` has a type
  and type-specific config. See the example for all supported fields.
- `fonts/EBGaramond.ttf` — bundled serif font (SIL OFL). Falls back to
  system Georgia, then PIL default.
