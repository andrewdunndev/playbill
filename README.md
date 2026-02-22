# Playbill

TOML-driven video program assembler. Define your program in `program.toml`,
drop video files into `input/`, and Playbill renders title cards, intros,
episode pass-throughs, and credits into a single concatenated video.

## Quick Start

```bash
make venv
# Edit program.toml — see the example for all segment types
# Drop your video/audio/photo files into input/
make preview          # Fast preview (~15s per segment)
make render           # Full assembly
```

Output: `output/playbill.mp4`

## How It Works

1. Each `[[segment]]` in `program.toml` is rendered to a temp file
2. All segments are concatenated with fade-to-black transitions
3. If segments have `act` fields, per-act files are also produced

## Segment Types

**`title`** — Text card (white on black). Set `lines`, `duration`, `fade_in`,
`fade_out`. Optional background music via `[output].title_music`.

**`intro`** — Portrait video on the left, title/subtitle/description on the
right. Point `video` at a clip file.

**`episode`** / **`scene`** — Pass-through video, scaled and padded to output
resolution. Supports `trim_end` to cut early and `outro_duration` for black
padding. HDR content is automatically tonemapped.

**`quote`** — Photo or video on the left, name and quote text on the right.
Use `photo` for still images or `video` for clips.

**`credits`** — Scrolling photo wall with friend quotes interspersed.
Background music with automatic audio ducking when video quotes play.

## Program Format

```toml
[output]
width = 1920
height = 1080
fps = 30
# title_music = "input/music/background.mp3"
# title_music_volume = "-22dB"

[[segment]]
type = "title"
lines = ["Your Event", "A Subtitle"]
duration = 6.0

[[segment]]
type = "episode"
file = "input/episodes/show.mkv"
outro_duration = 3.0
```

See `program.toml` for a complete example with all segment types.

## Input Structure

```
input/
  intros/       # Presenter video clips (.mov, .mp4)
  episodes/     # Full video files (.mkv, .mp4)
  scenes/       # Short clips (.mkv, .mp4)
  music/        # Background music (.mp3, .m4a)
  photos/       # Photos for credits scroll (.jpg, .png)
  friends/      # Friend video/photo quotes
```

## CLI Options

```
python assemble.py                  # Full assembly
python assemble.py --preview        # Truncate long segments to ~15s
python assemble.py --segment 3      # Render only segment 3
python assemble.py --skip-render    # Concat only (reuse existing temps)
python assemble.py --list           # List segments and exit
```

## Requirements

- Python 3.11+
- ffmpeg 7+
- `pip install numpy Pillow`

## License

MIT
