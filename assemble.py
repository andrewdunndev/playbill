#!/usr/bin/env python3
"""
Playbill — Video Program Assembler

Reads program.toml and assembles a complete video program from:
  - Title cards (text on black)
  - Intro cards (portrait video + text overlay)
  - Episode pass-through (scaled/padded to output resolution)
  - Scene clips (scaled/padded to output resolution)
  - External video pass-through (e.g. photo wall output)

Each segment is rendered to a temporary file, then concatenated with
fade-to-black transitions via ffmpeg.

Usage:
    python assemble.py                  # Full assembly
    python assemble.py --preview        # Preview: truncate long segments to ~15s
    python assemble.py --segment 3      # Render only segment 3 (0-indexed)
    python assemble.py --skip-render    # Concat only (reuse existing temps)
    python assemble.py --list           # List segments and exit
"""

import subprocess
import sys
import threading
from pathlib import Path

import tomllib

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).parent
FONTS_DIR = ROOT / "fonts"
OUTPUT_DIR = ROOT / "output"
TEMP_DIR = OUTPUT_DIR / "segments"

AUDIO_LAYOUT = "5.1"
AUDIO_RATE = 48000
AUDIO_CODEC = "pcm_s24le"
AFORMAT = f"aformat=sample_fmts=s32:sample_rates={AUDIO_RATE}:channel_layouts={AUDIO_LAYOUT}"
ANULLSRC = f"anullsrc=r={AUDIO_RATE}:cl={AUDIO_LAYOUT}"
AUDIO_ENC = ["-c:a", AUDIO_CODEC]


def load_program(path=None):
    path = path or ROOT / "program.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


def find_font(size):
    font_paths = [
        FONTS_DIR / "EBGaramond.ttf",
        Path("/System/Library/Fonts/Supplemental/Georgia.ttf"),
    ]
    for fp in font_paths:
        if fp.exists():
            try:
                return ImageFont.truetype(str(fp), size)
            except OSError:
                continue
    return ImageFont.load_default()


def probe_duration(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path)
    ]).decode().strip()
    return float(out)


def probe_video_info(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "csv=p=0", str(path)
    ]).decode().strip()
    parts = out.split(",")
    w, h = int(parts[0]), int(parts[1])
    num, den = parts[2].split("/")
    fps = int(num) / int(den)
    return w, h, fps


def probe_is_hdr(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=color_transfer",
        "-of", "csv=p=0", str(path)
    ]).decode().strip().rstrip(",")
    return out in ("smpte2084", "arib-std-b67")


HDR_OUTPUT_FLAGS = ["-bsf:v", "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1"]
HDR_INIT_HW = ["-init_hw_device", "videotoolbox=vt"]


_title_music_path = None
_title_music_volume = "-22dB"
_title_music_offset = 0.0


def _init_title_music(output_cfg):
    global _title_music_path, _title_music_volume
    path = output_cfg.get("title_music")
    if path:
        p = ROOT / path
        if p.exists():
            _title_music_path = p
    _title_music_volume = output_cfg.get("title_music_volume", "-22dB")


def render_title_card(seg, output_cfg, out_path, music_fade_in=True, music_fade_out=True):
    global _title_music_offset
    w = output_cfg["width"]
    h = output_cfg["height"]
    fps = output_cfg["fps"]
    duration = seg.get("duration", 6.0)
    fade_in = seg.get("fade_in", 2.0)
    fade_out = seg.get("fade_out", 2.0)
    lines = seg.get("lines", [])
    total_frames = int(duration * fps)

    font = find_font(80)
    small_font = find_font(50)

    base_img = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(base_img)

    line_data = []
    for i, line in enumerate(lines):
        f = font if i == 0 else small_font
        if not line:
            line_data.append((line, f, 0, 0))
            continue
        bb = draw.textbbox((0, 0), line, font=f)
        lw, lh = bb[2] - bb[0], bb[3] - bb[1]
        line_data.append((line, f, lw, lh))

    spacing = 20
    visible = [d for d in line_data if d[0]]
    block_h = sum(d[3] for d in visible) + spacing * (len(visible) - 1)
    empty_h = 40
    total_block = block_h + empty_h * sum(1 for d in line_data if not d[0])
    y = (h - total_block) // 2

    for line, f, lw, lh in line_data:
        if not line:
            y += empty_h
            continue
        x = (w - lw) // 2
        draw.text((x, y), line, fill=(255, 255, 255), font=f)
        y += lh + spacing

    base_arr = np.array(base_img, dtype=np.float32)

    use_music = _title_music_path is not None
    if use_music:
        music_offset = _title_music_offset
        afades = ""
        if music_fade_in:
            afades += f"afade=t=in:st=0:d={fade_in},"
        if music_fade_out:
            afades += f"afade=t=out:st={duration - fade_out}:d={fade_out},"
        audio_filter = (
            f"[1:a]atrim={music_offset}:{music_offset + duration},asetpts=PTS-STARTPTS,"
            f"volume={_title_music_volume},"
            f"{afades}"
            f"pan=5.1|FL=FL|FR=FR|FC=0.5*FL+0.5*FR|LFE=0.25*FL+0.25*FR|BL=0.5*FL|BR=0.5*FR,"
            f"{AFORMAT}[aout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "pipe:0",
            "-i", str(_title_music_path),
            "-filter_complex", audio_filter,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            *AUDIO_ENC,
            "-map", "0:v", "-map", "[aout]",
            "-t", str(duration),
            str(out_path),
        ]
        _title_music_offset = music_offset + duration
    else:
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "pipe:0",
            "-f", "lavfi", "-i", f"{ANULLSRC}:d={duration}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            *AUDIO_ENC,
            "-map", "0:v", "-map", "1:a",
            "-t", str(duration),
            str(out_path),
        ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    fade_in_frames = int(fade_in * fps)
    fade_out_frames = int(fade_out * fps)
    black = bytes(w * h * 3)

    for i in range(total_frames):
        alpha = 1.0
        if i < fade_in_frames:
            alpha = i / fade_in_frames
        elif i >= total_frames - fade_out_frames:
            alpha = (total_frames - i) / fade_out_frames
        if alpha <= 0:
            proc.stdin.write(black)
        elif alpha >= 1.0:
            proc.stdin.write(base_img.tobytes())
        else:
            frame = (base_arr * alpha).astype(np.uint8)
            proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed for title card: {stderr[-2000:]}")
    print(f"    Title card -> {out_path}")


def render_intro(seg, output_cfg, out_path):
    w = output_cfg["width"]
    h = output_cfg["height"]
    fps = output_cfg["fps"]
    video_path = ROOT / seg["video"]

    if not Path(video_path).exists():
        print(f"    WARNING: {video_path} not found, skipping intro")
        return False

    src_w, src_h, src_fps = probe_video_info(video_path)
    duration = probe_duration(video_path)

    video_display_h = h
    video_display_w = int(src_w * (h / src_h))
    if video_display_w > w * 0.45:
        video_display_w = int(w * 0.45)
        video_display_h = int(src_h * (video_display_w / src_w))

    text_area_x = video_display_w + 80
    text_area_w = w - text_area_x - 80

    title_font = find_font(90)
    subtitle_font = find_font(60)
    desc_font = find_font(40)

    text_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_overlay)

    text_y = h // 3

    title = seg.get("title", "")
    if title:
        draw.text((text_area_x, text_y), title, fill=(255, 255, 255), font=title_font)
        bb = draw.textbbox((0, 0), title, font=title_font)
        text_y += (bb[3] - bb[1]) + 30

    subtitle = seg.get("subtitle", "")
    if subtitle:
        draw.text((text_area_x, text_y), subtitle, fill=(200, 200, 200), font=subtitle_font)
        bb = draw.textbbox((0, 0), subtitle, font=subtitle_font)
        text_y += (bb[3] - bb[1]) + 40

    description = seg.get("description", "")
    if description:
        words = description.split()
        lines = []
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            bb = draw.textbbox((0, 0), test, font=desc_font)
            if bb[2] - bb[0] > text_area_w and current:
                lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)
        for line in lines:
            draw.text((text_area_x, text_y), line, fill=(180, 180, 180), font=desc_font)
            bb = draw.textbbox((0, 0), line, font=desc_font)
            text_y += (bb[3] - bb[1]) + 10

    text_overlay_path = TEMP_DIR / "text_overlay.png"
    text_overlay.save(str(text_overlay_path))

    fade_dur = 1.0

    filter_complex = (
        f"[0:v]scale={video_display_w}:{video_display_h}:force_original_aspect_ratio=decrease,"
        f"pad={video_display_w}:{video_display_h}:(ow-iw)/2:(oh-ih)/2:black,"
        f"setsar=1[vid];"
        f"color=black:s={w}x{h}:r={fps}:d={duration}[bg];"
        f"[bg][vid]overlay=0:(main_h-overlay_h)/2[bv];"
        f"[1:v]format=rgba[txt];"
        f"[bv][txt]overlay=0:0,"
        f"fade=t=in:st=0:d={fade_dur},"
        f"fade=t=out:st={duration - fade_dur}:d={fade_dur}[vout];"
        f"[0:a]{AFORMAT},"
        f"afade=t=in:st=0:d={fade_dur},"
        f"afade=t=out:st={duration - fade_dur}:d={fade_dur}[aout]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(text_overlay_path),
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        *AUDIO_ENC,
        "-r", str(fps),
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"    Intro -> {out_path}")
    return True


def render_passthrough(seg, output_cfg, out_path, file_key="file",
                       preview_duration=None):
    w = output_cfg["width"]
    h = output_cfg["height"]
    fps = output_cfg["fps"]
    src = ROOT / seg[file_key]

    if not src.exists():
        print(f"    WARNING: {src} not found, skipping")
        return False

    fade_dur = 1.0
    full_duration = probe_duration(src)
    trim_end = seg.get("trim_end")
    if trim_end and trim_end < full_duration:
        full_duration = trim_end
    outro = seg.get("outro_duration", 0)
    hdr = probe_is_hdr(src)

    if preview_duration and full_duration > preview_duration + 5:
        return _render_preview_passthrough(
            src, w, h, fps, fade_dur, full_duration, preview_duration, out_path,
            outro_duration=outro, hdr=hdr
        )

    total_duration = full_duration + outro

    scale_filters = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,"
        f"setsar=1,fps={fps}"
    )

    if hdr:
        vt_tonemap = (
            f"format=nv12,hwupload,"
            f"scale_vt=w={w}:h={h}"
            f":color_matrix=bt709:color_primaries=bt709:color_transfer=bt709,"
            f"hwdownload,format=nv12"
        )
        video_pre = (
            f"[0:v]{vt_tonemap},"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1,fps={fps}"
        )
    else:
        video_pre = f"[0:v]{scale_filters}"

    if outro > 0:
        total_duration = full_duration + outro
        filter_complex = (
            f"{video_pre},"
            f"fade=t=in:st=0:d={fade_dur},"
            f"fade=t=out:st={full_duration - fade_dur}:d={fade_dur}[main];"
            f"color=black:s={w}x{h}:r={fps}:d={outro}[black];"
            f"[main][black]concat=n=2:v=1:a=0[vout];"
            f"[0:a]{AFORMAT},"
            f"afade=t=in:st=0:d={fade_dur},"
            f"afade=t=out:st={full_duration - fade_dur}:d={fade_dur}[amain];"
            f"{ANULLSRC}:d={outro}[asilent];"
            f"[amain][asilent]concat=n=2:v=0:a=1[aout]"
        )
    else:
        filter_complex = (
            f"{video_pre},"
            f"fade=t=in:st=0:d={fade_dur},"
            f"fade=t=out:st={full_duration - fade_dur}:d={fade_dur}[vout];"
            f"[0:a]{AFORMAT},"
            f"afade=t=in:st=0:d={fade_dur},"
            f"afade=t=out:st={full_duration - fade_dur}:d={fade_dur}[aout]"
        )

    trim_flags = ["-t", str(full_duration)] if trim_end else []
    cmd = [
        "ffmpeg", "-y",
        *(HDR_INIT_HW if hdr else []),
        *trim_flags,
        "-i", str(src),
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        *AUDIO_ENC,
        *(HDR_OUTPUT_FLAGS if hdr else []),
        "-r", str(fps),
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    outro_label = f" + {outro}s black" if outro > 0 else ""
    print(f"    Passthrough{outro_label} -> {out_path}")
    return True



def _render_preview_passthrough(src, w, h, fps, fade_dur, full_duration,
                                preview_duration, out_path, outro_duration=0,
                                hdr=False):
    chunk = preview_duration / 3.0
    mid_start = full_duration / 2.0 - chunk / 2.0
    total_out = preview_duration + outro_duration

    if hdr:
        scale_pad = (
            f"format=nv12,hwupload,"
            f"scale_vt=w={w}:h={h}"
            f":color_matrix=bt709:color_primaries=bt709:color_transfer=bt709,"
            f"hwdownload,format=nv12,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1,fps={fps}"
        )
    else:
        scale_pad = (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1,fps={fps}"
        )

    parts = (
        f"[0:a]{AFORMAT}[anorm];"
        f"[0:v]trim=0:{chunk},setpts=PTS-STARTPTS,{scale_pad}[v0];"
        f"[anorm]atrim=0:{chunk},asetpts=PTS-STARTPTS[a0];"
        f"[0:v]trim={mid_start}:{mid_start + chunk},setpts=PTS-STARTPTS,{scale_pad}[v1];"
        f"[anorm]atrim={mid_start}:{mid_start + chunk},asetpts=PTS-STARTPTS[a1];"
        f"[0:v]trim={full_duration - chunk}:{full_duration},setpts=PTS-STARTPTS,{scale_pad}[v2];"
        f"[anorm]atrim={full_duration - chunk}:{full_duration},asetpts=PTS-STARTPTS[a2];"
    )

    parts += (
        f"[v2]fade=t=out:st={chunk - fade_dur}:d={fade_dur}[v2f];"
        f"[a2]afade=t=out:st={chunk - fade_dur}:d={fade_dur}[a2f];"
    )

    if outro_duration > 0:
        parts += (
            f"color=black:s={w}x{h}:r={fps}:d={outro_duration}[vblack];"
            f"{ANULLSRC}:d={outro_duration}[ablack];"
            f"[v0][a0][v1][a1][v2f][a2f][vblack][ablack]concat=n=4:v=1:a=1[cv][ca];"
        )
    else:
        parts += f"[v0][a0][v1][a1][v2f][a2f]concat=n=3:v=1:a=1[cv][ca];"

    parts += (
        f"[cv]fade=t=in:st=0:d={fade_dur}[vout];"
        f"[ca]afade=t=in:st=0:d={fade_dur}[aout]"
    )

    cmd = [
        "ffmpeg", "-y",
        *(HDR_INIT_HW if hdr else []),
        "-i", str(src),
        "-filter_complex", parts,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        *AUDIO_ENC,
        *(HDR_OUTPUT_FLAGS if hdr else []),
        "-r", str(fps),
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    outro_label = f" + {outro_duration:.0f}s black" if outro_duration > 0 else ""
    dur_label = f"{full_duration:.0f}s -> {preview_duration:.0f}s{outro_label}"
    print(f"    Preview passthrough ({dur_label}) -> {out_path}")
    return True


class VideoFrameReader:
    def __init__(self, path, display_w, display_h, fps=30):
        self.w = display_w
        self.h = display_h
        self.frame_size = display_w * display_h * 3
        vf = (
            f"scale={display_w}:{display_h}"
            f":force_original_aspect_ratio=decrease,"
            f"pad={display_w}:{display_h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1"
        )
        self.proc = subprocess.Popen([
            "ffmpeg", "-i", str(path),
            "-vf", vf,
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-r", str(fps),
            "pipe:1",
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read_frame(self):
        data = self.proc.stdout.read(self.frame_size)
        if len(data) < self.frame_size:
            return None
        return np.frombuffer(data, dtype=np.uint8).reshape(self.h, self.w, 3)

    def close(self):
        self.proc.stdout.close()
        self.proc.terminate()
        self.proc.wait()


def prepare_credits_photos(photo_dir, photo_count, card_w=500, card_h=700):
    import random
    rng = random.Random(42)
    photos_path = ROOT / photo_dir
    files = sorted(
        list(photos_path.glob("*.jpg")) + list(photos_path.glob("*.jpeg"))
    )
    if not files:
        return []
    step = max(1, len(files) // photo_count)
    selected = files[::step][:photo_count]

    border = 10
    inner_w = card_w - 2 * border
    inner_h = card_h - 2 * border
    cards = []
    for f in selected:
        img = ImageOps.exif_transpose(Image.open(str(f))).convert("RGB")
        w, h = img.size
        scale = min(inner_w / w, inner_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        card = Image.new("RGBA", (card_w, card_h), (255, 255, 255, 255))
        ox = border + (inner_w - new_w) // 2
        oy = border + (inner_h - new_h) // 2
        card.paste(img, (ox, oy))
        angle = rng.uniform(-15, 15)
        card = card.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))
        card_rgb = Image.new("RGB", card.size, (0, 0, 0))
        card_rgb.paste(card, mask=card.split()[3])
        x_offset = rng.randint(-200, 200)
        cards.append({
            "arr": np.array(card_rgb, dtype=np.uint8),
            "mask": np.array(card.split()[3], dtype=np.uint8),
            "x_offset": x_offset,
            "w": card_rgb.width,
            "h": card_rgb.height,
        })
        img.close()
    return cards


def render_credits_title(lines, w, h):
    font = find_font(80)
    small_font = find_font(50)
    img = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    line_data = []
    for i, line in enumerate(lines):
        f = font if i == 0 else small_font
        if not line:
            line_data.append((line, f, 0, 0))
            continue
        bb = draw.textbbox((0, 0), line, font=f)
        lw, lh = bb[2] - bb[0], bb[3] - bb[1]
        line_data.append((line, f, lw, lh))

    spacing = 20
    visible = [d for d in line_data if d[0]]
    block_h = sum(d[3] for d in visible) + spacing * (len(visible) - 1)
    empty_h = 40
    total_block = block_h + empty_h * sum(1 for d in line_data if not d[0])
    y = (h - total_block) // 2

    for line, f, lw, lh in line_data:
        if not line:
            y += empty_h
            continue
        x = (w - lw) // 2
        draw.text((x, y), line, fill=(255, 255, 255), font=f)
        y += lh + spacing

    return np.array(img, dtype=np.float32)


def render_quote_overlay(name, quote, area_w, area_h):
    name_font = find_font(90)
    quote_font = find_font(50)
    img = Image.new("RGB", (area_w, area_h), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = 60
    text_w = area_w - 2 * margin
    text_y = area_h // 3

    if name:
        draw.text((margin, text_y), name, fill=(255, 255, 255), font=name_font)
        bb = draw.textbbox((0, 0), name, font=name_font)
        text_y += (bb[3] - bb[1]) + 40

    if quote:
        words = quote.split()
        lines = []
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            bb = draw.textbbox((0, 0), test, font=quote_font)
            if bb[2] - bb[0] > text_w and current:
                lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)

        wrapped = ["\u201c" + lines[0]] + lines[1:] if lines else []
        if wrapped:
            wrapped[-1] = wrapped[-1] + "\u201d"

        for line in wrapped:
            draw.text((margin, text_y), line, fill=(200, 200, 200), font=quote_font)
            bb = draw.textbbox((0, 0), line, font=quote_font)
            text_y += (bb[3] - bb[1]) + 10

    return np.array(img, dtype=np.uint8)


def render_photo_quote_overlay(photo_path, name, quote, area_w, area_h):
    img = ImageOps.exif_transpose(Image.open(str(photo_path))).convert("RGB")
    pw, ph = img.size

    photo_max_h = area_h // 2
    photo_max_w = area_w - 120
    scale = min(photo_max_w / pw, photo_max_h / ph)
    new_w, new_h = int(pw * scale), int(ph * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (area_w, area_h), (0, 0, 0))
    photo_x = (area_w - new_w) // 2
    photo_y = 80
    canvas.paste(img, (photo_x, photo_y))
    img.close()

    draw = ImageDraw.Draw(canvas)
    name_font = find_font(90)
    quote_font = find_font(50)
    margin = 60
    text_w = area_w - 2 * margin
    text_y = photo_y + new_h + 40

    if name:
        draw.text((margin, text_y), name, fill=(255, 255, 255), font=name_font)
        bb = draw.textbbox((0, 0), name, font=name_font)
        text_y += (bb[3] - bb[1]) + 30

    if quote:
        words = quote.split()
        lines = []
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            bb = draw.textbbox((0, 0), test, font=quote_font)
            if bb[2] - bb[0] > text_w and current:
                lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)
        wrapped = ["\u201c" + lines[0]] + lines[1:] if lines else []
        if wrapped:
            wrapped[-1] = wrapped[-1] + "\u201d"
        for line in wrapped:
            draw.text((margin, text_y), line, fill=(200, 200, 200), font=quote_font)
            bb = draw.textbbox((0, 0), line, font=quote_font)
            text_y += (bb[3] - bb[1]) + 10

    return np.array(canvas, dtype=np.uint8)


def premix_credits_audio(music_path, music_end, voice_clips, total_duration, out_path):
    inputs = ["-i", str(music_path)]
    for vc in voice_clips:
        inputs += ["-i", str(vc["file"])]

    n_voices = len(voice_clips)

    duck_ramp = 2.0
    duck_level = 0.15
    duck_parts = []
    for vc in voice_clips:
        s = vc["start"]
        e = vc["start"] + vc["duration"]
        duck_parts.append(
            f"clip((t-{s:.2f})/{duck_ramp:.1f},0,1)*clip(({e:.2f}-t)/{duck_ramp:.1f},0,1)"
        )
    if duck_parts:
        duck_env = "+".join(f"({p})" for p in duck_parts)
        vol_expr = f"1.0-({1.0-duck_level})*min({duck_env},1)"
        music_filter = (
            f"[0:a]atrim=0:{music_end},asetpts=PTS-STARTPTS,"
            f"{AFORMAT},"
            f"volume='{vol_expr}':eval=frame,"
            f"afade=t=in:st=0:d=2,"
            f"afade=t=out:st={music_end - 3}:d=3,"
            f"apad=whole_dur={total_duration}[music];"
        )
    else:
        music_filter = (
            f"[0:a]atrim=0:{music_end},asetpts=PTS-STARTPTS,"
            f"{AFORMAT},"
            f"afade=t=in:st=0:d=2,"
            f"afade=t=out:st={music_end - 3}:d=3,"
            f"apad=whole_dur={total_duration}[music];"
        )

    voice_filters = ""
    voice_labels = []
    for i, vc in enumerate(voice_clips):
        delay_ms = int(vc["start"] * 1000)
        delays = "|".join([str(delay_ms)] * 6)
        voice_filters += (
            f"[{i+1}:a]{AFORMAT},"
            f"adelay={delays},"
            f"apad=whole_dur={total_duration}[v{i}];"
        )
        voice_labels.append(f"[v{i}]")

    if voice_labels:
        all_labels = "[music]" + "".join(voice_labels)
        n_inputs = 1 + n_voices
        mix_filter = f"{all_labels}amix=inputs={n_inputs}:duration=first:normalize=0[aout]"
    else:
        mix_filter = "[music]acopy[aout]"

    filter_complex = music_filter + voice_filters + mix_filter

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        *AUDIO_ENC,
        "-t", str(total_duration),
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def render_credits(seg, output_cfg, out_path, preview_duration=None):
    w = output_cfg["width"]
    h = output_cfg["height"]
    fps = output_cfg["fps"]

    music_path = ROOT / seg["music"]
    music_end = seg.get("music_content_end")
    if music_end is None:
        music_end = probe_duration(music_path)
    thanks_lines = seg.get("thanks_lines", ["Thank You"])
    thanks_dur = seg.get("thanks_duration", 8.0)
    closing_lines = seg.get("closing_lines", ["The End"])
    closing_dur = seg.get("closing_duration", 8.0)
    photo_dir = seg.get("photo_dir", "input/photos")
    photo_count = seg.get("photo_count", 35)
    photo_dur = seg.get("photo_duration", 3.0)
    thanks_video_path = ROOT / seg["thanks_video"] if seg.get("thanks_video") else None
    quotes = seg.get("quotes", [])
    fade_dur = 2.0

    for q in quotes:
        q["_path"] = ROOT / q["file"]
        if q["type"] == "video":
            q["_duration"] = probe_duration(q["_path"])
        else:
            q["_duration"] = q.get("duration", 8.0)

    thanks_vid_dur = 0
    if thanks_video_path and thanks_video_path.exists():
        thanks_vid_dur = probe_duration(thanks_video_path)

    timeline = []
    t = 0.0

    black_intro = 1.0
    timeline.append({"type": "black", "start": t, "end": t + black_intro})
    t += black_intro

    scroll_gap = 8.0
    quotes_total = sum(q["_duration"] for q in quotes)
    available = music_end - t - quotes_total
    per_gap = max(3.0, min(scroll_gap, available / max(1, len(quotes))))

    for q in quotes:
        timeline.append({"type": "scroll", "start": t, "end": t + per_gap})
        t += per_gap
        timeline.append({
            "type": "quote",
            "start": t,
            "end": t + q["_duration"],
            "quote": q,
        })
        t += q["_duration"]

    remaining_music = music_end - t
    if remaining_music > 1.0:
        timeline.append({"type": "scroll", "start": t, "end": t + remaining_music})
        t += remaining_music

    post_music_gap = 2.0
    timeline.append({"type": "black", "start": t, "end": t + post_music_gap})
    t += post_music_gap

    if thanks_video_path and thanks_video_path.exists() and thanks_vid_dur > 0:
        timeline.append({
            "type": "thanks_video",
            "start": t,
            "end": t + thanks_vid_dur,
        })
        t += thanks_vid_dur
        timeline.append({"type": "black", "start": t, "end": t + 1.0})
        t += 1.0

    timeline.append({"type": "title", "start": t, "end": t + closing_dur, "lines": closing_lines})
    t += closing_dur

    total_duration = t

    if preview_duration:
        credits_preview = max(preview_duration, 60.0)
        total_duration = min(credits_preview, total_duration)
        timeline = [b for b in timeline if b["start"] < total_duration]
        for b in timeline:
            if b["end"] > total_duration:
                b["end"] = total_duration
                if b["type"] == "quote":
                    b["quote"]["_duration"] = b["end"] - b["start"]

    total_frames = int(total_duration * fps)

    print(f"    Credits timeline ({total_duration:.0f}s, {len(timeline)} blocks):")
    for b in timeline:
        dur = b["end"] - b["start"]
        label = b["type"]
        if label == "quote":
            label += f" ({b['quote']['name']})"
        elif label == "title":
            label += f" ({b['lines'][0]})"
        print(f"      {b['start']:6.1f}s - {b['end']:6.1f}s  {label} ({dur:.1f}s)")

    voice_clips = []
    for b in timeline:
        if b["type"] == "quote" and b["quote"]["type"] == "video":
            voice_clips.append({
                "file": b["quote"]["_path"],
                "start": b["start"],
                "duration": b["end"] - b["start"],
            })
        elif b["type"] == "thanks_video" and thanks_video_path:
            voice_clips.append({
                "file": thanks_video_path,
                "start": b["start"],
                "duration": b["end"] - b["start"],
            })

    print("    Pre-mixing credits audio...")
    audio_path = TEMP_DIR / "credits_audio.wav"
    premix_credits_audio(music_path, music_end, voice_clips, total_duration, audio_path)

    print("    Loading credits photos...")
    card_w, card_h = 650, 900
    cards = prepare_credits_photos(photo_dir, photo_count, card_w, card_h)

    card_gap = 400
    max_card_h = max((c["h"] for c in cards), default=card_h)
    target_card_time = 8.0
    scroll_speed = (max_card_h + card_gap) / (target_card_time * fps)

    quote_overlays = {}
    video_readers = {}
    for b in timeline:
        if b["type"] == "quote":
            q = b["quote"]
            qid = id(b)
            left_w = int(w * 0.45)
            if q["type"] == "photo":
                quote_overlays[qid] = render_photo_quote_overlay(
                    q["_path"], q["name"], q.get("quote", ""), left_w, h
                )
            else:
                quote_overlays[qid] = render_quote_overlay(
                    q["name"], q.get("quote", ""), left_w, h
                )
        elif b["type"] == "thanks_video":
            pass

    title_arrays = {}
    for b in timeline:
        if b["type"] == "title":
            title_arrays[id(b)] = render_credits_title(b["lines"], w, h)

    print(f"    Rendering {total_frames} frames...")
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-i", str(audio_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        *AUDIO_ENC,
        "-map", "0:v", "-map", "1:a",
        "-t", str(total_duration),
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    stderr_chunks = []
    def drain_stderr():
        while True:
            chunk = proc.stderr.read(4096)
            if not chunk:
                break
            stderr_chunks.append(chunk)
    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()

    black = bytes(w * h * 3)
    scroll_y = -float(h)
    active_video_readers = {}

    fade_in_frames = int(fade_dur * fps)
    fade_out_frames = int(fade_dur * fps)

    for frame_idx in range(total_frames):
        t = frame_idx / fps

        cur_block = None
        cur_block_idx = len(timeline) - 1
        for bi, b in enumerate(timeline):
            if b["start"] <= t < b["end"]:
                cur_block = b
                cur_block_idx = bi
                break
        if cur_block is None:
            cur_block = timeline[-1]
        next_block = timeline[cur_block_idx + 1] if cur_block_idx + 1 < len(timeline) else None

        is_title = cur_block["type"] == "title"
        is_quote = cur_block["type"] in ("quote", "thanks_video")
        is_black = cur_block["type"] == "black"

        if is_black:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        elif is_title:
            block_t = t - cur_block["start"]
            block_dur = cur_block["end"] - cur_block["start"]
            title_arr = title_arrays[id(cur_block)]
            title_fade_in = min(2.0, block_dur / 3)
            title_fade_out = min(2.0, block_dur / 3)
            title_alpha = 1.0
            if block_t < title_fade_in:
                title_alpha = block_t / title_fade_in
            elif block_t > block_dur - title_fade_out:
                title_alpha = (block_dur - block_t) / title_fade_out
            canvas = (title_arr * max(0.0, min(1.0, title_alpha))).astype(np.uint8)
        else:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            block_t = t - cur_block["start"]

            left_w = int(w * 0.45)
            scroll_margin = 150
            left_col_x = scroll_margin + 200
            right_col_x = w - card_w - scroll_margin - 200
            video_fade_dur = 4.0

            if is_quote:
                bid = id(cur_block)
                block_dur = cur_block["end"] - cur_block["start"]
                remaining = block_dur - block_t
                fade = 1.5 if cur_block["type"] == "thanks_video" else video_fade_dur
                vid_alpha = min(1.0, block_t / fade)
                if remaining < fade:
                    vid_alpha = min(vid_alpha, remaining / fade)
                if cur_block["type"] == "quote":
                    q = cur_block["quote"]
                    if q["type"] == "video":
                        if bid not in active_video_readers:
                            vw, vh, _ = probe_video_info(q["_path"])
                            disp_h = h
                            disp_w = int(vw * (h / vh))
                            if disp_w > left_w:
                                disp_w = left_w
                                disp_h = int(vh * (left_w / vw))
                            active_video_readers[bid] = VideoFrameReader(
                                q["_path"], disp_w, disp_h, fps
                            )
                        reader = active_video_readers[bid]
                        vframe = reader.read_frame()
                        if vframe is not None:
                            vy = (h - reader.h) // 2
                            vx = (left_w - reader.w) // 2
                            if vid_alpha >= 1.0:
                                canvas[vy:vy+reader.h, vx:vx+reader.w] = vframe
                            else:
                                region = canvas[vy:vy+reader.h, vx:vx+reader.w]
                                blended = (region.astype(np.float32) * (1 - vid_alpha) + vframe.astype(np.float32) * vid_alpha)
                                canvas[vy:vy+reader.h, vx:vx+reader.w] = blended.astype(np.uint8)

                        overlay = quote_overlays.get(bid)
                        if overlay is not None:
                            oh, ow = overlay.shape[:2]
                            if vid_alpha >= 1.0:
                                mask = overlay > 0
                                target = canvas[0:oh, 0:ow]
                                np.copyto(target, overlay, where=mask)
                            else:
                                mask = overlay > 0
                                target = canvas[0:oh, 0:ow]
                                blended = (target.astype(np.float32) * (1 - vid_alpha) + overlay.astype(np.float32) * vid_alpha)
                                np.copyto(target, blended.astype(np.uint8), where=mask)
                    else:
                        overlay = quote_overlays.get(bid)
                        if overlay is not None:
                            oh, ow = overlay.shape[:2]
                            if vid_alpha >= 1.0:
                                canvas[0:oh, 0:ow] = overlay
                            else:
                                region = canvas[0:oh, 0:ow]
                                mask = overlay > 0
                                blended = (region.astype(np.float32) * (1 - vid_alpha) + overlay.astype(np.float32) * vid_alpha)
                                np.copyto(region, blended.astype(np.uint8), where=mask)
                elif cur_block["type"] == "thanks_video":
                    if bid not in active_video_readers:
                        if thanks_video_path and thanks_video_path.exists():
                            vw, vh, _ = probe_video_info(thanks_video_path)
                            disp_h = int(h * 0.8)
                            disp_w = int(vw * (disp_h / vh))
                            if disp_w > w:
                                disp_w = w
                                disp_h = int(vh * (w / vw))
                            active_video_readers[bid] = VideoFrameReader(
                                thanks_video_path, disp_w, disp_h, fps
                            )
                    reader = active_video_readers.get(bid)
                    if reader:
                        vframe = reader.read_frame()
                        if vframe is not None:
                            vy = (h - reader.h) // 2
                            vx = (w - reader.w) // 2
                            if vid_alpha >= 1.0:
                                canvas[vy:vy+reader.h, vx:vx+reader.w] = vframe
                            else:
                                region = canvas[vy:vy+reader.h, vx:vx+reader.w]
                                blended = (region.astype(np.float32) * (1 - vid_alpha) + vframe.astype(np.float32) * vid_alpha)
                                canvas[vy:vy+reader.h, vx:vx+reader.w] = blended.astype(np.uint8)

            def _blit_card(card, base_x):
                c_arr = card["arr"]
                c_mask = card["mask"]
                c_w = card["w"]
                c_h_card = card["h"]
                c_xoff = card["x_offset"]
                dx = base_x + c_xoff + (card_w - c_w) // 2
                dx = max(scroll_margin, min(dx, w - c_w - scroll_margin))
                dx1 = min(dx + c_w, w - scroll_margin)
                return dx, dx1, c_arr, c_mask, c_w, c_h_card

            if is_quote:
                left_col_alpha = 0.0
            elif (next_block and next_block["type"] in ("quote", "thanks_video")
                  and cur_block["end"] - t < video_fade_dur):
                left_col_alpha = (cur_block["end"] - t) / video_fade_dur
            else:
                prev_block = timeline[cur_block_idx - 1] if cur_block_idx > 0 else None
                if (prev_block and prev_block["type"] in ("quote", "thanks_video")
                        and block_t < video_fade_dur):
                    left_col_alpha = block_t / video_fade_dur
                else:
                    left_col_alpha = 1.0

            end_fade = 4.0
            if t >= music_end:
                photo_alpha = 0.0
            elif music_end - t < end_fade:
                photo_alpha = (music_end - t) / end_fade
            else:
                photo_alpha = 1.0

            scroll_view_start = int(scroll_y)
            for ci, card in enumerate(cards):
                c_h_card = card["h"]
                card_top = ci * (max_card_h + card_gap)
                screen_top = card_top - scroll_view_start

                if screen_top + c_h_card < 0 or screen_top >= h:
                    continue

                src_y0 = max(0, -screen_top)
                src_y1 = min(c_h_card, h - screen_top)
                dst_y0 = max(0, screen_top)
                dst_y1 = dst_y0 + (src_y1 - src_y0)

                is_left = ci % 2 == 0
                base_x = left_col_x if is_left else right_col_x

                dx, dx1, c_arr, c_mask, c_w_card, _ = _blit_card(card, base_x)
                src_x1 = dx1 - dx

                if dst_y1 > dst_y0 and dx1 > dx and src_x1 > 0:
                    region = canvas[dst_y0:dst_y1, dx:dx1]
                    src = c_arr[src_y0:src_y1, :src_x1]
                    mask_slice = c_mask[src_y0:src_y1, :src_x1]
                    mask3 = mask_slice[:, :, np.newaxis] > 128
                    card_alpha = photo_alpha
                    if is_left:
                        card_alpha = min(card_alpha, left_col_alpha)
                    if card_alpha <= 0:
                        continue
                    elif card_alpha < 1.0:
                        blended = (region.astype(np.float32) * (1 - card_alpha) +
                                   src.astype(np.float32) * card_alpha)
                        np.copyto(region, blended.astype(np.uint8), where=mask3)
                    else:
                        np.copyto(region, src, where=mask3)

            if cur_block["type"] != "thanks_video" and frame_idx >= fade_in_frames and t < music_end:
                scroll_y += scroll_speed

        alpha = 1.0
        if frame_idx < fade_in_frames:
            alpha = frame_idx / fade_in_frames
        elif frame_idx >= total_frames - fade_out_frames:
            alpha = (total_frames - frame_idx) / fade_out_frames

        if alpha <= 0:
            proc.stdin.write(black)
        elif alpha >= 1.0:
            proc.stdin.write(canvas.tobytes())
        else:
            frame = (canvas.astype(np.float32) * alpha).astype(np.uint8)
            proc.stdin.write(frame.tobytes())

        if frame_idx % (fps * 10) == 0 and frame_idx > 0:
            print(f"      {frame_idx / fps:.0f}s / {total_duration:.0f}s")

    for reader in active_video_readers.values():
        reader.close()

    proc.stdin.close()
    stderr_thread.join(timeout=30)
    proc.wait()
    if proc.returncode != 0:
        stderr = b"".join(stderr_chunks).decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed for credits: {stderr[-2000:]}")
    print(f"    Credits -> {out_path}")
    return True


def render_quote(seg, output_cfg, out_path, preview_duration=None):
    w = output_cfg["width"]
    h = output_cfg["height"]
    fps = output_cfg["fps"]

    has_video = "video" in seg
    has_photo = "photo" in seg
    if not has_video and not has_photo:
        print("    WARNING: quote segment has no video or photo, skipping")
        return False

    media_path = ROOT / (seg["video"] if has_video else seg["photo"])
    if not media_path.exists():
        print(f"    WARNING: {media_path} not found, skipping quote")
        return False

    name = seg.get("name", "")
    quote = seg.get("quote", "")
    fade_dur = 1.0

    if has_video:
        src_w, src_h, src_fps = probe_video_info(media_path)
        duration = probe_duration(media_path)
    else:
        duration = seg.get("duration", 8.0)
        img = Image.open(str(media_path))
        src_w, src_h = img.size
        img.close()

    media_display_h = h
    media_display_w = int(src_w * (h / src_h))
    if media_display_w > w * 0.45:
        media_display_w = int(w * 0.45)
        media_display_h = int(src_h * (media_display_w / src_w))

    text_area_x = media_display_w + 80
    text_area_w = w - text_area_x - 80

    name_font = find_font(90)
    quote_font = find_font(50)

    text_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_overlay)

    text_y = h // 3

    if name:
        draw.text((text_area_x, text_y), name, fill=(255, 255, 255), font=name_font)
        bb = draw.textbbox((0, 0), name, font=name_font)
        text_y += (bb[3] - bb[1]) + 40

    if quote:
        words = quote.split()
        lines = []
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            bb = draw.textbbox((0, 0), test, font=quote_font)
            if bb[2] - bb[0] > text_area_w and current:
                lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)

        quote_text = "\u201c" + lines[0] if lines else ""
        wrapped = [quote_text] + lines[1:]
        if wrapped:
            wrapped[-1] = wrapped[-1] + "\u201d"

        for line in wrapped:
            draw.text((text_area_x, text_y), line, fill=(200, 200, 200), font=quote_font)
            bb = draw.textbbox((0, 0), line, font=quote_font)
            text_y += (bb[3] - bb[1]) + 10

    text_overlay_path = TEMP_DIR / "quote_overlay.png"
    text_overlay.save(str(text_overlay_path))

    if has_video:
        filter_complex = (
            f"[0:v]scale={media_display_w}:{media_display_h}:force_original_aspect_ratio=decrease,"
            f"pad={media_display_w}:{media_display_h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1[vid];"
            f"color=black:s={w}x{h}:r={fps}:d={duration}[bg];"
            f"[bg][vid]overlay=0:(main_h-overlay_h)/2[bv];"
            f"[1:v]format=rgba[txt];"
            f"[bv][txt]overlay=0:0,"
            f"fade=t=in:st=0:d={fade_dur},"
            f"fade=t=out:st={duration - fade_dur}:d={fade_dur}[vout];"
            f"[0:a]{AFORMAT},"
            f"afade=t=in:st=0:d={fade_dur},"
            f"afade=t=out:st={duration - fade_dur}:d={fade_dur}[aout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(media_path),
            "-i", str(text_overlay_path),
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            *AUDIO_ENC,
            "-r", str(fps),
            str(out_path),
        ]
    else:
        filter_complex = (
            f"[0:v]scale={media_display_w}:{media_display_h}:force_original_aspect_ratio=decrease,"
            f"pad={media_display_w}:{media_display_h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1[vid];"
            f"color=black:s={w}x{h}:r={fps}:d={duration}[bg];"
            f"[bg][vid]overlay=0:(main_h-overlay_h)/2[bv];"
            f"[1:v]format=rgba[txt];"
            f"[bv][txt]overlay=0:0,"
            f"fade=t=in:st=0:d={fade_dur},"
            f"fade=t=out:st={duration - fade_dur}:d={fade_dur}[vout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-framerate", str(fps),
            "-i", str(media_path),
            "-i", str(text_overlay_path),
            "-f", "lavfi", "-i", f"{ANULLSRC}:d={duration}",
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "2:a",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            *AUDIO_ENC,
            "-t", str(duration),
            str(out_path),
        ]

    subprocess.run(cmd, check=True, capture_output=True)
    label = "video" if has_video else "photo"
    print(f"    Quote ({label}: {name}) -> {out_path}")
    return True


def concat_segments(segment_files, output_cfg, out_path):
    concat_list = TEMP_DIR / "concat.txt"
    with open(concat_list, "w") as f:
        for sf in segment_files:
            f.write(f"file '{sf}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "copy",
        "-af", f"volume=-4dB,alimiter=limit=0.9:attack=5:release=50,{AFORMAT}",
        "-c:a", "aac", "-b:a", "384k",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  Final output -> {out_path}")


PREVIEW_DURATION = 15.0


def main():
    list_only = "--list" in sys.argv
    skip_render = "--skip-render" in sys.argv
    preview_mode = "--preview" in sys.argv
    single_segment = None
    if "--segment" in sys.argv:
        idx = sys.argv.index("--segment")
        single_segment = int(sys.argv[idx + 1])

    program = load_program()
    output_cfg = program.get("output", {
        "width": 3840, "height": 2160, "fps": 30, "transition_duration": 1.0,
    })
    segments = program.get("segment", [])

    OUTPUT_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)
    _init_title_music(output_cfg)

    if list_only:
        print(f"Program segments{' (preview ~15s each)' if preview_mode else ''}:")
        cur_act = None
        for i, seg in enumerate(segments):
            act = seg.get("act")
            if act != cur_act:
                cur_act = act
                print(f"\n  --- Act {act} ---")
            stype = seg["type"]
            label = ""
            if stype == "title":
                label = " | ".join(l for l in seg.get("lines", []) if l)
            elif stype == "intro":
                label = f"{seg.get('title', '')} — {seg.get('subtitle', '')}"
            elif stype == "quote":
                label = f"{seg.get('name', '')} — {seg.get('quote', '')[:40]}"
            elif stype == "credits":
                n_quotes = len(seg.get("quotes", []))
                label = f"{n_quotes} quotes + photo scroll"
            elif stype in ("episode", "scene"):
                label = seg.get("file", "")
            print(f"  [{i:2d}] {stype:10s}  {label}")
        return

    mode_label = "PREVIEW" if preview_mode else "FULL"
    print(f"=== Assembling {len(segments)} segments ({mode_label}) ===")
    if preview_mode:
        print(f"  Long segments truncated to ~{PREVIEW_DURATION:.0f}s (head/mid/tail)")

    segment_files = []
    segment_acts = []

    preview_dur = PREVIEW_DURATION if preview_mode else None

    for i, seg in enumerate(segments):
        suffix = "_preview" if preview_mode else ""
        out_file = TEMP_DIR / f"seg_{i:03d}{suffix}.mp4"
        stype = seg["type"]
        act = seg.get("act")

        if single_segment is not None and i != single_segment:
            if out_file.exists():
                segment_files.append(out_file)
                segment_acts.append(act)
            continue

        if skip_render and out_file.exists():
            print(f"  [{i}] {stype}: reusing {out_file.name}")
            segment_files.append(out_file)
            segment_acts.append(act)
            continue

        print(f"  [{i}] {stype}...")

        ok = False
        if stype == "title":
            render_title_card(seg, output_cfg, out_file)
            ok = True
        elif stype == "intro":
            ok = render_intro(seg, output_cfg, out_file)
        elif stype in ("episode", "scene"):
            ok = render_passthrough(seg, output_cfg, out_file,
                                    preview_duration=preview_dur)
        elif stype == "quote":
            ok = render_quote(seg, output_cfg, out_file,
                              preview_duration=preview_dur)
        elif stype == "credits":
            ok = render_credits(seg, output_cfg, out_file,
                                preview_duration=preview_dur)
        else:
            print(f"    WARNING: unknown segment type '{stype}'")

        if ok:
            segment_files.append(out_file)
            segment_acts.append(act)

    if single_segment is not None:
        print(f"\n=== Rendered segment {single_segment} only ===")
        return

    if not segment_files:
        print("ERROR: No segments to concatenate")
        sys.exit(1)

    suffix = "_preview" if preview_mode else ""

    acts = sorted(set(a for a in segment_acts if a is not None))
    if len(acts) > 1:
        for act in acts:
            act_files = [f for f, a in zip(segment_files, segment_acts) if a == act]
            if not act_files:
                continue
            act_output = OUTPUT_DIR / f"act{act}{suffix}.mp4"
            print(f"\n=== Concatenating Act {act} ({len(act_files)} segments) ===")
            concat_segments(act_files, output_cfg, act_output)

    print(f"\n=== Concatenating all {len(segment_files)} segments ===")
    final_output = OUTPUT_DIR / (f"playbill{suffix}.mp4")
    concat_segments(segment_files, output_cfg, final_output)
    print(f"\n=== Done! ===")
    for act in acts:
        print(f"  Act {act}: output/act{act}{suffix}.mp4")
    print(f"  Full:   {final_output}")


if __name__ == "__main__":
    main()
