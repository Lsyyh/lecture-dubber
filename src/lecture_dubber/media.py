from __future__ import annotations

import os
from pathlib import Path

from .utils import ensure_command, run


def extract_audio(video: Path, out_wav: Path) -> Path:
    ensure_command("ffmpeg")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(out_wav)
    ])
    return out_wav


def probe_duration(path: Path) -> float:
    ensure_command("ffprobe")
    p = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ])
    return float(p.stdout.strip())


def fit_audio(in_wav: Path, out_wav: Path, target_duration: float, soft_min: float, soft_max: float) -> float:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    actual = probe_duration(in_wav)
    if target_duration <= 0 or actual <= 0:
        out_wav.write_bytes(in_wav.read_bytes())
        return 1.0
    speed = actual / target_duration
    if soft_min <= speed <= soft_max:
        if abs(speed - 1.0) < 0.015:
            out_wav.write_bytes(in_wav.read_bytes())
        else:
            run(["ffmpeg", "-y", "-i", str(in_wav), "-filter:a", f"atempo={speed:.6f}", str(out_wav)])
        return speed
    out_wav.write_bytes(in_wav.read_bytes())
    return 1.0


def compose_timeline(segment_files: list[tuple[float, Path]], total_duration: float, out_wav: Path) -> Path:
    ensure_command("ffmpeg")
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if not segment_files:
        raise ValueError("No TTS segment files to compose")
    # One -i per segment builds a command line that can exceed the 8191-char
    # limit of the cmd.exe spawn on this machine, so keep the line short:
    # inputs are passed relative to the output directory and the filtergraph
    # goes into a script file instead of the command line.
    graph = []
    labels = []
    for i, (start, _) in enumerate(segment_files):
        delay_ms = max(0, round(start * 1000))
        label = f"a{i}"
        graph.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[{label}]")
        labels.append(f"[{label}]")
    graph.append("".join(labels) + f"amix=inputs={len(labels)}:normalize=0,atrim=0:{total_duration:.3f}[mix]")
    graph_file = out_wav.parent / (out_wav.name + ".graph")
    graph_file.write_text(";\n".join(graph), encoding="utf-8")
    inputs: list[str] = []
    for _, path in segment_files:
        inputs += ["-i", os.path.relpath(path, out_wav.parent)]
    try:
        run(
            ["ffmpeg", "-y", *inputs, "-/filter_complex", graph_file.name,
             "-map", "[mix]", "-ar", "48000", out_wav.name],
            cwd=out_wav.parent,
        )
    finally:
        graph_file.unlink(missing_ok=True)
    return out_wav


def render_video(video: Path, dub_wav: Path, subtitle: Path | None, out_mp4: Path, original_gain_db: float, dub_gain_db: float) -> Path:
    ensure_command("ffmpeg")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    filter_parts = [
        f"[0:a]volume={original_gain_db}dB[orig]",
        f"[1:a]volume={dub_gain_db}dB[dub]",
        "[orig][dub]amix=inputs=2:duration=first:normalize=0[aout]",
    ]
    cmd = ["ffmpeg", "-y", "-i", str(video), "-i", str(dub_wav)]
    if subtitle is not None:
        escaped = str(subtitle).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        filter_parts.append(f"[0:v]subtitles='{escaped}'[vout]")
        video_map = "[vout]"
    else:
        video_map = "0:v"
    cmd.extend([
        "-filter_complex", ";".join(filter_parts),
        "-map", video_map, "-map", "[aout]", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_mp4)
    ])
    run(cmd)
    return out_mp4
