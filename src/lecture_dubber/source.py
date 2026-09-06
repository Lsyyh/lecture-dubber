from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

from .models import Config, SourceInfo
from .utils import ensure_command, run, slugify

# Bilibili desktop client cache: <id>/<id>-<page>-<code>.m4s where codes 300xx
# are video streams and 302xx are audio streams, and every file starts with 9
# junk bytes ("000000000") before the ftyp box.
_BILI_PAGE_RE = re.compile(r"-(\d+)-(\d+)\.m4s$")
_BILI_HEADER = b"000000000"


def is_url(value: str) -> bool:
    p = urlparse(value)
    return p.scheme in {"http", "https"} and bool(p.netloc)


def _cache_pages(folder: Path) -> dict[int, dict[str, list[tuple[int, Path]]]]:
    pages: dict[int, dict[str, list[tuple[int, Path]]]] = {}
    for path in sorted(folder.glob("*.m4s")):
        m = _BILI_PAGE_RE.search(path.name)
        if not m:
            continue
        page, code = int(m.group(1)), int(m.group(2))
        kind = "audio" if code >= 30200 else "video"
        pages.setdefault(page, {"video": [], "audio": []})[kind].append((code, path))
    return pages


def _read_cache_title(folder: Path) -> str | None:
    info_path = folder / "videoInfo.json"
    if not info_path.exists():
        return None
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    title = info.get("title")
    return str(title) if title else None


def bilibili_cache_title(value: str) -> str | None:
    """Title of a bilibili cache folder, or None if value is not one."""
    folder = Path(value).expanduser()
    if not folder.is_dir() or not _cache_pages(folder):
        return None
    return _read_cache_title(folder)


def _copy_stripped(src: Path, dst: Path) -> Path:
    with src.open("rb") as fsrc, dst.open("wb") as fdst:
        head = fsrc.read(len(_BILI_HEADER))
        if head != _BILI_HEADER:
            fdst.write(head)
        shutil.copyfileobj(fsrc, fdst)
    return dst


def _import_bilibili_cache(folder: Path, job_dir: Path, cfg: Config) -> SourceInfo:
    ensure_command("ffmpeg")
    download_dir = job_dir / "source"
    download_dir.mkdir(parents=True, exist_ok=True)

    pages = _cache_pages(folder)
    page = min(pages)
    streams = pages[page]
    if not streams["video"]:
        raise RuntimeError(f"No video stream in bilibili cache: {folder}")

    title = _read_cache_title(folder) or folder.name
    video = _copy_stripped(
        max(streams["video"], key=lambda t: t[0])[1], download_dir / f"page{page}_video.m4s"
    )
    inputs = ["-i", str(video)]
    if streams["audio"]:
        audio = _copy_stripped(
            max(streams["audio"], key=lambda t: t[0])[1], download_dir / f"page{page}_audio.m4s"
        )
        inputs += ["-i", str(audio)]

    out = download_dir / f"{slugify(title)[:80]}.mp4"
    run(["ffmpeg", "-y", *inputs, "-c", "copy", "-movflags", "+faststart", str(out)])

    try:
        info = json.loads((folder / "videoInfo.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        info = {}
    bvid = info.get("bvid")
    return SourceInfo(
        kind="local",
        input=str(folder),
        video_path=str(out.resolve()),
        title=title,
        webpage_url=f"https://www.bilibili.com/video/{bvid}" if bvid else None,
    )


def import_source(value: str, job_dir: Path, cfg: Config) -> SourceInfo:
    job_dir.mkdir(parents=True, exist_ok=True)
    if not is_url(value):
        src = Path(value).expanduser().resolve()
        if _cache_pages(src):
            return _import_bilibili_cache(src, job_dir, cfg)
        if not src.exists():
            raise FileNotFoundError(src)
        return SourceInfo(kind="local", input=value, video_path=str(src), title=src.stem)

    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError(
            "yt-dlp is not installed. Install the base package with: pip install -e ."
        ) from e

    download_dir = job_dir / "source"
    download_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(download_dir / "%(title).80s-%(id)s.%(ext)s")
    sublangs = list(dict.fromkeys([*cfg.subtitle_languages, "en.*"]))
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": f"bv*[height<={cfg.source_max_height}]+ba/b[height<={cfg.source_max_height}]/best",
        "merge_output_format": "mp4",
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": sublangs,
        "subtitlesformat": "vtt/srt/best",
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(value, download=True)
        requested = info.get("requested_downloads") or []
        candidates = []
        for item in requested:
            fp = item.get("filepath")
            if fp:
                candidates.append(Path(fp))
        prepared = Path(ydl.prepare_filename(info))
        candidates.extend([prepared, prepared.with_suffix(".mp4"), prepared.with_suffix(".mkv")])
        video_path = next((p for p in candidates if p.exists()), None)
        if video_path is None:
            media = [
                p
                for p in download_dir.iterdir()
                if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
            ]
            if not media:
                raise RuntimeError("yt-dlp finished but no downloaded media file was found")
            video_path = max(media, key=lambda p: p.stat().st_size)
        subtitles = [str(p) for p in download_dir.iterdir() if p.suffix.lower() in {".vtt", ".srt"}]
        return SourceInfo(
            kind="url",
            input=value,
            video_path=str(video_path.resolve()),
            title=info.get("title") or slugify(info.get("id", "video")),
            webpage_url=info.get("webpage_url") or value,
            subtitle_paths=subtitles,
        )
