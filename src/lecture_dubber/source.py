from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
from .models import Config, SourceInfo
from .utils import slugify


def is_url(value: str) -> bool:
    p = urlparse(value)
    return p.scheme in {"http", "https"} and bool(p.netloc)


def import_source(value: str, job_dir: Path, cfg: Config) -> SourceInfo:
    job_dir.mkdir(parents=True, exist_ok=True)
    if not is_url(value):
        src = Path(value).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        return SourceInfo(kind="local", input=value, video_path=str(src), title=src.stem)

    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError("yt-dlp is not installed. Install the base package with: pip install -e .") from e

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
            media = [p for p in download_dir.iterdir() if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
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
