from __future__ import annotations

from pathlib import Path
from .models import TranslationUnit


def _fmt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(units: list[TranslationUnit], path: Path) -> Path:
    lines = []
    for i, u in enumerate(units, 1):
        lines.extend([str(i), f"{_fmt(u.start)} --> {_fmt(u.end)}", u.translation or u.source, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
