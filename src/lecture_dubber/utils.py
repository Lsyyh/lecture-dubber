from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

# On this Windows machine children are spawned through cmd.exe (CreateProcess is
# hooked), so any argument containing shell metacharacters must be quoted with
# MSVCRT rules or cmd splits it at | ; & etc.
_CMD_META = re.compile(r'[\s"|&;<>^()]')


def _join_cmdline(cmd: list[str]) -> str:
    parts: list[str] = []
    for arg in cmd:
        if _CMD_META.search(arg):
            escaped = arg.replace('"', r"\"")
            parts.append(f'"{escaped}"')
        else:
            parts.append(arg)
    return " ".join(parts)


def ensure_command(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required command not found: {name}")
    return path


def run(
    cmd: list[str], *, check: bool = True, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _join_cmdline(cmd), check=check, text=True, capture_output=True, errors="replace", cwd=cwd
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def slugify(value: str) -> str:
    value = re.sub(r"[^\w\-\.]+", "-", value.strip(), flags=re.UNICODE)
    value = re.sub(r"-+", "-", value).strip("-.")
    return value[:80] or "job"


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating markdown fences or prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def cleanup_intermediates(job_dir: Path) -> None:
    """Remove artifacts that are not reused across resumes of a finished job."""
    shutil.rmtree(job_dir / "tts", ignore_errors=True)
    (job_dir / "dub.wav").unlink(missing_ok=True)
    source_dir = job_dir / "source"
    if source_dir.is_dir():
        for f in source_dir.glob("*.m4s"):
            f.unlink(missing_ok=True)
