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
            escaped = arg.replace('"', r'\"')
            parts.append(f'"{escaped}"')
        else:
            parts.append(arg)
    return " ".join(parts)


def ensure_command(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required command not found: {name}")
    return path


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _join_cmdline(cmd), check=check, text=True, capture_output=True, errors="replace"
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
