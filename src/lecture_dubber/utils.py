from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def ensure_command(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required command not found: {name}")
    return path


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def slugify(value: str) -> str:
    value = re.sub(r"[^\w\-\.]+", "-", value.strip(), flags=re.UNICODE)
    value = re.sub(r"-+", "-", value).strip("-.")
    return value[:80] or "job"
