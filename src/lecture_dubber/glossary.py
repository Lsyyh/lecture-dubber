from __future__ import annotations

from pathlib import Path
import yaml


def load_glossary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("glossary must be a YAML mapping")
    return {str(k): str(v) for k, v in data.items()}
