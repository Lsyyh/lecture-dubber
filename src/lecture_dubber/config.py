from __future__ import annotations

from pathlib import Path

import yaml

from .models import Config


def load_config(path: Path | None) -> Config:
    if path is None:
        return Config()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Config.model_validate(data)
