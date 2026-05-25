from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "project_config.json"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_project_path(config: dict[str, Any], key: str, root: str | Path | None = None) -> Path:
    base = Path(root or config.get("drive_root", "."))
    value = Path(config[key])
    return value if value.is_absolute() else base / value
