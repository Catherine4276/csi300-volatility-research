"""Configuration loading with project-relative path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and resolve common project paths."""

    config_path = Path(path) if path else PROJECT_ROOT / "config" / "default.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config["_config_path"] = str(config_path.resolve())
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def project_path(*parts: str) -> Path:
    """Return a path rooted at the project directory."""

    return PROJECT_ROOT.joinpath(*parts)

