"""Configuration loading: packaged defaults + optional profile overlay + optional
user overrides from ./config/*.toml in the working directory."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

PACKAGED_CONFIG_DIR = Path(__file__).resolve().parent / "config"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def load_config(profile: str | None = None, extra_path: str | Path | None = None) -> dict[str, Any]:
    """Load default.toml, overlay a named profile (e.g. 'pi5'), then an explicit file.

    Named files are looked up first in ./config/ (working directory, so users can
    edit without touching the installed package), then in the packaged config dir.
    """
    config = _read_toml(PACKAGED_CONFIG_DIR / "default.toml")

    def resolve(name: str) -> Path:
        candidates = (Path.cwd() / "config" / f"{name}.toml", PACKAGED_CONFIG_DIR / f"{name}.toml")
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"config profile '{name}' not found in ./config/ or packaged")

    if profile:
        config = _deep_merge(config, _read_toml(resolve(profile)))
    if extra_path:
        config = _deep_merge(config, _read_toml(Path(extra_path)))
    return config
