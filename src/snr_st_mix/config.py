from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration."""
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply CLI overrides such as ``training.epochs=5``."""
    result = deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value")
        dotted_key, raw_value = override.split("=", 1)
        keys = dotted_key.split(".")
        cursor = result
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = yaml.safe_load(raw_value)
    return result

