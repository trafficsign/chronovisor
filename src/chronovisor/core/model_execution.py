"""Owner-controlled model execution policy, separate from quality gates."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chronovisor.core.store import CHRONOVISOR_ROOT


def _config(root: Path | None = None) -> Mapping[str, Any]:
    root = root or Path(os.environ.get("CHRONOVISOR_ROOT") or CHRONOVISOR_ROOT)
    try:
        return tomllib.loads((root / "config.toml").read_text())
    except FileNotFoundError:
        return {}


def _settings(root: Path | None = None) -> Mapping[str, Any]:
    section = _config(root).get("model_execution", {})
    if not isinstance(section, Mapping):
        raise ValueError("model_execution must be a table")
    return section


def automatic_cloud_allowed(root: Path | None = None) -> bool:
    """Fail closed, including when an old lane or environment asks for cloud."""

    try:
        return _settings(root).get("allow_automatic_cloud_models") is True
    except (OSError, UnicodeError, ValueError):
        return False


def local_reasoning_settings(
    role: str, model: str, *, root: Path | None = None
) -> dict[str, Any] | None:
    """Resolve only explicitly configured local roles and the exact model."""

    settings = _settings(root).get("local_reasoning", {})
    if not isinstance(settings, Mapping):
        raise ValueError("local_reasoning must be a table")
    if role not in settings.get("roles", ()):
        return None
    if not model:
        model = str(_config(root).get("llm", {}).get("roles", {}).get(role, {}).get("model") or "")
    if model != settings.get("model"):
        return None
    level = settings.get("level")
    if level not in {"low", "medium", "xhigh"}:
        raise ValueError("unsupported local reasoning level")
    budgets: dict[str, int] = {}
    for key in ("num_predict", "num_ctx"):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("invalid local reasoning budgets")
        budgets[key] = value
    if budgets["num_predict"] > 131_072 or budgets["num_ctx"] > 262_144:
        raise ValueError("local reasoning exceeds runtime capacity")
    return {"level": level, **budgets}
