"""Compatibility helpers for legacy experiment module paths."""

from __future__ import annotations

import sys
from types import ModuleType


def alias_legacy_module(name: str, implementation: ModuleType) -> None:
    """Expose a relocated lab module through its legacy import path."""
    sys.modules[name] = implementation
