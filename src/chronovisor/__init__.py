"""Chronovisor package bootstrap and compatibility boundary."""

from __future__ import annotations

import os


def _alias_environment(old: str, new: str) -> None:
    """Expose a legacy environment variable under its canonical name.

    Conflicting values are rejected instead of silently selecting one.  This
    keeps rollback configuration readable without allowing split-brain runtime
    policy.
    """

    old_value = os.environ.get(old)
    if old_value is None:
        return
    new_value = os.environ.get(new)
    if new_value is not None and new_value != old_value:
        raise RuntimeError(
            f"conflicting compatibility environment variables: {old} and {new}"
        )
    os.environ.setdefault(new, old_value)


for _name in tuple(os.environ):
    if _name.startswith("LLM_WIKI_"):
        _alias_environment(_name, "CHRONOVISOR_" + _name.removeprefix("LLM_WIKI_"))

_alias_environment(
    "CODEX_WIKI_SAVE_ENABLED", "CODEX_CHRONOVISOR_RECORD_ENABLED"
)
_alias_environment(
    "CLAUDE_CODE_WIKI_SAVE_ENABLED", "CLAUDE_CODE_CHRONOVISOR_RECORD_ENABLED"
)
