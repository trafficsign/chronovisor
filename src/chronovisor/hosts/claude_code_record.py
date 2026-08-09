"""Compatibility alias for the raw Claude Code record module."""

from __future__ import annotations

import sys

from chronovisor.raw import claude_code_record as _claude_code_record


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-claude-code-record`` command-line entry point."""
    return _claude_code_record.main(argv)


sys.modules[__name__] = _claude_code_record
