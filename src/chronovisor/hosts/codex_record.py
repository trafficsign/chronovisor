"""Compatibility alias for the raw Codex record module."""

from __future__ import annotations

import sys

from chronovisor.raw import codex_record as _codex_record


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-codex-record`` command-line entry point."""
    return _codex_record.main(argv)


sys.modules[__name__] = _codex_record
