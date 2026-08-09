"""Compatibility alias for the recall-owned Librarian CLI module."""

from __future__ import annotations

import sys

from chronovisor.recall import librarian as _librarian


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-librarian`` command-line entry point."""
    return _librarian.main(argv)


sys.modules[__name__] = _librarian
