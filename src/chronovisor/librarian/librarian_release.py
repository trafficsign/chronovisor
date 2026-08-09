"""Compatibility alias for the recall-owned Librarian release CLI module."""

from __future__ import annotations

import sys

from chronovisor.recall import librarian_release as _librarian_release


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-librarian-release`` command-line entry point."""
    return _librarian_release.main(argv)


sys.modules[__name__] = _librarian_release
