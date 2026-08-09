"""Console owner for core retention score maintenance."""

from __future__ import annotations

from chronovisor.core.retention import main as _main


def main(argv: list[str] | None = None) -> int:
    """Run the retention command-line interface."""
    return _main(argv)
