"""Compatibility alias for the ingest self-heal module."""

from __future__ import annotations

import sys

from chronovisor.ingest import self_heal as _self_heal


def main(argv: list[str] | None = None) -> int:
    """Run the self-heal command-line interface."""
    return _self_heal.main(argv)


sys.modules[__name__] = _self_heal
