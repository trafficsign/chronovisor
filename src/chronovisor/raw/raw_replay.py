"""Compatibility alias for the canonical ingest raw replay module."""

from __future__ import annotations

import sys

from chronovisor.ingest import raw_replay as _raw_replay


def main(argv: list[str] | None = None) -> int:
    """Run the canonical raw replay CLI."""
    return _raw_replay.main(argv)


sys.modules[__name__] = _raw_replay
