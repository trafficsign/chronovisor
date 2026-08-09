"""Compatibility alias for the ingest snapshot module."""

from __future__ import annotations

import sys

from chronovisor.ingest import snapshot as _snapshot


def main() -> int:
    """Run the snapshot command-line interface."""
    return _snapshot.main()


sys.modules[__name__] = _snapshot
