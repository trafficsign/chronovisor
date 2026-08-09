"""Compatibility alias for the search retention module."""

from __future__ import annotations

import sys

from chronovisor.search import retention as _retention


def main() -> int:
    """Run the retention command-line interface."""
    return _retention.main()


sys.modules[__name__] = _retention
