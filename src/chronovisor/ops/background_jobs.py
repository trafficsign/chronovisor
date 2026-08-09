"""Compatibility alias for the core background jobs module."""

from __future__ import annotations

import sys

from chronovisor.core import background_jobs as _background_jobs


def main() -> int:
    """Run the background jobs command-line interface."""
    return _background_jobs.main()


sys.modules[__name__] = _background_jobs
