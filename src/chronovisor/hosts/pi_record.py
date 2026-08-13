"""Compatibility alias for the raw Pi record module."""

from __future__ import annotations

import sys

from chronovisor.raw import pi_record as _pi_record


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-pi-record`` command-line entry point."""
    return _pi_record.main(argv)


sys.modules[__name__] = _pi_record
