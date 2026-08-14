"""Compatibility alias for the raw Hermes record module."""

from __future__ import annotations

import sys

from chronovisor.raw import hermes_record as _hermes_record


def main(argv: list[str] | None = None) -> int:
    return _hermes_record.main(argv)


sys.modules[__name__] = _hermes_record