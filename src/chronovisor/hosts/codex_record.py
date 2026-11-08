"""Persisted job module name (background-jobs state.json, hook_dispatcher). Do not move or delete."""

from __future__ import annotations

import sys

from chronovisor.capture import codex_record as _codex_record


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-codex-record`` command-line entry point."""
    return _codex_record.main(argv)


sys.modules[__name__] = _codex_record

if __name__ == "__main__":
    raise SystemExit(main())
