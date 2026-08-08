"""Compatibility shim for the Librarian-owned preemption burn."""

from __future__ import annotations

import sys

from chronovisor.librarian import librarian_burn as _implementation


def main(argv: list[str] | None = None) -> int:
    return _implementation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

sys.modules[__name__] = _implementation
