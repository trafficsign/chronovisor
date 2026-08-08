"""Compatibility shim for classification-owned metadata migration."""

from __future__ import annotations

import sys

from chronovisor.librarian import classification_migration as _implementation


def main(argv: list[str] | None = None) -> int:
    return _implementation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

sys.modules[__name__] = _implementation
