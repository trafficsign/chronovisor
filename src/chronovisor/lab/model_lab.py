"""Compatibility shim for the ops-owned model lab."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from chronovisor.ops import model_lab as _implementation


def main(argv: Sequence[str] | None = None) -> int:
    return _implementation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

sys.modules[__name__] = _implementation
