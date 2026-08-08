"""Compatibility shim for classification-owned calibration."""

from __future__ import annotations

import sys

from chronovisor.classification import classification_calibration as _implementation


def main(argv: list[str] | None = None) -> int:
    return _implementation.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

sys.modules[__name__] = _implementation
