"""Compatibility shim for the decision-owned local model evaluator."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from chronovisor.decision import local_model_eval as _implementation


def _live_transport(request: Any) -> Any:
    return _implementation._live_transport(request)


def main(argv: Sequence[str] | None = None) -> int:
    return _implementation.main(argv)

if __name__ == "__main__":
    raise SystemExit(main())

sys.modules[__name__] = _implementation
