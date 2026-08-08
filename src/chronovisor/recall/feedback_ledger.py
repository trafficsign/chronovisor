"""Compatibility adapter for the search-owned feedback ledger."""

from __future__ import annotations

import sys

from chronovisor.search import feedback_ledger as _implementation

sys.modules[__name__] = _implementation
