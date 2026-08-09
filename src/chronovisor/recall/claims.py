"""Compatibility alias for the search claim ledger module."""

from __future__ import annotations

import sys

from chronovisor.search import claims as _claims

sys.modules[__name__] = _claims
