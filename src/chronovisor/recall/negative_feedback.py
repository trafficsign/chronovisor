"""Compatibility adapter for the search-owned negative-feedback implementation."""

from __future__ import annotations

import sys

from chronovisor.search import negative_feedback as _implementation

sys.modules[__name__] = _implementation
