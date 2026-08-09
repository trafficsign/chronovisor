"""Compatibility alias for the core runtime status module."""

from __future__ import annotations

import sys

from chronovisor.core import runtime_status as _runtime_status

sys.modules[__name__] = _runtime_status
