"""Compatibility alias for the raw Codex record module."""

from __future__ import annotations

import sys

from chronovisor.raw import codex_record as _codex_record

sys.modules[__name__] = _codex_record
