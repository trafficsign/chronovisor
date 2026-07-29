"""Compatibility alias for :mod:`chronovisor.core.jsonl_write`."""

from chronovisor.core import jsonl_write as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
