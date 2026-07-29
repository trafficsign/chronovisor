"""Compatibility alias for :mod:`chronovisor.ops.memory_integrity`."""

from chronovisor.ops import memory_integrity as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
