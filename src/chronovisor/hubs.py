"""Compatibility alias for :mod:`chronovisor.ops.hubs`."""

from chronovisor.ops import hubs as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
