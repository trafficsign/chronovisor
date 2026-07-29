"""Compatibility alias for :mod:`chronovisor.ops.runtime_status`."""

from chronovisor.ops import runtime_status as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
