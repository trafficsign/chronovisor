"""Compatibility alias for :mod:`chronovisor.ops.state_register`."""

from chronovisor.ops import state_register as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
