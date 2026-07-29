"""Compatibility alias for :mod:`chronovisor.ops.convergence_drain`."""

from chronovisor.ops import convergence_drain as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
