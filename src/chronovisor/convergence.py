"""Compatibility alias for :mod:`chronovisor.ops.convergence`."""

from chronovisor.ops import convergence as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
