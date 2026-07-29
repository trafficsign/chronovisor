"""Compatibility alias for :mod:`chronovisor.ops.converge_worker`."""

from chronovisor.ops import converge_worker as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
