"""Compatibility alias for :mod:`chronovisor.ops.deadman`."""

from chronovisor.ops import deadman as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
