"""Compatibility alias for :mod:`chronovisor.ops.entities`."""

from chronovisor.ops import entities as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
