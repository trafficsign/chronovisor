"""Compatibility alias for :mod:`chronovisor.ops.reflection`."""

from chronovisor.ops import reflection as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
