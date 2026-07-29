"""Compatibility alias for :mod:`chronovisor.ops.retention`."""

from chronovisor.ops import retention as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
