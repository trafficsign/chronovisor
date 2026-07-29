"""Compatibility alias for :mod:`chronovisor.ops.dashboard`."""

from chronovisor.ops import dashboard as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
