"""Compatibility alias for :mod:`chronovisor.ops.burn_monitor`."""

from chronovisor.ops import burn_monitor as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
