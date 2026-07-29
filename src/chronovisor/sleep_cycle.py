"""Compatibility alias for :mod:`chronovisor.ops.sleep_cycle`."""

from chronovisor.ops import sleep_cycle as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
