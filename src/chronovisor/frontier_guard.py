"""Compatibility alias for :mod:`chronovisor.decision.frontier_guard`."""

from chronovisor.decision import frontier_guard as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
