"""Compatibility alias for :mod:`chronovisor.decision.frontier_review`."""

from chronovisor.decision import frontier_review as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
