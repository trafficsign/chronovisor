"""Compatibility alias for :mod:`chronovisor.recall.freshness_candidates`."""

from chronovisor.recall import freshness_candidates as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
