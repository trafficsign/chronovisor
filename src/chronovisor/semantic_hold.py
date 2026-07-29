"""Compatibility alias for :mod:`chronovisor.search.semantic_hold`."""

from chronovisor.search import semantic_hold as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
