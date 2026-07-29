"""Compatibility alias for :mod:`chronovisor.search.semantic_index`."""

from chronovisor.search import semantic_index as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
