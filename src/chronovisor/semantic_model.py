"""Compatibility alias for :mod:`chronovisor.search.semantic_model`."""

from chronovisor.search import semantic_model as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
