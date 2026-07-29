"""Compatibility alias for :mod:`chronovisor.search.lexical_index`."""

from chronovisor.search import lexical_index as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
