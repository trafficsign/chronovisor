"""Compatibility alias for :mod:`chronovisor.search.index_store`."""

from chronovisor.search import index_store as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
