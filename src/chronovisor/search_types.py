"""Compatibility alias for :mod:`chronovisor.search.search_types`."""

from chronovisor.search import search_types as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
