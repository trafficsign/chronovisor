"""Compatibility alias for :mod:`chronovisor.search.embedding`."""

from chronovisor.search import embedding as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
