"""Compatibility alias for :mod:`chronovisor.search.cofire`."""

from chronovisor.search import cofire as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
