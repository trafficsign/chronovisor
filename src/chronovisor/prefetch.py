"""Compatibility alias for :mod:`chronovisor.search.prefetch`."""

from chronovisor.search import prefetch as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
