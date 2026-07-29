"""Compatibility alias for :mod:`chronovisor.search.semantic_service`."""

from chronovisor.search import semantic_service as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
