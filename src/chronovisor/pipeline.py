"""Compatibility alias for :mod:`chronovisor.search.pipeline`."""

from chronovisor.search import pipeline as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
