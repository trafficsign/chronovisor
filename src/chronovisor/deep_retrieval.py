"""Compatibility alias for :mod:`chronovisor.research.deep_retrieval`."""

from chronovisor.research import deep_retrieval as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
