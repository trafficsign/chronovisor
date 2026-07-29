"""Compatibility alias for :mod:`chronovisor.research.deep_retrieval_worker`."""

from chronovisor.research import deep_retrieval_worker as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
