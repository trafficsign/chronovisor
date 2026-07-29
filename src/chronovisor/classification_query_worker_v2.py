"""Compatibility alias for :mod:`chronovisor.classification.classification_query_worker_v2`."""

from chronovisor.classification import classification_query_worker_v2 as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
