"""Compatibility alias for :mod:`chronovisor.librarian.collection_anomaly_worker`."""

from chronovisor.librarian import collection_anomaly_worker as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
