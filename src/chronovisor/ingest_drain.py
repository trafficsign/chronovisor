"""Compatibility alias for :mod:`chronovisor.ingest.ingest_drain`."""

from chronovisor.ingest import ingest_drain as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
