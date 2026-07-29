"""Compatibility alias for :mod:`chronovisor.ingest.ingest_transport`."""

from chronovisor.ingest import ingest_transport as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
