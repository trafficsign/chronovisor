"""Compatibility alias for :mod:`chronovisor.ingest.ingest_prepare`."""

from chronovisor.ingest import ingest_prepare as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
