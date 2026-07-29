"""Compatibility alias for :mod:`chronovisor.ingest.ingest_review`."""

from chronovisor.ingest import ingest_review as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
