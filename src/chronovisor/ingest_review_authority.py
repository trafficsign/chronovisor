"""Compatibility alias for :mod:`chronovisor.ingest.ingest_review_authority`."""

from chronovisor.ingest import ingest_review_authority as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
