"""Compatibility alias for :mod:`chronovisor.ingest.ingest_triage`."""

from chronovisor.ingest import ingest_triage as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
