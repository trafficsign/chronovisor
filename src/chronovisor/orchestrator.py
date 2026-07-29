"""Compatibility alias for :mod:`chronovisor.ingest.orchestrator`."""

from chronovisor.ingest import orchestrator as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
