"""Compatibility alias for :mod:`chronovisor.ingest.page_registry`."""

from chronovisor.ingest import page_registry as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
