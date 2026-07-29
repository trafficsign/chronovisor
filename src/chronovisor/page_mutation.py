"""Compatibility alias for :mod:`chronovisor.ingest.page_mutation`."""

from chronovisor.ingest import page_mutation as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
