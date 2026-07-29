"""Compatibility alias for :mod:`chronovisor.ingest.read_back_integrity`."""

from chronovisor.ingest import read_back_integrity as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
