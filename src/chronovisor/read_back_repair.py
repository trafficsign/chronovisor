"""Compatibility alias for :mod:`chronovisor.ingest.read_back_repair`."""

from chronovisor.ingest import read_back_repair as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
