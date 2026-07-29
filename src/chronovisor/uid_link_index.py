"""Compatibility alias for :mod:`chronovisor.ingest.uid_link_index`."""

from chronovisor.ingest import uid_link_index as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
