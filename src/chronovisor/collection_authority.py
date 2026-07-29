"""Compatibility alias for :mod:`chronovisor.librarian.collection_authority`."""

from chronovisor.librarian import collection_authority as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
