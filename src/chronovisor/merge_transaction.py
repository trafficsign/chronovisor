"""Compatibility alias for :mod:`chronovisor.librarian.merge_transaction`."""

from chronovisor.librarian import merge_transaction as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
