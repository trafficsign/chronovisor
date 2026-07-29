"""Compatibility alias for :mod:`chronovisor.librarian.merge_ledger`."""

from chronovisor.librarian import merge_ledger as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
