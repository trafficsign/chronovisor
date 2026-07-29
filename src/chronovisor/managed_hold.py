"""Compatibility alias for :mod:`chronovisor.librarian.managed_hold`."""

from chronovisor.librarian import managed_hold as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
