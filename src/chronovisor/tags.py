"""Compatibility alias for :mod:`chronovisor.librarian.tags`."""

from chronovisor.librarian import tags as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
