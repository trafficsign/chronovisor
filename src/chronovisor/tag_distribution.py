"""Compatibility alias for :mod:`chronovisor.librarian.tag_distribution`."""

from chronovisor.librarian import tag_distribution as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
