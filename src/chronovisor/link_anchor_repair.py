"""Compatibility alias for :mod:`chronovisor.librarian.link_anchor_repair`."""

from chronovisor.librarian import link_anchor_repair as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
