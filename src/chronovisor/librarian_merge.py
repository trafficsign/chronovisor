"""Compatibility alias for :mod:`chronovisor.librarian.librarian_merge`."""

from chronovisor.librarian import librarian_merge as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
