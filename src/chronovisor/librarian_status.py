"""Compatibility alias for :mod:`chronovisor.librarian.librarian_status`."""

from chronovisor.librarian import librarian_status as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
