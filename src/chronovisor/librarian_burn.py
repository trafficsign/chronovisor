"""Compatibility alias for :mod:`chronovisor.lab.librarian_burn`."""

from chronovisor.lab import librarian_burn as _implementation
from chronovisor.lab._compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
