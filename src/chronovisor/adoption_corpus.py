"""Compatibility alias for :mod:`chronovisor.lab.adoption_corpus`."""

from chronovisor.lab import adoption_corpus as _implementation
from chronovisor.lab._compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
