"""Compatibility alias for :mod:`chronovisor.lab.model_lab`."""

from chronovisor.lab import model_lab as _implementation
from chronovisor.lab._compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
