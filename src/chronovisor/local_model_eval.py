"""Compatibility alias for :mod:`chronovisor.lab.local_model_eval`."""

from chronovisor.lab import local_model_eval as _implementation
from chronovisor.lab._compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
