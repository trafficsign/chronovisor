"""Compatibility alias for the relocated classification pilot v2."""

from chronovisor.lab import classification_pilot_v2 as _implementation
from chronovisor.lab._compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
