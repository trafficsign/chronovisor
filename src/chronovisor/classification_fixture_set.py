"""Compatibility alias for the relocated classification fixture support."""

from chronovisor.lab import classification_fixture_set as _implementation
from chronovisor.lab._compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
