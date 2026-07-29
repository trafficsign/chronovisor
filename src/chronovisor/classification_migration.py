"""Compatibility alias for the relocated classification migration."""

from chronovisor.lab import classification_migration as _implementation
from chronovisor.lab._compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
