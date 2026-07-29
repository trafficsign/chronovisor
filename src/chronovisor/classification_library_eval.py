"""Compatibility alias for the relocated library evaluation."""

from chronovisor.lab import classification_library_eval as _implementation
from chronovisor.lab._compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
