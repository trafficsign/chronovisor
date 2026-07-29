"""Compatibility alias for the relocated Annif experiment."""

from chronovisor.lab import classification_annif as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
