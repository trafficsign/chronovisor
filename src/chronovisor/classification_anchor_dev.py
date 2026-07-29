"""Compatibility alias for the relocated anchor development gate."""

from chronovisor.lab import classification_anchor_dev as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
