"""Compatibility alias for the relocated anchor-set development gate."""

from chronovisor.lab import classification_anchor_set_dev as _implementation
from chronovisor.lab._compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
