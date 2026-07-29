"""Compatibility alias for the relocated anchor complement auditor."""

from chronovisor.lab import classification_anchor_complement_auditor as _implementation
from chronovisor.lab._compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
