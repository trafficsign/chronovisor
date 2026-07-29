"""Compatibility alias for the relocated second-anchor auditor."""

from chronovisor.lab import classification_anchor_second_auditor as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
