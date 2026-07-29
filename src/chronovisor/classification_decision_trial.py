"""Compatibility alias for the relocated classification decision trial."""

from chronovisor.lab import classification_decision_trial as _implementation
from chronovisor.lab._compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
