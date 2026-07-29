"""Compatibility alias for the relocated classification calibration."""

from chronovisor.lab import classification_calibration as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
