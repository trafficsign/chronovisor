"""Compatibility alias for the relocated library calibration."""

from chronovisor.lab import classification_library_calibration as _implementation
from chronovisor.lab._compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
