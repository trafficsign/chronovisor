"""Compatibility alias for the relocated classification artifact runner."""

from chronovisor.lab import classification_artifact_runner as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
