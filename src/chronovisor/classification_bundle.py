"""Compatibility alias for :mod:`chronovisor.classification.classification_bundle`."""

from chronovisor.classification import classification_bundle as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
