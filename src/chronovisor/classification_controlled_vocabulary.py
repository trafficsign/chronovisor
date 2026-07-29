"""Compatibility alias for :mod:`chronovisor.classification.classification_controlled_vocabulary`."""

from chronovisor.classification import classification_controlled_vocabulary as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
