"""Compatibility alias for :mod:`chronovisor.classification.classification_hierarchy`."""

from chronovisor.classification import classification_hierarchy as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
