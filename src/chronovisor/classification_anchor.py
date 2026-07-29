"""Compatibility alias for :mod:`chronovisor.classification.classification_anchor`."""

from chronovisor.classification import classification_anchor as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
