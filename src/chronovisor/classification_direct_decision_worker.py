"""Compatibility alias for :mod:`chronovisor.classification.classification_direct_decision_worker`."""

from chronovisor.classification import classification_direct_decision_worker as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
