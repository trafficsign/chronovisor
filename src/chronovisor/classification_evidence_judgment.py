"""Compatibility alias for :mod:`chronovisor.classification.classification_evidence_judgment`."""

from chronovisor.classification import classification_evidence_judgment as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
