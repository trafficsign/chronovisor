"""Compatibility alias for :mod:`chronovisor.decision.decision_artifact`."""

from chronovisor.decision import decision_artifact as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
