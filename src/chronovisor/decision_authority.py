"""Compatibility alias for :mod:`chronovisor.decision.decision_authority`."""

from chronovisor.decision import decision_authority as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
