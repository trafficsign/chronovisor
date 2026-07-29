"""Compatibility alias for :mod:`chronovisor.decision.decision_policy`."""

from chronovisor.decision import decision_policy as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
