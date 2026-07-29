"""Compatibility alias for :mod:`chronovisor.decision.decision_lane_contracts`."""

from chronovisor.decision import decision_lane_contracts as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
