"""Compatibility alias for :mod:`chronovisor.decision.decision_lane_prompts`."""

from chronovisor.decision import decision_lane_prompts as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
