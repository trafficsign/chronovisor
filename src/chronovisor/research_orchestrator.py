"""Compatibility alias for :mod:`chronovisor.research.research_orchestrator`."""

from chronovisor.research import research_orchestrator as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
