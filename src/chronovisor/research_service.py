"""Compatibility alias for :mod:`chronovisor.research.research_service`."""

from chronovisor.research import research_service as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
