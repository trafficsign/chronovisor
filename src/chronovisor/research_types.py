"""Compatibility alias for :mod:`chronovisor.research.research_types`."""

from chronovisor.research import research_types as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
