"""Compatibility alias for :mod:`chronovisor.research.research_auditor`."""

from chronovisor.research import research_auditor as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
