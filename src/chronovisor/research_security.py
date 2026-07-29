"""Compatibility alias for :mod:`chronovisor.research.research_security`."""

from chronovisor.research import research_security as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
