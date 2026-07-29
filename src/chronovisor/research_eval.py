"""Compatibility alias for :mod:`chronovisor.lab.research_eval`."""

from chronovisor.lab import research_eval as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
