"""Compatibility alias for :mod:`chronovisor.research.web_provider`."""

from chronovisor.research import web_provider as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
