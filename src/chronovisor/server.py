"""Compatibility alias for :mod:`chronovisor.hosts.server`."""

from chronovisor.hosts import server as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
