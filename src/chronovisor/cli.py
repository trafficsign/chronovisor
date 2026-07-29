"""Compatibility alias for :mod:`chronovisor.hosts.cli`."""

from chronovisor.hosts import cli as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
