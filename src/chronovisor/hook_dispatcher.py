"""Compatibility alias for :mod:`chronovisor.hosts.hook_dispatcher`."""

from chronovisor.hosts import hook_dispatcher as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
