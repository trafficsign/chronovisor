"""Compatibility alias for :mod:`chronovisor.hosts.agent_save_base`."""

from chronovisor.hosts import agent_save_base as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
