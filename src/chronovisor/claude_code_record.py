"""Compatibility alias for :mod:`chronovisor.hosts.claude_code_record`."""

from chronovisor.hosts import claude_code_record as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
