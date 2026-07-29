"""Compatibility alias for :mod:`chronovisor.hosts.codex_record`."""

from chronovisor.hosts import codex_record as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
