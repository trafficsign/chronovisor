"""Compatibility alias for :mod:`chronovisor.decision.local_repair`."""

from chronovisor.decision import local_repair as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
