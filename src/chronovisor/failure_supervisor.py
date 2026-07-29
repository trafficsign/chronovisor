"""Compatibility alias for :mod:`chronovisor.decision.failure_supervisor`."""

from chronovisor.decision import failure_supervisor as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
