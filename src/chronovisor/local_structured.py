"""Compatibility alias for :mod:`chronovisor.decision.local_structured`."""

from chronovisor.decision import local_structured as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
