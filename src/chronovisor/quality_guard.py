"""Compatibility alias for :mod:`chronovisor.decision.quality_guard`."""

from chronovisor.decision import quality_guard as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
