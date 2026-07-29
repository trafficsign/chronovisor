"""Compatibility alias for :mod:`chronovisor.recall.claims`."""

from chronovisor.recall import claims as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
