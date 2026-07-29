"""Compatibility alias for :mod:`chronovisor.recall.provisional_recall`."""

from chronovisor.recall import provisional_recall as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
