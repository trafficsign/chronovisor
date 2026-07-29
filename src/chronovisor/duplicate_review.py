"""Compatibility alias for :mod:`chronovisor.recall.duplicate_review`."""

from chronovisor.recall import duplicate_review as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
