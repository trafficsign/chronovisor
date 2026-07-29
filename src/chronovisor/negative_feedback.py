"""Compatibility alias for :mod:`chronovisor.recall.negative_feedback`."""

from chronovisor.recall import negative_feedback as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
