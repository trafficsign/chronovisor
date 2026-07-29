"""Compatibility alias for :mod:`chronovisor.recall.recall_auto_apply`."""

from chronovisor.recall import recall_auto_apply as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
