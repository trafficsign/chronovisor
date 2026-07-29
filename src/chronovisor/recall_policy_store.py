"""Compatibility alias for :mod:`chronovisor.recall.recall_policy_store`."""

from chronovisor.recall import recall_policy_store as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
