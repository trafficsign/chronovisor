"""Compatibility alias for :mod:`chronovisor.recall.feedback_ledger`."""

from chronovisor.recall import feedback_ledger as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
