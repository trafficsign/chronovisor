"""Compatibility alias for :mod:`chronovisor.recall.recall_auditor`."""

from chronovisor.recall import recall_auditor as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
