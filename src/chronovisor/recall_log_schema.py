"""Compatibility alias for :mod:`chronovisor.recall.recall_log_schema`."""

from chronovisor.recall import recall_log_schema as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
