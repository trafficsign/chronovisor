"""Compatibility alias for the relocated query2doc v2 experiment."""

from chronovisor.lab import classification_query2doc_v2 as _implementation
from chronovisor.lab._compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
