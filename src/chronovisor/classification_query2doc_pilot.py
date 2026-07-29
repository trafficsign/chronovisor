"""Compatibility alias for the relocated query2doc pilot."""

from chronovisor.lab import classification_query2doc_pilot as _implementation
from chronovisor.lab._compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
