"""Compatibility alias for the relocated query2doc unseen gate."""

from chronovisor.lab import classification_query2doc_unseen as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
