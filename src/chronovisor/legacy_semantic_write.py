"""Compatibility alias for :mod:`chronovisor.raw.legacy_semantic_write`."""

from chronovisor.raw import legacy_semantic_write as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
