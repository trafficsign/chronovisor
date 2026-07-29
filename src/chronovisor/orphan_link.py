"""Compatibility alias for :mod:`chronovisor.ops.orphan_link`."""

from chronovisor.ops import orphan_link as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
