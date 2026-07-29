"""Compatibility alias for :mod:`chronovisor.ops.golden_expand`."""

from chronovisor.ops import golden_expand as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
