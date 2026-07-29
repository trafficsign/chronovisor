"""Compatibility alias for :mod:`chronovisor.ops.autonomy`."""

from chronovisor.ops import autonomy as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
