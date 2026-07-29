"""Compatibility alias for :mod:`chronovisor.ops.distill`."""

from chronovisor.ops import distill as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
