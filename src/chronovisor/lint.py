"""Compatibility alias for :mod:`chronovisor.ops.lint`."""

from chronovisor.ops import lint as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
