"""Compatibility alias for :mod:`chronovisor.ops.lint_repair`."""

from chronovisor.ops import lint_repair as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
