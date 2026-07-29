"""Compatibility alias for :mod:`chronovisor.ops.auto_apply_error_supervisor`."""

from chronovisor.ops import auto_apply_error_supervisor as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
