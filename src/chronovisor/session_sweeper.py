"""Compatibility alias for :mod:`chronovisor.ops.session_sweeper`."""

from chronovisor.ops import session_sweeper as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
