"""Compatibility alias for :mod:`chronovisor.ops.background_jobs`."""

from chronovisor.ops import background_jobs as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
