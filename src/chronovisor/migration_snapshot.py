"""Compatibility alias for :mod:`chronovisor.ops.migration_snapshot`."""

from chronovisor.ops import migration_snapshot as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
