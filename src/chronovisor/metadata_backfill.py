"""Compatibility alias for :mod:`chronovisor.ops.metadata_backfill`."""

from chronovisor.ops import metadata_backfill as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
