"""Compatibility alias for :mod:`chronovisor.ops.page_normalize`."""

from chronovisor.ops import page_normalize as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
