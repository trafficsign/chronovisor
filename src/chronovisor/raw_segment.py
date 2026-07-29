"""Compatibility alias for :mod:`chronovisor.raw.raw_segment`."""

from chronovisor.raw import raw_segment as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
