"""Compatibility alias for :mod:`chronovisor.raw.raw_replay`."""

from chronovisor.raw import raw_replay as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
