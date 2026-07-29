"""Compatibility alias for :mod:`chronovisor.core.timeutil`."""

from chronovisor.core import timeutil as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
