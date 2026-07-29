"""Compatibility alias for :mod:`chronovisor.core.store`."""

from chronovisor.core import store as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
