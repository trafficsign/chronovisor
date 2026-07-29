"""Compatibility alias for :mod:`chronovisor.core.alias_store`."""

from chronovisor.core import alias_store as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
