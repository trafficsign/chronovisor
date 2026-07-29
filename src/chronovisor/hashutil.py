"""Compatibility alias for :mod:`chronovisor.core.hashutil`."""

from chronovisor.core import hashutil as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
