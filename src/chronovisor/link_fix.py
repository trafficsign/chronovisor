"""Compatibility alias for :mod:`chronovisor.core.link_fix`."""

from chronovisor.core import link_fix as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
