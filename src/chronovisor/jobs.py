"""Compatibility alias for :mod:`chronovisor.core.jobs`."""

from chronovisor.core import jobs as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
