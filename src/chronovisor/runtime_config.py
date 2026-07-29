"""Compatibility alias for :mod:`chronovisor.core.runtime_config`."""

from chronovisor.core import runtime_config as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
