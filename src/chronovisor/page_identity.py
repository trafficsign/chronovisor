"""Compatibility alias for :mod:`chronovisor.core.page_identity`."""

from chronovisor.core import page_identity as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
