"""Compatibility alias for :mod:`chronovisor.raw.save_transaction`."""

from chronovisor.raw import save_transaction as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
