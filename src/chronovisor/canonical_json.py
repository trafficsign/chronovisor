"""Compatibility alias for :mod:`chronovisor.core.canonical_json`."""

from chronovisor.core import canonical_json as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
