"""Compatibility alias for :mod:`chronovisor.research.evidence_bundle`."""

from chronovisor.research import evidence_bundle as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
