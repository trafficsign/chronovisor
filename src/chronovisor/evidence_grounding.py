"""Compatibility alias for :mod:`chronovisor.research.evidence_grounding`."""

from chronovisor.research import evidence_grounding as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
