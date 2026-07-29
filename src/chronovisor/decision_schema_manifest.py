"""Compatibility alias for :mod:`chronovisor.decision.decision_schema_manifest`."""

from chronovisor.decision import decision_schema_manifest as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
