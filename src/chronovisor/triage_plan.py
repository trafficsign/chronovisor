"""Compatibility alias for :mod:`chronovisor.ingest.triage_plan`."""

from chronovisor.ingest import triage_plan as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
