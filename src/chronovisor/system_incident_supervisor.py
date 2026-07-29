"""Compatibility alias for :mod:`chronovisor.ops.system_incident_supervisor`."""

from chronovisor.ops import system_incident_supervisor as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
