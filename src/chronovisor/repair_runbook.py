"""Compatibility alias for :mod:`chronovisor.ops.repair_runbook`."""

from chronovisor.ops import repair_runbook as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
