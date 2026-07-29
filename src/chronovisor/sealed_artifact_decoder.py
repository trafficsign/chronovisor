"""Compatibility alias for :mod:`chronovisor.core.sealed_artifact_decoder`."""

from chronovisor.core import sealed_artifact_decoder as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
