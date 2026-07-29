"""Compatibility alias for :mod:`chronovisor.recall.content_correction`."""

from chronovisor.recall import content_correction as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
