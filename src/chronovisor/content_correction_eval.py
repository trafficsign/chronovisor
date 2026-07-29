"""Compatibility alias for :mod:`chronovisor.recall.content_correction_eval`."""

from chronovisor.recall import content_correction_eval as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
