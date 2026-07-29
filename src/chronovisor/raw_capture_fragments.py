"""Compatibility alias for :mod:`chronovisor.raw.raw_capture_fragments`."""

from chronovisor.raw import raw_capture_fragments as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
