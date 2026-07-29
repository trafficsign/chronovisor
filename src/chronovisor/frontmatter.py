"""Compatibility alias for :mod:`chronovisor.core.frontmatter`."""

from chronovisor.core import frontmatter as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
