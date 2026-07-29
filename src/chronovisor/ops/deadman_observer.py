"""Domain alias for the package-independent deadman observer."""

from chronovisor import deadman_observer as _implementation
from chronovisor.core.compat import alias_legacy_module

alias_legacy_module(__name__, _implementation)
