"""Expose the package-independent observer under the operations namespace."""

import sys

from chronovisor import deadman_observer as _implementation

sys.modules[__name__] = _implementation
