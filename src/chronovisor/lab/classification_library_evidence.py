"""Compatibility shim for classification-owned library evidence."""

from __future__ import annotations

import sys

from chronovisor.classification import (
    classification_library_evidence as _implementation,
)

sys.modules[__name__] = _implementation
