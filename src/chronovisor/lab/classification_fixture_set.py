"""Compatibility shim for the classification-owned fixture set."""

from __future__ import annotations

import sys

from chronovisor.classification import classification_fixture_set as _implementation

sys.modules[__name__] = _implementation
