"""Small path module shared by recall components without import cycles."""

from __future__ import annotations

from chronovisor.core.store import CHRONOVISOR_ROOT

RECALL_DIR = CHRONOVISOR_ROOT / "recall"
