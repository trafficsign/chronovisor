from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_operator_raw_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit behavior independent from the operator's live rollout mode."""

    monkeypatch.setenv("CHRONOVISOR_RAW_LAYOUT", "legacy")
