from __future__ import annotations

from chronovisor.classification_anchor_unseen import (
    MAXIMUM_CATASTROPHIC,
    MAXIMUM_HOLDS,
    MINIMUM_EXACT,
    SAMPLE_SIZE,
)


def test_unseen_gate_contract_is_stricter_than_development_tolerance() -> None:
    assert SAMPLE_SIZE == 30
    assert MINIMUM_EXACT == 27
    assert MAXIMUM_HOLDS == 3
    assert MAXIMUM_CATASTROPHIC == 0
