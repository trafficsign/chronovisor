"""Batch-wide ingest prerequisite tests."""

from __future__ import annotations

import pytest

from chronovisor.ingest import ingest_review_authority
from chronovisor.ingest import orchestrator


def test_authority_preflight_validates_the_adopted_batch_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = {
        "router": {
            "artifact_sha256": "b" * 64,
        }
    }
    monkeypatch.setattr(
        ingest_review_authority,
        "current_ingest_review_authority",
        lambda *, injected_reviewer: (
            authority,
            None,
        ),
    )
    monkeypatch.setattr(
        ingest_review_authority,
        "ingest_review_authority_shape_error",
        lambda value: None if value is authority else "wrong authority",
    )

    result = orchestrator.ingest_authority_preflight()

    assert result == {
        "ok": True,
        "status": "ready",
        "blocked_by": None,
        "retryable": False,
        "error": None,
        "artifact_sha256": "b" * 64,
    }


def test_authority_preflight_fails_closed_on_invalid_adoption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ingest_review_authority,
        "current_ingest_review_authority",
        lambda *, injected_reviewer: (
            None,
            "adoption_artifact_invalid:policy version mismatch",
        ),
    )

    result = orchestrator.ingest_authority_preflight()

    assert result["ok"] is False
    assert result["blocked_by"] == "decision_authority"
    assert result["retryable"] is True
    assert result["error"] == (
        "local consensus authority unavailable: "
        "adoption_artifact_invalid:policy version mismatch"
    )
