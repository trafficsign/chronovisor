from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor.classification.classification import ClassificationError
from chronovisor.classification.classification_retention import (
    build_audit_retention_manifest,
    refuse_automatic_audit_deletion,
    required_update_validation,
    validate_update_validation,
)


def test_semantic_updates_require_fresh_sentinel() -> None:
    semantic = required_update_validation("source-or-index-semantic")
    model = required_update_validation("model-policy-taxonomy")

    assert semantic["fixture_requirement"] == ("new-one-time-300-evaluable-sentinel")
    assert semantic["sentinel_reuse_forbidden"] is True
    assert model["fixture_requirement"] == "new-200-dev-and-300-holdout-epoch"


def test_audit_retention_deduplicates_content(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text("same", encoding="utf-8")
    right.write_text("same", encoding="utf-8")

    result = build_audit_retention_manifest(
        tmp_path / "retention.json",
        artifact_paths=[left, right],
    )

    assert result["deduplicated_artifact_count"] == 1
    assert result["source_bytes"] == 4
    assert result["object_count"] == 1
    assert Path(result["artifacts"][0]["stored_path"]).is_file()
    assert result["delete_automatically"] is False


def test_semantic_update_validation_fails_closed_without_fresh_sentinel() -> None:
    rejected = validate_update_validation(
        "source-or-index-semantic",
        {
            "locked_before_results": True,
            "one_time": True,
            "sentinel_reused": True,
            "group_disjoint": True,
            "evaluable_n": 300,
            "severe_error_count": 0,
            "expected_hold_escape_count": 0,
            "system_gates_passed": True,
            "recall_gate_passed": True,
            "powered": True,
        },
    )
    assert rejected["status"] == "inactive-manual-review"
    assert rejected["gates"]["sentinel_not_reused"] is False


def test_audit_deletion_requires_explicit_external_retention_decision() -> None:
    with pytest.raises(ClassificationError, match="user-approved"):
        refuse_automatic_audit_deletion()
