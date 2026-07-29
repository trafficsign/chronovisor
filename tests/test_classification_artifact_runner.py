from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor.lab import classification_migration
from chronovisor.lab.classification_artifact_runner import (
    _resolver_drill,
    storage_manifest,
)
from chronovisor.classification.classification_bundle import (
    ADOPTED_MANIFEST_SCHEMA,
    activate_decision_only,
)
from chronovisor.lab.classification_fixture_set import create_disabled_baseline_manifest
from chronovisor.core.durable_state import write_sealed_json


def test_decision_only_pointer_blocks_mutating_migration(
    tmp_path: Path,
) -> None:
    disabled = tmp_path / "classification" / "bundles" / "disabled.json"
    create_disabled_baseline_manifest(
        tmp_path,
        a0_config={"candidate_limit": 12},
        receipt_path=disabled,
    )
    activate_decision_only(tmp_path, target_path=disabled)

    with pytest.raises(RuntimeError, match="decision-only"):
        classification_migration.run_full_model_shadow(tmp_path)


def test_decision_only_pointer_blocks_page_metadata_migration(
    tmp_path: Path,
) -> None:
    adopted = tmp_path / "classification" / "bundles" / "adopted.json"
    write_sealed_json(
        adopted,
        {
            "schema": ADOPTED_MANIFEST_SCHEMA,
            "candidate_bundle_path": "synthetic",
            "adoption_payload": {"adoption_policy": {"authority_epoch": 3}},
            "authority": {"authority_digest": "sha256:synthetic"},
            "mutation_capability": False,
        },
    )
    activate_decision_only(tmp_path, target_path=adopted)

    with pytest.raises(RuntimeError, match="decision-only"):
        classification_migration.migrate_active_metadata(tmp_path)


def test_storage_manifest_separates_working_and_audit(tmp_path: Path) -> None:
    working = tmp_path / "working.bin"
    audit = tmp_path / "audit.bin"
    working.write_bytes(b"x" * 10)
    audit.write_bytes(b"y" * 20)

    result = storage_manifest(
        working_paths=[working],
        audit_paths=[audit],
    )

    assert result["working_set_bytes"] == 10
    assert result["audit_store_bytes"] == 20
    assert result["working_set_passed"] is True


def test_resolver_drill_proves_disabled_rollback_and_fail_closed(
    tmp_path: Path,
) -> None:
    disabled = tmp_path / "classification" / "bundles" / "disabled.json"
    create_disabled_baseline_manifest(
        tmp_path,
        a0_config={"candidate_limit": 12},
        receipt_path=disabled,
    )

    result = _resolver_drill(disabled)

    assert result["status"] == "passed"
    assert all(result["gates"].values())
    assert result["rollback_seconds"] <= 60
