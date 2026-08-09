from __future__ import annotations

import json
from pathlib import Path

from chronovisor.core.durable_state import write_sealed_json
from chronovisor.recall.classification_bundle import (
    activate_decision_only,
    probe_decision_only_authority,
    resolve_authority,
    rollback_authority,
)
from chronovisor.recall.classification_fixture_set import (
    create_disabled_baseline_manifest,
)


def test_disabled_baseline_is_distinct_from_missing_and_corrupt(
    tmp_path: Path,
) -> None:
    assert resolve_authority(tmp_path)["reason"] == "missing_active_pointer"
    disabled = tmp_path / "classification" / "bundles" / "disabled.json"
    create_disabled_baseline_manifest(
        tmp_path,
        a0_config={"candidate_limit": 12},
        receipt_path=disabled,
    )

    resolved = activate_decision_only(tmp_path, target_path=disabled)

    assert resolved["status"] == "disabled"
    assert resolved["reason"] == "intentional_disabled_baseline"
    assert resolved["mutation_capability"] is False


def test_rollback_disables_mutation_before_pointer_restore(tmp_path: Path) -> None:
    first = tmp_path / "classification" / "bundles" / "first.json"
    second = tmp_path / "classification" / "bundles" / "second.json"
    for path, name in ((first, "first"), (second, "second")):
        write_sealed_json(
            path,
            {
                "schema": "chronovisor.classification-disabled-baseline.v1",
                "name": name,
                "mutation_capability": False,
            },
        )
    activate_decision_only(tmp_path, target_path=first)
    activate_decision_only(tmp_path, target_path=second)

    resolved = rollback_authority(tmp_path)

    assert resolved["status"] == "disabled"
    mutation = json.loads(
        (tmp_path / "classification" / "authority" / "mutation.json").read_text()
    )
    assert mutation["enabled"] is False
    assert mutation["reason"] == "rollback-first-step"


def test_decision_only_probe_detects_mutation_breach(tmp_path: Path) -> None:
    adopted = tmp_path / "classification" / "bundles" / "adopted.json"
    write_sealed_json(
        adopted,
        {
            "schema": "chronovisor.classification-adopted-manifest.v1",
            "adopted_bundle_manifest_digest": "sha256:adopted",
            "authority": {"authority_digest": "sha256:authority"},
            "mode": "decision-only/canary",
            "mutation_capability": False,
        },
    )
    activate_decision_only(tmp_path, target_path=adopted)
    assert (
        probe_decision_only_authority(
            tmp_path,
            expected_manifest_path=adopted,
        )["status"]
        == "passed"
    )

    write_sealed_json(
        tmp_path / "classification" / "authority" / "mutation.json",
        {
            "schema": "chronovisor.classification-mutation-capability.v1",
            "enabled": True,
        },
    )
    probe = probe_decision_only_authority(
        tmp_path,
        expected_manifest_path=adopted,
    )
    assert probe["status"] == "critical-breach"
    assert probe["gates"]["mutation_disabled"] is False
