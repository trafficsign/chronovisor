"""Update-validation and content-addressed audit retention policy."""

from __future__ import annotations

import gzip
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.classification.classification import ClassificationError
from chronovisor.classification.classification_fixture_contract import (
    sha256_bytes,
    sha256_file,
)
from chronovisor.core.durable_state import write_sealed_json

RETENTION_SCHEMA = "chronovisor.classification-audit-retention.v1"
UPDATE_POLICY_SCHEMA = "chronovisor.classification-update-policy.v1"
AUDIT_ANNUAL_BUDGET = 1024**3


def required_update_validation(change_kind: str) -> dict[str, Any]:
    policies = {
        "bit-identical": {
            "auto_flip": True,
            "fixture_requirement": "equivalence-proof",
        },
        "metadata-only": {
            "auto_flip": True,
            "fixture_requirement": "nonsemantic-equivalence-proof",
        },
        "source-or-index-semantic": {
            "auto_flip": False,
            "fixture_requirement": "new-one-time-300-evaluable-sentinel",
        },
        "model-policy-taxonomy": {
            "auto_flip": False,
            "fixture_requirement": "new-200-dev-and-300-holdout-epoch",
        },
    }
    if change_kind not in policies:
        raise ClassificationError("unknown classification update kind")
    return {
        "schema": UPDATE_POLICY_SCHEMA,
        "change_kind": change_kind,
        **policies[change_kind],
        "result_dependent_replenishment_forbidden": True,
        "sentinel_reuse_forbidden": change_kind == "source-or-index-semantic",
    }


def validate_update_validation(
    change_kind: str,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless an update carries the pre-required independent evidence."""

    policy = required_update_validation(change_kind)
    if change_kind == "bit-identical":
        gates = {
            "equivalence_proof": receipt.get("bit_identical") is True,
        }
    elif change_kind == "metadata-only":
        gates = {
            "equivalence_proof": receipt.get("nonsemantic_equivalence") is True,
            "semantic_digest_unchanged": (
                receipt.get("before_semantic_digest")
                == receipt.get("after_semantic_digest")
                and bool(receipt.get("before_semantic_digest"))
            ),
        }
    elif change_kind == "source-or-index-semantic":
        gates = {
            "locked_before_results": receipt.get("locked_before_results") is True,
            "one_time_sentinel": receipt.get("one_time") is True,
            "sentinel_not_reused": receipt.get("sentinel_reused") is False,
            "group_disjoint": receipt.get("group_disjoint") is True,
            "evaluable_n": int(receipt.get("evaluable_n") or 0) >= 300,
            "severe_zero": int(receipt.get("severe_error_count") or -1) == 0,
            "expected_hold_escape_zero": (
                int(receipt.get("expected_hold_escape_count") or -1) == 0
            ),
            "system_gates": receipt.get("system_gates_passed") is True,
            "recall_gate": receipt.get("recall_gate_passed") is True,
            "powered": receipt.get("powered") is True,
        }
    else:
        gates = {
            "new_fixture_epoch": receipt.get("new_fixture_epoch") is True,
            "group_disjoint": receipt.get("group_disjoint") is True,
            "dev_n": int(receipt.get("dev_n") or 0) >= 200,
            "holdout_n": int(receipt.get("holdout_n") or 0) >= 300,
            "holdout_one_time": receipt.get("holdout_one_time") is True,
            "system_gates": receipt.get("system_gates_passed") is True,
            "recall_gate": receipt.get("recall_gate_passed") is True,
            "powered": receipt.get("powered") is True,
        }
    return {
        **policy,
        "status": "passed" if all(gates.values()) else "inactive-manual-review",
        "gates": gates,
    }


def refuse_automatic_audit_deletion() -> None:
    raise ClassificationError(
        "audit artifacts require user-approved external retention or budget change"
    )


def build_audit_retention_manifest(
    output_path: Path,
    *,
    artifact_paths: Sequence[Path],
) -> dict[str, Any]:
    object_root = output_path.parent / "objects"
    object_root.mkdir(parents=True, exist_ok=True)
    unique: dict[str, dict[str, Any]] = {}
    for path in artifact_paths:
        if not path.is_file():
            continue
        digest = sha256_file(path)
        raw = path.read_bytes()
        compressed = gzip.compress(raw, compresslevel=9, mtime=0)
        object_path = object_root / f"{digest.removeprefix('sha256:')}.gz"
        if not object_path.exists():
            temporary = object_path.with_suffix(".gz.tmp")
            temporary.write_bytes(compressed)
            temporary.replace(object_path)
        stored_digest = sha256_file(object_path)
        row = unique.setdefault(
            digest,
            {
                "sha256": digest,
                "source_size": len(raw),
                "stored_sha256": stored_digest,
                "stored_size": object_path.stat().st_size,
                "stored_path": str(object_path),
                "paths": [],
            },
        )
        if row["stored_sha256"] != stored_digest:
            raise ClassificationError("audit object digest collision")
        row["paths"].append(str(path))
    all_objects = sorted(object_root.glob("*.gz"))
    stored_total = sum(path.stat().st_size for path in all_objects)
    source_total = sum(int(row["source_size"]) for row in unique.values())
    payload = {
        "schema": RETENTION_SCHEMA,
        "content_addressed": True,
        "compression": "deterministic-gzip-9",
        "retention_year": datetime.now(UTC).year,
        "deduplicated_artifact_count": len(unique),
        "source_bytes": source_total,
        "bytes": stored_total,
        "object_count": len(all_objects),
        "object_set_sha256": sha256_bytes(
            "\n".join(
                f"{path.name}:{sha256_file(path)}:{path.stat().st_size}"
                for path in all_objects
            ).encode("utf-8")
        ),
        "annual_budget_bytes": AUDIT_ANNUAL_BUDGET,
        "warning": stored_total >= int(AUDIT_ANNUAL_BUDGET * 0.8),
        "adoption_blocked": stored_total > AUDIT_ANNUAL_BUDGET,
        "delete_automatically": False,
        "artifacts": sorted(unique.values(), key=lambda row: str(row["sha256"])),
    }
    write_sealed_json(output_path, payload, backup=True)
    return payload
