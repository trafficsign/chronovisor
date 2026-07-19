"""Durable continuation, repair, and no-progress recovery records."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from chronovisor.canonical_json import canonical_json_sha256_stringifying_strict
from chronovisor.ingest_review_plan import IngestReviewShardPlan
from chronovisor.ingest_review_store import write_ingest_artifact
from chronovisor.ingest_schemas import INGEST_REVIEW_SHARD_SCHEMA_VERSION


def seal_ingest_review_repair_transition(
    path: Path,
    *,
    source_key: str,
    previous_full_proposal_sha256: str,
    repaired_operations_sha256: str,
) -> str | None:
    """Allow one idempotent exact repair transition per source raw."""

    if (
        re.fullmatch(r"[0-9a-f]{64}", previous_full_proposal_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", repaired_operations_sha256) is None
    ):
        return "exact repair transition identity is invalid"
    identity = {
        "source_key": source_key,
        "previous_full_proposal_sha256": previous_full_proposal_sha256,
        "repaired_operations_sha256": repaired_operations_sha256,
    }
    payload = {
        "schema_version": INGEST_REVIEW_SHARD_SCHEMA_VERSION,
        "kind": "ingest_review_exact_repair_transition",
        **identity,
        "identity_sha256": canonical_json_sha256_stringifying_strict(identity),
    }
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return f"exact repair transition is unreadable: {type(exc).__name__}: {exc}"
        if existing == payload:
            return None
        if not isinstance(existing, dict):
            return "exact repair transition binding is invalid"
        existing_identity = {
            key: existing.get(key)
            for key in (
                "source_key",
                "previous_full_proposal_sha256",
                "repaired_operations_sha256",
            )
        }
        if (
            set(existing) != set(payload)
            or existing.get("schema_version") != INGEST_REVIEW_SHARD_SCHEMA_VERSION
            or existing.get("kind") != "ingest_review_exact_repair_transition"
            or existing.get("identity_sha256")
            != canonical_json_sha256_stringifying_strict(existing_identity)
        ):
            return "exact repair transition binding is invalid"
        return "exact repair transition limit exceeded"
    try:
        write_ingest_artifact(path, payload)
        readback = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return (
            "exact repair transition write/readback failed: "
            f"{type(exc).__name__}: {exc}"
        )
    if readback != payload:
        return "exact repair transition readback verification failed"
    return None


def persist_ingest_review_continuation_marker(
    path: Path,
    *,
    source_key: str,
    plan: IngestReviewShardPlan,
    reason: str,
    previous_full_proposal_sha256: str,
    previous_authority: dict[str, Any] | str,
    current_authority: dict[str, Any],
) -> str | None:
    """Seal the one exceptional zero-approval resume permission."""

    if reason not in {
        "exact_repair_reseed",
        "authority_epoch_reseed",
        "router_config_reseed",
    }:
        return "ingest review continuation marker reason is invalid"
    if re.fullmatch(r"[0-9a-f]{64}", previous_full_proposal_sha256) is None or (
        reason == "exact_repair_reseed"
        and previous_full_proposal_sha256 == plan.full_proposal_sha256
    ):
        return "ingest review continuation marker transition is invalid"
    if isinstance(previous_authority, str):
        if re.fullmatch(r"[0-9a-f]{64}", previous_authority) is None:
            return "ingest review continuation previous authority is invalid"
        previous_authority_sha256 = previous_authority
    else:
        previous_authority_sha256 = canonical_json_sha256_stringifying_strict(
            previous_authority
        )
    transition = {
        "source_key": source_key,
        "previous_full_proposal_sha256": previous_full_proposal_sha256,
        "full_proposal_sha256": plan.full_proposal_sha256,
        "manifest_sha256": plan.manifest_sha256,
        "reason": reason,
        "previous_authority_sha256": previous_authority_sha256,
        "current_authority_sha256": canonical_json_sha256_stringifying_strict(
            current_authority
        ),
    }
    payload = {
        "schema_version": INGEST_REVIEW_SHARD_SCHEMA_VERSION,
        "kind": "ingest_review_zero_progress_continuation",
        **transition,
        "transition_sha256": canonical_json_sha256_stringifying_strict(transition),
        "state": "available",
    }
    try:
        write_ingest_artifact(path, payload)
        readback = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return f"continuation marker write/readback failed: {type(exc).__name__}: {exc}"
    if readback != payload:
        return "continuation marker readback verification failed"
    return None


def load_ingest_review_continuation_marker(
    path: Path,
    *,
    source_key: str,
    plan: IngestReviewShardPlan,
    authority: dict[str, Any],
    allow_stale_identity: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"continuation marker is unreadable: {type(exc).__name__}: {exc}"
    expected_fields = {
        "schema_version",
        "kind",
        "source_key",
        "previous_full_proposal_sha256",
        "full_proposal_sha256",
        "manifest_sha256",
        "reason",
        "previous_authority_sha256",
        "current_authority_sha256",
        "transition_sha256",
        "state",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_fields
        or payload.get("schema_version") != INGEST_REVIEW_SHARD_SCHEMA_VERSION
        or payload.get("kind") != "ingest_review_zero_progress_continuation"
        or payload.get("source_key") != source_key
        or not isinstance(payload.get("previous_full_proposal_sha256"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}", str(payload.get("previous_full_proposal_sha256"))
        )
        is None
        or payload.get("reason")
        not in {
            "exact_repair_reseed",
            "authority_epoch_reseed",
            "router_config_reseed",
        }
        or payload.get("state") not in {"available", "claimed"}
    ):
        return None, "continuation marker schema is invalid"
    transition = {
        key: payload.get(key)
        for key in (
            "source_key",
            "previous_full_proposal_sha256",
            "full_proposal_sha256",
            "manifest_sha256",
            "reason",
            "previous_authority_sha256",
            "current_authority_sha256",
        )
    }
    if (
        not isinstance(payload.get("previous_authority_sha256"), str)
        or not isinstance(payload.get("current_authority_sha256"), str)
        or re.fullmatch(
            r"[0-9a-f]{64}", str(payload.get("previous_authority_sha256"))
        )
        is None
        or re.fullmatch(
            r"[0-9a-f]{64}", str(payload.get("current_authority_sha256"))
        )
        is None
        or payload.get("transition_sha256")
        != canonical_json_sha256_stringifying_strict(transition)
        or (
            payload.get("reason") == "exact_repair_reseed"
            and payload.get("previous_full_proposal_sha256")
            == plan.full_proposal_sha256
        )
    ):
        return None, "continuation marker transition binding is invalid"
    if payload.get("full_proposal_sha256") != plan.full_proposal_sha256:
        return None, None
    if not allow_stale_identity and (
        payload.get("current_authority_sha256")
        != canonical_json_sha256_stringifying_strict(authority)
        or payload.get("manifest_sha256") != plan.manifest_sha256
    ):
        return None, None
    return payload, None


def consume_ingest_review_continuation_marker(
    path: Path,
    marker: dict[str, Any],
) -> str | None:
    consumed = dict(marker)
    consumed["state"] = "claimed"
    try:
        write_ingest_artifact(path, consumed)
        readback = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return f"continuation marker consume failed: {type(exc).__name__}: {exc}"
    if readback != consumed:
        return "continuation marker consume readback verification failed"
    return None


def persist_ingest_review_stall(
    path: Path,
    *,
    source_key: str,
    plan: IngestReviewShardPlan,
    authority: dict[str, Any],
    approved_indices: tuple[int, ...],
) -> str | None:
    """Tombstone one same-plan, same-authority no-progress review epoch."""

    identity = {
        "source_key": source_key,
        "full_proposal_sha256": plan.full_proposal_sha256,
        "manifest_sha256": plan.manifest_sha256,
        "authority_sha256": canonical_json_sha256_stringifying_strict(authority),
        "approved_indices": list(approved_indices),
    }
    payload = {
        "schema_version": INGEST_REVIEW_SHARD_SCHEMA_VERSION,
        "kind": "ingest_review_no_progress_stall",
        **identity,
        "identity_sha256": canonical_json_sha256_stringifying_strict(identity),
    }
    try:
        write_ingest_artifact(path, payload)
        readback = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return f"review stall write/readback failed: {type(exc).__name__}: {exc}"
    if readback != payload:
        return "review stall readback verification failed"
    return None


def matching_ingest_review_stall_error(
    path: Path,
    *,
    source_key: str,
    plan: IngestReviewShardPlan,
    authority: dict[str, Any],
    approved_indices: tuple[int, ...],
) -> str | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return f"review stall is unreadable: {type(exc).__name__}: {exc}"
    identity = {
        "source_key": payload.get("source_key") if isinstance(payload, dict) else None,
        "full_proposal_sha256": (
            payload.get("full_proposal_sha256") if isinstance(payload, dict) else None
        ),
        "manifest_sha256": (
            payload.get("manifest_sha256") if isinstance(payload, dict) else None
        ),
        "authority_sha256": (
            payload.get("authority_sha256") if isinstance(payload, dict) else None
        ),
        "approved_indices": (
            payload.get("approved_indices") if isinstance(payload, dict) else None
        ),
    }
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "schema_version",
            "kind",
            "source_key",
            "full_proposal_sha256",
            "manifest_sha256",
            "authority_sha256",
            "approved_indices",
            "identity_sha256",
        }
        or payload.get("schema_version") != INGEST_REVIEW_SHARD_SCHEMA_VERSION
        or payload.get("kind") != "ingest_review_no_progress_stall"
        or payload.get("identity_sha256")
        != canonical_json_sha256_stringifying_strict(identity)
    ):
        return "review stall binding is invalid"
    expected_identity = {
        "source_key": source_key,
        "full_proposal_sha256": plan.full_proposal_sha256,
        "manifest_sha256": plan.manifest_sha256,
        "authority_sha256": canonical_json_sha256_stringifying_strict(authority),
        "approved_indices": list(approved_indices),
    }
    if identity == expected_identity:
        return "same ingest review plan and authority previously made no progress"
    return None
