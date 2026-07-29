"""Durable path, codec, and readback contracts for ingest review artifacts."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Callable, TypeVar

from chronovisor.decision import decision_authority
from chronovisor.core.canonical_json import canonical_json_sha256_stringifying_strict
from chronovisor.ingest.ingest_schemas import (
    INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION,
    INGEST_FRONTIER_REVIEW_ARTIFACT_SCHEMA_VERSION,
)

AuthorityValidator = Callable[[dict[str, Any], dict[str, Any]], str | None]
AuthorityShapeValidator = Callable[[dict[str, Any]], str | None]
PreparedT = TypeVar("PreparedT")


def ingest_artifact_root(pages_dir: Path) -> Path:
    """Resolve the runtime root from the caller-owned Wiki path seam."""

    return pages_dir.parent / "runtime" / "ingest-frontier"


def ingest_artifact_paths(pages_dir: Path, source_key: str) -> tuple[Path, Path]:
    root = ingest_artifact_root(pages_dir)
    return root / f"{source_key}.proposal.json", root / f"{source_key}.review.json"


def continuation_marker_path(pages_dir: Path, source_key: str) -> Path:
    return ingest_artifact_root(pages_dir) / f"{source_key}.continuation.json"


def repair_transition_path(pages_dir: Path, source_key: str) -> Path:
    return ingest_artifact_root(pages_dir) / f"{source_key}.repair-transition.json"


def review_stall_path(pages_dir: Path, source_key: str) -> Path:
    return ingest_artifact_root(pages_dir) / f"{source_key}.stalled.json"


def write_ingest_artifact(path: Path, payload: dict[str, Any]) -> None:
    """Persist the exact historical pretty-JSON artifact contract atomically."""

    from chronovisor.core.link_fix import atomic_write

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
    )


def load_ingest_proposal(
    path: Path,
    *,
    source_key: str,
    raw_content: str,
    decode_prepared: Callable[[object], PreparedT | None],
    targets_reserved_system_page: Callable[[PreparedT], bool],
    plan_is_recoverable: Callable[[PreparedT], bool],
) -> tuple[dict[str, Any], PreparedT] | None:
    """Read and validate a proposal envelope using caller-owned page checks."""

    from chronovisor.decision.decision_lane_prompts import validate_ingest_proposal_envelope

    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(artifact, dict):
        return None
    proposal = artifact.get("proposal")
    if not isinstance(proposal, dict):
        return None
    proposal_sha256 = canonical_json_sha256_stringifying_strict(proposal)
    if (
        artifact.get("schema_version") != INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION
        or artifact.get("kind") != "ingest_frontier_proposal_artifact"
        or artifact.get("source_key") != source_key
        or artifact.get("proposal_sha256") != proposal_sha256
        or not validate_ingest_proposal_envelope(proposal)
        or proposal.get("source_key") != source_key
        or proposal.get("raw_content") != raw_content
        or proposal.get("raw_sha256")
        != hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
    ):
        return None
    prepared = decode_prepared(proposal.get("prepared_operations"))
    if (
        prepared is None
        or targets_reserved_system_page(prepared)
        or not plan_is_recoverable(prepared)
    ):
        return None
    return proposal, prepared


def load_ingest_review_artifact(
    path: Path,
    *,
    source_key: str,
    proposal_sha256: str,
    authority_shape_error: AuthorityShapeValidator,
    authority_error: AuthorityValidator,
    require_integrity: bool = False,
) -> dict[str, Any] | None:
    """Load and validate a durable authority-sealed review artifact."""

    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(artifact, dict):
        return None
    artifact_sha256 = artifact.get("artifact_sha256")
    if artifact_sha256 is not None:
        unsigned_artifact = dict(artifact)
        unsigned_artifact.pop("artifact_sha256", None)
        if (
            not isinstance(artifact_sha256, str)
            or artifact_sha256
            != canonical_json_sha256_stringifying_strict(unsigned_artifact)
        ):
            return None
    elif require_integrity:
        return None
    review = artifact.get("review")
    authority = artifact.get("authority")
    if (
        artifact.get("schema_version")
        != INGEST_FRONTIER_REVIEW_ARTIFACT_SCHEMA_VERSION
        or artifact.get("kind") != "ingest_frontier_review_artifact"
        or artifact.get("source_key") != source_key
        or artifact.get("proposal_sha256") != proposal_sha256
        or not isinstance(authority, dict)
        or authority_shape_error(authority) is not None
        or not isinstance(review, dict)
        or authority_error(review, authority) is not None
        or review.get("decision")
        not in {"apply_available", "confirmed_noop", "approved", "rejected"}
    ):
        return None
    return artifact


def load_ingest_review(
    path: Path,
    *,
    source_key: str,
    proposal_sha256: str,
    authority_shape_error: AuthorityShapeValidator,
    authority_error: AuthorityValidator,
) -> dict[str, Any] | None:
    artifact = load_ingest_review_artifact(
        path,
        source_key=source_key,
        proposal_sha256=proposal_sha256,
        authority_shape_error=authority_shape_error,
        authority_error=authority_error,
    )
    if artifact is None:
        return None
    review = artifact.get("review")
    return review if isinstance(review, dict) else None


def sealed_ingest_review_artifact(
    *,
    source_key: str,
    proposal_sha256: str,
    review: dict[str, Any],
    authority: dict[str, Any],
    integrity: bool = False,
) -> dict[str, Any]:
    """Build the common authority-sealed terminal ingest artifact."""

    sealed = decision_authority.seal_semantic_artifact(
        {
            "schema_version": INGEST_FRONTIER_REVIEW_ARTIFACT_SCHEMA_VERSION,
            "kind": "ingest_frontier_review_artifact",
            "source_key": source_key,
            "proposal_sha256": proposal_sha256,
            "review": review,
        },
        authority=authority,
        lane="ingest_reconciliation",
    )
    if integrity:
        sealed["artifact_sha256"] = canonical_json_sha256_stringifying_strict(sealed)
    return sealed


def write_and_readback_ingest_review_artifact(
    path: Path,
    *,
    source_key: str,
    proposal_sha256: str,
    review: dict[str, Any],
    authority: dict[str, Any],
    authority_shape_error: AuthorityShapeValidator,
    authority_error: AuthorityValidator,
    integrity: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    """Atomically persist and verify a terminal verdict before any effect."""

    try:
        sealed = sealed_ingest_review_artifact(
            source_key=source_key,
            proposal_sha256=proposal_sha256,
            review=review,
            authority=authority,
            integrity=integrity,
        )
        write_ingest_artifact(path, sealed)
    except (OSError, ValueError) as exc:
        return None, f"frontier review artifact write failed: {exc}"
    readback = load_ingest_review_artifact(
        path,
        source_key=source_key,
        proposal_sha256=proposal_sha256,
        authority_shape_error=authority_shape_error,
        authority_error=authority_error,
        require_integrity=integrity,
    )
    if readback != sealed:
        return None, "frontier review artifact readback verification failed"
    return readback, None
