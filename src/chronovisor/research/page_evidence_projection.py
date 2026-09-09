"""Compatibility exports and ResearchStore persistence for page projections."""

from __future__ import annotations

import hashlib
from pathlib import Path

from chronovisor.core.page_evidence import (
    MAX_CONTEXT_FIELD_CHARS,
    MAX_SECTION_PIECE_CHARS,
    PAGE_EVIDENCE_PROJECTION_SCHEMA,
    PAGE_EVIDENCE_TRANSFORM_VERSION,
    PageEvidenceProjection,
    PageEvidenceProjectionError,
    PageEvidenceRecord,
    build_page_evidence_artifact,
    load_page_evidence_artifact,
    page_evidence_projection_key,
    project_page_evidence,
    reconstruct_page_body,
)
from chronovisor.search.research_store import ResearchStore
from chronovisor.search.research_types import EvidenceArtifact

__all__ = [
    "MAX_CONTEXT_FIELD_CHARS",
    "MAX_SECTION_PIECE_CHARS",
    "PAGE_EVIDENCE_PROJECTION_SCHEMA",
    "PAGE_EVIDENCE_TRANSFORM_VERSION",
    "PageEvidenceProjection",
    "PageEvidenceProjectionError",
    "PageEvidenceRecord",
    "build_page_evidence_artifact",
    "checkpoint_page_evidence_projection",
    "load_page_evidence_artifact",
    "page_evidence_projection_key",
    "project_page_evidence",
    "reconstruct_page_body",
    "store_page_evidence_projection",
]


def store_page_evidence_projection(
    store: ResearchStore, source: bytes, page_id: str, *, durable: bool = False
) -> EvidenceArtifact:
    projection = project_page_evidence(source, page_id)
    artifact = store.put_artifact(
        projection.canonical_bytes(),
        source_type="page-evidence-projection",
        source_uri=f"page:{page_id}",
        title=page_id,
        mime_type="application/json",
        trust="page-source",
        durable=durable,
        metadata={
            "schema": PAGE_EVIDENCE_PROJECTION_SCHEMA,
            "projection_id": projection.projection_id,
            "projection_key": page_evidence_projection_key(source, page_id),
            "page_id": page_id,
            "content_sha256": projection.content_sha256,
            "transform_version": projection.transform_version,
        },
    )
    stored = store.read_artifact(artifact.artifact_id)
    if stored != projection.canonical_bytes():
        raise PageEvidenceProjectionError(
            "stored page projection differs from source transform"
        )
    load_page_evidence_artifact(stored, source=source, page_id=page_id)
    return artifact


def checkpoint_page_evidence_projection(
    store: ResearchStore,
    session_id: str,
    artifact: EvidenceArtifact,
    *,
    active: bool = False,
    durable_receipt: bool = False,
) -> Path:
    # Keep resumable page backfill receipts under the caller-owned store root.
    session_id_sha256 = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
    return store.write_summary(
        session_id_sha256,
        {
            "schema_version": 1,
            "session_id_sha256": session_id_sha256,
            "active": bool(active),
            "durable_receipt": bool(durable_receipt),
            "payload": {
                "kind": "page-evidence-projection",
                "artifact_id": artifact.artifact_id,
                "projection_id": artifact.metadata.get("projection_id", ""),
                "projection_key": artifact.metadata.get("projection_key", ""),
                "page_id": artifact.metadata.get("page_id", ""),
                "content_sha256": artifact.metadata.get("content_sha256", ""),
                "transform_version": artifact.metadata.get("transform_version", ""),
            },
        },
    )
