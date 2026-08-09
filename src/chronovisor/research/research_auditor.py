"""Post-run observation-only audit for research quality and waste."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.jsonl_write import append_jsonl_durable
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.research.evidence_bundle import EvidenceBundle
from chronovisor.search.research_store import ResearchStore

AUDIT_LOG = CHRONOVISOR_ROOT / "review" / "research-audit.jsonl"


def audit_research_run(
    summary: Mapping[str, Any],
    bundle: EvidenceBundle,
    *,
    store: ResearchStore | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    store = store or ResearchStore()
    path = path or AUDIT_LOG
    events = store.events(bundle.research_run_id)
    unknown = [
        claim.claim for claim in bundle.claims if claim.status.value == "unknown"
    ]
    unsupported = [
        claim.claim
        for claim in bundle.claims
        if claim.status.value == "supported" and not claim.evidence_ids
    ]
    evidence_actions = {
        "chronovisor_read",
        "verified_claims",
        "raw_search",
        "web_search",
        "web_fetch",
    }
    observations = [row for row in events if row.get("kind") == "observation"]
    wasted = [
        {
            "iteration": row.get("iteration"),
            "action": (row.get("action") or {}).get("type"),
        }
        for row in observations
        if (row.get("action") or {}).get("type") in evidence_actions
        and not row.get("artifact_id")
        and row.get("status") not in {"blocked", "degraded", "error"}
    ]
    deep = [
        (row.get("action") or {}).get("type")
        for row in events
        if row.get("kind") == "action"
        and (row.get("action") or {}).get("type")
        in {"raw_search", "web_search", "web_fetch"}
    ]
    local_source_types = {"chronovisor_read", "verified_claims"}
    artifact_source_types = {
        artifact.artifact_id: artifact.source_type for artifact in bundle.artifacts
    }
    resolved_claims = [
        claim
        for claim in bundle.claims
        if claim.status.value in {"supported", "contradicted"}
    ]
    avoidable_deep = bool(
        deep
        and resolved_claims
        and all(
            any(
                artifact_source_types.get(artifact_id) in local_source_types
                for artifact_id in (*claim.evidence_ids, *claim.contradiction_ids)
            )
            for claim in resolved_claims
        )
    )
    record = {
        "schema_version": 1,
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "research_run_id": bundle.research_run_id,
        "bundle_id": bundle.bundle_id,
        "missing_evidence": unknown,
        "unsupported_claim": unsupported,
        "wasted_action": wasted,
        "avoidable_deep": avoidable_deep,
        "deep_actions": deep,
        "stop_reason": str(summary.get("stop_reason") or ""),
        "status": "attention"
        if unknown or unsupported or wasted or avoidable_deep
        else "ok",
    }
    append_jsonl_durable(path, [record], sort_keys=True)
    store.append_event(
        bundle.research_run_id,
        {"kind": "post_answer_audit", "audit": record, "terminal": True},
    )
    return record
