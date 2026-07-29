"""Source-backed claim synthesis and deterministic citation rendering."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from chronovisor.research.research_store import ResearchStore
from chronovisor.research.research_types import ClaimKind, ClaimStatus, EvidenceArtifact


@dataclass(frozen=True)
class ClaimAssessment:
    claim: str
    kind: ClaimKind
    status: ClaimStatus
    evidence_ids: tuple[str, ...] = ()
    contradiction_ids: tuple[str, ...] = ()
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["status"] = self.status.value
        payload["evidence_ids"] = list(self.evidence_ids)
        payload["contradiction_ids"] = list(self.contradiction_ids)
        return payload


@dataclass(frozen=True)
class EvidenceBundle:
    bundle_id: str
    research_run_id: str
    created_at: str
    claims: tuple[ClaimAssessment, ...]
    artifacts: tuple[EvidenceArtifact, ...]
    trace: tuple[dict[str, Any], ...] = ()
    challenge: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "bundle_id": self.bundle_id,
            "research_run_id": self.research_run_id,
            "created_at": self.created_at,
            "claims": [claim.to_dict() for claim in self.claims],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "trace": list(self.trace),
            "challenge": self.challenge,
        }


def classify_claim(claim: str, *, user_reported: bool = False) -> ClaimKind:
    if user_reported:
        return ClaimKind.USER_REPORTED
    folded = claim.casefold()
    temporal = (
        "current",
        "currently",
        "latest",
        "today",
        "now",
        "最近",
        "現在",
        "最新",
        "今日",
        "価格",
        "version",
        "バージョン",
    )
    return (
        ClaimKind.FRESHNESS_SENSITIVE
        if any(term in folded for term in temporal)
        else ClaimKind.STABLE
    )


def deterministic_citations(
    assessment: ClaimAssessment,
    artifacts: Mapping[str, EvidenceArtifact],
) -> list[str]:
    citations: list[str] = []
    artifact_ids = dict.fromkeys(
        (*assessment.evidence_ids, *assessment.contradiction_ids)
    )
    for artifact_id in artifact_ids:
        artifact = artifacts.get(artifact_id)
        if artifact is None:
            continue
        target = artifact.citation or artifact.source_uri
        label = artifact.title.strip() or artifact.source_type
        if target.startswith("http://") or target.startswith("https://"):
            citations.append(f"[{label}]({target})")
        else:
            citations.append(f"{label} ({target})")
    return citations


def _claim_tokens(claim: str) -> set[str]:
    return {
        token
        for token in re.findall(
            r"[a-z0-9][a-z0-9_.:/-]{2,}|[\u3040-\u30ff\u3400-\u9fff]{2,}",
            claim.casefold(),
        )
        if token not in {"that", "this", "with", "from", "して", "いる", "ある"}
    }


def _negated_near_match(haystack: str, matched_tokens: Iterable[str]) -> bool:
    negations = (" not ", "ない", "ではない", "false", "contradict")
    for token in matched_tokens:
        start = 0
        while (position := haystack.find(token, start)) >= 0:
            window = haystack[
                max(0, position - 48) : min(len(haystack), position + len(token) + 48)
            ]
            if any(term in window for term in negations):
                return True
            start = position + max(1, len(token))
    return False


def simple_assess_claims(
    claims: Iterable[tuple[str, bool]],
    artifacts: Iterable[EvidenceArtifact],
) -> tuple[ClaimAssessment, ...]:
    """Conservative lexical bootstrap; model challenge may only narrow it."""

    artifact_rows = list(artifacts)
    out: list[ClaimAssessment] = []
    for claim, user_reported in claims:
        tokens = _claim_tokens(claim)
        matched: list[str] = []
        contradicted: list[str] = []
        ranked = sorted(
            artifact_rows,
            key=lambda item: (
                0 if item.source_type in {"verified_claims", "chronovisor_read"} else 1,
                0 if item.trust in {"official", "local"} else 1,
                item.artifact_id,
            ),
        )
        negated = any(
            term in claim.casefold() for term in (" not ", "ない", "ではない", "誤り")
        )
        for artifact in ranked:
            haystack = f"{artifact.title} {artifact.preview}".casefold()
            matched_tokens = [token for token in tokens if token in haystack]
            if tokens and len(matched_tokens) >= max(1, min(2, len(tokens))):
                evidence_negated = _negated_near_match(haystack, matched_tokens)
                if evidence_negated != negated:
                    contradicted.append(artifact.artifact_id)
                else:
                    matched.append(artifact.artifact_id)
        status = (
            ClaimStatus.CONTRADICTED
            if contradicted and not matched
            else ClaimStatus.SUPPORTED
            if matched
            else ClaimStatus.UNKNOWN
        )
        out.append(
            ClaimAssessment(
                claim=claim,
                kind=classify_claim(claim, user_reported=user_reported),
                status=status,
                evidence_ids=tuple(matched[:5]),
                contradiction_ids=tuple(contradicted[:5]),
                rationale=(
                    "source-backed lexical evidence match"
                    if matched
                    else "source-backed contradiction"
                    if contradicted
                    else "no source-backed match"
                ),
            )
        )
    return tuple(out)


def build_bundle(
    *,
    run_id: str,
    claims: Iterable[ClaimAssessment],
    artifacts: Iterable[EvidenceArtifact],
    trace: Iterable[Mapping[str, Any]] = (),
    challenge: Mapping[str, Any] | None = None,
    store: ResearchStore | None = None,
) -> EvidenceBundle:
    claim_rows = tuple(claims)
    artifact_rows = tuple(artifacts)
    canonical = json.dumps(
        {
            "run_id": run_id,
            "claims": [row.to_dict() for row in claim_rows],
            "artifact_ids": [row.artifact_id for row in artifact_rows],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    import hashlib

    bundle = EvidenceBundle(
        bundle_id="bundle:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        research_run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        claims=claim_rows,
        artifacts=artifact_rows,
        trace=tuple(dict(row) for row in trace),
        challenge=dict(challenge or {}),
    )
    if store is not None:
        store.put_artifact(
            json.dumps(bundle.to_dict(), ensure_ascii=False, sort_keys=True, indent=2),
            source_type="evidence_bundle",
            source_uri=f"research:{run_id}",
            title=f"Evidence bundle {run_id}",
            mime_type="application/json",
            citation=f"research:{run_id}",
            trust="derived",
            durable=True,
        )
        store.write_bundle(run_id, bundle.to_dict())
    return bundle
