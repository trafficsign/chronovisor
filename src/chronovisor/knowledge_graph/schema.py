"""Versioned, privacy-safe contracts for the typed graph data plane."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

SCHEMA_VERSION = 1
RELATION_STATUSES = frozenset(
    {
        "proposed",
        "held",
        "verified",
        "repeatedly_used",
        "authoritative",
        "stale",
        "retracted",
    }
)
RELATION_ACTIONS = frozenset(
    {"propose", "hold", "verify", "use", "promote", "stale", "retract"}
)
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256(value: Any) -> str:
    encoded = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def relation_id(
    *,
    source_page_id: str,
    target_page_id: str,
    predicate: str,
    evidence_sha256: str,
    model_sha256: str,
    rubric_sha256: str,
) -> str:
    identity = {
        "source_page_id": source_page_id,
        "target_page_id": target_page_id,
        "predicate": predicate.strip().casefold(),
        "evidence_sha256": evidence_sha256,
        "model_sha256": model_sha256,
        "rubric_sha256": rubric_sha256,
    }
    return f"rel_{sha256(identity)[:24]}"


def entity_candidate_id(*, mention: str, page_id: str, content_sha256: str) -> str:
    return f"entc_{sha256([mention.casefold(), page_id, content_sha256])[:24]}"


@dataclass(frozen=True)
class EvidenceRef:
    page_id: str
    content_sha256: str
    span_sha256: str
    source_line: int
    raw_sha256: str = ""

    def validate(self) -> None:
        if not SAFE_ID_RE.fullmatch(self.page_id):
            raise ValueError("invalid evidence page_id")
        for name, value in (
            ("content_sha256", self.content_sha256),
            ("span_sha256", self.span_sha256),
        ):
            if HEX64_RE.fullmatch(value) is None:
                raise ValueError(f"invalid {name}")
        if self.raw_sha256 and HEX64_RE.fullmatch(self.raw_sha256) is None:
            raise ValueError("invalid raw_sha256")
        if self.source_line < 1:
            raise ValueError("source_line must be positive")


@dataclass(frozen=True)
class ConsensusVote:
    role: str
    model_sha256: str
    decision: str
    confidence: float
    vote_sha256: str

    def validate(self) -> None:
        if self.role not in {"primary", "challenger", "tie_break"}:
            raise ValueError("invalid consensus role")
        if self.decision not in {"approve", "reject", "abstain"}:
            raise ValueError("invalid vote decision")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("invalid vote confidence")
        if HEX64_RE.fullmatch(self.model_sha256) is None:
            raise ValueError("invalid model_sha256")
        if HEX64_RE.fullmatch(self.vote_sha256) is None:
            raise ValueError("invalid vote_sha256")


@dataclass(frozen=True)
class ConsensusReceipt:
    receipt_id: str
    producer_role: str
    quorum: int
    outcome: str
    votes: tuple[ConsensusVote, ...] = ()
    hold_reason: str = ""

    def validate(self) -> None:
        if not SAFE_ID_RE.fullmatch(self.receipt_id):
            raise ValueError("invalid receipt_id")
        if self.producer_role not in {
            "primary",
            "challenger",
            "tie_break",
            "deterministic",
        }:
            raise ValueError("invalid producer_role")
        if self.outcome not in {"verified", "held", "rejected"}:
            raise ValueError("invalid consensus outcome")
        independent = [vote for vote in self.votes if vote.role != self.producer_role]
        if len({vote.role for vote in self.votes}) != len(self.votes):
            raise ValueError("duplicate consensus roles")
        for vote in self.votes:
            vote.validate()
        approvals = sum(vote.decision == "approve" for vote in independent)
        if self.outcome == "verified" and approvals < self.quorum:
            raise ValueError("producer-independent quorum not satisfied")
        if self.outcome == "held" and not self.hold_reason:
            raise ValueError("held receipt requires a reason")


@dataclass(frozen=True)
class RelationRecord:
    relation_id: str
    source_page_id: str
    target_page_id: str
    predicate: str
    direction: str
    status: str
    evidence: tuple[EvidenceRef, ...]
    model_sha256: str
    rubric_sha256: str
    producer_role: str
    confidence: float
    consensus: ConsensusReceipt | None = None
    valid_from: str = ""
    valid_to: str = ""
    used_count: int = 0
    used_sessions: tuple[str, ...] = ()
    reason_code: str = ""

    def validate(self) -> None:
        if not SAFE_ID_RE.fullmatch(self.relation_id):
            raise ValueError("invalid relation_id")
        if not SAFE_ID_RE.fullmatch(self.source_page_id) or not SAFE_ID_RE.fullmatch(
            self.target_page_id
        ):
            raise ValueError("invalid relation endpoint")
        if self.source_page_id == self.target_page_id:
            raise ValueError("self relation is not allowed")
        if not self.predicate.strip() or len(self.predicate) > 128:
            raise ValueError("invalid predicate")
        if self.direction not in {"forward", "reverse", "bidirectional"}:
            raise ValueError("invalid direction")
        if self.status not in RELATION_STATUSES:
            raise ValueError("invalid relation status")
        if not self.evidence:
            raise ValueError("relation requires evidence")
        for evidence in self.evidence:
            evidence.validate()
        if HEX64_RE.fullmatch(self.model_sha256) is None:
            raise ValueError("invalid model_sha256")
        if HEX64_RE.fullmatch(self.rubric_sha256) is None:
            raise ValueError("invalid rubric_sha256")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("invalid confidence")
        expected = relation_id(
            source_page_id=self.source_page_id,
            target_page_id=self.target_page_id,
            predicate=self.predicate,
            evidence_sha256=sha256([asdict(row) for row in self.evidence]),
            model_sha256=self.model_sha256,
            rubric_sha256=self.rubric_sha256,
        )
        if self.relation_id != expected:
            raise ValueError("relation_id digest mismatch")
        if self.consensus is not None:
            self.consensus.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RelationRecord:
        evidence = tuple(EvidenceRef(**dict(row)) for row in value.get("evidence", []))
        consensus_value = value.get("consensus")
        consensus = None
        if isinstance(consensus_value, Mapping):
            votes = tuple(
                ConsensusVote(**dict(row)) for row in consensus_value.get("votes", [])
            )
            consensus = ConsensusReceipt(
                **{
                    key: item for key, item in consensus_value.items() if key != "votes"
                },
                votes=votes,
            )
        record = cls(
            **{
                key: item
                for key, item in value.items()
                if key
                not in {"evidence", "consensus", "schema_version", "used_sessions"}
            },
            evidence=evidence,
            consensus=consensus,
            used_sessions=tuple(value.get("used_sessions", ())),
        )
        record.validate()
        return record


@dataclass(frozen=True)
class EntityCandidate:
    candidate_id: str
    mention: str
    normalized: str
    page_id: str
    content_sha256: str
    entity_type: str = "unknown"
    alias_evidence_sha256: str = ""
    cluster_id: str = ""
    status: str = "proposed"


@dataclass(frozen=True)
class CommunityRecord:
    community_id: str
    member_page_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    source_digests: tuple[str, ...]
    summary_sha256: str
    generated_at: str
    summary: str = ""
    model_sha256: str = ""


def validate_event(value: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "event_id",
        "previous_hash",
        "event_hash",
        "action",
        "created_at",
        "relation",
        "reason_code",
    }
    if set(value) != required:
        raise ValueError("relation event keys mismatch")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported relation event schema")
    if value.get("action") not in RELATION_ACTIONS:
        raise ValueError("invalid relation event action")
    if (
        value.get("previous_hash")
        and HEX64_RE.fullmatch(str(value["previous_hash"])) is None
    ):
        raise ValueError("invalid previous_hash")
    unsigned = {key: item for key, item in value.items() if key != "event_hash"}
    if value.get("event_hash") != sha256(unsigned):
        raise ValueError("relation event hash mismatch")
    RelationRecord.from_dict(value["relation"])
