"""Candidate-only entity consolidation and reversible merge proposals."""

from __future__ import annotations

import math
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import read_sealed_json
from chronovisor.core.jsonl_write import append_jsonl_durable
from chronovisor.core.knowledge_graph_schema import sha256
from chronovisor.core.knowledge_graph_store import KnowledgeGraphStore
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.decision.graph_decisions import (
    ENTITY_MERGE_VERIFICATION_SCHEMA,
    build_entity_merge_verification_prompt,
)
from chronovisor.knowledge_graph.consensus import _router_for_producer


def normalize_mention(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def generate_merge_candidates(
    entities: Sequence[Mapping[str, Any]],
    *,
    embeddings: Mapping[str, Sequence[float]] | None = None,
    similarity_threshold: float = 0.92,
) -> list[dict[str, Any]]:
    """Generate deterministic DBSCAN-like connected components, never authority."""

    values = [dict(row) for row in entities if str(row.get("candidate_id") or "")]
    parent = list(range(len(values)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            left_norm = normalize_mention(str(values[left].get("mention") or ""))
            right_norm = normalize_mention(str(values[right].get("mention") or ""))
            exact = bool(left_norm and left_norm == right_norm)
            semantic = 0.0
            if embeddings is not None:
                semantic = cosine(
                    embeddings.get(str(values[left]["candidate_id"]), ()),
                    embeddings.get(str(values[right]["candidate_id"]), ()),
                )
            same_type = str(values[left].get("entity_type")) == str(
                values[right].get("entity_type")
            )
            if exact or (same_type and semantic >= similarity_threshold):
                union(left, right)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(values):
        groups[find(index)].append(row)
    candidates = []
    for members in groups.values():
        if len(members) < 2:
            continue
        ids = sorted(str(row["candidate_id"]) for row in members)
        candidates.append(
            {
                "merge_candidate_id": f"merge_{sha256(ids)[:24]}",
                "member_candidate_ids": ids,
                "status": "proposed",
                "authority": False,
                "reason": "exact_or_embedding_cluster",
            }
        )
    return sorted(candidates, key=lambda row: row["merge_candidate_id"])


def apply_merge_decision(
    candidate: Mapping[str, Any],
    *,
    approved: bool,
    receipt_id: str,
) -> dict[str, Any]:
    """Return an event-like decision; callers never mutate AliasStore directly."""

    return {
        **dict(candidate),
        "status": "verified" if approved else "held",
        "authority": False,
        "receipt_id": receipt_id,
        "reversible": True,
    }


def split_merge(merge: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        **dict(merge),
        "status": "retracted",
        "authority": False,
        "split_reason": reason[:160],
        "reversible": True,
    }


def consolidate_entity_candidates(
    *,
    root: Path = CHRONOVISOR_ROOT,
    store: KnowledgeGraphStore | None = None,
    limit: int = 3,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Build Nemotron-assisted merge candidates and verify them independently."""

    graph_store = store or KnowledgeGraphStore(root / "knowledge-graph")
    try:
        snapshot = read_sealed_json(
            graph_store.entity_snapshot_file, recover_backup=True
        )
    except Exception:
        snapshot = {}
    candidate_values = snapshot.get("candidates")
    candidates = (
        [dict(row) for row in candidate_values.values() if isinstance(row, dict)]
        if isinstance(candidate_values, dict)
        else []
    )[:500]
    embeddings: dict[str, Sequence[float]] = {}
    embedding_status = "not_needed"
    embedding_route: dict[str, str | None] | None = None
    if len(candidates) >= 2:
        try:
            from chronovisor.core.embedding import embed_texts

            embedding_result = embed_texts(
                [str(row.get("mention") or "") for row in candidates],
                return_route=True,
            )
            if not isinstance(embedding_result, tuple):
                raise RuntimeError("embedding route identity is unavailable")
            vectors, embedding_route = embedding_result
            embeddings = {
                str(row["candidate_id"]): vector
                for row, vector in zip(candidates, vectors, strict=True)
            }
            embedding_status = str(embedding_route["model"])
        except Exception as exc:
            embedding_status = f"fallback_exact:{type(exc).__name__}"
            embedding_route = None
    proposals = generate_merge_candidates(candidates, embeddings=embeddings)
    previous_merges = snapshot.get("merge_candidates")
    merges = (
        {
            str(key): dict(value)
            for key, value in previous_merges.items()
            if isinstance(value, dict)
        }
        if isinstance(previous_merges, dict)
        else {}
    )
    processed = approved = held = 0
    for proposal in proposals:
        merge_id = str(proposal["merge_candidate_id"])
        if merges.get(merge_id, {}).get("status") in {"verified", "held", "retracted"}:
            continue
        if processed >= max(0, limit):
            break
        members = [
            row
            for row in candidates
            if row.get("candidate_id") in proposal["member_candidate_ids"]
        ]
        evidence = {
            "merge_candidate_id": merge_id,
            "member_candidate_ids": proposal["member_candidate_ids"],
            "mention_sha256s": [
                sha256(str(row.get("mention") or "")) for row in members
            ],
            "entity_types": sorted(
                {str(row.get("entity_type") or "unknown") for row in members}
            ),
            "page_ids": sorted({str(row.get("page_id") or "") for row in members}),
            "alias_evidence_sha256s": [
                str(row.get("alias_evidence_sha256") or "") for row in members
            ],
        }
        result = _router_for_producer("tie_break", "entity_merge_verification").decide(
            build_entity_merge_verification_prompt(evidence),
            ENTITY_MERGE_VERIFICATION_SCHEMA,
            decision_lane="entity_merge_verification",
        )
        value = result.value if isinstance(result.value, dict) else {}
        is_approved = bool(
            result.ok
            and value.get("decision") == "approved"
            and value.get("same_identity") is True
            and value.get("alias_supported") is True
            and value.get("collision_risk") is False
            and value.get("split_required") is False
        )
        receipt_id = (
            "entity_receipt_"
            + sha256([merge_id, result.agreement_sha256, is_approved])[:20]
        )
        decided = apply_merge_decision(
            proposal, approved=is_approved, receipt_id=receipt_id
        )
        decided["reason_code"] = (
            "local_quorum" if is_approved else result.failure_class or "held"
        )
        votes = []
        seen_roles: set[str] = set()
        for vote in result.votes:
            if vote.role in seen_roles:
                continue
            seen_roles.add(vote.role)
            vote_value = (
                vote.result.value if isinstance(vote.result.value, dict) else {}
            )
            raw_decision = str(vote_value.get("decision") or "abstained")
            votes.append(
                {
                    "role": vote.role,
                    "model_sha256": sha256(vote.model),
                    "decision": (
                        "approve"
                        if raw_decision == "approved"
                        else "reject"
                        if raw_decision == "rejected"
                        else "abstain"
                    ),
                    "confidence": float(vote_value.get("confidence") or 0.0),
                    "vote_sha256": sha256(
                        [
                            vote.signature_sha256,
                            vote_value,
                            vote.result.failure_class,
                        ]
                    ),
                }
            )
        decided["consensus"] = {
            "receipt_id": receipt_id,
            "producer_role": "tie_break",
            "quorum": 2,
            "outcome": "verified" if is_approved else "held",
            "hold_reason": "" if is_approved else decided["reason_code"],
            "votes": votes,
        }
        merges[merge_id] = decided
        if not dry_run:
            append_jsonl_durable(
                root / "runtime" / "typed-graph" / "entity-decisions.jsonl",
                [
                    {
                        "schema_version": 1,
                        "entity_merge_id": merge_id,
                        "subject_id": merge_id,
                        "receipt_id": receipt_id,
                        "polarity": "positive" if is_approved else "exposure",
                        "quality": "silver",
                        "outcome": decided["status"],
                        "vote_manifest_sha256": sha256(
                            [vote.signature_sha256 for vote in result.votes]
                        ),
                        "votes": votes,
                    }
                ],
                sort_keys=True,
            )
        processed += 1
        approved += int(is_approved)
        held += int(not is_approved)
    if not dry_run:
        graph_store.write_derived_snapshot(
            "entities",
            {
                **{
                    key: value
                    for key, value in snapshot.items()
                    if key != "snapshot_sha256"
                },
                "schema_version": 1,
                "candidates": candidate_values
                if isinstance(candidate_values, dict)
                else {},
                "merge_candidates": dict(sorted(merges.items())),
                "embedding_backend": embedding_status,
                "embedding_route": embedding_route,
            },
        )
    return {
        "status": "ok",
        "candidates": len(candidates),
        "merge_candidates": len(proposals),
        "processed": processed,
        "verified": approved,
        "held": held,
        "embedding_backend": embedding_status,
        "embedding_route": embedding_route,
        "external_model_calls": 0,
    }
