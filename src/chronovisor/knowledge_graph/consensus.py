"""Background-only local consensus for relation and merge proposals."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from chronovisor.core import index_store, store
from chronovisor.core.jsonl_write import append_jsonl_durable
from chronovisor.core.knowledge_graph_schema import (
    ConsensusReceipt,
    ConsensusVote,
    sha256,
)
from chronovisor.core.knowledge_graph_store import KnowledgeGraphStore
from chronovisor.decision.decision_router import DecisionRouter, DecisionRouterResult
from chronovisor.decision.graph_decisions import (
    RELATION_VERIFICATION_SCHEMA,
    build_relation_verification_prompt,
)

RECEIPT_LEDGER = (
    store.CHRONOVISOR_ROOT
    / "runtime"
    / "typed-graph"
    / "consensus-receipts.jsonl"
)
RouterFactory = Callable[[str], DecisionRouter]
_CANONICAL_PRODUCER_ROLES = frozenset(("primary", "challenger", "tie_break"))


def _router_for_producer(
    producer_role: str, decision_lane: str = "relation_verification"
) -> DecisionRouter:
    return DecisionRouter(
        decision_lane=decision_lane,
        audit_role=f"typed_graph_{decision_lane}",
        excluded_roles=(
            (producer_role,) if producer_role in _CANONICAL_PRODUCER_ROLES else ()
        ),
    )


router_for_producer = _router_for_producer


def canonical_graph_page_paths(root: Path) -> dict[str, Path]:
    """Return the canonical stable graph corpus keyed by unique page stem."""

    paths = index_store.canonical_document_paths(
        root / "pages", system_dir=root / "system", require_stable=True
    )
    by_id: dict[str, Path] = {}
    for path in paths:
        if path.stem in by_id:
            raise ValueError(f"duplicate graph page id: {path.stem}")
        by_id[path.stem] = path
    return by_id


def _source_digest_valid(
    page_paths: Mapping[str, Path], page_id: str, digest: str
) -> bool:
    path = page_paths.get(page_id)
    if path is None:
        return False
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() == digest
    except OSError:
        return False


def verify_pending_relations(
    *,
    root: Path = store.CHRONOVISOR_ROOT,
    store: KnowledgeGraphStore | None = None,
    receipt_file: Path = RECEIPT_LEDGER,
    router_factory: RouterFactory = _router_for_producer,
    limit: int = 3,
    dry_run: bool = False,
) -> dict[str, Any]:
    graph_store = store or KnowledgeGraphStore(root / "knowledge-graph")
    pending = [row for row in graph_store.relations(statuses={"proposed"})][
        : max(0, limit)
    ]
    try:
        page_paths = canonical_graph_page_paths(root)
    except ValueError:
        return {
            "status": "blocked",
            "reason": "duplicate_page_id",
            "verified": 0,
            "held": 0,
            "vetoed": 0,
            "external_model_calls": 0,
        }
    verified = held = vetoed = external_model_calls = 0
    for record in pending:
        digest_valid = all(
            _source_digest_valid(page_paths, evidence.page_id, evidence.content_sha256)
            for evidence in record.evidence
        )
        endpoints_known = (
            record.source_page_id in page_paths
            and record.target_page_id in page_paths
        )
        if not digest_valid or not endpoints_known:
            outcome = "held"
            result: DecisionRouterResult | None = None
            reason = "stale_digest" if not digest_valid else "unknown_endpoint"
            vetoed += 1
        else:
            evidence = {
                "relation_id": record.relation_id,
                "source_page_id": record.source_page_id,
                "target_page_id": record.target_page_id,
                "predicate": record.predicate,
                "direction": record.direction,
                "content_sha256s": [row.content_sha256 for row in record.evidence],
                "span_sha256s": [row.span_sha256 for row in record.evidence],
                "source_lines": [row.source_line for row in record.evidence],
            }
            result = router_factory(record.producer_role).decide(
                build_relation_verification_prompt(evidence),
                RELATION_VERIFICATION_SCHEMA,
                decision_lane="relation_verification",
            )
            value = result.value if isinstance(result.value, dict) else {}
            approved = bool(
                result.ok
                and value.get("decision") == "approved"
                and value.get("evidence_supported") is True
                and value.get("contradiction_found") is False
                and value.get("unknown_endpoint") is False
                and value.get("digest_valid") is True
            )
            outcome = "verified" if approved else "held"
            reason = (
                "local_quorum"
                if approved
                else result.failure_class or str(value.get("decision") or "no_quorum")
            )
        receipt_id = "receipt_" + sha256([record.relation_id, outcome, reason])[:24]
        vote_rows = []
        consensus_votes: list[ConsensusVote] = []
        record_external_model_calls = 0
        if result is not None:
            for vote in result.votes:
                route_provenance = getattr(vote, "route_provenance", None)
                if (
                    isinstance(route_provenance, Mapping)
                    and route_provenance.get("location") == "remote"
                ):
                    record_external_model_calls += 1
                value = vote.result.value if isinstance(vote.result.value, dict) else {}
                raw_decision = str(value.get("decision") or "abstained")
                decision = (
                    "approve"
                    if raw_decision == "approved"
                    else "reject"
                    if raw_decision == "rejected"
                    else "abstain"
                )
                role = vote.role
                if role not in {row.role for row in consensus_votes}:
                    consensus_votes.append(
                        ConsensusVote(
                            role=role,
                            model_sha256=sha256(vote.model),
                            decision=decision,
                            confidence=float(value.get("confidence") or 0.0),
                            vote_sha256=sha256(
                                [vote.signature_sha256, value, vote.result.failure_class]
                            ),
                        )
                    )
                vote_rows.append(
                    {
                        "role": vote.role,
                        "model_sha256": sha256(vote.model),
                        "decision_sha256": sha256(vote.result.value or {}),
                        "valid": vote.result.ok,
                    }
                )
        external_model_calls += record_external_model_calls
        receipt = {
            "schema_version": 1,
            "receipt_id": receipt_id,
            "relation_id": record.relation_id,
            "outcome": outcome,
            "producer_role": record.producer_role,
            "vote_manifest_sha256": sha256(vote_rows),
            "quorum": 2,
            "reason_code": reason[:160],
            "external_model_calls": record_external_model_calls,
        }
        if not dry_run:
            append_jsonl_durable(receipt_file, [receipt], sort_keys=True)
            updated = replace(
                record,
                status=outcome,
                reason_code=reason[:160],
                consensus=ConsensusReceipt(
                    receipt_id=receipt_id,
                    producer_role=record.producer_role,
                    quorum=2,
                    outcome=outcome,
                    votes=tuple(consensus_votes),
                    hold_reason=reason[:160] if outcome == "held" else "",
                ),
            )
            graph_store.append(
                updated,
                action="verify" if outcome == "verified" else "hold",
                reason_code=reason[:160],
            )
        if outcome == "verified":
            verified += 1
        else:
            held += 1
    return {
        "status": "ok",
        "processed": len(pending),
        "verified": verified,
        "held": held,
        "deterministic_veto": vetoed,
        "external_model_calls": external_model_calls,
        "dry_run": dry_run,
    }
