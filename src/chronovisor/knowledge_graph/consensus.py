"""Background-only local consensus for relation and merge proposals."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from chronovisor.core.jsonl_write import append_jsonl_durable
from chronovisor.core.runtime_config import load_decision_router_config
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.decision.decision_router import DecisionRouter, DecisionRouterResult
from chronovisor.decision.graph_decisions import (
    RELATION_VERIFICATION_SCHEMA,
    build_relation_verification_prompt,
)
from chronovisor.knowledge_graph.schema import (
    ConsensusReceipt,
    ConsensusVote,
    sha256,
)
from chronovisor.knowledge_graph.store import KnowledgeGraphStore

RECEIPT_LEDGER = (
    CHRONOVISOR_ROOT / "runtime" / "typed-graph" / "consensus-receipts.jsonl"
)
RouterFactory = Callable[[str], DecisionRouter]


def _router_for_producer(
    producer_role: str, decision_lane: str = "relation_verification"
) -> DecisionRouter:
    config = load_decision_router_config()
    models = {
        "primary": config.primary_model,
        "challenger": config.challenger_model,
        "tie_break": config.tie_break_model,
    }
    independent = [model for role, model in models.items() if role != producer_role]
    if len(independent) < 2:
        independent = [config.primary_model, config.challenger_model]
    selected = replace(
        config,
        primary_model=independent[0],
        challenger_model=independent[1],
        tie_break_model=independent[2] if len(independent) >= 3 else independent[1],
        quorum=2,
        adoption_artifact="",
    )
    return DecisionRouter(
        config=selected,
        decision_lane=decision_lane,
        audit_role=f"typed_graph_{decision_lane}",
    )


def _page_exists(root: Path, page_id: str) -> bool:
    return any(
        path.exists()
        for path in (
            root / "pages" / f"{page_id}.md",
            root / "system" / f"{page_id}.md",
        )
    )


def _source_digest_valid(root: Path, page_id: str, digest: str) -> bool:
    for path in (root / "pages" / f"{page_id}.md", root / "system" / f"{page_id}.md"):
        try:
            if hashlib.sha256(path.read_bytes()).hexdigest() == digest:
                return True
        except OSError:
            continue
    return False


def verify_pending_relations(
    *,
    root: Path = CHRONOVISOR_ROOT,
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
    verified = held = vetoed = 0
    for record in pending:
        digest_valid = all(
            _source_digest_valid(root, evidence.page_id, evidence.content_sha256)
            for evidence in record.evidence
        )
        endpoints_known = _page_exists(root, record.source_page_id) and _page_exists(
            root, record.target_page_id
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
        if result is not None:
            configured = load_decision_router_config()
            roles_by_model = {
                configured.primary_model: "primary",
                configured.challenger_model: "challenger",
                configured.tie_break_model: "tie_break",
            }
            for vote in result.votes:
                value = vote.result.value if isinstance(vote.result.value, dict) else {}
                raw_decision = str(value.get("decision") or "abstained")
                decision = (
                    "approve"
                    if raw_decision == "approved"
                    else "reject"
                    if raw_decision == "rejected"
                    else "abstain"
                )
                role = roles_by_model.get(vote.model, vote.role)
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
        receipt = {
            "schema_version": 1,
            "receipt_id": receipt_id,
            "relation_id": record.relation_id,
            "outcome": outcome,
            "producer_role": record.producer_role,
            "vote_manifest_sha256": sha256(vote_rows),
            "quorum": 2,
            "reason_code": reason[:160],
            "external_model_calls": 0,
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
        "external_model_calls": 0,
        "dry_run": dry_run,
    }
