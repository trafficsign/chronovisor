"""End-to-end explicit research service: retrieve, synthesize, challenge, audit."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import replace
from typing import Any, Iterable, Mapping

from llm_wiki_mcp.background_jobs import enqueue_job
from llm_wiki_mcp.evidence_bundle import (
    ClaimAssessment,
    EvidenceBundle,
    build_bundle,
    deterministic_citations,
    simple_assess_claims,
)
from llm_wiki_mcp.research_auditor import audit_research_run
from llm_wiki_mcp.research_challenge import challenge_bundle
from llm_wiki_mcp.research_config import ResearchConfig, load_research_config
from llm_wiki_mcp.research_orchestrator import Planner, run_research
from llm_wiki_mcp.research_store import ResearchStore
from llm_wiki_mcp.research_types import BudgetUsage, ClaimStatus


def _usage_from_summary(summary: Mapping[str, Any]) -> BudgetUsage:
    raw = summary.get("usage") if isinstance(summary.get("usage"), Mapping) else {}
    fields = {
        "iterations",
        "planner_calls",
        "challenge_calls",
        "tie_break_calls",
        "repair_calls",
        "searches",
        "fetches",
        "observation_bytes",
    }
    return BudgetUsage(**{key: int(raw.get(key) or 0) for key in fields})


def _claim_inputs(goal: str, claims: Iterable[str | Mapping[str, Any]] | None) -> list[tuple[str, bool]]:
    rows: list[tuple[str, bool]] = []
    for item in claims or ():
        if isinstance(item, str):
            text, user_reported = item.strip(), False
        elif isinstance(item, Mapping):
            text = str(item.get("claim") or "").strip()
            user_reported = item.get("user_reported") is True
        else:
            continue
        if text:
            rows.append((text[:2_000], user_reported))
    return rows or [(goal.strip()[:2_000], False)]


def _reconcile(
    claims: tuple[ClaimAssessment, ...],
    challenge: Mapping[str, Any],
) -> tuple[ClaimAssessment, ...]:
    challenger = challenge.get("challenger") if isinstance(challenge.get("challenger"), Mapping) else {}
    verdict = str(challenger.get("verdict") or "")
    tie = challenge.get("tie_break") if isinstance(challenge.get("tie_break"), Mapping) else {}
    if verdict not in {"reject", "inconclusive"} or tie.get("choice") == "planner":
        return claims
    unsupported = {str(item).casefold() for item in challenger.get("unsupported_claims", []) if isinstance(item, str)}
    contradictions = bool(challenger.get("contradictions"))
    out: list[ClaimAssessment] = []
    for claim in claims:
        targeted = not unsupported or claim.claim.casefold() in unsupported
        if claim.status == ClaimStatus.SUPPORTED and targeted:
            out.append(
                replace(
                    claim,
                    status=ClaimStatus.CONTRADICTED if verdict == "reject" and contradictions else ClaimStatus.UNKNOWN,
                    rationale=f"local challenge {verdict}; planner support not adopted",
                )
            )
        else:
            out.append(claim)
    return tuple(out)


def run_evidence_research(
    goal: str,
    *,
    claims: Iterable[str | Mapping[str, Any]] | None = None,
    config: ResearchConfig | None = None,
    planner: Planner | None = None,
    challenge: bool = True,
    purpose: str = "explicit",
    run_id: str | None = None,
    store: ResearchStore | None = None,
    web_provider: Any = None,
    challenge_runner: Any = None,
) -> dict[str, Any]:
    config = config or load_research_config()
    store = store or ResearchStore()
    run_id = run_id or uuid.uuid4().hex
    summary = run_research(
        goal,
        config=config,
        planner=planner,
        purpose=purpose,
        run_id=run_id,
        store=store,
        web_provider=web_provider,
    )
    artifacts = tuple(
        artifact
        for artifact_id in summary.get("artifact_ids", [])
        if (artifact := store.artifact_manifest(str(artifact_id))) is not None
    )
    assessed = simple_assess_claims(_claim_inputs(goal, claims), artifacts)
    provisional = build_bundle(
        run_id=run_id,
        claims=assessed,
        artifacts=artifacts,
        trace=store.events(run_id),
        store=store,
    )
    challenge_result: dict[str, Any] = {"status": "disabled"}
    if challenge and artifacts and config.budgets.max_challenge_calls > 0:
        challenge_result = challenge_bundle(
            provisional,
            config=config,
            usage=_usage_from_summary(summary),
            store=store,
            runner=challenge_runner,
        )
    final_claims = _reconcile(assessed, challenge_result)
    bundle = build_bundle(
        run_id=run_id,
        claims=final_claims,
        artifacts=artifacts,
        trace=store.events(run_id),
        challenge=challenge_result,
        store=store,
    )
    by_id = {artifact.artifact_id: artifact for artifact in artifacts}
    citations = {
        claim.claim: deterministic_citations(claim, by_id)
        for claim in bundle.claims
    }
    audit = audit_research_run(summary, bundle, store=store)
    store.append_event(
        run_id,
        {
            "kind": "durable_receipt",
            "bundle_id": bundle.bundle_id,
            "terminal": True,
        },
    )
    result = {
        **summary,
        "evidence_bundle_id": bundle.bundle_id,
        "claims": [claim.to_dict() for claim in bundle.claims],
        "citations": citations,
        "challenge": challenge_result,
        "audit": audit,
    }
    store.write_summary(run_id, result)
    return result


def enqueue_evidence_research(
    goal: str,
    *,
    claims: Iterable[str | Mapping[str, Any]] | None = None,
    challenge: bool = True,
    purpose: str = "explicit",
) -> dict[str, Any]:
    run_id = uuid.uuid4().hex
    return enqueue_job(
        name="research",
        module="llm_wiki_mcp.research_worker",
        args=["--run-id", run_id, "--purpose", purpose],
        env={"LLM_WIKI_RESEARCH_RUN_ID": run_id, **{key: value for key, value in os.environ.items() if key.startswith("OLLAMA_")}},
        stdin_text=json.dumps(
            {"goal": goal, "claims": list(claims or ()), "challenge": bool(challenge)},
            ensure_ascii=False,
        ),
    )
