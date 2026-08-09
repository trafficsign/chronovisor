"""End-to-end explicit research service: retrieve, synthesize, challenge, audit."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from chronovisor.core.background_jobs import enqueue_job
from chronovisor.research.evidence_bundle import (
    ClaimAssessment,
    build_bundle,
    deterministic_citations,
    simple_assess_claims,
)
from chronovisor.research.research_auditor import audit_research_run
from chronovisor.research.research_challenge import challenge_bundle
from chronovisor.research.research_orchestrator import Planner, run_research
from chronovisor.search.research_config import ResearchConfig, load_research_config
from chronovisor.search.research_store import ResearchStore
from chronovisor.search.research_types import BudgetUsage, ClaimStatus


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


def _claim_inputs(
    goal: str, claims: Iterable[str | Mapping[str, Any]] | None
) -> list[tuple[str, bool]]:
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
    challenger = (
        challenge.get("challenger")
        if isinstance(challenge.get("challenger"), Mapping)
        else {}
    )
    verdict = str(challenger.get("verdict") or "")
    tie = (
        challenge.get("tie_break")
        if isinstance(challenge.get("tie_break"), Mapping)
        else {}
    )
    if verdict not in {"reject", "inconclusive"} or tie.get("choice") == "planner":
        return claims
    unsupported = {
        str(item).casefold()
        for item in challenger.get("unsupported_claims", [])
        if isinstance(item, str)
    }
    contradictions = bool(challenger.get("contradictions"))
    out: list[ClaimAssessment] = []
    for claim in claims:
        targeted = not unsupported or claim.claim.casefold() in unsupported
        if claim.status == ClaimStatus.SUPPORTED and targeted:
            out.append(
                replace(
                    claim,
                    status=ClaimStatus.CONTRADICTED
                    if verdict == "reject" and contradictions
                    else ClaimStatus.UNKNOWN,
                    rationale=f"local challenge {verdict}; planner support not adopted",
                )
            )
        else:
            out.append(claim)
    return tuple(out)


def _render_deterministic_answer(
    claims: tuple[ClaimAssessment, ...],
    citations: Mapping[str, list[str]],
    *,
    stop_reason: str,
) -> str:
    """Render a conservative answer when the planner never emits ``finish``."""

    lines = [
        f"Planner synthesis stopped with {stop_reason or 'unknown'}; "
        "deterministic evidence assessment follows."
    ]
    for claim in claims:
        line = f"- [{claim.status.value}] {claim.claim}: {claim.rationale}"
        sources = citations.get(claim.claim, [])
        if sources:
            line += " Sources: " + "; ".join(sources)
        lines.append(line)
    return "\n".join(lines)[:8_000]


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
        claim.claim: deterministic_citations(claim, by_id) for claim in bundle.claims
    }
    planner_answer = str(summary.get("answer") or "").strip()
    answer_mode = "planner"
    if not planner_answer:
        planner_answer = _render_deterministic_answer(
            bundle.claims,
            citations,
            stop_reason=str(summary.get("stop_reason") or ""),
        )
        answer_mode = "deterministic_claim_assessment"
    effective_summary = {
        **summary,
        "answer": planner_answer,
        "answer_mode": answer_mode,
    }
    audit = audit_research_run(effective_summary, bundle, store=store)
    store.append_event(
        run_id,
        {
            "kind": "durable_receipt",
            "bundle_id": bundle.bundle_id,
            "terminal": True,
        },
    )
    result = {
        **effective_summary,
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
        module="chronovisor.research.research_worker",
        args=["--run-id", run_id, "--purpose", purpose],
        env={
            "CHRONOVISOR_RESEARCH_RUN_ID": run_id,
            **{
                key: value
                for key, value in os.environ.items()
                if key.startswith("OLLAMA_")
            },
        },
        stdin_text=json.dumps(
            {"goal": goal, "claims": list(claims or ()), "challenge": bool(challenge)},
            ensure_ascii=False,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or durably enqueue source-backed Chronovisor research."
    )
    parser.add_argument("query", help="Research question or goal.")
    parser.add_argument(
        "--claim",
        action="append",
        default=[],
        help="Claim to assess; repeat for multiple claims.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Run now instead of using the default durable background queue.",
    )
    parser.add_argument("--no-challenge", dest="challenge", action="store_false")
    parser.add_argument("--purpose", default="explicit")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(challenge=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-research`` command-line entry point."""
    args = build_parser().parse_args(argv)
    if args.sync:
        payload = run_evidence_research(
            args.query,
            claims=args.claim,
            challenge=args.challenge,
            purpose=args.purpose,
        )
    else:
        payload = enqueue_evidence_research(
            args.query,
            claims=args.claim,
            challenge=args.challenge,
            purpose=args.purpose,
        )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(json.dumps(payload, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
