"""Self-improving recall policy loop.

The loop is intentionally offline: local models and replay evaluation produce
policy proposals, while a durable local-consensus verdict is the final
adoption authority for every active-policy mutation.  ``frontier_*`` names in
this module are retained only for historical artifact compatibility.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from chronovisor.core.canonical_json import (
    canonical_json_sha256_stringifying as _canonical_json_sha256,
)
from chronovisor.core.feedback_ledger import active_feedback_rows
from chronovisor.core.page_mutation import decision_authority_lock
from chronovisor.core.runtime_config import load_toml_file, runtime_repo_root
from chronovisor.core.runtime_status import safe_append_event, safe_append_metric
from chronovisor.decision import decision_authority
from chronovisor.decision.local_structured import (
    LocalStructuredSession,
    ValidationIssue,
)
from chronovisor.decision.recall_improvement_contract import (
    PROPOSER_VISIBLE_BLOCKERS as PROPOSER_VISIBLE_BLOCKERS,
)
from chronovisor.decision.recall_improvement_contract import (
    PolicyProposal as PolicyProposal,
)
from chronovisor.decision.recall_improvement_contract import (
    adoption_gate_summary as _adoption_gate_summary,
)
from chronovisor.decision.recall_improvement_contract import (
    build_frontier_audit_prompt as build_frontier_audit_prompt,
)
from chronovisor.decision.recall_improvement_contract import (
    build_recall_improvement_candidate_record as build_recall_improvement_candidate_record,
)
from chronovisor.decision.recall_improvement_contract import (
    candidate_blocker_summary as _candidate_blocker_summary,
)
from chronovisor.decision.recall_improvement_contract import (
    candidate_blockers,
    filter_blocker_counts,
    gate_candidate,
    latency_p95,
    top_blocker_rows,
)
from chronovisor.decision.recall_improvement_contract import (
    json_default as _json_default,
)
from chronovisor.decision.recall_improvement_contract import (
    proposer_adoption_gate_summary as _proposer_adoption_gate_summary,
)
from chronovisor.decision.recall_improvement_contract import (
    proposer_visible_rejection_blockers as _proposer_visible_rejection_blockers,
)
from chronovisor.decision.recall_policy_contract import (
    ALLOWED_POLICY_FIELDS as ALLOWED_POLICY_FIELDS,
)
from chronovisor.decision.recall_policy_contract import RecallPolicy as RecallPolicy
from chronovisor.decision.recall_policy_contract import (
    apply_policy_overrides as apply_policy_overrides,
)
from chronovisor.decision.recall_policy_contract import (
    normalize_policy_overrides as normalize_policy_overrides,
)
from chronovisor.decision.recall_policy_contract import (
    policy_snapshot as policy_snapshot,
)
from chronovisor.recall.recall_eval import (
    RecallExample,
    build_dataset,
    evaluate_examples,
)
from chronovisor.recall.recall_policy_store import (
    ACTIVE_POLICY_FILE,
    EPISODES_FILE,
    FRONTIER_AUDIT_DIR,
    IMPROVEMENT_DIR,
    LIVE_EPISODES_FILE,
    REGISTRY_FILE,
    RUNS_DIR,
    SCHEDULE_FILE,
    append_jsonl,
    atomic_write_json,
    read_active_policy,
    read_json_file,
    read_jsonl,
)
from chronovisor.recall.recall_runtime import (
    RECALL_FEEDBACK_FILE,
    RECALL_LOG_FILE,
    load_policy,
)

_candidate_blockers = candidate_blockers
_filter_blocker_counts = filter_blocker_counts
_gate_candidate = gate_candidate
_latency_p95 = latency_p95
_top_blocker_rows = top_blocker_rows

DEFAULT_IMPROVEMENT_MODELS = (
    "maxwell1500/ornith-35b:Q5_K_M",
    "gemma4:26b",
)

FRONTIER_AUDIT_MODES = {"off", "auto", "always"}
RUN_DUE_LOCK_FILE = IMPROVEMENT_DIR / "run-due.lock"
DEFAULT_QUARANTINE_RETRY_SECONDS = 6 * 60 * 60
FRONTIER_POLICY_ARTIFACT_SCHEMA_VERSION = 3
RECALL_IMPROVEMENT_DECISION_LANE = "recall_improvement"
FRONTIER_POLICY_DECISIONS = {
    "approved",
    "rejected",
    "quarantined",
    "needs_retry",
}


def _quarantine_retry_seconds() -> int:
    try:
        return max(
            0,
            int(
                os.getenv(
                    "CHRONOVISOR_CONVERGENCE_QUARANTINE_RETRY_SECONDS",
                    str(DEFAULT_QUARANTINE_RETRY_SECONDS),
                )
            ),
        )
    except (TypeError, ValueError):
        return DEFAULT_QUARANTINE_RETRY_SECONDS


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _run_id() -> str:
    return f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _stable_bucket(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) % 100


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default)
        + "\n"
        for row in rows
    )
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _example_key(example: RecallExample) -> str:
    return "|".join(
        (
            example.kind,
            example.ref,
            example.host,
            example.prompt,
            ",".join(example.expected_pages),
        )
    )


def split_examples(
    examples: list[RecallExample],
    *,
    holdout_ratio: float = 0.2,
) -> tuple[list[RecallExample], list[RecallExample]]:
    if not examples:
        return [], []
    cutoff = int(max(1, min(80, round(holdout_ratio * 100))))
    holdout = [
        example
        for example in examples
        if _stable_bucket(_example_key(example)) < cutoff
    ]
    dev = [example for example in examples if example not in holdout]
    return dev or examples, holdout or examples


def write_episode_snapshot(
    examples: list[RecallExample],
    *,
    path: Path = EPISODES_FILE,
) -> Path:
    rows = [
        {
            "prompt": example.prompt,
            "host": example.host,
            "cwd": example.cwd,
            "session_id": example.session_id,
            "expected_pages": list(example.expected_pages),
            "negative_pages": list(example.negative_pages),
            "injected_pages": list(example.injected_pages),
            "kind": example.kind,
            "ref": example.ref,
            "ts": example.ts,
        }
        for example in examples
    ]
    _write_jsonl(path, rows)
    return path


def metric_score(metrics: dict[str, Any]) -> float:
    recall_at_1 = float(metrics.get("recall_at_1") or 0.0)
    recall_at_3 = float(metrics.get("recall_at_3") or 0.0)
    mrr = float(metrics.get("mrr") or 0.0)
    waste = float(metrics.get("waste_injection_rate") or 0.0)
    avg_pages = float(metrics.get("avg_pages") or 0.0)
    latency = (
        metrics.get("latency_ms") if isinstance(metrics.get("latency_ms"), dict) else {}
    )
    p95 = float((latency or {}).get("p95") or 0.0)
    latency_penalty = min(0.10, p95 / 20_000.0)
    page_penalty = max(0.0, avg_pages - 3.0) * 0.015
    score = (
        0.45 * recall_at_3
        + 0.25 * recall_at_1
        + 0.20 * mrr
        + 0.10 * (1.0 - waste)
        - latency_penalty
        - page_penalty
    )
    return round(max(0.0, min(1.0, score)), 6)


def _evaluate(
    examples: list[RecallExample],
    *,
    policy: RecallPolicy,
    deadline: float | None = None,
) -> dict[str, Any]:
    payload = evaluate_examples(
        examples,
        policy=policy,
        replay=True,
        deadline=deadline,
    )
    payload["score"] = metric_score(payload.get("metrics", {}))
    return payload


def _policy_hash(policy: RecallPolicy) -> str:
    data = json.dumps(
        policy_snapshot(policy),
        ensure_ascii=False,
        sort_keys=True,
        default=_json_default,
    )
    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:12]


def _examples_hash(examples: list[RecallExample]) -> str:
    data = json.dumps(
        [
            {
                "prompt": example.prompt,
                "kind": example.kind,
                "expected_pages": list(example.expected_pages),
                "injected_pages": list(example.injected_pages),
                "ref": example.ref,
            }
            for example in examples
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:12]


def _evaluate_cached(
    examples: list[RecallExample],
    *,
    policy: RecallPolicy,
    cache: dict[tuple[str, str], dict[str, Any]],
    deadline: float | None = None,
) -> dict[str, Any]:
    key = (_policy_hash(policy), _examples_hash(examples))
    if key not in cache:
        cache[key] = _evaluate(examples, policy=policy, deadline=deadline)
    return cache[key]


def _clone_policy(policy: RecallPolicy) -> RecallPolicy:
    return RecallPolicy(**dict(policy.__dict__))


def _failure_samples(
    eval_payload: dict[str, Any], *, limit: int = 10
) -> list[dict[str, Any]]:
    rows = (
        eval_payload.get("rows") if isinstance(eval_payload.get("rows"), list) else []
    )
    samples: list[dict[str, Any]] = []
    for row in rows:
        expected = {
            page for page in row.get("expected_pages", []) if isinstance(page, str)
        }
        pages = [page for page in row.get("pages", []) if isinstance(page, str)]
        kind = str(row.get("kind") or "")
        failed = False
        reason = ""
        if expected and not (expected & set(pages[:3])):
            failed = True
            reason = "expected page missing from top3"
        elif kind in {"false-positive", "injection_ignored"} and pages:
            failed = True
            reason = "false-positive injected pages"
        if not failed:
            continue
        samples.append(
            {
                "kind": kind,
                "reason": reason,
                "prompt": str(row.get("prompt") or "")[:220],
                "expected_pages": sorted(expected),
                "pages": pages[:6],
                "decision": row.get("decision"),
            }
        )
        if len(samples) >= limit:
            break
    return samples


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    return float(ordered[max(0, min(idx, len(ordered) - 1))])


def live_episode_summary(
    path: Path = LIVE_EPISODES_FILE, *, limit: int = 200
) -> dict[str, Any]:
    rows = read_jsonl(path, limit=limit)
    decisions: dict[str, int] = {}
    statuses: dict[str, int] = {}
    hosts: dict[str, int] = {}
    latencies: list[float] = []
    page_counts: list[int] = []
    samples: list[dict[str, Any]] = []
    for row in rows:
        decision = str(row.get("decision") or "unknown")
        status = str(row.get("status") or "unknown")
        host = str(row.get("host") or "unknown")
        decisions[decision] = decisions.get(decision, 0) + 1
        statuses[status] = statuses.get(status, 0) + 1
        hosts[host] = hosts.get(host, 0) + 1
        latency = row.get("latency_ms")
        if isinstance(latency, int | float):
            latencies.append(float(latency))
        pages = row.get("pages")
        if isinstance(pages, list):
            page_counts.append(len(pages))
        if len(samples) < 5:
            samples.append(
                {
                    "ts": row.get("ts"),
                    "decision": decision,
                    "pages": pages[:5] if isinstance(pages, list) else [],
                    "prompt_preview": str(row.get("prompt_preview") or "")[:160],
                }
            )
    return {
        "episodes": len(rows),
        "latest_ts": rows[-1].get("ts") if rows else None,
        "decisions": decisions,
        "statuses": statuses,
        "hosts": hosts,
        "avg_pages": round(
            (sum(page_counts) / len(page_counts)) if page_counts else 0.0, 3
        ),
        "latency_ms": {
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
        },
        "samples": samples,
    }


def _allowed_field_summary() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for field, spec in ALLOWED_POLICY_FIELDS.items():
        row = {"type": spec["type"].__name__}
        if "min" in spec:
            row["min"] = spec["min"]
        if "max" in spec:
            row["max"] = spec["max"]
        out[field] = row
    return out


def _proposal_schema() -> dict[str, Any]:
    override_properties: dict[str, dict[str, Any]] = {}
    for field, spec in ALLOWED_POLICY_FIELDS.items():
        wanted = spec["type"]
        field_schema: dict[str, Any] = {
            "type": (
                "boolean"
                if wanted is bool
                else "integer"
                if wanted is int
                else "number"
            )
        }
        if "min" in spec:
            field_schema["minimum"] = spec["min"]
        if "max" in spec:
            field_schema["maximum"] = spec["max"]
        override_properties[field] = field_schema

    # The documented wrapper remains the preferred response, while direct
    # policy fields stay schema-valid for backward compatibility.  Semantic
    # validation below still fails closed when neither form contains an
    # allowed override.
    return {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        "properties": {
            "summary": {"type": "string"},
            "rationale": {"type": "string"},
            "risk": {"type": "string", "enum": ["low", "medium", "high"]},
            "audit_recommended": {"type": "boolean"},
            "overrides": {
                "type": "object",
                "additionalProperties": False,
                "minProperties": 1,
                "properties": override_properties,
            },
            **override_properties,
        },
    }


def _proposal_prompt(
    *,
    model: str,
    baseline_policy: RecallPolicy,
    baseline_eval: dict[str, Any],
    baseline_holdout: dict[str, Any] | None = None,
    failure_samples: list[dict[str, Any]],
    live_summary: dict[str, Any],
    min_improvement: float = 0.05,
    recent_rejection_blockers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "task": "Propose one small recall policy patch for Chronovisor.",
        "model_role": model,
        "rules": [
            "Use only allowed fields.",
            "Prefer one to four field changes.",
            "Do not propose graph-heavy changes; graph is not the primary strategy.",
            "Optimize recall quality without increasing false-positive memory injection.",
            "Your patch is rejected unless it passes every adoption_gate check.",
            "Do not repeat recent rejected patterns unless your rationale explains why the blocker will not recur.",
            "Output JSON only.",
        ],
        "allowed_fields": _allowed_field_summary(),
        "adoption_gate": _proposer_adoption_gate_summary(
            baseline_dev=baseline_eval,
            live_summary=live_summary,
            min_improvement=min_improvement,
        ),
        "baseline_policy": policy_snapshot(baseline_policy),
        "baseline_metrics": baseline_eval.get("metrics", {}),
        "baseline_score": baseline_eval.get("score"),
        "failure_samples": failure_samples,
        "live_traffic": live_summary,
        "recent_rejection_blockers": _proposer_visible_rejection_blockers(
            recent_rejection_blockers or {"counts": {}, "runs": []}
        ),
        "output": {
            "summary": "short title",
            "rationale": "why this should improve replay eval",
            "risk": "low|medium|high",
            "audit_recommended": False,
            "overrides": {"max_pages": 4},
        },
    }


def _proposal_overrides(parsed: Any) -> tuple[dict[str, Any], bool]:
    """Return normalized overrides and whether the model skipped the wrapper."""
    if not isinstance(parsed, dict):
        return {}, False
    overrides = normalize_policy_overrides(parsed.get("overrides"))
    if overrides:
        return overrides, False
    direct = normalize_policy_overrides(parsed)
    if direct:
        return direct, True
    return {}, False


def _proposal_value_issues(value: Any) -> list[ValidationIssue]:
    overrides, _direct = _proposal_overrides(value)
    if overrides:
        return []
    return [
        ValidationIssue(
            pointer="",
            keyword="required",
            expected="at least one allowed recall policy override",
            received={"type": type(value).__name__},
            message="proposal contained no valid overrides",
        )
    ]


def _call_ollama_proposer(
    *,
    model: str,
    baseline_policy: RecallPolicy,
    baseline_eval: dict[str, Any],
    baseline_holdout: dict[str, Any] | None = None,
    failure_samples: list[dict[str, Any]],
    live_summary: dict[str, Any],
    min_improvement: float = 0.05,
    recent_rejection_blockers: dict[str, Any] | None = None,
    timeout_seconds: float = 180.0,
) -> PolicyProposal:
    prompt = _proposal_prompt(
        model=model,
        baseline_policy=baseline_policy,
        baseline_eval=baseline_eval,
        baseline_holdout=baseline_holdout,
        failure_samples=failure_samples,
        live_summary=live_summary,
        min_improvement=min_improvement,
        recent_rejection_blockers=recent_rejection_blockers,
    )
    started = time.perf_counter()
    try:
        result = LocalStructuredSession(
            model=model,
            role="recall_policy_proposer",
            num_ctx=32_768,
            num_predict=1_024,
            keep_alive="20m",
            read_timeout_ms=max(200, int(timeout_seconds * 1_000)),
            max_input_chars=262_144,
            max_output_chars=8_000,
            max_feedback_chars=2_000,
        ).run(
            json.dumps(prompt, ensure_ascii=False),
            _proposal_schema(),
            value_validator=_proposal_value_issues,
        )
        if not result.ok:
            elapsed_ms = int(round((time.perf_counter() - started) * 1000))
            failure_class = result.failure_class or "structured_failure"
            return PolicyProposal(
                source="ollama",
                model=model,
                proposal_id=f"{model}:error",
                summary="proposal failed",
                rationale=(
                    f"{failure_class}: "
                    f"{result.failure_reason or 'proposal generation failed'} "
                    f"after {elapsed_ms}ms"
                )[:400],
                overrides={},
                risk="high",
                audit_recommended=True,
                error=failure_class,
            )
        parsed = result.value
        overrides, direct_overrides = _proposal_overrides(parsed)
        if not overrides:
            raise ValueError("proposal contained no valid overrides")
        rationale = (
            str(parsed.get("rationale") or "")[:1200]
            if isinstance(parsed, dict)
            else ""
        )
        if direct_overrides and not rationale:
            rationale = (
                "Model returned allowed policy fields directly; accepted as overrides."
            )
        return PolicyProposal(
            source="ollama",
            model=model,
            proposal_id=f"{model}:{hashlib.sha1(json.dumps(overrides, sort_keys=True).encode()).hexdigest()[:8]}",
            summary=str(parsed.get("summary") or "policy patch")[:160]
            if isinstance(parsed, dict)
            else "policy patch",
            rationale=rationale,
            risk=str(parsed.get("risk") or "medium")
            if isinstance(parsed, dict)
            else "medium",
            audit_recommended=bool(parsed.get("audit_recommended"))
            if isinstance(parsed, dict)
            else False,
            overrides=overrides,
        )
    except Exception as exc:
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))
        return PolicyProposal(
            source="ollama",
            model=model,
            proposal_id=f"{model}:error",
            summary="proposal failed",
            rationale=f"{exc.__class__.__name__}: {str(exc)[:200]} after {elapsed_ms}ms",
            overrides={},
            risk="high",
            audit_recommended=True,
            error=exc.__class__.__name__,
        )


def configured_models(
    models: str | list[str] | tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    if isinstance(models, str):
        parsed = tuple(item.strip() for item in models.split(",") if item.strip())
        if parsed:
            return parsed
    if isinstance(models, (list, tuple)):
        parsed = tuple(str(item).strip() for item in models if str(item).strip())
        if parsed:
            return parsed
    env = os.environ.get("CHRONOVISOR_RECALL_IMPROVEMENT_MODELS")
    if env:
        parsed = tuple(item.strip() for item in env.split(",") if item.strip())
        if parsed:
            return parsed
    data = load_toml_file()
    section = data.get("recall_improvement")
    if isinstance(section, dict):
        raw = section.get("models")
        if isinstance(raw, list):
            parsed = tuple(str(item).strip() for item in raw if str(item).strip())
            if parsed:
                return parsed
    return DEFAULT_IMPROVEMENT_MODELS


def propose_with_models(
    *,
    models: tuple[str, ...],
    baseline_policy: RecallPolicy,
    baseline_eval: dict[str, Any],
    baseline_holdout: dict[str, Any],
    failure_samples: list[dict[str, Any]],
    live_summary: dict[str, Any],
    min_improvement: float,
    recent_rejection_blockers: dict[str, Any],
) -> list[PolicyProposal]:
    return [
        _call_ollama_proposer(
            model=model,
            baseline_policy=baseline_policy,
            baseline_eval=baseline_eval,
            baseline_holdout=baseline_holdout,
            failure_samples=failure_samples,
            live_summary=live_summary,
            min_improvement=min_improvement,
            recent_rejection_blockers=recent_rejection_blockers,
        )
        for model in models
    ]


def heuristic_proposals(
    *,
    baseline_policy: RecallPolicy,
    baseline_eval: dict[str, Any],
) -> list[PolicyProposal]:
    metrics = baseline_eval.get("metrics", {})
    recall_at_3 = float(metrics.get("recall_at_3") or 0.0)
    waste = float(metrics.get("waste_injection_rate") or 0.0)
    proposals: list[PolicyProposal] = []
    if recall_at_3 < 0.85:
        proposals.append(
            PolicyProposal(
                source="heuristic",
                model="deterministic",
                proposal_id="heuristic:broaden",
                summary="Broaden top-k recall",
                rationale="Positive examples are missing expected pages, so try one more page/query and a slightly higher semantic weight.",
                overrides=normalize_policy_overrides(
                    {
                        "max_pages": min(6, baseline_policy.max_pages + 1),
                        "max_queries": min(6, baseline_policy.max_queries + 1),
                        "fusion_semantic": min(
                            2.0, baseline_policy.fusion_semantic + 0.1
                        ),
                    }
                ),
                risk="medium",
            )
        )
    if waste > 0.0:
        proposals.append(
            PolicyProposal(
                source="heuristic",
                model="deterministic",
                proposal_id="heuristic:tighten",
                summary="Tighten noisy injections",
                rationale="False-positive examples injected memory, so raise the decision thresholds slightly.",
                overrides=normalize_policy_overrides(
                    {
                        "search_threshold": min(
                            0.9, baseline_policy.search_threshold + 0.03
                        ),
                        "read_threshold": min(
                            0.95, baseline_policy.read_threshold + 0.03
                        ),
                    }
                ),
                risk="low",
            )
        )
    if not proposals:
        proposals.append(
            PolicyProposal(
                source="heuristic",
                model="deterministic",
                proposal_id="heuristic:rank",
                summary="Nudge rank fusion",
                rationale="No dominant failure class; try a small BM25 rank bonus adjustment.",
                overrides=normalize_policy_overrides(
                    {
                        "fusion_bm25_rank_bonus": min(
                            0.05, baseline_policy.fusion_bm25_rank_bonus + 0.002
                        ),
                    }
                ),
                risk="low",
            )
        )
    return [proposal for proposal in proposals if proposal.overrides]


def _recent_rejection_blockers(*, runs_dir: Path, limit: int = 5) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    aggregate: dict[str, int] = {}
    for path in sorted(runs_dir.glob("*.json"), reverse=True):
        record = read_json_file(path)
        if record.get("status") != "rejected":
            continue
        summary = _candidate_blocker_summary(record.get("candidates") or [])
        if not summary["counts"]:
            continue
        runs.append(
            {
                "run_id": record.get("run_id") or path.stem,
                "ts": record.get("ts"),
                "reason": record.get("reason"),
                "counts": summary["counts"],
                "top": summary["top"],
            }
        )
        for key, count in summary["counts"].items():
            aggregate[key] = aggregate.get(key, 0) + int(count)
        if len(runs) >= limit:
            break
    top = sorted(aggregate.items(), key=lambda item: (-item[1], item[0]))
    return {
        "counts": dict(top),
        "top": [{"name": name, "count": count} for name, count in top[:5]],
        "runs": runs,
    }


def _proposal_record(
    proposal: PolicyProposal,
    *,
    baseline_policy: RecallPolicy,
    dev_examples: list[RecallExample],
    holdout_examples: list[RecallExample],
    baseline_dev: dict[str, Any],
    baseline_holdout: dict[str, Any],
    min_improvement: float,
    eval_cache: dict[tuple[str, str], dict[str, Any]],
    deadline: float | None = None,
) -> dict[str, Any]:
    candidate_policy = _clone_policy(baseline_policy)
    applied_fields = apply_policy_overrides(candidate_policy, proposal.overrides)
    if not applied_fields:
        return {
            "proposal": asdict(proposal),
            "status": "invalid",
            "reason": "no valid policy fields",
        }
    candidate_dev = _evaluate_cached(
        dev_examples,
        policy=candidate_policy,
        cache=eval_cache,
        deadline=deadline,
    )
    candidate_holdout = _evaluate_cached(
        holdout_examples,
        policy=candidate_policy,
        cache=eval_cache,
        deadline=deadline,
    )
    return build_recall_improvement_candidate_record(
        proposal,
        applied_fields=applied_fields,
        candidate_policy=candidate_policy,
        baseline_dev=baseline_dev,
        baseline_holdout=baseline_holdout,
        candidate_dev=candidate_dev,
        candidate_holdout=candidate_holdout,
        min_improvement=min_improvement,
    )


def _best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [
        candidate
        for candidate in candidates
        if candidate.get("status") == "candidate_pass"
    ]
    if not passing:
        return None
    return max(
        passing,
        key=lambda item: (
            item.get("dev", {}).get("score", 0.0),
            item.get("holdout", {}).get("score", 0.0),
        ),
    )


def _frontier_audit_needed(
    best: dict[str, Any], *, mode: str
) -> tuple[bool, list[str]]:
    if mode == "off":
        return False, []
    if mode == "always":
        return True, ["frontier mode is always"]
    proposal = best.get("proposal") if isinstance(best.get("proposal"), dict) else {}
    checks = best.get("checks") if isinstance(best.get("checks"), dict) else {}
    overrides = (
        proposal.get("overrides") if isinstance(proposal.get("overrides"), dict) else {}
    )
    reasons: list[str] = []
    if proposal.get("audit_recommended"):
        reasons.append("proposal requested audit")
    if proposal.get("risk") == "high":
        reasons.append("proposal risk is high")
    if len(overrides) >= 4:
        reasons.append("proposal changes four or more fields")
    if any(field in overrides for field in ("semantic", "rewrite_enabled")):
        reasons.append("proposal toggles search strategy")
    if (
        "search_threshold" in overrides
        and abs(float(overrides.get("search_threshold", 0.0)) - 0.35) > 0.15
    ):
        reasons.append("proposal makes a large search threshold move")
    if (
        "read_threshold" in overrides
        and abs(float(overrides.get("read_threshold", 0.0)) - 0.65) > 0.15
    ):
        reasons.append("proposal makes a large read threshold move")
    relative_gain = checks.get("relative_gain")
    if isinstance(relative_gain, int | float) and relative_gain < 0.08:
        reasons.append("eval gain is close to adoption threshold")
    return bool(reasons), reasons


def _frontier_policy_evidence(
    record: dict[str, Any],
    best: dict[str, Any],
    reasons: list[str],
) -> dict[str, Any]:
    """Return stable, complete evidence that a local verdict authorizes."""
    proposal = best.get("proposal") if isinstance(best.get("proposal"), dict) else {}
    stable_proposal = {
        key: value for key, value in proposal.items() if key != "proposal_id"
    }
    dataset = record.get("dataset") if isinstance(record.get("dataset"), dict) else {}
    stable_dataset = {
        key: dataset.get(key)
        for key in ("examples", "dev", "holdout")
        if key in dataset
    }
    return {
        "dataset": stable_dataset,
        "baseline": record.get("baseline"),
        "candidate": {
            **best,
            "proposal": stable_proposal,
        },
        "failure_samples": record.get("failure_samples", [])[:5],
        "audit_reasons": list(reasons),
    }


def _frontier_policy_artifact_path(
    *,
    audit_dir: Path,
    candidate_sha256: str,
) -> Path:
    return audit_dir / f"candidate-{candidate_sha256}.json"


def _decorate_durable_frontier_review(
    review: dict[str, Any],
    *,
    candidate_sha256: str,
    artifact_path: Path,
    authority: dict[str, Any],
    reused: bool,
) -> dict[str, Any]:
    return {
        **review,
        "candidate_sha256": candidate_sha256,
        "_artifact_durable": True,
        "_artifact_path": str(artifact_path),
        "_artifact_authority": authority,
        "_artifact_reused": reused,
    }


def load_frontier_policy_audit(
    record: dict[str, Any],
    best: dict[str, Any],
    *,
    reasons: list[str],
    audit_dir: Path = FRONTIER_AUDIT_DIR,
    authority: dict[str, Any],
) -> dict[str, Any] | None:
    evidence = _frontier_policy_evidence(record, best, reasons)
    candidate_sha256 = _canonical_json_sha256(evidence)
    artifact_path = _frontier_policy_artifact_path(
        audit_dir=audit_dir,
        candidate_sha256=candidate_sha256,
    )
    try:
        envelope = read_json_file(artifact_path)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(envelope, dict):
        return None
    if (
        envelope.get("schema_version") != FRONTIER_POLICY_ARTIFACT_SCHEMA_VERSION
        or envelope.get("kind") != "recall_policy_frontier_verdict"
        or envelope.get("candidate_sha256") != candidate_sha256
        or envelope.get("evidence") != evidence
        or _canonical_json_sha256(envelope.get("evidence")) != candidate_sha256
    ):
        return None
    review = envelope.get("review")
    artifact_authority = envelope.get("authority")
    if (
        not isinstance(review, dict)
        or review.get("decision") not in FRONTIER_POLICY_DECISIONS
        or decision_authority.compare_semantic_authority(
            artifact_authority,
            authority,
            lane=RECALL_IMPROVEMENT_DECISION_LANE,
        )
        is not None
        or decision_authority.semantic_verdict_authority_error(
            review,
            artifact_authority,
            lane=RECALL_IMPROVEMENT_DECISION_LANE,
        )
        is not None
    ):
        return None
    return _decorate_durable_frontier_review(
        review,
        candidate_sha256=candidate_sha256,
        artifact_path=artifact_path,
        authority=authority,
        reused=True,
    )


def run_frontier_policy_audit(
    record: dict[str, Any],
    best: dict[str, Any],
    *,
    reasons: list[str],
    repo_root: Path | None = None,
    timeout: int | None = None,
    audit_dir: Path = FRONTIER_AUDIT_DIR,
    reviewer: Any | None = None,
    authority: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from chronovisor.decision import routine_review

    if authority is None:
        authority, authority_error = decision_authority.current_semantic_authority(
            RECALL_IMPROVEMENT_DECISION_LANE,
            injected_reviewer=reviewer is not None,
        )
        if authority_error is not None or authority is None:
            raise OSError(
                authority_error or "recall improvement review authority is missing"
            )
    reused = load_frontier_policy_audit(
        record,
        best,
        reasons=reasons,
        audit_dir=audit_dir,
        authority=authority,
    )
    if reused is not None:
        return reused

    repo = repo_root or runtime_repo_root()
    timeout_seconds = timeout or int(
        os.environ.get("CHRONOVISOR_RECALL_IMPROVE_FRONTIER_TIMEOUT", "1800")
    )
    prompt = build_frontier_audit_prompt(record, best, reasons)
    if reviewer is None:
        payload = routine_review.run_structured_review(
            prompt,
            routine_review.FRONTIER_DECISION_SCHEMA,
            repo_root=repo,
            timeout=timeout_seconds,
            execute_patch=False,
            decision_lane="recall_improvement",
        )
    else:
        payload = reviewer(prompt, best)
    if not isinstance(payload, dict):
        payload = {
            "decision": "needs_retry",
            "summary": "frontier policy reviewer returned invalid payload",
        }
    payload["audit_reasons"] = reasons
    payload["run_id"] = record.get("run_id")
    payload["ts"] = _now_iso()
    current_authority, current_authority_error = (
        decision_authority.current_semantic_authority(
            RECALL_IMPROVEMENT_DECISION_LANE,
            injected_reviewer=reviewer is not None,
        )
    )
    verdict_authority_error = (
        current_authority_error
        or decision_authority.compare_semantic_authority(
            authority,
            current_authority,
            lane=RECALL_IMPROVEMENT_DECISION_LANE,
        )
        or decision_authority.semantic_verdict_authority_error(
            payload,
            authority,
            lane=RECALL_IMPROVEMENT_DECISION_LANE,
        )
    )
    if verdict_authority_error is not None:
        raise OSError(verdict_authority_error)
    evidence = _frontier_policy_evidence(record, best, reasons)
    candidate_sha256 = _canonical_json_sha256(evidence)
    artifact_path = _frontier_policy_artifact_path(
        audit_dir=audit_dir,
        candidate_sha256=candidate_sha256,
    )
    envelope = decision_authority.seal_semantic_artifact(
        {
            "schema_version": FRONTIER_POLICY_ARTIFACT_SCHEMA_VERSION,
            "kind": "recall_policy_frontier_verdict",
            "candidate_sha256": candidate_sha256,
            "evidence": evidence,
            "review": payload,
            "created_at": _now_iso(),
        },
        authority=authority,
        lane=RECALL_IMPROVEMENT_DECISION_LANE,
    )
    audit_dir.mkdir(parents=True, exist_ok=True)
    # This write is deliberately before the active-policy mutation. A crash
    # after this point can reuse the exact verdict without asking the frontier
    # model again, while a changed candidate/eval gets a different digest.
    atomic_write_json(artifact_path, envelope)
    verified = load_frontier_policy_audit(
        record,
        best,
        reasons=reasons,
        audit_dir=audit_dir,
        authority=authority,
    )
    if verified is None:
        raise OSError("frontier policy artifact read-back validation failed")
    run_id = str(record.get("run_id") or "unknown")
    try:
        atomic_write_json(audit_dir / f"run-{run_id}.json", envelope)
    except OSError:
        # The content-addressed artifact above is the authorization boundary;
        # the run alias is audit convenience only.
        pass
    return _decorate_durable_frontier_review(
        payload,
        candidate_sha256=candidate_sha256,
        artifact_path=artifact_path,
        authority=authority,
        reused=False,
    )


def _frontier_blocks_adoption(audit: dict[str, Any] | None) -> bool:
    return not (
        audit
        and audit.get("decision") == "approved"
        and audit.get("_artifact_durable") is True
        and isinstance(audit.get("candidate_sha256"), str)
        and bool(audit.get("candidate_sha256"))
    )


def _write_active_policy_under_authority(
    active_file: Path,
    active_policy: dict[str, Any],
    *,
    review: dict[str, Any],
    authority: dict[str, Any],
    injected_reviewer: bool,
) -> str | None:
    """Write one active policy while its adopted review epoch is stable."""

    with decision_authority_lock():
        current_authority, current_authority_error = (
            decision_authority.current_semantic_authority(
                RECALL_IMPROVEMENT_DECISION_LANE,
                injected_reviewer=injected_reviewer,
            )
        )
        effect_authority_error = (
            current_authority_error
            or decision_authority.compare_semantic_authority(
                authority,
                current_authority,
                lane=RECALL_IMPROVEMENT_DECISION_LANE,
            )
            or decision_authority.semantic_verdict_authority_error(
                review,
                authority,
                lane=RECALL_IMPROVEMENT_DECISION_LANE,
            )
        )
        if effect_authority_error is not None:
            return effect_authority_error
        atomic_write_json(active_file, active_policy)
    return None


def run_improvement(
    *,
    config_file: Path | None = None,
    log_file: Path = RECALL_LOG_FILE,
    feedback_file: Path = RECALL_FEEDBACK_FILE,
    models: str | list[str] | tuple[str, ...] | None = None,
    apply: bool = True,
    include_heuristic: bool = True,
    min_improvement: float = 0.05,
    max_examples: int = 120,
    max_elapsed_seconds: float = 15 * 60,
    frontier_mode: str = "auto",
    frontier_timeout: int | None = None,
    active_file: Path = ACTIVE_POLICY_FILE,
    registry_file: Path = REGISTRY_FILE,
    runs_dir: Path = RUNS_DIR,
    episodes_file: Path = EPISODES_FILE,
    live_episodes_file: Path = LIVE_EPISODES_FILE,
    frontier_budget: Any | None = None,
    frontier_audit_dir: Path = FRONTIER_AUDIT_DIR,
    frontier_reviewer: Any | None = None,
) -> dict[str, Any]:
    run_id = _run_id()
    started = _now_iso()
    deadline = (
        time.monotonic() + max(0.0, float(max_elapsed_seconds))
        if max_elapsed_seconds > 0
        else None
    )
    examples = [
        example
        for example in build_dataset(log_file=log_file, feedback_file=feedback_file)
        if example.kind != "page_ignored"
    ]
    if max_examples > 0:
        examples = examples[-max_examples:]
    write_episode_snapshot(examples, path=episodes_file)
    live_telemetry = live_episode_summary(live_episodes_file)
    model_list = configured_models(models)
    frontier_mode = frontier_mode if frontier_mode in FRONTIER_AUDIT_MODES else "auto"
    if not examples:
        record = {
            "schema_version": 1,
            "run_id": run_id,
            "ts": started,
            "status": "blocked",
            "applied": False,
            "reason": "no recall feedback examples available",
            "dataset": {
                "examples": 0,
                "log_file": str(log_file),
                "feedback_file": str(feedback_file),
            },
            "live_telemetry": live_telemetry,
            "models": list(model_list),
        }
        _persist_run(record, runs_dir=runs_dir, registry_file=registry_file)
        return record

    dev_examples, holdout_examples = split_examples(examples)
    baseline_policy = load_policy(config_file) if config_file else load_policy()
    eval_cache: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        baseline_dev = _evaluate_cached(
            dev_examples,
            policy=baseline_policy,
            cache=eval_cache,
            deadline=deadline,
        )
        baseline_holdout = _evaluate_cached(
            holdout_examples,
            policy=baseline_policy,
            cache=eval_cache,
            deadline=deadline,
        )
    except TimeoutError as exc:
        record = {
            "schema_version": 1,
            "run_id": run_id,
            "ts": started,
            "status": "budget_deferred",
            "applied": False,
            "reason": str(exc),
            "dataset": {
                "examples": len(examples),
                "dev": len(dev_examples),
                "holdout": len(holdout_examples),
            },
            "eval_cache_entries": len(eval_cache),
            "models": list(model_list),
        }
        _persist_run(record, runs_dir=runs_dir, registry_file=registry_file)
        return record
    failures = _failure_samples(baseline_dev)
    recent_blockers = _recent_rejection_blockers(runs_dir=runs_dir)
    proposer_blockers = _proposer_visible_rejection_blockers(recent_blockers)

    if deadline is not None and time.monotonic() >= deadline:
        record = {
            "schema_version": 1,
            "run_id": run_id,
            "ts": started,
            "status": "budget_deferred",
            "applied": False,
            "reason": "recall improvement runtime budget exhausted",
            "dataset": {
                "examples": len(examples),
                "dev": len(dev_examples),
                "holdout": len(holdout_examples),
            },
            "eval_cache_entries": len(eval_cache),
            "models": list(model_list),
        }
        _persist_run(record, runs_dir=runs_dir, registry_file=registry_file)
        return record

    proposals = propose_with_models(
        models=model_list,
        baseline_policy=baseline_policy,
        baseline_eval=baseline_dev,
        baseline_holdout=baseline_holdout,
        failure_samples=failures,
        live_summary=live_telemetry,
        min_improvement=min_improvement,
        recent_rejection_blockers=proposer_blockers,
    )
    if include_heuristic:
        proposals.extend(
            heuristic_proposals(
                baseline_policy=baseline_policy, baseline_eval=baseline_dev
            )
        )

    try:
        candidates = [
            _proposal_record(
                proposal,
                baseline_policy=baseline_policy,
                dev_examples=dev_examples,
                holdout_examples=holdout_examples,
                baseline_dev=baseline_dev,
                baseline_holdout=baseline_holdout,
                min_improvement=min_improvement,
                eval_cache=eval_cache,
                deadline=deadline,
            )
            for proposal in proposals
            if proposal.overrides
        ]
    except TimeoutError as exc:
        record = {
            "schema_version": 1,
            "run_id": run_id,
            "ts": started,
            "status": "budget_deferred",
            "applied": False,
            "reason": str(exc),
            "dataset": {
                "examples": len(examples),
                "dev": len(dev_examples),
                "holdout": len(holdout_examples),
            },
            "eval_cache_entries": len(eval_cache),
            "models": list(model_list),
        }
        _persist_run(record, runs_dir=runs_dir, registry_file=registry_file)
        return record
    blocker_summary = _candidate_blocker_summary(candidates)
    best = _best_candidate(candidates)
    active_policy: dict[str, Any] | None = None
    applied = False
    status = "rejected"
    reason = "no candidate passed adoption gate"
    frontier_audit: dict[str, Any] | None = None
    frontier_audit_reasons: list[str] = []
    review_authority: dict[str, Any] | None = None
    injected_reviewer = False
    authority_error: str | None = None
    if best:
        status = "applied" if apply else "shadow_pass"
        reason = "candidate passed adoption gate"
        active_policy = {
            "schema_version": 1,
            "run_id": run_id,
            "ts": started,
            "status": status,
            "source": "recall-improvement",
            "models": list(model_list),
            "overrides": best["proposal"]["overrides"],
            "summary": best["proposal"].get("summary", ""),
            "checks": best.get("checks", {}),
            "dev": best.get("dev", {}),
            "holdout": best.get("holdout", {}),
        }
        audit_needed, frontier_audit_reasons = _frontier_audit_needed(
            best, mode=frontier_mode
        )
        if apply:
            audit_needed = True
            mandatory_reason = (
                "active policy adoption requires a durable local-consensus verdict"
            )
            if mandatory_reason not in frontier_audit_reasons:
                frontier_audit_reasons.append(mandatory_reason)
        if apply and audit_needed:
            provisional = {
                "schema_version": 1,
                "run_id": run_id,
                "ts": started,
                "status": status,
                "applied": False,
                "reason": reason,
                "dataset": {
                    "examples": len(examples),
                    "dev": len(dev_examples),
                    "holdout": len(holdout_examples),
                    "log_file": str(log_file),
                    "feedback_file": str(feedback_file),
                    "episodes_file": str(episodes_file),
                },
                "baseline": {
                    "dev": {
                        "score": baseline_dev["score"],
                        "metrics": baseline_dev["metrics"],
                    },
                    "holdout": {
                        "score": baseline_holdout["score"],
                        "metrics": baseline_holdout["metrics"],
                    },
                },
                "failure_samples": failures,
            }
            injected_reviewer = (
                frontier_reviewer is not None
                or getattr(run_frontier_policy_audit, "__module__", None) != __name__
            )
            review_authority, authority_error = (
                decision_authority.current_semantic_authority(
                    RECALL_IMPROVEMENT_DECISION_LANE,
                    injected_reviewer=injected_reviewer,
                )
            )
            frontier_audit = (
                load_frontier_policy_audit(
                    provisional,
                    best,
                    reasons=frontier_audit_reasons,
                    audit_dir=frontier_audit_dir,
                    authority=review_authority,
                )
                if authority_error is None and review_authority is not None
                else None
            )
            allowed = True
            budget_reason = authority_error or "ok"
            if authority_error is not None or review_authority is None:
                allowed = False
            if frontier_audit is None and frontier_budget is not None and allowed:
                allowed, budget_reason = frontier_budget.consume("frontier")
            if frontier_audit is None:
                try:
                    frontier_audit = (
                        run_frontier_policy_audit(
                            provisional,
                            best,
                            reasons=frontier_audit_reasons,
                            timeout=frontier_timeout,
                            audit_dir=frontier_audit_dir,
                            reviewer=frontier_reviewer,
                            authority=review_authority,
                        )
                        if allowed
                        else {
                            "decision": "needs_retry",
                            "summary": budget_reason,
                            "rescue_status": "pending_frontier_review",
                            "human_required": False,
                        }
                    )
                except Exception as exc:
                    frontier_audit = {
                        "decision": "needs_retry",
                        "summary": (
                            "local-consensus verdict could not be durably recorded: "
                            f"{exc.__class__.__name__}: {exc}"
                        ),
                        "rescue_status": "pending_frontier_review",
                        "human_required": False,
                    }
            if (
                injected_reviewer
                and isinstance(frontier_audit, dict)
                and frontier_audit.get("decision") in FRONTIER_POLICY_DECISIONS
                and "_artifact_authority" not in frontier_audit
            ):
                # A monkeypatched review function is an explicit unit-test
                # dependency injection boundary. Production artifacts are
                # always sealed by run_frontier_policy_audit itself.
                frontier_audit["_artifact_authority"] = review_authority
            if (
                isinstance(frontier_audit, dict)
                and frontier_audit.get("decision") in FRONTIER_POLICY_DECISIONS
                and (
                    decision_authority.compare_semantic_authority(
                        frontier_audit.get("_artifact_authority"),
                        review_authority,
                        lane=RECALL_IMPROVEMENT_DECISION_LANE,
                    )
                    or decision_authority.semantic_verdict_authority_error(
                        frontier_audit,
                        review_authority,
                        lane=RECALL_IMPROVEMENT_DECISION_LANE,
                    )
                )
            ):
                frontier_audit = {
                    "decision": "needs_retry",
                    "summary": "recall improvement verdict authority is invalid",
                    "rescue_status": "pending_frontier_review",
                    "human_required": False,
                }
            if (
                isinstance(frontier_audit, dict)
                and frontier_audit.get("decision") not in FRONTIER_POLICY_DECISIONS
            ):
                frontier_audit = {
                    "decision": "needs_retry",
                    "summary": "recall improvement verdict decision is invalid",
                    "rescue_status": "pending_frontier_review",
                    "human_required": False,
                }
            if _frontier_blocks_adoption(frontier_audit):
                if authority_error is not None:
                    status = "pending_frontier_review"
                elif not allowed:
                    status = "budget_deferred"
                elif frontier_audit.get("decision") == "quarantined":
                    status = "frontier_quarantined"
                elif frontier_audit.get("decision") == "rejected":
                    status = "frontier_rejected"
                else:
                    status = "pending_frontier_review"
                reason = (
                    budget_reason
                    if not allowed
                    else f"frontier audit did not approve: {frontier_audit.get('summary') or frontier_audit.get('decision')}"
                )
                active_policy = None
            elif active_policy is not None:
                active_policy["frontier_verdict"] = {
                    "decision": "approved",
                    "candidate_sha256": frontier_audit.get("candidate_sha256"),
                    "artifact_path": frontier_audit.get("_artifact_path"),
                    "authority": review_authority,
                }
        if apply and active_policy:
            mutation_allowed = True
            mutation_reason = "ok"
            if frontier_budget is not None:
                mutation_allowed, mutation_reason = frontier_budget.consume("mutation")
            if not mutation_allowed:
                status = "budget_deferred"
                reason = mutation_reason
                active_policy = None
            else:
                assert review_authority is not None
                effect_authority_error = _write_active_policy_under_authority(
                    active_file,
                    active_policy,
                    review=frontier_audit or {},
                    authority=review_authority,
                    injected_reviewer=injected_reviewer,
                )
                if effect_authority_error is not None:
                    status = "pending_frontier_review"
                    reason = effect_authority_error
                    active_policy = None
                else:
                    applied = True

    record = {
        "schema_version": 1,
        "run_id": run_id,
        "ts": started,
        "status": status,
        "applied": applied,
        "reason": reason,
        "dataset": {
            "examples": len(examples),
            "dev": len(dev_examples),
            "holdout": len(holdout_examples),
            "log_file": str(log_file),
            "feedback_file": str(feedback_file),
            "episodes_file": str(episodes_file),
            "live_episodes_file": str(live_episodes_file),
        },
        "models": list(model_list),
        "baseline_policy": policy_snapshot(baseline_policy),
        "baseline": {
            "dev": {"score": baseline_dev["score"], "metrics": baseline_dev["metrics"]},
            "holdout": {
                "score": baseline_holdout["score"],
                "metrics": baseline_holdout["metrics"],
            },
        },
        "failure_samples": failures,
        "live_telemetry": live_telemetry,
        "adoption_gate": _adoption_gate_summary(
            baseline_dev=baseline_dev,
            baseline_holdout=baseline_holdout,
            min_improvement=min_improvement,
        ),
        "recent_rejection_blockers": recent_blockers,
        "proposer_visible_rejection_blockers": proposer_blockers,
        "candidate_blockers": blocker_summary,
        "proposals": [asdict(proposal) for proposal in proposals],
        "candidates": candidates,
        "best": best,
        "active_policy": active_policy,
        "frontier_audit_recommended": bool(best and frontier_audit_reasons),
        "frontier_audit_reasons": frontier_audit_reasons,
        "frontier_audit": frontier_audit,
        "eval_cache_entries": len(eval_cache),
    }
    if (
        status in {"frontier_rejected", "frontier_quarantined"}
        and review_authority is not None
    ):
        # Terminal semantic dispositions are re-resolved under the shared
        # authority lease before they are persisted, just as for a policy
        # mutation. A changed epoch converts the result into a retry hold.
        with decision_authority_lock():
            current_authority, current_authority_error = (
                decision_authority.current_semantic_authority(
                    RECALL_IMPROVEMENT_DECISION_LANE,
                    injected_reviewer=injected_reviewer,
                )
            )
            terminal_authority_error = (
                current_authority_error
                or decision_authority.compare_semantic_authority(
                    review_authority,
                    current_authority,
                    lane=RECALL_IMPROVEMENT_DECISION_LANE,
                )
                or decision_authority.semantic_verdict_authority_error(
                    frontier_audit,
                    review_authority,
                    lane=RECALL_IMPROVEMENT_DECISION_LANE,
                )
            )
            if terminal_authority_error is not None:
                status = "pending_frontier_review"
                reason = terminal_authority_error
                record["status"] = status
                record["reason"] = reason
            _persist_run(record, runs_dir=runs_dir, registry_file=registry_file)
    else:
        _persist_run(record, runs_dir=runs_dir, registry_file=registry_file)
    level = "info" if applied or status == "shadow_pass" else "warn"
    safe_append_event(level, f"recall-improve | {status}", run_id=run_id, reason=reason)
    return record


def _persist_run(
    record: dict[str, Any], *, runs_dir: Path, registry_file: Path
) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(runs_dir / f"{record['run_id']}.json", record)
    append_jsonl(registry_file, _compact_registry_record(record))


def _compact_registry_record(record: dict[str, Any]) -> dict[str, Any]:
    best = record.get("best") if isinstance(record.get("best"), dict) else None
    return {
        "schema_version": record.get("schema_version", 1),
        "run_id": record.get("run_id"),
        "ts": record.get("ts"),
        "status": record.get("status"),
        "applied": record.get("applied", False),
        "reason": record.get("reason"),
        "dataset": record.get("dataset"),
        "models": record.get("models"),
        "baseline": record.get("baseline"),
        "best": best,
        "active_policy": record.get("active_policy"),
        "adoption_gate": record.get("adoption_gate"),
        "candidate_blockers": record.get("candidate_blockers"),
        "recent_rejection_blockers": record.get("recent_rejection_blockers"),
        "proposer_visible_rejection_blockers": record.get(
            "proposer_visible_rejection_blockers"
        ),
        "frontier_audit_recommended": record.get("frontier_audit_recommended", False),
        "frontier_audit_reasons": record.get("frontier_audit_reasons", []),
        "frontier_audit": record.get("frontier_audit"),
        "eval_cache_entries": record.get("eval_cache_entries"),
        "live_telemetry": record.get("live_telemetry"),
    }


def rollback_policy(
    *,
    active_file: Path = ACTIVE_POLICY_FILE,
    registry_file: Path = REGISTRY_FILE,
) -> dict[str, Any]:
    # Serialize every active-policy writer with evaluated authority readers.
    # Rollback is explicit operational authority, not reuse of an old model
    # verdict, so it does not require a semantic authority seal of its own.
    with decision_authority_lock():
        active = read_active_policy(active_file)
        current_run = str(active.get("run_id") or "")
        rows = read_jsonl(registry_file)
        applied_rows = [
            row
            for row in rows
            if row.get("status") == "applied"
            and isinstance(row.get("active_policy"), dict)
            and row.get("run_id") != current_run
        ]
        if applied_rows:
            previous = applied_rows[-1]["active_policy"]
            atomic_write_json(active_file, previous)
            status = "rolled_back"
            target = previous.get("run_id")
        else:
            with contextlib.suppress(FileNotFoundError):
                active_file.unlink()
            status = "cleared"
            target = None
    record = {
        "schema_version": 1,
        "run_id": _run_id(),
        "ts": _now_iso(),
        "status": "rollback",
        "applied": True,
        "reason": status,
        "from_run_id": current_run,
        "to_run_id": target,
    }
    append_jsonl(registry_file, record)
    safe_append_event(
        "warn",
        f"recall-improve | rollback {status}",
        from_run_id=current_run,
        to_run_id=target,
    )
    return record


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _feedback_count(path: Path = RECALL_FEEDBACK_FILE) -> int:
    return sum(
        1 for row in active_feedback_rows(path) if row.get("kind") != "page_ignored"
    )


def _current_recall_authority_sha256() -> tuple[str | None, str | None]:
    authority, error = decision_authority.current_semantic_authority(
        RECALL_IMPROVEMENT_DECISION_LANE
    )
    if error is not None or authority is None:
        return None, error or "recall improvement authority is unavailable"
    shape_error = decision_authority.semantic_authority_shape_error(
        authority,
        lane=RECALL_IMPROVEMENT_DECISION_LANE,
    )
    if shape_error is not None:
        return None, shape_error
    return _canonical_json_sha256(authority), None


def _durable_frontier_hold(
    result: dict[str, Any],
    *,
    feedback_count: int,
) -> dict[str, Any] | None:
    """Return schedule fields for one authority-bound non-terminal verdict."""

    audit = (
        result.get("frontier_audit")
        if isinstance(result.get("frontier_audit"), dict)
        else None
    )
    if (
        audit is None
        or audit.get("_artifact_durable") is not True
        or audit.get("decision") not in {"quarantined", "needs_retry"}
    ):
        return None
    candidate_sha256 = audit.get("candidate_sha256")
    authority = audit.get("_artifact_authority")
    if (
        not _is_sha256(candidate_sha256)
        or decision_authority.semantic_authority_shape_error(
            authority,
            lane=RECALL_IMPROVEMENT_DECISION_LANE,
        )
        is not None
        or decision_authority.semantic_verdict_authority_error(
            audit,
            authority,
            lane=RECALL_IMPROVEMENT_DECISION_LANE,
        )
        is not None
    ):
        return None
    return {
        "frontier_hold_candidate_sha256": candidate_sha256,
        "frontier_hold_authority_sha256": _canonical_json_sha256(authority),
        "frontier_hold_feedback_count": feedback_count,
        "frontier_hold_decision": audit["decision"],
    }


def _try_acquire_run_due_lock(lock_file: Path = RUN_DUE_LOCK_FILE):
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_file.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps({"pid": os.getpid(), "ts": _now_iso()}, ensure_ascii=False) + "\n"
    )
    handle.flush()
    return handle


def run_due(
    *,
    config_file: Path | None = None,
    log_file: Path = RECALL_LOG_FILE,
    feedback_file: Path = RECALL_FEEDBACK_FILE,
    models: str | list[str] | tuple[str, ...] | None = None,
    apply: bool = True,
    include_heuristic: bool = True,
    min_improvement: float = 0.05,
    max_examples: int = 80,
    max_elapsed_seconds: float = 15 * 60,
    min_interval_hours: float = 24.0,
    min_new_feedback: int = 5,
    min_total_feedback: int = 3,
    frontier_mode: str = "auto",
    frontier_timeout: int | None = None,
    dry_run: bool = False,
    schedule_file: Path = SCHEDULE_FILE,
    lock_file: Path = RUN_DUE_LOCK_FILE,
    frontier_budget: Any | None = None,
) -> dict[str, Any]:
    lock_handle = None
    if not dry_run:
        lock_handle = _try_acquire_run_due_lock(lock_file)
        if lock_handle is None:
            return {
                "schema_version": 1,
                "checked_at": _now_iso(),
                "status": "skipped",
                "reason": "recall improvement run already in progress",
                "dry_run": dry_run,
                "locked": True,
                "lock_file": str(lock_file),
            }
    try:
        return _run_due_locked(
            config_file=config_file,
            log_file=log_file,
            feedback_file=feedback_file,
            models=models,
            apply=apply,
            include_heuristic=include_heuristic,
            min_improvement=min_improvement,
            max_examples=max_examples,
            max_elapsed_seconds=max_elapsed_seconds,
            min_interval_hours=min_interval_hours,
            min_new_feedback=min_new_feedback,
            min_total_feedback=min_total_feedback,
            frontier_mode=frontier_mode,
            frontier_timeout=frontier_timeout,
            dry_run=dry_run,
            schedule_file=schedule_file,
            frontier_budget=frontier_budget,
        )
    finally:
        if lock_handle is not None:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                lock_handle.close()


def _run_due_locked(
    *,
    config_file: Path | None = None,
    log_file: Path = RECALL_LOG_FILE,
    feedback_file: Path = RECALL_FEEDBACK_FILE,
    models: str | list[str] | tuple[str, ...] | None = None,
    apply: bool = True,
    include_heuristic: bool = True,
    min_improvement: float = 0.05,
    max_examples: int = 80,
    max_elapsed_seconds: float = 15 * 60,
    min_interval_hours: float = 24.0,
    min_new_feedback: int = 5,
    min_total_feedback: int = 3,
    frontier_mode: str = "auto",
    frontier_timeout: int | None = None,
    dry_run: bool = False,
    schedule_file: Path = SCHEDULE_FILE,
    frontier_budget: Any | None = None,
) -> dict[str, Any]:
    now = datetime.now()
    state = read_json_file(schedule_file)
    feedback_count = _feedback_count(feedback_file)
    last_run = _parse_ts(state.get("last_run_at"))
    last_feedback_count = int(state.get("last_feedback_count") or 0)
    age_hours = ((now - last_run).total_seconds() / 3600.0) if last_run else None
    new_feedback = max(0, feedback_count - last_feedback_count)
    first_run = last_run is None
    enough_total = feedback_count >= min_total_feedback
    interval_due = last_run is None or (
        age_hours is not None and age_hours >= min_interval_hours
    )
    feedback_due = new_feedback >= min_new_feedback
    last_status = str(state.get("last_status") or "")
    quarantine_pending = last_status == "frontier_quarantined"
    retry_pending = last_status in {
        "pending_frontier_review",
        "frontier_quarantined",
    }
    hold_candidate_sha256 = state.get("frontier_hold_candidate_sha256")
    hold_authority_sha256 = state.get("frontier_hold_authority_sha256")
    hold_feedback_count = state.get("frontier_hold_feedback_count")
    hold_decision = state.get("frontier_hold_decision")
    durable_hold_pending = bool(
        retry_pending
        and _is_sha256(hold_candidate_sha256)
        and _is_sha256(hold_authority_sha256)
        and isinstance(hold_feedback_count, int)
        and hold_feedback_count >= 0
        and hold_decision in {"quarantined", "needs_retry"}
    )
    current_authority_sha256: str | None = None
    current_authority_error: str | None = None
    if durable_hold_pending:
        current_authority_sha256, current_authority_error = (
            _current_recall_authority_sha256()
        )
    hold_feedback_release = bool(
        durable_hold_pending
        and isinstance(hold_feedback_count, int)
        and feedback_count > hold_feedback_count
    )
    hold_authority_release = bool(
        durable_hold_pending
        and current_authority_sha256 is not None
        and current_authority_sha256 != hold_authority_sha256
    )
    hold_release = bool(
        durable_hold_pending
        and current_authority_error is None
        and current_authority_sha256 is not None
        and (hold_feedback_release or hold_authority_release)
    )
    legacy_quarantine_retried = bool(state.get("frontier_legacy_hold_retried"))
    legacy_retry_hold_pending = bool(
        retry_pending and not durable_hold_pending and legacy_quarantine_retried
    )
    legacy_feedback_release = bool(
        legacy_retry_hold_pending and feedback_count > last_feedback_count
    )
    quarantine_retry_at = _parse_ts(state.get("frontier_quarantine_retry_at"))
    if (
        quarantine_pending
        and not durable_hold_pending
        and not legacy_quarantine_retried
        and quarantine_retry_at is None
    ):
        quarantine_started = _parse_ts(
            state.get("frontier_quarantined_at")
            or state.get("last_checked_at")
            or state.get("last_run_at")
        )
        if quarantine_started is not None:
            quarantine_retry_at = quarantine_started + timedelta(
                seconds=_quarantine_retry_seconds()
            )
    quarantine_retry_due = bool(
        quarantine_pending
        and not durable_hold_pending
        and not legacy_quarantine_retried
        and (quarantine_retry_at is None or quarantine_retry_at <= now)
    )
    retry_at = (
        quarantine_retry_at
        if quarantine_pending and not durable_hold_pending
        else _parse_ts(state.get("frontier_next_retry_at"))
    )
    if durable_hold_pending:
        retry_due = hold_release
    elif legacy_retry_hold_pending:
        retry_due = legacy_feedback_release
    elif quarantine_pending:
        retry_due = quarantine_retry_due
    else:
        retry_due = retry_at is None or retry_at <= now
    due = enough_total and (
        (durable_hold_pending and hold_release)
        or (retry_pending and not durable_hold_pending and retry_due)
        or (not retry_pending and interval_due and (feedback_due or first_run))
    )
    decision = {
        "schema_version": 1,
        "checked_at": _now_iso(),
        "status": "due" if due else "skipped",
        "dry_run": dry_run,
        "feedback_count": feedback_count,
        "last_feedback_count": last_feedback_count,
        "new_feedback": new_feedback,
        "age_hours": round(age_hours, 3) if age_hours is not None else None,
        "min_interval_hours": min_interval_hours,
        "min_new_feedback": min_new_feedback,
        "min_total_feedback": min_total_feedback,
        "reasons": {
            "enough_total": enough_total,
            "interval_due": interval_due,
            "feedback_due": feedback_due,
            "first_run": first_run,
            "frontier_retry_pending": retry_pending,
            "frontier_retry_due": retry_due,
            "frontier_quarantine_pending": quarantine_pending,
            "frontier_quarantine_retry_due": quarantine_retry_due,
            "frontier_durable_hold_pending": durable_hold_pending,
            "frontier_hold_feedback_release": hold_feedback_release,
            "frontier_hold_authority_release": hold_authority_release,
            "frontier_hold_authority_available": current_authority_error is None,
            "frontier_hold_authority_error": current_authority_error,
            "frontier_legacy_retry_hold_pending": legacy_retry_hold_pending,
            "frontier_legacy_feedback_release": legacy_feedback_release,
        },
    }
    if dry_run:
        preview_decision = dict(decision)
        decision["would_update_schedule"] = {
            **state,
            "last_checked_at": decision["checked_at"],
            "last_status": decision["status"],
            "last_decision": preview_decision,
        }
        return decision

    if not due:
        next_state = {
            **state,
            "last_checked_at": decision["checked_at"],
            # A backoff check is not a terminal decision. Preserve the pending
            # frontier state so the next cycle still uses frontier_next_retry_at
            # instead of silently falling back to the ordinary daily gate.
            "last_status": state.get("last_status")
            if retry_pending
            else decision["status"],
            "last_decision": decision,
        }
        atomic_write_json(schedule_file, next_state)
        return decision

    result = run_improvement(
        config_file=config_file,
        log_file=log_file,
        feedback_file=feedback_file,
        models=models,
        apply=apply,
        include_heuristic=include_heuristic,
        min_improvement=min_improvement,
        max_examples=max_examples,
        max_elapsed_seconds=max_elapsed_seconds,
        frontier_mode=frontier_mode,
        frontier_timeout=frontier_timeout,
        frontier_budget=frontier_budget,
    )
    result_status = str(result.get("status") or "")
    if result_status == "budget_deferred":
        # A cycle budget is ephemeral and must not acknowledge durable
        # progress. Leaving the schedule byte-for-byte unchanged makes this
        # exact candidate immediately eligible in a later funded cycle.
        safe_append_metric(
            "recall_improve",
            status=result_status,
            applied=False,
            examples=(result.get("dataset") or {}).get("examples"),
            eval_cache_entries=result.get("eval_cache_entries"),
        )
        return {
            "status": "budget_deferred",
            "decision": decision,
            "result": result,
            "schedule_state": state,
        }
    audit = (
        result.get("frontier_audit")
        if isinstance(result.get("frontier_audit"), dict)
        else {}
    )
    durable_hold = _durable_frontier_hold(result, feedback_count=feedback_count)
    if durable_hold is not None:
        next_state = {
            **state,
            "last_checked_at": decision["checked_at"],
            "last_run_at": result.get("ts") or decision["checked_at"],
            "last_run_id": result.get("run_id"),
            "last_status": result_status,
            "last_feedback_count": feedback_count,
            "last_decision": decision,
            "frontier_retry_candidate": None,
            "frontier_retry_attempts": 0,
            "frontier_next_retry_at": None,
            "frontier_quarantined_at": None,
            "frontier_quarantine_retry_at": None,
            "frontier_legacy_hold_retried": False,
            **durable_hold,
        }
    elif result_status in {"pending_frontier_review", "frontier_quarantined"}:
        budget_retry = (
            str(audit.get("summary") or "") == "frontier cycle budget exhausted"
        )
        candidate_payload = {
            "overrides": ((result.get("best") or {}).get("proposal") or {}).get(
                "overrides", {}
            ),
            "examples": (result.get("dataset") or {}).get("examples"),
            "feedback_count": feedback_count,
        }
        candidate_hash = hashlib.sha256(
            json.dumps(
                candidate_payload, ensure_ascii=False, sort_keys=True, default=str
            ).encode("utf-8")
        ).hexdigest()
        previous_hash = str(state.get("frontier_retry_candidate") or "")
        attempts = (
            0
            if quarantine_retry_due
            else int(state.get("frontier_retry_attempts") or 0)
            if previous_hash == candidate_hash
            else 0
        )
        if not budget_retry:
            attempts += 1
        if not budget_retry and attempts >= 3:
            result = {
                **result,
                "status": "frontier_quarantined",
                "reason": f"{result.get('reason', '')}; frontier retry limit exhausted",
            }
            next_state = {
                **state,
                "last_checked_at": decision["checked_at"],
                "last_run_at": result.get("ts") or decision["checked_at"],
                "last_run_id": result.get("run_id"),
                "last_status": "frontier_quarantined",
                # The candidate remains unresolved. Do not acknowledge its
                # feedback progress or it can disappear from future retries.
                "last_feedback_count": last_feedback_count,
                "last_decision": decision,
                "frontier_retry_candidate": candidate_hash,
                "frontier_retry_attempts": attempts,
                "frontier_next_retry_at": None,
                "frontier_quarantined_at": now.isoformat(timespec="seconds"),
                "frontier_quarantine_retry_at": (
                    now + timedelta(seconds=_quarantine_retry_seconds())
                ).isoformat(timespec="seconds"),
                "frontier_legacy_hold_retried": bool(
                    legacy_quarantine_retried or quarantine_retry_due
                ),
                "frontier_hold_candidate_sha256": None,
                "frontier_hold_authority_sha256": None,
                "frontier_hold_feedback_count": None,
                "frontier_hold_decision": None,
            }
        else:
            delay_seconds = (
                5 * 60
                if budget_retry
                else min(6 * 60 * 60, 15 * 60 * (2 ** max(0, attempts - 1)))
            )
            next_state = {
                **state,
                "last_checked_at": decision["checked_at"],
                "last_run_id": result.get("run_id"),
                "last_status": "pending_frontier_review",
                "last_decision": decision,
                "frontier_retry_candidate": candidate_hash,
                "frontier_retry_attempts": attempts,
                "frontier_next_retry_at": (
                    now + timedelta(seconds=delay_seconds)
                ).isoformat(timespec="seconds"),
                "frontier_quarantined_at": None,
                "frontier_quarantine_retry_at": None,
                "frontier_legacy_hold_retried": bool(
                    legacy_quarantine_retried or quarantine_retry_due
                ),
                "frontier_hold_candidate_sha256": None,
                "frontier_hold_authority_sha256": None,
                "frontier_hold_feedback_count": None,
                "frontier_hold_decision": None,
            }
    else:
        next_state = {
            **state,
            "last_checked_at": decision["checked_at"],
            "last_run_at": result.get("ts") or decision["checked_at"],
            "last_run_id": result.get("run_id"),
            "last_status": result_status,
            "last_feedback_count": feedback_count,
            "last_decision": decision,
            "frontier_retry_candidate": None,
            "frontier_retry_attempts": 0,
            "frontier_next_retry_at": None,
            "frontier_quarantined_at": None,
            "frontier_quarantine_retry_at": None,
            "frontier_legacy_hold_retried": False,
            "frontier_hold_candidate_sha256": None,
            "frontier_hold_authority_sha256": None,
            "frontier_hold_feedback_count": None,
            "frontier_hold_decision": None,
        }
    atomic_write_json(schedule_file, next_state)
    safe_append_metric(
        "recall_improve",
        status=result.get("status"),
        applied=bool(result.get("applied")),
        examples=(result.get("dataset") or {}).get("examples"),
        eval_cache_entries=result.get("eval_cache_entries"),
    )
    return {
        "status": "ran",
        "decision": decision,
        "result": result,
        "schedule_state": next_state,
    }


def improvement_snapshot(
    *,
    active_file: Path = ACTIVE_POLICY_FILE,
    registry_file: Path = REGISTRY_FILE,
    limit: int = 12,
) -> dict[str, Any]:
    active = read_json_file(active_file)
    history = []
    for row in read_jsonl(registry_file, limit=limit):
        hydrated = dict(row)
        run_id = hydrated.get("run_id")
        if isinstance(run_id, str) and not hydrated.get("candidate_blockers"):
            full = read_json_file(RUNS_DIR / f"{run_id}.json")
            if full:
                hydrated["candidate_blockers"] = full.get(
                    "candidate_blockers"
                ) or _candidate_blocker_summary(full.get("candidates") or [])
                hydrated["adoption_gate"] = full.get("adoption_gate")
                hydrated["recent_rejection_blockers"] = full.get(
                    "recent_rejection_blockers"
                )
                hydrated["proposer_visible_rejection_blockers"] = full.get(
                    "proposer_visible_rejection_blockers"
                )
        history.append(hydrated)
    schedule = read_json_file(SCHEDULE_FILE)
    latest = history[-1] if history else None
    counts: dict[str, int] = {}
    for row in history:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    status = "active" if active else "quiet"
    if latest and not active:
        status = str(latest.get("status") or status)
    return {
        "status": status,
        "active": active or None,
        "latest": latest,
        "history": history,
        "counts": counts,
        "models": list(configured_models(None)),
        "schedule": schedule or None,
        "paths": {
            "active_policy": str(active_file),
            "registry": str(registry_file),
            "episodes": str(EPISODES_FILE),
            "live_episodes": str(LIVE_EPISODES_FILE),
            "runs": str(RUNS_DIR),
            "schedule": str(SCHEDULE_FILE),
            "frontier_audits": str(FRONTIER_AUDIT_DIR),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local self-improving recall policy experiments."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run", help="Propose, replay-evaluate, and optionally adopt a policy patch."
    )
    run.add_argument("--config")
    run.add_argument("--log-file", default=str(RECALL_LOG_FILE))
    run.add_argument("--feedback-file", default=str(RECALL_FEEDBACK_FILE))
    run.add_argument("--models", help="Comma-separated Ollama proposer models.")
    run.add_argument("--no-apply", dest="apply", action="store_false", default=True)
    run.add_argument(
        "--no-heuristic", dest="include_heuristic", action="store_false", default=True
    )
    run.add_argument("--min-improvement", type=float, default=0.05)
    run.add_argument("--max-examples", type=int, default=120)
    run.add_argument("--frontier", choices=sorted(FRONTIER_AUDIT_MODES), default="auto")
    run.add_argument("--frontier-timeout", type=int)
    run.add_argument("--json", action="store_true")

    due = sub.add_parser(
        "run-due",
        help="Run the improvement loop only when schedule/feedback gates are due.",
    )
    due.add_argument("--config")
    due.add_argument("--log-file", default=str(RECALL_LOG_FILE))
    due.add_argument("--feedback-file", default=str(RECALL_FEEDBACK_FILE))
    due.add_argument("--models", help="Comma-separated Ollama proposer models.")
    due.add_argument("--no-apply", dest="apply", action="store_false", default=True)
    due.add_argument(
        "--no-heuristic", dest="include_heuristic", action="store_false", default=True
    )
    due.add_argument("--min-improvement", type=float, default=0.05)
    due.add_argument("--max-examples", type=int, default=80)
    due.add_argument("--min-interval-hours", type=float, default=24.0)
    due.add_argument("--min-new-feedback", type=int, default=5)
    due.add_argument("--min-total-feedback", type=int, default=3)
    due.add_argument("--frontier", choices=sorted(FRONTIER_AUDIT_MODES), default="auto")
    due.add_argument("--frontier-timeout", type=int)
    due.add_argument("--dry-run", action="store_true")
    due.add_argument("--json", action="store_true")

    status = sub.add_parser(
        "status", help="Show active policy and recent improvement runs."
    )
    status.add_argument("--json", action="store_true")

    rollback = sub.add_parser(
        "rollback", help="Rollback to the previous accepted policy."
    )
    rollback.add_argument("--json", action="store_true")
    return parser


def _print_status(payload: dict[str, Any]) -> None:
    print(f"status\t{payload.get('status')}")
    active = payload.get("active") or {}
    print(f"active\t{active.get('run_id') or '--'}")
    latest = payload.get("latest") or {}
    print(f"latest\t{latest.get('run_id') or '--'}\t{latest.get('status') or '--'}")


def _print_run(payload: dict[str, Any]) -> None:
    print(f"run\t{payload['run_id']}")
    print(f"status\t{payload['status']}")
    print(f"applied\t{payload['applied']}")
    print(f"reason\t{payload['reason']}")
    baseline = payload.get("baseline", {}).get("dev", {})
    print(f"baseline_dev_score\t{baseline.get('score', 0):.3f}")
    best = payload.get("best") or {}
    if best:
        print(f"best_dev_score\t{best.get('dev', {}).get('score', 0):.3f}")
        print(f"best_summary\t{best.get('proposal', {}).get('summary', '')}")


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-recall-improve`` command-line entry point."""
    args = build_parser().parse_args(argv)
    if args.command == "run":
        payload = run_improvement(
            config_file=Path(args.config).expanduser() if args.config else None,
            log_file=Path(args.log_file).expanduser(),
            feedback_file=Path(args.feedback_file).expanduser(),
            models=args.models,
            apply=args.apply,
            include_heuristic=args.include_heuristic,
            min_improvement=max(0.0, args.min_improvement),
            max_examples=max(1, args.max_examples),
            frontier_mode=args.frontier,
            frontier_timeout=args.frontier_timeout,
        )
        if args.json:
            print(
                json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
            )
        else:
            _print_run(payload)
        return (
            0
            if payload["status"] in {"applied", "shadow_pass", "rejected", "blocked"}
            else 1
        )
    if args.command == "run-due":
        payload = run_due(
            config_file=Path(args.config).expanduser() if args.config else None,
            log_file=Path(args.log_file).expanduser(),
            feedback_file=Path(args.feedback_file).expanduser(),
            models=args.models,
            apply=args.apply,
            include_heuristic=args.include_heuristic,
            min_improvement=max(0.0, args.min_improvement),
            max_examples=max(1, args.max_examples),
            min_interval_hours=max(0.0, args.min_interval_hours),
            min_new_feedback=max(0, args.min_new_feedback),
            min_total_feedback=max(0, args.min_total_feedback),
            frontier_mode=args.frontier,
            frontier_timeout=args.frontier_timeout,
            dry_run=args.dry_run,
        )
        if args.json:
            print(
                json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
            )
        else:
            print(f"status\t{payload.get('status')}")
            if payload.get("result"):
                print(
                    f"run\t{payload['result'].get('run_id')}\t{payload['result'].get('status')}"
                )
        return 0
    if args.command == "status":
        payload = improvement_snapshot()
        if args.json:
            print(
                json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
            )
        else:
            _print_status(payload)
        return 0
    if args.command == "rollback":
        payload = rollback_policy()
        if args.json:
            print(
                json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
            )
        else:
            print(
                f"rollback\t{payload['reason']}\t{payload.get('from_run_id') or '--'} -> {payload.get('to_run_id') or '--'}"
            )
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
