"""Self-improving recall policy loop.

The loop is intentionally offline: local models can propose policy changes,
but replay evaluation is the only adoption judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from llm_wiki_mcp.recall_eval import (
    RecallExample,
    build_dataset,
    evaluate_examples,
)
from llm_wiki_mcp.recall_policy_store import (
    ACTIVE_POLICY_FILE,
    ALLOWED_POLICY_FIELDS,
    EPISODES_FILE,
    FRONTIER_AUDIT_DIR,
    IMPROVEMENT_DIR,
    LIVE_EPISODES_FILE,
    REGISTRY_FILE,
    RUNS_DIR,
    SCHEDULE_FILE,
    append_jsonl,
    apply_policy_overrides,
    atomic_write_json,
    normalize_policy_overrides,
    policy_snapshot,
    read_active_policy,
    read_json_file,
    read_jsonl,
)
from llm_wiki_mcp.recall_runtime import (
    RECALL_FEEDBACK_FILE,
    RECALL_LOG_FILE,
    RecallPolicy,
    load_policy,
)
from llm_wiki_mcp.runtime_config import load_toml_file
from llm_wiki_mcp.runtime_status import safe_append_event, safe_append_metric


DEFAULT_IMPROVEMENT_MODELS = (
    "qwen3.6:35b-a3b-mxfp8",
    "gemma4:26b-mxfp8",
)

FRONTIER_AUDIT_MODES = {"off", "auto", "always"}


@dataclass(frozen=True)
class PolicyProposal:
    source: str
    model: str
    proposal_id: str
    summary: str
    rationale: str
    overrides: dict[str, Any]
    risk: str = "medium"
    audit_recommended: bool = False
    error: str = ""


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _run_id() -> str:
    return f"{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _stable_bucket(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) % 100


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n" for row in rows)
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
    holdout = [example for example in examples if _stable_bucket(_example_key(example)) < cutoff]
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
    latency = metrics.get("latency_ms") if isinstance(metrics.get("latency_ms"), dict) else {}
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
) -> dict[str, Any]:
    payload = evaluate_examples(examples, policy=policy, replay=True)
    payload["score"] = metric_score(payload.get("metrics", {}))
    return payload


def _policy_hash(policy: RecallPolicy) -> str:
    data = json.dumps(policy_snapshot(policy), ensure_ascii=False, sort_keys=True, default=_json_default)
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
) -> dict[str, Any]:
    key = (_policy_hash(policy), _examples_hash(examples))
    if key not in cache:
        cache[key] = _evaluate(examples, policy=policy)
    return cache[key]


def _clone_policy(policy: RecallPolicy) -> RecallPolicy:
    return RecallPolicy(**dict(policy.__dict__))


def _failure_samples(eval_payload: dict[str, Any], *, limit: int = 10) -> list[dict[str, Any]]:
    rows = eval_payload.get("rows") if isinstance(eval_payload.get("rows"), list) else []
    samples: list[dict[str, Any]] = []
    for row in rows:
        expected = {page for page in row.get("expected_pages", []) if isinstance(page, str)}
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


def live_episode_summary(path: Path = LIVE_EPISODES_FILE, *, limit: int = 200) -> dict[str, Any]:
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
        "avg_pages": round((sum(page_counts) / len(page_counts)) if page_counts else 0.0, 3),
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


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _proposal_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary", "rationale", "risk", "audit_recommended", "overrides"],
        "properties": {
            "summary": {"type": "string"},
            "rationale": {"type": "string"},
            "risk": {"type": "string", "enum": ["low", "medium", "high"]},
            "audit_recommended": {"type": "boolean"},
            "overrides": {"type": "object"},
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
        "task": "Propose one small recall policy patch for LLM Wiki.",
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
        "adoption_gate": _adoption_gate_summary(
            baseline_dev=baseline_eval,
            baseline_holdout=baseline_holdout or {},
            min_improvement=min_improvement,
        ),
        "baseline_policy": policy_snapshot(baseline_policy),
        "baseline_metrics": baseline_eval.get("metrics", {}),
        "baseline_holdout_metrics": (baseline_holdout or {}).get("metrics", {}),
        "baseline_score": baseline_eval.get("score"),
        "baseline_holdout_score": (baseline_holdout or {}).get("score"),
        "failure_samples": failure_samples,
        "live_traffic": live_summary,
        "recent_rejection_blockers": recent_rejection_blockers or {"counts": {}, "runs": []},
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
    from llm_wiki_mcp.ollama import OLLAMA_URL

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
    timeout = httpx.Timeout(connect=10.0, read=timeout_seconds, write=10.0, pool=10.0)
    started = time.perf_counter()
    try:
        with httpx.Client(base_url=OLLAMA_URL, timeout=timeout) as client:
            resp = client.post(
                "/api/generate",
                json={
                    "model": model,
                    "prompt": json.dumps(prompt, ensure_ascii=False),
                    "stream": False,
                    "think": False,
                    "keep_alive": "20m",
                    "format": _proposal_schema(),
                    "options": {
                        "temperature": 0.2,
                        "num_ctx": 32768,
                        "num_predict": 1024,
                    },
                },
            )
            resp.raise_for_status()
        raw = resp.json().get("response", "{}")
        parsed = json.loads(_strip_json_fence(str(raw)))
        overrides, direct_overrides = _proposal_overrides(parsed)
        if not overrides:
            raise ValueError("proposal contained no valid overrides")
        rationale = str(parsed.get("rationale") or "")[:1200] if isinstance(parsed, dict) else ""
        if direct_overrides and not rationale:
            rationale = "Model returned allowed policy fields directly; accepted as overrides."
        return PolicyProposal(
            source="ollama",
            model=model,
            proposal_id=f"{model}:{hashlib.sha1(json.dumps(overrides, sort_keys=True).encode()).hexdigest()[:8]}",
            summary=str(parsed.get("summary") or "policy patch")[:160] if isinstance(parsed, dict) else "policy patch",
            rationale=rationale,
            risk=str(parsed.get("risk") or "medium") if isinstance(parsed, dict) else "medium",
            audit_recommended=bool(parsed.get("audit_recommended")) if isinstance(parsed, dict) else False,
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


def configured_models(models: str | list[str] | tuple[str, ...] | None = None) -> tuple[str, ...]:
    if isinstance(models, str):
        parsed = tuple(item.strip() for item in models.split(",") if item.strip())
        if parsed:
            return parsed
    if isinstance(models, (list, tuple)):
        parsed = tuple(str(item).strip() for item in models if str(item).strip())
        if parsed:
            return parsed
    env = os.environ.get("LLM_WIKI_RECALL_IMPROVEMENT_MODELS")
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
                        "fusion_semantic": min(2.0, baseline_policy.fusion_semantic + 0.1),
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
                        "search_threshold": min(0.9, baseline_policy.search_threshold + 0.03),
                        "read_threshold": min(0.95, baseline_policy.read_threshold + 0.03),
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
                        "fusion_bm25_rank_bonus": min(0.05, baseline_policy.fusion_bm25_rank_bonus + 0.002),
                    }
                ),
                risk="low",
            )
        )
    return [proposal for proposal in proposals if proposal.overrides]


def _latency_p95(metrics: dict[str, Any]) -> float:
    latency = metrics.get("latency_ms") if isinstance(metrics.get("latency_ms"), dict) else {}
    return float((latency or {}).get("p95") or 0.0)


def _adoption_gate_summary(
    *,
    baseline_dev: dict[str, Any],
    baseline_holdout: dict[str, Any],
    min_improvement: float,
) -> dict[str, Any]:
    base_dev_score = float(baseline_dev.get("score") or 0.0)
    holdout_metrics = baseline_holdout.get("metrics", {}) if isinstance(baseline_holdout, dict) else {}
    holdout_score = float(baseline_holdout.get("score") or 0.0) if isinstance(baseline_holdout, dict) else 0.0
    holdout_recall_at_3 = float(holdout_metrics.get("recall_at_3") or 0.0)
    holdout_waste = float(holdout_metrics.get("waste_injection_rate") or 0.0)
    holdout_p95 = _latency_p95(holdout_metrics)
    return {
        "candidate_must_pass_all": {
            "dev_improved": (
                f"relative_gain >= {min_improvement:.3f} "
                "or absolute_gain >= 0.030 against baseline dev score"
            ),
            "holdout_score_ok": "candidate holdout score >= baseline holdout score - 0.010",
            "holdout_recall_ok": "candidate holdout recall_at_3 >= baseline holdout recall_at_3 - 0.001",
            "holdout_waste_ok": "candidate holdout waste_injection_rate <= baseline holdout waste + 0.020",
            "latency_ok": "candidate holdout p95 latency <= baseline holdout p95 * 1.5 + 500ms",
        },
        "baseline_for_gate": {
            "dev_score": base_dev_score,
            "holdout_score": holdout_score,
            "holdout_recall_at_3": holdout_recall_at_3,
            "holdout_waste_injection_rate": holdout_waste,
            "holdout_latency_p95_ms": holdout_p95,
            "holdout_latency_p95_ceiling_ms": round(holdout_p95 * 1.5 + 500.0, 3),
        },
        "advice": [
            "A patch that broadens search but hurts holdout recall/score or p95 latency will be rejected.",
            "Prefer targeted changes supported by failure_samples, not broad max_pages/threshold moves by default.",
        ],
    }


def _candidate_blockers(checks: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if checks.get("dev_improved") is False:
        blockers.append("dev_improved")
    for key in ("holdout_score_ok", "holdout_recall_ok", "holdout_waste_ok", "latency_ok"):
        if checks.get(key) is False:
            blockers.append(key)
    return blockers


def _candidate_blocker_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    total_blocked = 0
    for candidate in candidates:
        blockers = candidate.get("blockers")
        if not isinstance(blockers, list):
            checks = candidate.get("checks") if isinstance(candidate.get("checks"), dict) else {}
            blockers = _candidate_blockers(checks)
        if blockers:
            total_blocked += 1
        for blocker in blockers:
            key = str(blocker)
            counts[key] = counts.get(key, 0) + 1
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {
        "total_candidates": len(candidates),
        "blocked_candidates": total_blocked,
        "counts": dict(top),
        "top": [{"name": name, "count": count} for name, count in top[:4]],
    }


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


def _gate_candidate(
    *,
    baseline_dev: dict[str, Any],
    baseline_holdout: dict[str, Any],
    candidate_dev: dict[str, Any],
    candidate_holdout: dict[str, Any],
    min_improvement: float,
) -> tuple[bool, dict[str, Any]]:
    base_dev_score = float(baseline_dev.get("score") or 0.0)
    cand_dev_score = float(candidate_dev.get("score") or 0.0)
    base_holdout_score = float(baseline_holdout.get("score") or 0.0)
    cand_holdout_score = float(candidate_holdout.get("score") or 0.0)
    absolute_gain = cand_dev_score - base_dev_score
    relative_gain = absolute_gain / max(0.10, abs(base_dev_score))

    base_holdout_metrics = baseline_holdout.get("metrics", {})
    cand_holdout_metrics = candidate_holdout.get("metrics", {})
    holdout_score_ok = cand_holdout_score >= base_holdout_score - 0.01
    holdout_recall_ok = float(cand_holdout_metrics.get("recall_at_3") or 0.0) >= (
        float(base_holdout_metrics.get("recall_at_3") or 0.0) - 0.001
    )
    holdout_waste_ok = float(cand_holdout_metrics.get("waste_injection_rate") or 0.0) <= (
        float(base_holdout_metrics.get("waste_injection_rate") or 0.0) + 0.02
    )
    latency_ok = _latency_p95(cand_holdout_metrics) <= (_latency_p95(base_holdout_metrics) * 1.5 + 500.0)
    dev_improved = relative_gain >= min_improvement or absolute_gain >= 0.03
    checks = {
        "dev_score": cand_dev_score,
        "baseline_dev_score": base_dev_score,
        "absolute_gain": round(absolute_gain, 6),
        "relative_gain": round(relative_gain, 6),
        "dev_improved": dev_improved,
        "holdout_score_ok": holdout_score_ok,
        "holdout_recall_ok": holdout_recall_ok,
        "holdout_waste_ok": holdout_waste_ok,
        "latency_ok": latency_ok,
    }
    accepted = all((dev_improved, holdout_score_ok, holdout_recall_ok, holdout_waste_ok, latency_ok))
    return accepted, checks


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
) -> dict[str, Any]:
    candidate_policy = _clone_policy(baseline_policy)
    applied_fields = apply_policy_overrides(candidate_policy, proposal.overrides)
    if not applied_fields:
        return {
            "proposal": asdict(proposal),
            "status": "invalid",
            "reason": "no valid policy fields",
        }
    candidate_dev = _evaluate_cached(dev_examples, policy=candidate_policy, cache=eval_cache)
    candidate_holdout = _evaluate_cached(holdout_examples, policy=candidate_policy, cache=eval_cache)
    accepted, checks = _gate_candidate(
        baseline_dev=baseline_dev,
        baseline_holdout=baseline_holdout,
        candidate_dev=candidate_dev,
        candidate_holdout=candidate_holdout,
        min_improvement=min_improvement,
    )
    blockers = _candidate_blockers(checks)
    return {
        "proposal": asdict(proposal),
        "status": "candidate_pass" if accepted else "candidate_blocked",
        "applied_fields": applied_fields,
        "candidate_policy": policy_snapshot(candidate_policy),
        "dev": {
            "score": candidate_dev["score"],
            "metrics": candidate_dev["metrics"],
        },
        "holdout": {
            "score": candidate_holdout["score"],
            "metrics": candidate_holdout["metrics"],
        },
        "checks": checks,
        "blockers": blockers,
    }


def _best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [candidate for candidate in candidates if candidate.get("status") == "candidate_pass"]
    if not passing:
        return None
    return max(
        passing,
        key=lambda item: (
            item.get("dev", {}).get("score", 0.0),
            item.get("holdout", {}).get("score", 0.0),
        ),
    )


def _frontier_audit_needed(best: dict[str, Any], *, mode: str) -> tuple[bool, list[str]]:
    if mode == "off":
        return False, []
    if mode == "always":
        return True, ["frontier mode is always"]
    proposal = best.get("proposal") if isinstance(best.get("proposal"), dict) else {}
    checks = best.get("checks") if isinstance(best.get("checks"), dict) else {}
    overrides = proposal.get("overrides") if isinstance(proposal.get("overrides"), dict) else {}
    reasons: list[str] = []
    if proposal.get("audit_recommended"):
        reasons.append("proposal requested audit")
    if proposal.get("risk") == "high":
        reasons.append("proposal risk is high")
    if len(overrides) >= 4:
        reasons.append("proposal changes four or more fields")
    if any(field in overrides for field in ("semantic", "rewrite_enabled")):
        reasons.append("proposal toggles search strategy")
    if "search_threshold" in overrides and abs(float(overrides.get("search_threshold", 0.0)) - 0.35) > 0.15:
        reasons.append("proposal makes a large search threshold move")
    if "read_threshold" in overrides and abs(float(overrides.get("read_threshold", 0.0)) - 0.65) > 0.15:
        reasons.append("proposal makes a large read threshold move")
    relative_gain = checks.get("relative_gain")
    if isinstance(relative_gain, int | float) and relative_gain < 0.08:
        reasons.append("eval gain is close to adoption threshold")
    return bool(reasons), reasons


def build_frontier_audit_prompt(record: dict[str, Any], best: dict[str, Any], reasons: list[str]) -> str:
    excerpt = {
        "run_id": record.get("run_id"),
        "status": record.get("status"),
        "dataset": record.get("dataset"),
        "baseline": record.get("baseline"),
        "best": best,
        "audit_reasons": reasons,
        "failure_samples": record.get("failure_samples", [])[:5],
    }
    return f"""\
You are the frontier auditor for LLM Wiki recall self-improvement.

Review whether the proposed recall policy patch is safe to adopt.
Do not edit files. Do not run commands. Return JSON only with this exact shape:
{{
  "decision": "approved|rejected|quarantined|needs_retry",
  "summary": "...",
  "tests_run": ["reviewed replay eval payload"],
  "commit": null,
  "committed": false,
  "pushed": false,
  "risk": "low|medium|high",
  "notes": "..."
}}

Approval criteria:
- The replay eval improved dev score and did not degrade holdout recall, waste, or latency.
- The policy patch is small and rollback-safe.
- It does not increase stale/noisy recall risk.
- If evidence is insufficient, return needs_retry.

Payload:
{json.dumps(excerpt, ensure_ascii=False, indent=2, default=_json_default)}
"""


def run_frontier_policy_audit(
    record: dict[str, Any],
    best: dict[str, Any],
    *,
    reasons: list[str],
    repo_root: Path | None = None,
    timeout: int | None = None,
    audit_dir: Path = FRONTIER_AUDIT_DIR,
) -> dict[str, Any]:
    from llm_wiki_mcp import frontier_review

    repo = repo_root or Path(__file__).resolve().parents[2]
    timeout_seconds = timeout or int(os.environ.get("LLM_WIKI_RECALL_IMPROVE_FRONTIER_TIMEOUT", "1800"))
    prompt = build_frontier_audit_prompt(record, best, reasons)
    result = frontier_review._run_codex(
        prompt,
        repo_root=repo,
        timeout=timeout_seconds,
        execute_patch=False,
    )
    payload = result.to_dict()
    payload["audit_reasons"] = reasons
    payload["run_id"] = record.get("run_id")
    payload["ts"] = _now_iso()
    audit_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(audit_dir / f"{record.get('run_id', 'unknown')}.json", payload)
    return payload


def _frontier_blocks_adoption(audit: dict[str, Any] | None) -> bool:
    if not audit:
        return False
    return audit.get("decision") != "approved"


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
    frontier_mode: str = "auto",
    frontier_timeout: int | None = None,
    active_file: Path = ACTIVE_POLICY_FILE,
    registry_file: Path = REGISTRY_FILE,
    runs_dir: Path = RUNS_DIR,
    episodes_file: Path = EPISODES_FILE,
    live_episodes_file: Path = LIVE_EPISODES_FILE,
) -> dict[str, Any]:
    run_id = _run_id()
    started = _now_iso()
    examples = build_dataset(log_file=log_file, feedback_file=feedback_file)
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
            "dataset": {"examples": 0, "log_file": str(log_file), "feedback_file": str(feedback_file)},
            "live_telemetry": live_telemetry,
            "models": list(model_list),
        }
        _persist_run(record, runs_dir=runs_dir, registry_file=registry_file)
        return record

    dev_examples, holdout_examples = split_examples(examples)
    baseline_policy = load_policy(config_file) if config_file else load_policy()
    eval_cache: dict[tuple[str, str], dict[str, Any]] = {}
    baseline_dev = _evaluate_cached(dev_examples, policy=baseline_policy, cache=eval_cache)
    baseline_holdout = _evaluate_cached(holdout_examples, policy=baseline_policy, cache=eval_cache)
    failures = _failure_samples(baseline_dev)
    recent_blockers = _recent_rejection_blockers(runs_dir=runs_dir)

    proposals = propose_with_models(
        models=model_list,
        baseline_policy=baseline_policy,
        baseline_eval=baseline_dev,
        baseline_holdout=baseline_holdout,
        failure_samples=failures,
        live_summary=live_telemetry,
        min_improvement=min_improvement,
        recent_rejection_blockers=recent_blockers,
    )
    if include_heuristic:
        proposals.extend(heuristic_proposals(baseline_policy=baseline_policy, baseline_eval=baseline_dev))

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
        )
        for proposal in proposals
        if proposal.overrides
    ]
    blocker_summary = _candidate_blocker_summary(candidates)
    best = _best_candidate(candidates)
    active_policy: dict[str, Any] | None = None
    applied = False
    status = "rejected"
    reason = "no candidate passed adoption gate"
    frontier_audit: dict[str, Any] | None = None
    frontier_audit_reasons: list[str] = []
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
        audit_needed, frontier_audit_reasons = _frontier_audit_needed(best, mode=frontier_mode)
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
                    "dev": {"score": baseline_dev["score"], "metrics": baseline_dev["metrics"]},
                    "holdout": {"score": baseline_holdout["score"], "metrics": baseline_holdout["metrics"]},
                },
                "failure_samples": failures,
            }
            frontier_audit = run_frontier_policy_audit(
                provisional,
                best,
                reasons=frontier_audit_reasons,
                timeout=frontier_timeout,
            )
            if _frontier_blocks_adoption(frontier_audit):
                status = (
                    "pending_frontier_review"
                    if frontier_audit.get("human_required") or frontier_audit.get("rescue_status") == "pending_frontier_review"
                    else "frontier_rejected"
                )
                reason = f"frontier audit did not approve: {frontier_audit.get('summary') or frontier_audit.get('decision')}"
                active_policy = None
        if apply and active_policy:
            atomic_write_json(active_file, active_policy)
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
            "holdout": {"score": baseline_holdout["score"], "metrics": baseline_holdout["metrics"]},
        },
        "failure_samples": failures,
        "live_telemetry": live_telemetry,
        "adoption_gate": _adoption_gate_summary(
            baseline_dev=baseline_dev,
            baseline_holdout=baseline_holdout,
            min_improvement=min_improvement,
        ),
        "recent_rejection_blockers": recent_blockers,
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
    _persist_run(record, runs_dir=runs_dir, registry_file=registry_file)
    level = "info" if applied or status == "shadow_pass" else "warn"
    safe_append_event(level, f"recall-improve | {status}", run_id=run_id, reason=reason)
    return record


def _persist_run(record: dict[str, Any], *, runs_dir: Path, registry_file: Path) -> None:
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
    active = read_active_policy(active_file)
    current_run = str(active.get("run_id") or "")
    rows = read_jsonl(registry_file)
    applied_rows = [
        row for row in rows
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
        try:
            active_file.unlink()
        except FileNotFoundError:
            pass
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
    safe_append_event("warn", f"recall-improve | rollback {status}", from_run_id=current_run, to_run_id=target)
    return record


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _feedback_count(path: Path = RECALL_FEEDBACK_FILE) -> int:
    return len(read_jsonl(path))


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
    min_interval_hours: float = 24.0,
    min_new_feedback: int = 5,
    min_total_feedback: int = 3,
    frontier_mode: str = "auto",
    frontier_timeout: int | None = None,
    dry_run: bool = False,
    schedule_file: Path = SCHEDULE_FILE,
) -> dict[str, Any]:
    now = datetime.now()
    state = read_json_file(schedule_file)
    feedback_count = _feedback_count(feedback_file)
    last_run = _parse_ts(state.get("last_run_at"))
    last_feedback_count = int(state.get("last_feedback_count") or 0)
    age_hours = ((now - last_run).total_seconds() / 3600.0) if last_run else None
    new_feedback = max(0, feedback_count - last_feedback_count)
    enough_total = feedback_count >= min_total_feedback
    interval_due = last_run is None or (age_hours is not None and age_hours >= min_interval_hours)
    feedback_due = new_feedback >= min_new_feedback
    due = enough_total and (interval_due or feedback_due)
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
            "last_status": decision["status"],
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
        frontier_mode=frontier_mode,
        frontier_timeout=frontier_timeout,
    )
    next_state = {
        **state,
        "last_checked_at": decision["checked_at"],
        "last_run_at": result.get("ts") or decision["checked_at"],
        "last_run_id": result.get("run_id"),
        "last_status": result.get("status"),
        "last_feedback_count": feedback_count,
        "last_decision": decision,
    }
    atomic_write_json(schedule_file, next_state)
    safe_append_metric(
        "recall_improve",
        status=result.get("status"),
        applied=bool(result.get("applied")),
        examples=(result.get("dataset") or {}).get("examples"),
        eval_cache_entries=result.get("eval_cache_entries"),
    )
    return {"status": "ran", "decision": decision, "result": result, "schedule_state": next_state}


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
                hydrated["candidate_blockers"] = full.get("candidate_blockers") or _candidate_blocker_summary(
                    full.get("candidates") or []
                )
                hydrated["adoption_gate"] = full.get("adoption_gate")
                hydrated["recent_rejection_blockers"] = full.get("recent_rejection_blockers")
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
    parser = argparse.ArgumentParser(description="Run local self-improving recall policy experiments.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Propose, replay-evaluate, and optionally adopt a policy patch.")
    run.add_argument("--config")
    run.add_argument("--log-file", default=str(RECALL_LOG_FILE))
    run.add_argument("--feedback-file", default=str(RECALL_FEEDBACK_FILE))
    run.add_argument("--models", help="Comma-separated Ollama proposer models.")
    run.add_argument("--no-apply", dest="apply", action="store_false", default=True)
    run.add_argument("--no-heuristic", dest="include_heuristic", action="store_false", default=True)
    run.add_argument("--min-improvement", type=float, default=0.05)
    run.add_argument("--max-examples", type=int, default=120)
    run.add_argument("--frontier", choices=sorted(FRONTIER_AUDIT_MODES), default="auto")
    run.add_argument("--frontier-timeout", type=int)
    run.add_argument("--json", action="store_true")

    due = sub.add_parser("run-due", help="Run the improvement loop only when schedule/feedback gates are due.")
    due.add_argument("--config")
    due.add_argument("--log-file", default=str(RECALL_LOG_FILE))
    due.add_argument("--feedback-file", default=str(RECALL_FEEDBACK_FILE))
    due.add_argument("--models", help="Comma-separated Ollama proposer models.")
    due.add_argument("--no-apply", dest="apply", action="store_false", default=True)
    due.add_argument("--no-heuristic", dest="include_heuristic", action="store_false", default=True)
    due.add_argument("--min-improvement", type=float, default=0.05)
    due.add_argument("--max-examples", type=int, default=80)
    due.add_argument("--min-interval-hours", type=float, default=24.0)
    due.add_argument("--min-new-feedback", type=int, default=5)
    due.add_argument("--min-total-feedback", type=int, default=3)
    due.add_argument("--frontier", choices=sorted(FRONTIER_AUDIT_MODES), default="auto")
    due.add_argument("--frontier-timeout", type=int)
    due.add_argument("--dry-run", action="store_true")
    due.add_argument("--json", action="store_true")

    status = sub.add_parser("status", help="Show active policy and recent improvement runs.")
    status.add_argument("--json", action="store_true")

    rollback = sub.add_parser("rollback", help="Rollback to the previous accepted policy.")
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
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
        else:
            _print_run(payload)
        return 0 if payload["status"] in {"applied", "shadow_pass", "rejected", "blocked"} else 1
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
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
        else:
            print(f"status\t{payload.get('status')}")
            if payload.get("result"):
                print(f"run\t{payload['result'].get('run_id')}\t{payload['result'].get('status')}")
        return 0
    if args.command == "status":
        payload = improvement_snapshot()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
        else:
            _print_status(payload)
        return 0
    if args.command == "rollback":
        payload = rollback_policy()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
        else:
            print(f"rollback\t{payload['reason']}\t{payload.get('from_run_id') or '--'} -> {payload.get('to_run_id') or '--'}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
