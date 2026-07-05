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
    IMPROVEMENT_DIR,
    REGISTRY_FILE,
    RUNS_DIR,
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
from llm_wiki_mcp.runtime_status import safe_append_event


DEFAULT_IMPROVEMENT_MODELS = (
    "qwen3.6:35b-a3b-q8_0",
    "hf.co/douyamv/Gemma-4-31B-JANG_4M-CRACK-GGUF:latest",
)


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
    failure_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "task": "Propose one small recall policy patch for LLM Wiki.",
        "model_role": model,
        "rules": [
            "Use only allowed fields.",
            "Prefer one to four field changes.",
            "Do not propose graph-heavy changes; graph is not the primary strategy.",
            "Optimize recall quality without increasing false-positive memory injection.",
            "Output JSON only.",
        ],
        "allowed_fields": _allowed_field_summary(),
        "baseline_policy": policy_snapshot(baseline_policy),
        "baseline_metrics": baseline_eval.get("metrics", {}),
        "baseline_score": baseline_eval.get("score"),
        "failure_samples": failure_samples,
        "output": {
            "summary": "short title",
            "rationale": "why this should improve replay eval",
            "risk": "low|medium|high",
            "audit_recommended": False,
            "overrides": {"max_pages": 4},
        },
    }


def _call_ollama_proposer(
    *,
    model: str,
    baseline_policy: RecallPolicy,
    baseline_eval: dict[str, Any],
    failure_samples: list[dict[str, Any]],
    timeout_seconds: float = 180.0,
) -> PolicyProposal:
    from llm_wiki_mcp.ollama import OLLAMA_URL

    prompt = _proposal_prompt(
        model=model,
        baseline_policy=baseline_policy,
        baseline_eval=baseline_eval,
        failure_samples=failure_samples,
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
        overrides = normalize_policy_overrides(parsed.get("overrides"))
        if not overrides:
            raise ValueError("proposal contained no valid overrides")
        return PolicyProposal(
            source="ollama",
            model=model,
            proposal_id=f"{model}:{hashlib.sha1(json.dumps(overrides, sort_keys=True).encode()).hexdigest()[:8]}",
            summary=str(parsed.get("summary") or "policy patch")[:160],
            rationale=str(parsed.get("rationale") or "")[:1200],
            risk=str(parsed.get("risk") or "medium"),
            audit_recommended=bool(parsed.get("audit_recommended")),
            overrides=overrides,
        )
    except Exception as exc:
        elapsed_ms = int(round((time.perf_counter() - started) * 1000))
        return PolicyProposal(
            source="ollama",
            model=model,
            proposal_id=f"{model}:error",
            summary="proposal failed",
            rationale=f"{exc.__class__.__name__} after {elapsed_ms}ms",
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
    failure_samples: list[dict[str, Any]],
) -> list[PolicyProposal]:
    return [
        _call_ollama_proposer(
            model=model,
            baseline_policy=baseline_policy,
            baseline_eval=baseline_eval,
            failure_samples=failure_samples,
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
) -> dict[str, Any]:
    candidate_policy = _clone_policy(baseline_policy)
    applied_fields = apply_policy_overrides(candidate_policy, proposal.overrides)
    if not applied_fields:
        return {
            "proposal": asdict(proposal),
            "status": "invalid",
            "reason": "no valid policy fields",
        }
    candidate_dev = _evaluate(dev_examples, policy=candidate_policy)
    candidate_holdout = _evaluate(holdout_examples, policy=candidate_policy)
    accepted, checks = _gate_candidate(
        baseline_dev=baseline_dev,
        baseline_holdout=baseline_holdout,
        candidate_dev=candidate_dev,
        candidate_holdout=candidate_holdout,
        min_improvement=min_improvement,
    )
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
    active_file: Path = ACTIVE_POLICY_FILE,
    registry_file: Path = REGISTRY_FILE,
    runs_dir: Path = RUNS_DIR,
    episodes_file: Path = EPISODES_FILE,
) -> dict[str, Any]:
    run_id = _run_id()
    started = _now_iso()
    examples = build_dataset(log_file=log_file, feedback_file=feedback_file)
    if max_examples > 0:
        examples = examples[-max_examples:]
    write_episode_snapshot(examples, path=episodes_file)
    model_list = configured_models(models)
    if not examples:
        record = {
            "schema_version": 1,
            "run_id": run_id,
            "ts": started,
            "status": "blocked",
            "applied": False,
            "reason": "no recall feedback examples available",
            "dataset": {"examples": 0, "log_file": str(log_file), "feedback_file": str(feedback_file)},
            "models": list(model_list),
        }
        _persist_run(record, runs_dir=runs_dir, registry_file=registry_file)
        return record

    dev_examples, holdout_examples = split_examples(examples)
    baseline_policy = load_policy(config_file) if config_file else load_policy()
    baseline_dev = _evaluate(dev_examples, policy=baseline_policy)
    baseline_holdout = _evaluate(holdout_examples, policy=baseline_policy)
    failures = _failure_samples(baseline_dev)

    proposals = propose_with_models(
        models=model_list,
        baseline_policy=baseline_policy,
        baseline_eval=baseline_dev,
        failure_samples=failures,
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
        )
        for proposal in proposals
        if proposal.overrides
    ]
    best = _best_candidate(candidates)
    active_policy: dict[str, Any] | None = None
    applied = False
    status = "rejected"
    reason = "no candidate passed adoption gate"
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
        if apply:
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
        },
        "models": list(model_list),
        "baseline_policy": policy_snapshot(baseline_policy),
        "baseline": {
            "dev": {"score": baseline_dev["score"], "metrics": baseline_dev["metrics"]},
            "holdout": {"score": baseline_holdout["score"], "metrics": baseline_holdout["metrics"]},
        },
        "failure_samples": failures,
        "proposals": [asdict(proposal) for proposal in proposals],
        "candidates": candidates,
        "best": best,
        "active_policy": active_policy,
        "frontier_audit_recommended": bool(best and best["proposal"].get("audit_recommended")),
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
        "frontier_audit_recommended": record.get("frontier_audit_recommended", False),
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


def improvement_snapshot(
    *,
    active_file: Path = ACTIVE_POLICY_FILE,
    registry_file: Path = REGISTRY_FILE,
    limit: int = 12,
) -> dict[str, Any]:
    active = read_json_file(active_file)
    history = read_jsonl(registry_file, limit=limit)
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
        "paths": {
            "active_policy": str(active_file),
            "registry": str(registry_file),
            "episodes": str(EPISODES_FILE),
            "runs": str(RUNS_DIR),
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
    run.add_argument("--json", action="store_true")

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
        )
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
        else:
            _print_run(payload)
        return 0 if payload["status"] in {"applied", "shadow_pass", "rejected", "blocked"} else 1
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
