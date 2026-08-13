#!/usr/bin/env python3
"""Capture and compare privacy-safe Recall parity receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

SCHEMA = "chronovisor.recall-parity.v1"
PROTOCOL = "deterministic-teacher-rerank-filter-render.v2"
COHORT_SIZE = 100
_STORE_WARM = False
PROJECTION_FIELDS = (
    "candidate_set_sha256",
    "teacher_top30_sha256",
    "authoritative_rerank_sha256",
    "sensitive_decisions_sha256",
    "filtered_authority_sha256",
    "context_items_sha256",
    "rendered_context_sha256",
)
RECEIPT_FIELDS = (
    "case_id",
    "input_sha256",
    "status",
    "decision",
    "search_mode",
    *PROJECTION_FIELDS,
)


def _sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _case_id(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        f"{row.get('prompt_hash', '')}{row.get('decision_id', '')}".encode()
    ).hexdigest()


def _stable_prompt_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha1(normalized.encode()).hexdigest()[:16]


def _input_sha256(row: dict[str, Any]) -> str:
    preview = str(row.get("prompt_preview") or "")
    prompt_hash = _stable_prompt_hash(preview)
    if prompt_hash != row.get("prompt_hash"):
        raise ValueError("prompt preview does not match prompt hash")
    return _sha(
        {
            "prompt_hash": prompt_hash,
            "decision_id_sha256": hashlib.sha256(
                str(row.get("decision_id") or "").encode()
            ).hexdigest(),
            "host_sha256": hashlib.sha256(
                str(row.get("host") or "codex").encode()
            ).hexdigest(),
            "cwd_sha256": hashlib.sha256(
                str(row.get("cwd") or "").encode()
            ).hexdigest(),
            "session_id_sha256": hashlib.sha256(
                str(row.get("session_id") or "").encode()
            ).hexdigest(),
        }
    )


def eligible_rows(log_file: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in log_file.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        preview = row.get("prompt_preview") if isinstance(row, dict) else None
        if (
            isinstance(row, dict)
            and row.get("schema_version") == 2
            and row.get("event") == "UserPromptSubmit"
            and row.get("status") == "ok"
            and row.get("decision") == "read"
            and isinstance(preview, str)
            and row.get("prompt_chars") == len(preview)
            and isinstance(row.get("prompt_hash"), str)
            and row["prompt_hash"]
            and isinstance(row.get("decision_id"), str)
            and row["decision_id"]
            and _stable_prompt_hash(preview) == row.get("prompt_hash")
        ):
            rows.append(row)
    rows.sort(key=lambda row: (str(row["prompt_hash"]), str(row["decision_id"])))
    return rows


def select_cohort(log_file: Path, *, limit: int = 100) -> list[dict[str, Any]]:
    rows = eligible_rows(log_file)
    if len(rows) < limit:
        raise ValueError(f"eligible cohort has {len(rows)} rows; need {limit}")
    return rows[:limit]


def _page_projection(value: Any, uid_lookup: Callable[[str], str]) -> dict[str, Any]:
    page_id = str(getattr(value, "page_id", "") or "")
    uid = str(getattr(value, "uid", "") or "") or uid_lookup(page_id)
    return {
        "page_uid": uid,
        "page_id_sha256": hashlib.sha256(page_id.encode()).hexdigest(),
        "score_hex": float(getattr(value, "score", 0.0) or 0.0).hex(),
        "sensitivity": str(getattr(value, "sensitivity", "normal") or "normal"),
    }


def _page_identity(value: Any, uid_lookup: Callable[[str], str]) -> dict[str, str]:
    page_id = str(getattr(value, "page_id", "") or "")
    uid = str(getattr(value, "uid", "") or "") or uid_lookup(page_id)
    return {
        "page_uid": uid,
        "page_id_sha256": hashlib.sha256(page_id.encode()).hexdigest(),
    }


def _context_projection(value: Any, uid_lookup: Callable[[str], str]) -> dict[str, Any]:
    projected = _page_projection(value, uid_lookup)
    projected.update(
        {
            "title": str(getattr(value, "title", "") or ""),
            "updated": str(getattr(value, "updated", "") or ""),
            "snippets": [str(item) for item in (getattr(value, "snippets", []) or [])],
            "certificate_id": str(getattr(value, "certificate_id", "") or ""),
            "evidence_kind": str(getattr(value, "evidence_kind", "") or ""),
            "source_line": int(getattr(value, "source_line", 0) or 0),
        }
    )
    return projected


def _projection_hash(
    values: list[Any],
    uid_lookup: Callable[[str], str],
    *,
    unordered: bool = False,
) -> str:
    projected = [_page_projection(value, uid_lookup) for value in values]
    if unordered:
        projected = sorted(
            {
                json.dumps(item, sort_keys=True, separators=(",", ":"))
                for item in projected
            }
        )
    return _sha(projected)


def _context_hash(values: list[Any], uid_lookup: Callable[[str], str]) -> str:
    return _sha([_context_projection(value, uid_lookup) for value in values])


def _elapsed_ms(started_ns: int) -> int:
    return max(0, round((time.perf_counter_ns() - started_ns) / 1_000_000))


def _aggregate(values: list[int]) -> dict[str, int]:
    ordered = sorted(values)

    def percentile(percent: int) -> int:
        if not ordered:
            return 0
        rank = max(0, ((len(ordered) * percent + 99) // 100) - 1)
        return ordered[rank]

    return {
        "count": len(ordered),
        "p50": percentile(50),
        "p95": percentile(95),
        "p99": percentile(99),
    }


def _runtime_identity(source_root: Path, source_commit: str) -> dict[str, str]:
    from chronovisor.recall import recall_runtime

    source_root = source_root.resolve()
    module = Path(recall_runtime.__file__).resolve()
    try:
        module.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"runtime module is outside --source-root: {module}") from exc
    if len(source_commit) != 40 or set(source_commit) - set("0123456789abcdef"):
        raise ValueError("--source-commit must be a full lowercase Git SHA")
    try:
        head = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD^{tree}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            [
                "git",
                "-C",
                str(source_root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("--source-root must be a Git worktree") from exc
    if head != source_commit:
        raise ValueError("--source-commit does not match source worktree HEAD")
    if dirty:
        raise ValueError("source worktree has tracked changes")
    return {
        "source_commit": source_commit,
        "source_tree": tree,
        "runtime_module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
    }


def _input_snapshot(_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Hash the immutable inputs used by a replay without exposing their content."""

    from chronovisor.core import reranker_client, runtime_config, semantic_client
    from chronovisor.recall import recall_runtime, recall_session

    root = recall_runtime.CHRONOVISOR_ROOT
    paths = [
        root / "config.toml",
        root / ".index" / "backlinks.json",
        root / ".index" / "lexical.sqlite",
        root / ".index" / "lexical.sqlite-wal",
        root / ".index" / "pages.json",
        root / ".index" / "semantic" / "active.json",
        root / "recall" / "content-feedback.jsonl",
        root / "recall" / "feedback.jsonl",
        root / "recall" / "query-hints.json",
        root / "recall" / "retention.json",
        root / "recall" / "search-policy.json",
        root / "runtime" / "evidence-reconstruction" / "promotion.json",
        root / "runtime" / "recall-field" / "promotion.json",
        root / "runtime" / "recall-improvement" / "active-policy.json",
        root / "runtime" / "recall-rubric" / "active.json",
        root / "runtime" / "typed-graph" / "promotion.json",
    ]
    paths.extend(sorted(recall_session.SESSIONS_DIR.glob("*.json")))
    for directory in (
        root / ".index" / "semantic",
        root / "knowledge-graph",
        root / "pages",
        root / "system",
    ):
        if directory.is_dir():
            paths.extend(
                path
                for path in sorted(directory.rglob("*"))
                if path.is_file() and path.suffix not in {".bak", ".lock", ".shm"}
            )

    digest = hashlib.sha256()
    try:
        semantic = semantic_client.health(runtime_config.load_search_embedding_config())
        semantic_identity = {
            key: semantic.get(key)
            for key in (
                "status",
                "ready",
                "generation_id",
                "model",
                "revision",
                "device",
                "routes",
            )
        }
        semantic_index = semantic.get("index")
        if isinstance(semantic_index, dict):
            semantic_identity["index"] = {
                key: semantic_index.get(key)
                for key in ("status", "generation_id", "error")
            }
    except Exception as exc:
        semantic_identity = {"error": type(exc).__name__}
    try:
        reranker = reranker_client.health(runtime_config.load_reranker_config())
        reranker_runtime = reranker.get("runtime")
        reranker_warmup = reranker.get("warmup")
        reranker_identity = {
            "status": reranker.get("status"),
            "ready": reranker.get("ready"),
            "route": reranker.get("route"),
            "warmup_status": (
                reranker_warmup.get("status")
                if isinstance(reranker_warmup, dict)
                else None
            ),
            "runtime_commit": (
                reranker_runtime.get("commit_id")
                if isinstance(reranker_runtime, dict)
                else None
            ),
        }
    except Exception as exc:
        reranker_identity = {"error": type(exc).__name__}
    digest.update(
        _sha(
            {
                "semantic_service": semantic_identity,
                "reranker_service": reranker_identity,
            }
        ).encode()
    )
    count = 0
    for path in sorted(set(paths), key=lambda value: str(value)):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = hashlib.sha256(str(path).encode()).hexdigest()
        digest.update(relative.encode() + b"\0")
        try:
            stream = path.open("rb")
        except OSError:
            digest.update(b"missing\0")
            continue
        count += 1
        with stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return {"sha256": digest.hexdigest(), "file_count": count}


def _protocol_controls_sha256() -> str:
    return _sha(
        {
            "judge": "production-auto-with-zero-invocation-required",
            "rewrite": "production-off-required",
            "state_context": "production-snapshot",
            "session_read": "preserved",
            "session_write": "disabled",
            "store_refresh": "first_case_then_frozen",
            "field_processor_evidence": "non_authoritative_only_disabled",
            "scheduler": "no_op",
            "durable_writes": "disabled",
        }
    )


def _empty_receipt(case_id: str, *, status: str = "error") -> dict[str, str]:
    return {
        "case_id": case_id,
        "input_sha256": "",
        "status": status,
        "decision": "",
        "search_mode": "",
        **{field: _sha([]) for field in PROJECTION_FIELDS[:-1]},
        "rendered_context_sha256": hashlib.sha256(b"").hexdigest(),
    }


def run_production(row: dict[str, Any]) -> dict[str, Any]:
    """Run production Recall read-only and return receipts, never raw values."""

    previous = os.environ.get("CHRONOVISOR_READ_ONLY")
    os.environ["CHRONOVISOR_READ_ONLY"] = "1"
    try:
        return _run_production(row)
    finally:
        if previous is None:
            os.environ.pop("CHRONOVISOR_READ_ONLY", None)
        else:
            os.environ["CHRONOVISOR_READ_ONLY"] = previous


def _run_production(row: dict[str, Any]) -> dict[str, Any]:
    global _STORE_WARM

    from chronovisor.core import index_store, research_scheduler
    from chronovisor.core import search as core_search
    from chronovisor.recall import (
        recall_compiler,
        recall_field,
        recall_field_candidate,
        recall_field_schema,
        recall_processor,
        recall_runtime,
        recall_session,
    )
    from chronovisor.research.evidence_runtime import load_evidence_rollout

    observed: dict[str, Any] = {
        "raw": [],
        "teacher": [],
        "authority": [],
        "sensitive": [],
        "latency": {"teacher": [], "reranker": [], "context": []},
    }
    raw_search = recall_runtime.run_search
    teacher_search = recall_runtime.search_candidates
    rerank = recall_processor.rank_recall_candidates
    sensitive = recall_runtime.should_filter_sensitive_result
    collect_context = recall_runtime.collect_context
    uid_lookup = recall_runtime.page_uid_for_id

    def raw_wrapper(*args: Any, **kwargs: Any) -> Any:
        result = raw_search(*args, **kwargs)
        observed["raw"].extend(result[0])
        return result

    def teacher_wrapper(*args: Any, **kwargs: Any) -> Any:
        observed["raw"] = []
        observed["sensitive"] = []
        started_ns = time.perf_counter_ns()
        try:
            result = teacher_search(*args, **kwargs)
        finally:
            observed["latency"]["teacher"].append(_elapsed_ms(started_ns))
        observed["teacher"] = list(result[0])
        observed["authority"] = list(observed["teacher"])
        return result

    def rerank_wrapper(*args: Any, **kwargs: Any) -> Any:
        started_ns = time.perf_counter_ns()
        try:
            result = rerank(*args, **kwargs)
        finally:
            observed["latency"]["reranker"].append(_elapsed_ms(started_ns))
        metadata = result[1] if isinstance(result[1], dict) else {}
        if (
            metadata.get("mode") in {"canary", "on"}
            and metadata.get("status") == "applied"
        ):
            observed["authority"] = list(result[0])
        return result

    def sensitive_wrapper(value: Any, request: Any) -> bool:
        decision = sensitive(value, request)
        observed["sensitive"].append(
            {**_page_identity(value, uid_lookup), "filtered": bool(decision)}
        )
        return decision

    def context_wrapper(*args: Any, **kwargs: Any) -> Any:
        started_ns = time.perf_counter_ns()
        try:
            result = collect_context(*args, **kwargs)
        finally:
            observed["latency"]["context"].append(_elapsed_ms(started_ns))
        return result

    policy = recall_runtime.load_policy()
    policy.log_decisions = False
    policy.processor_shadow_enabled = False
    if policy.rewrite_enabled:
        raise RuntimeError(
            "paired replay requires the production rewrite lane to be off"
        )
    field_config = recall_field_candidate.effective_rollout(
        recall_field_schema.load_recall_field_config()
    )
    if field_config.mode == "active":
        raise RuntimeError("paired replay refuses to disable active Field authority")

    total_started_ns = time.perf_counter_ns()
    request = recall_runtime.RecallRequest(
        host=str(row.get("host") or "codex"),
        event="UserPromptSubmit",
        prompt=str(row["prompt_preview"]),
        cwd=str(row.get("cwd") or ""),
        session_id=str(row.get("session_id") or ""),
        decision_id=str(row["decision_id"]),
    )
    if recall_runtime.processor_authority_for_request(policy, request):
        raise RuntimeError("paired replay refuses to disable Processor authority")
    evidence_rollout = load_evidence_rollout(recall_runtime.CHRONOVISOR_ROOT)
    if int(evidence_rollout.get("canary_percent") or 0) > 0:
        raise RuntimeError("paired replay refuses to disable evidence authority")
    with ExitStack() as stack:
        stack.enter_context(patch.object(recall_runtime, "run_search", raw_wrapper))
        stack.enter_context(
            patch.object(recall_runtime, "search_candidates", teacher_wrapper)
        )
        stack.enter_context(
            patch.object(recall_processor, "rank_recall_candidates", rerank_wrapper)
        )
        stack.enter_context(
            patch.object(
                recall_runtime,
                "should_filter_sensitive_result",
                sensitive_wrapper,
            )
        )
        stack.enter_context(
            patch.object(recall_runtime, "collect_context", context_wrapper)
        )
        stack.enter_context(
            patch.object(
                recall_runtime,
                "run_local_judge",
                lambda *_a, **_k: (_ for _ in ()).throw(
                    RuntimeError("paired replay excludes cases invoking the judge")
                ),
            )
        )
        stack.enter_context(
            patch.object(
                recall_runtime,
                "okf_startup_status",
                lambda *_a: SimpleNamespace(allowed=True, layout="okf_v0_2"),
            )
        )
        stack.enter_context(
            patch.object(
                index_store, "okf_runtime_operation", lambda *_a: nullcontext()
            )
        )
        if _STORE_WARM:
            stack.enter_context(
                patch.object(
                    index_store.IndexStore,
                    "refresh_if_stale",
                    lambda *_a, **_k: None,
                )
            )
            stack.enter_context(
                patch.object(core_search.BM25Index, "build", lambda *_a, **_k: None)
            )
        stack.enter_context(
            patch.object(recall_session, "cleanup_sessions", lambda *_a, **_k: None)
        )
        stack.enter_context(
            patch.object(
                recall_session,
                "update_session_after_recall",
                lambda *_a, **_k: None,
            )
        )
        stack.enter_context(
            patch.object(
                recall_runtime,
                "observe_evidence_reconstruction",
                lambda *_a, **_k: {"status": "skipped", "reason": "parity"},
            )
        )
        stack.enter_context(
            patch.object(
                recall_field,
                "run_field_turn",
                lambda *_a, **_k: {"status": "disabled", "reason": "parity"},
            )
        )
        stack.enter_context(
            patch.object(
                research_scheduler,
                "foreground_lane",
                lambda **_k: nullcontext(
                    SimpleNamespace(
                        resource_wait_ms=0,
                        research_overlap=False,
                        preempted=False,
                    )
                ),
            )
        )
        for owner, name in (
            (recall_runtime, "append_recall_log"),
            (recall_runtime, "append_jsonl_durable"),
            (recall_field, "queue_teacher_commits"),
            (recall_field_candidate, "append_candidate_trace"),
            (recall_compiler, "append_shadow_trace"),
            (recall_processor, "append_certificates"),
        ):
            stack.enter_context(patch.object(owner, name, lambda *_a, **_k: None))
        result = recall_runtime.run_recall(request, policy)
    _STORE_WARM = True
    observed["latency"]["total"] = [_elapsed_ms(total_started_ns)]

    teacher = observed["teacher"]
    authority = observed["authority"] or teacher
    context = list(result.context_items)
    sensitive_projection = sorted(
        {
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            for item in observed["sensitive"]
        }
    )
    filtered_ids = {
        str(item["page_id_sha256"])
        for item in observed["sensitive"]
        if item["filtered"]
    }
    filtered_authority = [
        value
        for value in authority
        if hashlib.sha256(str(getattr(value, "page_id", "") or "").encode()).hexdigest()
        not in filtered_ids
    ]
    receipt = {
        "case_id": _case_id(row),
        "input_sha256": _input_sha256(row),
        "status": result.status,
        "decision": result.decision,
        "search_mode": result.search_mode,
        "candidate_set_sha256": _projection_hash(
            observed["raw"], uid_lookup, unordered=True
        ),
        "teacher_top30_sha256": _projection_hash(teacher[:30], uid_lookup),
        "authoritative_rerank_sha256": _projection_hash(authority, uid_lookup),
        "sensitive_decisions_sha256": _sha(sensitive_projection),
        "filtered_authority_sha256": _projection_hash(filtered_authority, uid_lookup),
        "context_items_sha256": _context_hash(context, uid_lookup),
        "rendered_context_sha256": hashlib.sha256(result.context.encode()).hexdigest(),
    }
    receipt["_stage_latency_ms"] = {
        name: sum(values) for name, values in observed["latency"].items() if values
    }
    return receipt


def _cases(payload: dict[str, Any]) -> list[str]:
    paired = payload.get("paired_replay")
    projections = paired.get("projections") if isinstance(paired, dict) else None
    if (
        not isinstance(paired, dict)
        or paired.get("schema") != SCHEMA
        or paired.get("protocol") != PROTOCOL
        or paired.get("candidate_count") != COHORT_SIZE
        or not isinstance(projections, list)
        or len(projections) != COHORT_SIZE
    ):
        raise ValueError("cases artifact must contain exactly 100 parity projections")
    case_ids = [str(row.get("case_id") or "") for row in projections]
    if any(not case_id for case_id in case_ids) or len(set(case_ids)) != COHORT_SIZE:
        raise ValueError("cases artifact contains invalid or duplicate case IDs")
    return case_ids


def _field_latency(row: dict[str, Any]) -> int | None:
    evidence = row.get("evidence_features")
    field_shadow = evidence.get("field_shadow") if isinstance(evidence, dict) else None
    value = field_shadow.get("latency_ms") if isinstance(field_shadow, dict) else None
    return value if isinstance(value, int) and value >= 0 else None


def capture(
    log_file: Path,
    *,
    limit: int = COHORT_SIZE,
    cases: dict[str, Any] | None = None,
    runner: Callable[[dict[str, Any]], dict[str, Any]] = run_production,
    runtime_identity: dict[str, str] | None = None,
    snapshotter: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if limit != COHORT_SIZE:
        raise ValueError(f"paired replay cohort is fixed at {COHORT_SIZE} cases")
    source = eligible_rows(log_file)
    snapshot_before = snapshotter(source) if snapshotter else None
    if cases is None:
        selected = []
        projections = []
        for row in source:
            try:
                receipt = runner(row)
            except Exception:
                continue
            if receipt.get("status") != "ok" or receipt.get("decision") != "read":
                continue
            receipt["case_id"] = _case_id(row)
            receipt["input_sha256"] = _input_sha256(row)
            selected.append(row)
            projections.append(receipt)
            if len(selected) == COHORT_SIZE:
                break
        if len(selected) != COHORT_SIZE:
            raise ValueError(f"successful replay has {len(selected)} rows; need 100")
    else:
        case_ids = _cases(cases)
        by_case = {_case_id(row): row for row in source}
        missing = [case_id for case_id in case_ids if case_id not in by_case]
        if missing:
            raise ValueError(f"source log is missing {len(missing)} baseline cases")
        selected = [by_case[case_id] for case_id in case_ids]
        projections = []
        for row in selected:
            try:
                receipt = runner(row)
                receipt["case_id"] = _case_id(row)
                receipt["input_sha256"] = _input_sha256(row)
                projections.append(receipt)
            except Exception:
                failed = _empty_receipt(_case_id(row))
                failed["input_sha256"] = _input_sha256(row)
                projections.append(failed)
    snapshot_after = snapshotter(source) if snapshotter else None
    if snapshot_before != snapshot_after:
        raise ValueError("replay input snapshot changed during capture")
    latency: dict[str, list[int]] = {
        "teacher": [],
        "reranker": [],
        "context": [],
        "total": [],
        "field": [],
    }
    public_projections: list[dict[str, str]] = []
    for row, receipt in zip(selected, projections, strict=True):
        stage_latency = receipt.pop("_stage_latency_ms", None)
        if isinstance(stage_latency, dict):
            for name in ("teacher", "reranker", "context", "total"):
                value = stage_latency.get(name)
                if isinstance(value, int) and value >= 0:
                    latency[name].append(value)
        field_latency = _field_latency(row)
        if field_latency is not None:
            latency["field"].append(field_latency)
        public_projections.append(
            {field: str(receipt.get(field, "")) for field in RECEIPT_FIELDS}
        )
    case_ids = [_case_id(row) for row in selected]
    return {
        "paired_replay": {
            "schema": SCHEMA,
            "protocol": PROTOCOL,
            "protocol_controls_sha256": _protocol_controls_sha256(),
            "runtime_identity": runtime_identity
            or {
                "source_commit": "0" * 40,
                "source_tree": "0" * 40,
                "runtime_module_sha256": "0" * 64,
            },
            "input_snapshot": snapshot_before or {"sha256": "0" * 64, "file_count": 0},
            "candidate_count": len(public_projections),
            "source_cohort_sha256": _sha(case_ids),
            "input_cohort_sha256": _sha(
                [receipt["input_sha256"] for receipt in public_projections]
            ),
            "baseline_stage_latency_ms": {
                name: _aggregate(values) for name, values in latency.items()
            },
            "projections": public_projections,
        }
    }


def comparison_errors(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        baseline_cases = _cases(baseline)
        candidate_cases = _cases(candidate)
    except ValueError as exc:
        return [str(exc)]
    baseline_pair = baseline["paired_replay"]
    candidate_pair = candidate["paired_replay"]
    baseline_identity = baseline_pair.get("runtime_identity")
    candidate_identity = candidate_pair.get("runtime_identity")

    def valid_identity(value: Any) -> bool:
        return isinstance(value, dict) and all(
            isinstance(value.get(key), str)
            and len(value[key]) == length
            and not set(value[key]) - set("0123456789abcdef")
            for key, length in (
                ("source_commit", 40),
                ("source_tree", 40),
                ("runtime_module_sha256", 64),
            )
        )

    if not valid_identity(baseline_identity) or not valid_identity(candidate_identity):
        errors.append("runtime identity invalid")
    elif (
        baseline_identity["source_commit"] == candidate_identity["source_commit"]
        or baseline_identity["source_tree"] == candidate_identity["source_tree"]
    ):
        errors.append("candidate runtime is not a distinct source tree")
    if baseline_pair.get("protocol") != candidate_pair.get("protocol"):
        errors.append("protocol mismatch")
    if baseline_pair.get("source_cohort_sha256") != candidate_pair.get(
        "source_cohort_sha256"
    ):
        errors.append("source cohort mismatch")
    if baseline_pair.get("input_cohort_sha256") != candidate_pair.get(
        "input_cohort_sha256"
    ):
        errors.append("input cohort mismatch")
    if baseline_pair.get("protocol_controls_sha256") != candidate_pair.get(
        "protocol_controls_sha256"
    ):
        errors.append("protocol controls mismatch")
    if baseline_pair.get("input_snapshot") != candidate_pair.get("input_snapshot"):
        errors.append("input snapshot mismatch")
    if baseline_cases != candidate_cases:
        errors.append("case order mismatch")
        return errors
    for index, (expected, actual) in enumerate(
        zip(
            baseline_pair["projections"],
            candidate_pair["projections"],
            strict=True,
        )
    ):
        for field in RECEIPT_FIELDS:
            if expected.get(field) != actual.get(field):
                errors.append(f"projection {index} {field} mismatch")
    return errors


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    capture_parser = commands.add_parser("capture")
    capture_parser.add_argument("--log", type=Path, required=True)
    capture_parser.add_argument("--output", type=Path)
    capture_parser.add_argument("--cases", type=Path)
    capture_parser.add_argument("--source-root", type=Path, required=True)
    capture_parser.add_argument("--source-commit", required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "capture":
        case_payload = _load(args.cases) if args.cases else None
        text = (
            json.dumps(
                capture(
                    args.log,
                    cases=case_payload,
                    runtime_identity=_runtime_identity(
                        args.source_root, args.source_commit
                    ),
                    snapshotter=_input_snapshot,
                ),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        return 0
    errors = comparison_errors(_load(args.baseline), _load(args.candidate))
    if errors:
        sys.stderr.write("\n".join(errors) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
