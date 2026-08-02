"""Autonomous supervision, learning, and rollout control for Recall Field."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import exclusive_text_file_lock
from chronovisor.core.link_fix import atomic_write
from chronovisor.core.store import CHRONOVISOR_ROOT, find_page
from chronovisor.recall.recall_answer_eval import (
    ANSWER_ADAPTER_REGISTRY,
    ANSWER_EPISODE_LEDGER,
    ANSWER_EXECUTION_LEDGER,
    ANSWER_REVIEW_LEDGER,
    LOCKED_ANSWER_EVAL_ARTIFACT,
    TRAIN_ANSWER_EVAL_ARTIFACT,
    builtin_field_environment_identity,
    validate_answer_artifact_set,
    validate_answer_outcome_artifact,
    validate_locked_answer_artifact,
)
from chronovisor.recall.recall_confidence import (
    cluster_rate_wilson_interval,
    manifest_sha256,
)
from chronovisor.recall.recall_label_factory import (
    build_label_ledger,
    default_label_ledger_inputs,
    materialize_label_ledger,
)
from chronovisor.recall.recall_learning import (
    append_policy_history,
    decide_learning_update,
    load_last_known_good,
    verify_policy_history,
    write_last_known_good,
)
from chronovisor.recall.recall_log_schema import join_used_recall_episodes

RUNTIME_DIR = CHRONOVISOR_ROOT / "runtime" / "recall-field"
GROWTH_STATE_FILE = RUNTIME_DIR / "growth-state.json"
GROWTH_HISTORY_FILE = RUNTIME_DIR / "growth-history.jsonl"
CANDIDATE_TRACE_FILE = RUNTIME_DIR / "candidate-trace.jsonl"
PROMOTION_ARTIFACT = RUNTIME_DIR / "promotion.json"
POLICY_HISTORY_FILE = RUNTIME_DIR / "policy-history.jsonl"
LAST_KNOWN_GOOD_POLICY_FILE = RUNTIME_DIR / "last-known-good-policy.json"
LOCKED_E2E_ARTIFACT = (
    CHRONOVISOR_ROOT / "runtime" / "search-eval" / "recall-field-locked-e2e.json"
)
COMPILER_TRACE_FILE = (
    CHRONOVISOR_ROOT / "runtime" / "recall-compiler" / "shadow-trace.jsonl"
)

MIN_STRONG_POSITIVES = 200
MIN_STRONG_SESSIONS = 20
MIN_CANDIDATE_TRACES = 100
MIN_CANDIDATE_SESSIONS = 20
MIN_PROCESSOR_USED_EPISODES = 50
MIN_TEACHER_COVERAGE = 0.99
MIN_PROCESSOR_USED_COVERAGE = 0.99
MIN_PROCESSOR_USED_PRECISION = 0.90
MIN_TEACHER_COVERAGE_LCB = 0.95
MIN_PROCESSOR_USED_COVERAGE_LCB = 0.95
MIN_PROCESSOR_USED_PRECISION_LCB = 0.85
CANARY_STEPS = (5, 25, 100)
CANARY_ADVANCE_SAMPLES = 100
QUALITY_WINDOW = 200


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path, *, limit: int | None = 20_000) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if limit is not None:
            lines = lines[-max(1, limit) :]
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _candidate_confidence_case(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the exact authority confidence case for one complete quality row."""

    if row.get("schema_version") != 3 or row.get("quality_eligible") is not True:
        return None
    session = str(row.get("session_hash") or "")
    query_sha = str(row.get("query_sha256") or row.get("prompt_sha256") or "")
    raw_lists = [
        row.get("field_page_ids"),
        row.get("teacher_page_ids"),
        row.get("committed_page_ids"),
    ]
    if (
        not session
        or len(query_sha) != 64
        or any(char not in "0123456789abcdef" for char in query_sha.casefold())
        or any(
            not isinstance(values, list)
            or any(not isinstance(value, str) or not value for value in values)
            for values in raw_lists
        )
    ):
        return None
    field_ids, teacher_ids, committed_ids = (
        set(values) for values in raw_lists if isinstance(values, list)
    )
    bound_page_ids = field_ids | teacher_ids | committed_ids
    hashes_value = row.get("page_content_sha256")
    uids_value = row.get("page_uids")
    hashes = hashes_value if isinstance(hashes_value, Mapping) else {}
    uids = uids_value if isinstance(uids_value, Mapping) else {}
    if (
        not bound_page_ids
        or not bound_page_ids <= set(hashes)
        or not bound_page_ids <= set(uids)
        or any(
            not isinstance(hashes.get(page_id), str)
            or len(str(hashes[page_id])) != 64
            or any(
                char not in "0123456789abcdef"
                for char in str(hashes[page_id]).casefold()
            )
            or not isinstance(uids.get(page_id), str)
            or not str(uids[page_id])
            for page_id in bound_page_ids
        )
    ):
        return None
    expected_nodes = list(
        dict.fromkeys(
            [f"session:{session}", f"query:{query_sha}"]
            + [f"page:{page_id}" for page_id in sorted(bound_page_ids)]
            + [f"uid:{uids[page_id]}" for page_id in sorted(bound_page_ids)]
            + [f"content:{hashes[page_id]}" for page_id in sorted(bound_page_ids)]
        )
    )
    if row.get("cluster_nodes") != expected_nodes:
        return None
    case = {
        "session_hash": session,
        "query_sha256": query_sha,
        "cluster_nodes": expected_nodes,
        "teacher_coverage": (
            len(teacher_ids & field_ids) / len(teacher_ids) if teacher_ids else 0.0
        ),
        "field_precision": (
            len(teacher_ids & field_ids) / len(field_ids) if field_ids else 0.0
        ),
        "commit_coverage": (
            len(committed_ids & field_ids) / len(committed_ids)
            if committed_ids
            else 0.0
        ),
    }
    case["case_sha256"] = hashlib.sha256(
        json.dumps(case, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return case


def _validated_candidate_trace_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Validate the complete append-only candidate trace chain."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            return rows, "candidate_trace_chain_invalid"
        if not isinstance(row, dict):
            return rows, "candidate_trace_chain_invalid"
        rows.append(row)
    previous = "0" * 64
    cumulative = 0
    for row in rows:
        unsigned = {key: value for key, value in row.items() if key != "record_sha256"}
        expected_cumulative = cumulative + int(
            _candidate_confidence_case(row) is not None
        )
        if (
            row.get("schema_version") != 3
            or row.get("previous_record_sha256") != previous
            or row.get("record_sha256")
            != hashlib.sha256(
                json.dumps(
                    unsigned,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            or row.get("cumulative_eligible_trace_count") != expected_cumulative
        ):
            return rows, "candidate_trace_chain_invalid"
        previous = str(row["record_sha256"])
        cumulative = expected_cumulative
    return rows, ""


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
    return round(ordered[max(0, index)], 3)


def _p95_improvement(teacher: list[float], field: list[float]) -> float | None:
    teacher_p95 = _percentile(teacher, 0.95)
    field_p95 = _percentile(field, 0.95)
    if teacher_p95 is None or field_p95 is None:
        return None
    return round(teacher_p95 - field_p95, 3)


def _is_stable_candidate_row(row: Mapping[str, Any]) -> bool:
    """Return whether a trace contains an actual stable-topic Field comparison.

    Topic resets, empty working sets, and canary exclusions correctly fall back to
    the full teacher.  They remain part of operational safety metrics, but they
    are not evidence about Field ranking quality.

    Schema-v1 traces did not identify the quality population explicitly, so an
    observed/active row remains the conservative compatibility signal.  Newer
    traces mark stable-topic attempts directly.  A failed verifier is still a
    quality miss even though the request correctly falls back to the teacher.
    """

    if "quality_eligible" in row:
        return row.get("quality_eligible") is True
    return str(row.get("status") or "") in {"observed", "active"} and (
        "full_search_required" not in row or row.get("full_search_required") is False
    )


def candidate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate privacy-safe Field/teacher comparisons.

    End-to-end latency, deadline misses, and fallback rate cover every turn.
    Ranking coverage and Field/teacher latency compare only stable-topic rows
    where the Field was actually attempted; verifier failures remain quality
    misses while the request itself falls back safely.
    """

    sessions: set[str] = set()
    stable_sessions: set[str] = set()
    coverage_sessions: set[str] = set()
    commit_sessions: set[str] = set()
    paired_latency_sessions: set[str] = set()
    stable_traces = 0
    coverage_evidence_traces = 0
    commit_evidence_traces = 0
    paired_latency_traces = 0
    teacher_pages = 0
    teacher_overlap = 0
    field_pages = 0
    committed_pages = 0
    committed_overlap = 0
    latencies: list[float] = []
    field_latencies: list[float] = []
    teacher_latencies: list[float] = []
    over_4s = 0
    fallbacks = 0
    full_searches = 0
    active_rows = 0
    confidence_rows: list[dict[str, Any]] = []
    for row in rows:
        session = str(row.get("session_hash") or "")
        if session:
            sessions.add(session)
        latency = row.get("latency_ms")
        if isinstance(latency, int | float) and not isinstance(latency, bool):
            latencies.append(max(0.0, float(latency)))
        if row.get("over_4s") is True or (
            isinstance(latency, int | float) and float(latency) > 4_000
        ):
            over_4s += 1
        is_fallback = str(row.get("status") or "") == "fallback"
        if is_fallback:
            fallbacks += 1
        if is_fallback or (
            "full_search_required" in row
            and row.get("full_search_required") is not False
        ):
            full_searches += 1
        if row.get("authority") == "field":
            active_rows += 1
        if not _is_stable_candidate_row(row):
            continue

        stable_traces += 1
        if session:
            stable_sessions.add(session)
        raw_field_ids = row.get("field_page_ids")
        raw_teacher_ids = row.get("teacher_page_ids")
        raw_committed_ids = row.get("committed_page_ids")
        field_ids = {
            str(value)
            for value in (raw_field_ids if isinstance(raw_field_ids, list) else [])
            if isinstance(value, str) and value
        }
        teacher_ids = {
            str(value)
            for value in (raw_teacher_ids if isinstance(raw_teacher_ids, list) else [])
            if isinstance(value, str) and value
        }
        committed_ids = {
            str(value)
            for value in (
                raw_committed_ids if isinstance(raw_committed_ids, list) else []
            )
            if isinstance(value, str) and value
        }
        confidence_row = _candidate_confidence_case(row)
        if confidence_row is not None:
            confidence_rows.append(confidence_row)
        if isinstance(raw_field_ids, list) and isinstance(raw_teacher_ids, list):
            teacher_overlap += len(teacher_ids & field_ids)
            field_pages += len(field_ids)
            if teacher_ids:
                coverage_evidence_traces += 1
                if session:
                    coverage_sessions.add(session)
                teacher_pages += len(teacher_ids)
        if (
            isinstance(raw_field_ids, list)
            and isinstance(raw_committed_ids, list)
            and committed_ids
        ):
            commit_evidence_traces += 1
            if session:
                commit_sessions.add(session)
            committed_pages += len(committed_ids)
            committed_overlap += len(committed_ids & field_ids)
        field_latency = row.get("field_latency_ms")
        teacher_latency = row.get("teacher_latency_ms")
        if (
            isinstance(field_latency, int | float)
            and not isinstance(field_latency, bool)
            and isinstance(teacher_latency, int | float)
            and not isinstance(teacher_latency, bool)
        ):
            paired_latency_traces += 1
            if session:
                paired_latency_sessions.add(session)
            field_latencies.append(max(0.0, float(field_latency)))
            teacher_latencies.append(max(0.0, float(teacher_latency)))
    confidence_manifest = manifest_sha256(
        sorted(
            str(row["case_sha256"])
            for row in confidence_rows
        )
    )
    confidence = {
        "method": "connected-cluster-bootstrap-percentile",
        "confidence": 0.95,
        "seed": 1729,
        "manifest_sha256": confidence_manifest,
        "samples": len(confidence_rows),
        "legacy_or_incomplete_samples": stable_traces - len(confidence_rows),
        "cases": confidence_rows,
        "teacher_coverage": cluster_rate_wilson_interval(
            confidence_rows,
            value_key="teacher_coverage",
            success_threshold=MIN_TEACHER_COVERAGE,
        ),
        "field_precision": cluster_rate_wilson_interval(
            confidence_rows,
            value_key="field_precision",
            success_threshold=MIN_PROCESSOR_USED_PRECISION,
        ),
        "commit_coverage": cluster_rate_wilson_interval(
            confidence_rows,
            value_key="commit_coverage",
            success_threshold=MIN_TEACHER_COVERAGE,
        ),
    }
    confidence["clusters"] = min(
        int(bound.get("clusters") or 0)
        for bound in (
            confidence["teacher_coverage"],
            confidence["field_precision"],
            confidence["commit_coverage"],
        )
    )
    return {
        "traces": len(rows),
        "sessions": len(sessions),
        "stable_traces": stable_traces,
        "stable_sessions": len(stable_sessions),
        "coverage_evidence_traces": coverage_evidence_traces,
        "coverage_evidence_sessions": len(coverage_sessions),
        "commit_evidence_traces": commit_evidence_traces,
        "commit_evidence_sessions": len(commit_sessions),
        "paired_latency_traces": paired_latency_traces,
        "paired_latency_sessions": len(paired_latency_sessions),
        "teacher_pages": teacher_pages,
        "field_pages": field_pages,
        "field_teacher_overlap": teacher_overlap,
        "field_precision_against_teacher": round(
            teacher_overlap / field_pages if field_pages else 0.0,
            6,
        ),
        "teacher_top30_coverage": round(
            teacher_overlap / teacher_pages if teacher_pages else 0.0,
            6,
        ),
        "teacher_committed_pages": committed_pages,
        "teacher_commit_coverage": round(
            committed_overlap / committed_pages if committed_pages else 0.0,
            6,
        ),
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "field_latency_ms": {
            "p50": _percentile(field_latencies, 0.50),
            "p95": _percentile(field_latencies, 0.95),
            "max": round(max(field_latencies), 3) if field_latencies else None,
        },
        "teacher_latency_ms": {
            "p50": _percentile(teacher_latencies, 0.50),
            "p95": _percentile(teacher_latencies, 0.95),
            "max": round(max(teacher_latencies), 3) if teacher_latencies else None,
        },
        "p95_improvement_ms": _p95_improvement(teacher_latencies, field_latencies),
        "over_4s": over_4s,
        "fallbacks": fallbacks,
        "fallback_rate": round(fallbacks / len(rows) if rows else 0.0, 6),
        "full_searches": full_searches,
        "full_search_rate": round(full_searches / len(rows) if rows else 0.0, 6),
        "active_traces": active_rows,
        "confidence": confidence,
    }


def split_integrity(labels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Prove identity, content, timestamp, and embargo isolation."""

    sessions: dict[str, set[str]] = {}
    queries: dict[str, set[str]] = {}
    pages: dict[str, set[str]] = {}
    page_uids: dict[str, set[str]] = {}
    contents: dict[str, set[str]] = {}
    timestamp_leakage = 0
    embargo_leakage = 0
    for row in labels:
        split = str(row.get("split") or "")
        session = str(row.get("session_hash") or "")
        query = str(row.get("query_sha256") or "")
        if session and split:
            sessions.setdefault(session, set()).add(split)
        if query and split:
            queries.setdefault(query, set()).add(split)
        page_uid = str(row.get("page_uid") or "")
        page_id = str(row.get("page_id") or "")
        content = str(row.get("content_sha256") or "")
        if page_id and split:
            pages.setdefault(page_id, set()).add(split)
        if page_uid and split:
            page_uids.setdefault(page_uid, set()).add(split)
        if content and split:
            contents.setdefault(content, set()).add(split)
        if split in {"train", "holdout", "locked-test"} and not str(
            row.get("observed_at") or ""
        ).endswith("Z"):
            timestamp_leakage += 1
        if row.get("split_diagnostic") == "boundary_embargo" and split != "embargo":
            embargo_leakage += 1
    session_leaks = sum(len(values) > 1 for values in sessions.values())
    query_leaks = sum(len(values) > 1 for values in queries.values())
    page_leaks = sum(len(values & {"train", "holdout", "locked-test"}) > 1 for values in pages.values())
    page_uid_leaks = sum(
        len(values & {"train", "holdout", "locked-test"}) > 1
        for values in page_uids.values()
    )
    content_leaks = sum(
        len(values & {"train", "holdout", "locked-test"}) > 1
        for values in contents.values()
    )
    return {
        "sessions": len(sessions),
        "queries": len(queries),
        "session_leakage": session_leaks,
        "query_leakage": query_leaks,
        "pages": len(pages),
        "page_uids": len(page_uids),
        "contents": len(contents),
        "page_leakage": page_leaks,
        "page_uid_leakage": page_uid_leaks,
        "content_leakage": content_leaks,
        "timestamp_leakage": timestamp_leakage,
        "embargo_leakage": embargo_leakage,
        "passed": all(
            value == 0
            for value in (
                session_leaks,
                query_leaks,
                page_leaks,
                page_uid_leaks,
                content_leaks,
                timestamp_leakage,
                embargo_leakage,
            )
        ),
    }


def compiler_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Measure exact fast-path precision separately from traffic coverage."""

    exact = [row for row in rows if row.get("compiler_status") == "exact"]
    predicted = sum(len(set(row.get("compiler_page_ids", []))) for row in exact)
    overlap = sum(
        len(
            set(row.get("compiler_page_ids", [])) & set(row.get("teacher_page_ids", []))
        )
        for row in exact
    )
    commit_overlap = sum(
        len(
            set(row.get("compiler_page_ids", []))
            & set(row.get("committed_page_ids", []))
        )
        for row in exact
    )
    return {
        "traces": len(rows),
        "exact_traces": len(exact),
        "coverage": round(len(exact) / len(rows) if rows else 0.0, 6),
        "predicted_pages": predicted,
        "teacher_overlap": overlap,
        "commit_overlap": commit_overlap,
        "precision": round(overlap / predicted if predicted else 0.0, 6),
        "authority_eligible": bool(predicted and overlap / predicted >= 0.99),
    }


def retrieval_locked_e2e_status(path: Path) -> dict[str, Any]:
    """Recompute the case-level, reviewed manual-94 retrieval gate."""

    from chronovisor.recall.recall_runtime import page_uid_for_id
    from chronovisor.search import search_eval

    payload = _read_json(path)
    seal = payload.get("snapshot_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "snapshot_sha256"}
    manifest = payload.get("manifest")
    entries = manifest.get("entries") if isinstance(manifest, Mapping) else None
    cases = payload.get("cases")
    entries_by_sha = {
        str(entry.get("entry_sha256") or ""): entry
        for entry in entries
        if isinstance(entry, Mapping)
    } if isinstance(entries, list) else {}
    cases_valid = isinstance(cases, list) and len(cases) == 94
    normalized_cases: list[Mapping[str, Any]] = []
    seen_queries: set[str] = set()
    seen_entries: set[str] = set()
    if cases_valid:
        for value in cases:
            case = value if isinstance(value, Mapping) else {}
            entry = entries_by_sha.get(str(case.get("manifest_entry_sha256") or ""))
            ranked = case.get("ranked_page_bindings")
            selected = case.get("selected_evidence")
            committed = case.get("committed_page_ids")
            certificates = case.get("certificate_ids")
            query_sha = str(case.get("query_sha256") or "")
            expected_review = (
                hashlib.sha256(
                    json.dumps(
                        {
                            "kind": "manual94-human-review-v1",
                            "entry_sha256": entry.get("entry_sha256"),
                            "ref": entry.get("ref"),
                            "source": entry.get("source"),
                            "reviewed": entry.get("reviewed"),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                if isinstance(entry, Mapping)
                else ""
            )
            entry_unsigned = (
                {key: item for key, item in entry.items() if key != "entry_sha256"}
                if isinstance(entry, Mapping)
                else {}
            )
            live_bindings = True
            if isinstance(ranked, list):
                for rank, binding in enumerate(ranked, start=1):
                    page_id = str(binding.get("page_id") or "") if isinstance(binding, Mapping) else ""
                    page = find_page(page_id)
                    try:
                        live_sha = hashlib.sha256(page.read_bytes()).hexdigest() if page else ""
                    except OSError:
                        live_sha = ""
                    if (
                        not isinstance(binding, Mapping)
                        or binding.get("rank") != rank
                        or binding.get("page_uid") != page_uid_for_id(page_id)
                        or not binding.get("page_uid")
                        or binding.get("content_sha256") != live_sha
                    ):
                        live_bindings = False
                        break
            else:
                live_bindings = False
            selected_rows = [item for item in selected if isinstance(item, Mapping)] if isinstance(selected, list) else []
            selected_certificates = [str(item.get("certificate_id") or "") for item in selected_rows]
            selected_pages = [str(item.get("page_id") or "") for item in selected_rows]
            ranked_pages = {
                str(item.get("page_id") or "")
                for item in ranked
                if isinstance(item, Mapping)
            } if isinstance(ranked, list) else set()
            expected_pages = list(entry.get("expected_pages") or []) if isinstance(entry, Mapping) else []
            expected_bad_pages = list(
                dict.fromkeys(
                    [
                        *list(entry.get("negative_pages") or []),
                        *list(entry.get("stale_pages") or []),
                    ]
                )
            ) if isinstance(entry, Mapping) else []
            expected_commit = hashlib.sha256(
                json.dumps(
                    {
                        "query_sha256": query_sha,
                        "committed_page_ids": committed,
                        "certificate_ids": certificates,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            case_unsigned = {key: item for key, item in case.items() if key != "case_sha256"}
            if (
                not isinstance(entry, Mapping)
                or entry.get("entry_sha256")
                != hashlib.sha256(
                    json.dumps(
                        entry_unsigned,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                or entry.get("reviewed") is not True
                or entry.get("query_sha256") != query_sha
                or case.get("expected_pages") != expected_pages
                or case.get("bad_pages") != expected_bad_pages
                or str(entry.get("entry_sha256") or "") in seen_entries
                or query_sha in seen_queries
                or case.get("reviewed") is not True
                or case.get("review_receipt_sha256") != expected_review
                or case.get("case_sha256")
                != hashlib.sha256(
                    json.dumps(
                        case_unsigned,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                or not live_bindings
                or not isinstance(committed, list)
                or not committed
                or committed != selected_pages
                or len(selected_rows) != len(selected or [])
                or not set(selected_pages) <= ranked_pages
                or not isinstance(certificates, list)
                or not certificates
                or certificates != selected_certificates
                or any(not value for value in selected_pages + selected_certificates)
                or case.get("commit_ids") != [expected_commit]
                or isinstance(case.get("latency_ms"), bool)
                or not isinstance(case.get("latency_ms"), int | float)
            ):
                cases_valid = False
                break
            seen_queries.add(query_sha)
            seen_entries.add(str(entry["entry_sha256"]))
            normalized_cases.append(case)
    cases_valid = bool(
        cases_valid
        and isinstance(entries, list)
        and len(entries) == 94
        and len(entries_by_sha) == 94
        and seen_entries == set(entries_by_sha)
    )
    positives = [case for case in normalized_cases if case.get("expected_pages")]
    negative_cases = [case for case in normalized_cases if case.get("bad_pages")]
    recall_at_5 = (
        sum(
            bool(
                set(case.get("expected_pages", []))
                & {
                    str(binding.get("page_id") or "")
                    for binding in case.get("ranked_page_bindings", [])[:5]
                    if isinstance(binding, Mapping)
                }
            )
            for case in positives
        )
        / len(positives)
        if positives
        else 0.0
    )
    negative_rate = (
        sum(
            bool(
                set(case.get("bad_pages", []))
                & {
                    str(binding.get("page_id") or "")
                    for binding in case.get("ranked_page_bindings", [])[:20]
                    if isinstance(binding, Mapping)
                }
            )
            for case in negative_cases
        )
        / len(negative_cases)
        if negative_cases
        else 0.0
    )
    committed_pairs = [
        (case, str(page_id))
        for case in normalized_cases
        for page_id in case.get("committed_page_ids", [])
    ]
    committed_tp = sum(
        page in set(case.get("expected_pages", [])) for case, page in committed_pairs
    )
    # ``bad_pages`` is an explanatory annotation, not an exhaustive negative
    # universe. Every selected page outside the exact expected labels is a FP.
    committed_fp = sum(
        page not in set(case.get("expected_pages", []))
        for case, page in committed_pairs
    )
    precision = committed_tp / (committed_tp + committed_fp) if committed_tp + committed_fp else None
    related_recall = (
        sum(bool(set(case.get("expected_pages", [])) & set(case.get("committed_page_ids", []))) for case in positives)
        / len(positives)
        if positives
        else 0.0
    )
    evidence_precision: dict[str, float | None] = {}
    for kind in ("rich", "pointer"):
        selected_pairs = [
            (case, str(item.get("page_id") or ""))
            for case in normalized_cases
            for item in case.get("selected_evidence", [])
            if isinstance(item, Mapping) and item.get("evidence_kind") == kind
        ]
        tp = sum(page in set(case.get("expected_pages", [])) for case, page in selected_pairs)
        fp = sum(
            page not in set(case.get("expected_pages", []))
            for case, page in selected_pairs
        )
        evidence_precision[kind] = tp / (tp + fp) if tp + fp else None
    latency_max = max((float(case["latency_ms"]) for case in normalized_cases), default=math.inf)
    expected_metrics = {
        "recall_at_5": recall_at_5,
        "negative_hit_rate_at_20": negative_rate,
        "latency_ms": {"max": latency_max},
        "processor": {
            "precision": precision,
            "related_recall": related_recall,
            "evidence_kind": {
                "rich": {"precision": evidence_precision["rich"]},
                "pointer": {"precision": evidence_precision["pointer"]},
            },
        },
    }
    expected_gates = {
        "sealed_manual_94": cases_valid and len(entries_by_sha) == 94,
        "rerank_recall_at_5": recall_at_5 >= 0.535,
        "negative_hit_rate": negative_rate <= 0.20,
        "processor_precision": precision is not None and precision >= 0.90,
        "processor_related_recall": related_recall >= 0.535,
        "rich_precision": evidence_precision["rich"] is not None and evidence_precision["rich"] >= 0.90,
        "pointer_precision": evidence_precision["pointer"] is not None and evidence_precision["pointer"] >= 0.90,
        "latency": latency_max <= 4_000,
    }
    environment = payload.get("environment_epoch")
    environment_valid = (
        isinstance(environment, Mapping)
        and dict(environment) == builtin_field_environment_identity()
        and payload.get("environment_epoch_sha256")
        == hashlib.sha256(
            json.dumps(
                dict(environment),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    canonical_manifest_path = search_eval.MANUAL_MANIFEST_FILE
    canonical_manifest = _read_json(canonical_manifest_path)
    try:
        canonical_manifest_file_sha256 = hashlib.sha256(
            canonical_manifest_path.read_bytes()
        ).hexdigest()
    except OSError:
        canonical_manifest_file_sha256 = ""
    canonical_unsigned = {
        key: value
        for key, value in canonical_manifest.items()
        if key != "manifest_sha256"
    }
    canonical_entries = canonical_manifest.get("entries")
    canonical_entry_rows = (
        canonical_entries if isinstance(canonical_entries, list) else []
    )
    canonical_review_ledger = canonical_manifest.get("review_ledger")
    try:
        review_ledger_path = Path(
            str(canonical_review_ledger.get("path") or "")
        ).expanduser().resolve(strict=False)
        live_review_ledger_sha256 = hashlib.sha256(
            review_ledger_path.read_bytes()
        ).hexdigest()
    except (AttributeError, OSError):
        live_review_ledger_sha256 = ""
    expected_review_ledger_head = hashlib.sha256(
        json.dumps(
            {
                "file_sha256": live_review_ledger_sha256,
                "entry_sha256": [
                    str(entry.get("entry_sha256") or "")
                    for entry in canonical_entry_rows
                    if isinstance(entry, Mapping)
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    canonical_manifest_valid = bool(
        canonical_manifest.get("schema_version") == 2
        and canonical_manifest.get("examples") == 94
        and isinstance(canonical_entries, list)
        and len(canonical_entries) == 94
        and len(
            {
                str(entry.get("entry_sha256") or "")
                for entry in canonical_entry_rows
                if isinstance(entry, Mapping)
            }
        )
        == 94
        and all(
            isinstance(entry, Mapping) and entry.get("reviewed") is True
            for entry in canonical_entry_rows
        )
        and canonical_manifest.get("manifest_sha256")
        == hashlib.sha256(
            json.dumps(
                canonical_unsigned,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        and isinstance(canonical_review_ledger, Mapping)
        and len(live_review_ledger_sha256) == 64
        and canonical_review_ledger.get("file_sha256")
        == live_review_ledger_sha256
        and canonical_review_ledger.get("head_sha256")
        == expected_review_ledger_head
        and manifest == canonical_manifest
        and payload.get("frozen_manifest_sha256")
        == canonical_manifest_file_sha256
        and payload.get("frozen_at") == canonical_manifest.get("frozen_at")
    )
    try:
        frozen_at = datetime.fromisoformat(
            str(payload.get("frozen_at") or "").replace("Z", "+00:00")
        )
        generated_at = datetime.fromisoformat(
            str(payload.get("generated_at") or "").replace("Z", "+00:00")
        )
        preregistered = bool(
            frozen_at.tzinfo is not None
            and generated_at.tzinfo is not None
            and frozen_at.utcoffset() == timedelta(0)
            and generated_at.utcoffset() == timedelta(0)
            and frozen_at < generated_at
        )
    except (TypeError, ValueError):
        preregistered = False
    valid = bool(
        payload.get("schema_version") == 2
        and payload.get("status") == "passed"
        and payload.get("examples") == 94
        and payload.get("metrics") == expected_metrics
        and payload.get("gates") == expected_gates
        and all(value is True for value in expected_gates.values())
        and isinstance(manifest, Mapping)
        and manifest.get("manifest_sha256") == payload.get("manifest_sha256")
        and manifest.get("manifest_sha256")
        == hashlib.sha256(
            json.dumps(
                {key: value for key, value in manifest.items() if key != "manifest_sha256"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        and environment_valid
        and canonical_manifest_valid
        and preregistered
        and isinstance(seal, str)
        and seal == hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    return {
        "passed": valid,
        "reason": "verified" if valid else "retrieval_locked_e2e_invalid",
        "manifest_sha256": str(payload.get("manifest_sha256") or ""),
        "examples": payload.get("examples") if isinstance(payload.get("examples"), int) else 0,
        "environment_epoch_sha256": str(payload.get("environment_epoch_sha256") or ""),
        "artifact_sha256": str(seal or ""),
    }


def locked_e2e_status(path: Path) -> dict[str, Any]:
    """Compatibility alias for the manual-94 retrieval gate."""

    return retrieval_locked_e2e_status(path)


def processor_used_metrics(
    recall_rows: Sequence[Mapping[str, Any]],
    pull_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure whether shadow Processor retained pages explicitly used later."""

    joined = join_used_recall_episodes(list(recall_rows), list(pull_rows))
    observed: list[tuple[str, set[str], set[str]]] = []
    confidence_rows: list[dict[str, Any]] = []
    all_sessions: set[str] = set()
    for episode in joined["episodes"]:
        recall = episode.get("recall")
        if not isinstance(recall, Mapping):
            continue
        features = recall.get("evidence_features")
        shadow = (
            features.get("processor_shadow") if isinstance(features, Mapping) else None
        )
        if not isinstance(shadow, Mapping):
            continue
        selected = {
            str(value)
            for value in shadow.get("committed_page_ids", [])
            if isinstance(value, str) and value
        }
        used = {
            str(value)
            for value in episode.get("page_ids", [])
            if isinstance(value, str) and value
        }
        if not used:
            continue
        session = str(episode.get("session_id") or "")
        if session:
            all_sessions.add(session)
        observed.append((session, selected, used))
        context_hashes = {
            str(item.get("page_id") or ""): str(
                item.get("content_sha256") or item.get("page_sha256") or ""
            )
            for item in recall.get("context_items", [])
            if isinstance(item, Mapping)
            and str(item.get("page_id") or "")
            and len(str(item.get("content_sha256") or item.get("page_sha256") or ""))
            == 64
        }
        context_uids = {
            str(item.get("page_id") or ""): str(item.get("page_uid") or "")
            for item in recall.get("context_items", [])
            if isinstance(item, Mapping)
            and str(item.get("page_id") or "")
            and str(item.get("page_uid") or "")
        }
        query_sha = str(
            recall.get("prompt_sha256")
            or recall.get("prompt_hash")
            or recall.get("query_sha256")
            or ""
        )
        bound_pages = selected | used
        if (
            session
            and query_sha
            and bound_pages
            and bound_pages <= set(context_hashes)
            and bound_pages <= set(context_uids)
        ):
            confidence_row = {
                    "session_hash": hashlib.sha256(session.encode()).hexdigest()[:16],
                    "query_sha256": query_sha,
                    "cluster_nodes": list(
                        dict.fromkeys(
                            [
                                f"session:{hashlib.sha256(session.encode()).hexdigest()[:16]}",
                                f"query:{query_sha}",
                            ]
                            + [f"page:{page}" for page in sorted(bound_pages)]
                            + [f"uid:{context_uids[page]}" for page in sorted(bound_pages)]
                            + [
                                f"content:{context_hashes[page]}"
                                for page in sorted(bound_pages)
                            ]
                        )
                    ),
                    "coverage": len(used & selected) / len(used),
                    "precision": len(used & selected) / len(selected)
                    if selected
                    else 0.0,
                }
            confidence_row["case_sha256"] = hashlib.sha256(
                json.dumps(
                    confidence_row,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            confidence_rows.append(confidence_row)

    quality_window = observed[-QUALITY_WINDOW:]
    used_pages = 0
    covered_pages = 0
    selected_pages = 0
    selected_used_pages = 0
    for _session, selected, used in quality_window:
        used_pages += len(used)
        covered_pages += len(used & selected)
        selected_pages += len(selected)
        selected_used_pages += len(used & selected)
    confidence = {
        "method": "connected-cluster-bootstrap-percentile",
        "confidence": 0.95,
        "seed": 1729,
        "manifest_sha256": manifest_sha256(
            sorted(
                str(row["case_sha256"])
                for row in confidence_rows
            )
        ),
        "samples": len(confidence_rows),
        "legacy_or_incomplete_samples": len(observed) - len(confidence_rows),
        "cases": confidence_rows,
        "coverage": cluster_rate_wilson_interval(
            confidence_rows,
            value_key="coverage",
            success_threshold=MIN_PROCESSOR_USED_COVERAGE,
        ),
        "precision": cluster_rate_wilson_interval(
            confidence_rows,
            value_key="precision",
            success_threshold=MIN_PROCESSOR_USED_PRECISION,
        ),
    }
    confidence["clusters"] = min(
        int(confidence[key].get("clusters") or 0) for key in ("coverage", "precision")
    )
    return {
        "episodes": len(observed),
        "sessions": len(all_sessions),
        "quality_window_episodes": len(quality_window),
        "used_pages": used_pages,
        "covered_pages": covered_pages,
        "selected_pages": selected_pages,
        "selected_used_pages": selected_used_pages,
        "used_page_coverage": round(
            covered_pages / used_pages if used_pages else 0.0,
            6,
        ),
        "used_precision_proxy": round(
            selected_used_pages / selected_pages if selected_pages else 0.0,
            6,
        ),
        "joined_used": int(joined["accepted"]),
        "rejected_used": int(joined["rejected"]),
        "confidence": confidence,
    }


def _promotion_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    candidate = metrics["candidate"]
    used = metrics["processor_used"]
    learning = metrics["learning"]
    return {
        "schema_version": 3,
        "status": "passed",
        "metrics": {
            "stable_traces": int(candidate["quality_window_stable_traces"]),
            "stable_sessions": int(candidate["quality_window_stable_sessions"]),
            "coverage_evidence_traces": int(candidate["coverage_evidence_traces"]),
            "coverage_evidence_sessions": int(candidate["coverage_evidence_sessions"]),
            "commit_evidence_traces": int(candidate["commit_evidence_traces"]),
            "commit_evidence_sessions": int(candidate["commit_evidence_sessions"]),
            "paired_latency_traces": int(candidate["paired_latency_traces"]),
            "paired_latency_sessions": int(candidate["paired_latency_sessions"]),
            "validated_confidence_traces": int(
                candidate["validated_confidence_traces"]
            ),
            "incomplete_quality_traces": int(
                candidate["incomplete_quality_traces"]
            ),
            "teacher_commit_coverage": float(candidate["teacher_commit_coverage"]),
            "precision_delta_points": float(learning["precision_delta_points"]),
            "recall_delta_points": float(learning["recall_delta_points"]),
            "over_4s": int(candidate["over_4s"]),
            "fallback_rate": float(candidate["fallback_rate"]),
            "full_search_rate": float(candidate["full_search_rate"]),
            "processor_used_page_coverage": float(used["used_page_coverage"]),
            "processor_used_precision_proxy": float(used["used_precision_proxy"]),
        },
        "confidence_evidence": {
            "candidate": candidate["confidence"],
            "processor_used": used["confidence"],
            "answer_reward": learning["confidence_bounds"]["answer_reward"],
        },
        "answer_evaluation": learning["answer_evaluation"],
        "locked_answer_evaluation": learning["locked_answer_evaluation"],
        "retrieval_locked_e2e": learning["retrieval_locked_e2e"],
        "answer_artifact_set": learning["answer_artifact_set"],
    }


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {**payload, "snapshot_sha256": hashlib.sha256(encoded).hexdigest()}


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _append_only_prefix_identity(path: Path) -> dict[str, Any]:
    """Seal the exact byte/line prefix while permitting later complete appends."""

    try:
        data = path.read_bytes()
    except OSError:
        data = b""
    lines = data.splitlines()
    return {
        "path": str(path.resolve(strict=False)),
        "byte_length": len(data),
        "line_count": len(lines),
        "prefix_sha256": hashlib.sha256(data).hexdigest(),
        "head_sha256": hashlib.sha256(lines[-1]).hexdigest()
        if lines
        else "0" * 64,
    }


def _append_only_prefix_rows(
    path: Path, identity: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    """Read the sealed JSONL ancestor without incorporating later appends."""

    byte_length = identity.get("byte_length")
    line_count = identity.get("line_count")
    if (
        not isinstance(byte_length, int)
        or isinstance(byte_length, bool)
        or byte_length < 0
        or not isinstance(line_count, int)
        or isinstance(line_count, bool)
        or line_count < 0
    ):
        return [], "source_prefix_identity_invalid"
    try:
        current = path.read_bytes()
    except OSError:
        return [], "source_prefix_unreadable"
    if len(current) < byte_length:
        return [], "source_prefix_truncated"
    prefix = current[:byte_length]
    lines = prefix.splitlines()
    head_sha = hashlib.sha256(lines[-1]).hexdigest() if lines else "0" * 64
    if (
        hashlib.sha256(prefix).hexdigest() != identity.get("prefix_sha256")
        or len(lines) != line_count
        or head_sha != identity.get("head_sha256")
        or (len(current) > byte_length and prefix and not prefix.endswith(b"\n"))
    ):
        return [], "source_prefix_mismatch"
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return [], "source_prefix_jsonl_invalid"
        if not isinstance(row, dict):
            return [], "source_prefix_jsonl_invalid"
        rows.append(row)
    return rows, ""


def _failed_promotion(reason: str, metrics: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": 3,
        "status": "held",
        "reason": reason,
        "metrics": {
            "stable_traces": int(metrics["candidate"]["quality_window_stable_traces"]),
            "stable_sessions": int(
                metrics["candidate"]["quality_window_stable_sessions"]
            ),
            "coverage_evidence_traces": int(
                metrics["candidate"]["coverage_evidence_traces"]
            ),
            "coverage_evidence_sessions": int(
                metrics["candidate"]["coverage_evidence_sessions"]
            ),
            "commit_evidence_traces": int(
                metrics["candidate"]["commit_evidence_traces"]
            ),
            "commit_evidence_sessions": int(
                metrics["candidate"]["commit_evidence_sessions"]
            ),
            "paired_latency_traces": int(metrics["candidate"]["paired_latency_traces"]),
            "paired_latency_sessions": int(
                metrics["candidate"]["paired_latency_sessions"]
            ),
            "teacher_commit_coverage": float(
                metrics["candidate"]["teacher_commit_coverage"]
            ),
            "processor_used_page_coverage": float(
                metrics["processor_used"]["used_page_coverage"]
            ),
            "processor_used_precision_proxy": float(
                metrics["processor_used"]["used_precision_proxy"]
            ),
            "over_4s": int(metrics["candidate"]["over_4s"]),
            "fallback_rate": float(metrics["candidate"]["fallback_rate"]),
        },
        "confidence_evidence": {
            "candidate": metrics["candidate"]["confidence"],
            "processor_used": metrics["processor_used"]["confidence"],
            "answer_reward": metrics["learning"]["confidence_bounds"][
                "answer_reward"
            ],
        },
        "answer_evaluation": metrics["learning"]["answer_evaluation"],
        "locked_answer_evaluation": metrics["learning"]["locked_answer_evaluation"],
        "retrieval_locked_e2e": metrics["learning"]["retrieval_locked_e2e"],
        "answer_artifact_set": metrics["learning"]["answer_artifact_set"],
    }
    return _seal(payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    os.chmod(path, 0o600)


def _append_history(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    with exclusive_text_file_lock(lock):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.chmod(path, 0o600)


def _advance_rollout(
    previous: Mapping[str, Any],
    *,
    authority_eligible: bool,
    candidate_trace_count: int,
) -> tuple[str, int, int]:
    if not authority_eligible:
        return "candidate", 100, candidate_trace_count
    previous_mode = str(previous.get("effective_mode") or "candidate")
    previous_percent = int(previous.get("canary_percent") or 0)
    if previous_mode != "active" or previous_percent not in CANARY_STEPS:
        return "active", CANARY_STEPS[0], candidate_trace_count
    if "stage_started_confidence_sample_count" not in previous:
        # Legacy counters included point-only traces. Never subtract unlike
        # units: preserve the current canary and rebase it.
        return "active", previous_percent, candidate_trace_count
    try:
        started_at = max(0, int(previous["stage_started_confidence_sample_count"]))
    except (TypeError, ValueError):
        return "active", previous_percent, candidate_trace_count
    new_samples = max(0, candidate_trace_count - started_at)
    if new_samples < CANARY_ADVANCE_SAMPLES:
        return "active", previous_percent, started_at
    index = CANARY_STEPS.index(previous_percent)
    next_percent = CANARY_STEPS[min(index + 1, len(CANARY_STEPS) - 1)]
    return "active", next_percent, candidate_trace_count


def _persist_growth_artifacts(
    *,
    state_file: Path,
    promotion_file: Path,
    history_file: Path,
    policy_history_file: Path,
    last_known_good_file: Path,
    payload: dict[str, Any],
    promotion: dict[str, Any],
    learning_decision: dict[str, Any],
    current_policy: dict[str, Any],
    integrity: dict[str, Any],
) -> None:
    metrics = payload["metrics"]
    extensions = payload["extensions"]
    _write_json(state_file, payload)
    _write_json(promotion_file, promotion)
    _append_history(
        history_file,
        {
            key: payload[key]
            for key in (
                "generated_at",
                "stage",
                "effective_mode",
                "canary_percent",
                "field_learning_allowed",
                "positive_learning_allowed",
                "policy_update_allowed",
                "authority_enabled",
                "gates",
            )
        }
        | {
            "labels": metrics["labels"],
            "candidate": metrics["candidate"],
            "processor_used": metrics["processor_used"],
            "learning": metrics["learning"],
            "compiler": payload["compiler"],
            "extensions": {
                "typed_graph": extensions["typed_graph"]["gates"],
                "rubric": extensions["rubric"]["gates"],
            },
        },
    )
    policy_record = append_policy_history(
        {
            "generated_at": payload["generated_at"],
            "status": str(learning_decision.get("status") or "held"),
            "reason": str(learning_decision.get("reason") or ""),
            "policy": learning_decision.get("policy", current_policy),
            "label_counts": metrics["labels"],
            "metrics": metrics["learning"],
            "split_integrity": integrity,
            "authority_enabled": payload["authority_enabled"],
            "extensions": {
                "typed_graph": extensions["typed_graph"]["gates"],
                "rubric": extensions["rubric"]["gates"],
            },
        },
        path=policy_history_file,
    )
    chain = verify_policy_history(policy_history_file)
    if payload["policy_update_allowed"] and chain.get("status") == "ok":
        write_last_known_good(
            {
                key: float(value)
                for key, value in learning_decision.get(
                    "policy", current_policy
                ).items()
            },
            evaluation={
                "candidate": metrics["candidate"],
                "processor_used": metrics["processor_used"],
                "split_integrity": integrity,
            },
            history_head_sha256=str(policy_record["record_sha256"]),
            path=last_known_good_file,
        )


def _candidate_growth_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build stable-quality and all-turn operational windows independently."""

    totals = candidate_metrics(rows)
    operational = candidate_metrics(rows[-QUALITY_WINDOW:])
    stable_rows = [row for row in rows if _is_stable_candidate_row(row)]
    quality = candidate_metrics(stable_rows[-QUALITY_WINDOW:])
    return {
        **quality,
        "traces": totals["traces"],
        "sessions": totals["sessions"],
        "stable_traces": totals["stable_traces"],
        "stable_sessions": totals["stable_sessions"],
        "validated_confidence_traces": totals["confidence"]["samples"],
        "incomplete_quality_traces": totals["confidence"][
            "legacy_or_incomplete_samples"
        ],
        "active_traces": totals["active_traces"],
        "quality_window_traces": operational["traces"],
        "quality_window_sessions": operational["sessions"],
        "quality_window_stable_traces": quality["stable_traces"],
        "quality_window_stable_sessions": quality["stable_sessions"],
        "latency_ms": operational["latency_ms"],
        "over_4s": operational["over_4s"],
        "fallbacks": operational["fallbacks"],
        "fallback_rate": operational["fallback_rate"],
        "full_searches": operational["full_searches"],
        "full_search_rate": operational["full_search_rate"],
    }


def run_growth_cycle(
    *,
    dry_run: bool = False,
    state_file: Path = GROWTH_STATE_FILE,
    history_file: Path = GROWTH_HISTORY_FILE,
    candidate_trace_file: Path = CANDIDATE_TRACE_FILE,
    promotion_file: Path = PROMOTION_ARTIFACT,
    policy_history_file: Path | None = None,
    last_known_good_file: Path | None = None,
    compiler_trace_file: Path | None = None,
    locked_e2e_file: Path | None = None,
    locked_answer_eval_file: Path | None = None,
    train_answer_eval_file: Path | None = None,
    answer_episode_file: Path = ANSWER_EPISODE_LEDGER,
    answer_review_ledger_file: Path = ANSWER_REVIEW_LEDGER,
    answer_execution_ledger_file: Path = ANSWER_EXECUTION_LEDGER,
    answer_adapter_registry: Path | Mapping[str, Any] = ANSWER_ADAPTER_REGISTRY,
    label_inputs: dict[str, Path] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Refresh supervision and advance a fail-closed autonomous rollout."""

    policy_history_file = (
        policy_history_file or state_file.parent / POLICY_HISTORY_FILE.name
    )
    last_known_good_file = (
        last_known_good_file or state_file.parent / LAST_KNOWN_GOOD_POLICY_FILE.name
    )
    compiler_trace_file = (
        compiler_trace_file
        or state_file.parent.parent / "recall-compiler" / COMPILER_TRACE_FILE.name
    )
    locked_e2e_file = locked_e2e_file or LOCKED_E2E_ARTIFACT
    locked_answer_eval_file = locked_answer_eval_file or LOCKED_ANSWER_EVAL_ARTIFACT
    train_answer_eval_file = train_answer_eval_file or TRAIN_ANSWER_EVAL_ARTIFACT
    inputs = label_inputs or default_label_ledger_inputs()
    if not dry_run and isinstance(inputs.get("feedback_file"), Path):
        from chronovisor.recall.recall_field import sync_reviewed_negative_feedback

        sync_reviewed_negative_feedback(inputs["feedback_file"])
    labels = (
        build_label_ledger(**inputs) if dry_run else materialize_label_ledger(**inputs)
    )
    recall_rows = _read_jsonl(inputs["recall_log_file"])
    pull_rows = _read_jsonl(inputs["pull_log_file"])
    candidate_rows, candidate_chain_error = _validated_candidate_trace_rows(
        candidate_trace_file
    )
    candidate = _candidate_growth_metrics(candidate_rows)
    candidate["trace_chain_valid"] = not candidate_chain_error
    candidate["trace_chain_head_sha256"] = (
        str(candidate_rows[-1].get("record_sha256") or "")
        if candidate_rows
        else "0" * 64
    )
    candidate["cumulative_eligible_trace_count"] = (
        int(candidate_rows[-1].get("cumulative_eligible_trace_count") or 0)
        if candidate_rows
        else 0
    )
    processor = processor_used_metrics(recall_rows, pull_rows)
    counts = dict(labels["counts"])
    label_gate = bool(labels["gates"]["field_learning_allowed"])
    subject_gate_value = labels.get("gates")
    subject_gates = subject_gate_value if isinstance(subject_gate_value, dict) else {}
    typed_graph_eval = _read_json(
        state_file.parent.parent / "typed-graph" / "evaluation.json"
    )
    rubric_status = _read_json(
        state_file.parent.parent / "recall-rubric" / "status.json"
    )
    rubric_state = rubric_status if isinstance(rubric_status, dict) else {}
    graph_eval = typed_graph_eval if isinstance(typed_graph_eval, dict) else {}
    comparison_value = graph_eval.get("comparison")
    comparison = comparison_value if isinstance(comparison_value, dict) else {}
    metric_value = comparison.get("metrics")
    metrics = metric_value if isinstance(metric_value, dict) else {}
    winner_metrics_value = metrics.get(str(graph_eval.get("winner") or "current"))
    winner_metrics = (
        winner_metrics_value if isinstance(winner_metrics_value, dict) else {}
    )
    extension_gates = {
        "relation_learning": subject_gates.get("relation_learning_allowed") is True,
        "entity_learning": subject_gates.get("entity_learning_allowed") is True,
        "rubric_learning": subject_gates.get("rubric_learning_allowed") is True,
        "rubric_adopted": rubric_state.get("status") == "adopted",
        "four_arm_evaluation": graph_eval.get("status") == "passed",
        "external_calls_zero": winner_metrics.get("external_model_calls", 0) == 0,
    }
    integrity = split_integrity(labels["labels"])
    compiler = compiler_metrics(_read_jsonl(compiler_trace_file))
    retrieval_locked_e2e = retrieval_locked_e2e_status(locked_e2e_file)
    locked_answer_e2e = validate_locked_answer_artifact(
        locked_answer_eval_file,
        episode_file=answer_episode_file,
        review_ledger_file=answer_review_ledger_file,
        execution_ledger_file=answer_execution_ledger_file,
        adapter_registry=answer_adapter_registry,
    )
    train_answer_e2e = validate_answer_outcome_artifact(
        train_answer_eval_file,
        required_split="train",
        episode_file=answer_episode_file,
        review_ledger_file=answer_review_ledger_file,
        execution_ledger_file=answer_execution_ledger_file,
        adapter_registry=answer_adapter_registry,
    )
    answer_artifact_set = validate_answer_artifact_set(
        train=train_answer_eval_file,
        locked=locked_answer_eval_file,
        episode_file=answer_episode_file,
        review_ledger_file=answer_review_ledger_file,
        execution_ledger_file=answer_execution_ledger_file,
        adapter_registry=answer_adapter_registry,
    )
    last_known_good = load_last_known_good(last_known_good_file)
    current_policy = {
        "spread_gain": 0.35,
        "global_inhibition": 0.08,
        "turn_decay": 0.82,
        **(
            last_known_good.get("policy", {})
            if isinstance(last_known_good.get("policy"), dict)
            else {}
        ),
    }
    proposed_policy = {
        "spread_gain": current_policy["spread_gain"]
        + (0.01 if candidate["teacher_commit_coverage"] >= 0.99 else -0.02),
        "global_inhibition": current_policy["global_inhibition"]
        + (0.01 if processor["used_precision_proxy"] < 0.90 else -0.005),
        "turn_decay": current_policy["turn_decay"]
        + (0.01 if candidate["full_search_rate"] > 0.25 else 0.0),
    }
    answer_point = train_answer_e2e.get("point")
    learning_metrics = {
        "session_leakage": float(integrity["session_leakage"]),
        "query_leakage": float(integrity["query_leakage"]),
        "page_leakage": float(integrity["page_leakage"]),
        "content_leakage": float(integrity["content_leakage"]),
        "timestamp_leakage": float(integrity["timestamp_leakage"]),
        "embargo_leakage": float(integrity["embargo_leakage"]),
        "precision_delta_points": (
            round(float(answer_point) * 100.0, 6)
            if isinstance(answer_point, int | float)
            else -100.0
        ),
        "recall_delta_points": (
            round(float(answer_point) * 100.0, 6)
            if isinstance(answer_point, int | float)
            else -100.0
        ),
    }
    answer_confidence = {
        "valid": train_answer_e2e.get("passed") is True,
        "method": str(train_answer_e2e.get("method") or ""),
        "confidence": train_answer_e2e.get("confidence"),
        "seed": train_answer_e2e.get("seed"),
        "manifest_sha256": str(train_answer_e2e.get("manifest_sha256") or ""),
        "samples": int(train_answer_e2e.get("samples") or 0),
        "clusters": int(train_answer_e2e.get("distinct_clusters") or 0),
        "point": train_answer_e2e.get("point"),
        "lower": train_answer_e2e.get("lower"),
        "upper": train_answer_e2e.get("upper"),
        "point_floor": 0.02,
        "lower_floor": 0.0,
    }
    learning_metrics["answer_evaluation"] = train_answer_e2e
    learning_metrics["retrieval_locked_e2e"] = retrieval_locked_e2e
    learning_metrics["locked_answer_evaluation"] = locked_answer_e2e
    learning_metrics["answer_artifact_set"] = answer_artifact_set
    learning_metrics["confidence_bounds"] = {"answer_reward": answer_confidence}
    learning_decision = decide_learning_update(
        current={key: float(value) for key, value in current_policy.items()},
        proposed={key: float(value) for key, value in proposed_policy.items()},
        label_counts=counts,
        metrics=learning_metrics,
        answer_evaluation=train_answer_e2e,
        confidence_bounds={"answer_reward": answer_confidence},
    )
    candidate_confidence = candidate["confidence"]
    processor_confidence = processor["confidence"]
    candidate_teacher_bound = candidate_confidence["teacher_coverage"]
    candidate_precision_bound = candidate_confidence["field_precision"]
    candidate_commit_bound = candidate_confidence["commit_coverage"]
    processor_coverage_bound = processor_confidence["coverage"]
    processor_precision_bound = processor_confidence["precision"]
    previous = load_growth_state(state_file)
    previous_candidate = (
        previous.get("metrics", {}).get("candidate", {})
        if isinstance(previous.get("metrics"), Mapping)
        else {}
    )
    previous_count = int(
        previous_candidate.get("cumulative_eligible_trace_count") or 0
    ) if isinstance(previous_candidate, Mapping) else 0
    previous_head = str(
        previous_candidate.get("trace_chain_head_sha256") or ""
    ) if isinstance(previous_candidate, Mapping) else ""
    candidate_trace_monotonic = bool(
        candidate["cumulative_eligible_trace_count"] >= previous_count
        and (
            not previous_head
            or any(
                row.get("record_sha256") == previous_head
                for row in candidate_rows
            )
        )
    )
    gates = {
        "candidate_trace_chain": candidate["trace_chain_valid"],
        "candidate_trace_monotonic": candidate_trace_monotonic,
        "labels": label_gate,
        "split_integrity": integrity["passed"],
        "retrieval_locked_e2e": retrieval_locked_e2e["passed"],
        "locked_e2e": retrieval_locked_e2e["passed"],
        "locked_answer_e2e": locked_answer_e2e["passed"],
        "shared_environment_epoch": bool(
            retrieval_locked_e2e.get("environment_epoch_sha256")
            and retrieval_locked_e2e.get("environment_epoch_sha256")
            == locked_answer_e2e.get("environment_epoch_sha256")
        ),
        "train_answer_e2e": train_answer_e2e["passed"],
        "answer_artifact_set": answer_artifact_set["passed"],
        "candidate_samples": (
            candidate["quality_window_stable_traces"] >= MIN_CANDIDATE_TRACES
        ),
        "candidate_sessions": (
            candidate["quality_window_stable_sessions"] >= MIN_CANDIDATE_SESSIONS
        ),
        "candidate_coverage_evidence": (
            candidate["coverage_evidence_traces"] >= MIN_CANDIDATE_TRACES
            and candidate["coverage_evidence_sessions"] >= MIN_CANDIDATE_SESSIONS
        ),
        "candidate_commit_evidence": (
            candidate["commit_evidence_traces"] >= MIN_CANDIDATE_TRACES
            and candidate["commit_evidence_sessions"] >= MIN_CANDIDATE_SESSIONS
        ),
        "candidate_latency_evidence": (
            candidate["paired_latency_traces"] >= MIN_CANDIDATE_TRACES
            and candidate["paired_latency_sessions"] >= MIN_CANDIDATE_SESSIONS
        ),
        "teacher_top30_coverage": (
            candidate["teacher_top30_coverage"] >= MIN_TEACHER_COVERAGE
        ),
        "teacher_top30_coverage_lcb": (
            candidate_teacher_bound.get("valid") is True
            and float(candidate_teacher_bound.get("lower") or 0.0)
            >= MIN_TEACHER_COVERAGE_LCB
        ),
        "teacher_commit_coverage": (
            candidate["teacher_commit_coverage"] >= MIN_TEACHER_COVERAGE
        ),
        "teacher_commit_coverage_lcb": (
            candidate_commit_bound.get("valid") is True
            and float(candidate_commit_bound.get("lower") or 0.0)
            >= MIN_TEACHER_COVERAGE_LCB
        ),
        "candidate_precision": candidate["field_precision_against_teacher"]
        >= MIN_PROCESSOR_USED_PRECISION,
        "candidate_precision_lcb": (
            candidate_precision_bound.get("valid") is True
            and float(candidate_precision_bound.get("lower") or 0.0)
            >= MIN_PROCESSOR_USED_PRECISION_LCB
        ),
        "candidate_confidence_manifest": (
            candidate_confidence.get("samples", 0) >= MIN_CANDIDATE_TRACES
            and candidate_confidence.get("clusters", 0) >= MIN_CANDIDATE_SESSIONS
            and len(str(candidate_confidence.get("manifest_sha256") or "")) == 64
        ),
        "candidate_confidence_complete": (
            candidate["incomplete_quality_traces"] == 0
        ),
        "candidate_cumulative_exact": (
            candidate["cumulative_eligible_trace_count"]
            == candidate["validated_confidence_traces"]
        ),
        "latency": candidate["over_4s"] == 0,
        "full_search_rate": candidate["full_search_rate"] < 1.0,
        "p95_improvement": isinstance(candidate["p95_improvement_ms"], int | float)
        and float(candidate["p95_improvement_ms"]) > 0.0,
        "processor_used_samples": (
            processor["episodes"] >= MIN_PROCESSOR_USED_EPISODES
        ),
        "processor_used_sessions": (processor["sessions"] >= MIN_STRONG_SESSIONS),
        "processor_used_coverage": (
            processor["used_page_coverage"] >= MIN_PROCESSOR_USED_COVERAGE
        ),
        "processor_used_coverage_lcb": (
            processor_coverage_bound.get("valid") is True
            and float(processor_coverage_bound.get("lower") or 0.0)
            >= MIN_PROCESSOR_USED_COVERAGE_LCB
        ),
        "processor_used_precision": (
            processor["used_precision_proxy"] >= MIN_PROCESSOR_USED_PRECISION
        ),
        "processor_used_precision_lcb": (
            processor_precision_bound.get("valid") is True
            and float(processor_precision_bound.get("lower") or 0.0)
            >= MIN_PROCESSOR_USED_PRECISION_LCB
        ),
        "processor_confidence_manifest": (
            processor_confidence.get("samples", 0) >= MIN_PROCESSOR_USED_EPISODES
            and processor_confidence.get("clusters", 0) >= MIN_STRONG_SESSIONS
            and len(str(processor_confidence.get("manifest_sha256") or "")) == 64
        ),
    }
    # Positive co-fire is the mechanism that lets a candidate Field improve its
    # teacher coverage.  Requiring teacher coverage (and therefore full
    # authority eligibility) before enabling that learning creates a circular
    # gate: the Field cannot learn until it is already good enough to ship.
    #
    # Keep the two trust boundaries separate.  Strong/diverse labels, a clean
    # temporal split, and the sealed non-degradation evaluation may unlock
    # positive co-fire while the Field remains candidate-only.  Production
    # authority and scalar policy adoption still require every live gate.
    positive_learning_allowed = bool(
        label_gate
        and integrity["passed"]
        and train_answer_e2e["passed"]
        and learning_decision.get("field_learning_allowed") is True
    )
    authority_eligible = all(gates.values())
    # The paired answer replay is sealed against the currently active LKG.
    # A newly proposed scalar policy has not yet been replayed, so this cycle
    # may record it as a candidate but must not activate it after evaluation.
    policy_update_allowed = False
    # Backwards-compatible public name used by existing dashboards and clients.
    field_learning_allowed = positive_learning_allowed
    effective_mode, canary_percent, stage_started = _advance_rollout(
        previous,
        authority_eligible=authority_eligible,
        candidate_trace_count=int(candidate["cumulative_eligible_trace_count"]),
    )
    if not label_gate:
        stage = "collecting_labels"
    elif not all(
        gates[key]
        for key in (
            "split_integrity",
            "retrieval_locked_e2e",
            "locked_answer_e2e",
            "shared_environment_epoch",
            "train_answer_e2e",
            "answer_artifact_set",
            "candidate_trace_chain",
            "candidate_trace_monotonic",
            "candidate_samples",
            "candidate_sessions",
            "candidate_coverage_evidence",
            "candidate_commit_evidence",
            "candidate_latency_evidence",
            "teacher_top30_coverage",
            "teacher_top30_coverage_lcb",
            "teacher_commit_coverage",
            "teacher_commit_coverage_lcb",
            "candidate_precision",
            "candidate_precision_lcb",
            "candidate_confidence_manifest",
            "candidate_confidence_complete",
            "candidate_cumulative_exact",
            "latency",
            "full_search_rate",
            "p95_improvement",
        )
    ):
        stage = "collecting_candidate_evidence"
    elif not all(
        gates[key]
        for key in (
            "processor_used_samples",
            "processor_used_sessions",
            "processor_used_coverage",
            "processor_used_coverage_lcb",
            "processor_used_precision",
            "processor_used_precision_lcb",
            "processor_confidence_manifest",
        )
    ):
        stage = "collecting_processor_evidence"
    else:
        stage = "canary" if canary_percent < 100 else "active"
    observed_at = now or datetime.now(UTC)
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=UTC)
    generated_at = observed_at.isoformat(timespec="seconds").replace("+00:00", "Z")
    metrics = {
        "labels": counts,
        "candidate": candidate,
        "processor_used": processor,
        "learning": learning_metrics,
    }
    payload = {
        "schema_version": 3,
        "status": "ok",
        "generated_at": generated_at,
        "stage": stage,
        "effective_mode": effective_mode,
        "canary_percent": canary_percent,
        "stage_started_trace_count": stage_started,
        "stage_started_stable_trace_count": stage_started,
        "stage_started_confidence_sample_count": stage_started,
        "field_learning_allowed": field_learning_allowed,
        "positive_learning_allowed": positive_learning_allowed,
        "policy_update_allowed": policy_update_allowed,
        "label_learning_gate": label_gate,
        "authority_enabled": authority_eligible,
        "gates": gates,
        "thresholds": {
            "strong_positive": MIN_STRONG_POSITIVES,
            "strong_positive_sessions": MIN_STRONG_SESSIONS,
            "candidate_traces": MIN_CANDIDATE_TRACES,
            "candidate_sessions": MIN_CANDIDATE_SESSIONS,
            "candidate_trace_scope": "stable_topic_attempt",
            "candidate_evidence_traces": MIN_CANDIDATE_TRACES,
            "candidate_evidence_sessions": MIN_CANDIDATE_SESSIONS,
            "processor_used_episodes": MIN_PROCESSOR_USED_EPISODES,
            "teacher_coverage": MIN_TEACHER_COVERAGE,
            "teacher_coverage_lcb": MIN_TEACHER_COVERAGE_LCB,
            "processor_used_coverage": MIN_PROCESSOR_USED_COVERAGE,
            "processor_used_coverage_lcb": MIN_PROCESSOR_USED_COVERAGE_LCB,
            "processor_used_precision": MIN_PROCESSOR_USED_PRECISION,
            "processor_used_precision_lcb": MIN_PROCESSOR_USED_PRECISION_LCB,
            "quality_window": QUALITY_WINDOW,
        },
        "metrics": metrics,
        "learning": {
            **learning_decision,
            "metrics": learning_metrics,
            "split_integrity": integrity,
        },
        "compiler": compiler,
        "retrieval_locked_e2e": retrieval_locked_e2e,
        "locked_e2e": retrieval_locked_e2e,
        "locked_answer_evaluation": locked_answer_e2e,
        "train_answer_evaluation": train_answer_e2e,
        "answer_artifact_set": answer_artifact_set,
        "extensions": {
            "typed_graph": {
                "gates": extension_gates,
                "authority_mature": all(extension_gates.values()),
                "evaluation_status": str(
                    typed_graph_eval.get("status") or "not_started"
                ),
            },
            "rubric": {
                "status": str(rubric_status.get("status") or "builtin"),
                "gates": rubric_status.get("gates") or {},
            },
        },
    }
    payload = _seal(payload)
    promotion_unsigned = (
        _promotion_payload(metrics)
        if authority_eligible
        else {
            key: value
            for key, value in _failed_promotion(stage, metrics).items()
            if key != "snapshot_sha256"
        }
    )
    registry_path = (
        answer_adapter_registry if isinstance(answer_adapter_registry, Path) else None
    )
    environment_epoch = builtin_field_environment_identity()
    candidate_policy = learning_decision.get("policy")
    candidate_policy = (
        candidate_policy if isinstance(candidate_policy, dict) else current_policy
    )
    promotion_unsigned.update(
        {
            "schema_version": 4,
            "generated_at": generated_at,
            "expires_at": (observed_at + timedelta(hours=24))
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "environment_epoch": environment_epoch,
            "lkg_policy_transition": {
                "base_artifact_sha256": environment_epoch.get(
                    "lkg_base_artifact_sha256", ""
                ),
                "base_snapshot_sha256": str(
                    environment_epoch.get("lkg_base_snapshot_sha256") or ""
                ),
                "base_policy_sha256": _seal(current_policy)["snapshot_sha256"],
                "candidate_result_policy_sha256": _seal(candidate_policy)[
                    "snapshot_sha256"
                ],
                "activated_artifact_sha256": str(
                    environment_epoch.get("lkg_base_artifact_sha256") or ""
                ),
                "evaluated_effective_config_sha256": str(
                    environment_epoch.get("effective_field_config_sha256") or ""
                ),
                "activated": False,
            },
            "growth_state_snapshot_sha256": payload["snapshot_sha256"],
            "candidate_trace": {
                "path": str(candidate_trace_file.resolve(strict=False)),
                "file_sha256": _file_sha256(candidate_trace_file),
                "head_sha256": candidate["trace_chain_head_sha256"],
                "cumulative_eligible_trace_count": candidate[
                    "cumulative_eligible_trace_count"
                ],
            },
            "source_artifacts": {
                "manual94": {
                    "path": str(locked_e2e_file.resolve(strict=False)),
                    "file_sha256": _file_sha256(locked_e2e_file),
                },
                "recall_log": _append_only_prefix_identity(
                    inputs["recall_log_file"]
                ),
                "pull_log": _append_only_prefix_identity(inputs["pull_log_file"]),
                "train_answer": {
                    "path": str(train_answer_eval_file.resolve(strict=False)),
                    "file_sha256": _file_sha256(train_answer_eval_file),
                },
                "locked_answer": {
                    "path": str(locked_answer_eval_file.resolve(strict=False)),
                    "file_sha256": _file_sha256(locked_answer_eval_file),
                },
                "answer_episodes": {
                    "path": str(answer_episode_file.resolve(strict=False)),
                    "file_sha256": _file_sha256(answer_episode_file),
                },
                "answer_reviews": {
                    "path": str(answer_review_ledger_file.resolve(strict=False)),
                    "file_sha256": _file_sha256(answer_review_ledger_file),
                },
                "answer_executions": {
                    "path": str(answer_execution_ledger_file.resolve(strict=False)),
                    "file_sha256": _file_sha256(answer_execution_ledger_file),
                },
                "answer_registry": {
                    "path": str(registry_path.resolve(strict=False)) if registry_path else "",
                    "file_sha256": _file_sha256(registry_path) if registry_path else "",
                },
            },
        }
    )
    promotion = _seal(promotion_unsigned)
    if not dry_run:
        _persist_growth_artifacts(
            state_file=state_file,
            promotion_file=promotion_file,
            history_file=history_file,
            policy_history_file=policy_history_file,
            last_known_good_file=last_known_good_file,
            payload=payload,
            promotion=promotion,
            learning_decision=learning_decision,
            current_policy=current_policy,
            integrity=integrity,
        )
    return payload


def load_growth_state(path: Path = GROWTH_STATE_FILE) -> dict[str, Any]:
    """Load the bounded public operational state used by runtime and UI."""

    payload = _read_json(path)
    seal = payload.get("snapshot_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "snapshot_sha256"}
    if (
        payload.get("schema_version") != 3
        or not isinstance(seal, str)
        or seal != hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    ):
        return {}
    return payload


def automatic_rollout(
    *,
    enabled: bool,
    state_file: Path = GROWTH_STATE_FILE,
) -> tuple[str, int]:
    """Return the fail-closed effective Field rollout."""

    if not enabled:
        return "", 0
    state = load_growth_state(state_file)
    mode = str(state.get("effective_mode") or "candidate")
    percent = state.get("canary_percent")
    if mode != "active" or not isinstance(percent, int) or percent not in range(1, 101):
        return "candidate", 100
    if state.get("authority_enabled") is not True:
        return "candidate", 100
    from chronovisor.recall.recall_field_candidate import authority_allowed

    if not authority_allowed(state_file.parent / PROMOTION_ARTIFACT.name):
        return "candidate", 100
    return "active", percent


def automatic_learning_allowed(
    *,
    enabled: bool,
    state_file: Path = GROWTH_STATE_FILE,
) -> bool:
    if not enabled:
        return False
    state = load_growth_state(state_file)
    return bool(
        state.get("positive_learning_allowed") is True
        and state.get("train_answer_evaluation", {}).get("passed") is True
    )


def automatic_processor_authority_allowed(
    *,
    enabled: bool,
    state_file: Path = GROWTH_STATE_FILE,
) -> bool:
    if not enabled:
        return False
    state = load_growth_state(state_file)
    from chronovisor.recall.recall_field_candidate import authority_allowed

    return bool(
        state.get("authority_enabled") is True
        and state.get("effective_mode") == "active"
        and int(state.get("canary_percent") or 0) > 0
        and authority_allowed(state_file.parent / PROMOTION_ARTIFACT.name)
    )


__all__ = [
    "automatic_learning_allowed",
    "automatic_processor_authority_allowed",
    "automatic_rollout",
    "candidate_metrics",
    "load_growth_state",
    "processor_used_metrics",
    "run_growth_cycle",
]
