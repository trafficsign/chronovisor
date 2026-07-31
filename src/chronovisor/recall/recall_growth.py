"""Autonomous supervision, learning, and rollout control for Recall Field."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import exclusive_text_file_lock
from chronovisor.core.link_fix import atomic_write
from chronovisor.core.store import CHRONOVISOR_ROOT
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
CANARY_STEPS = (5, 25, 100)
CANARY_ADVANCE_SAMPLES = 100
QUALITY_WINDOW = 200


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path, *, limit: int = 20_000) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-max(1, limit) :]
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


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
    return round(ordered[max(0, index)], 3)


def candidate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate privacy-safe Field/teacher comparisons."""

    sessions: set[str] = set()
    teacher_pages = 0
    teacher_overlap = 0
    field_pages = 0
    committed_pages = 0
    committed_overlap = 0
    latencies: list[float] = []
    field_latencies: list[float] = []
    teacher_latencies: list[float] = []
    over_4s = 0
    full_searches = 0
    active_rows = 0
    for row in rows:
        session = str(row.get("session_hash") or "")
        if session:
            sessions.add(session)
        field_ids = {
            str(value)
            for value in row.get("field_page_ids", [])
            if isinstance(value, str) and value
        }
        teacher_ids = {
            str(value)
            for value in row.get("teacher_page_ids", [])
            if isinstance(value, str) and value
        }
        committed_ids = {
            str(value)
            for value in row.get("committed_page_ids", [])
            if isinstance(value, str) and value
        }
        teacher_pages += len(teacher_ids)
        teacher_overlap += len(teacher_ids & field_ids)
        field_pages += len(field_ids)
        committed_pages += len(committed_ids)
        committed_overlap += len(committed_ids & field_ids)
        latency = row.get("latency_ms")
        if isinstance(latency, int | float) and not isinstance(latency, bool):
            latencies.append(max(0.0, float(latency)))
        field_latency = row.get("field_latency_ms")
        if isinstance(field_latency, int | float) and not isinstance(
            field_latency, bool
        ):
            field_latencies.append(max(0.0, float(field_latency)))
        teacher_latency = row.get("teacher_latency_ms")
        if isinstance(teacher_latency, int | float) and not isinstance(
            teacher_latency, bool
        ):
            teacher_latencies.append(max(0.0, float(teacher_latency)))
        if row.get("over_4s") is True or (
            isinstance(latency, int | float) and float(latency) > 4_000
        ):
            over_4s += 1
        if row.get("full_search_required") is True:
            full_searches += 1
        if row.get("authority") == "field":
            active_rows += 1
    return {
        "traces": len(rows),
        "sessions": len(sessions),
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
        "p95_improvement_ms": (
            round(
                float(_percentile(teacher_latencies, 0.95))
                - float(_percentile(field_latencies, 0.95)),
                3,
            )
            if field_latencies and teacher_latencies
            else None
        ),
        "over_4s": over_4s,
        "full_searches": full_searches,
        "full_search_rate": round(full_searches / len(rows) if rows else 0.0, 6),
        "active_traces": active_rows,
    }


def split_integrity(labels: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Prove that neither a session nor a query spans evaluation splits."""

    sessions: dict[str, set[str]] = {}
    queries: dict[str, set[str]] = {}
    for row in labels:
        split = str(row.get("split") or "")
        session = str(row.get("session_hash") or "")
        query = str(row.get("query_sha256") or "")
        if session and split:
            sessions.setdefault(session, set()).add(split)
        if query and split:
            queries.setdefault(query, set()).add(split)
    session_leaks = sum(len(values) > 1 for values in sessions.values())
    query_leaks = sum(len(values) > 1 for values in queries.values())
    return {
        "sessions": len(sessions),
        "queries": len(queries),
        "session_leakage": session_leaks,
        "query_leakage": query_leaks,
        "passed": session_leaks == 0 and query_leaks == 0,
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


def locked_e2e_status(path: Path) -> dict[str, Any]:
    """Verify the sealed manual-94 E2E artifact used by promotion gates."""

    payload = _read_json(path)
    if not payload:
        return {"status": "missing", "passed": False}
    seal = str(payload.get("snapshot_sha256") or "")
    unsigned = {
        key: value for key, value in payload.items() if key != "snapshot_sha256"
    }
    if (
        not seal
        or seal
        != hashlib.sha256(
            json.dumps(
                unsigned,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    ):
        return {"status": "invalid", "passed": False}
    gates = payload.get("gates")
    passed = bool(
        payload.get("status") == "passed"
        and isinstance(gates, dict)
        and gates
        and all(value is True for value in gates.values())
    )
    return {
        "status": str(payload.get("status") or "invalid"),
        "passed": passed,
        "manifest_sha256": str(payload.get("manifest_sha256") or ""),
        "precision_delta_points": payload.get("precision_delta_points"),
        "recall_delta_points": payload.get("recall_delta_points"),
        "precision_lower_95": payload.get("precision_lower_95"),
        "gates": gates if isinstance(gates, dict) else {},
    }


def processor_used_metrics(
    recall_rows: Sequence[Mapping[str, Any]],
    pull_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure whether shadow Processor retained pages explicitly used later."""

    joined = join_used_recall_episodes(list(recall_rows), list(pull_rows))
    observed: list[tuple[str, set[str], set[str]]] = []
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
    }


def _promotion_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    candidate = metrics["candidate"]
    used = metrics["processor_used"]
    learning = metrics["learning"]
    return {
        "schema_version": 1,
        "status": "passed",
        "metrics": {
            "teacher_commit_coverage": float(candidate["teacher_commit_coverage"]),
            "precision_delta_points": float(learning["precision_delta_points"]),
            "recall_delta_points": float(learning["recall_delta_points"]),
            "over_4s": int(candidate["over_4s"]),
            "full_search_rate": float(candidate["full_search_rate"]),
            "processor_used_page_coverage": float(used["used_page_coverage"]),
            "processor_used_precision_proxy": float(used["used_precision_proxy"]),
        },
    }


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {**payload, "snapshot_sha256": hashlib.sha256(encoded).hexdigest()}


def _failed_promotion(reason: str, metrics: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "status": "held",
        "reason": reason,
        "metrics": {
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
        },
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
    started_at = int(previous.get("stage_started_trace_count") or 0)
    if previous_mode != "active" or previous_percent not in CANARY_STEPS:
        return "active", CANARY_STEPS[0], candidate_trace_count
    new_samples = max(0, candidate_trace_count - started_at)
    if new_samples < CANARY_ADVANCE_SAMPLES:
        return "active", previous_percent, started_at
    index = CANARY_STEPS.index(previous_percent)
    next_percent = CANARY_STEPS[min(index + 1, len(CANARY_STEPS) - 1)]
    return "active", next_percent, candidate_trace_count


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
    inputs = label_inputs or default_label_ledger_inputs()
    labels = (
        build_label_ledger(**inputs) if dry_run else materialize_label_ledger(**inputs)
    )
    recall_rows = _read_jsonl(inputs["recall_log_file"])
    pull_rows = _read_jsonl(inputs["pull_log_file"])
    candidate_rows = _read_jsonl(candidate_trace_file)
    candidate_totals = candidate_metrics(candidate_rows)
    candidate_quality = candidate_metrics(candidate_rows[-QUALITY_WINDOW:])
    candidate = {
        **candidate_quality,
        "traces": candidate_totals["traces"],
        "sessions": candidate_totals["sessions"],
        "active_traces": candidate_totals["active_traces"],
        "quality_window_traces": candidate_quality["traces"],
    }
    processor = processor_used_metrics(recall_rows, pull_rows)
    counts = dict(labels["counts"])
    label_gate = bool(labels["gates"]["field_learning_allowed"])
    integrity = split_integrity(labels["labels"])
    compiler = compiler_metrics(_read_jsonl(compiler_trace_file))
    locked_e2e = locked_e2e_status(locked_e2e_file)
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
    locked_precision_delta = locked_e2e.get("precision_delta_points")
    locked_recall_delta = locked_e2e.get("recall_delta_points")
    learning_metrics = {
        "session_leakage": float(integrity["session_leakage"]),
        "query_leakage": float(integrity["query_leakage"]),
        "precision_delta_points": (
            float(locked_precision_delta)
            if isinstance(locked_precision_delta, int | float)
            else round(
                (float(candidate["field_precision_against_teacher"]) - 1.0) * 100.0,
                6,
            )
        ),
        "recall_delta_points": (
            float(locked_recall_delta)
            if isinstance(locked_recall_delta, int | float)
            else round(
                (float(candidate["teacher_commit_coverage"]) - 1.0) * 100.0,
                6,
            )
        ),
    }
    learning_decision = decide_learning_update(
        current={key: float(value) for key, value in current_policy.items()},
        proposed={key: float(value) for key, value in proposed_policy.items()},
        label_counts=counts,
        metrics=learning_metrics,
    )
    gates = {
        "labels": label_gate,
        "split_integrity": integrity["passed"],
        "locked_e2e": locked_e2e["passed"],
        "candidate_samples": candidate["traces"] >= MIN_CANDIDATE_TRACES,
        "candidate_sessions": candidate["sessions"] >= MIN_CANDIDATE_SESSIONS,
        "teacher_top30_coverage": (
            candidate["teacher_top30_coverage"] >= MIN_TEACHER_COVERAGE
        ),
        "teacher_commit_coverage": (
            candidate["teacher_commit_coverage"] >= MIN_TEACHER_COVERAGE
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
        "processor_used_precision": (
            processor["used_precision_proxy"] >= MIN_PROCESSOR_USED_PRECISION
        ),
    }
    authority_eligible = all(gates.values())
    field_learning_allowed = bool(
        authority_eligible and learning_decision.get("field_learning_allowed") is True
    )
    previous = _read_json(state_file)
    effective_mode, canary_percent, stage_started = _advance_rollout(
        previous,
        authority_eligible=authority_eligible,
        candidate_trace_count=int(candidate["traces"]),
    )
    if not label_gate:
        stage = "collecting_labels"
    elif not all(
        gates[key]
        for key in (
            "split_integrity",
            "locked_e2e",
            "candidate_samples",
            "candidate_sessions",
            "teacher_top30_coverage",
            "teacher_commit_coverage",
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
            "processor_used_precision",
        )
    ):
        stage = "collecting_processor_evidence"
    else:
        stage = "canary" if canary_percent < 100 else "active"
    generated_at = (now or datetime.now()).isoformat(timespec="seconds")
    metrics = {
        "labels": counts,
        "candidate": candidate,
        "processor_used": processor,
        "learning": learning_metrics,
    }
    payload = {
        "schema_version": 1,
        "status": "ok",
        "generated_at": generated_at,
        "stage": stage,
        "effective_mode": effective_mode,
        "canary_percent": canary_percent,
        "stage_started_trace_count": stage_started,
        "field_learning_allowed": field_learning_allowed,
        "label_learning_gate": label_gate,
        "authority_enabled": authority_eligible,
        "gates": gates,
        "thresholds": {
            "strong_positive": MIN_STRONG_POSITIVES,
            "strong_positive_sessions": MIN_STRONG_SESSIONS,
            "candidate_traces": MIN_CANDIDATE_TRACES,
            "candidate_sessions": MIN_CANDIDATE_SESSIONS,
            "processor_used_episodes": MIN_PROCESSOR_USED_EPISODES,
            "teacher_coverage": MIN_TEACHER_COVERAGE,
            "processor_used_coverage": MIN_PROCESSOR_USED_COVERAGE,
            "processor_used_precision": MIN_PROCESSOR_USED_PRECISION,
            "quality_window": QUALITY_WINDOW,
        },
        "metrics": metrics,
        "learning": {
            **learning_decision,
            "metrics": learning_metrics,
            "split_integrity": integrity,
        },
        "compiler": compiler,
        "locked_e2e": locked_e2e,
    }
    promotion = (
        _seal(_promotion_payload(metrics))
        if authority_eligible
        else _failed_promotion(stage, metrics)
    )
    if not dry_run:
        _write_json(state_file, payload)
        _write_json(promotion_file, promotion)
        _append_history(
            history_file,
            {
                "generated_at": generated_at,
                "stage": stage,
                "effective_mode": effective_mode,
                "canary_percent": canary_percent,
                "field_learning_allowed": field_learning_allowed,
                "authority_enabled": authority_eligible,
                "gates": gates,
                "labels": counts,
                "candidate": candidate,
                "processor_used": processor,
                "learning": learning_metrics,
                "compiler": compiler,
            },
        )
        policy_record = append_policy_history(
            {
                "generated_at": generated_at,
                "status": str(learning_decision.get("status") or "held"),
                "reason": str(learning_decision.get("reason") or ""),
                "policy": learning_decision.get("policy", current_policy),
                "label_counts": counts,
                "metrics": learning_metrics,
                "split_integrity": integrity,
                "authority_enabled": authority_eligible,
            },
            path=policy_history_file,
        )
        chain = verify_policy_history(policy_history_file)
        if (
            authority_eligible
            and field_learning_allowed
            and chain.get("status") == "ok"
        ):
            write_last_known_good(
                {
                    key: float(value)
                    for key, value in learning_decision.get(
                        "policy", current_policy
                    ).items()
                },
                evaluation={
                    "candidate": candidate,
                    "processor_used": processor,
                    "split_integrity": integrity,
                },
                history_head_sha256=str(policy_record["record_sha256"]),
                path=last_known_good_file,
            )
    return payload


def load_growth_state(path: Path = GROWTH_STATE_FILE) -> dict[str, Any]:
    """Load the bounded public operational state used by runtime and UI."""

    return _read_json(path)


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
    return "active", percent


def automatic_learning_allowed(
    *,
    enabled: bool,
    state_file: Path = GROWTH_STATE_FILE,
) -> bool:
    if not enabled:
        return False
    return load_growth_state(state_file).get("field_learning_allowed") is True


def automatic_processor_authority_allowed(
    *,
    enabled: bool,
    state_file: Path = GROWTH_STATE_FILE,
) -> bool:
    if not enabled:
        return False
    state = load_growth_state(state_file)
    return bool(
        state.get("authority_enabled") is True
        and state.get("effective_mode") == "active"
        and int(state.get("canary_percent") or 0) > 0
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
