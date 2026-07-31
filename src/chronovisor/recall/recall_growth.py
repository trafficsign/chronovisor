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
from chronovisor.recall.recall_log_schema import join_used_recall_episodes

RUNTIME_DIR = CHRONOVISOR_ROOT / "runtime" / "recall-field"
GROWTH_STATE_FILE = RUNTIME_DIR / "growth-state.json"
GROWTH_HISTORY_FILE = RUNTIME_DIR / "growth-history.jsonl"
CANDIDATE_TRACE_FILE = RUNTIME_DIR / "candidate-trace.jsonl"
PROMOTION_ARTIFACT = RUNTIME_DIR / "promotion.json"

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
    committed_pages = 0
    committed_overlap = 0
    latencies: list[float] = []
    over_4s = 0
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
        committed_pages += len(committed_ids)
        committed_overlap += len(committed_ids & field_ids)
        latency = row.get("latency_ms")
        if isinstance(latency, int | float) and not isinstance(latency, bool):
            latencies.append(max(0.0, float(latency)))
        if row.get("over_4s") is True or (
            isinstance(latency, int | float) and float(latency) > 4_000
        ):
            over_4s += 1
        if row.get("authority") == "field":
            active_rows += 1
    return {
        "traces": len(rows),
        "sessions": len(sessions),
        "teacher_pages": teacher_pages,
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
        "over_4s": over_4s,
        "active_traces": active_rows,
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
    coverage = min(
        float(candidate["teacher_top30_coverage"]),
        float(candidate["teacher_commit_coverage"]),
        float(used["used_page_coverage"]),
    )
    return {
        "schema_version": 1,
        "status": "passed",
        "metrics": {
            "teacher_commit_coverage": float(candidate["teacher_commit_coverage"]),
            "precision_delta_points": round((coverage - 1.0) * 100.0, 6),
            "recall_delta_points": round((coverage - 1.0) * 100.0, 6),
            "over_4s": int(candidate["over_4s"]),
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
    label_inputs: dict[str, Path] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Refresh supervision and advance a fail-closed autonomous rollout."""

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
    gates = {
        "labels": label_gate,
        "candidate_samples": candidate["traces"] >= MIN_CANDIDATE_TRACES,
        "candidate_sessions": candidate["sessions"] >= MIN_CANDIDATE_SESSIONS,
        "teacher_top30_coverage": (
            candidate["teacher_top30_coverage"] >= MIN_TEACHER_COVERAGE
        ),
        "teacher_commit_coverage": (
            candidate["teacher_commit_coverage"] >= MIN_TEACHER_COVERAGE
        ),
        "latency": candidate["over_4s"] == 0,
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
            "candidate_samples",
            "candidate_sessions",
            "teacher_top30_coverage",
            "teacher_commit_coverage",
            "latency",
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
    }
    payload = {
        "schema_version": 1,
        "status": "ok",
        "generated_at": generated_at,
        "stage": stage,
        "effective_mode": effective_mode,
        "canary_percent": canary_percent,
        "stage_started_trace_count": stage_started,
        "field_learning_allowed": label_gate,
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
                "field_learning_allowed": label_gate,
                "authority_enabled": authority_eligible,
                "gates": gates,
                "labels": counts,
                "candidate": candidate,
                "processor_used": processor,
            },
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
