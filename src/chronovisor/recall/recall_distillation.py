"""Autonomous, point-in-time Recall distillation contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import sys
import threading
import time
import unicodedata
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from chronovisor.core import (
    canonical_json,
    claude_code_transcript,
    codex_transcript,
    pi_transcript,
    runtime_config,
)
from chronovisor.core.raw_store import (
    RawSegmentCorrupt,
    RawStore,
    committed_event_spans,
    committed_raw_watermark,
)
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall.recall_calibration import sigmoid

RALLY_SCHEMA = "chronovisor.recall-rally.rally-v1"
BASELINE_SCHEMA = "chronovisor.recall-distill-baseline.v1"
POLICY_SCHEMA = "chronovisor.recall-distill-policy.v2"
OUTCOME_SCHEMA = "chronovisor.recall-closed-outcome.v1"
VETO_SCHEMA = "chronovisor.recall-authenticated-negative-veto.v1"
SHADOW_OBSERVATION_SCHEMA = "chronovisor.recall-distill-shadow-observation.v1"
SPLIT_PLAN_SCHEMA = "chronovisor.recall-distill-split-plan.v1"
ASSIGNMENT_REVISION = "assignment-v2"
PROBE_REVISION = "probe-v2"
TEACHER_ROLES = (
    "recall.distill.teacher.a",
    "recall.distill.teacher.b",
    "recall.distill.teacher.c",
)
UTILITY_LABELS = frozenset({"helpful", "neutral", "harmful", "uncertain"})
RELEVANCE_LABELS = frozenset({"relevant", "irrelevant", "uncertain"})
AUTHORITIES = frozenset({"verified", "teacher-only", "uncertain", "reject"})
CLOSED_PREDICATES = frozenset(
    {
        "exact_claim_supported",
        "exact_claim_contradicted",
        "exact_supersession",
        "exact_test_outcome",
        "exact_rollback_outcome",
        "exact_correction_link",
        "exact_commit_overlap",
        "exact_path_overlap",
        "exact_task_uuid_overlap",
    }
)
RELEVANCE_CLOSED_PREDICATES = frozenset(
    {
        "exact_claim_supported",
        "exact_claim_contradicted",
        "exact_supersession",
        "exact_correction_link",
        "exact_commit_overlap",
        "exact_path_overlap",
        "exact_task_uuid_overlap",
    }
)
TEXT_FEATURE_REVISION = "recall-distill-text-v2"
FAST_FEATURE_KEYS = ("query_chargram_coverage", "candidate_chargram_precision")
FORBIDDEN_LIVE_FEATURES = frozenset(
    {
        "a_t",
        "a1",
        "actual_answer",
        "answer",
        "answer_delta",
        "gap",
        "outcome",
        "future_entity",
        "future_timestamp",
        "historical_fts_score",
        "used",
        "ignored",
    }
)
FALSE_VALUES = frozenset({"0", "false", "no", "off"})
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


class DistillationError(ValueError):
    """A distillation input violates a deterministic or safety contract."""


class DistillationDeferred(RuntimeError):
    """A foreground-safe worker call was deferred and must remain resumable."""


class Teacher(Protocol):
    role: str
    local: bool

    def evaluate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


class CounterfactualGenerator(Protocol):
    local: bool

    def compare(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


class _WorkerTeacher:
    local = True

    def __init__(
        self,
        role: str,
        max_input_bytes: int,
        expected_route: Mapping[str, str],
        expected_digest: str,
        deadline_ms: int = 60_000,
    ) -> None:
        self.role = role
        self.max_input_bytes = max_input_bytes
        self.expected_route = expected_route
        self.expected_digest = expected_digest
        self.deadline_ms = deadline_ms

    def evaluate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return _worker_call(
            "teacher",
            self.role,
            payload,
            max_input_bytes=self.max_input_bytes,
            expected_route=self.expected_route,
            expected_digest=self.expected_digest,
            deadline_ms=self.deadline_ms,
        )


class _WorkerCounterfactual:
    local = True

    def __init__(
        self,
        max_input_bytes: int,
        routes: Mapping[str, Mapping[str, str]],
        digests: Mapping[str, str],
        deadline_ms: int = 60_000,
    ) -> None:
        self.max_input_bytes = max_input_bytes
        self.routes = routes
        self.digests = digests
        self.deadline_ms = deadline_ms

    def compare(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        base = {
            key: payload[key]
            for key in ("rally_id", "candidate_id", "query", "context")
        }
        arm0 = _worker_call(
            "answer",
            "recall.distill.answer_generator",
            {**base, "evidence": payload.get("a0_evidence", [])},
            max_input_bytes=self.max_input_bytes,
            expected_route=self.routes["recall.distill.answer_generator"],
            expected_digest=self.digests["recall.distill.answer_generator"],
            deadline_ms=self.deadline_ms,
        )
        arm1 = _worker_call(
            "answer",
            "recall.distill.answer_generator",
            {**base, "evidence": payload.get("a1_evidence", [])},
            max_input_bytes=self.max_input_bytes,
            expected_route=self.routes["recall.distill.answer_generator"],
            expected_digest=self.digests["recall.distill.answer_generator"],
            deadline_ms=self.deadline_ms,
        )
        a0 = str(arm0.get("answer") or "")
        a1 = str(arm1.get("answer") or "")
        first_a = (
            int.from_bytes(
                hashlib.sha256(
                    f"blind-order-v1\0{base['rally_id']}\0{base['candidate_id']}".encode()
                ).digest()[:8],
                "big",
            )
            % 2
            == 0
        )

        def judge(a_first: bool) -> Mapping[str, Any]:
            return _worker_call(
                "utility",
                "recall.distill.utility_judge",
                {
                    "rally_id": base["rally_id"],
                    "candidate_id": base["candidate_id"],
                    "answer_a": a0 if a_first else a1,
                    "answer_b": a1 if a_first else a0,
                    "blind_order": "a_first" if a_first else "b_first",
                    "actual_answer_diagnostic": payload.get("actual_answer", ""),
                },
                max_input_bytes=self.max_input_bytes,
                expected_route=self.routes["recall.distill.utility_judge"],
                expected_digest=self.digests["recall.distill.utility_judge"],
                deadline_ms=self.deadline_ms,
            )

        first = judge(first_a)
        second = judge(not first_a)

        def decoded_choice(result: Mapping[str, Any], a0_first: bool) -> str:
            choice = result.get("blind_choice")
            if choice == "tie":
                return "neutral"
            if choice not in {"a", "b"}:
                return "uncertain"
            candidate_arm = "b" if a0_first else "a"
            return "helpful" if choice == candidate_arm else "harmful"

        first_verdict = decoded_choice(first, first_a)
        second_verdict = decoded_choice(second, not first_a)
        distinct_models = arm0.get("_model_digest") != first.get("_model_digest")
        verdict = (
            first_verdict
            if distinct_models and first_verdict == second_verdict
            else "uncertain"
        )
        return {
            "verdict": verdict,
            "reason": str(first.get("rationale") or "")[:500],
            "a0_sha256": hashlib.sha256(a0.encode()).hexdigest(),
            "a1_sha256": hashlib.sha256(a1.encode()).hexdigest(),
            "blind_orders": [
                "a0_first" if first_a else "a1_first",
                "a1_first" if first_a else "a0_first",
            ],
            "blind_choices": [first.get("blind_choice"), second.get("blind_choice")],
            "order_agreement": distinct_models and first_verdict == second_verdict,
            "generator_route_identity": arm0.get("_route_identity", {}),
            "generator_model_digest": arm0.get("_model_digest", ""),
            "judge_route_identity": first.get("_route_identity", {}),
            "judge_model_digest": first.get("_model_digest", ""),
        }


def _worker_call(
    operation: str,
    role: str,
    payload: Mapping[str, Any],
    *,
    max_input_bytes: int,
    expected_route: Mapping[str, str],
    expected_digest: str,
    deadline_ms: int = 60_000,
) -> Mapping[str, Any]:
    from chronovisor.core import ollama
    from chronovisor.core.research_scheduler import (
        research_lane,
        run_cancellable_command,
    )

    route = ollama.runtime_generation_routes((role,))[0]
    if route.role != role or route.location != "local" or not route.structured_output:
        raise DistillationError("distillation worker route is not local structured")
    request_id = canonical_json.canonical_json_sha256_strict(
        {"operation": operation, "role": role, "input": payload}
    )
    request = {
        "schema": "chronovisor.recall-distillation-worker.v1",
        "operation": operation,
        "role": role,
        "request_id": request_id,
        "input": dict(payload),
        "deadline_ms": deadline_ms,
    }
    input_ceiling = min(max_input_bytes, 12_000)
    if len(canonical_json.canonical_json_bytes_strict(payload)) > input_ceiling:
        raise DistillationError("distillation worker input exceeds fixed byte limit")
    encoded = canonical_json.canonical_json_strict(request)
    with research_lane(
        f"recall-distill-{request_id[:16]}",
        enabled=True,
        mode="auto",
        purpose="sleep",
        needs_model=True,
    ) as lease:
        outcome = run_cancellable_command(
            [sys.executable, "-m", "chronovisor.recall.recall_distillation_worker"],
            encoded,
            lease,
            timeout_seconds=deadline_ms / 1_000 + 5,
        )
    if outcome.status != "completed" or not isinstance(outcome.value, dict):
        if outcome.status in {"deferred", "cancelled", "timeout"}:
            raise DistillationDeferred(f"distillation worker {outcome.status}")
        raise DistillationError(f"distillation worker {outcome.status}")
    response = outcome.value
    route_identity = response.get("route_identity")
    model_digest = response.get("model_digest")
    if (
        response.get("schema") != "chronovisor.recall-distillation-worker.v1"
        or response.get("operation") != operation
        or response.get("role") != role
        or response.get("request_id") != request_id
    ):
        raise DistillationError("distillation worker response is invalid")
    if response.get("ok") is not True:
        failure_class = str(response.get("failure_class") or "output_invalid")
        if failure_class in {
            "route_unavailable",
            "backend_error",
            "transport_timeout",
            "transport_error",
            "capacity_unavailable",
            "foreground_preempted",
            "resource_busy",
            "cancelled",
            "timeout",
        }:
            raise DistillationDeferred(f"distillation worker {failure_class}")
        raise DistillationError("distillation worker output is invalid")
    if (
        not isinstance(response.get("result"), dict)
        or not isinstance(route_identity, dict)
        or route_identity != dict(expected_route)
        or not isinstance(model_digest, str)
        or model_digest != expected_digest
    ):
        raise DistillationError("distillation worker response is invalid")
    return {
        **response["result"],
        "_route_identity": route_identity,
        "_model_digest": model_digest,
    }


def _default_workers(
    config: DistillationConfig,
    *,
    deadline_ms: int = 60_000,
) -> tuple[dict[str, Teacher], CounterfactualGenerator | None]:
    from chronovisor.core import ollama

    roles = (
        *TEACHER_ROLES,
        "recall.distill.answer_generator",
        "recall.distill.utility_judge",
    )
    try:
        routes = ollama.runtime_generation_routes(roles)
    except Exception:
        return {}, None
    if tuple(route.role for route in routes) != roles or any(
        route.provider != "ollama"
        or route.location != "local"
        or not route.structured_output
        for route in routes
    ):
        return {}, None
    try:
        model_digests = ollama.model_digests([route.model for route in routes])
    except Exception:
        return {}, None
    if any(not model_digests.get(route.model) for route in routes):
        return {}, None
    identities = {
        route.role: {
            "role": route.role,
            "provider": route.provider,
            "model": route.model,
            "location": route.location,
        }
        for route in routes
    }
    digests = {route.role: model_digests[route.model] for route in routes}
    if len({digests[role] for role in TEACHER_ROLES}) != len(TEACHER_ROLES):
        return {}, None
    return (
        {
            role: _WorkerTeacher(
                role,
                config.max_input_bytes,
                identities[role],
                digests[role],
                deadline_ms,
            )
            for role in TEACHER_ROLES
        },
        _WorkerCounterfactual(
            config.max_input_bytes, identities, digests, deadline_ms
        ),
    )


@dataclass(frozen=True)
class DistillationConfig:
    enabled: bool = False
    chunk_size: int = 25
    max_input_bytes: int = 12_000
    max_candidates: int = 200
    hard_floor_rallies: int = 1_000
    hard_floor_days: int = 30
    hard_floor_windows: int = 3
    hard_floor_teacher_labels: int = 500
    hard_floor_teacher_per_class: int = 100
    hard_floor_probe_pairs: int = 100
    hard_floor_counterfactual_pairs: int = 100
    rollout_stages: tuple[int, ...] = (5, 25, 100)
    canary_min_days: int = 7


def _bool_override(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in FALSE_VALUES:
        return False
    if normalized in TRUE_VALUES:
        return True
    raise DistillationError("invalid CHRONOVISOR_RECALL_DISTILLATION value")


def _config_table(config_path: Path | None = None) -> Mapping[str, Any]:
    data = (
        runtime_config.load_toml_file()
        if config_path is None
        else runtime_config.load_toml_file(config_path)
    )
    recall = data.get("recall") if isinstance(data, dict) else None
    table = recall.get("distillation") if isinstance(recall, dict) else None
    return table if isinstance(table, dict) else {}


def distillation_enabled(config_path: Path | None = None) -> bool:
    override = os.environ.get("CHRONOVISOR_RECALL_DISTILLATION")
    if override is not None:
        return _bool_override(override)
    return _config_table(config_path).get("enabled") is True


def load_distillation_config(config_path: Path | None = None) -> DistillationConfig:
    table = _config_table(config_path)

    def positive(name: str, default: int) -> int:
        value = table.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise DistillationError(f"recall.distillation.{name} must be positive")
        return value

    stages = table.get("rollout_stages", [5, 25, 100])
    if not isinstance(stages, list) or tuple(stages) != (5, 25, 100):
        raise DistillationError("rollout_stages must be [5, 25, 100]")
    return DistillationConfig(
        enabled=distillation_enabled(config_path),
        chunk_size=positive("chunk_size", 25),
        max_input_bytes=positive("max_input_bytes", 12_000),
        max_candidates=positive("max_candidates", 200),
        hard_floor_rallies=positive("hard_floor_rallies", 1_000),
        hard_floor_days=positive("hard_floor_days", 30),
        hard_floor_windows=positive("hard_floor_windows", 3),
        hard_floor_teacher_labels=positive("hard_floor_teacher_labels", 500),
        hard_floor_teacher_per_class=positive("hard_floor_teacher_per_class", 100),
        hard_floor_probe_pairs=positive("hard_floor_probe_pairs", 100),
        hard_floor_counterfactual_pairs=positive(
            "hard_floor_counterfactual_pairs", 100
        ),
        rollout_stages=(5, 25, 100),
        canary_min_days=positive("canary_min_days", 7),
    )


def _timestamp(value: object, fallback: str) -> tuple[str, int]:
    selected = value if isinstance(value, str) and value else fallback
    try:
        parsed = datetime.fromisoformat(selected.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DistillationError("Raw event timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise DistillationError("Raw event timestamp has no timezone")
    normalized = parsed.astimezone(UTC)
    return selected, int(normalized.timestamp() * 1_000_000)


def _event_semantics(host: str, event: Mapping[str, Any]) -> tuple[str, str]:
    if host == "codex":
        return codex_transcript.codex_semantic_view(
            event.get("type"), event.get("payload")
        )
    if host == "claude-code":
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        return claude_code_transcript.claude_semantic_view(event.get("type"), content)
    if host == "pi":
        item_type, content = pi_transcript._pi_message_view(dict(event))
        return pi_transcript.claude_semantic_view(item_type, content)
    if host == "hermes":
        message = event.get("message")
        if not isinstance(message, dict):
            raise DistillationError("Hermes Raw event has no message")
        role = message.get("role")
        content = message.get("content")
        text = (
            content
            if isinstance(content, str)
            else ""
            if content is None
            else json.dumps(content, ensure_ascii=False, sort_keys=True)
        )
        return (str(role) if isinstance(role, str) else "event"), text
    raise DistillationError(f"unsupported committed Raw host: {host}")


def _event_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "raw_id": row["raw_id"],
        "byte_range": [row["byte_start"], row["byte_end"]],
        "raw_sha256": row["raw_sha256"],
        "receipt_sha256": row["receipt_sha256"],
        "event_index": row["event_index"],
        "source_index": row["source_index"],
        "timestamp": row["timestamp"],
        "timestamp_us": row["timestamp_us"],
        "role": row["role"],
        "semantic_sha256": row["semantic_sha256"],
        "structural": row["structural"],
    }


def _structural_tokens(event: Mapping[str, Any]) -> dict[str, list[str]]:
    found: dict[str, set[str]] = {
        "commit": set(),
        "path": set(),
        "task_uuid": set(),
    }

    def visit(value: object, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                visit(child, str(child_key).casefold())
            return
        if isinstance(value, list):
            for child in value:
                visit(child, key)
            return
        if not isinstance(value, str):
            return
        text = value.strip()
        if key in {"commit", "commit_sha", "commit_hash", "sha"} and re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}", text
        ):
            found["commit"].add(text)
        elif key in {"task_id", "task_uuid"} and re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
            text.casefold(),
        ):
            found["task_uuid"].add(text.casefold())
        elif key in {"path", "file_path", "filepath"} and text:
            normalized = Path(text).as_posix().rstrip("/")
            if normalized and "\x00" not in normalized:
                found["path"].add(normalized)

    visit(event)
    return {kind: sorted(values) for kind, values in found.items()}


def _events(raw_dir: Path) -> list[dict[str, Any]]:
    raw_store = RawStore(raw_dir, mode="v2")
    rows: list[dict[str, Any]] = []
    for unit, raw in raw_store.iter_segment_bytes():
        commit = unit.commit
        if commit is None or unit.sha256 is None or unit.captured_at is None:
            raise DistillationError("Raw v2 unit has no committed receipt")
        if raw_store.is_archived_legacy_markdown(unit, raw):
            continue
        receipt_sha256 = canonical_json.canonical_json_sha256_strict(commit.to_dict())
        try:
            spans = committed_event_spans(raw, commit.record_count)
        except RawSegmentCorrupt as exc:
            raise DistillationError(str(exc)) from exc
        for event_index, (start, encoded) in enumerate(spans):
            try:
                event = json.loads(encoded)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise DistillationError("committed Raw event is invalid") from exc
            if not isinstance(event, dict):
                raise DistillationError("committed Raw event is not an object")
            role, text = _event_semantics(commit.host, event)
            if role not in {"user", "assistant", "tool"}:
                continue
            timestamp, timestamp_us = _timestamp(
                event.get("timestamp"), commit.captured_at
            )
            rows.append(
                {
                    "host": commit.host,
                    "session_key": commit.session_key,
                    "session_cluster_id": hashlib.sha256(
                        f"{commit.host}\0{commit.session_key}".encode()
                    ).hexdigest(),
                    "session_id_sha256": hashlib.sha256(
                        str(commit.session_id or commit.session_key).encode()
                    ).hexdigest(),
                    "raw_id": unit.raw_id,
                    "raw_sha256": unit.sha256,
                    "receipt_sha256": receipt_sha256,
                    "event_index": event_index,
                    "source_index": commit.after_line + event_index + 1,
                    "byte_start": start,
                    "byte_end": start + len(encoded),
                    "timestamp": timestamp,
                    "timestamp_us": timestamp_us,
                    "role": role,
                    "text": text,
                    "semantic_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "structural": _structural_tokens(event),
                }
            )
    rows.sort(key=lambda row: (row["host"], row["session_key"], row["source_index"]))
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        key = (row["host"], row["session_key"], row["source_index"])
        if key in seen:
            raise DistillationError("overlapping committed Raw source intervals")
        seen.add(key)
    return rows


def _prompt_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha1(normalized.encode()).hexdigest()[:16]


def _exposure_map(root: Path) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    path = store.distillation_dir(root) / "exposure-receipts.jsonl"
    result: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in store.read_chain(path):
        binding = {
            key: row.get(key)
            for key in (
                "decision_id",
                "host",
                "session_id_sha256",
                "query_semantic_sha256",
                "policy_id",
                "candidate_ids",
                "candidate_refs_sha256",
                "candidate_pool_refs_sha256",
                "candidate_feature_snapshot_sha256",
                "runtime_observation_sha256",
                "render_sha256",
                "renderer_revision",
                "context_style",
                "candidate_snapshot_sha256",
                "observed_at",
            )
        }
        if row.get("kind") != "prospective-exact-exposure-v1" or row.get(
            "binding_sha256"
        ) != canonical_json.canonical_json_sha256_strict(binding):
            continue
        artifact_id = row.get("exposure_artifact_id")
        try:
            artifact = store.read_sealed(
                store.distillation_dir(root) / "exposures" / f"{artifact_id}.json",
                schema="chronovisor.recall-exact-exposure.v1",
            )
        except store.DistillationStoreError:
            continue
        if artifact.get("artifact_id") != artifact_id or any(
            artifact.get(key) != value for key, value in binding.items()
        ):
            continue
        result[
            (
                str(row["host"]),
                str(row["session_id_sha256"]),
                str(row["query_semantic_sha256"]),
            )
        ].append(
            {
                "decision_id": row["decision_id"],
                "policy_id": row["policy_id"],
                "candidate_ids": row["candidate_ids"],
                "candidate_snapshot_sha256": row["candidate_snapshot_sha256"],
                "candidate_feature_snapshot_sha256": row[
                    "candidate_feature_snapshot_sha256"
                ],
                "observed_at": row["observed_at"],
                "record_sha256": row["record_sha256"],
                "exposure_artifact_id": artifact_id,
            }
        )
    return result


def extract_rallies(
    raw_dir: Path,
    *,
    root: Path | None = None,
    max_context_bytes: int = 12_000,
    _event_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Extract deterministic rally-v1 manifests without persisting conversation text."""

    root = root or raw_dir.parent
    exposure = _exposure_map(root)
    rallies: list[dict[str, Any]] = []
    by_session: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in _event_rows if _event_rows is not None else _events(raw_dir):
        by_session[(event["host"], event["session_key"])].append(event)
    for (host, session_key), events in sorted(by_session.items()):
        if max_context_bytes <= 0:
            raise DistillationError("max_context_bytes must be positive")
        prefix: list[tuple[dict[str, Any], int]] = []
        current: dict[str, Any] | None = None

        def finish(host: str = host, session_key: str = session_key) -> None:
            nonlocal current
            if current is None:
                return
            q = current.pop("_q")
            answer_refs = current.pop("_answer_refs")
            tool_refs = current.pop("_tool_refs")
            identity = {
                "schema": RALLY_SCHEMA,
                "host": host,
                "session_key": session_key,
                "raw_id": q["raw_id"],
                "event_index": q["event_index"],
                "raw_sha256": q["raw_sha256"],
            }
            rally_id = canonical_json.canonical_json_sha256_strict(identity)
            possible_receipts = exposure.get(
                (host, current["session_id_sha256"], current["query_sha256"]), []
            )
            answer_end_us = max(
                (int(ref["timestamp_us"]) for ref in answer_refs), default=-1
            )
            receipt_rows = [
                row
                for row in possible_receipts
                if int(current["as_of_us"])
                <= _timestamp(row["observed_at"], row["observed_at"])[1]
                <= answer_end_us
            ]
            exposure_ambiguous = len(receipt_rows) > 1
            if len(receipt_rows) != 1:
                receipt_rows = []
            has_answer = bool(answer_refs)
            has_exposure = bool(receipt_rows)
            current.update(
                {
                    "schema": RALLY_SCHEMA,
                    "boundary_revision": "rally-v1",
                    "rally_id": rally_id,
                    "query_ref": _event_ref(q),
                    "context_refs": list(current.pop("_context_refs")),
                    "actual_answer_refs": answer_refs,
                    "tool_refs": tool_refs,
                    "exposure_receipts": receipt_rows,
                    "eligibility": {
                        "relevance": True,
                        "answer_utility": has_answer and has_exposure,
                        "reason": "eligible"
                        if has_answer and has_exposure
                        else "missing_answer"
                        if not has_answer
                        else "ambiguous_exact_exposure"
                        if exposure_ambiguous
                        else "missing_exact_exposure",
                    },
                }
            )
            rallies.append(current)
            current = None

        for event in events:
            ref = _event_ref(event)
            if event["role"] == "user" and event["text"].strip():
                finish()
                selected: list[dict[str, Any]] = []
                used_bytes = 0
                for prefix_ref, semantic_bytes in reversed(prefix):
                    if used_bytes + semantic_bytes > max_context_bytes:
                        break
                    selected.append(prefix_ref)
                    used_bytes += semantic_bytes
                selected.reverse()
                full_refs = [prefix_ref for prefix_ref, _ in prefix]
                current = {
                    "host": host,
                    "session_cluster_id": event["session_cluster_id"],
                    "session_id_sha256": event["session_id_sha256"],
                    "as_of": event["timestamp"],
                    "as_of_us": event["timestamp_us"],
                    "source_index": event["source_index"],
                    "query_sha256": event["semantic_sha256"],
                    "prompt_hash": _prompt_hash(event["text"]),
                    "_q": event,
                    "_context_refs": selected,
                    "context_suffix_bytes": used_bytes,
                    "full_context": {
                        "event_count": len(prefix),
                        "refs_sha256": canonical_json.canonical_json_sha256_strict(
                            full_refs
                        ),
                        "first_ref": full_refs[0] if full_refs else None,
                        "last_ref": full_refs[-1] if full_refs else None,
                    },
                    "_answer_refs": [],
                    "_tool_refs": [],
                }
            elif (
                current is not None
                and event["role"] == "assistant"
                and event["text"].strip()
            ):
                current["_answer_refs"].append(ref)
            elif current is not None and event["role"] == "tool":
                current["_tool_refs"].append(ref)
            prefix.append((ref, len(event["text"].encode("utf-8"))))
        finish()
    return sorted(rallies, key=lambda row: (row["as_of_us"], row["rally_id"]))


def build_historical_index(
    raw_dir: Path,
    path: Path,
    *,
    _event_rows: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    atoms = []
    for event in _event_rows if _event_rows is not None else _events(raw_dir):
        if event["role"] != "assistant" or not event["text"].strip():
            continue
        ref = _event_ref(event)
        atoms.append(
            {
                "atom_id": canonical_json.canonical_json_sha256_strict(
                    {"kind": "assistant-atom-v1", "ref": ref}
                ),
                "host": event["host"],
                "session_cluster_id": event["session_cluster_id"],
                "source_index": event["source_index"],
                "timestamp_us": event["timestamp_us"],
                "text_sha256": event["semantic_sha256"],
                "ref": ref,
                "text": event["text"],
            }
        )
    return store.create_historical_index(path, atoms)


def candidate_snapshot(
    index_path: Path,
    rally: Mapping[str, Any],
    query_text: str,
    *,
    limit: int = 200,
    candidate_texts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    candidates = store.search_historical_index(
        index_path,
        query=query_text,
        as_of_us=int(rally["as_of_us"]),
        host=str(rally["host"]),
        session_cluster_id=str(rally["session_cluster_id"]),
        source_index=int(rally["source_index"]),
        limit=limit,
    )
    feature_rows = []
    query_feature_bytes = _bounded_normalized_text(
        query_text, max_bytes=2_048
    ).encode()
    for candidate in candidates:
        candidate_text = (candidate_texts or {}).get(str(candidate["text_sha256"]))
        feature_rows.append(
            {
                **candidate,
                **(
                    {
                        "feature_revision": TEXT_FEATURE_REVISION,
                        "candidate_feature_text_sha256": hashlib.sha256(
                            _bounded_normalized_text(
                                candidate_text, max_bytes=4_096
                            ).encode()
                        ).hexdigest(),
                        "features": build_text_features(query_text, candidate_text),
                    }
                    if isinstance(candidate_text, str)
                    else {}
                ),
            }
        )
    unsigned = {
        "schema": "chronovisor.recall-candidate-snapshot.v1",
        "rally_id": rally["rally_id"],
        "as_of": rally["as_of"],
        "retriever_revision": "historical-fts-v1",
        "feature_revision": TEXT_FEATURE_REVISION,
        "query_feature_text_sha256": hashlib.sha256(query_feature_bytes).hexdigest(),
        "candidates": feature_rows,
    }
    return {
        **unsigned,
        "snapshot_sha256": canonical_json.canonical_json_sha256_strict(unsigned),
    }


def teacher_assignment(
    rally_id: str, candidate_id: str, *, routes: Sequence[str] = TEACHER_ROLES
) -> dict[str, Any]:
    if len(routes) != 3 or len(set(routes)) != 3:
        raise DistillationError("exactly three distinct teacher routes are required")
    key = f"{ASSIGNMENT_REVISION}\0{rally_id}\0{candidate_id}".encode()
    probe_key = f"{PROBE_REVISION}\0{rally_id}\0{candidate_id}".encode()
    owner = routes[int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 3]
    probe = (
        int.from_bytes(hashlib.sha256(probe_key).digest()[:8], "big") % 10_000 < 1_500
    )
    return {
        "revision": ASSIGNMENT_REVISION,
        "owner": owner,
        "probe_revision": PROBE_REVISION,
        "probe": probe,
        "routes": list(routes) if probe else [owner],
    }


def _ordered_teacher_routes(
    pending: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Sequence[Mapping[str, Any]],
) -> list[str]:
    counts = {
        role: sum(row.get("route") == role for row in labels) for role in TEACHER_ROLES
    }
    return sorted(pending, key=lambda role: (counts.get(role, 0), role))


def _is_counterfactual_turn(
    teacher_calls: int, counterfactual_calls: int, *, available: bool
) -> bool:
    return available and teacher_calls >= 3 * (counterfactual_calls + 1)


def adjudicate_label(
    verdict: str,
    *,
    closed_predicate: str | None,
    reason: str = "",
    dimension: str = "answer_utility",
) -> dict[str, Any]:
    allowed = RELEVANCE_LABELS if dimension == "relevance" else UTILITY_LABELS
    if dimension not in {"relevance", "answer_utility"} or verdict not in allowed:
        return {
            "verdict": "uncertain",
            "authority": "reject",
            "dimension": dimension,
            "reason_code": "invalid_verdict",
            "rationale_sha256": hashlib.sha256(b"invalid_verdict").hexdigest(),
            "rationale_chars": 15,
        }
    authority = "uncertain" if verdict == "uncertain" else "teacher-only"
    reason_code = (
        reason
        if re.fullmatch(r"[a-z0-9_]{1,64}", reason)
        else "model_rationale"
        if reason
        else "none"
    )
    return {
        "verdict": verdict,
        "authority": authority,
        "dimension": dimension,
        "closed_predicate": closed_predicate or "",
        "reason_code": reason_code,
        "rationale_sha256": hashlib.sha256(reason.encode()).hexdigest(),
        "rationale_chars": len(reason),
    }


def _append_unique_receipt(
    path: Path,
    payload: Mapping[str, Any],
    *,
    nonblocking: bool,
    prepare: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(nonblocking, bool):
        raise DistillationError("receipt nonblocking flag is invalid")
    if not nonblocking:
        return store.append_chain_unique(
            path,
            {**payload, **(prepare() if prepare else {})},
            unique_field="decision_id",
            binding_field="idempotency_sha256",
        )
    lock = store.acquire_nonblocking_lock(path.with_suffix(path.suffix + ".lock"))
    if lock is None:
        return {"status": "deferred", "reason": "receipt_ledger_busy"}
    try:
        try:
            return store.append_chain_unique_locked(
                path,
                {**payload, **(prepare() if prepare else {})},
                unique_field="decision_id",
                binding_field="idempotency_sha256",
            )
        except store.DistillationStoreBusy:
            return {"status": "deferred", "reason": "receipt_ledger_busy"}
    finally:
        store.release_lock(lock)


def record_exposure(
    *,
    decision_id: str,
    host: str,
    session_id: str,
    prompt_hash: str,
    policy_id: str,
    candidate_ids: Sequence[str],
    candidate_snapshot_sha256: str,
    observed_at: str,
    decision_latency_ms: float | None = None,
    timed_out: bool | None = None,
    error_code: str = "",
    nonblocking: bool = False,
    root: Path | None = None,
) -> dict[str, Any]:
    if not all(
        isinstance(value, str) and value
        for value in (decision_id, host, session_id, prompt_hash, policy_id)
    ):
        raise DistillationError("exposure identity fields must be non-empty")
    _timestamp(observed_at, observed_at)
    if not re.fullmatch(r"[0-9a-f]{64}", candidate_snapshot_sha256):
        raise DistillationError("candidate snapshot digest is invalid")
    if (
        (decision_latency_ms is None) != (timed_out is None)
        or re.fullmatch(r"[a-z0-9_]{0,64}", error_code) is None
        or (
            decision_latency_ms is not None
            and (
                isinstance(decision_latency_ms, bool)
                or not isinstance(decision_latency_ms, (int, float))
                or not math.isfinite(float(decision_latency_ms))
                or not 0 <= float(decision_latency_ms) <= 60_000
                or not isinstance(timed_out, bool)
            )
        )
    ):
        raise DistillationError("page exposure runtime observation is invalid")
    ids = list(candidate_ids)
    if len(ids) != len(set(ids)) or any(
        not isinstance(item, str) or not item for item in ids
    ):
        raise DistillationError("candidate ids must be unique non-empty strings")
    root = root or CHRONOVISOR_ROOT
    runtime_observation = (
        {
            "decision": "read" if ids else "none",
            "selected_count": len(ids),
            "latency_ms": float(decision_latency_ms),
            "timed_out": timed_out,
            "error_code": error_code,
        }
        if decision_latency_ms is not None
        else None
    )
    binding = {
        "decision_id": decision_id,
        "host": host,
        "session_id_sha256": hashlib.sha256(session_id.encode()).hexdigest(),
        "prompt_hash": prompt_hash,
        "policy_id": policy_id,
        "candidate_ids": ids,
        "candidate_snapshot_sha256": candidate_snapshot_sha256,
        "runtime_observation_sha256": canonical_json.canonical_json_sha256_strict(
            runtime_observation
        ),
        "observed_at": observed_at,
    }
    return _append_unique_receipt(
        store.distillation_dir(root) / "exposure-receipts.jsonl",
        {
            "kind": "prospective-page-exposure",
            **binding,
            "runtime_observation": runtime_observation,
            "binding_sha256": canonical_json.canonical_json_sha256_strict(binding),
            "idempotency_sha256": canonical_json.canonical_json_sha256_strict(
                {key: value for key, value in binding.items() if key != "observed_at"}
            ),
        },
        nonblocking=nonblocking,
    )


def _validate_exposure_policy_identity(root: Path, policy_id: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", policy_id) is None:
        raise DistillationError("exact exposure policy identity is invalid")
    locations = (
        ("policies", POLICY_SCHEMA),
        ("baselines", BASELINE_SCHEMA),
    )
    for directory, schema in locations:
        try:
            artifact = store.read_sealed(
                store.distillation_dir(root) / directory / f"{policy_id}.json",
                schema=schema,
            )
        except store.DistillationStoreError:
            continue
        if artifact.get("artifact_id") == policy_id:
            return
    raise DistillationError("exact exposure policy identity is not sealed")


def load_capture_policy_identity(root: Path | None = None) -> str:
    """Return the sealed incumbent/baseline identity for capture-only receipts."""

    root = root or CHRONOVISOR_ROOT
    if not _enabled_for_root(root):
        return ""
    try:
        policy_id = str(store.read_pointer(root, "active")["policy_id"])
        _validate_exposure_policy_identity(root, policy_id)
        return policy_id
    except (KeyError, store.DistillationStoreError, DistillationError):
        pass
    try:
        state = _read_worker_state(root)
        policy_id = str(state.get("baseline_artifact_id") or "")
        _validate_exposure_policy_identity(root, policy_id)
    except (store.DistillationStoreError, DistillationError):
        return ""
    return policy_id


def record_exact_exposure(
    *,
    decision_id: str,
    host: str,
    session_id: str,
    query_semantic_sha256: str,
    policy_id: str,
    candidate_refs: Sequence[Mapping[str, Any]],
    candidate_feature_snapshot: Sequence[Mapping[str, Any]] = (),
    candidate_pool_refs: Sequence[Mapping[str, Any]] = (),
    render_sha256: str,
    candidate_snapshot_sha256: str,
    observed_at: str,
    renderer_revision: str = "recall-card-v1",
    context_style: str = "default",
    decision_latency_ms: float | None = None,
    timed_out: bool | None = None,
    nonblocking: bool = False,
    root: Path | None = None,
) -> dict[str, Any]:
    digests = (query_semantic_sha256, render_sha256, candidate_snapshot_sha256)
    if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in digests):
        raise DistillationError("exact exposure digest is invalid")
    if not all(
        isinstance(value, str) and value
        for value in (
            decision_id,
            host,
            session_id,
            policy_id,
            renderer_revision,
            context_style,
        )
    ) or any(
        re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", value) is None
        for value in (renderer_revision, context_style)
    ):
        raise DistillationError("exact exposure identity is invalid")
    _, observed_us = _timestamp(observed_at, observed_at)
    refs: list[dict[str, Any]] = []
    ids: list[str] = []
    for candidate in candidate_refs:
        candidate_id = candidate.get("candidate_id")
        evidence = candidate.get("evidence_refs")
        page_id = candidate.get("page_id")
        rendered_context = candidate.get("rendered_context")
        content_sha256 = candidate.get("content_sha256")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise DistillationError("exact exposure candidate binding is invalid")
        if isinstance(evidence, list) and evidence:
            if re.fullmatch(r"[0-9a-f]{64}", str(content_sha256)) is None:
                raise DistillationError("exact exposure Raw content digest is invalid")
            for ref in evidence:
                if (
                    not isinstance(ref, dict)
                    or not isinstance(ref.get("raw_id"), str)
                    or Path(ref["raw_id"]).name != ref["raw_id"]
                    or not isinstance(ref.get("byte_range"), list)
                    or len(ref["byte_range"]) != 2
                    or re.fullmatch(r"[0-9a-f]{64}", str(ref.get("raw_sha256"))) is None
                    or re.fullmatch(r"[0-9a-f]{64}", str(ref.get("receipt_sha256")))
                    is None
                ):
                    raise DistillationError("exact exposure Raw ref is invalid")
            exact_source = {
                "content_sha256": content_sha256,
                "evidence_refs": evidence,
            }
        elif (
            isinstance(page_id, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,239}", page_id)
            and re.fullmatch(r"[0-9a-f]{64}", str(candidate.get("page_content_sha256")))
            is not None
            and re.fullmatch(
                r"[0-9a-f]{64}", str(candidate.get("rendered_context_sha256"))
            )
            is not None
            and isinstance(rendered_context, str)
            and rendered_context
            and len(rendered_context.encode()) <= 12_000
            and hashlib.sha256(rendered_context.encode()).hexdigest()
            == candidate.get("rendered_context_sha256")
        ):
            exact_source = {
                "page_id": page_id,
                "page_content_sha256": candidate["page_content_sha256"],
                "rendered_context": rendered_context,
                "rendered_context_sha256": candidate["rendered_context_sha256"],
            }
        else:
            raise DistillationError("exact exposure has no verifiable source version")
        ids.append(candidate_id)
        refs.append(
            {
                "candidate_id": candidate_id,
                **exact_source,
            }
        )
    if len(ids) != len(set(ids)):
        raise DistillationError("exact exposure candidate ids are duplicated")
    if len(candidate_pool_refs) > 12:
        raise DistillationError("exact exposure candidate pool is too large")
    pool_rows: list[dict[str, Any]] = []
    pool_ids: set[str] = set()
    for candidate in candidate_pool_refs:
        candidate_id = candidate.get("candidate_id")
        page_id = candidate.get("page_id")
        rendered_context = candidate.get("rendered_context")
        rendered_sha256 = candidate.get("rendered_context_sha256")
        selected = candidate.get("selected")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in pool_ids
            or not isinstance(selected, bool)
            or not isinstance(page_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,239}", page_id) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(candidate.get("page_content_sha256")))
            is None
            or not isinstance(rendered_context, str)
            or not rendered_context
            or len(rendered_context.encode()) > 12_000
            or hashlib.sha256(rendered_context.encode()).hexdigest() != rendered_sha256
        ):
            raise DistillationError("exact exposure pool candidate is invalid")
        pool_ids.add(candidate_id)
        pool_rows.append(
            {
                "candidate_id": candidate_id,
                "selected": selected,
                "page_id": page_id,
                "page_content_sha256": candidate["page_content_sha256"],
                "rendered_context": rendered_context,
                "rendered_context_sha256": rendered_sha256,
            }
        )
    if pool_rows and {
        row["candidate_id"] for row in pool_rows if row["selected"]
    } != set(ids):
        raise DistillationError("exact exposure selected pool does not match E_t")
    if pool_rows:
        selected_pool = {
            str(row["candidate_id"]): row for row in pool_rows if row["selected"]
        }
        for ref in refs:
            if "page_id" not in ref:
                continue
            pool_ref = selected_pool.get(str(ref["candidate_id"]))
            if pool_ref is None or any(
                pool_ref.get(key) != ref.get(key)
                for key in (
                    "page_id",
                    "page_content_sha256",
                    "rendered_context_sha256",
                )
            ):
                raise DistillationError(
                    "exact exposure selected source version does not match pool"
                )
    if len(candidate_feature_snapshot) > 200:
        raise DistillationError("exact exposure feature snapshot is too large")
    feature_rows: list[dict[str, Any]] = []
    feature_ids: set[str] = set()
    for row in candidate_feature_snapshot:
        candidate_id = row.get("candidate_id")
        values = row.get("features")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in feature_ids
            or not isinstance(values, Mapping)
        ):
            raise DistillationError("exact exposure feature binding is invalid")
        canonical_features = build_fast_features(values)
        if set(values) != set(FAST_FEATURE_KEYS) or dict(values) != canonical_features:
            raise DistillationError("exact exposure features are not canonical")
        feature_ids.add(candidate_id)
        feature_rows.append(
            {"candidate_id": candidate_id, "features": canonical_features}
        )
    if not set(ids).issubset(feature_ids) and feature_rows:
        raise DistillationError("selected candidates are missing exact features")
    if pool_rows and pool_ids != feature_ids:
        raise DistillationError("candidate pool and feature snapshot do not match")
    if (decision_latency_ms is None) != (timed_out is None) or (
        decision_latency_ms is not None
        and (
            isinstance(decision_latency_ms, bool)
            or not isinstance(decision_latency_ms, (int, float))
            or not math.isfinite(float(decision_latency_ms))
            or not 0 <= float(decision_latency_ms) <= 60_000
            or not isinstance(timed_out, bool)
        )
    ):
        raise DistillationError("exact exposure runtime observation is invalid")
    runtime_observation = (
        {
            "decision": "read" if ids else "none",
            "selected_count": len(ids),
            "evaluated_count": len(feature_rows),
            "latency_ms": float(decision_latency_ms),
            "timed_out": timed_out,
        }
        if decision_latency_ms is not None
        else None
    )
    refs_sha256 = canonical_json.canonical_json_sha256_strict(refs)
    pool_sha256 = canonical_json.canonical_json_sha256_strict(pool_rows)
    features_sha256 = canonical_json.canonical_json_sha256_strict(feature_rows)
    runtime_observation_sha256 = canonical_json.canonical_json_sha256_strict(
        runtime_observation
    )
    session_id_sha256 = hashlib.sha256(session_id.encode()).hexdigest()
    binding = {
        "decision_id": decision_id,
        "host": host,
        "session_id_sha256": session_id_sha256,
        "query_semantic_sha256": query_semantic_sha256,
        "policy_id": policy_id,
        "candidate_ids": ids,
        "candidate_refs_sha256": refs_sha256,
        "candidate_pool_refs_sha256": pool_sha256,
        "candidate_feature_snapshot_sha256": features_sha256,
        "runtime_observation_sha256": runtime_observation_sha256,
        "render_sha256": render_sha256,
        "renderer_revision": renderer_revision,
        "context_style": context_style,
        "candidate_snapshot_sha256": candidate_snapshot_sha256,
        "observed_at": observed_at,
    }
    root = root or CHRONOVISOR_ROOT
    _validate_exposure_policy_identity(root, policy_id)
    def prepare() -> Mapping[str, Any]:
        _, _, artifact = store.write_immutable(
            store.distillation_dir(root) / "exposures",
            {
                "kind": "exact-rendered-exposure",
                **binding,
                "candidate_refs": refs,
                "candidate_pool_refs": pool_rows,
                "candidate_feature_snapshot": feature_rows,
                "runtime_observation": runtime_observation,
            },
            schema="chronovisor.recall-exact-exposure.v1",
            nonblocking=nonblocking,
        )
        return {"exposure_artifact_id": artifact["artifact_id"]}

    return _append_unique_receipt(
        store.distillation_dir(root) / "exposure-receipts.jsonl",
        {
            "kind": "prospective-exact-exposure-v1",
            **binding,
            "binding_sha256": canonical_json.canonical_json_sha256_strict(binding),
            "idempotency_sha256": canonical_json.canonical_json_sha256_strict(
                {key: value for key, value in binding.items() if key != "observed_at"}
            ),
        },
        nonblocking=nonblocking,
        prepare=prepare,
    )


def record_shadow_observation(
    *,
    decision_id: str,
    host: str,
    session_id: str,
    query_semantic_sha256: str,
    policy_id: str,
    incumbent_policy_id: str,
    served_policy_id: str,
    selected_candidate_ids: Sequence[str],
    incumbent_selected_candidate_ids: Sequence[str],
    paired_eligible: bool,
    candidate_feature_snapshot: Sequence[Mapping[str, Any]],
    candidate_pool_refs: Sequence[Mapping[str, Any]],
    observed_at: str,
    decision_latency_ms: float,
    timed_out: bool,
    error_code: str = "",
    nonblocking: bool = False,
    root: Path | None = None,
) -> dict[str, Any]:
    """Record non-causal shadow scoring without persisting candidate text."""

    root = root or CHRONOVISOR_ROOT
    if not all(
        isinstance(value, str) and value for value in (decision_id, host, session_id)
    ):
        raise DistillationError("shadow observation identity is invalid")
    if (
        re.fullmatch(r"[0-9a-f]{64}", query_semantic_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", policy_id) is None
        or re.fullmatch(r"[0-9a-f]{64}", incumbent_policy_id) is None
        or re.fullmatch(r"[0-9a-f]{64}", served_policy_id) is None
        or re.fullmatch(r"[a-z0-9_]{0,64}", error_code) is None
        or isinstance(decision_latency_ms, bool)
        or not isinstance(decision_latency_ms, (int, float))
        or not math.isfinite(float(decision_latency_ms))
        or not 0 <= float(decision_latency_ms) <= 60_000
        or not isinstance(timed_out, bool)
        or not isinstance(paired_eligible, bool)
    ):
        raise DistillationError("shadow observation runtime binding is invalid")
    _, observed_us = _timestamp(observed_at, observed_at)
    selected = list(selected_candidate_ids)
    incumbent_selected = list(incumbent_selected_candidate_ids)
    if any(
        len(values) != len(set(values))
        or any(
            not isinstance(candidate_id, str) or not candidate_id
            for candidate_id in values
        )
        for values in (selected, incumbent_selected)
    ):
        raise DistillationError("shadow selected candidates are invalid")
    if len(candidate_pool_refs) > 12 or len(candidate_feature_snapshot) > 12:
        raise DistillationError("shadow candidate pool is too large")
    pool_rows: list[dict[str, Any]] = []
    pool_ids: set[str] = set()
    for row in candidate_pool_refs:
        candidate_id = row.get("candidate_id")
        page_id = row.get("page_id")
        row_selected = row.get("selected")
        page_sha256 = row.get("page_content_sha256")
        rendered_sha256 = row.get("rendered_context_sha256")
        rendered = row.get("rendered_context")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in pool_ids
            or not isinstance(page_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,239}", page_id) is None
            or not isinstance(row_selected, bool)
            or re.fullmatch(r"[0-9a-f]{64}", str(page_sha256)) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(rendered_sha256)) is None
            or (
                rendered is not None
                and (
                    not isinstance(rendered, str)
                    or len(rendered.encode()) > 12_000
                    or hashlib.sha256(rendered.encode()).hexdigest() != rendered_sha256
                )
            )
        ):
            raise DistillationError("shadow candidate source binding is invalid")
        pool_ids.add(candidate_id)
        pool_rows.append(
            {
                "candidate_id": candidate_id,
                "selected": row_selected,
                "page_id": page_id,
                "page_content_sha256": page_sha256,
                "rendered_context_sha256": rendered_sha256,
            }
        )
    if {row["candidate_id"] for row in pool_rows if row["selected"]} != set(selected):
        raise DistillationError("shadow selected pool does not match decision")
    if not set(incumbent_selected).issubset(pool_ids):
        raise DistillationError("shadow incumbent decision is outside candidate pool")
    feature_rows: list[dict[str, Any]] = []
    feature_ids: set[str] = set()
    for row in candidate_feature_snapshot:
        candidate_id = row.get("candidate_id")
        values = row.get("features")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in feature_ids
            or not isinstance(values, Mapping)
        ):
            raise DistillationError("shadow feature binding is invalid")
        features = build_fast_features(values)
        if set(values) != set(FAST_FEATURE_KEYS) or dict(values) != features:
            raise DistillationError("shadow features are not canonical")
        feature_ids.add(candidate_id)
        feature_rows.append({"candidate_id": candidate_id, "features": features})
    if feature_ids != pool_ids:
        raise DistillationError("shadow pool and features do not match")
    qualified = load_policy_observation_context(session_id, root)
    if (
        qualified.get("candidate_policy_id") != policy_id
        or qualified.get("incumbent_policy_id") != incumbent_policy_id
        or qualified.get("served_policy_id") != served_policy_id
    ):
        raise DistillationError("paired policy observation is not qualified")
    _, stage_started_us = _timestamp(
        qualified.get("stage_started_at"), observed_at
    )
    if observed_us < stage_started_us:
        raise DistillationError("paired observation predates rollout stage")
    observation = {
        "decision": "read" if selected else "none",
        "selected_count": len(selected),
        "evaluated_count": len(feature_rows),
        "latency_ms": float(decision_latency_ms),
        "timed_out": timed_out,
        "error_code": error_code,
    }
    binding = {
        "decision_id": decision_id,
        "host": host,
        "session_id_sha256": hashlib.sha256(session_id.encode()).hexdigest(),
        "query_semantic_sha256": query_semantic_sha256,
        "policy_id": policy_id,
        "incumbent_policy_id": incumbent_policy_id,
        "served_policy_id": served_policy_id,
        "stage": qualified["stage"],
        "stage_started_at": qualified["stage_started_at"],
        "qualified_run_id": qualified["qualified_run_id"],
        "selected_candidate_ids": selected,
        "incumbent_selected_candidate_ids": incumbent_selected,
        "paired_eligible": paired_eligible,
        "candidate_pool_sha256": canonical_json.canonical_json_sha256_strict(pool_rows),
        "candidate_feature_snapshot_sha256": canonical_json.canonical_json_sha256_strict(
            feature_rows
        ),
        "runtime_observation_sha256": canonical_json.canonical_json_sha256_strict(
            observation
        ),
        "observed_at": observed_at,
    }
    def prepare() -> Mapping[str, Any]:
        _, _, artifact = store.write_immutable(
            store.distillation_dir(root) / "shadow-observations",
            {
                "kind": "non-causal-shadow-observation",
                **binding,
                "candidate_pool_refs": pool_rows,
                "candidate_feature_snapshot": feature_rows,
                "runtime_observation": observation,
            },
            schema=SHADOW_OBSERVATION_SCHEMA,
            nonblocking=nonblocking,
        )
        return {"shadow_observation_artifact_id": artifact["artifact_id"]}

    return _append_unique_receipt(
        store.distillation_dir(root) / "shadow-observation-receipts.jsonl",
        {
            "kind": "shadow-policy-observation",
            **binding,
            "binding_sha256": canonical_json.canonical_json_sha256_strict(binding),
            "idempotency_sha256": canonical_json.canonical_json_sha256_strict(
                {key: value for key, value in binding.items() if key != "observed_at"}
            ),
        },
        nonblocking=nonblocking,
        prepare=prepare,
    )


def record_closed_outcome(
    *,
    outcome_id: str,
    exposure_artifact_id: str,
    candidate_id: str,
    candidate_version_sha256: str,
    kind: str,
    status: str,
    evidence_sha256: str,
    observed_at: str,
    root: Path | None = None,
) -> dict[str, Any]:
    """Seal one closed, version-bound outcome; never accept prose evidence."""

    allowed = {
        "test": {"passed", "failed"},
        "rollback": {"rolled_back", "not_rolled_back"},
        "correction": {"accepted", "rejected"},
    }
    if (
        re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", outcome_id) is None
        or not isinstance(candidate_id, str)
        or not candidate_id
        or kind not in allowed
        or status not in allowed[kind]
        or re.fullmatch(r"[0-9a-f]{64}", exposure_artifact_id) is None
        or re.fullmatch(r"[0-9a-f]{64}", candidate_version_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", evidence_sha256) is None
    ):
        raise DistillationError("closed outcome binding is invalid")
    _timestamp(observed_at, observed_at)
    root = root or CHRONOVISOR_ROOT
    exposure = store.read_sealed(
        store.distillation_dir(root) / "exposures" / f"{exposure_artifact_id}.json",
        schema="chronovisor.recall-exact-exposure.v1",
    )
    versions = {
        str(row.get("candidate_id")): str(
            row.get("page_content_sha256") or row.get("content_sha256") or ""
        )
        for row in exposure.get("candidate_refs", [])
        if isinstance(row, Mapping)
    }
    if versions.get(candidate_id) != candidate_version_sha256:
        raise DistillationError("closed outcome candidate version is not exposed")
    binding = {
        "outcome_id": outcome_id,
        "exposure_artifact_id": exposure_artifact_id,
        "candidate_id": candidate_id,
        "candidate_version_sha256": candidate_version_sha256,
        "kind": kind,
        "status": status,
        "evidence_sha256": evidence_sha256,
        "observed_at": observed_at,
    }
    _, _, artifact = store.write_immutable(
        store.distillation_dir(root) / "outcomes",
        {"kind": "closed-outcome", "authority": "capture-only", **binding},
        schema=OUTCOME_SCHEMA,
    )
    return store.append_chain_unique(
        store.distillation_dir(root) / "outcome-receipts.jsonl",
        {
            "kind": "closed-outcome-receipt",
            **binding,
            "outcome_artifact_id": artifact["artifact_id"],
            "binding_sha256": canonical_json.canonical_json_sha256_strict(binding),
        },
        unique_field="outcome_id",
        binding_field="binding_sha256",
    )


def record_authenticated_exact_correction_veto(
    *,
    decision_id: str,
    correction_id: str,
    candidate_id: str,
    page_id: str,
    preimage_bytes: bytes,
    postimage_bytes: bytes,
    readback_bytes: bytes,
    cas_status: str,
    observed_at: str,
    root: Path | None = None,
) -> dict[str, Any]:
    """Bind a negative correction veto to one exact selected Page version."""

    if (
        re.fullmatch(r"[A-Za-z0-9._:-]{1,160}", decision_id) is None
        or re.fullmatch(r"[A-Za-z0-9._:-]{1,160}", correction_id) is None
        or not candidate_id
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,239}", page_id) is None
        or cas_status not in {"applied", "already_applied"}
        or any(
            not isinstance(value, bytes) or not value or len(value) > 2_000_000
            for value in (preimage_bytes, postimage_bytes, readback_bytes)
        )
    ):
        raise DistillationError("authenticated correction veto binding is invalid")
    _, observed_us = _timestamp(observed_at, observed_at)
    root = root or CHRONOVISOR_ROOT
    receipts = [
        receipt
        for values in _exposure_map(root).values()
        for receipt in values
        if receipt.get("decision_id") == decision_id
    ]
    if len(receipts) != 1:
        raise DistillationError("correction veto exact exposure is not unique")
    receipt = receipts[0]
    exposure_id = str(receipt["exposure_artifact_id"])
    exposure = store.read_sealed(
        store.distillation_dir(root) / "exposures" / f"{exposure_id}.json",
        schema="chronovisor.recall-exact-exposure.v1",
    )
    selected = exposure.get("candidate_refs")
    if not isinstance(selected, list) or len(selected) != 1 or not isinstance(
        selected[0], Mapping
    ):
        raise DistillationError("correction veto requires one selected candidate")
    candidate = selected[0]
    preimage_sha256 = hashlib.sha256(preimage_bytes).hexdigest()
    postimage_sha256 = hashlib.sha256(postimage_bytes).hexdigest()
    readback_sha256 = hashlib.sha256(readback_bytes).hexdigest()
    if (
        candidate.get("candidate_id") != candidate_id
        or candidate.get("page_id") != page_id
        or candidate.get("page_content_sha256") != preimage_sha256
        or postimage_sha256 != readback_sha256
        or postimage_sha256 == preimage_sha256
        or observed_us
        < _timestamp(str(exposure.get("observed_at") or ""), observed_at)[1]
    ):
        raise DistillationError("correction veto version/readback binding is invalid")
    policy_id = str(exposure.get("policy_id") or "")
    _validate_exposure_policy_identity(root, policy_id)
    binding = {
        "decision_id": decision_id,
        "correction_id": correction_id,
        "exposure_artifact_id": exposure_id,
        "policy_id": policy_id,
        "candidate_id": candidate_id,
        "page_id": page_id,
        "preimage_sha256": preimage_sha256,
        "postimage_sha256": postimage_sha256,
        "cas_status": cas_status,
        "observed_at": observed_at,
        "producer_revision": "content-correction-cas-readback-v1",
    }
    _, _, artifact = store.write_immutable(
        store.distillation_dir(root) / "negative-vetoes",
        {"kind": "authenticated-exact-correction-veto", **binding},
        schema=VETO_SCHEMA,
    )
    return store.append_chain_unique(
        store.distillation_dir(root) / "negative-veto-receipts.jsonl",
        {
            "kind": "authenticated-negative-veto",
            **binding,
            "veto_artifact_id": artifact["artifact_id"],
            "binding_sha256": canonical_json.canonical_json_sha256_strict(binding),
        },
        unique_field="correction_id",
        binding_field="binding_sha256",
    )


def _resolve_closed_outcome(
    root: Path,
    receipt_id: object,
    *,
    exposure_artifact_id: str,
    candidate_id: str,
) -> None:
    if (
        not isinstance(receipt_id, str)
        or re.fullmatch(r"[0-9a-f]{64}", receipt_id) is None
    ):
        return None
    rows = store.read_chain(store.distillation_dir(root) / "outcome-receipts.jsonl")
    receipt = next(
        (row for row in rows if row.get("record_sha256") == receipt_id), None
    )
    if (
        receipt is None
        or receipt.get("exposure_artifact_id") != exposure_artifact_id
        or receipt.get("candidate_id") != candidate_id
    ):
        return None
    try:
        artifact = store.read_sealed(
            store.distillation_dir(root)
            / "outcomes"
            / f"{receipt['outcome_artifact_id']}.json",
            schema=OUTCOME_SCHEMA,
        )
    except (KeyError, store.DistillationStoreError):
        return None
    binding = {
        key: receipt.get(key)
        for key in (
            "outcome_id",
            "exposure_artifact_id",
            "candidate_id",
            "candidate_version_sha256",
            "kind",
            "status",
            "evidence_sha256",
            "observed_at",
        )
    }
    if (
        artifact.get("artifact_id") != receipt.get("outcome_artifact_id")
        or any(artifact.get(key) != value for key, value in binding.items())
        or receipt.get("binding_sha256")
        != canonical_json.canonical_json_sha256_strict(binding)
    ):
        return None
    return None


def _bounded_normalized_text(value: str, *, max_bytes: int) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    collapsed = " ".join(
        "".join(
            character if character.isalnum() else " " for character in normalized
        ).split()
    )
    encoded = collapsed.encode("utf-8")[:max_bytes]
    return encoded.decode("utf-8", errors="ignore").strip()


def _text_chargrams(value: str) -> frozenset[str]:
    grams: set[str] = set()
    for token in value.split():
        if len(token) < 3:
            grams.add(token)
        else:
            grams.update(token[index : index + 3] for index in range(len(token) - 2))
    return frozenset(grams)


def build_text_features(query_text: str, candidate_text: str) -> dict[str, float]:
    """Build the byte-identical historical/live Recall feature contract."""

    if not isinstance(query_text, str) or not isinstance(candidate_text, str):
        raise DistillationError("text feature inputs must be strings")
    query = _text_chargrams(_bounded_normalized_text(query_text, max_bytes=2_048))
    candidate = _text_chargrams(
        _bounded_normalized_text(candidate_text, max_bytes=4_096)
    )
    overlap = len(query.intersection(candidate))
    return build_fast_features(
        query_chargram_coverage=round(overlap / len(query), 8) if query else 0.0,
        candidate_chargram_precision=(
            round(overlap / len(candidate), 8) if candidate else 0.0
        ),
    )


def build_fast_features(
    values: Mapping[str, Any] | None = None, /, **extra: Any
) -> dict[str, float]:
    merged = {**(dict(values) if values is not None else {}), **extra}
    forbidden = FORBIDDEN_LIVE_FEATURES.intersection(key.lower() for key in merged)
    unknown = set(merged).difference(FAST_FEATURE_KEYS)
    if forbidden or unknown:
        raise DistillationError(
            f"live feature payload is not whitelisted: {sorted(forbidden | unknown)}"
        )
    features: dict[str, float] = {}
    for key in FAST_FEATURE_KEYS:
        value = merged.get(key, 0.0)
        if isinstance(value, bool):
            value = float(value)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise DistillationError(f"live feature {key} must be finite numeric")
        features[key] = max(0.0, min(1.0, float(value)))
    return features


def score_fast_features(
    features: Mapping[str, Any], policy: Mapping[str, Any]
) -> float:
    clean = build_fast_features(features)
    if tuple(policy.get("feature_keys", ())) != FAST_FEATURE_KEYS:
        raise DistillationError("policy feature schema mismatch")
    weights = policy.get("weights")
    if not isinstance(weights, dict) or set(weights) != set(FAST_FEATURE_KEYS):
        raise DistillationError("policy weights are incomplete")
    try:
        total = float(policy.get("bias", 0.0)) + sum(
            float(weights[key]) * clean[key] for key in FAST_FEATURE_KEYS
        )
    except (TypeError, ValueError) as exc:
        raise DistillationError("policy weights are invalid") from exc
    return sigmoid(total)


def policy_decision(
    score: float,
    policy: Mapping[str, Any],
    *,
    runner_up_score: float = 0.0,
) -> dict[str, Any]:
    threshold = float(policy.get("threshold", 1.0))
    margin = float(policy.get("abstain_margin", 0.0))
    max_cards = int(policy.get("max_cards", 0))
    accepted = score >= threshold and score - runner_up_score >= margin
    return {
        "decision": "read" if accepted else "none",
        "max_cards": max_cards if accepted else 0,
    }


def _load_policy(policy_id: str, root: Path) -> dict[str, Any]:
    artifact = store.read_sealed(
        store.distillation_dir(root) / "policies" / f"{policy_id}.json",
        schema=POLICY_SCHEMA,
    )
    if artifact.get("artifact_id") != policy_id:
        raise DistillationError("policy identity mismatch")
    return artifact


def _enabled_for_root(root: Path) -> bool:
    config_path = root / "config.toml"
    return distillation_enabled(config_path if config_path.exists() else None)


def _read_worker_state(root: Path) -> dict[str, Any]:
    state = store.read_sealed(
        store.distillation_dir(root) / store.STATE_FILE,
        schema=store.DISTILLATION_SCHEMA,
    )
    return {
        key: value
        for key, value in state.items()
        if key not in {"schema", "namespace", "seal_sha256"}
    }


def _load_serving_policy(root: Path, *, allow_lkg: bool) -> dict[str, Any]:
    kinds = ("active", "lkg") if allow_lkg else ("active",)
    for kind in kinds:
        try:
            pointer = store.read_pointer(root, kind)
            return _load_policy(str(pointer["policy_id"]), root)
        except (store.DistillationStoreError, DistillationError, KeyError):
            continue
    return {}


def _is_legacy_bootstrap_policy(policy: Mapping[str, Any]) -> bool:
    return policy.get("serve_mode") == "legacy" and policy.get("kind") == (
        "tiny-logistic-policy"
    )


def _non_bootstrap_policy(policy: dict[str, Any]) -> dict[str, Any]:
    return {} if _is_legacy_bootstrap_policy(policy) else policy


def load_active_policy(root: Path | None = None) -> dict[str, Any]:
    root = root or CHRONOVISOR_ROOT
    if not _enabled_for_root(root):
        return {}
    try:
        rollout_status = str(_read_worker_state(root).get("status") or "")
    except store.DistillationStoreError:
        return {}
    if rollout_status not in {"canary", "active", "rolled_back"}:
        return {}
    return _non_bootstrap_policy(
        _load_serving_policy(root, allow_lkg=rollout_status == "rolled_back")
    )


def load_policy_for_session(
    session_id: str, root: Path | None = None
) -> dict[str, Any]:
    root = root or CHRONOVISOR_ROOT
    if not _enabled_for_root(root):
        return {}
    try:
        status = str(_read_worker_state(root).get("status") or "")
    except store.DistillationStoreError:
        return {}
    if status not in {"canary", "active", "rolled_back"}:
        return {}
    from chronovisor.recall import recall_distillation_rollout

    policy_id = recall_distillation_rollout.select_policy_id(root, session_id)
    if not policy_id:
        return {}
    try:
        return _non_bootstrap_policy(_load_policy(policy_id, root))
    except (store.DistillationStoreError, DistillationError):
        return {}


def load_policy_pair_for_session(
    session_id: str, root: Path | None = None
) -> dict[str, Any]:
    context = load_policy_observation_context(session_id, root)
    return context if context.get("stage") == "canary" else {}


def load_policy_observation_context(
    session_id: str, root: Path | None = None
) -> dict[str, Any]:
    """Atomically select and validate both arms for shadow/canary observation."""

    root = root or CHRONOVISOR_ROOT
    if not _enabled_for_root(root) or not session_id:
        return {}
    lock = store.distillation_dir(root) / "rollout.lock"
    try:
        from chronovisor.recall import recall_distillation_rollout as rollout

        with store._locked(lock):
            state = _read_worker_state(root)
            stage = str(state.get("status") or "")
            if stage not in {"shadow", "canary"} or state.get("learning_halted"):
                return {}
            candidate_id = str(store.read_pointer(root, "candidate")["policy_id"])
            incumbent_id = str(store.read_pointer(root, "active")["policy_id"])
            if str(store.read_pointer(root, "lkg")["policy_id"]) != incumbent_id:
                return {}
            candidate = _load_policy(candidate_id, root)
            incumbent = _load_policy(incumbent_id, root)
            lineage = candidate.get("lineage")
            baseline_id = (
                str(lineage.get("baseline_artifact_id") or "")
                if isinstance(lineage, Mapping)
                else ""
            )
            baseline = store.read_sealed(
                store.distillation_dir(root) / "baselines" / f"{baseline_id}.json",
                schema=BASELINE_SCHEMA,
            )
            receipt_id = str(state.get("evaluation_receipt_id") or "")
            receipt = store.read_sealed(
                store.distillation_dir(root) / "rollout-runs" / f"{receipt_id}.json",
                schema=rollout.EVALUATION_SCHEMA,
            )
            hard_floor = baseline.get("hard_floor")
            if (
                baseline.get("artifact_id") != baseline_id
                or not isinstance(hard_floor, Mapping)
                or hard_floor.get("p5_allowed") is not True
                or receipt.get("artifact_id") != receipt_id
                or receipt.get("policy_id") != candidate_id
                or receipt.get("incumbent_policy_id") != incumbent_id
                or receipt.get("baseline_id") != baseline_id
            ):
                return {}
            if stage == "shadow":
                served_id = incumbent_id
            else:
                percent = int(state.get("rollout_percent") or 0)
                bucket = int.from_bytes(
                    hashlib.sha256(
                        f"recall-distill-rollout-v2\0{session_id}".encode()
                    ).digest()[:8],
                    "big",
                ) % 10_000
                served_id = candidate_id if bucket < percent * 100 else incumbent_id
            return {
                "stage": stage,
                "stage_started_at": str(state.get("stage_started_at") or ""),
                "qualified_run_id": str(state.get("stage_run_id") or ""),
                "served_policy_id": served_id,
                "candidate_policy_id": candidate_id,
                "incumbent_policy_id": incumbent_id,
                "served_policy": _non_bootstrap_policy(
                    candidate if served_id == candidate_id else incumbent
                ),
                "candidate_policy": _non_bootstrap_policy(candidate),
                "incumbent_policy": _non_bootstrap_policy(incumbent),
            }
    except (KeyError, store.DistillationStoreError, DistillationError):
        return {}


def load_shadow_policy(root: Path | None = None) -> dict[str, Any]:
    """Load a qualified candidate for observation only, never live serving."""
    context = load_policy_observation_context("shadow-observation", root)
    if context.get("stage") != "shadow":
        return {}
    return {
        **context["candidate_policy"],
        "shadow_incumbent_policy_id": context["incumbent_policy_id"],
        "shadow_incumbent_policy": context["incumbent_policy"],
    }


def train_tiny_policy(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    samples: list[tuple[str, str, dict[str, float], float, float]] = []
    validation: list[tuple[dict[str, float], float]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        authority = row.get("authority")
        verdict = row.get("verdict")
        dimension = str(row.get("dimension") or "answer_utility")
        allowed = UTILITY_LABELS if dimension == "answer_utility" else RELEVANCE_LABELS
        if (
            authority != "teacher-only"
            or verdict not in allowed
            or verdict == "uncertain"
            or row.get("probe") is True
            or (
                row.get("source") == "counterfactual-label"
                and row.get("order_agreement") is not True
            )
        ):
            continue
        raw_features = row.get("features")
        if not isinstance(raw_features, Mapping) or set(raw_features) != set(
            FAST_FEATURE_KEYS
        ):
            continue
        rally_id = str(row.get("rally_id") or "")
        candidate_id = str(row.get("candidate_id") or "")
        key = (dimension, rally_id, candidate_id)
        if key in seen:
            continue
        seen.add(key)
        target = {
            "helpful": 1.0,
            "neutral": 0.5,
            "harmful": 0.0,
            "relevant": 1.0,
            "irrelevant": 0.0,
        }[str(verdict)]
        split = str(row.get("split") or "train")
        if split == "validation":
            validation.append((build_fast_features(raw_features), target))
            continue
        if split != "train":
            continue
        weight = 1.0
        samples.append(
            (rally_id, candidate_id, build_fast_features(raw_features), target, weight)
        )
    per_rally: dict[str, int] = defaultdict(int)
    for rally_id, _candidate_id, _features, _target, _weight in samples:
        per_rally[rally_id] += 1
    training = [
        (features, target, weight / per_rally[rally_id])
        for rally_id, _candidate_id, features, target, weight in samples
    ]
    weights = {key: 0.0 for key in FAST_FEATURE_KEYS}
    bias = 0.0
    if training:
        for _ in range(180):
            gradients = {key: 0.0 for key in FAST_FEATURE_KEYS}
            gradient_bias = 0.0
            scale = sum(weight for _, _, weight in training)
            for features, label, sample_weight in training:
                predicted = sigmoid(
                    bias
                    + sum(weights[key] * features[key] for key in FAST_FEATURE_KEYS)
                )
                error = (predicted - label) * sample_weight
                gradient_bias += error
                for key in FAST_FEATURE_KEYS:
                    gradients[key] += error * features[key]
            for key in FAST_FEATURE_KEYS:
                weights[key] -= 0.15 * gradients[key] / scale
            bias -= 0.15 * gradient_bias / scale
    threshold = 0.65
    if validation:
        scored = [
            (
                sigmoid(
                    bias
                    + sum(weights[key] * features[key] for key in FAST_FEATURE_KEYS)
                ),
                target,
            )
            for features, target in validation
        ]

        def threshold_quality(candidate: float) -> tuple[float, float]:
            positive = [
                score >= candidate for score, target in scored if target >= 0.75
            ]
            negative = [score < candidate for score, target in scored if target <= 0.25]
            if not positive or not negative:
                return (-1.0, -candidate)
            balanced = (
                sum(positive) / len(positive) + sum(negative) / len(negative)
            ) / 2
            return (balanced, -candidate)

        threshold = max(
            (round(value / 100, 2) for value in range(35, 86, 5)),
            key=threshold_quality,
        )
    return {
        "feature_keys": list(FAST_FEATURE_KEYS),
        "feature_revision": TEXT_FEATURE_REVISION,
        "weights": {key: round(value, 8) for key, value in weights.items()},
        "bias": round(bias, 8),
        "threshold": threshold,
        "abstain_margin": 0.08,
        "max_cards": 3,
        "training_rows": len(training),
        "validation_rows": len(validation),
    }


def materialize_training_rows(
    root: Path | None = None, *, limit: int = 10_000
) -> dict[str, Any]:
    """Join labels only to byte-identical prospective live feature snapshots."""

    if limit <= 0 or limit > 10_000:
        raise DistillationError("training materialization limit is invalid")
    root = root or CHRONOVISOR_ROOT
    rally_rows = store.read_chain(store.distillation_dir(root) / "rally-manifest.jsonl")
    rallies = {
        str(manifest["rally_id"]): manifest
        for row in rally_rows
        if isinstance((manifest := row.get("manifest")), Mapping)
    }
    features_by_pair: dict[tuple[str, str], dict[str, float]] = {}
    for row in store.read_chain(
        store.distillation_dir(root) / "candidate-ledger.jsonl"
    ):
        rally_id = str(row.get("rally_id") or "")
        snapshot = row.get("snapshot")
        if (
            not rally_id
            or not isinstance(snapshot, Mapping)
            or snapshot.get("feature_revision") != TEXT_FEATURE_REVISION
            or not isinstance(snapshot.get("candidates"), list)
        ):
            continue
        for candidate in snapshot["candidates"]:
            if not isinstance(candidate, Mapping) or not isinstance(
                candidate.get("features"), Mapping
            ):
                continue
            try:
                features = build_fast_features(candidate["features"])
            except DistillationError:
                continue
            if dict(candidate["features"]) == features:
                features_by_pair[
                    (rally_id, str(candidate.get("candidate_id") or ""))
                ] = features
    for rally_id, rally in rallies.items():
        for receipt in rally.get("exposure_receipts", []):
            if not isinstance(receipt, Mapping):
                continue
            artifact_id = str(receipt.get("exposure_artifact_id") or "")
            try:
                exposure = store.read_sealed(
                    store.distillation_dir(root) / "exposures" / f"{artifact_id}.json",
                    schema="chronovisor.recall-exact-exposure.v1",
                )
            except store.DistillationStoreError:
                continue
            rows = exposure.get("candidate_feature_snapshot")
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, Mapping) or not isinstance(
                    row.get("features"), Mapping
                ):
                    continue
                try:
                    features = build_fast_features(row["features"])
                except DistillationError:
                    continue
                if dict(row["features"]) == features:
                    features_by_pair[(rally_id, str(row.get("candidate_id") or ""))] = (
                        features
                    )
    materialized: list[dict[str, Any]] = []
    for label in store.read_chain(store.distillation_dir(root) / "label-ledger.jsonl"):
        rally_id = str(label.get("rally_id") or "")
        candidate_id = str(label.get("candidate_id") or "")
        rally = rallies.get(rally_id)
        if rally is None:
            continue
        raw_features = label.get("features") or features_by_pair.get(
            (rally_id, candidate_id)
        )
        if not isinstance(raw_features, Mapping):
            continue
        try:
            features = build_fast_features(raw_features)
        except DistillationError:
            continue
        if (
            set(raw_features) != set(FAST_FEATURE_KEYS)
            or dict(raw_features) != features
        ):
            continue
        verdict = str(label.get("verdict") or "")
        dimension = str(label.get("dimension") or "")
        if (
            label.get("authority") != "teacher-only"
            or verdict == "uncertain"
            or verdict
            not in (
                UTILITY_LABELS if dimension == "answer_utility" else RELEVANCE_LABELS
            )
        ):
            continue
        assignment = label.get("assignment")
        probe = isinstance(assignment, Mapping) and assignment.get("probe") is True
        materialized.append(
            {
                "rally_id": rally_id,
                "candidate_id": candidate_id,
                "session_cluster_id": rally["session_cluster_id"],
                "as_of": rally["as_of"],
                "dimension": dimension,
                "verdict": verdict,
                "authority": label["authority"],
                "features": features,
                "route": str(label.get("route") or ""),
                "model_digest": str(label.get("model_digest") or ""),
                "generator_model_digest": str(
                    label.get("generator_model_digest") or ""
                ),
                "judge_model_digest": str(label.get("judge_model_digest") or ""),
                "probe": probe,
                "source": str(label.get("kind") or ""),
                "order_agreement": label.get("order_agreement") is True,
                "label_record_sha256": label["record_sha256"],
            }
        )
    materialized = materialized[-limit:]
    split_plan_id = ""
    split: dict[str, str] = {}
    try:
        split_plan = _read_split_plan(root)
        split_plan_id = str(split_plan["artifact_id"])
        split = {
            str(rally_id): str(value)
            for rally_id, value in split_plan["assignments"].items()
        }
    except (KeyError, DistillationError, store.DistillationStoreError):
        split = grouped_rolling_split(materialized) if materialized else {}
    rows = [
        {
            **row,
            "split": split.get(row["rally_id"], "embargo"),
            "split_plan_id": split_plan_id,
        }
        for row in materialized
    ]
    _, _, artifact = store.write_immutable(
        store.distillation_dir(root) / "training-snapshots",
        {
            "kind": "text-parity-training-snapshot",
            "feature_revision": TEXT_FEATURE_REVISION,
            "rows": rows,
            "label_chain_head": store.verify_chain(
                store.distillation_dir(root) / "label-ledger.jsonl"
            )["head_sha256"],
        },
        schema="chronovisor.recall-distill-training.v1",
    )
    return artifact


def _wilson_lower(successes: int, total: int) -> float:
    if total <= 0 or successes < 0 or successes > total:
        return 0.0
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = proportion + z * z / (2 * total)
    spread = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    return max(0.0, (center - spread) / denominator)


def _wilson_upper(successes: int, total: int) -> float:
    if total <= 0 or successes < 0 or successes > total:
        return 1.0
    return min(1.0, 1.0 - _wilson_lower(total - successes, total))


def _active_training_cohort(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the latest coherent local-model cohort and resplit only it."""

    teacher_digests = {role: "" for role in TEACHER_ROLES}
    counterfactual_digests: tuple[str, str] | None = None
    for row in rows:
        route = str(row.get("route") or "")
        digest = str(row.get("model_digest") or "")
        if (
            row.get("source") == "teacher-label"
            and route in teacher_digests
            and re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            teacher_digests[route] = digest
        generator = str(row.get("generator_model_digest") or "")
        judge = str(row.get("judge_model_digest") or "")
        if (
            row.get("source") == "counterfactual-label"
            and row.get("order_agreement") is True
            and re.fullmatch(r"[0-9a-f]{64}", generator)
            and re.fullmatch(r"[0-9a-f]{64}", judge)
            and generator != judge
        ):
            counterfactual_digests = (generator, judge)
    selected: list[dict[str, Any]] = []
    for row in rows:
        route = str(row.get("route") or "")
        if row.get("source") == "teacher-label":
            if row.get("model_digest") != teacher_digests.get(route):
                continue
        elif row.get("source") == "counterfactual-label":
            if counterfactual_digests is None or (
                row.get("generator_model_digest"),
                row.get("judge_model_digest"),
            ) != counterfactual_digests:
                continue
        else:
            continue
        selected.append(dict(row))
    fixed_ids = {str(row.get("split_plan_id") or "") for row in selected}
    if len(fixed_ids) != 1 or not next(iter(fixed_ids), ""):
        split = grouped_rolling_split(selected) if selected else {}
        selected = [
            {**row, "split": split[str(row["rally_id"])]} for row in selected
        ]
    cohort = {
        "revision": "latest-model-cohort-v1",
        "teacher_model_digests": teacher_digests,
        "counterfactual_model_digests": list(counterfactual_digests or ()),
    }
    return selected, {
        **cohort,
        "cohort_sha256": canonical_json.canonical_json_sha256_strict(cohort),
    }


def _offline_training_gate(
    rows: Sequence[Mapping[str, Any]],
    config: DistillationConfig,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Qualify untrusted local-model data without calling it verified truth."""

    rows, model_cohort = _active_training_cohort(rows)

    owner = [
        row
        for row in rows
        if row.get("source") == "teacher-label"
        and row.get("probe") is not True
        and row.get("verdict") in {"relevant", "irrelevant"}
    ]
    probes = [
        row
        for row in rows
        if row.get("source") == "teacher-label"
        and row.get("probe") is True
        and row.get("verdict") in {"relevant", "irrelevant"}
    ]
    counterfactual = [
        row
        for row in rows
        if row.get("source") == "counterfactual-label"
        and row.get("order_agreement") is True
        and row.get("verdict") in {"helpful", "harmful", "neutral"}
    ]
    teacher_counts = {
        verdict: sum(row.get("verdict") == verdict for row in owner)
        for verdict in ("relevant", "irrelevant")
    }
    probe_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in probes:
        probe_groups[(str(row["rally_id"]), str(row["candidate_id"]))].append(row)
    complete_probes = [
        group
        for group in probe_groups.values()
        if {str(row.get("route") or "") for row in group} == set(TEACHER_ROLES)
        and all(row.get("split") == "test" for row in group)
    ]
    stable = sum(
        len({str(row["verdict"]) for row in group}) == 1 for group in complete_probes
    )
    digests = {
        role: {
            str(row.get("model_digest") or "")
            for row in [*owner, *probes]
            if row.get("route") == role and row.get("model_digest")
        }
        for role in TEACHER_ROLES
    }
    route_models = {
        next(iter(values)) for values in digests.values() if len(values) == 1
    }
    reasons: list[str] = []
    fixed_ids = {str(row.get("split_plan_id") or "") for row in rows}
    try:
        split_plan = _read_split_plan(root) if root is not None else {}
    except (DistillationError, store.DistillationStoreError):
        split_plan = {}
    assignments = split_plan.get("assignments")
    fixed_split_valid = (
        len(fixed_ids) == 1
        and bool(next(iter(fixed_ids), ""))
        and split_plan.get("artifact_id") == next(iter(fixed_ids), "")
        and split_plan.get("model_cohort_sha256") == model_cohort["cohort_sha256"]
        and split_plan.get("feature_revision") == TEXT_FEATURE_REVISION
        and isinstance(assignments, Mapping)
        and all(
            assignments.get(str(row.get("rally_id") or "")) == row.get("split")
            for row in rows
        )
    )
    if not fixed_split_valid:
        reasons.append("fixed_split_plan_missing")
    if len(owner) < config.hard_floor_teacher_labels:
        reasons.append("teacher_labels_below_floor")
    if any(
        count < config.hard_floor_teacher_per_class for count in teacher_counts.values()
    ):
        reasons.append("teacher_class_below_floor")
    if len(complete_probes) < config.hard_floor_probe_pairs:
        reasons.append("probe_pairs_below_floor")
    if any(len(values) != 1 for values in digests.values()) or len(route_models) != 3:
        reasons.append("teacher_models_not_distinct")
    if _wilson_lower(stable, len(complete_probes)) < 0.60:
        reasons.append("probe_route_stability_below_gate")
    if not {"train", "validation", "test"}.issubset(
        {str(row.get("split") or "") for row in [*owner, *counterfactual]}
    ):
        reasons.append("chronological_split_incomplete")

    route_folds: dict[str, Any] = {}
    for holdout in TEACHER_ROLES:
        fold_policy = train_tiny_policy(
            [
                row
                for row in rows
                if row.get("route") != holdout
                and (
                    row.get("probe") is not True or row.get("source") != "teacher-label"
                )
            ]
        )
        test = [
            row
            for row in probes
            if row.get("route") == holdout and row.get("split") == "test"
        ]
        relevant = [row for row in test if row.get("verdict") == "relevant"]
        irrelevant = [row for row in test if row.get("verdict") == "irrelevant"]

        def accepted(
            row: Mapping[str, Any], policy: Mapping[str, Any] = fold_policy
        ) -> bool:
            score = score_fast_features(row["features"], policy)
            return policy_decision(score, policy)["decision"] == "read"

        recall = _wilson_lower(sum(accepted(row) for row in relevant), len(relevant))
        specificity = _wilson_lower(
            sum(not accepted(row) for row in irrelevant), len(irrelevant)
        )
        passed = (
            len(relevant) >= 30
            and len(irrelevant) >= 30
            and recall >= 0.65
            and specificity >= 0.65
        )
        route_folds[holdout] = {
            "relevant": len(relevant),
            "irrelevant": len(irrelevant),
            "recall_wilson_lower": round(recall, 8),
            "specificity_wilson_lower": round(specificity, 8),
            "passed": passed,
        }
        if not passed:
            reasons.append(f"route_holdout_failed_{holdout.rsplit('.', 1)[-1]}")

    cf_test = [
        row
        for row in counterfactual
        if row.get("split") == "test" and row.get("verdict") in {"helpful", "harmful"}
    ]
    if len(cf_test) < config.hard_floor_counterfactual_pairs:
        reasons.append("counterfactual_pairs_below_floor")
    final_policy = train_tiny_policy(rows)
    directional = sum(
        (
            policy_decision(
                score_fast_features(row["features"], final_policy), final_policy
            )["decision"]
            == "read"
        )
        == (row.get("verdict") == "helpful")
        for row in cf_test
    )
    cf_lower = _wilson_lower(directional, len(cf_test))
    if len(cf_test) < 30 or cf_lower < 0.60:
        reasons.append("counterfactual_direction_below_gate")
    return {
        "schema": "chronovisor.recall-offline-training-gate.v2",
        "truth_authority": "teacher_only_not_verified",
        "model_cohort": model_cohort,
        "passed": not reasons,
        "reasons": sorted(set(reasons)),
        "teacher_counts": {"total": len(owner), **teacher_counts},
        "counterfactual_pairs": len(cf_test),
        "probe": {
            "pairs": len(complete_probes),
            "locked_test_only": True,
            "stable": stable,
            "route_stability_wilson_lower": round(
                _wilson_lower(stable, len(complete_probes)), 8
            ),
            "is_truth": False,
        },
        "route_folds": route_folds,
        "counterfactual_direction": {
            "denominator": len(cf_test),
            "wilson_lower": round(cf_lower, 8),
        },
    }


def _ensure_bootstrap_policy(root: Path, baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Seed rollout identity with a sealed marker for unchanged legacy Recall."""

    lock = store.distillation_dir(root) / "bootstrap.lock"
    with store._locked(lock):
        active: dict[str, Any] = {}
        lkg: dict[str, Any] = {}
        try:
            active = _load_policy(
                str(store.read_pointer(root, "active")["policy_id"]), root
            )
        except (KeyError, store.DistillationStoreError, DistillationError):
            pass
        try:
            lkg = _load_policy(str(store.read_pointer(root, "lkg")["policy_id"]), root)
        except (KeyError, store.DistillationStoreError, DistillationError):
            pass
        if active and lkg:
            return active
        if active:
            store.write_pointer(
                root, "lkg", str(active["artifact_id"]), bootstrap=False
            )
            return active
        if lkg:
            store.write_pointer(
                root, "active", str(lkg["artifact_id"]), bootstrap=False
            )
            return lkg
        policy = {
            **train_tiny_policy([]),
            "threshold": 1.0,
            "abstain_margin": 1.0,
            "max_cards": 1,
            "serve_mode": "legacy",
        }
        policy_id, _, artifact = store.write_immutable(
            store.distillation_dir(root) / "policies",
            {
                "kind": "tiny-logistic-policy",
                **policy,
                "lineage": {
                    "bootstrap_revision": "legacy-incumbent-v1",
                    "baseline_artifact_id": str(baseline.get("artifact_id") or ""),
                },
            },
            schema=POLICY_SCHEMA,
        )
        store.write_pointer(root, "active", policy_id, bootstrap=True)
        store.write_pointer(root, "lkg", policy_id, bootstrap=True)
        return artifact


def _maybe_publish_candidate(
    root: Path,
    config: DistillationConfig,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    if baseline.get("hard_floor", {}).get("p5_allowed") is not True:
        return {"status": "held", "reason": "p5_hard_floor"}
    try:
        candidate = store.read_pointer(root, "candidate")
        candidate_policy = _load_policy(str(candidate["policy_id"]), root)
        lineage = candidate_policy.get("lineage")
        if not isinstance(lineage, Mapping) or lineage.get(
            "baseline_artifact_id"
        ) != baseline.get("artifact_id"):
            return {"status": "held", "reason": "candidate_baseline_mismatch"}
        return {"status": "candidate", "policy_id": candidate["policy_id"]}
    except (store.DistillationStoreError, DistillationError, KeyError):
        pass
    training = materialize_training_rows(root)
    rows = training["rows"]
    offline_gate = _offline_training_gate(rows, config, root=root)
    active_rows, model_cohort = _active_training_cohort(rows)
    if offline_gate != baseline.get("offline_training_gate"):
        return {"status": "held", "reason": "offline_gate_baseline_mismatch"}
    if offline_gate["passed"] is not True:
        return {"status": "held", "reason": "offline_training_gate"}
    try:
        store.read_pointer(root, "lkg")
        store.read_pointer(root, "active")
    except store.DistillationStoreError:
        return {"status": "held", "reason": "sealed_incumbent_missing"}
    policy = train_tiny_policy(active_rows)
    replay_id, _, _ = store.write_immutable(
        store.distillation_dir(root) / "locked-replays",
        {
            "kind": "locked-replay-input",
            "training_snapshot_id": training["artifact_id"],
            "baseline_artifact_id": baseline["artifact_id"],
            "policy_sha256": canonical_json.canonical_json_sha256_strict(policy),
            "offline_gate_sha256": canonical_json.canonical_json_sha256_strict(
                offline_gate
            ),
            "model_cohort_sha256": model_cohort["cohort_sha256"],
            "split_revision": "grouped-rolling-v1",
        },
        schema="chronovisor.recall-distill-locked-replay.v1",
    )
    artifact = publish_policy(
        policy,
        lineage={
            "training_snapshot_id": training["artifact_id"],
            "locked_replay_id": replay_id,
            "baseline_artifact_id": baseline["artifact_id"],
            "model_cohort_sha256": model_cohort["cohort_sha256"],
        },
        root=root,
    )
    return {"status": "candidate", "policy_id": artifact["artifact_id"]}


def _operational_rollout_metrics(
    root: Path, candidate_id: str, incumbent_id: str
) -> dict[str, dict[str, Any]]:
    missing = {
        "denominator": 0,
        "min_denominator": 1,
        "min_days": 7,
        "ci_lower": 0.0,
        "min_ci_lower": 1.0,
    }

    def gate(denominator: int, score: float, threshold: float) -> dict[str, Any]:
        return {
            "denominator": denominator,
            "min_denominator": 500,
            "min_days": 7,
            "ci_lower": max(0.0, min(1.0, score)),
            "min_ci_lower": threshold,
        }

    try:
        state = _read_worker_state(root)
    except store.DistillationStoreError:
        state = {}
    stage = str(state.get("status") or "")
    stage_started_at = str(state.get("stage_started_at") or "")
    qualified_run_id = str(state.get("stage_run_id") or "")
    candidate_only = False
    if stage == "canary" and int(state.get("rollout_percent") or 0) == 100:
        try:
            candidate_only = _load_policy(incumbent_id, root).get("serve_mode") == "legacy"
        except (store.DistillationStoreError, DistillationError):
            candidate_only = False
    pairs: list[dict[str, Any]] = []
    try:
        shadow_receipts = store.read_chain(
            store.distillation_dir(root) / "shadow-observation-receipts.jsonl"
        )
    except store.DistillationStoreError:
        shadow_receipts = []
    for receipt in shadow_receipts:
        binding = {
            key: receipt.get(key)
            for key in (
                "decision_id",
                "host",
                "session_id_sha256",
                "query_semantic_sha256",
                "policy_id",
                "incumbent_policy_id",
                "served_policy_id",
                "stage",
                "stage_started_at",
                "qualified_run_id",
                "selected_candidate_ids",
                "incumbent_selected_candidate_ids",
                "paired_eligible",
                "candidate_pool_sha256",
                "candidate_feature_snapshot_sha256",
                "runtime_observation_sha256",
                "observed_at",
            )
        }
        if receipt.get("kind") != "shadow-policy-observation" or receipt.get(
            "binding_sha256"
        ) != canonical_json.canonical_json_sha256_strict(binding):
            continue
        artifact_id = receipt.get("shadow_observation_artifact_id")
        try:
            artifact = store.read_sealed(
                store.distillation_dir(root)
                / "shadow-observations"
                / f"{artifact_id}.json",
                schema=SHADOW_OBSERVATION_SCHEMA,
            )
        except store.DistillationStoreError:
            continue
        if artifact.get("artifact_id") != artifact_id or any(
            artifact.get(key) != value for key, value in binding.items()
        ):
            continue
        observation = artifact.get("runtime_observation")
        candidate_selected = artifact.get("selected_candidate_ids")
        incumbent_selected = artifact.get("incumbent_selected_candidate_ids")
        features = artifact.get("candidate_feature_snapshot")
        if (
            artifact.get("policy_id") != candidate_id
            or artifact.get("incumbent_policy_id") != incumbent_id
            or artifact.get("stage") != stage
            or artifact.get("stage_started_at") != stage_started_at
            or artifact.get("qualified_run_id") != qualified_run_id
            or (
                artifact.get("paired_eligible") is not True
                and not candidate_only
            )
            or not isinstance(observation, Mapping)
            or not isinstance(candidate_selected, list)
            or not isinstance(incumbent_selected, list)
            or not isinstance(features, list)
        ):
            continue
        try:
            latency_ms = float(observation["latency_ms"])
            timed_out = observation["timed_out"]
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not math.isfinite(latency_ms)
            or not 0 <= latency_ms <= 60_000
            or not isinstance(timed_out, bool)
        ):
            continue
        pairs.append(
            {
                "candidate_covered": bool(candidate_selected),
                "incumbent_covered": bool(incumbent_selected),
                "latency_ms": latency_ms,
                "timed_out": timed_out,
            }
        )
    metrics = {
        name: dict(missing)
        for name in (
            "coverage_abstain",
            "latency_timeout",
            "cohort_delta",
            "feature_parity",
        )
    }
    if pairs:
        candidate_coverage = sum(row["candidate_covered"] for row in pairs) / len(
            pairs
        )
        incumbent_coverage = sum(row["incumbent_covered"] for row in pairs) / len(
            pairs
        )
        coverage_ok = float(candidate_coverage >= incumbent_coverage * 0.90)
        metrics["coverage_abstain"] = gate(len(pairs), coverage_ok, 1.0)
        metrics["cohort_delta"] = gate(len(pairs), coverage_ok, 1.0)
        ordered = sorted(row["latency_ms"] for row in pairs)
        p95 = ordered[math.ceil(len(ordered) * 0.95) - 1]
        timeouts = sum(row["timed_out"] for row in pairs)
        latency_ok = float(
            p95 <= 180 and _wilson_upper(timeouts, len(pairs)) <= 0.03
        )
        metrics["latency_timeout"] = gate(len(pairs), latency_ok, 1.0)
        metrics["feature_parity"] = gate(len(pairs), 1.0, 1.0)
    return metrics


def _authenticated_negative_vetoes(root: Path, policy_id: str) -> int:
    rows = store.read_chain(
        store.distillation_dir(root) / "negative-veto-receipts.jsonl"
    )
    valid = 0
    for row in rows:
        binding = {
            key: row.get(key)
            for key in (
                "decision_id",
                "correction_id",
                "exposure_artifact_id",
                "policy_id",
                "candidate_id",
                "page_id",
                "preimage_sha256",
                "postimage_sha256",
                "cas_status",
                "observed_at",
                "producer_revision",
            )
        }
        if (
            row.get("kind") != "authenticated-negative-veto"
            or row.get("policy_id") != policy_id
            or row.get("binding_sha256")
            != canonical_json.canonical_json_sha256_strict(binding)
        ):
            continue
        artifact_id = str(row.get("veto_artifact_id") or "")
        try:
            artifact = store.read_sealed(
                store.distillation_dir(root)
                / "negative-vetoes"
                / f"{artifact_id}.json",
                schema=VETO_SCHEMA,
            )
        except store.DistillationStoreError:
            continue
        if artifact.get("artifact_id") == artifact_id and all(
            artifact.get(key) == value for key, value in binding.items()
        ):
            valid += 1
    return valid


def _automatic_rollout_evaluation(
    root: Path,
    baseline: Mapping[str, Any],
    promotion: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize sealed runtime receipts into a fail-closed rollout evaluation."""

    if promotion.get("status") != "candidate":
        return {"status": "not_applicable"}
    try:
        from chronovisor.recall import recall_distillation_rollout as rollout

        state = _read_worker_state(root)
        if state.get("status") not in {"replay", "shadow", "canary"}:
            return {"status": "not_due"}
        candidate_id = str(store.read_pointer(root, "candidate")["policy_id"])
        incumbent_id = str(store.read_pointer(root, "active")["policy_id"])
        policy = _load_policy(candidate_id, root)
    except (KeyError, store.DistillationStoreError, DistillationError):
        return {"status": "held", "reason": "rollout_identity_unavailable"}
    vetoes = _authenticated_negative_vetoes(root, candidate_id)
    if vetoes:
        veto_run_id = canonical_json.canonical_json_sha256_strict(
            {
                "kind": "authenticated-negative-veto-run-v1",
                "policy_id": candidate_id,
                "veto_head": store.verify_chain(
                    store.distillation_dir(root) / "negative-veto-receipts.jsonl"
                )["head_sha256"],
            }
        )
        result = rollout.rollback_to_lkg(
            root, veto_run_id, "authenticated_negative_veto"
        )
        return {"status": "rolled_back", "negative_vetoes": vetoes, **result}
    label_head = store.verify_chain(
        store.distillation_dir(root) / "label-ledger.jsonl"
    )["head_sha256"]
    exposure_head = store.verify_chain(
        store.distillation_dir(root) / "exposure-receipts.jsonl"
    )["head_sha256"]
    run_id = canonical_json.canonical_json_sha256_strict(
        {
            "kind": "automatic-rollout-evaluation-v2",
            "policy_id": candidate_id,
            "status": state.get("status"),
            "rollout_percent": state.get("rollout_percent", 0),
            "label_head": label_head,
            "exposure_head": exposure_head,
            "shadow_head": store.verify_chain(
                store.distillation_dir(root) / "shadow-observation-receipts.jsonl"
            )["head_sha256"],
        }
    )
    measured_metrics = _operational_rollout_metrics(root, candidate_id, incumbent_id)
    replay_gate = {
        "denominator": 1,
        "min_denominator": 1,
        "min_days": 0,
        "ci_lower": 1.0,
        "min_ci_lower": 1.0,
    }
    replay_metrics = {
        name: dict(replay_gate)
        for name in (
            "coverage_abstain",
            "latency_timeout",
            "cohort_delta",
            "feature_parity",
        )
    }
    lineage = (
        policy.get("lineage") if isinstance(policy.get("lineage"), Mapping) else {}
    )
    split_sha256 = str(lineage.get("locked_replay_id") or "")
    if re.fullmatch(r"[0-9a-f]{64}", split_sha256) is None:
        split_sha256 = hashlib.sha256(b"missing-split").hexdigest()
    raw_watermark = str(baseline.get("raw_watermark") or "")
    if re.fullmatch(r"[0-9a-f]{64}", raw_watermark) is None:
        raw_watermark = hashlib.sha256(raw_watermark.encode()).hexdigest()
    evaluation_id, _, _ = store.write_immutable(
        store.distillation_dir(root) / "evaluations",
        {
            "kind": "automatic-closed-metrics",
            "run_id": run_id,
            "policy_id": candidate_id,
            "baseline_id": baseline["artifact_id"],
            "raw_watermark": raw_watermark,
            "incumbent_policy_id": incumbent_id,
            "split_sha256": split_sha256,
            "feature_revision": TEXT_FEATURE_REVISION,
            "feature_parity_sha256": canonical_json.canonical_json_sha256_strict(
                {"feature_keys": list(FAST_FEATURE_KEYS), "policy_id": candidate_id}
            ),
            "offline_gate_sha256": canonical_json.canonical_json_sha256_strict(
                baseline["offline_training_gate"]
            ),
            "observation_mode": (
                "candidate_only_legacy_incumbent"
                if state.get("status") == "canary"
                and int(state.get("rollout_percent") or 0) == 100
                and _load_policy(incumbent_id, root).get("serve_mode") == "legacy"
                else "paired"
            ),
            "replay_metrics": replay_metrics,
            "shadow_metrics": measured_metrics,
            "canary_metrics": measured_metrics,
        },
        schema=rollout.EVALUATION_SCHEMA,
    )
    try:
        result = rollout.evaluate_and_advance(
            root,
            datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            {"run_id": run_id, "evaluation_artifact_id": evaluation_id},
        )
    except rollout.RolloutError:
        return {"status": "held", "reason": "automatic_evaluation_invalid"}
    return {"status": "held", "reason": "metrics_denominator_missing", **result}


def publish_policy(
    policy: Mapping[str, Any],
    *,
    lineage: Mapping[str, Any],
    root: Path | None = None,
) -> dict[str, Any]:
    root = root or CHRONOVISOR_ROOT
    with store._locked(store.distillation_dir(root) / "rollout.lock"):
        policy_id, _, artifact = store.write_immutable(
            store.distillation_dir(root) / "policies",
            {"kind": "tiny-logistic-policy", **policy, "lineage": dict(lineage)},
            schema=POLICY_SCHEMA,
        )
        try:
            active = store.read_pointer(root, "active")
            store.write_pointer(root, "lkg", str(active["policy_id"]))
        except (store.DistillationStoreError, KeyError):
            pass
        store.write_pointer(root, "candidate", policy_id)
        try:
            previous_state = _read_worker_state(root)
        except store.DistillationStoreError:
            previous_state = {"kind": "worker-state"}
        store.write_sealed_state(
            store.distillation_dir(root) / store.STATE_FILE,
            {
                **previous_state,
                "status": "replay",
                "rollout_percent": 0,
                "stage_started_at": datetime.now(UTC)
                .isoformat()
                .replace("+00:00", "Z"),
                "learning_halted": False,
                "error_code": "",
            },
        )
        return artifact


def grouped_rolling_split(
    rows: Sequence[Mapping[str, Any]], *, embargo_components: int = 1
) -> dict[str, str]:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    owner: dict[str, int] = {}
    for index, row in enumerate(rows):
        keys = [
            f"session:{row.get('session_cluster_id') or row.get('session_key', '')}"
        ]
        keys.extend(
            f"{field}:{value}"
            for field in ("task_id", "entity_id")
            if (value := row.get(field))
        )
        for key in keys:
            if key in owner:
                union(index, owner[key])
            else:
                owner[key] = index
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[find(index)].append(index)
    ordered = sorted(
        components.values(),
        key=lambda indexes: (
            max(str(rows[index].get("as_of", "")) for index in indexes),
            min(str(rows[index].get("rally_id", "")) for index in indexes),
        ),
    )
    train_end = int(len(ordered) * 0.70)
    validation_end = int(len(ordered) * 0.85)
    use_embargo = (
        embargo_components > 0
        and validation_end - train_end > embargo_components * 2
        and len(ordered) - validation_end > embargo_components
    )
    result: dict[str, str] = {}
    for position, indexes in enumerate(ordered):
        split = (
            "train"
            if position < train_end
            else "validation"
            if position < validation_end
            else "test"
        )
        if use_embargo and position in {train_end, validation_end}:
            split = "embargo"
        for index in indexes:
            result[str(rows[index]["rally_id"])] = split
    return result


def _read_split_plan(root: Path) -> dict[str, Any]:
    pointer = store.read_sealed(
        store.distillation_dir(root) / "split-plan.json",
        schema=store.DISTILLATION_SCHEMA,
    )
    plan_id = str(pointer.get("split_plan_id") or "")
    if re.fullmatch(r"[0-9a-f]{64}", plan_id) is None:
        raise DistillationError("split plan pointer is invalid")
    artifact = store.read_sealed(
        store.distillation_dir(root) / "split-plans" / f"{plan_id}.json",
        schema=SPLIT_PLAN_SCHEMA,
    )
    assignments = artifact.get("assignments")
    if (
        artifact.get("artifact_id") != plan_id
        or not isinstance(assignments, Mapping)
        or any(value not in {"train", "validation", "test", "embargo"} for value in assignments.values())
    ):
        raise DistillationError("split plan artifact is invalid")
    return artifact


def _ensure_split_plan(
    root: Path,
    rallies: Sequence[Mapping[str, Any]],
    *,
    raw_watermark: str,
    model_cohort_sha256: str,
) -> dict[str, Any]:
    assignments = grouped_rolling_split(rallies)
    plan_id, _, artifact = store.write_immutable(
        store.distillation_dir(root) / "split-plans",
        {
            "kind": "fixed-chronological-group-split",
            "raw_watermark": raw_watermark,
            "feature_revision": TEXT_FEATURE_REVISION,
            "model_cohort_sha256": model_cohort_sha256,
            "split_revision": "grouped-rolling-v1",
            "assignments": assignments,
        },
        schema=SPLIT_PLAN_SCHEMA,
    )
    store.write_sealed_state(
        store.distillation_dir(root) / "split-plan.json",
        {"kind": "split-plan-pointer", "split_plan_id": plan_id},
    )
    return artifact


def _verified_counts(root: Path) -> dict[str, int]:
    counts = {label: 0 for label in ("helpful", "neutral", "harmful")}
    for row in store.read_chain(store.distillation_dir(root) / "label-ledger.jsonl"):
        if row.get("authority") == "verified" and row.get("verdict") in counts:
            counts[str(row["verdict"])] += 1
    return counts


_AGGREGATE_METRIC_KEYS = frozenset(
    {
        "candidate_recall",
        "wrong_domain_rate",
        "strong_anchor_rescue_rate",
        "coverage_rate",
        "abstain_rate",
        "latency_p50_ms",
        "latency_p95_ms",
        "timeout_rate",
        "exact_structural_links",
        "exact_outcome_links",
        "archive_commit",
        "expected_commit",
        "drift",
    }
)


def _baseline_metrics(values: Mapping[str, Any] | None) -> dict[str, Any]:
    supplied = dict(values or {})
    unknown = set(supplied).difference(_AGGREGATE_METRIC_KEYS)
    if unknown:
        raise DistillationError(f"unknown aggregate metrics: {sorted(unknown)}")
    numeric = _AGGREGATE_METRIC_KEYS.difference(
        {"archive_commit", "expected_commit", "drift"}
    )
    for key in numeric:
        value = supplied.get(key)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise DistillationError(f"aggregate metric {key} must be non-negative")
    for key in ("archive_commit", "expected_commit"):
        value = supplied.get(key)
        if value is not None and (
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{7,64}|unknown", value) is None
        ):
            raise DistillationError(f"aggregate metric {key} is invalid")
    if supplied.get("drift") is not None and not isinstance(supplied["drift"], bool):
        raise DistillationError("aggregate metric drift must be boolean")
    return {key: supplied.get(key) for key in sorted(_AGGREGATE_METRIC_KEYS)}


def _automatic_baseline_metrics(root: Path) -> dict[str, Any]:
    """Derive only privacy-safe metrics from sealed/runtime-owned observations."""

    metrics: dict[str, Any] = {
        "wrong_domain_rate": None,
        "exact_outcome_links": 0,
    }
    try:
        identity = runtime_config.runtime_identity()
    except Exception:
        identity = {}
    archive_commit = identity.get("commit_id")
    expected_commit = identity.get("expected_commit")
    if isinstance(archive_commit, str) and re.fullmatch(
        r"[0-9a-f]{7,64}", archive_commit
    ):
        metrics["archive_commit"] = archive_commit
    if isinstance(expected_commit, str) and re.fullmatch(
        r"[0-9a-f]{7,64}", expected_commit
    ):
        metrics["expected_commit"] = expected_commit
    if (
        "archive_commit" in metrics
        and "expected_commit" in metrics
        and isinstance(identity.get("drift"), bool)
    ):
        metrics["drift"] = identity["drift"]

    log_rows: dict[str, dict[str, Any]] = {}
    log_path = root / "recall" / "recall-log.jsonl"
    try:
        with log_path.open(encoding="utf-8") as handle:
            lines = deque(handle, maxlen=10_000)
    except (OSError, UnicodeError):
        lines = deque()
    for line in lines:
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeError):
            continue
        if not isinstance(row, dict):
            continue
        latency = row.get("latency_ms")
        decision = row.get("decision")
        status = row.get("status")
        stage = row.get("stage")
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or not 0 <= float(latency) <= 60_000
            or decision not in {"none", "read", "search"}
            or not isinstance(status, str)
            or stage not in {"decision", "injected"}
        ):
            continue
        decision_id = row.get("decision_id")
        key = (
            str(decision_id)
            if isinstance(decision_id, str) and decision_id
            else hashlib.sha256(line.encode()).hexdigest()
        )
        log_rows[key] = {
            "latency_ms": float(latency),
            "covered": stage == "injected",
            "abstained": decision == "none",
            "timed_out": status in {"timeout", "degraded"},
        }

    exact_artifacts: list[dict[str, Any]] = []
    exact_receipts = [
        receipt for receipts in _exposure_map(root).values() for receipt in receipts
    ]
    for artifact_id in sorted(
        {str(receipt["exposure_artifact_id"]) for receipt in exact_receipts}
    ):
        try:
            exact_artifacts.append(
                store.read_sealed(
                    store.distillation_dir(root) / "exposures" / f"{artifact_id}.json",
                    schema="chronovisor.recall-exact-exposure.v1",
                )
            )
        except store.DistillationStoreError:
            continue

    runtime_rows: dict[str, dict[str, Any]] = {}
    for artifact in exact_artifacts:
        observation = artifact.get("runtime_observation")
        selected = artifact.get("candidate_ids")
        if not isinstance(selected, list):
            continue
        if not isinstance(observation, Mapping):
            continue
        try:
            latency = float(observation["latency_ms"])
            timed_out = observation["timed_out"]
        except (KeyError, TypeError, ValueError):
            continue
        if (
            not math.isfinite(latency)
            or not 0 <= latency <= 60_000
            or not isinstance(timed_out, bool)
        ):
            continue
        runtime_rows[str(artifact.get("decision_id") or artifact["artifact_id"])] = {
            "latency_ms": latency,
            "covered": bool(selected),
            "abstained": not bool(selected),
            "timed_out": timed_out,
        }
    exact_decision_ids = {str(receipt["decision_id"]) for receipt in exact_receipts}
    try:
        receipt_rows = store.read_chain(
            store.distillation_dir(root) / "exposure-receipts.jsonl"
        )
    except store.DistillationStoreError:
        receipt_rows = []
    for receipt in receipt_rows:
        binding = {
            key: receipt.get(key)
            for key in (
                "decision_id",
                "host",
                "session_id_sha256",
                "prompt_hash",
                "policy_id",
                "candidate_ids",
                "candidate_snapshot_sha256",
                "runtime_observation_sha256",
                "observed_at",
            )
        }
        observation = receipt.get("runtime_observation")
        if (
            receipt.get("kind") != "prospective-page-exposure"
            or str(receipt.get("decision_id") or "") in exact_decision_ids
            or receipt.get("binding_sha256")
            != canonical_json.canonical_json_sha256_strict(binding)
            or receipt.get("runtime_observation_sha256")
            != canonical_json.canonical_json_sha256_strict(observation)
            or not isinstance(observation, Mapping)
        ):
            continue
        try:
            _validate_exposure_policy_identity(
                root, str(receipt.get("policy_id") or "")
            )
            latency = float(observation["latency_ms"])
            timed_out = observation["timed_out"]
        except (KeyError, TypeError, ValueError, DistillationError):
            continue
        selected = receipt.get("candidate_ids")
        if (
            not isinstance(selected, list)
            or not math.isfinite(latency)
            or not 0 <= latency <= 60_000
            or not isinstance(timed_out, bool)
        ):
            continue
        runtime_rows[str(receipt["decision_id"])] = {
            "latency_ms": latency,
            "covered": bool(selected),
            "abstained": not bool(selected),
            "timed_out": timed_out,
        }

    try:
        shadow_receipts = store.read_chain(
            store.distillation_dir(root) / "shadow-observation-receipts.jsonl"
        )
    except store.DistillationStoreError:
        shadow_receipts = []
    for receipt in shadow_receipts:
        binding = {
            key: receipt.get(key)
            for key in (
                "decision_id",
                "host",
                "session_id_sha256",
                "query_semantic_sha256",
                "policy_id",
                "incumbent_policy_id",
                "served_policy_id",
                "stage",
                "stage_started_at",
                "qualified_run_id",
                "selected_candidate_ids",
                "incumbent_selected_candidate_ids",
                "paired_eligible",
                "candidate_pool_sha256",
                "candidate_feature_snapshot_sha256",
                "runtime_observation_sha256",
                "observed_at",
            )
        }
        artifact_id = receipt.get("shadow_observation_artifact_id")
        if receipt.get("kind") != "shadow-policy-observation" or receipt.get(
            "binding_sha256"
        ) != canonical_json.canonical_json_sha256_strict(binding):
            continue
        try:
            artifact = store.read_sealed(
                store.distillation_dir(root)
                / "shadow-observations"
                / f"{artifact_id}.json",
                schema=SHADOW_OBSERVATION_SCHEMA,
            )
        except store.DistillationStoreError:
            continue
        if artifact.get("artifact_id") != artifact_id or any(
            artifact.get(key) != value for key, value in binding.items()
        ):
            continue
    rows = list(log_rows.values()) or list(runtime_rows.values())
    if rows:
        ordered = sorted(float(row["latency_ms"]) for row in rows)
        metrics.update(
            {
                "coverage_rate": sum(bool(row["covered"]) for row in rows) / len(rows),
                "abstain_rate": sum(bool(row["abstained"]) for row in rows) / len(rows),
                "latency_p50_ms": ordered[(len(ordered) - 1) // 2],
                "latency_p95_ms": ordered[math.ceil(len(ordered) * 0.95) - 1],
                "timeout_rate": sum(bool(row["timed_out"]) for row in rows) / len(rows),
            }
        )
    try:
        labels = store.read_chain(store.distillation_dir(root) / "label-ledger.jsonl")
    except store.DistillationStoreError:
        labels = []
    metrics["exact_structural_links"] = sum(
        row.get("authority") == "verified"
        and row.get("dimension") == "relevance"
        and row.get("closed_predicate") in RELEVANCE_CLOSED_PREDICATES
        for row in labels
    )
    return metrics


def preflight(
    *,
    raw_dir: Path,
    root: Path | None = None,
    config_path: Path | None = None,
    runtime_commit: str = "",
    aggregate_metrics: Mapping[str, Any] | None = None,
    _rallies: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    root = root or raw_dir.parent
    config = load_distillation_config(config_path)
    rallies = (
        list(_rallies)
        if _rallies is not None
        else extract_rallies(
            raw_dir, root=root, max_context_bytes=config.max_input_bytes
        )
    )
    dates = [
        datetime.fromtimestamp(row["as_of_us"] / 1_000_000, UTC).date()
        for row in rallies
    ]
    span_days = (max(dates) - min(dates)).days + 1 if dates else 0
    windows = len({(date - min(dates)).days // 7 for date in dates}) if dates else 0
    training_snapshot = materialize_training_rows(root)
    offline_gate = _offline_training_gate(training_snapshot["rows"], config, root=root)
    supplied_metrics = dict(aggregate_metrics or {})
    automatic_metrics = _automatic_baseline_metrics(root)
    metrics = _baseline_metrics({**automatic_metrics, **supplied_metrics})
    resolved_runtime_commit = (
        runtime_commit
        or metrics.get("archive_commit")
        or os.environ.get("CHRONOVISOR_RUNTIME_COMMIT", "unknown")
    )
    if re.fullmatch(r"[0-9a-f]{7,64}|unknown", resolved_runtime_commit) is None:
        raise DistillationError("runtime commit identity is invalid")
    reasons = []
    if len(rallies) < config.hard_floor_rallies:
        reasons.append("rallies_below_floor")
    if span_days < config.hard_floor_days:
        reasons.append("days_below_floor")
    if windows < config.hard_floor_windows:
        reasons.append("windows_below_floor")
    reasons.extend(str(value) for value in offline_gate["reasons"])
    if any(
        metrics[key] is None for key in ("archive_commit", "expected_commit", "drift")
    ):
        reasons.append("runtime_identity_unavailable")
    elif (
        metrics["drift"] is not False
        or metrics["archive_commit"] != metrics["expected_commit"]
    ):
        reasons.append("runtime_identity_drift")
    config_file = runtime_config.active_config_file(config_path)
    try:
        config_sha256 = hashlib.sha256(config_file.read_bytes()).hexdigest()
    except OSError:
        config_sha256 = hashlib.sha256(b"").hexdigest()
    payload = {
        "kind": "privacy-safe-baseline",
        "raw_watermark": committed_raw_watermark(raw_dir),
        "config_sha256": config_sha256,
        "runtime_commit": resolved_runtime_commit,
        "counts": {
            "rallies": len(rallies),
            "sessions": len(
                {(row["host"], row["session_cluster_id"]) for row in rallies}
            ),
            "independent_session_clusters": len(
                {(row["host"], row["session_cluster_id"]) for row in rallies}
            ),
            "independent_task_clusters": None,
            "task_cluster_status": "unavailable_no_closed_task_join",
            "host_distribution": {
                host: sum(row["host"] == host for row in rallies)
                for host in ("codex", "claude-code", "pi", "hermes")
            },
            "answer_utility_eligible": sum(
                bool(row["eligibility"]["answer_utility"]) for row in rallies
            ),
            "answer_utility_eligibility_rate": round(
                sum(bool(row["eligibility"]["answer_utility"]) for row in rallies)
                / len(rallies),
                8,
            )
            if rallies
            else 0.0,
            "span_days": span_days,
            "windows": windows,
            "teacher_only_labels": offline_gate["teacher_counts"]["total"],
            "verified_truth_labels": 0,
            "probe_pairs": offline_gate["probe"]["pairs"],
            "counterfactual_pairs": offline_gate["counterfactual_pairs"],
            "locked_test_probe_pairs": offline_gate["probe"]["pairs"],
            "locked_test_counterfactual_pairs": offline_gate[
                "counterfactual_pairs"
            ],
        },
        "offline_training_gate": offline_gate,
        "hard_floor": {
            "p5_allowed": not reasons,
            "reasons": reasons,
        },
        "metrics": metrics,
        "frozen_contract": {
            "rally_revision": "rally-v1",
            "assignment_revision": ASSIGNMENT_REVISION,
            "probe_revision": PROBE_REVISION,
            "probe_rate": 0.15,
            "feature_revision": TEXT_FEATURE_REVISION,
            "feature_whitelist": list(FAST_FEATURE_KEYS),
            "closed_predicates": sorted(CLOSED_PREDICATES),
            "teacher": {
                "routes": list(TEACHER_ROLES),
                "local_only": True,
                "max_input_bytes": config.max_input_bytes,
                "max_candidates": config.max_candidates,
                "max_load_skew": 0.10,
                "order_bias_max": 0.05,
            },
            "promotion": {
                "rolling_windows": config.hard_floor_windows,
                "canary_min_days": config.canary_min_days,
                "rollout_stages": list(config.rollout_stages),
                "paired_min_observations": 500,
                "paired_min_days": 7,
                "latency_p95_max_ms": 180,
                "timeout_wilson_upper_max": 0.03,
                "ci_method": "wilson-v1",
                "exact_outcome_mode": "authenticated_negative_veto_only",
            },
            "privacy": {
                "raw_remote_egress": 0,
                "persistent_raw_text": False,
                "persistent_session_ids": False,
            },
        },
    }
    _, _, artifact = store.write_immutable(
        store.distillation_dir(root) / "baselines", payload, schema=BASELINE_SCHEMA
    )
    return artifact


def _matching_p5_baseline(
    root: Path, current: Mapping[str, Any]
) -> dict[str, Any] | None:
    current_floor = current.get("hard_floor")
    if (
        not isinstance(current_floor, Mapping)
        or current_floor.get("p5_allowed") is not True
    ):
        return None
    matches: list[dict[str, Any]] = []
    directory = store.distillation_dir(root) / "baselines"
    for path in sorted(directory.glob("*.json")):
        try:
            artifact = store.read_sealed(path, schema=BASELINE_SCHEMA)
        except store.DistillationStoreError:
            continue
        hard_floor = artifact.get("hard_floor")
        if (
            artifact.get("raw_watermark") == current.get("raw_watermark")
            and artifact.get("config_sha256") == current.get("config_sha256")
            and artifact.get("runtime_commit") == current.get("runtime_commit")
            and artifact.get("metrics") == current.get("metrics")
            and artifact.get("frozen_contract") == current.get("frozen_contract")
            and artifact.get("offline_training_gate")
            == current.get("offline_training_gate")
            and isinstance(hard_floor, Mapping)
            and hard_floor.get("p5_allowed") is True
        ):
            matches.append(artifact)
    return matches[-1] if matches else None


def _teacher_payload(
    rally: Mapping[str, Any],
    candidate: Mapping[str, Any],
    texts: Mapping[str, str],
    *,
    max_input_bytes: int,
) -> dict[str, Any] | None:
    query = texts.get(str(rally["query_sha256"]), "")
    candidate_text = texts.get(str(candidate["text_sha256"]), "")
    context = [
        texts.get(str(ref["semantic_sha256"]), "")
        for ref in rally.get("context_refs", [])
    ]
    payload = {
        "schema": "chronovisor.recall-distill-teacher-input.v1",
        "rally_id": rally["rally_id"],
        "candidate_id": candidate["candidate_id"],
        "query": query,
        "context": context,
        "candidate": candidate_text,
    }
    while (
        context
        and len(canonical_json.canonical_json_bytes_strict(payload)) > max_input_bytes
    ):
        context.pop(0)
    if (
        not query
        or not candidate_text
        or len(canonical_json.canonical_json_bytes_strict(payload)) > max_input_bytes
    ):
        return None
    return payload


def _teacher_label(
    response: Mapping[str, Any],
    *,
    verified_predicate: str | None,
) -> dict[str, Any]:
    verdict = response.get("verdict")
    if verdict in RELEVANCE_LABELS:
        dimension = "relevance"
    elif verdict in UTILITY_LABELS:
        dimension = "answer_utility"
    else:
        return adjudicate_label(
            "uncertain",
            closed_predicate=None,
            reason="invalid_teacher_output",
            dimension="relevance",
        )
    if dimension == "relevance" and verdict != "relevant":
        verified_predicate = None
    if dimension == "answer_utility" and verified_predicate in {
        "exact_commit_overlap",
        "exact_path_overlap",
        "exact_task_uuid_overlap",
    }:
        verified_predicate = None
    return adjudicate_label(
        str(verdict),
        closed_predicate=verified_predicate,
        reason=str(response.get("reason") or response.get("rationale") or "")[:500],
        dimension=dimension,
    )


def _default_structural_verifier(
    rally: Mapping[str, Any],
    candidate: Mapping[str, Any],
    _response: Mapping[str, Any],
) -> str | None:
    query_ref = rally.get("query_ref")
    candidate_ref = candidate.get("ref")
    query_structural = (
        query_ref.get("structural") if isinstance(query_ref, Mapping) else None
    )
    candidate_structural = (
        candidate_ref.get("structural")
        if isinstance(candidate_ref, Mapping)
        else candidate.get("structural")
    )
    if not isinstance(query_structural, Mapping) or not isinstance(
        candidate_structural, Mapping
    ):
        return None
    for kind, predicate in (
        ("commit", "exact_commit_overlap"),
        ("task_uuid", "exact_task_uuid_overlap"),
        ("path", "exact_path_overlap"),
    ):
        query_values = query_structural.get(kind)
        candidate_values = candidate_structural.get(kind)
        if (
            isinstance(query_values, list)
            and isinstance(candidate_values, list)
            and set(query_values).intersection(candidate_values)
        ):
            return predicate
    return None


@dataclass(frozen=True)
class _TeacherBatchResult:
    labels_written: int = 0
    model_calls: int = 0
    deferred: bool = False


@dataclass(frozen=True)
class _CounterfactualBlockResult:
    pending: bool = False
    written: int = 0
    model_calls: int = 0
    deferred: bool = False


@dataclass(frozen=True)
class _CandidateCaptureResult:
    snapshots: dict[str, Mapping[str, Any]]
    work: list[Mapping[str, Any]]
    split_plan: Mapping[str, Any]
    deadline_deferred: bool


def _run_teacher_batch(
    *,
    root: Path,
    config: DistillationConfig,
    teachers: Mapping[str, Teacher],
    snapshots: Mapping[str, Mapping[str, Any]],
    rally_by_id: Mapping[str, Mapping[str, Any]],
    texts: Mapping[str, str],
    label_path: Path,
    label_rows: Sequence[Mapping[str, Any]],
    structural_verifier: Callable[
        [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], str | None
    ],
) -> _TeacherBatchResult:
    completed = {
        (str(row.get("rally_id")), str(row.get("candidate_id")), str(row.get("route")))
        for row in label_rows
    }
    pending_by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rally_id, snapshot in sorted(snapshots.items()):
        rally = rally_by_id.get(rally_id)
        if rally is None:
            continue
        candidates = list(snapshot.get("candidates", []))
        selected_candidates = candidates[:3]
        if candidates[3:]:
            selected_candidates.append(candidates[-1])
        for candidate in selected_candidates:
            assignment = teacher_assignment(rally_id, str(candidate["candidate_id"]))
            payload = _teacher_payload(
                rally,
                candidate,
                texts,
                max_input_bytes=config.max_input_bytes,
            )
            for route in assignment["routes"]:
                key = (rally_id, str(candidate["candidate_id"]), str(route))
                if key in completed or payload is None:
                    continue
                pending_by_route[str(route)].append(
                    {
                        "key": key,
                        "rally": rally,
                        "candidate": candidate,
                        "assignment": assignment,
                        "input": {
                            "candidate_id": candidate["candidate_id"],
                            "rally_id": rally_id,
                            "query": payload["query"],
                            "context": payload["context"],
                            "evidence": payload["candidate"],
                        },
                    }
                )
    # Exactly one cancellable batch per sleep chunk. Other pairs remain pending.
    for route in _ordered_teacher_routes(pending_by_route, label_rows):
        batch: list[dict[str, Any]] = []
        batch_candidate_ids: set[str] = set()
        for task in pending_by_route[route]:
            candidate_id = str(task["candidate"]["candidate_id"])
            if candidate_id in batch_candidate_ids:
                continue
            proposed = [*batch, task]
            worker_input = {
                "schema": "chronovisor.recall-distill-teacher-batch.v1",
                "candidates": [item["input"] for item in proposed],
            }
            if len(proposed) > 16 or len(
                canonical_json.canonical_json_bytes_strict(worker_input)
            ) > min(config.max_input_bytes, 12_000):
                break
            batch = proposed
            batch_candidate_ids.add(candidate_id)
        if not batch:
            continue
        response: Mapping[str, Any] = {}
        try:
            response = teachers[route].evaluate(
                {
                    "schema": "chronovisor.recall-distill-teacher-batch.v1",
                    "candidates": [item["input"] for item in batch],
                }
            )
            labels = response.get("labels")
            if not isinstance(labels, list) or {
                str(label.get("candidate_id"))
                for label in labels
                if isinstance(label, Mapping)
            } != {str(item["candidate"]["candidate_id"]) for item in batch}:
                raise DistillationError("teacher batch response coverage is invalid")
        except (DistillationDeferred, TimeoutError, OSError):
            return _TeacherBatchResult(deferred=True)
        except Exception:
            labels = [
                {
                    "candidate_id": item["candidate"]["candidate_id"],
                    "verdict": "uncertain",
                    "reason": "invalid_teacher_output",
                }
                for item in batch
            ]
        labels_by_id = {
            str(label["candidate_id"]): label
            for label in labels
            if isinstance(label, Mapping)
        }
        for task in batch:
            candidate = task["candidate"]
            rally = task["rally"]
            label_response = labels_by_id[str(candidate["candidate_id"])]
            predicate = structural_verifier(rally, candidate, label_response)
            if predicate not in CLOSED_PREDICATES:
                predicate = None
            label = _teacher_label(label_response, verified_predicate=predicate)
            store.append_chain(
                label_path,
                {
                    "kind": "teacher-label",
                    "rally_id": rally["rally_id"],
                    "candidate_id": candidate["candidate_id"],
                    "route": route,
                    "route_identity": response.get("_route_identity", {}),
                    "model_digest": response.get("_model_digest", ""),
                    "assignment": task["assignment"],
                    **label,
                },
            )
        return _TeacherBatchResult(labels_written=len(batch), model_calls=1)
    return _TeacherBatchResult()


def _run_counterfactual_block(
    *,
    execute: bool,
    root: Path,
    config: DistillationConfig,
    counterfactual: CounterfactualGenerator | None,
    snapshots: Mapping[str, Mapping[str, Any]],
    rally_by_id: Mapping[str, Mapping[str, Any]],
    texts: Mapping[str, str],
    label_path: Path,
    label_rows: Sequence[Mapping[str, Any]],
) -> _CounterfactualBlockResult:
    if counterfactual is None or not counterfactual.local:
        return _CounterfactualBlockResult()
    counterfactual_done = {
        (str(row.get("rally_id")), str(row.get("candidate_id")))
        for row in label_rows
        if row.get("route") == "counterfactual"
    }
    pending = False
    for rally_id, snapshot in sorted(snapshots.items()):
        rally = rally_by_id.get(rally_id)
        if rally is None or not rally.get("actual_answer_refs"):
            continue
        candidate_ids = set(
            str(item.get("candidate_id") or "")
            for item in snapshot.get("candidates", [])[:3]
            if isinstance(item, Mapping)
        )
        if any(
            candidate_id and (rally_id, candidate_id) not in counterfactual_done
            for candidate_id in candidate_ids
        ):
            pending = True
            break
    if not execute or not pending:
        return _CounterfactualBlockResult(pending=pending)

    for rally_id, snapshot in sorted(snapshots.items()):
        rally = rally_by_id.get(rally_id)
        if rally is None or not rally.get("actual_answer_refs"):
            continue
        exposure = (
            rally["exposure_receipts"][0] if rally.get("exposure_receipts") else {}
        )
        exposure_artifact: Mapping[str, Any] = {}
        if exposure:
            try:
                exposure_artifact = store.read_sealed(
                    store.distillation_dir(root)
                    / "exposures"
                    / f"{exposure['exposure_artifact_id']}.json",
                    schema="chronovisor.recall-exact-exposure.v1",
                )
            except (KeyError, store.DistillationStoreError):
                exposure = {}
                exposure_artifact = {}
        exact_candidates = exposure_artifact.get("candidate_refs", [])
        raw_feature_rows = [
            *(
                exposure_artifact.get("candidate_feature_snapshot", [])
                if isinstance(
                    exposure_artifact.get("candidate_feature_snapshot", []), list
                )
                else []
            ),
            *[
                {
                    "candidate_id": item.get("candidate_id"),
                    "features": item.get("features"),
                }
                for item in snapshot.get("candidates", [])
                if isinstance(item, Mapping)
                and item.get("feature_revision") == TEXT_FEATURE_REVISION
            ],
        ]
        if not isinstance(exact_candidates, list):
            exact_candidates = []
        exact_features = {
            str(row["candidate_id"]): dict(row["features"])
            for row in raw_feature_rows
            if isinstance(row, Mapping)
            and isinstance(row.get("candidate_id"), str)
            and isinstance(row.get("features"), Mapping)
            and set(row["features"]) == set(FAST_FEATURE_KEYS)
        }
        original_evidence: dict[str, str] = {}
        removal_candidates: list[dict[str, Any]] = []
        for item in exact_candidates:
            if not isinstance(item, Mapping):
                continue
            candidate_id = str(item.get("candidate_id") or "")
            rendered = item.get("rendered_context")
            evidence_refs = item.get("evidence_refs")
            if not isinstance(rendered, str) or not rendered:
                if not isinstance(evidence_refs, list):
                    continue
                parts = [
                    texts.get(str(ref.get("semantic_sha256") or ""), "")
                    for ref in evidence_refs
                    if isinstance(ref, Mapping)
                ]
                if not parts or any(not part for part in parts):
                    continue
                rendered = "\n".join(parts)
            original_evidence[candidate_id] = rendered
            removal_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "text_sha256": item.get("content_sha256"),
                    "ref": evidence_refs[0]
                    if isinstance(evidence_refs, list) and evidence_refs
                    else {"structural": {}},
                    "rendered_context": rendered,
                }
            )
        if exposure and set(original_evidence) != set(exposure["candidate_ids"]):
            continue
        live_additions = [
            {
                "candidate_id": item["candidate_id"],
                "rendered_context": item["rendered_context"],
                "ref": {"structural": {}},
            }
            for item in exposure_artifact.get("candidate_pool_refs", [])
            if isinstance(item, Mapping) and item.get("selected") is False
        ]
        historical_additions = [
            candidate
            for candidate in snapshot.get("candidates", [])[:3]
            if candidate["candidate_id"] not in original_evidence
            and candidate["candidate_id"]
            not in {item["candidate_id"] for item in live_additions}
        ]
        for candidate in [
            *removal_candidates,
            *live_additions,
            *historical_additions,
        ]:
            candidate_id = str(candidate["candidate_id"])
            if (rally_id, candidate_id) in counterfactual_done:
                continue
            candidate_text = candidate.get("rendered_context") or texts.get(
                str(candidate.get("text_sha256") or ""), ""
            )
            if not isinstance(candidate_text, str) or not candidate_text:
                continue
            mode = "remove" if candidate_id in original_evidence else "add"
            context = [
                texts.get(str(ref["semantic_sha256"]), "")
                for ref in rally.get("context_refs", [])
            ]
            original = list(original_evidence.values())
            payload = {
                "rally_id": rally_id,
                "candidate_id": candidate_id,
                "query": texts.get(str(rally["query_sha256"]), ""),
                "context": context,
                "actual_answer": "\n".join(
                    texts.get(str(ref["semantic_sha256"]), "")
                    for ref in rally.get("actual_answer_refs", [])
                ),
                "a0_evidence": [
                    value
                    for key, value in original_evidence.items()
                    if key != candidate_id
                ]
                if mode == "remove"
                else original,
                "a1_evidence": original
                if mode == "remove"
                else [*original, candidate_text],
                "mode": mode,
            }
            while (
                context
                and len(canonical_json.canonical_json_bytes_strict(payload))
                > config.max_input_bytes
            ):
                context.pop(0)
            if (
                len(canonical_json.canonical_json_bytes_strict(payload))
                > config.max_input_bytes
            ):
                continue
            response: Mapping[str, Any] = {}
            try:
                response = counterfactual.compare(payload)
                generator_digest = str(response.get("generator_model_digest") or "")
                judge_digest = str(response.get("judge_model_digest") or "")
                verdict = str(response.get("verdict") or "uncertain")
                if (
                    response.get("order_agreement") is not True
                    or re.fullmatch(r"[0-9a-f]{64}", generator_digest) is None
                    or re.fullmatch(r"[0-9a-f]{64}", judge_digest) is None
                    or generator_digest == judge_digest
                ):
                    verdict = "uncertain"
                label = adjudicate_label(
                    verdict,
                    closed_predicate=None,
                    reason=str(response.get("reason") or "")[:500],
                    dimension="answer_utility",
                )
            except (DistillationDeferred, TimeoutError, OSError):
                return _CounterfactualBlockResult(pending=True, deferred=True)
            except Exception:
                label = adjudicate_label(
                    "uncertain",
                    closed_predicate=None,
                    reason="counterfactual_failed",
                )
            store.append_chain(
                label_path,
                {
                    "kind": "counterfactual-label",
                    "rally_id": rally_id,
                    "candidate_id": candidate["candidate_id"],
                    "route": "counterfactual",
                    "mode": mode,
                    "exposure_artifact_id": str(
                        exposure.get("exposure_artifact_id") or ""
                    ),
                    "a0_sha256": response.get("a0_sha256", ""),
                    "a1_sha256": response.get("a1_sha256", ""),
                    "blind_orders": response.get("blind_orders", []),
                    "order_agreement": response.get("order_agreement", False),
                    "generator_route_identity": response.get(
                        "generator_route_identity", {}
                    ),
                    "generator_model_digest": response.get(
                        "generator_model_digest", ""
                    ),
                    "judge_route_identity": response.get("judge_route_identity", {}),
                    "judge_model_digest": response.get("judge_model_digest", ""),
                    **(
                        {"features": exact_features[candidate_id]}
                        if candidate_id in exact_features
                        else {}
                    ),
                    **label,
                },
            )
            return _CounterfactualBlockResult(pending=True, written=1, model_calls=1)
    return _CounterfactualBlockResult(pending=pending)


def _capture_candidate_snapshots(
    *,
    root: Path,
    raw_dir: Path,
    config: DistillationConfig,
    rallies: Sequence[Mapping[str, Any]],
    texts: Mapping[str, str],
    index_path: Path,
    cold_start: bool,
    deadline: float,
) -> _CandidateCaptureResult:
    candidate_path = store.distillation_dir(root) / "candidate-ledger.jsonl"
    snapshots = {
        str(row["rally_id"]): row["snapshot"]
        for row in store.read_chain(candidate_path)
        if isinstance(row.get("snapshot"), dict)
    }
    candidate_limit = 100 if cold_start else config.chunk_size
    planned = [rally for rally in rallies if rally["rally_id"] not in snapshots]
    split_plan: Mapping[str, Any] = {}
    if cold_start:
        _, cohort = _active_training_cohort(materialize_training_rows(root)["rows"])
        split_plan = _ensure_split_plan(
            root,
            rallies,
            raw_watermark=committed_raw_watermark(raw_dir),
            model_cohort_sha256=cohort["cohort_sha256"],
        )
        assignments = split_plan["assignments"]
        planned.sort(
            key=lambda rally: (
                0 if assignments.get(rally["rally_id"]) == "test" else 1,
                str(rally["rally_id"]),
            )
        )
    planned = planned[:candidate_limit]
    work: list[Mapping[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    deferred = False
    for rally in planned:
        if cold_start and deadline - time.monotonic() < 30:
            deferred = True
            break
        snapshot = candidate_snapshot(
            index_path,
            rally,
            texts.get(str(rally["query_sha256"]), ""),
            limit=config.max_candidates,
            candidate_texts=texts,
        )
        payloads.append(
            {
                "kind": "candidate-snapshot",
                "rally_id": rally["rally_id"],
                "snapshot": snapshot,
            }
        )
        snapshots[str(rally["rally_id"])] = snapshot
        work.append(rally)
    store.append_chain_batch(candidate_path, payloads)
    return _CandidateCaptureResult(snapshots, work, split_plan, deferred)


def _worker_state_transition(
    root: Path,
    *,
    p5_allowed: bool,
    local_teachers: bool,
    model_deferred: bool,
    gate_baseline: Mapping[str, Any],
    promotion: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        previous = _read_worker_state(root)
    except store.DistillationStoreError:
        previous = {"kind": "worker-state"}
    rollout_status = str(previous.get("status") or "")
    if rollout_status not in {
        "shadow",
        "replay",
        "canary",
        "active",
        "rolled_back",
        "quarantined",
    }:
        rollout_status = "ready" if p5_allowed else "capture_only"
    hold_reason = str(previous.get("hold_reason") or "")
    if rollout_status in {"ready", "capture_only"}:
        reasons = gate_baseline["hard_floor"]["reasons"]
        hold_reason = str((reasons[0] if reasons else promotion.get("reason")) or "")
    return {
        "previous": previous,
        "worker_status": (
            "deferred"
            if model_deferred
            else "ready"
            if p5_allowed and local_teachers
            else "capture_only"
        ),
        "rollout_status": rollout_status,
        "hold_reason": hold_reason,
        "last_success_at": (
            str(previous.get("last_success_at") or "")
            if model_deferred
            else datetime.now(UTC).isoformat()
        ),
    }


def _run_distillation_chunk_impl(
    *,
    root: Path | None = None,
    raw_dir: Path | None = None,
    config_path: Path | None = None,
    teachers: Mapping[str, Teacher] | None = None,
    counterfactual: CounterfactualGenerator | None = None,
    structural_verifier: (
        Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], str | None]
        | None
    ) = None,
    dry_run: bool = False,
    cold_start: bool = False,
    max_elapsed_seconds: int = 300,
) -> dict[str, Any]:
    root = root or CHRONOVISOR_ROOT
    raw_dir = raw_dir or root / "raw"
    config = load_distillation_config(config_path)
    deadline = time.monotonic() + max_elapsed_seconds
    if not config.enabled:
        return {"status": "disabled", "processed": 0}
    try:
        scheduler_state = _read_worker_state(root)
    except store.DistillationStoreError:
        scheduler_state = {}
    teacher_model_calls = int(scheduler_state.get("teacher_model_calls", 0))
    counterfactual_model_calls = int(
        scheduler_state.get("counterfactual_model_calls", 0)
    )
    # ponytail: cold start rescans Raw; add a sealed projection catalog only if the
    # 5-minute lane ceiling is measured to fail.
    event_rows = _events(raw_dir)
    rallies = extract_rallies(
        raw_dir,
        root=root,
        max_context_bytes=config.max_input_bytes,
        _event_rows=event_rows,
    )
    if teachers is None:
        teachers, default_counterfactual = _default_workers(
            config, deadline_ms=45_000 if cold_start else 60_000
        )
        if counterfactual is None:
            counterfactual = default_counterfactual
    structural_verifier = structural_verifier or _default_structural_verifier
    ledger_path = store.distillation_dir(root) / "rally-manifest.jsonl"
    existing = {
        manifest.get("rally_id")
        for row in store.read_chain(ledger_path)
        if isinstance((manifest := row.get("manifest")), dict)
    }
    manifest_limit = 500 if cold_start else config.chunk_size
    pending = [row for row in rallies if row["rally_id"] not in existing][
        :manifest_limit
    ]
    local_teachers = bool(teachers) and all(
        role in teachers and teachers[role].local for role in TEACHER_ROLES
    )
    if dry_run:
        return {
            "status": "dry_run",
            "pending": len(pending),
            "teachers_available": local_teachers,
        }
    store.append_chain_batch(
        ledger_path,
        ({"kind": "rally-manifest", "manifest": rally} for rally in pending),
    )
    index_path = store.distillation_dir(root) / "historical-index.sqlite"
    index_sha256 = build_historical_index(raw_dir, index_path, _event_rows=event_rows)
    texts = {str(row["semantic_sha256"]): str(row["text"]) for row in event_rows}
    rally_by_id = {str(rally["rally_id"]): rally for rally in rallies}
    candidate_path = store.distillation_dir(root) / "candidate-ledger.jsonl"
    capture = _capture_candidate_snapshots(
        root=root,
        raw_dir=raw_dir,
        config=config,
        rallies=rallies,
        texts=texts,
        index_path=index_path,
        cold_start=cold_start,
        deadline=deadline,
    )
    snapshots = capture.snapshots
    candidate_work = capture.work
    split_plan = capture.split_plan
    deadline_deferred = capture.deadline_deferred

    label_path = store.distillation_dir(root) / "label-ledger.jsonl"
    label_rows = store.read_chain(label_path)
    model_snapshots = snapshots
    if cold_start and split_plan:
        prioritize_test = int(scheduler_state.get("cold_start_lane_turn", 0)) % 4 != 3
        if prioritize_test:
            assignments = split_plan["assignments"]
            model_snapshots = {
                rally_id: snapshot
                for rally_id, snapshot in snapshots.items()
                if assignments.get(rally_id) == "test"
            }
    counterfactual_probe = _run_counterfactual_block(
        execute=False,
        root=root,
        config=config,
        counterfactual=counterfactual,
        snapshots=model_snapshots,
        rally_by_id=rally_by_id,
        texts=texts,
        label_path=label_path,
        label_rows=label_rows,
    )
    prefer_counterfactual = _is_counterfactual_turn(
        teacher_model_calls,
        counterfactual_model_calls,
        available=counterfactual_probe.pending,
    )
    teacher_result = _TeacherBatchResult()
    minimum_model_seconds = 220 if prefer_counterfactual else 55
    model_work_available = local_teachers or counterfactual_probe.pending
    model_deferred = deadline_deferred or (
        model_work_available
        and cold_start
        and deadline - time.monotonic() < minimum_model_seconds
    )
    if local_teachers and not prefer_counterfactual and not model_deferred:
        teacher_result = _run_teacher_batch(
            root=root,
            config=config,
            teachers=teachers or {},
            snapshots=model_snapshots,
            rally_by_id=rally_by_id,
            texts=texts,
            label_path=label_path,
            label_rows=label_rows,
            structural_verifier=structural_verifier,
        )
    labels_written = teacher_result.labels_written
    teacher_model_calls += teacher_result.model_calls
    model_deferred = model_deferred or teacher_result.deferred
    counterfactual_result = _CounterfactualBlockResult(
        pending=counterfactual_probe.pending
    )
    if (
        not model_deferred
        and labels_written == 0
        and (
            not cold_start
            or not counterfactual_probe.pending
            or deadline - time.monotonic() >= 220
        )
    ):
        counterfactual_result = _run_counterfactual_block(
            execute=True,
            root=root,
            config=config,
            counterfactual=counterfactual,
            snapshots=model_snapshots,
            rally_by_id=rally_by_id,
            texts=texts,
            label_path=label_path,
            label_rows=label_rows,
        )
    elif (
        not model_deferred
        and labels_written == 0
        and counterfactual_probe.pending
        and cold_start
    ):
        model_deferred = True
    counterfactual_written = counterfactual_result.written
    counterfactual_model_calls += counterfactual_result.model_calls
    model_deferred = model_deferred or counterfactual_result.deferred
    if cold_start:
        _, current_cohort = _active_training_cohort(
            materialize_training_rows(root)["rows"]
        )
        split_plan = _ensure_split_plan(
            root,
            rallies,
            raw_watermark=committed_raw_watermark(raw_dir),
            model_cohort_sha256=current_cohort["cohort_sha256"],
        )
    baseline = preflight(
        raw_dir=raw_dir,
        root=root,
        config_path=config_path,
        _rallies=rallies,
    )
    gate_baseline = _matching_p5_baseline(root, baseline) or baseline
    p5_allowed = bool(gate_baseline["hard_floor"]["p5_allowed"])
    manifest_backlog = max(0, len(rallies) - len(existing) - len(pending))
    candidate_backlog = max(0, len(rallies) - len(snapshots))
    cold_start_pending = bool(manifest_backlog or candidate_backlog or not p5_allowed)
    split_plan_id = str(
        split_plan.get("artifact_id") or scheduler_state.get("split_plan_id") or ""
    )
    bootstrap = _ensure_bootstrap_policy(root, gate_baseline)
    promotion = _maybe_publish_candidate(root, config, gate_baseline)
    rollout_evaluation = _automatic_rollout_evaluation(root, gate_baseline, promotion)
    transition = _worker_state_transition(
        root,
        p5_allowed=p5_allowed,
        local_teachers=local_teachers,
        model_deferred=model_deferred,
        gate_baseline=gate_baseline,
        promotion=promotion,
    )
    manifest_head = store.verify_chain(ledger_path)["head_sha256"]
    candidate_head = store.verify_chain(candidate_path)["head_sha256"]
    label_head = store.verify_chain(label_path)["head_sha256"]
    run_id, _, _ = store.write_immutable(
        store.distillation_dir(root) / "runs",
        {
            "kind": "bounded-chunk",
            "raw_watermark": baseline["raw_watermark"],
            "baseline_artifact_id": gate_baseline["artifact_id"],
            "manifest_head": manifest_head,
            "candidate_head": candidate_head,
            "label_head": label_head,
            "processed": len(pending),
            "candidate_snapshots": len(candidate_work),
            "labels_written": labels_written,
            "counterfactuals_written": counterfactual_written,
            "p5_allowed": p5_allowed,
        },
        schema="chronovisor.recall-distill-run.v1",
    )
    # ponytail: the final small state commit is uninterruptible; isolate the worker
    # process only if a storage stall at this boundary is measured.
    if (
        threading.current_thread() is threading.main_thread()
        and hasattr(signal, "setitimer")
        and hasattr(signal, "ITIMER_REAL")
    ):
        signal.setitimer(signal.ITIMER_REAL, 0)
    state = store.write_sealed_state(
        store.distillation_dir(root) / store.STATE_FILE,
        {
            **transition["previous"],
            "status": transition["rollout_status"],
            "worker_status": transition["worker_status"],
            "rollout_percent": int(transition["previous"].get("rollout_percent", 0)),
            "raw_watermark": baseline["raw_watermark"],
            "baseline_artifact_id": gate_baseline["artifact_id"],
            "historical_index_sha256": index_sha256,
            "manifest_chain_head": manifest_head,
            "run_id": run_id,
            "processed": len(pending),
            "candidate_snapshots": len(candidate_work),
            "labels_written": labels_written,
            "counterfactuals_written": counterfactual_written,
            "teacher_model_calls": teacher_model_calls,
            "counterfactual_model_calls": counterfactual_model_calls,
            "cold_start_pending": cold_start_pending,
            "cold_start_lane_turn": int(
                scheduler_state.get("cold_start_lane_turn", 0)
            )
            + int(cold_start and not model_deferred),
            "split_plan_id": split_plan_id,
            "manifest_backlog": manifest_backlog,
            "candidate_backlog": candidate_backlog,
            "promotion_status": promotion["status"],
            "promotion_reason": promotion.get("reason", ""),
            "incumbent_policy_id": bootstrap["artifact_id"],
            "rollout_evaluation_status": rollout_evaluation["status"],
            "hold_reason": transition["hold_reason"],
            "capture_only_reasons": gate_baseline["hard_floor"]["reasons"]
            + ([] if local_teachers else ["local_teachers_unavailable"]),
            "last_success_at": transition["last_success_at"],
            "error_code": "worker_deferred" if model_deferred else "",
        },
    )
    return {
        "status": transition["worker_status"],
        "processed": len(pending),
        "p5_allowed": p5_allowed,
        "teachers_available": local_teachers,
        "counterfactual_available": bool(counterfactual and counterfactual.local),
        "candidate_snapshots": len(candidate_work),
        "labels_written": labels_written,
        "counterfactuals_written": counterfactual_written,
        "cold_start_pending": cold_start_pending,
        "split_plan_id": split_plan_id,
        "manifest_backlog": manifest_backlog,
        "candidate_backlog": candidate_backlog,
        "promotion": promotion,
        "rollout_evaluation": rollout_evaluation,
        "run_id": run_id,
        "state_sha256": state["seal_sha256"],
    }


def cold_start_due(root: Path | None = None) -> bool:
    """Return a scan-free, fail-closed hint for the existing converge lane."""

    root = root or CHRONOVISOR_ROOT
    if not _enabled_for_root(root):
        return False
    state_path = store.distillation_dir(root) / store.STATE_FILE
    if not state_path.exists():
        return True
    try:
        state = _read_worker_state(root)
    except store.DistillationStoreError:
        return False
    if state.get("status") == "active":
        return False
    return state.get("cold_start_pending") is not False


def run_distillation_chunk(
    *,
    root: Path | None = None,
    raw_dir: Path | None = None,
    config_path: Path | None = None,
    teachers: Mapping[str, Teacher] | None = None,
    counterfactual: CounterfactualGenerator | None = None,
    structural_verifier: (
        Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], str | None]
        | None
    ) = None,
    dry_run: bool = False,
    cold_start: bool = False,
    max_elapsed_seconds: int = 300,
) -> dict[str, Any]:
    """Run one bounded single-writer distillation unit."""

    if (
        isinstance(max_elapsed_seconds, bool)
        or not isinstance(max_elapsed_seconds, int)
        or not 60 <= max_elapsed_seconds <= 600
    ):
        raise DistillationError("distillation elapsed limit is invalid")
    from chronovisor.recall.recall_runtime import (
        RecallWallClockTimeout,
        recall_wall_clock_deadline,
    )

    resolved_root = root or CHRONOVISOR_ROOT
    lock = store.acquire_nonblocking_lock(
        store.distillation_dir(resolved_root) / "distillation-worker.lock"
    )
    if lock is None:
        return {"status": "deferred", "processed": 0, "reason": "worker_busy"}
    try:
        try:
            with recall_wall_clock_deadline(max_elapsed_seconds * 1_000):
                return _run_distillation_chunk_impl(
                    root=resolved_root,
                    raw_dir=raw_dir,
                    config_path=config_path,
                    teachers=teachers,
                    counterfactual=counterfactual,
                    structural_verifier=structural_verifier,
                    dry_run=dry_run,
                    cold_start=cold_start,
                    max_elapsed_seconds=max_elapsed_seconds,
                )
        except RecallWallClockTimeout:
            # ponytail: no cross-ledger transaction; completed atomic batches are
            # idempotent resume points and are never rolled back after timeout.
            return {
                "status": "deferred",
                "processed": 0,
                "reason": "wall_clock_timeout",
                "atomic_progress_may_be_present": True,
            }
    finally:
        store.release_lock(lock)


def distillation_snapshot(root: Path | None = None) -> dict[str, Any]:
    return store.snapshot(root or CHRONOVISOR_ROOT)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subcommands.add_parser("preflight")
    preflight_parser.add_argument("--root", type=Path, default=CHRONOVISOR_ROOT)
    preflight_parser.add_argument("--raw-dir", type=Path)
    preflight_parser.add_argument("--config", type=Path)
    preflight_parser.add_argument("--runtime-commit", default="")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight(
            raw_dir=args.raw_dir or args.root / "raw",
            root=args.root,
            config_path=args.config,
            runtime_commit=args.runtime_commit,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
