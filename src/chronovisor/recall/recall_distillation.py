"""Autonomous, point-in-time Recall distillation contracts."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib
import json
import math
import os
import re
import signal
import sqlite3
import sys
import threading
import time
import tomllib
import unicodedata
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, cast

from chronovisor.core import (
    canonical_json,
    claude_code_transcript,
    codex_transcript,
    pi_transcript,
    runtime_config,
)
from chronovisor.core.durable_state import atomic_write_bytes, sidecar_exclusive_lock
from chronovisor.core.raw_store import (
    RawSegmentCorrupt,
    RawStore,
    committed_event_spans,
    committed_raw_watermark,
)
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.recall import recall_distillation_store as store
from chronovisor.recall.recall_calibration import sigmoid
from chronovisor.recall.recall_distillation_remote_teacher import (
    OX_ALPHA_FIXED_IDENTITY,
    OpenCodeOxAlphaTeacher,
    ox_alpha_response_metadata,
    ox_alpha_source_binding,
    validate_ox_alpha_labels,
)
from chronovisor.recall.recall_distillation_single_teacher_gate import (
    evaluate_single_teacher_gate,
    expected_ox_provider_request_sha256,
    expected_ox_request_sha256,
)

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
OX_TEACHER_ROLE = "recall.distill.teacher.ox-alpha"
LOCAL_TRIAD_PROFILE = "local-triad-v1"
OX_SINGLE_PROFILE = "ox-alpha-single-v1"
OX_SINGLE_COHORT = "ox-alpha-backfill-v1"
OX_PROBE_REVISION = "single-teacher-repeat-v2"
OX_RAMP_REQUEST_REVISION = "json-schema-core-label-abstain-16k-240s-v6"
TEACHER_PROFILES = frozenset({LOCAL_TRIAD_PROFILE, OX_SINGLE_PROFILE})
OX_ALPHA_ENDPOINT = "https://opencode.ai/zen/go/v1"
OX_ALPHA_CREDENTIAL_REF = "oskeyring:codex-router-opencode-go/default"
OX_PROFILE_SCHEMA = "chronovisor.recall-distill-ox-profile.v1"
R4_CANDIDATE_ANCHOR_SCHEMA = "chronovisor.recall-r4-candidate-anchor.v1"
R4_CANDIDATE_ANCHOR_FILE = "r4-candidate-anchor.json"
R4_R0_EVIDENCE_ID = "4de2cfe3f33e5c9c5153b264ebee8fae24d814856e0ac339e53c3077dc7efb33"
OX_RAMP_RECEIPTS_PER_CAP = 20
_OX_EXPIRY_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})\Z"
)
_OX_MAX_EXPIRY = datetime(2100, 1, 1, tzinfo=UTC)
# Deterministic local payload rejects may be skipped in one run, but never
# indefinitely: after this many extra claims, leave the remainder ready.
OX_PREFLIGHT_SCAN_CLAIM_BUDGET = 500
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
REPLAY_OBSERVATION_SCHEMA = "chronovisor.recall-rollout-replay-observation.v1"
SHADOW_PRODUCER_NAME = "chronovisor.recall-runtime"
SHADOW_PRODUCER_VERSION = 1
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


# This is deliberately a value object rather than an arbitrary Mapping.  The
# live writer is the only code allowed to construct operational evidence;
# callers cannot smuggle self-reported gate values into the sealed ledger.
@dataclass(frozen=True)
class ShadowOperationalEvidence:
    candidate_quality: bool
    baseline_quality: bool
    candidate_covered: bool
    baseline_covered: bool
    candidate_anchor_retained: bool
    baseline_anchor_retained: bool
    candidate_abstained: bool
    baseline_abstained: bool
    candidate_score_ms: int
    live_latency_ms: int
    resource_ok: bool
    integrity_ok: bool
    negative_veto: bool
    deadline_ms: int
    stage: str
    run_id: str
    cohort: str
    host: str
    producer_name: str = SHADOW_PRODUCER_NAME
    producer_version: int = SHADOW_PRODUCER_VERSION
    synthetic_fixture: bool = False
    pair_id: str = ""
    candidate_decision_sha256: str = ""
    baseline_decision_sha256: str = ""
    candidate_pool_sha256: str = ""
    baseline_pool_sha256: str = ""
    candidate_feature_snapshot_sha256: str = ""
    baseline_feature_snapshot_sha256: str = ""
    candidate_feature_bytes_sha256: str = ""
    baseline_feature_bytes_sha256: str = ""
    feature_snapshot_sha256: str = ""
    feature_parity: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the one closed wire schema used by runtime evidence."""

        return {
            "candidate_quality": self.candidate_quality,
            "baseline_quality": self.baseline_quality,
            "candidate_covered": self.candidate_covered,
            "baseline_covered": self.baseline_covered,
            "candidate_anchor_retained": self.candidate_anchor_retained,
            "baseline_anchor_retained": self.baseline_anchor_retained,
            "candidate_abstained": self.candidate_abstained,
            "baseline_abstained": self.baseline_abstained,
            "candidate_score_ms": self.candidate_score_ms,
            "live_latency_ms": self.live_latency_ms,
            "resource_ok": self.resource_ok,
            "integrity_ok": self.integrity_ok,
            "negative_veto": self.negative_veto,
            "deadline_ms": self.deadline_ms,
            "producer": {
                "name": self.producer_name,
                "version": self.producer_version,
                "synthetic_fixture": self.synthetic_fixture,
            },
            "stage": self.stage,
            "run_id": self.run_id,
            "cohort": self.cohort,
            "host": self.host,
            "pair_id": self.pair_id,
            "candidate_decision_sha256": self.candidate_decision_sha256,
            "baseline_decision_sha256": self.baseline_decision_sha256,
            "candidate_pool_sha256": self.candidate_pool_sha256,
            "baseline_pool_sha256": self.baseline_pool_sha256,
            "candidate_feature_snapshot_sha256": self.candidate_feature_snapshot_sha256,
            "baseline_feature_snapshot_sha256": self.baseline_feature_snapshot_sha256,
            "candidate_feature_bytes_sha256": self.candidate_feature_bytes_sha256,
            "baseline_feature_bytes_sha256": self.baseline_feature_bytes_sha256,
            "feature_snapshot_sha256": self.feature_snapshot_sha256,
            "feature_parity": self.feature_parity,
        }


_OPERATIONAL_EVIDENCE_KEYS = frozenset(
    ShadowOperationalEvidence(
        candidate_quality=False,
        baseline_quality=False,
        candidate_covered=False,
        baseline_covered=False,
        candidate_anchor_retained=False,
        baseline_anchor_retained=False,
        candidate_abstained=True,
        baseline_abstained=True,
        candidate_score_ms=0,
        live_latency_ms=0,
        resource_ok=False,
        integrity_ok=False,
        negative_veto=False,
        deadline_ms=1,
        stage="",
        run_id="",
        cohort="",
        host="",
    ).to_dict()
)

_DISTILLATION_ROLES = {
    "recall.distill.teacher.a": {
        "capability": "generation",
        "provider": "local",
        "model": runtime_config.DEFAULT_DECISION_PRIMARY_MODEL,
        "required_capabilities": ["structured_output"],
    },
    "recall.distill.teacher.b": {
        "capability": "generation",
        "provider": "local",
        "model": "gpt-oss:20b",
        "required_capabilities": ["structured_output"],
    },
    "recall.distill.teacher.c": {
        "capability": "generation",
        "provider": "local",
        "model": runtime_config.DEFAULT_DECISION_TIE_BREAK_MODEL,
        "required_capabilities": ["structured_output"],
    },
    "recall.distill.answer_generator": {
        "capability": "generation",
        "provider": "local",
        "model": runtime_config.DEFAULT_DECISION_PRIMARY_MODEL,
        "required_capabilities": ["structured_output"],
    },
    "recall.distill.utility_judge": {
        "capability": "generation",
        "provider": "local",
        "model": runtime_config.DEFAULT_DECISION_TIE_BREAK_MODEL,
        "required_capabilities": ["structured_output"],
    },
}


class DistillationError(ValueError):
    """A distillation input violates a deterministic or safety contract."""


class DistillationDeferred(RuntimeError):
    """A foreground-safe worker call was deferred and must remain resumable."""

    def __init__(
        self,
        message: str = "",
        *,
        failure_class: str = "deferred",
        attempted: bool = False,
    ) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.attempted = attempted


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
        mode="sleep",
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
            raise DistillationDeferred(
                f"distillation worker {outcome.status}",
                failure_class=outcome.status,
                attempted=outcome.status not in {"deferred", "cancelled"},
            )
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
            raise DistillationDeferred(
                f"distillation worker {failure_class}",
                failure_class=failure_class,
                attempted=failure_class
                not in {
                    "capacity_unavailable",
                    "foreground_preempted",
                    "resource_busy",
                    "cancelled",
                },
            )
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
    teacher_deadline_ms: int = 60_000,
    counterfactual_deadline_ms: int = 60_000,
) -> tuple[dict[str, Teacher], CounterfactualGenerator | None]:
    if config.teacher_profile == OX_SINGLE_PROFILE:
        if not config.ox_enabled or not config.ox_free_only:
            return {}, None
        try:
            from chronovisor.core.llm_config import (
                compose_remote_generation_backend,
            )
            from chronovisor.core.llm_security import CredentialRef, CredentialResolver
            from chronovisor.core.provider_profiles import generic_openai_profile
            from chronovisor.recall.recall_distillation_remote_teacher import (
                OpenCodeOxAlphaTeacher,
            )

            profile = generic_openai_profile(
                "opencode-go",
                OX_ALPHA_ENDPOINT,
                CredentialRef.parse(OX_ALPHA_CREDENTIAL_REF),
                structured_output_models={"ox-alpha-free"},
            )
            resolver = CredentialResolver()
            # Do not advertise a runnable remote teacher until the keyring entry
            # is present. The secret is intentionally neither retained nor logged.
            credential = resolver.resolve(profile.credential_ref)
            del credential
            teacher = OpenCodeOxAlphaTeacher(
                compose_remote_generation_backend(profile, resolver),
                enabled=True,
                free_only=True,
                allow_paid_fallback=False,
                max_input_bytes=config.max_input_bytes,
                timeout_ms=max(240_000, teacher_deadline_ms),
            )
        except Exception:
            return {}, None
        # Counterfactuals remain local even while OX supplies the temporary
        # relevance teacher.  They never receive the remote backend.
        try:
            from chronovisor.core import ollama

            roles = (
                "recall.distill.answer_generator",
                "recall.distill.utility_judge",
            )
            routes = ollama.runtime_generation_routes(roles)
            digests = ollama.runtime_generation_route_fingerprints(routes)
            identities = {
                route.role: {
                    "role": route.role,
                    "provider": route.provider,
                    "model": route.model,
                    "location": route.location,
                }
                for route in routes
            }
            counterfactual = (
                _WorkerCounterfactual(
                    config.max_input_bytes,
                    identities,
                    digests,
                    counterfactual_deadline_ms,
                )
                if tuple(route.role for route in routes) == roles
                and all(
                    route.location == "local" and route.structured_output
                    for route in routes
                )
                and digests[roles[0]] != digests[roles[1]]
                else None
            )
        except Exception:
            counterfactual = None
        return {teacher.role: teacher}, counterfactual

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
        route.location != "local" or not route.structured_output for route in routes
    ):
        return {}, None
    try:
        digests = ollama.runtime_generation_route_fingerprints(routes)
    except Exception:
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
    if len({digests[role] for role in TEACHER_ROLES}) != len(TEACHER_ROLES):
        return {}, None
    return (
        {
            role: _WorkerTeacher(
                role,
                config.max_input_bytes,
                identities[role],
                digests[role],
                teacher_deadline_ms,
            )
            for role in TEACHER_ROLES
        },
        _WorkerCounterfactual(
            config.max_input_bytes,
            identities,
            digests,
            counterfactual_deadline_ms,
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
    teacher_profile: str = LOCAL_TRIAD_PROFILE
    teacher_max_inflight: int = 10
    teacher_claim_limit: int = 500
    ox_enabled: bool = False
    ox_free_only: bool = True
    ox_expires_at: str = "2099-01-01T00:00:00Z"


def _ox_expiry(value: object) -> str:
    """Return a strict future UTC RFC3339 expiry, or fail before egress."""

    if (
        not isinstance(value, str)
        or _OX_EXPIRY_RE.fullmatch(value) is None
    ):
        raise DistillationError("OX expiry must be strict UTC RFC3339")
    try:
        expires_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise DistillationError("OX expiry must be strict UTC RFC3339") from exc
    if expires_at.tzinfo is None or expires_at.utcoffset() is None:
        raise DistillationError("OX expiry must be strict UTC RFC3339")
    normalized = expires_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if expires_at <= datetime.now(UTC) or expires_at >= _OX_MAX_EXPIRY:
        raise DistillationError("OX expiry is not in the future")
    return normalized


def _same_future_ox_expiry(value: object, expected: object) -> bool:
    try:
        value_normalized = _ox_expiry(value)
        expected_normalized = _ox_expiry(expected)
        return (
            value_normalized == value
            and value_normalized == expected_normalized
        )
    except DistillationError:
        return False


def _canonical_hard_floors() -> dict[str, int]:
    defaults = DistillationConfig()
    return {
        name: getattr(defaults, name)
        for name in (
            "hard_floor_rallies",
            "hard_floor_days",
            "hard_floor_windows",
            "hard_floor_teacher_labels",
            "hard_floor_teacher_per_class",
            "hard_floor_probe_pairs",
            "hard_floor_counterfactual_pairs",
        )
    }


def _has_canonical_hard_floors(config: DistillationConfig) -> bool:
    return all(
        getattr(config, name) >= value
        for name, value in _canonical_hard_floors().items()
    )


def _validate_ox_source_binding(value: Mapping[str, str] | None) -> dict[str, str]:
    source = dict(value or {})
    if (
        set(source)
        != {"source_commit", "source_tree_sha256", "source_ox_identity_sha256"}
        or re.fullmatch(r"[0-9a-f]{40}", source.get("source_commit", "")) is None
        or any(
            re.fullmatch(r"[0-9a-f]{64}", source.get(key, "")) is None
            for key in ("source_tree_sha256", "source_ox_identity_sha256")
        )
    ):
        raise DistillationError("OX source binding is invalid")
    return source


def _ensure_ox_profile_contract(
    root: Path,
    config: DistillationConfig,
    *,
    source_binding: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Seal the exact temporary remote-teacher boundary before any egress."""

    if (
        config.teacher_profile != OX_SINGLE_PROFILE
        or config.ox_enabled is not True
        or config.ox_free_only is not True
        or not 1 <= config.teacher_max_inflight <= 10
    ):
        raise DistillationError("OX profile contract is unsafe")
    expires_at = _ox_expiry(config.ox_expires_at)
    relevant_config = {
        "teacher_profile": config.teacher_profile,
        "teacher_max_inflight": config.teacher_max_inflight,
        "ox_enabled": config.ox_enabled,
        "ox_free_only": config.ox_free_only,
        "ox_expires_at": expires_at,
        "max_input_bytes": config.max_input_bytes,
        "max_candidates": config.max_candidates,
    }
    if source_binding is None:
        try:
            source_binding = ox_alpha_source_binding()
        except ValueError as exc:
            # There is no source-less "preflight" contract.  It would be a
            # second authority which can never bind to remote labels.
            raise DistillationError("OX source binding is unavailable") from exc
    source = _validate_ox_source_binding(source_binding)
    _, _, artifact = store.write_immutable(
        store.distillation_dir(root) / "ox-profile-contracts",
        {
            "kind": "ox-alpha-free-profile",
            "profile": OX_SINGLE_PROFILE,
            "cohort": OX_SINGLE_COHORT,
            "route": "opencode-go/ox-alpha-free",
            "endpoint": f"{OX_ALPHA_ENDPOINT}/chat/completions",
            "request_model": "ox-alpha-free",
            "required_returned_model": "ox-alpha-free",
            "request_revision": OX_RAMP_REQUEST_REVISION,
            "fixed_identity": OX_ALPHA_FIXED_IDENTITY,
            "free_only": True,
            "no_paid_fallback": True,
            "official_status": "limited_time",
            "expires_at": expires_at,
            "docs_url": "https://opencode.ai/docs/go/",
            "kill_categories": [
                "402",
                "payment_required",
                "model_unavailable",
                "route_model_drift",
                "privacy_gate",
            ],
            "max_inflight": config.teacher_max_inflight,
            "live_recall_model_calls": 0,
            **source,
            "relevant_config_sha256": canonical_json.canonical_json_sha256_strict(
                relevant_config
            ),
        },
        schema=OX_PROFILE_SCHEMA,
    )
    store.write_sealed_state(
        store.distillation_dir(root) / "ox-profile-contract.json",
        {
            "kind": "ox-profile-contract-pointer",
            "profile_contract_id": artifact["artifact_id"],
        },
    )
    return artifact


def _current_ox_profile_contract_id(root: Path) -> str:
    """Read the sole OX contract allowed to contribute to training."""

    try:
        pointer = store.read_sealed(
            store.distillation_dir(root) / "ox-profile-contract.json",
            schema=store.DISTILLATION_SCHEMA,
        )
    except store.DistillationStoreError:
        return ""
    contract_id = str(pointer.get("profile_contract_id") or "")
    return contract_id if re.fullmatch(r"[0-9a-f]{64}", contract_id) else ""


def _validate_ox_profile_contract(
    contract: Mapping[str, Any], contract_id: str
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "namespace",
        "artifact_id",
        "kind",
        "profile",
        "cohort",
        "route",
        "endpoint",
        "request_model",
        "required_returned_model",
        "request_revision",
        "fixed_identity",
        "free_only",
        "no_paid_fallback",
        "official_status",
        "expires_at",
        "docs_url",
        "kill_categories",
        "max_inflight",
        "live_recall_model_calls",
        "source_commit",
        "source_tree_sha256",
        "source_ox_identity_sha256",
        "relevant_config_sha256",
        "seal_sha256",
    }
    if set(contract) != expected_keys:
        return {}
    unsigned = {
        key: value
        for key, value in contract.items()
        if key not in {"artifact_id", "seal_sha256"}
    }
    if (
        contract.get("artifact_id") != contract_id
        or canonical_json.canonical_json_sha256_strict(unsigned) != contract_id
    ):
        return {}
    expected = {
        "schema": OX_PROFILE_SCHEMA,
        "namespace": "recall-distillation",
        "kind": "ox-alpha-free-profile",
        "profile": OX_SINGLE_PROFILE,
        "cohort": OX_SINGLE_COHORT,
        "route": "opencode-go/ox-alpha-free",
        "endpoint": f"{OX_ALPHA_ENDPOINT}/chat/completions",
        "request_model": "ox-alpha-free",
        "required_returned_model": "ox-alpha-free",
        "request_revision": OX_RAMP_REQUEST_REVISION,
        "fixed_identity": OX_ALPHA_FIXED_IDENTITY,
        "free_only": True,
        "no_paid_fallback": True,
        "official_status": "limited_time",
        "docs_url": "https://opencode.ai/docs/go/",
        "kill_categories": [
            "402",
            "payment_required",
            "model_unavailable",
            "route_model_drift",
            "privacy_gate",
        ],
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        return {}
    max_inflight = contract.get("max_inflight")
    if (
        isinstance(max_inflight, bool)
        or not isinstance(max_inflight, int)
        or not 1 <= max_inflight <= 10
    ):
        return {}
    live_recall_model_calls = contract.get("live_recall_model_calls")
    if (
        isinstance(live_recall_model_calls, bool)
        or not isinstance(live_recall_model_calls, int)
        or live_recall_model_calls != 0
    ):
        return {}
    source_keys = (
        "source_commit",
        "source_tree_sha256",
        "source_ox_identity_sha256",
    )
    if any(not isinstance(contract.get(key), str) for key in source_keys):
        return {}
    if not isinstance(contract.get("relevant_config_sha256"), str):
        return {}
    try:
        if _ox_expiry(contract.get("expires_at")) != contract.get("expires_at"):
            return {}
        _validate_ox_source_binding(
            {key: contract[key] for key in source_keys}
        )
    except DistillationError:
        return {}
    if any(
        re.fullmatch(r"[0-9a-f]{64}", contract.get(key, "")) is None
        for key in (
            "source_tree_sha256",
            "source_ox_identity_sha256",
            "relevant_config_sha256",
        )
    ) or re.fullmatch(r"[0-9a-f]{40}", contract.get("source_commit", "")) is None:
        return {}
    return dict(contract)


def _read_ox_profile_contract(root: Path, contract_id: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{64}", contract_id) is None:
        return {}
    try:
        contract = store.read_sealed(
            store.distillation_dir(root)
            / "ox-profile-contracts"
            / f"{contract_id}.json",
            schema=OX_PROFILE_SCHEMA,
        )
    except store.DistillationStoreError:
        return {}
    return _validate_ox_profile_contract(contract, contract_id)


def _ox_contract_source_binding(root: Path, contract_id: str) -> dict[str, str]:
    contract = _read_ox_profile_contract(root, contract_id)
    if not contract:
        return {}
    return _validate_ox_source_binding(
        {
            key: str(contract.get(key) or "")
            for key in (
                "source_commit",
                "source_tree_sha256",
                "source_ox_identity_sha256",
            )
            if contract.get(key) is not None
        }
    )


def _ox_teacher_source_binding(teacher: Teacher) -> dict[str, str]:
    """Read the adapter-owned source binding without trusting generic teachers."""

    if (
        type(teacher) is not OpenCodeOxAlphaTeacher
        or type(teacher).__module__
        != "chronovisor.recall.recall_distillation_remote_teacher"
        or teacher.role != OX_TEACHER_ROLE
        or teacher.local is not False
        or teacher._route_identity != OX_ALPHA_FIXED_IDENTITY["route_identity"]
    ):
        raise DistillationError("untrusted OX teacher adapter")
    binding = teacher.receipt_binding
    try:
        value = binding()
    except Exception as exc:
        raise DistillationError("OX source binding is unavailable") from exc
    if not isinstance(value, Mapping):
        raise DistillationError("OX source binding is invalid")
    result = {str(key): str(item) for key, item in value.items()}
    return _validate_ox_source_binding(result)


def _ox_source_binding_matches(teacher: Teacher, expected: Mapping[str, str]) -> bool:
    """Re-read the trusted adapter identity after HTTP, before publication."""

    if not isinstance(teacher, OpenCodeOxAlphaTeacher):
        return not expected
    try:
        observed = _ox_teacher_source_binding(teacher)
    except DistillationError:
        return False
    return set(observed) == set(expected) and all(
        hmac.compare_digest(observed[key], expected[key]) for key in expected
    )


def _ox_event_head(root: Path, name: str) -> dict[str, Any]:
    path = store.distillation_dir(root) / name
    if not path.exists():
        return {"records": 0, "head_sha256": ""}
    return store.chain_head(path)


def _ox_event_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The closed, stable deduplication identity for one OX event."""

    kind = str(payload.get("kind") or "")
    common = {
        key: payload.get(key)
        for key in (
            "profile_contract_id",
            "source_commit",
            "source_tree_sha256",
            "source_ox_identity_sha256",
            "request_revision",
        )
    }
    if kind == "ox-ramp-stage":
        return {"kind": kind, **common, "cap": payload.get("cap")}
    if kind == "ox-provider-failure":
        return {
            "kind": kind,
            **common,
            "category": payload.get("category"),
            "work_ids": payload.get("work_ids", []),
            "attempts_by_work": payload.get("attempts_by_work", {}),
            "provider_receipts": payload.get("provider_receipts", {}),
        }
    if kind == "ox-lease-reclaim":
        return {
            "kind": kind,
            **common,
            "receipt": payload.get("workset_receipt_sha256"),
        }
    raise DistillationError("OX event kind is invalid")


def _append_ox_event(
    root: Path,
    name: str,
    payload: Mapping[str, Any],
    *,
    unique_key: str = "",
) -> dict[str, Any]:
    path = store.distillation_dir(root) / name
    event_key = canonical_json.canonical_json_sha256_strict(_ox_event_identity(payload))
    binding = canonical_json.canonical_json_sha256_strict(payload)
    # Store's unique index makes the check+append one critical section.  The
    # content digest turns a same identity / different payload into a hard stop.
    row = store.append_chain_unique(
        path,
        {**payload, "event_key": event_key, "event_binding_sha256": binding},
        unique_field="event_key",
        binding_field="event_binding_sha256",
    )
    if unique_key and payload.get(unique_key) is None:
        raise DistillationError("OX event identity is incomplete")
    if not isinstance(row, Mapping):
        raise DistillationError("OX event append failed")
    store.write_immutable(
        store.distillation_dir(root) / "ox-event-anchors",
        {
            "kind": "ox-event-anchor",
            "ledger_name": name,
            "event_key": event_key,
            "event_binding_sha256": binding,
            "record_sha256": str(row.get("record_sha256") or ""),
        },
        schema="chronovisor.recall-distill-ox-event-anchor.v1",
    )
    return _ox_event_head(root, name)


def _ox_event_heads(root: Path) -> dict[str, Mapping[str, Any]]:
    return {
        "ox_ramp_receipts": _ox_event_head(root, "ox-ramp-receipts.jsonl"),
        "ox_failure_receipts": _ox_event_head(root, "ox-failure-receipts.jsonl"),
        "ox_lease_recovery_receipts": _ox_event_head(
            root, "ox-lease-recovery-receipts.jsonl"
        ),
    }


def _r4_critical_module_sha256() -> dict[str, str]:
    """Bind installed critical code to the exact runtime checkout bytes."""

    modules = {
        "recall_distillation": "chronovisor.recall.recall_distillation",
        "remote_teacher": "chronovisor.recall.recall_distillation_remote_teacher",
        "workset": "chronovisor.recall.recall_distillation_workset",
        "runtime_config": "chronovisor.core.runtime_config",
    }
    try:
        source_root = runtime_config.runtime_repo_root().resolve(strict=True)
    except Exception:
        return {}
    result: dict[str, str] = {}
    for label, module_name in modules.items():
        relative = Path(*module_name.split(".")).with_suffix(".py")
        source_path = source_root / "src" / relative
        try:
            installed = importlib.import_module(module_name)
            installed_path = Path(str(installed.__file__)).resolve(strict=True)
            source = source_path.read_bytes()
            deployed = installed_path.read_bytes()
        except (ImportError, OSError, TypeError, ValueError):
            return {}
        if source != deployed:
            return {}
        result[label] = hashlib.sha256(source).hexdigest()
    return result


def bootstrap_r4_candidate_anchor(
    *, root: Path, tracked_r0_evidence: Path, source_binding: Mapping[str, str]
) -> dict[str, Any]:
    """Create the one R4 candidate anchor after an explicit clone bootstrap.

    This is deliberately not called by the worker.  Operators must supply the
    tracked R0 artifact; a legacy production workset therefore cannot gain a
    new authority merely by being opened during a normal backfill.
    """

    anchor_path = store.distillation_dir(root) / R4_CANDIDATE_ANCHOR_FILE
    lock = store.acquire_nonblocking_lock(anchor_path.with_suffix(".lock"))
    if lock is None:
        raise DistillationError("R4 candidate anchor bootstrap is busy")
    try:
        if anchor_path.exists():
            raise DistillationError("R4 candidate anchor already exists")
        try:
            r0 = json.loads(tracked_r0_evidence.read_bytes())
            store.verify_seal(r0, schema="chronovisor.recall-r0.v1")
            checkpoint_path = (
                store.distillation_dir(root) / "candidate-ledger.jsonl.head.json"
            )
            checkpoint = store.read_sealed(checkpoint_path)
            candidate_path = store.distillation_dir(root) / "candidate-ledger.jsonl"
            file_state = checkpoint.get("file_state")
            if (
                r0.get("artifact_id") != R4_R0_EVIDENCE_ID
                or not isinstance(file_state, Mapping)
                or candidate_path.stat().st_size != file_state.get("size_bytes")
                or checkpoint.get("records")
                != r0["production"]["ledgers"]["candidate-ledger.jsonl"]["records"]
                or checkpoint.get("head_sha256")
                != r0["production"]["ledgers"]["candidate-ledger.jsonl"]["head_sha256"]
                or not isinstance(source_binding.get("source_commit"), str)
            ):
                raise DistillationError(
                    "R4 candidate anchor bootstrap preflight failed"
                )
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            store.DistillationStoreError,
        ) as exc:
            raise DistillationError(
                "R4 candidate anchor bootstrap preflight failed"
            ) from exc
        critical_modules = _r4_critical_module_sha256()
        if not critical_modules:
            raise DistillationError(
                "R4 candidate anchor runtime binding is unavailable"
            )
        candidate = {
            "head_sha256": checkpoint["head_sha256"],
            "records": checkpoint["records"],
            "bytes": file_state["size_bytes"],
            "file_state": dict(file_state),
        }
        unsigned = {
            "schema": R4_CANDIDATE_ANCHOR_SCHEMA,
            "namespace": "recall-distillation",
            "kind": "r4-candidate-anchor",
            "r0_artifact_id": R4_R0_EVIDENCE_ID,
            "r0_file_sha256": hashlib.sha256(
                tracked_r0_evidence.read_bytes()
            ).hexdigest(),
            "bootstrap_source_commit": source_binding["source_commit"],
            "candidate_checkpoint": candidate,
            "critical_module_sha256": critical_modules,
        }
        artifact = {
            "artifact_id": canonical_json.canonical_json_sha256_strict(unsigned),
            **unsigned,
        }
        artifact["seal_sha256"] = canonical_json.canonical_json_sha256_strict(artifact)
        atomic_write_bytes(
            anchor_path,
            canonical_json.canonical_json_bytes_strict(artifact) + b"\n",
            backup=False,
        )
        return store.verify_seal(
            json.loads(anchor_path.read_bytes()), schema=R4_CANDIDATE_ANCHOR_SCHEMA
        )
    finally:
        store.release_lock(lock)


def _r4_runtime_identity_projection(
    root: Path,
    *,
    config_path: Path | None,
    source_binding: Mapping[str, str],
    profile_contract_id: str,
    candidate_path: Path,
    label_path: Path,
) -> dict[str, Any]:
    """Project R4 runtime identity from the same durable artifacts it audits."""

    from chronovisor.recall.recall_distillation_workset import DistillationWorkset

    config_file = runtime_config.active_config_file(config_path)
    workset_path = store.distillation_dir(root) / "ox-workset.sqlite3"
    contract_path = (
        store.distillation_dir(root)
        / "ox-profile-contracts"
        / f"{profile_contract_id}.json"
    )
    candidate_checkpoint = candidate_path.with_suffix(
        candidate_path.suffix + ".head.json"
    )
    label_checkpoint = label_path.with_suffix(label_path.suffix + ".head.json")
    try:
        receipts = DistillationWorkset(
            workset_path, migrate=False
        ).audit_transition_receipts()
        workset_sha256 = hashlib.sha256(workset_path.read_bytes()).hexdigest()
        contract_sha256 = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        candidate = store.read_sealed(candidate_checkpoint)
        labels = store.read_sealed(label_checkpoint)
        config_sha256 = hashlib.sha256(config_file.read_bytes()).hexdigest()
        critical_modules = _r4_critical_module_sha256()
        if not critical_modules:
            return {}
    except (OSError, store.DistillationStoreError, ValueError):
        # A partially written runtime must not fabricate a compatible receipt.
        return {}
    # R4 authority is rooted in the managed runtime, never in a checkout
    # handoff.  Bootstrap/migration creates this sealed, content-addressed
    # anchor explicitly; normal worker execution only reads it.
    try:
        anchor = store.read_sealed(
            store.distillation_dir(root) / R4_CANDIDATE_ANCHOR_FILE,
            schema=R4_CANDIDATE_ANCHOR_SCHEMA,
        )
        anchor_id = anchor.get("artifact_id")
        anchor_candidate = anchor.get("candidate_checkpoint")
        if (
            anchor.get("kind") != "r4-candidate-anchor"
            or not isinstance(anchor_id, str)
            or len(anchor_id) != 64
            or not isinstance(anchor_candidate, Mapping)
        ):
            return {}
        anchor_fields = {
            "candidate_anchor_artifact_id": anchor_id,
            "candidate_anchor_head_sha256": anchor_candidate.get("head_sha256"),
            "candidate_anchor_records": anchor_candidate.get("records"),
            "candidate_anchor_bytes": anchor_candidate.get("bytes"),
            "candidate_anchor_file_state": anchor_candidate.get("file_state"),
            "candidate_anchor_r0_artifact_id": anchor.get("r0_artifact_id"),
            "candidate_anchor_r0_file_sha256": anchor.get("r0_file_sha256"),
            "candidate_anchor_critical_module_sha256": anchor.get(
                "critical_module_sha256"
            ),
            "candidate_anchor_bootstrap_source_commit": anchor.get(
                "bootstrap_source_commit"
            ),
        }
        if (
            candidate.get("head_sha256")
            != anchor_fields["candidate_anchor_head_sha256"]
            or candidate.get("records") != anchor_fields["candidate_anchor_records"]
            or candidate.get("file_state")
            != anchor_fields["candidate_anchor_file_state"]
            or candidate.get("file_state", {}).get("size_bytes")
            != anchor_fields["candidate_anchor_bytes"]
            or anchor_fields["candidate_anchor_critical_module_sha256"]
            != critical_modules
        ):
            return {}
    except (OSError, store.DistillationStoreError, TypeError, ValueError):
        return {}
    try:
        installed = runtime_config.runtime_identity()
    except Exception:
        installed = {}
    direct_url = installed.get("direct_url")
    if not isinstance(direct_url, Mapping):
        return {}
    try:
        direct_url_copy = json.loads(canonical_json.canonical_json_bytes_strict(direct_url))
    except (TypeError, ValueError):
        return {}
    return {
        "root": str(root.absolute()),
        "account_uid": os.getuid(),
        "account_home": str(Path.home()),
        **source_binding,
        "config_sha256": config_sha256,
        "critical_module_sha256": critical_modules,
        "workset_sha256": workset_sha256,
        "profile_contract_sha256": contract_sha256,
        "workset_receipt_head": receipts.get("head_sha256"),
        "candidate_checkpoint_head": candidate.get("head_sha256"),
        "candidate_checkpoint_records": candidate.get("records"),
        "candidate_checkpoint_file_state": candidate.get("file_state"),
        **anchor_fields,
        "candidate_tail_records": 0,
        "candidate_tail_bytes": 0,
        "label_receipt_head": labels.get("head_sha256"),
        "label_checkpoint_records": labels.get("records"),
        "label_checkpoint_file_state": labels.get("file_state"),
        "archive_commit": str(installed.get("commit_id") or ""),
        "expected_commit": str(installed.get("expected_commit") or ""),
        "drift": installed.get("drift"),
        "archive_path": str(installed.get("archive_path") or ""),
        "module_path": str(installed.get("module_path") or ""),
        "direct_url": direct_url_copy,
        "direct_url_sha256": canonical_json.canonical_json_sha256_strict(
            direct_url_copy
        ),
        "os_identity": {
            "system": os.uname().sysname,
            "release": os.uname().release,
            "machine": os.uname().machine,
        },
    }


def _ox_event_projection(
    root: Path,
    *,
    profile_contract_id: str,
    source_binding: Mapping[str, str],
    workset: Mapping[str, Any],
    label_path: Path,
    authoritative_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the immutable OX ledgers; never manufacture missing evidence."""

    contract = _read_ox_profile_contract(root, profile_contract_id)
    contract_revision = contract.get("request_revision")
    contract_expiry: str | None = None
    if profile_contract_id:
        try:
            contract_expiry = _ox_expiry(contract.get("expires_at"))
        except DistillationError as exc:
            raise DistillationError("OX profile contract expiry is invalid") from exc
        if (
            not contract
            or contract_revision != OX_RAMP_REQUEST_REVISION
            or contract_expiry != contract.get("expires_at")
        ):
            raise DistillationError("OX profile contract request revision is invalid")
    else:
        contract_revision = OX_RAMP_REQUEST_REVISION

    contract_cache: dict[str, dict[str, Any]] = {}
    if profile_contract_id:
        contract_cache[profile_contract_id] = contract

    workset_payload_digests: dict[str, str] | None = None

    def current_workset_payload_digests() -> dict[str, str]:
        """Read the exact OX payload inventory without opening it writable."""

        nonlocal workset_payload_digests
        if workset_payload_digests is not None:
            return workset_payload_digests
        path = store.distillation_dir(root) / "ox-workset.sqlite3"
        try:
            uri = path.absolute().as_uri() + "?mode=ro"
            with sqlite3.connect(uri, uri=True) as connection:
                rows = connection.execute(
                    "SELECT work_id, payload_digest, provenance_json "
                    "FROM work_items WHERE kind = ?",
                    ("ox",),
                ).fetchall()
        except (OSError, sqlite3.Error) as exc:
            raise DistillationError("OX workset payload inventory is unavailable") from exc
        inventory: dict[str, str] = {}
        for work_id, payload_digest, provenance_json in rows:
            if (
                not isinstance(work_id, str)
                or re.fullmatch(r"[0-9a-f]{64}", work_id) is None
                or not isinstance(payload_digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", payload_digest) is None
                or work_id in inventory
            ):
                raise DistillationError("OX workset payload inventory is invalid")
            try:
                provenance = json.loads(provenance_json)
            except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
                raise DistillationError("OX workset payload inventory is invalid") from exc
            if not isinstance(provenance, Mapping):
                raise DistillationError("OX workset payload inventory is invalid")
            if provenance.get("profile_contract_id") != profile_contract_id:
                continue
            if (
                provenance.get("profile") != OX_SINGLE_PROFILE
                or provenance.get("cohort") != OX_SINGLE_COHORT
                or provenance.get("route") != "opencode-go/ox-alpha-free"
            ):
                raise DistillationError("OX workset payload inventory is invalid")
            expected_work_id = canonical_json.canonical_json_sha256_strict(
                {
                    "kind": "ox-teacher-label-v1",
                    "profile": OX_SINGLE_PROFILE,
                    "cohort": OX_SINGLE_COHORT,
                    "route": "opencode-go/ox-alpha-free",
                    "profile_contract_id": profile_contract_id,
                    "payload_digest": payload_digest,
                }
            )
            if work_id != expected_work_id:
                raise DistillationError("OX workset payload inventory is unbound")
            inventory[work_id] = payload_digest
        workset_payload_digests = inventory
        return inventory

    def referenced_contract(record: Mapping[str, Any]) -> dict[str, Any]:
        record_contract_id = record.get("profile_contract_id")
        if not isinstance(record_contract_id, str) or re.fullmatch(
            r"[0-9a-f]{64}", record_contract_id
        ) is None:
            raise DistillationError("OX record profile contract reference is invalid")
        if record_contract_id not in contract_cache:
            referenced = _read_ox_profile_contract(root, record_contract_id)
            if not referenced:
                raise DistillationError("OX record profile contract is missing")
            contract_cache[record_contract_id] = referenced
        referenced = contract_cache[record_contract_id]
        source_keys = (
            "source_commit",
            "source_tree_sha256",
            "source_ox_identity_sha256",
        )
        if any(
            record.get(key) != referenced.get(key)
            for key in (*source_keys, "request_revision", "expires_at")
        ):
            raise DistillationError("OX record contract binding conflicts")
        return referenced

    def rows(name: str) -> list[dict[str, Any]]:
        path = store.distillation_dir(root) / name
        try:
            if path.stat().st_size > 4 * 1024 * 1024:
                raise DistillationError("OX event ledger exceeds bounded size")
        except FileNotFoundError:
            pass
        verified = store.verify_chain(path)
        records = store.read_chain(path)
        head = store.chain_head(path)
        if (
            verified != head
            or len(records) != head["records"]
            or (records[-1].get("record_sha256") if records else "")
            != head["head_sha256"]
        ):
            raise DistillationError("OX event ledger head does not reconcile")
        selected: list[dict[str, Any]] = []
        for record in records:
            referenced = referenced_contract(record)
            referenced_revision = referenced.get("request_revision")
            referenced_expiry = referenced.get("expires_at")
            if record.get("request_revision") != referenced_revision:
                raise DistillationError("OX event request revision conflicts")
            if record.get("expires_at") != referenced_expiry:
                raise DistillationError("OX event expiry conflicts")
            if record.get("profile_contract_id") != profile_contract_id:
                continue
            if any(record.get(key) != value for key, value in source_binding.items()):
                raise DistillationError("OX event source binding conflicts")
            chain_fields = {
                "schema",
                "namespace",
                "previous_sha256",
                "record_sha256",
                "event_key",
                "event_binding_sha256",
            }
            payload = {
                key: value for key, value in record.items() if key not in chain_fields
            }
            allowed = {
                "ox-ramp-stage": {
                    "kind",
                    "profile_contract_id",
                    "source_commit",
                    "source_tree_sha256",
                    "source_ox_identity_sha256",
                    "request_revision",
                    "expires_at",
                    "cap",
                    "valid_receipts",
                    "attempts",
                    "work_ids",
                    "label_count",
                    "label_head_sha256",
                    "captured_at",
                },
                "ox-provider-failure": {
                    "kind",
                    "profile_contract_id",
                    "source_commit",
                    "source_tree_sha256",
                    "source_ox_identity_sha256",
                    "request_revision",
                    "expires_at",
                    "category",
                    "status",
                    "attempts",
                    "bounded",
                    "before_cap",
                    "after_cap",
                    "work_ids",
                    "attempts_by_work",
                    "provider_receipts",
                    "captured_at",
                },
                "ox-lease-reclaim": {
                    "kind",
                    "profile_contract_id",
                    "source_commit",
                    "source_tree_sha256",
                    "source_ox_identity_sha256",
                    "request_revision",
                    "expires_at",
                    "workset_receipt_generation",
                    "workset_receipt_sha256",
                    "reclaimed",
                    "leased_after",
                    "captured_at",
                },
            }
            kind = str(payload.get("kind") or "")
            if kind not in allowed or set(payload) - allowed[kind]:
                raise DistillationError("OX event schema is invalid")
            common_required = {
                "profile_contract_id",
                "source_commit",
                "source_tree_sha256",
                "source_ox_identity_sha256",
                "request_revision",
                "expires_at",
                "captured_at",
            }
            required = {
                "ox-ramp-stage": common_required
                | {
                    "cap",
                    "valid_receipts",
                    "attempts",
                    "work_ids",
                    "label_head_sha256",
                },
                "ox-provider-failure": common_required
                | {
                    "category",
                    "status",
                    "work_ids",
                    "attempts_by_work",
                    "provider_receipts",
                },
                "ox-lease-reclaim": common_required
                | {
                    "workset_receipt_generation",
                    "workset_receipt_sha256",
                    "reclaimed",
                    "leased_after",
                },
            }[kind]
            if not required.issubset(payload) or not isinstance(
                payload.get("captured_at"), str
            ):
                raise DistillationError("OX event required fields are missing")
            if payload.get("expires_at") != contract_expiry:
                raise DistillationError("OX event expiry conflicts")
            try:
                if _ox_expiry(payload.get("expires_at")) != payload.get("expires_at"):
                    raise DistillationError("OX event expiry is not canonical")
            except DistillationError as exc:
                raise DistillationError("OX event expiry is invalid") from exc
            if kind == "ox-ramp-stage" and (
                payload.get("cap") not in {1, 2, 5, 10}
                or not isinstance(payload.get("valid_receipts"), int)
                or not isinstance(payload.get("attempts"), int)
                or not isinstance(payload.get("work_ids"), list)
            ):
                raise DistillationError("OX ramp event fields are invalid")
            if kind == "ox-provider-failure" and (
                payload.get("status") not in {"deferred", "hard_stop"}
                or not isinstance(payload.get("work_ids"), list)
                or not isinstance(payload.get("attempts_by_work"), Mapping)
                or not isinstance(payload.get("provider_receipts"), Mapping)
            ):
                raise DistillationError("OX failure event fields are invalid")
            if kind == "ox-provider-failure":
                work_ids = payload.get("work_ids")
                attempts_by_work = payload.get("attempts_by_work")
                provider_receipts = payload.get("provider_receipts")
                if (
                    not isinstance(work_ids, list)
                    or not work_ids
                    or any(
                        not isinstance(work_id, str)
                        or re.fullmatch(r"[0-9a-f]{64}", work_id) is None
                        for work_id in work_ids
                    )
                    or len(set(work_ids)) != len(work_ids)
                    or not isinstance(attempts_by_work, Mapping)
                    or set(attempts_by_work) != set(work_ids)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 1
                        for value in attempts_by_work.values()
                    )
                    or not isinstance(provider_receipts, Mapping)
                    or set(provider_receipts) != set(work_ids)
                ):
                    raise DistillationError("OX failure event fields are invalid")
                payload_inventory = current_workset_payload_digests()
                expected_receipts = {
                    work_id: expected_ox_provider_request_sha256(
                        profile_contract_id=profile_contract_id,
                        payload_digest=payload_inventory[work_id],
                        work_id=work_id,
                        expires_at=str(referenced_expiry),
                    )
                    for work_id in work_ids
                    if work_id in payload_inventory
                }
                if len(expected_receipts) != len(work_ids) or dict(provider_receipts) != expected_receipts:
                    raise DistillationError("OX failure provider receipt is unbound")
            if record.get("event_key") != canonical_json.canonical_json_sha256_strict(
                _ox_event_identity(payload)
            ) or record.get(
                "event_binding_sha256"
            ) != canonical_json.canonical_json_sha256_strict(payload):
                raise DistillationError("OX event identity is invalid")
            anchor_id = canonical_json.canonical_json_sha256_strict(
                {
                    "schema": "chronovisor.recall-distill-ox-event-anchor.v1",
                    "namespace": "recall-distillation",
                    "kind": "ox-event-anchor",
                    "ledger_name": name,
                    "event_key": record["event_key"],
                    "event_binding_sha256": record["event_binding_sha256"],
                    "record_sha256": record["record_sha256"],
                }
            )
            try:
                anchor = store.read_sealed(
                    store.distillation_dir(root)
                    / "ox-event-anchors"
                    / f"{anchor_id}.json",
                    schema="chronovisor.recall-distill-ox-event-anchor.v1",
                )
            except store.DistillationStoreError as exc:
                raise DistillationError("OX event immutable anchor is missing") from exc
            if anchor.get("artifact_id") != anchor_id:
                raise DistillationError("OX event immutable anchor conflicts")
            selected.append(dict(record))
        return selected

    ramp = rows("ox-ramp-receipts.jsonl")
    failures = rows("ox-failure-receipts.jsonl")
    recoveries = rows("ox-lease-recovery-receipts.jsonl")
    labels: list[dict[str, Any]] = []
    for row in store.read_chain(label_path):
        if row.get("profile") != OX_SINGLE_PROFILE:
            continue
        referenced = referenced_contract(row)
        if row.get("profile_contract_id") != profile_contract_id:
            continue
        if any(row.get(key) != value for key, value in source_binding.items()):
            raise DistillationError("OX label source binding conflicts")
        if row.get("request_revision") != referenced.get("request_revision"):
            raise DistillationError("OX label request revision conflicts")
        if row.get("expires_at") != referenced.get("expires_at"):
            raise DistillationError("OX label expiry conflicts")
        try:
            if _ox_expiry(row.get("expires_at")) != row.get("expires_at"):
                raise DistillationError("OX label expiry is not canonical")
        except DistillationError as exc:
            raise DistillationError("OX label expiry is invalid") from exc
        labels.append(row)
    pair_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in labels:
        assignment = row.get("assignment")
        pair_id = (
            str(assignment.get("repeat_pair_id") or "")
            if isinstance(assignment, Mapping)
            else ""
        )
        if pair_id:
            pair_rows[pair_id].append(row)

    def complete_pair(rows_for_pair: Sequence[Mapping[str, Any]], key: str) -> bool:
        assignments = [row.get("assignment") for row in rows_for_pair]
        if not all(isinstance(value, Mapping) for value in assignments):
            return False
        members = {
            str(row.get("work_id") or "")
            for row in rows_for_pair
            if str(row.get("work_id") or "")
        }
        orders = {
            str(cast(Mapping[str, Any], value).get("blind_order") or "")
            for value in assignments
        }
        return (
            len(members) >= 2
            and all(
                cast(Mapping[str, Any], value).get(key) is True for value in assignments
            )
            and orders == {"a_first", "b_first"}
        )

    repeat_pairs = {
        pair_id
        for pair_id, rows_for_pair in pair_rows.items()
        if complete_pair(rows_for_pair, "fixed_repeat")
    }
    order_pairs = {
        pair_id
        for pair_id, rows_for_pair in pair_rows.items()
        if complete_pair(rows_for_pair, "order_swap")
    }
    conflicts = sum(row.get("negative_veto_conflict") is True for row in labels)
    # Policy artifacts are authority, not an optimistic local alias.  Require
    # both the sealed pointer target and its baseline lineage.
    rollback_verified = False
    try:
        active_id = str(store.read_pointer(root, "active")["policy_id"])
        active = _load_policy(active_id, root)
        lineage = active.get("lineage")
        rollback_verified = (
            active.get("artifact_id") == active_id
            and isinstance(lineage, Mapping)
            and re.fullmatch(
                r"[0-9a-f]{64}", str(lineage.get("baseline_artifact_id") or "")
            )
            is not None
        )
    except (KeyError, store.DistillationStoreError, DistillationError):
        pass
    quality_gates: dict[str, Any] = {
        "negative_veto": {
            "authenticated": bool(labels),
            "exact_binding": bool(labels),
            "conflicts": conflicts,
        },
        "blind_repeat": {
            "revision": OX_PROBE_REVISION,
            "complete": bool(repeat_pairs),
            "stability_passed": bool(repeat_pairs) and conflicts == 0,
            "pairs": len(repeat_pairs),
        },
        "order_swap": {"complete": bool(order_pairs), "pairs": len(order_pairs)},
        "rollback": {
            "verified": rollback_verified,
            "active_unchanged": rollback_verified,
            "status": "not_rolled_back" if rollback_verified else "unverified",
        },
    }
    required_caps = (1, 2, 5, 10)
    ramp_complete = [row.get("cap") for row in ramp] == list(required_caps) and all(
        isinstance(row.get("valid_receipts"), int)
        and isinstance(row.get("attempts"), int)
        and row["valid_receipts"] >= OX_RAMP_RECEIPTS_PER_CAP
        and row["attempts"] >= row["valid_receipts"]
        and row["valid_receipts"] * 100 >= row["attempts"] * 95
        and isinstance(row.get("work_ids"), list)
        and len(row["work_ids"]) == row["valid_receipts"]
        and len(set(row["work_ids"])) == len(row["work_ids"])
        for row in ramp
    )
    failure_complete = {str(row.get("category") or "") for row in failures} >= {
        "429",
        "5xx",
        "timeout",
        "402",
        "paid",
        "model_drift",
    }
    gate_passed = authoritative_gate.get("passed") is True
    gate_reasons = authoritative_gate.get("reasons")
    authoritative_reasons = (
        [str(reason) for reason in gate_reasons]
        if isinstance(gate_reasons, list)
        else ["authoritative_gate_unavailable"]
    )
    quality_reasons = [
        name
        for name, passed in (
            (
                "negative_veto",
                quality_gates["negative_veto"]["authenticated"]
                and quality_gates["negative_veto"]["exact_binding"]
                and quality_gates["negative_veto"]["conflicts"] == 0,
            ),
            (
                "blind_repeat",
                quality_gates["blind_repeat"]["complete"]
                and quality_gates["blind_repeat"]["stability_passed"],
            ),
            ("order_swap", quality_gates["order_swap"]["complete"]),
            ("rollback", quality_gates["rollback"]["verified"]),
        )
        if not passed
    ]
    if not ramp_complete:
        quality_reasons.append("ramp_receipts_incomplete")
    if not failure_complete:
        quality_reasons.append("failure_receipts_incomplete")
    if int(workset.get("leased") or 0) != 0:
        quality_reasons.append("leased_work_present")
    if not gate_passed:
        quality_reasons.extend(f"offline:{reason}" for reason in authoritative_reasons)
    quality_reasons = sorted(set(quality_reasons))
    return {
        "ramp_receipts": ramp,
        "failure_receipts": failures,
        "lease_recovery": {
            "recovered": sum(int(row.get("reclaimed") or 0) for row in recoveries),
            "leased_after": int(workset.get("leased") or 0),
            "receipt_count": len(recoveries),
        },
        "quality_gates": {
            **quality_gates,
            "passed": not quality_reasons,
            "reasons": quality_reasons,
            "offline_training_gate": dict(authoritative_gate),
            "ramp_complete": ramp_complete,
            "failure_complete": failure_complete,
        },
    }


def _default_distillation_config() -> dict[str, Any]:
    defaults = dict(DistillationConfig().__dict__)
    defaults["rollout_stages"] = list(defaults["rollout_stages"])
    return defaults


_DISTILLATION_CONFIG = _default_distillation_config()
_OPTIONAL_PROFILE_CONFIG = frozenset(
    {
        "teacher_profile",
        "teacher_max_inflight",
        "teacher_claim_limit",
        "ox_enabled",
        "ox_free_only",
        "ox_expires_at",
    }
)


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
    teacher_profile = table.get("teacher_profile", LOCAL_TRIAD_PROFILE)
    if teacher_profile not in TEACHER_PROFILES:
        raise DistillationError("recall.distillation.teacher_profile is invalid")
    ox_enabled = table.get("ox_enabled", False)
    ox_free_only = table.get("ox_free_only", True)
    if not isinstance(ox_enabled, bool) or ox_free_only is not True:
        raise DistillationError("OX Alpha must be explicitly enabled and free-only")
    ox_expires_at = table.get("ox_expires_at", "")
    if teacher_profile == OX_SINGLE_PROFILE and ox_enabled is True:
        _ox_expiry(ox_expires_at)
    teacher_max_inflight = positive("teacher_max_inflight", 10)
    if teacher_max_inflight > 10:
        raise DistillationError("teacher_max_inflight must be at most 10")
    teacher_claim_limit = positive("teacher_claim_limit", 500)
    if teacher_claim_limit > 500:
        raise DistillationError("teacher_claim_limit must be at most 500")
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
        teacher_profile=str(teacher_profile),
        teacher_max_inflight=teacher_max_inflight,
        teacher_claim_limit=teacher_claim_limit,
        ox_enabled=ox_enabled,
        ox_free_only=True,
        ox_expires_at=str(ox_expires_at),
    )


def _migration_sections(
    data: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    recall = data.get("recall", {})
    llm = data.get("llm", {})
    if not isinstance(recall, dict) or not isinstance(llm, dict):
        raise DistillationError("distillation config parent conflicts")
    distillation = recall.get("distillation", {})
    roles = llm.get("roles", {})
    if not isinstance(distillation, dict) or not isinstance(roles, dict):
        raise DistillationError("distillation config section conflicts")
    return recall, distillation, roles


def _migration_additions(data: Mapping[str, Any]) -> tuple[str, ...]:
    recall, distillation, roles = _migration_sections(data)
    additions: list[str] = []
    if "distillation" in recall:
        expected = dict(_DISTILLATION_CONFIG)
        enabled = distillation.get("enabled")
        missing = set(expected).difference(distillation)
        if missing.difference(_OPTIONAL_PROFILE_CONFIG):
            raise DistillationError("recall.distillation conflicts or is incomplete")
        if not isinstance(enabled, bool) or set(distillation).difference(expected):
            raise DistillationError("recall.distillation conflicts or is incomplete")
        for name, value in distillation.items():
            if name == "enabled" or name in _OPTIONAL_PROFILE_CONFIG:
                continue
            if value != expected[name]:
                raise DistillationError(
                    "recall.distillation conflicts or is incomplete"
                )
        profile = distillation.get("teacher_profile", LOCAL_TRIAD_PROFILE)
        max_inflight = distillation.get("teacher_max_inflight", 10)
        claim_limit = distillation.get("teacher_claim_limit", 500)
        ox_enabled = distillation.get("ox_enabled", False)
        ox_free_only = distillation.get("ox_free_only", True)
        if (
            profile not in TEACHER_PROFILES
            or isinstance(max_inflight, bool)
            or not isinstance(max_inflight, int)
            or not 1 <= max_inflight <= 10
            or isinstance(claim_limit, bool)
            or not isinstance(claim_limit, int)
            or not 1 <= claim_limit <= 500
            or not isinstance(ox_enabled, bool)
            or ox_free_only is not True
        ):
            raise DistillationError("recall.distillation profile config is invalid")
        additions.extend(f"recall.distillation.{name}" for name in sorted(missing))
    else:
        additions.append("recall.distillation")
    for name, expected in _DISTILLATION_ROLES.items():
        if name not in roles:
            additions.append(name)
        elif not isinstance(roles[name], dict) or roles[name] != expected:
            raise DistillationError(f"{name} conflicts or is incomplete")
    return tuple(additions)


def _migration_appendix(additions: Sequence[str]) -> bytes:
    sections: list[str] = []
    if "recall.distillation" in additions:
        sections.append(
            "\n".join(
                ["[recall.distillation]"]
                + [
                    f"{name} = {json.dumps(value, ensure_ascii=False)}"
                    for name, value in _DISTILLATION_CONFIG.items()
                ]
            )
        )
    for name, expected in _DISTILLATION_ROLES.items():
        if name not in additions:
            continue
        sections.append(
            "\n".join(
                [f'[llm.roles."{name}"]']
                + [
                    f"{key} = {json.dumps(value, ensure_ascii=False)}"
                    for key, value in expected.items()
                ]
            )
        )
    return ("\n\n".join(sections) + "\n").encode("utf-8")


def _migration_insert_profile_defaults(
    original: bytes, additions: Sequence[str]
) -> bytes:
    keys = [
        name.removeprefix("recall.distillation.")
        for name in additions
        if name.startswith("recall.distillation.")
    ]
    if not keys:
        return original
    try:
        text = original.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DistillationError("config TOML is unavailable or invalid") from exc
    header = re.search(r"(?m)^\[recall\.distillation\][ \t]*(?:#.*)?$", text)
    if header is None:
        raise DistillationError("recall.distillation section is unavailable")
    following = re.search(r"(?m)^\[", text[header.end() :])
    insertion = len(text) if following is None else header.end() + following.start()
    prefix = "" if insertion == 0 or text[:insertion].endswith("\n") else "\n"
    block = "".join(
        f"{key} = {json.dumps(_DISTILLATION_CONFIG[key], ensure_ascii=False)}\n"
        for key in keys
    )
    return (text[:insertion] + prefix + block + text[insertion:]).encode("utf-8")


def migrate_distillation_config(
    config_path: Path | None = None, *, apply: bool = False
) -> dict[str, Any]:
    """Append the exact disabled distillation configuration, or fail closed.

    The result contains only stable section names; it never exposes config text.
    """

    path = runtime_config.active_config_file(config_path)
    with sidecar_exclusive_lock(path):
        try:
            original = path.read_bytes()
            parsed = tomllib.loads(original.decode("utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise DistillationError("config TOML is unavailable or invalid") from exc
        additions = _migration_additions(parsed)
        if not additions:
            return {"status": "noop", "additions": []}
        replacement = _migration_insert_profile_defaults(original, additions)
        section_additions = [
            name for name in additions if not name.startswith("recall.distillation.")
        ]
        suffix = _migration_appendix(section_additions)
        if suffix:
            separator = b"" if not replacement or replacement.endswith(b"\n") else b"\n"
            replacement += separator + b"\n" + suffix
        try:
            migrated = tomllib.loads(replacement.decode("utf-8"))
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise DistillationError("generated distillation config is invalid") from exc
        if _migration_additions(migrated):
            raise DistillationError("generated distillation config is incomplete")
        result = {
            "status": "applied" if apply else "dry_run",
            "additions": list(additions),
        }
        if not apply:
            return result
        atomic_write_bytes(path, replacement, backup=True)
        try:
            confirmed = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise DistillationError(
                "migrated config failed read-back validation"
            ) from exc
        if _migration_additions(confirmed):
            raise DistillationError("migrated config read-back is incomplete")
        return result


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
        item_type, content = pi_transcript.pi_message_view(dict(event))
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


def _read_chain(path: Path) -> list[dict[str, Any]]:
    return store.read_chain(path)


def _prompt_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha1(normalized.encode()).hexdigest()[:16]


def _exposure_map(root: Path) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    path = store.distillation_dir(root) / "exposure-receipts.jsonl"
    result: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _read_chain(path):
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
    prefetch = getattr(candidate_texts, "prefetch", None)
    if callable(prefetch):
        prefetch(str(candidate["text_sha256"]) for candidate in candidates)
    feature_rows = []
    query_feature_bytes = _bounded_normalized_text(query_text, max_bytes=2_048).encode()
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
        policy_id = str(_stable_pointer_read(root, "active")["policy_id"])
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


def _shadow_feature_rows(
    rows: Sequence[Mapping[str, Any]], label: str
) -> list[dict[str, Any]]:
    """Normalize one arm's feature bytes without sharing the caller object."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise DistillationError(f"{label} feature snapshot is invalid")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise DistillationError(f"{label} feature row is invalid")
        if set(row) != {"candidate_id", "features"}:
            raise DistillationError(f"{label} feature row schema is not closed")
        candidate_id = row.get("candidate_id")
        values = row.get("features")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in seen
            or not isinstance(values, Mapping)
            or set(values) != set(FAST_FEATURE_KEYS)
        ):
            raise DistillationError(f"{label} feature binding is invalid")
        features = build_fast_features(values)
        if dict(values) != features:
            raise DistillationError(f"{label} features are not canonical")
        seen.add(candidate_id)
        normalized.append({"candidate_id": candidate_id, "features": features})
    return normalized


def _shadow_pool_rows(
    rows: Sequence[Mapping[str, Any]], label: str
) -> list[dict[str, Any]]:
    """Normalize one arm's candidate pool, retaining no rendered text."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise DistillationError(f"{label} candidate pool is invalid")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise DistillationError(f"{label} candidate source binding is invalid")
        allowed_keys = {
            "candidate_id",
            "selected",
            "page_id",
            "page_content_sha256",
            "rendered_context_sha256",
        }
        row_keys = set(row)
        if row_keys != allowed_keys and row_keys != allowed_keys | {"rendered_context"}:
            raise DistillationError(f"{label} candidate source schema is not closed")
        candidate_id = row.get("candidate_id")
        page_id = row.get("page_id")
        selected = row.get("selected")
        page_sha256 = row.get("page_content_sha256")
        rendered_sha256 = row.get("rendered_context_sha256")
        rendered = row.get("rendered_context")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in seen
            or not isinstance(page_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,239}", page_id) is None
            or not isinstance(selected, bool)
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
            raise DistillationError(f"{label} candidate source binding is invalid")
        seen.add(candidate_id)
        normalized.append(
            {
                "candidate_id": candidate_id,
                "selected": selected,
                "page_id": page_id,
                "page_content_sha256": page_sha256,
                "rendered_context_sha256": rendered_sha256,
            }
        )
    return normalized


def shadow_observation_hashes(
    candidate_feature_snapshot: Sequence[Mapping[str, Any]],
    baseline_feature_snapshot: Sequence[Mapping[str, Any]],
    candidate_pool_refs: Sequence[Mapping[str, Any]],
    baseline_pool_refs: Sequence[Mapping[str, Any]],
    *,
    selected_candidate_ids: Sequence[str],
    baseline_selected_candidate_ids: Sequence[str],
) -> dict[str, str | bool]:
    """Compute independent arm hashes and the pair identity.

    The candidate and incumbent values are serialized in separate calls.  A
    caller cannot satisfy parity by copying one digest into the other field.
    """

    candidate_features = _shadow_feature_rows(candidate_feature_snapshot, "candidate")
    baseline_features = _shadow_feature_rows(baseline_feature_snapshot, "baseline")
    candidate_pool = _shadow_pool_rows(candidate_pool_refs, "candidate")
    baseline_pool = _shadow_pool_rows(baseline_pool_refs, "baseline")
    candidate_feature_bytes = canonical_json.canonical_json_bytes_strict(candidate_features)
    baseline_feature_bytes = canonical_json.canonical_json_bytes_strict(baseline_features)
    candidate_feature_sha = hashlib.sha256(candidate_feature_bytes).hexdigest()
    baseline_feature_sha = hashlib.sha256(baseline_feature_bytes).hexdigest()
    candidate_feature_snapshot_sha = canonical_json.canonical_json_sha256_strict(
        candidate_features
    )
    baseline_feature_snapshot_sha = canonical_json.canonical_json_sha256_strict(
        baseline_features
    )
    candidate_pool_sha = canonical_json.canonical_json_sha256_strict(candidate_pool)
    baseline_pool_sha = canonical_json.canonical_json_sha256_strict(baseline_pool)
    candidate_decision_sha = canonical_json.canonical_json_sha256_strict(
        list(selected_candidate_ids)
    )
    baseline_decision_sha = canonical_json.canonical_json_sha256_strict(
        list(baseline_selected_candidate_ids)
    )
    feature_snapshot_sha = canonical_json.canonical_json_sha256_strict(
        {"candidate": candidate_features, "baseline": baseline_features}
    )
    parity = (
        candidate_feature_sha == baseline_feature_sha
        and candidate_feature_snapshot_sha == baseline_feature_sha
        and baseline_feature_snapshot_sha == candidate_feature_sha
    )
    pair_id = canonical_json.canonical_json_sha256_strict(
        {
            "candidate_decision_sha256": candidate_decision_sha,
            "baseline_decision_sha256": baseline_decision_sha,
            "candidate_pool_sha256": candidate_pool_sha,
            "baseline_pool_sha256": baseline_pool_sha,
            "candidate_feature_bytes_sha256": candidate_feature_sha,
            "baseline_feature_bytes_sha256": baseline_feature_sha,
        }
    )
    return {
        "candidate_decision_sha256": candidate_decision_sha,
        "baseline_decision_sha256": baseline_decision_sha,
        "candidate_pool_sha256": candidate_pool_sha,
        "baseline_pool_sha256": baseline_pool_sha,
        "candidate_feature_snapshot_sha256": candidate_feature_snapshot_sha,
        "baseline_feature_snapshot_sha256": baseline_feature_snapshot_sha,
        "candidate_feature_bytes_sha256": candidate_feature_sha,
        "baseline_feature_bytes_sha256": baseline_feature_sha,
        "feature_snapshot_sha256": feature_snapshot_sha,
        "feature_parity": parity,
        "pair_id": pair_id,
    }


def _shadow_replay_source_fields(
    *,
    decision_id: str,
    query_semantic_sha256: str,
    observed_at: str,
    pool_rows: Sequence[Mapping[str, Any]],
    selected_candidate_ids: Sequence[str],
    baseline_pool_rows: Sequence[Mapping[str, Any]],
    baseline_selected_candidate_ids: Sequence[str],
    paired_eligible: bool,
) -> dict[str, str]:
    """Derive immutable replay source identity from one real request."""

    def selected_identity(
        rows: Sequence[Mapping[str, Any]],
        selected_ids: Sequence[str],
        *,
        label: str,
        required: bool,
    ) -> str:
        if any(not isinstance(value, str) or not value for value in selected_ids):
            raise DistillationError(f"{label} selected candidates are invalid")
        if len(selected_ids) != len(set(selected_ids)):
            raise DistillationError(f"{label} selected candidates are duplicated")
        if len(selected_ids) != 1:
            if required:
                raise DistillationError(
                    f"{label} replay source candidate identity is ambiguous"
                )
            return ""
        candidate_id = selected_ids[0]
        matches = [
            row
            for row in rows
            if row.get("candidate_id") == candidate_id and row.get("selected") is True
        ]
        if len(matches) != 1:
            raise DistillationError(
                f"{label} replay source candidate is not selected in its pool"
            )
        return candidate_id

    candidate_id = selected_identity(
        pool_rows,
        selected_candidate_ids,
        label="candidate",
        required=paired_eligible,
    )
    # A paired replay row must represent one incumbent-selected candidate as
    # well; unpaired/abstaining observations remain truthful non-replay data.
    selected_identity(
        baseline_pool_rows,
        baseline_selected_candidate_ids,
        label="baseline",
        required=paired_eligible,
    )
    row_id = canonical_json.canonical_json_sha256_strict(
        {
            "decision_id": decision_id,
            "query_semantic_sha256": query_semantic_sha256,
            "candidate_id": candidate_id,
            "observed_at": observed_at,
        }
    )
    split_bucket = int(query_semantic_sha256[:2], 16) if query_semantic_sha256 else 0
    split = "train" if split_bucket < 179 else "validation" if split_bucket < 217 else "test"
    return {
        "row_id": row_id,
        "rally_id": query_semantic_sha256,
        "candidate_id": candidate_id,
        "as_of": observed_at,
        "split": split,
        "split_role": split,
    }


def _shadow_evidence_with_hashes(
    evidence: ShadowOperationalEvidence,
    hashes: Mapping[str, str | bool],
    *,
    stage: str,
    run_id: str,
    cohort: str,
    host: str,
) -> dict[str, Any]:
    """Validate typed evidence against producer-derived identities."""

    if not isinstance(evidence, ShadowOperationalEvidence):
        raise DistillationError("shadow operational evidence must be typed")
    raw = evidence.to_dict()
    if set(raw) != _OPERATIONAL_EVIDENCE_KEYS:
        raise DistillationError("shadow operational evidence schema is not closed")
    producer = raw.get("producer")
    if (
        not isinstance(producer, Mapping)
        or set(producer) != {"name", "version", "synthetic_fixture"}
        or producer.get("name") != SHADOW_PRODUCER_NAME
        or producer.get("version") != SHADOW_PRODUCER_VERSION
        or producer.get("synthetic_fixture") is not False
    ):
        raise DistillationError("shadow operational evidence producer is invalid")
    for key in (
        "candidate_quality",
        "baseline_quality",
        "candidate_covered",
        "baseline_covered",
        "candidate_anchor_retained",
        "baseline_anchor_retained",
        "candidate_abstained",
        "baseline_abstained",
        "resource_ok",
        "integrity_ok",
        "negative_veto",
        "feature_parity",
    ):
        if not isinstance(raw.get(key), bool):
            raise DistillationError(f"shadow operational evidence {key} type")
    for key in ("candidate_score_ms", "live_latency_ms", "deadline_ms"):
        if not isinstance(raw.get(key), int) or isinstance(raw.get(key), bool):
            raise DistillationError(f"shadow operational evidence {key} type")
    for key in ("stage", "run_id", "cohort", "host"):
        if not isinstance(raw.get(key), str):
            raise DistillationError(f"shadow operational evidence {key} type")
    if (
        raw.get("stage") != stage
        or raw.get("run_id") != run_id
        or raw.get("cohort") != cohort
        or raw.get("host") != host
        or not stage
        or not run_id
        or not cohort
        or not host
    ):
        raise DistillationError("shadow operational evidence provenance drift")
    for key in (
        "candidate_decision_sha256",
        "baseline_decision_sha256",
        "candidate_pool_sha256",
        "baseline_pool_sha256",
        "candidate_feature_snapshot_sha256",
        "baseline_feature_snapshot_sha256",
        "candidate_feature_bytes_sha256",
        "baseline_feature_bytes_sha256",
        "feature_snapshot_sha256",
        "pair_id",
    ):
        expected = hashes[key]
        value = raw.get(key)
        if not isinstance(value, str) or value not in ("", expected):
            raise DistillationError(f"shadow operational evidence {key} drift")
        raw[key] = expected
    expected_parity = hashes["feature_parity"]
    if raw.get("feature_parity") not in {False, expected_parity}:
        raise DistillationError("shadow operational evidence parity drift")
    raw["feature_parity"] = expected_parity
    return raw


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
    baseline_feature_snapshot: Sequence[Mapping[str, Any]] | None = None,
    baseline_pool_refs: Sequence[Mapping[str, Any]] | None = None,
    operational_evidence: ShadowOperationalEvidence | None = None,
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
        any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in values)
        or len(values) != len(set(values))
        for values in (selected, incumbent_selected)
    ):
        raise DistillationError("shadow selected candidates are invalid")
    if len(candidate_pool_refs) > 12 or len(candidate_feature_snapshot) > 12:
        raise DistillationError("shadow candidate pool is too large")
    if baseline_feature_snapshot is not None and len(baseline_feature_snapshot) > 12:
        raise DistillationError("shadow baseline feature snapshot is too large")
    if baseline_pool_refs is not None and len(baseline_pool_refs) > 12:
        raise DistillationError("shadow baseline pool is too large")
    pool_rows = _shadow_pool_rows(candidate_pool_refs, "candidate")
    baseline_pool_rows = _shadow_pool_rows(
        baseline_pool_refs if baseline_pool_refs is not None else candidate_pool_refs,
        "baseline",
    )
    pool_ids = {row["candidate_id"] for row in pool_rows}
    baseline_pool_ids = {row["candidate_id"] for row in baseline_pool_rows}
    if {row["candidate_id"] for row in pool_rows if row["selected"]} != set(selected):
        raise DistillationError("shadow selected pool does not match decision")
    if {row["candidate_id"] for row in baseline_pool_rows if row["selected"]} != set(
        incumbent_selected
    ):
        raise DistillationError("shadow baseline selected pool does not match decision")
    if not set(incumbent_selected).issubset(baseline_pool_ids):
        raise DistillationError("shadow incumbent decision is outside candidate pool")
    feature_rows = _shadow_feature_rows(candidate_feature_snapshot, "candidate")
    baseline_feature_rows = _shadow_feature_rows(
        baseline_feature_snapshot
        if baseline_feature_snapshot is not None
        else candidate_feature_snapshot,
        "baseline",
    )
    feature_ids = {row["candidate_id"] for row in feature_rows}
    baseline_feature_ids = {row["candidate_id"] for row in baseline_feature_rows}
    if feature_ids != pool_ids or baseline_feature_ids != baseline_pool_ids:
        raise DistillationError("shadow pool and features do not match")
    if pool_ids != baseline_pool_ids:
        raise DistillationError("shadow candidate and baseline pools differ")
    qualified = load_policy_observation_context(session_id, root)
    if (
        qualified.get("candidate_policy_id") != policy_id
        or qualified.get("incumbent_policy_id") != incumbent_policy_id
        or qualified.get("served_policy_id") != served_policy_id
    ):
        raise DistillationError("paired policy observation is not qualified")
    _, stage_started_us = _timestamp(qualified.get("stage_started_at"), observed_at)
    if observed_us < stage_started_us:
        raise DistillationError("paired observation predates rollout stage")
    hashes = shadow_observation_hashes(
        feature_rows,
        baseline_feature_rows,
        pool_rows,
        baseline_pool_rows,
        selected_candidate_ids=selected,
        baseline_selected_candidate_ids=incumbent_selected,
    )
    if operational_evidence is None:
        evidence: dict[str, Any] = {}
    elif isinstance(operational_evidence, ShadowOperationalEvidence):
        evidence = _shadow_evidence_with_hashes(
            operational_evidence,
            hashes,
            stage=str(qualified.get("stage") or ""),
            run_id=str(qualified.get("qualified_run_id") or ""),
            cohort=str(qualified.get("cohort") or ""),
            host=host,
        )
    else:
        raise DistillationError("shadow operational evidence must be typed")
    observation = {
        "decision": "read" if selected else "none",
        "selected_count": len(selected),
        "evaluated_count": len(feature_rows),
        "latency_ms": float(decision_latency_ms),
        "timed_out": timed_out,
        "error_code": error_code,
    }
    evidence_sha256 = canonical_json.canonical_json_sha256_strict(evidence)
    replay_source = _shadow_replay_source_fields(
        decision_id=decision_id,
        query_semantic_sha256=query_semantic_sha256,
        observed_at=observed_at,
        pool_rows=pool_rows,
        selected_candidate_ids=selected,
        baseline_pool_rows=baseline_pool_rows,
        baseline_selected_candidate_ids=incumbent_selected,
        paired_eligible=paired_eligible,
    )
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
        "run_id": qualified["qualified_run_id"],
        "cohort": qualified.get("cohort", ""),
        "baseline_artifact_id": qualified.get("baseline_artifact_id", ""),
        "candidate_policy_id": policy_id,
        "baseline_policy_id": incumbent_policy_id,
        **replay_source,
        "selected_candidate_ids": selected,
        "incumbent_selected_candidate_ids": incumbent_selected,
        "paired_eligible": paired_eligible,
        "candidate_pool_sha256": canonical_json.canonical_json_sha256_strict(pool_rows),
        "candidate_feature_snapshot_sha256": canonical_json.canonical_json_sha256_strict(
            feature_rows
        ),
        "candidate_decision_sha256": hashes["candidate_decision_sha256"],
        "baseline_decision_sha256": hashes["baseline_decision_sha256"],
        "baseline_pool_sha256": hashes["baseline_pool_sha256"],
        "baseline_feature_snapshot_sha256": hashes["baseline_feature_snapshot_sha256"],
        "candidate_feature_bytes_sha256": hashes["candidate_feature_bytes_sha256"],
        "baseline_feature_bytes_sha256": hashes["baseline_feature_bytes_sha256"],
        "feature_snapshot_sha256": hashes["feature_snapshot_sha256"],
        "pair_id": hashes["pair_id"],
        "runtime_observation_sha256": canonical_json.canonical_json_sha256_strict(
            observation
        ),
        "operational_evidence_sha256": evidence_sha256,
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
                "baseline_pool_refs": baseline_pool_rows,
                "baseline_feature_snapshot": baseline_feature_rows,
                "runtime_observation": observation,
                "operational_evidence": evidence,
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
                {
                    key: value
                    for key, value in binding.items()
                    if key not in {"observed_at", "as_of", "row_id"}
                }
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
    if (
        not isinstance(selected, list)
        or len(selected) != 1
        or not isinstance(selected[0], Mapping)
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
    rows = _read_chain(store.distillation_dir(root) / "outcome-receipts.jsonl")
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
    from chronovisor.recall import recall_distillation_rollout as rollout

    artifact = rollout._stable_sealed(
        store.distillation_dir(root) / "policies" / f"{policy_id}.json",
        base=store.distillation_dir(root),
        schema=POLICY_SCHEMA,
        label="policy artifact",
    )
    if artifact.get("artifact_id") != policy_id:
        raise DistillationError("policy identity mismatch")
    return artifact


def _enabled_for_root(root: Path) -> bool:
    config_path = root / "config.toml"
    return distillation_enabled(config_path if config_path.exists() else None)


def _read_worker_state(root: Path) -> dict[str, Any]:
    from chronovisor.recall import recall_distillation_rollout as rollout

    state = rollout._stable_sealed(
        store.distillation_dir(root) / store.STATE_FILE,
        base=store.distillation_dir(root),
        schema=store.DISTILLATION_SCHEMA,
        label="worker state",
    )
    return {
        key: value
        for key, value in state.items()
        if key not in {"schema", "namespace", "seal_sha256"}
    }


def _stable_pointer_read(root: Path, kind: str) -> dict[str, Any]:
    from chronovisor.recall import recall_distillation_rollout as rollout

    return rollout._stable_pointer(root, kind)


def _load_serving_policy(root: Path, *, allow_lkg: bool) -> dict[str, Any]:
    kinds = ("active", "lkg") if allow_lkg else ("active",)
    for kind in kinds:
        try:
            pointer = _stable_pointer_read(root, kind)
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
            candidate_id = str(_stable_pointer_read(root, "candidate")["policy_id"])
            incumbent_id = str(_stable_pointer_read(root, "active")["policy_id"])
            if str(_stable_pointer_read(root, "lkg")["policy_id"]) != incumbent_id:
                return {}
            candidate = _load_policy(candidate_id, root)
            incumbent = _load_policy(incumbent_id, root)
            lineage = candidate.get("lineage")
            baseline_id = (
                str(lineage.get("baseline_artifact_id") or "")
                if isinstance(lineage, Mapping)
                else ""
            )
            cohort = (
                str(lineage.get("model_cohort_sha256") or "")
                if isinstance(lineage, Mapping)
                else ""
            )
            baseline = rollout._stable_sealed(
                store.distillation_dir(root) / "baselines" / f"{baseline_id}.json",
                base=store.distillation_dir(root),
                schema=BASELINE_SCHEMA,
                label="observation baseline",
            )
            receipt_id = str(state.get("evaluation_receipt_id") or "")
            receipt = rollout._stable_sealed(
                store.distillation_dir(root) / "rollout-runs" / f"{receipt_id}.json",
                base=store.distillation_dir(root),
                schema=rollout.EVALUATION_SCHEMA,
                label="observation rollout receipt",
            )
            hard_floor = baseline.get("hard_floor")
            if (
                baseline.get("artifact_id") != baseline_id
                or not isinstance(hard_floor, Mapping)
                or hard_floor.get("p5_allowed") is not True
                or receipt.get("artifact_id") != receipt_id
                or receipt.get("candidate_policy_id") != candidate_id
                or receipt.get("baseline_policy_id") != incumbent_id
                or receipt.get("baseline_artifact_id") != baseline_id
            ):
                return {}
            if stage == "shadow":
                served_id = incumbent_id
            else:
                percent = int(state.get("rollout_percent") or 0)
                bucket = (
                    int.from_bytes(
                        hashlib.sha256(
                            f"recall-distill-rollout-v2\0{session_id}".encode()
                        ).digest()[:8],
                        "big",
                    )
                    % 10_000
                )
                served_id = candidate_id if bucket < percent * 100 else incumbent_id
            return {
                "stage": stage,
                "stage_started_at": str(state.get("stage_started_at") or ""),
                "qualified_run_id": str(state.get("stage_run_id") or ""),
                "cohort": cohort,
                "baseline_artifact_id": baseline_id,
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


def _policy_payload_digest(policy: Mapping[str, Any]) -> str:
    """Hash only executable policy fields, never the sealed artifact envelope."""

    fields = train_tiny_policy(())
    return canonical_json.canonical_json_sha256_strict(
        {key: policy.get(key) for key in fields}
    )


def _materialization_rallies(
    root: Path, supplied: Sequence[Mapping[str, Any]] | None
) -> dict[str, Mapping[str, Any]]:
    if supplied is None:
        rally_rows = _read_chain(store.distillation_dir(root) / "rally-manifest.jsonl")
        return {
            str(manifest["rally_id"]): manifest
            for row in rally_rows
            if isinstance((manifest := row.get("manifest")), Mapping)
        }
    return {
        str(row["rally_id"]): row
        for row in supplied
        if isinstance(row, Mapping) and row.get("rally_id")
    }


def _materialization_snapshots(
    root: Path, supplied: Mapping[str, Mapping[str, Any]] | None
) -> Mapping[str, Mapping[str, Any]]:
    if supplied is not None:
        return supplied
    return {
        str(row.get("rally_id") or ""): row["snapshot"]
        for row in _read_chain(store.distillation_dir(root) / "candidate-ledger.jsonl")
        if isinstance(row.get("snapshot"), Mapping)
    }


def _materialization_feature_pairs(
    root: Path,
    rallies: Mapping[str, Mapping[str, Any]],
    snapshots: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, float]], dict[str, dict[str, Any]]]:
    features_by_pair: dict[tuple[str, str], dict[str, float]] = {}
    snapshot_contracts = {
        rally_id: {
            "as_of": str(snapshot.get("as_of") or ""),
            "snapshot_sha256": str(snapshot.get("snapshot_sha256") or ""),
            "candidate_text_sha256": {
                str(candidate.get("candidate_id") or ""): str(
                    candidate.get("text_sha256") or ""
                )
                for candidate in (
                    snapshot.get("candidates", [])
                    if isinstance(snapshot.get("candidates"), list)
                    else []
                )
                if isinstance(candidate, Mapping)
                and candidate.get("candidate_id")
            },
        }
        for rally_id, snapshot in snapshots.items()
        if isinstance(snapshot, Mapping)
    }
    for rally_id, snapshot in snapshots.items():
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
    return features_by_pair, snapshot_contracts


def _materialization_payload_source_matches(
    payload_source: object,
    *,
    rally_id: str,
    candidate_id: str,
    rally: Mapping[str, Any],
    snapshot_sha256: str,
    candidate_text_sha256: str | None = None,
    assignment: Mapping[str, Any] | None = None,
) -> bool:
    """Bind a payload envelope to its immutable rally/candidate source.

    The digest is checked by each caller; this helper only compares the
    closed semantic projection.  Production collection reuses the same
    projection with candidate text, while older materialized rows can omit
    that optional field when a snapshot predates text hashes.
    """

    if not isinstance(payload_source, Mapping):
        return False
    expected: dict[str, Any] = {
        "rally_id": rally_id,
        "candidate_id": candidate_id,
        "snapshot_sha256": snapshot_sha256,
        "query_sha256": rally.get("query_sha256", ""),
        "context_sha256": [
            ref.get("semantic_sha256", "")
            for ref in rally.get("context_refs", [])
            if isinstance(ref, Mapping)
        ],
    }
    if candidate_text_sha256 is not None:
        expected["candidate_text_sha256"] = candidate_text_sha256
    if isinstance(assignment, Mapping) and assignment.get("probe") is True:
        expected["assignment"] = dict(assignment)
    return dict(payload_source) == expected


def _training_assignment_authority(assignment: object) -> dict[str, Any]:
    """Keep the deterministic, training-relevant assignment projection."""

    source = assignment if isinstance(assignment, Mapping) else {}
    return {
        "revision": str(source.get("revision") or ""),
        "kind": str(source.get("kind") or ""),
        "profile": str(source.get("profile") or ""),
        "split": str(source.get("split") or ""),
        "probe": source.get("probe") is True,
        "owner": str(source.get("owner") or ""),
        "routes": [str(route) for route in source.get("routes", [])]
        if isinstance(source.get("routes"), list)
        else [],
        "probe_revision": str(source.get("probe_revision") or ""),
        "repeat_pair_id": str(source.get("repeat_pair_id") or ""),
        "fixed_repeat": source.get("fixed_repeat") is True,
        "order_swap": source.get("order_swap") is True,
        "blind_order": str(source.get("blind_order") or ""),
        "probe_batch_id": str(source.get("probe_batch_id") or ""),
        "order_variant": (
            int(source["order_variant"])
            if isinstance(source.get("order_variant"), int)
            and not isinstance(source.get("order_variant"), bool)
            else 0
        ),
        "candidate_position": (
            int(source["candidate_position"])
            if isinstance(source.get("candidate_position"), int)
            and not isinstance(source.get("candidate_position"), bool)
            else -1
        ),
    }


def _materialization_label_row(
    label: Mapping[str, Any],
    root: Path,
    rallies: Mapping[str, Mapping[str, Any]],
    features_by_pair: Mapping[tuple[str, str], Mapping[str, float]],
    snapshot_contracts: Mapping[str, Mapping[str, Any]],
    current_ox_contract_id: str,
) -> dict[str, Any] | None:
    rally_id = str(label.get("rally_id") or "")
    candidate_id = str(label.get("candidate_id") or "")
    rally_row = rallies.get(rally_id)
    if rally_row is None:
        return None
    rally_as_of = str(rally_row.get("as_of") or "")
    session_cluster_id = str(rally_row.get("session_cluster_id") or "")
    label_record_sha256 = str(label.get("record_sha256") or "")
    if (
        not rally_as_of
        or not session_cluster_id
        or re.fullmatch(r"[0-9a-f]{64}", label_record_sha256) is None
        or label.get("schema") != store.DISTILLATION_SCHEMA
        or label.get("namespace") != "recall-distillation"
        or (
            str(label.get("previous_sha256") or "")
            and re.fullmatch(r"[0-9a-f]{64}", str(label.get("previous_sha256") or ""))
            is None
        )
        or canonical_json.canonical_json_sha256_strict(
            {key: value for key, value in label.items() if key != "record_sha256"}
        )
        != label_record_sha256
    ):
        return None
    raw_features = label.get("features") or features_by_pair.get(
        (rally_id, candidate_id)
    )
    if not isinstance(raw_features, Mapping):
        return None
    try:
        features = build_fast_features(raw_features)
    except DistillationError:
        return None
    if set(raw_features) != set(FAST_FEATURE_KEYS) or dict(raw_features) != features:
        return None
    verdict = str(label.get("verdict") or "")
    dimension = str(label.get("dimension") or "")
    if (
        label.get("authority") != "teacher-only"
        or verdict == "uncertain"
        or verdict
        not in (UTILITY_LABELS if dimension == "answer_utility" else RELEVANCE_LABELS)
    ):
        return None
    assignment = label.get("assignment")
    probe = isinstance(assignment, Mapping) and assignment.get("probe") is True
    is_ox = label.get("profile") == OX_SINGLE_PROFILE
    if is_ox and label.get("profile_contract_id") != current_ox_contract_id:
        return None
    if is_ox:
        try:
            contract = _read_ox_profile_contract(root, current_ox_contract_id)
            expiry = _ox_expiry(label.get("expires_at"))
            contract_expiry = _ox_expiry(contract.get("expires_at"))
        except DistillationError:
            return None
        contract_revision = contract.get("request_revision")
        if (
            not contract
            or contract_revision != OX_RAMP_REQUEST_REVISION
            or label.get("request_revision") != contract_revision
            or expiry != label.get("expires_at")
            or contract_expiry != contract.get("expires_at")
            or expiry != contract_expiry
        ):
            return None
        source_binding = _ox_contract_source_binding(root, current_ox_contract_id)
        if not source_binding or any(
            label.get(key) != value for key, value in source_binding.items()
        ):
            return None
        payload_source = label.get("payload_source")
        payload_digest = str(label.get("payload_digest") or "")
        expected_work_id = canonical_json.canonical_json_sha256_strict(
            {
                "kind": "ox-teacher-label-v1",
                "profile": OX_SINGLE_PROFILE,
                "cohort": OX_SINGLE_COHORT,
                "route": "opencode-go/ox-alpha-free",
                "profile_contract_id": current_ox_contract_id,
                "payload_digest": payload_digest,
            }
        )
        snapshot_contract = snapshot_contracts.get(rally_id, {})
        candidate_texts = snapshot_contract.get("candidate_text_sha256")
        candidate_text_sha256 = (
            str(candidate_texts.get(candidate_id) or "")
            if isinstance(candidate_texts, Mapping) and candidate_id in candidate_texts
            else None
        )
        if (
            not isinstance(payload_source, Mapping)
            or canonical_json.canonical_json_sha256_strict(payload_source)
            != payload_digest
            or label.get("work_id") != expected_work_id
            or not _materialization_payload_source_matches(
                payload_source,
                rally_id=rally_id,
                candidate_id=candidate_id,
                rally=rally_row,
                snapshot_sha256=str(snapshot_contract.get("snapshot_sha256") or ""),
                candidate_text_sha256=candidate_text_sha256,
                assignment=assignment if isinstance(assignment, Mapping) else None,
            )
        ):
            return None
    snapshot_contract = snapshot_contracts.get(rally_id, {})
    temporal_as_of = str(label.get("as_of") or rally_as_of)
    temporal_group_id = str(label.get("group_id") or session_cluster_id)
    feature_parity = features_by_pair.get((rally_id, candidate_id)) == features
    future_safe = (
        temporal_as_of == rally_as_of
        and snapshot_contract.get("as_of") == rally_as_of
        and re.fullmatch(r"[0-9a-f]{64}", snapshot_contract.get("snapshot_sha256", ""))
        is not None
        and feature_parity
    )
    return {
        "rally_id": rally_id,
        "candidate_id": candidate_id,
        "session_cluster_id": session_cluster_id,
        "as_of": rally_as_of,
        "dimension": dimension,
        "verdict": verdict,
        "authority": label["authority"],
        "features": features,
        "route": str(label.get("route") or ""),
        "route_identity": dict(label.get("route_identity") or {}),
        "teacher_role": str(label.get("teacher_role") or ""),
        "model_digest": str(label.get("model_digest") or ""),
        "generator_model_digest": str(label.get("generator_model_digest") or ""),
        "judge_model_digest": str(label.get("judge_model_digest") or ""),
        "generator_route_identity": dict(label.get("generator_route_identity") or {}),
        "judge_route_identity": dict(label.get("judge_route_identity") or {}),
        "counterfactual_ref": str(label.get("exposure_artifact_id") or ""),
        "a0_sha256": str(label.get("a0_sha256") or ""),
        "a1_sha256": str(label.get("a1_sha256") or ""),
        "blind_orders": list(label.get("blind_orders") or []),
        "counterfactual_producer": str(label.get("counterfactual_producer") or ""),
        "counterfactual_revision": str(label.get("counterfactual_revision") or ""),
        "probe": probe,
        "source": str(label.get("kind") or ""),
        "profile": str(label.get("profile") or label.get("teacher_profile") or ""),
        "cohort": str(label.get("cohort") or label.get("teacher_profile") or ""),
        "assignment_revision": str(
            label.get("assignment_revision")
            or (assignment.get("revision") if isinstance(assignment, Mapping) else "")
            or ""
        ),
        "assignment_authority": _training_assignment_authority(assignment),
        "profile_contract_id": str(label.get("profile_contract_id") or ""),
        "expires_at": str(label.get("expires_at") or ""),
        "identity_revision": str(label.get("identity_revision") or ""),
        "request_revision": str(label.get("request_revision") or ""),
        "group_id": temporal_group_id,
        "label_split_plan_id": str(label.get("split_plan_id") or ""),
        "order_agreement": label.get("order_agreement") is True,
        "label_record_sha256": label_record_sha256,
        "payload_digest": str(label.get("payload_digest") or ""),
        "payload_source": dict(label.get("payload_source") or {}),
        "work_id": str(label.get("work_id") or ""),
        "source_commit": str(label.get("source_commit") or ""),
        "source_tree_sha256": str(label.get("source_tree_sha256") or ""),
        "source_ox_identity_sha256": str(label.get("source_ox_identity_sha256") or ""),
        "negative_veto_conflict": label.get("negative_veto_conflict") is not None
        and label.get("negative_veto_conflict") is not False,
        "feature_parity": feature_parity,
        "future_leakage": not future_safe,
        **(
            {
                "status": str(label.get("status") or ""),
                "error_class": label.get("error_class"),
                "profile": str(label.get("profile") or ""),
                "cohort": str(label.get("cohort") or ""),
                "profile_contract_id": str(label.get("profile_contract_id") or ""),
                "expires_at": str(label.get("expires_at") or ""),
                "identity_revision": str(label.get("identity_revision") or ""),
                "request_revision": str(label.get("request_revision") or ""),
                "route_digest": str(label.get("route_digest") or ""),
                "route_identity_exact": label.get("route_identity")
                == {
                    "provider": "opencode-go",
                    "model": "opencode-go/ox-alpha-free",
                    "location": "remote",
                },
                "prompt_sha256": str(label.get("prompt_sha256") or ""),
                "schema_sha256": str(label.get("schema_sha256") or ""),
                "request_sha256": str(label.get("request_sha256") or ""),
                "provider_request_sha256": str(
                    label.get("provider_request_sha256") or ""
                ),
                "provider_response_request_sha256": str(
                    label.get("provider_response_request_sha256") or ""
                ),
                "source_commit": str(label.get("source_commit") or ""),
                "source_tree_sha256": str(label.get("source_tree_sha256") or ""),
                "source_ox_identity_sha256": str(
                    label.get("source_ox_identity_sha256") or ""
                ),
                "group_identity_exact": temporal_group_id == session_cluster_id,
                "as_of": temporal_as_of,
                "future_leakage_evidence_ref": (
                    f"candidate-snapshot:{snapshot_contract['snapshot_sha256']}"
                    if future_safe
                    else ""
                ),
                "repeat_pair_id": str(assignment.get("repeat_pair_id") or "")
                if isinstance(assignment, Mapping)
                else "",
                "fixed_repeat": (
                    assignment.get("fixed_repeat") is True
                    if isinstance(assignment, Mapping)
                    else False
                ),
                "order_swap": (
                    assignment.get("order_swap") is True
                    if isinstance(assignment, Mapping)
                    else False
                ),
                "blind_order": str(assignment.get("blind_order") or "")
                if isinstance(assignment, Mapping)
                else "",
            }
            if is_ox
            else {}
        ),
    }


def _materialization_rows(
    labels: Sequence[Mapping[str, Any]],
    root: Path,
    rallies: Mapping[str, Mapping[str, Any]],
    features_by_pair: Mapping[tuple[str, str], Mapping[str, float]],
    snapshot_contracts: Mapping[str, Mapping[str, str]],
    current_ox_contract_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous = ""
    for label in labels:
        if not isinstance(label, Mapping):
            continue
        # A direct list is not trusted merely because it carries a digest: its
        # canonical predecessor must form the same ledger chain.
        if label.get("previous_sha256") != previous:
            return []
        record = str(label.get("record_sha256") or "")
        if re.fullmatch(r"[0-9a-f]{64}", record) is None:
            return []
        previous = record
        row = _materialization_label_row(
            label,
            root,
            rallies,
            features_by_pair,
            snapshot_contracts,
            current_ox_contract_id,
        )
        if row is not None:
            rows.append(row)
    return rows


def _allocate_materialized_rows(
    rows: Sequence[Mapping[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Keep deterministic old, recent, and locked-test coverage when bounded."""

    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (_source_epoch(row), str(row.get("rally_id") or "")),
    )
    if len(ordered) <= limit:
        return ordered
    locked = [row for row in ordered if row.get("locked_test_read_only") is True]
    unlocked = [row for row in ordered if row.get("locked_test_read_only") is not True]
    recent_count = max(1, len(unlocked) // 3)
    old, recent = unlocked[:-recent_count], unlocked[-recent_count:]
    bands = (old, recent, locked)
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(bands):
        for band in bands:
            if band and len(selected) < limit:
                selected.append(band.pop(0))
    return sorted(
        selected,
        key=lambda row: (_source_epoch(row), str(row.get("rally_id") or "")),
    )


def _finalize_materialized_training_rows(
    root: Path,
    materialized: list[dict[str, Any]],
    limit: int,
    *,
    excluded_prior_contract_rows: int = 0,
) -> dict[str, Any]:
    split_plan_id = ""
    split: dict[str, str] = {}
    compatible_split_plan_ids: set[str] = set()
    try:
        split_plan = _read_split_plan(root)
        split_plan_id = str(split_plan["artifact_id"])
        split = {
            str(rally_id): str(value)
            for rally_id, value in split_plan["assignments"].items()
        }
        compatible_split_plan_ids.add(split_plan_id)
        for prior_plan_id in {
            str(row.get("label_split_plan_id") or "") for row in materialized
        } - {split_plan_id, ""}:
            try:
                prior = _read_split_plan_artifact(root, prior_plan_id)
            except (DistillationError, store.DistillationStoreError):
                continue
            if (
                prior.get("feature_revision") == split_plan.get("feature_revision")
                and prior.get("split_revision") == split_plan.get("split_revision")
                and all(
                    split.get(str(rally_id)) == str(value)
                    for rally_id, value in prior["assignments"].items()
                )
            ):
                compatible_split_plan_ids.add(prior_plan_id)
    except (KeyError, DistillationError, store.DistillationStoreError):
        split = grouped_rolling_split(materialized) if materialized else {}
    rows = []
    for row in materialized:
        row_split = split.get(row["rally_id"], "embargo")
        split_bound = (
            row.get("profile") == OX_SINGLE_PROFILE
            and row.get("label_split_plan_id") in compatible_split_plan_ids
            and bool(split_plan_id)
            and row_split == "test"
        )
        fixed_split_plan = (
            row.get("profile") == OX_SINGLE_PROFILE
            and row.get("label_split_plan_id") in compatible_split_plan_ids
            and bool(split_plan_id)
            and row_split in {"train", "validation", "test"}
        )
        rows.append(
            {
                **row,
                "split": row_split,
                "split_plan_id": split_plan_id,
                "locked_test_read_only": bool(split_plan_id) and row_split == "test",
                "locked_test_evidence_ref": (
                    f"split-plan:{split_plan_id}"
                    if split_plan_id and row_split == "test"
                    else ""
                ),
                **(
                    {
                        "fixed_split_plan": fixed_split_plan,
                        "locked_test_read_only": split_bound,
                        "locked_test_evidence_ref": (
                            f"split-plan:{split_plan_id}" if split_bound else ""
                        ),
                    }
                    if row.get("profile") == OX_SINGLE_PROFILE
                    else {}
                ),
            }
        )
    rows = _allocate_materialized_rows(rows, limit)
    _, _, artifact = store.write_immutable(
        store.distillation_dir(root) / "training-snapshots",
        {
            "kind": "text-parity-training-snapshot",
            "feature_revision": TEXT_FEATURE_REVISION,
            "rows": rows,
            "excluded_prior_contract_rows": excluded_prior_contract_rows,
            "label_chain_head": store.chain_head(
                store.distillation_dir(root) / "label-ledger.jsonl"
            )["head_sha256"],
        },
        schema="chronovisor.recall-distill-training.v1",
    )
    return artifact


def materialize_training_rows(
    root: Path | None = None,
    *,
    limit: int = 10_000,
    _rallies: Sequence[Mapping[str, Any]] | None = None,
    _snapshots: Mapping[str, Mapping[str, Any]] | None = None,
    _label_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Join labels only to byte-identical prospective live feature snapshots."""

    if limit <= 0 or limit > 10_000:
        raise DistillationError("training materialization limit is invalid")
    root = root or CHRONOVISOR_ROOT
    rallies = _materialization_rallies(root, _rallies)
    snapshots = _materialization_snapshots(root, _snapshots)
    features_by_pair, snapshot_contracts = _materialization_feature_pairs(
        root, rallies, snapshots
    )
    labels = (
        _label_rows
        if _label_rows is not None
        else _read_chain(store.distillation_dir(root) / "label-ledger.jsonl")
    )
    current_ox_contract_id = _current_ox_profile_contract_id(root)
    excluded_prior_contract_rows = sum(
        isinstance(label, Mapping)
        and label.get("profile") == OX_SINGLE_PROFILE
        and label.get("profile_contract_id") != current_ox_contract_id
        for label in labels
    )
    materialized = _materialization_rows(
        labels,
        root,
        rallies,
        features_by_pair,
        snapshot_contracts,
        current_ox_contract_id,
    )
    return _finalize_materialized_training_rows(
        root,
        materialized,
        limit,
        excluded_prior_contract_rows=excluded_prior_contract_rows,
    )


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


def _authoritative_materialized_row_binding(
    root: Path, row: Mapping[str, Any], split_plan: Mapping[str, Any] | None
) -> bool:
    """Bind a sealed training row back to its canonical ledger source.

    Digest self-consistency is not provenance: a caller can recompute an
    envelope for an unrelated rally/candidate pair.  The label record, live
    rally manifest and candidate snapshot are the authority that make the
    envelope meaningful.
    """

    record_sha256 = str(row.get("label_record_sha256") or "")
    if re.fullmatch(r"[0-9a-f]{64}", record_sha256) is None:
        return False
    labels = _read_chain(store.distillation_dir(root) / "label-ledger.jsonl")
    labels_by_record = {
        str(label.get("record_sha256") or ""): label
        for label in labels
        if isinstance(label, Mapping)
    }
    label = labels_by_record.get(record_sha256)
    if label is None:
        return False
    rallies = _materialization_rallies(root, None)
    snapshots = _materialization_snapshots(root, None)
    features_by_pair, snapshot_contracts = _materialization_feature_pairs(
        root, rallies, snapshots
    )
    canonical = _materialization_label_row(
        label,
        root,
        rallies,
        features_by_pair,
        snapshot_contracts,
        _current_ox_profile_contract_id(root),
    )
    if canonical is None or any(
        row.get(key) != value for key, value in canonical.items()
    ):
        return False
    if str(
        row.get("source") or ""
    ) == "counterfactual-label" and not _sealed_counterfactual_exposure_binding(
        root, row, label, rallies.get(str(row.get("rally_id") or ""))
    ):
        return False
    if not _configured_local_route_binding(row):
        return False
    expected_plan = split_plan
    if expected_plan is None:
        try:
            expected_plan = _read_split_plan_artifact(
                root, str(row.get("split_plan_id") or "")
            )
        except (DistillationError, store.DistillationStoreError):
            return False
    plan_id = str(expected_plan.get("artifact_id") or "")
    rally_id = str(row.get("rally_id") or "")
    split = expected_plan.get("assignments", {}).get(rally_id)
    rally = rallies.get(rally_id)
    if str(row.get("source") or "") == "counterfactual-label" and (
        rally is None
        or row.get("label_split_plan_id") != plan_id
        or row.get("group_id") != rally.get("session_cluster_id")
        or row.get("as_of") != rally.get("as_of")
    ):
        return False
    return not (
        row.get("split_plan_id") != plan_id
        or split not in {"train", "validation", "test"}
        or row.get("split") != split
        or row.get("locked_test_read_only") is not (split == "test")
        or row.get("locked_test_evidence_ref")
        != (f"split-plan:{plan_id}" if split == "test" else "")
    )


def _sealed_counterfactual_exposure_binding(
    root: Path,
    row: Mapping[str, Any],
    label: Mapping[str, Any],
    rally: Mapping[str, Any] | None,
) -> bool:
    """Require CF rows to remain anchored to one verified prior exposure.

    The hash chain proves that the label existed, but not that its blind
    comparison used an actually captured exposure.  Reuse the verified
    exposure-receipt projection here so direct publication cannot substitute a
    self-consistent, nonexistent ``counterfactual_ref``.
    """

    reference = str(row.get("counterfactual_ref") or "")
    if (
        rally is None
        or re.fullmatch(r"[0-9a-f]{64}", reference) is None
        or label.get("exposure_artifact_id") != reference
    ):
        return False
    authority_fields = (
        "rally_id",
        "profile",
        "cohort",
        "assignment_revision",
        "as_of",
        "group_id",
        "a0_sha256",
        "a1_sha256",
        "blind_orders",
        "counterfactual_producer",
        "counterfactual_revision",
        "generator_route_identity",
        "generator_model_digest",
        "judge_route_identity",
        "judge_model_digest",
    )
    if (
        any(row.get(key) != label.get(key) for key in authority_fields)
        or (row.get("candidate_id") != str(label.get("candidate_id") or ""))
        or (row.get("label_split_plan_id") != label.get("split_plan_id"))
    ):
        return False
    rally_receipts = rally.get("exposure_receipts")
    if not isinstance(rally_receipts, list):
        return False
    matching_rally_receipts = [
        receipt
        for receipt in rally_receipts
        if isinstance(receipt, Mapping)
        and receipt.get("exposure_artifact_id") == reference
    ]
    if len(matching_rally_receipts) != 1:
        return False
    try:
        exact_receipts = _exposure_map(root).get(
            (
                str(rally.get("host") or ""),
                str(rally.get("session_id_sha256") or ""),
                str(rally.get("query_sha256") or ""),
            ),
            [],
        )
        artifact = store.read_sealed(
            store.distillation_dir(root) / "exposures" / f"{reference}.json",
            schema="chronovisor.recall-exact-exposure.v1",
        )
    except (KeyError, TypeError, store.DistillationStoreError):
        return False
    receipt = next(
        (
            item
            for item in exact_receipts
            if item.get("exposure_artifact_id") == reference
        ),
        None,
    )
    if receipt is None or artifact.get("artifact_id") != reference:
        return False
    candidate_ids = receipt.get("candidate_ids")
    candidate_id = str(row.get("candidate_id") or "")
    if (
        not isinstance(candidate_ids, list)
        or not candidate_ids
        or candidate_id not in {str(value) for value in candidate_ids}
        or artifact.get("candidate_ids") != candidate_ids
        or row.get("assignment_authority")
        != _training_assignment_authority(label.get("assignment"))
        or row.get("assignment_authority")
        != _training_assignment_authority(
            {"revision": ASSIGNMENT_REVISION, "kind": "counterfactual"}
        )
    ):
        return False
    refs = artifact.get("candidate_refs")
    if isinstance(refs, list):
        matching_refs = [
            item
            for item in refs
            if isinstance(item, Mapping)
            and str(item.get("candidate_id") or "") == candidate_id
        ]
        if not matching_refs:
            return False
        declared_hashes = {
            str(value)
            for item in matching_refs
            for value in (
                item.get("content_sha256"),
                item.get("page_content_sha256"),
                item.get("text_sha256"),
            )
            if isinstance(value, str) and value
        }
        if declared_hashes:
            try:
                snapshot = _materialization_snapshots(root, None).get(
                    str(row.get("rally_id") or ""), {}
                )
                candidates = snapshot.get("candidates", [])
            except (AttributeError, KeyError, TypeError, store.DistillationStoreError):
                return False
            candidate_hashes = {
                str(candidate.get("text_sha256") or "")
                for candidate in candidates
                if isinstance(candidate, Mapping)
                and str(candidate.get("candidate_id") or "") == candidate_id
            }
            if not candidate_hashes.intersection(declared_hashes):
                return False
    return matching_rally_receipts[0].get("candidate_ids") == receipt.get(
        "candidate_ids"
    )


def _configured_local_route_binding(row: Mapping[str, Any]) -> bool:
    """Bind local teacher/CF identities to the currently configured routes."""

    source = str(row.get("source") or "")
    if source not in {"teacher-label", "counterfactual-label"}:
        return True
    if source == "teacher-label" and row.get("profile") == OX_SINGLE_PROFILE:
        return True
    try:
        from chronovisor.core import ollama

        roles = (
            *TEACHER_ROLES,
            "recall.distill.answer_generator",
            "recall.distill.utility_judge",
        )
        routes = ollama.runtime_generation_routes(roles)
        digests = ollama.runtime_generation_route_fingerprints(routes)
    except Exception:
        return False
    expected = {
        route.role: {
            "role": route.role,
            "provider": route.provider,
            "model": route.model,
            "location": route.location,
        }
        for route in routes
    }
    if tuple(expected) != roles or any(
        identity["location"] != "local" for identity in expected.values()
    ):
        return False
    if source == "teacher-label":
        route = str(row.get("route") or "")
        authority = {
            "profile": LOCAL_TRIAD_PROFILE,
            "cohort": LOCAL_TRIAD_PROFILE,
            "profile_contract_id": "",
            "expires_at": "",
            "identity_revision": "local-teacher-v1",
            "request_revision": "local-teacher-v1",
            "assignment_revision": ASSIGNMENT_REVISION,
        }
        return (
            route in TEACHER_ROLES
            and all(row.get(key) == value for key, value in authority.items())
            and row.get("route_identity") == expected[route]
            and row.get("model_digest") == digests[route]
        )
    generator = "recall.distill.answer_generator"
    judge = "recall.distill.utility_judge"
    return (
        row.get("profile") in {LOCAL_TRIAD_PROFILE, OX_SINGLE_PROFILE}
        and row.get("cohort")
        == (
            OX_SINGLE_COHORT
            if row.get("profile") == OX_SINGLE_PROFILE
            else LOCAL_TRIAD_PROFILE
        )
        and row.get("assignment_revision") == ASSIGNMENT_REVISION
        and row.get("identity_revision") == "local-blind-counterfactual-v1"
        and row.get("request_revision") == "local-blind-counterfactual-v1"
        and (
            (
                row.get("profile") == LOCAL_TRIAD_PROFILE
                and row.get("profile_contract_id") == ""
                and row.get("expires_at") == ""
            )
            or (
                row.get("profile") == OX_SINGLE_PROFILE
                and re.fullmatch(
                    r"[0-9a-f]{64}", str(row.get("profile_contract_id") or "")
                )
                is not None
                and _same_future_ox_expiry(row.get("expires_at"), row.get("expires_at"))
            )
        )
        and row.get("counterfactual_producer") == "chronovisor-local-blind-v1"
        and row.get("counterfactual_revision") == "two-order-locked-v1"
        and set(row.get("blind_orders") or []) == {"a0_first", "a1_first"}
        and row.get("generator_route_identity") == expected[generator]
        and row.get("judge_route_identity") == expected[judge]
        and row.get("generator_model_digest") == digests[generator]
        and row.get("judge_model_digest") == digests[judge]
        and row.get("generator_model_digest") != row.get("judge_model_digest")
    )


def _materialized_row_integrity(
    row: Mapping[str, Any],
    *,
    root: Path | None = None,
    split_plan: Mapping[str, Any] | None = None,
) -> bool:
    """One gate for local and OX rows that came through materialization."""

    if re.fullmatch(r"[0-9a-f]{64}", str(row.get("label_record_sha256") or "")) is None:
        return False
    if (
        row.get("future_leakage") is not False
        or row.get("feature_parity") is not True
        or row.get("negative_veto_conflict") is True
        or not str(row.get("route") or "")
        or re.fullmatch(r"[0-9a-f]{64}", str(row.get("model_digest") or "")) is None
    ):
        return False
    if row.get("source") != "counterfactual-label":
        if row.get("profile") == OX_SINGLE_PROFILE:
            payload = str(row.get("payload_digest") or "")
            source = row.get("payload_source")
            contract = (
                _read_ox_profile_contract(
                    root, str(row.get("profile_contract_id") or "")
                )
                if root is not None
                else {}
            )
            contract_binding = (
                _validate_ox_source_binding(
                    {
                        key: str(contract.get(key) or "")
                        for key in (
                            "source_commit",
                            "source_tree_sha256",
                            "source_ox_identity_sha256",
                        )
                        if contract.get(key) is not None
                    }
                )
                if contract
                else {}
            )
            semantic = (
                isinstance(source, Mapping)
                and re.fullmatch(r"[0-9a-f]{64}", payload) is not None
                and canonical_json.canonical_json_sha256_strict(source) == payload
                and row.get("route") == "opencode-go/ox-alpha-free"
                and row.get("teacher_role") == OX_TEACHER_ROLE
                and row.get("cohort") == OX_SINGLE_COHORT
                and row.get("assignment_revision") == "single-teacher-v1"
                and row.get("route_identity")
                == OX_ALPHA_FIXED_IDENTITY["route_identity"]
                and row.get("route_identity_exact") is True
                and row.get("model_digest") == OX_ALPHA_FIXED_IDENTITY["model_digest"]
                and row.get("prompt_sha256")
                == OX_ALPHA_FIXED_IDENTITY["prompt_template_sha256"]
                and row.get("schema_sha256")
                == OX_ALPHA_FIXED_IDENTITY["schema_revision_sha256"]
                and row.get("route_digest") == OX_ALPHA_FIXED_IDENTITY["route_digest"]
                and row.get("identity_revision") == OX_ALPHA_FIXED_IDENTITY["revision"]
                and row.get("request_revision") == OX_RAMP_REQUEST_REVISION
                and contract.get("request_revision") == OX_RAMP_REQUEST_REVISION
                and expected_ox_request_sha256(
                    profile_contract_id=str(row.get("profile_contract_id") or ""),
                    payload_digest=payload,
                )
                == row.get("request_sha256")
                and expected_ox_provider_request_sha256(
                    profile_contract_id=str(row.get("profile_contract_id") or ""),
                    payload_digest=payload,
                    work_id=str(row.get("work_id") or ""),
                    expires_at=str(row.get("expires_at") or ""),
                )
                == row.get("provider_request_sha256")
                and row.get("provider_response_request_sha256")
                == row.get("provider_request_sha256")
                and bool(contract_binding)
                and all(
                    row.get(key) == value for key, value in contract_binding.items()
                )
            )
        else:
            semantic = True
    else:
        semantic = (
            row.get("profile") in {LOCAL_TRIAD_PROFILE, OX_SINGLE_PROFILE}
            and row.get("cohort")
            == (
                OX_SINGLE_COHORT
                if row.get("profile") == OX_SINGLE_PROFILE
                else LOCAL_TRIAD_PROFILE
            )
            and row.get("assignment_revision") == ASSIGNMENT_REVISION
            and row.get("assignment_authority")
            == _training_assignment_authority(
                {"revision": ASSIGNMENT_REVISION, "kind": "counterfactual"}
            )
            and row.get("identity_revision") == "local-blind-counterfactual-v1"
            and row.get("request_revision") == "local-blind-counterfactual-v1"
            and (
                (
                    row.get("profile") == LOCAL_TRIAD_PROFILE
                    and row.get("profile_contract_id") == ""
                    and row.get("expires_at") == ""
                )
                or (
                    row.get("profile") == OX_SINGLE_PROFILE
                    and re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(row.get("profile_contract_id") or ""),
                    )
                    is not None
                    and _same_future_ox_expiry(
                        row.get("expires_at"), row.get("expires_at")
                    )
                )
            )
            and row.get("order_agreement") is True
            and re.fullmatch(r"[0-9a-f]{64}", str(row.get("counterfactual_ref") or ""))
            is not None
            and re.fullmatch(r"[0-9a-f]{64}", str(row.get("a0_sha256") or ""))
            is not None
            and re.fullmatch(r"[0-9a-f]{64}", str(row.get("a1_sha256") or ""))
            is not None
            and str(row.get("generator_model_digest") or "")
            != str(row.get("judge_model_digest") or "")
            and all(
                re.fullmatch(r"[0-9a-f]{64}", str(row.get(name) or "")) is not None
                for name in ("generator_model_digest", "judge_model_digest")
            )
            and bool(row.get("generator_route_identity"))
            and bool(row.get("judge_route_identity"))
            and row["generator_route_identity"].get("location") == "local"
            and row["judge_route_identity"].get("location") == "local"
            and bool(row["generator_route_identity"].get("provider"))
            and bool(row["judge_route_identity"].get("provider"))
            and bool(row["generator_route_identity"].get("model"))
            and bool(row["judge_route_identity"].get("model"))
            and row["generator_route_identity"] != row["judge_route_identity"]
            and row.get("counterfactual_producer") == "chronovisor-local-blind-v1"
            and row.get("counterfactual_revision") == "two-order-locked-v1"
            and set(row.get("blind_orders") or []) == {"a0_first", "a1_first"}
            and (
                row.get("split") != "test"
                or (
                    row.get("locked_test_read_only") is True
                    and str(row.get("locked_test_evidence_ref") or "")
                    == f"split-plan:{str(row.get('split_plan_id') or '')}"
                )
            )
        )
    return semantic and (
        root is None or _authoritative_materialized_row_binding(root, row, split_plan)
    )


def _active_training_cohort(
    rows: Sequence[Mapping[str, Any]],
    *,
    teacher_profile: str = LOCAL_TRIAD_PROFILE,
    profile_contract_id: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the latest coherent local-model cohort and resplit only it."""

    if teacher_profile == OX_SINGLE_PROFILE:
        single_rows = [
            dict(row)
            for row in rows
            if row.get("profile") == OX_SINGLE_PROFILE
            and row.get("cohort") == OX_SINGLE_COHORT
            and row.get("profile_contract_id") == profile_contract_id
        ]
        single_cohort = {
            "revision": "single-teacher-cohort-v1",
            "profile": OX_SINGLE_PROFILE,
            "cohort": OX_SINGLE_COHORT,
            "profile_contract_id": profile_contract_id,
            "model_digests": sorted(
                {str(row.get("model_digest") or "") for row in single_rows}
            ),
        }
        return single_rows, {
            **single_cohort,
            "cohort_sha256": canonical_json.canonical_json_sha256_strict(single_cohort),
        }

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
            if (
                counterfactual_digests is None
                or (
                    row.get("generator_model_digest"),
                    row.get("judge_model_digest"),
                )
                != counterfactual_digests
            ):
                continue
        else:
            continue
        selected.append(dict(row))
    fixed_ids = {str(row.get("split_plan_id") or "") for row in selected}
    if len(fixed_ids) != 1 or not next(iter(fixed_ids), ""):
        split = grouped_rolling_split(selected) if selected else {}
        selected = [{**row, "split": split[str(row["rally_id"])]} for row in selected]
    cohort: dict[str, Any] = {
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

    integrity_failed = any(
        isinstance(row, Mapping) and not _materialized_row_integrity(row)
        for row in rows
    )
    if config.teacher_profile == OX_SINGLE_PROFILE:
        invalid_input_rows = any(not isinstance(row, Mapping) for row in rows)
        current_contract_id = ""
        reasons: list[str] = []
        if integrity_failed:
            reasons.append("row_integrity_failed")
        if root is None:
            reasons.append("profile_contract_unavailable")
        else:
            try:
                current_contract_id = str(
                    _ensure_ox_profile_contract(root, config)["artifact_id"]
                )
            except (DistillationError, store.DistillationStoreError, KeyError):
                reasons.append("profile_contract_unavailable")
        single_rows = [
            row
            for row in rows
            if isinstance(row, Mapping) and row.get("profile") == OX_SINGLE_PROFILE
        ]
        gate = evaluate_single_teacher_gate(
            single_rows,
            profile=OX_SINGLE_PROFILE,
            cohort=OX_SINGLE_COHORT,
            profile_contract_id=current_contract_id,
            min_labels=config.hard_floor_teacher_labels,
            min_per_class=config.hard_floor_teacher_per_class,
            min_repeat_pairs=config.hard_floor_probe_pairs,
            row_validator=(
                (lambda row: _materialized_row_integrity(row, root=root))
                if root is not None
                else None
            ),
        )
        reasons.extend(gate["reasons"])
        if invalid_input_rows:
            reasons.append("input_row_invalid")
        if config.ox_enabled is not True:
            reasons.append("ox_profile_disabled")
        counterfactual_pairs = [
            row
            for row in single_rows
            if row.get("source") == "counterfactual-label"
            and row.get("split") == "test"
            and row.get("verdict") in {"helpful", "harmful"}
            and re.fullmatch(r"[0-9a-f]{64}", str(row.get("counterfactual_ref") or ""))
            and re.fullmatch(r"[0-9a-f]{64}", str(row.get("a0_sha256") or ""))
            and re.fullmatch(r"[0-9a-f]{64}", str(row.get("a1_sha256") or ""))
            and isinstance(row.get("generator_route_identity"), Mapping)
            and isinstance(row.get("judge_route_identity"), Mapping)
            and row.get("generator_route_identity")
            and row.get("judge_route_identity")
            and row.get("counterfactual_producer") == "chronovisor-local-blind-v1"
            and row.get("counterfactual_revision") == "two-order-locked-v1"
            and set(row.get("blind_orders") or []) == {"a0_first", "a1_first"}
            and row.get("profile") == OX_SINGLE_PROFILE
            and row.get("cohort") == OX_SINGLE_COHORT
            and row.get("profile_contract_id") == current_contract_id
            and row.get("identity_revision") == "local-blind-counterfactual-v1"
            and _same_future_ox_expiry(row.get("expires_at"), config.ox_expires_at)
        ]
        if len(counterfactual_pairs) < max(100, config.hard_floor_counterfactual_pairs):
            reasons.append("counterfactual_pairs_below_floor")
        if current_contract_id:
            if gate["identity"]["profile_contract_id"] != current_contract_id:
                reasons.append("profile_contract_mismatch")
            try:
                split_plan = _read_split_plan(root)
            except (DistillationError, store.DistillationStoreError, KeyError):
                reasons.append("split_plan_unavailable")
            else:
                _, model_cohort = _active_training_cohort(
                    single_rows,
                    teacher_profile=OX_SINGLE_PROFILE,
                    profile_contract_id=current_contract_id,
                )
                if (
                    split_plan.get("model_cohort_sha256")
                    != model_cohort["cohort_sha256"]
                ):
                    reasons.append("split_plan_cohort_mismatch")
        gate = {
            **gate,
            "passed": not reasons,
            "reasons": sorted(set(reasons)),
        }
        return {
            **gate,
            "teacher_counts": {
                "total": gate["labels"]["eligible"],
                **{
                    verdict: gate["labels"][verdict]
                    for verdict in ("relevant", "irrelevant")
                },
            },
            "counterfactual_pairs": len(counterfactual_pairs),
            "probe": {
                "pairs": gate["blind_repeat"]["complete_pairs"],
                "locked_test_only": gate["locked_test"]["read_only"],
                "stable": gate["blind_repeat"]["stable"],
                "route_stability_wilson_lower": gate["blind_repeat"]["wilson_lower"],
                "is_truth": False,
            },
            "route_folds": {},
            "counterfactual_direction": {"denominator": 0, "wilson_lower": 0.0},
        }

    if any(not isinstance(row, Mapping) for row in rows):
        return {
            "schema": "chronovisor.recall-offline-training-gate.v2",
            "truth_authority": "teacher_only_not_verified",
            "passed": False,
            "reasons": ["input_row_invalid"],
            "teacher_counts": {"total": 0, "relevant": 0, "irrelevant": 0},
            "counterfactual_pairs": 0,
            "probe": {
                "pairs": 0,
                "locked_test_only": False,
                "stable": 0,
                "route_stability_wilson_lower": 0.0,
                "is_truth": False,
            },
            "route_folds": {},
            "counterfactual_direction": {"denominator": 0, "wilson_lower": 0.0},
        }
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
    if integrity_failed:
        reasons.append("row_integrity_failed")
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
                str(_stable_pointer_read(root, "active")["policy_id"]), root
            )
        except (KeyError, store.DistillationStoreError, DistillationError):
            pass
        try:
            lkg = _load_policy(str(_stable_pointer_read(root, "lkg")["policy_id"]), root)
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


def _verify_candidate_lineage(
    root: Path,
    candidate: Mapping[str, Any],
    candidate_policy: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> str | None:
    """Validate every durable candidate path before it can be reused."""

    lineage = candidate_policy.get("lineage")
    if not isinstance(lineage, Mapping):
        return "candidate_lineage_incomplete"
    if lineage.get("baseline_artifact_id") != baseline.get("artifact_id"):
        return "candidate_baseline_mismatch"
    if candidate.get("policy_id") != candidate_policy.get("artifact_id"):
        return "candidate_lineage_incomplete"
    required = (
        "training_snapshot_id",
        "locked_replay_id",
        "model_cohort_sha256",
        "raw_watermark",
        "label_chain_head",
        "split_plan_id",
        "offline_gate_sha256",
        "training_rows_sha256",
        "candidate_head",
    )
    if any(
        re.fullmatch(r"[0-9a-f]{64}", str(lineage.get(key) or "")) is None
        for key in required
    ):
        return "candidate_lineage_incomplete"
    if lineage.get("feature_revision") != TEXT_FEATURE_REVISION:
        return "candidate_lineage_incomplete"
    try:
        snapshot = store.read_sealed(
            store.distillation_dir(root)
            / "training-snapshots"
            / f"{lineage['training_snapshot_id']}.json",
            schema="chronovisor.recall-distill-training.v1",
        )
        replay = store.read_sealed(
            store.distillation_dir(root)
            / "locked-replays"
            / f"{lineage['locked_replay_id']}.json",
            schema="chronovisor.recall-distill-locked-replay.v1",
        )
        sealed_baseline = store.read_sealed(
            store.distillation_dir(root)
            / "baselines"
            / f"{lineage['baseline_artifact_id']}.json",
            schema=BASELINE_SCHEMA,
        )
        split_plan = _read_split_plan_artifact(root, str(lineage["split_plan_id"]))
    except (DistillationError, store.DistillationStoreError, KeyError):
        return "candidate_lineage_incomplete"
    rows = snapshot.get("rows")
    replay_rows = replay.get("training_rows")
    expected_rows: list[dict[str, Any]] = []
    if isinstance(rows, list) and all(isinstance(row, Mapping) for row in rows):
        profile = (
            OX_SINGLE_PROFILE
            if any(row.get("profile") == OX_SINGLE_PROFILE for row in rows)
            else LOCAL_TRIAD_PROFILE
        )
        contract_id = str(lineage.get("profile_contract_id") or "")
        if profile == OX_SINGLE_PROFILE and not contract_id:
            contract_id = _current_ox_profile_contract_id(root)
        expected_rows, _ = _active_training_cohort(
            rows, teacher_profile=profile, profile_contract_id=contract_id
        )
    snapshot_hashes = [
        canonical_json.canonical_json_sha256_strict(row)
        for row in rows or []
        if isinstance(row, Mapping)
    ]
    replay_hashes = [
        canonical_json.canonical_json_sha256_strict(row)
        for row in replay_rows or []
        if isinstance(row, Mapping)
    ]
    expected_hashes = [
        canonical_json.canonical_json_sha256_strict(row) for row in expected_rows
    ]
    ox_rows = [
        row
        for row in rows or []
        if isinstance(row, Mapping) and row.get("profile") == OX_SINGLE_PROFILE
    ]
    if ox_rows:
        current_contract_id = _current_ox_profile_contract_id(root)
        try:
            current_contract = _read_ox_profile_contract(root, current_contract_id)
            if not current_contract:
                raise DistillationError("OX profile contract is invalid")
            expiry_valid = _ox_expiry(current_contract.get("expires_at"))
        except (DistillationError, store.DistillationStoreError):
            return "candidate_lineage_incomplete"
        if (
            not current_contract_id
            or lineage.get("profile_contract_id") != current_contract_id
            or any(
                row.get("profile_contract_id") != current_contract_id
                or _same_future_ox_expiry(row.get("expires_at"), expiry_valid)
                is not True
                for row in [*ox_rows, *(replay_rows or [])]
                if isinstance(row, Mapping) and row.get("profile") == OX_SINGLE_PROFILE
            )
        ):
            return "candidate_lineage_incomplete"
    if (
        snapshot.get("artifact_id") != lineage["training_snapshot_id"]
        or snapshot.get("label_chain_head") != lineage["label_chain_head"]
        or store.chain_head(store.distillation_dir(root) / "label-ledger.jsonl")[
            "head_sha256"
        ]
        != lineage["label_chain_head"]
        or store.chain_head(store.distillation_dir(root) / "candidate-ledger.jsonl")[
            "head_sha256"
        ]
        != lineage["candidate_head"]
        or snapshot.get("feature_revision") != TEXT_FEATURE_REVISION
        or not isinstance(rows, list)
        or not rows
        or not isinstance(replay_rows, list)
        or not replay_rows
        or len(replay_hashes) != len(replay_rows)
        or len(set(replay_hashes)) != len(replay_hashes)
        or set(replay_hashes) - set(snapshot_hashes)
        or set(replay_hashes) != set(expected_hashes)
        or any(
            not isinstance(row, Mapping)
            or not _materialized_row_integrity(row, root=root, split_plan=split_plan)
            for row in rows
        )
        or {str(row.get("split") or "") for row in rows if isinstance(row, Mapping)}
        < {"train", "validation", "test"}
        or not all(
            any(
                row.get("split") == split
                for row in replay_rows
                if isinstance(row, Mapping)
            )
            for split in ("train", "validation")
        )
        or not any(
            row.get("split") == "test" and row.get("locked_test_read_only") is True
            for row in replay_rows
            if isinstance(row, Mapping)
        )
        or replay.get("training_snapshot_id") != lineage["training_snapshot_id"]
        or replay.get("baseline_artifact_id") != lineage["baseline_artifact_id"]
        or replay.get("model_cohort_sha256") != lineage["model_cohort_sha256"]
        or replay.get("split_revision") != "grouped-rolling-v1"
        or split_plan.get("feature_revision") != TEXT_FEATURE_REVISION
        or split_plan.get("model_cohort_sha256") != lineage["model_cohort_sha256"]
        or sealed_baseline.get("raw_watermark") != lineage["raw_watermark"]
        or sealed_baseline.get("raw_watermark") != baseline.get("raw_watermark")
        or sealed_baseline.get("hard_floor", {}).get("p5_allowed") is not True
        or sealed_baseline.get("offline_training_gate", {}).get("passed") is not True
        or sealed_baseline.get("frozen_contract", {}).get("hard_floors")
        != _canonical_hard_floors()
        or canonical_json.canonical_json_sha256_strict(
            sealed_baseline.get("offline_training_gate")
        )
        != lineage["offline_gate_sha256"]
        or replay.get("offline_gate_sha256") != lineage["offline_gate_sha256"]
        or replay.get("training_rows_sha256") != lineage["training_rows_sha256"]
        or canonical_json.canonical_json_sha256_strict(replay_rows)
        != lineage["training_rows_sha256"]
        or replay.get("policy_sha256") != _policy_payload_digest(candidate_policy)
        or train_tiny_policy(replay_rows)
        != {key: candidate_policy.get(key) for key in train_tiny_policy(replay_rows)}
    ):
        return "candidate_lineage_incomplete"
    return None


def _maybe_publish_candidate(
    root: Path,
    config: DistillationConfig,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    if not _has_canonical_hard_floors(config):
        return {"status": "held", "reason": "canonical_floor_lowered"}
    if baseline.get("hard_floor", {}).get("p5_allowed") is not True:
        return {"status": "held", "reason": "p5_hard_floor"}
    try:
        candidate = _stable_pointer_read(root, "candidate")
        candidate_policy = _load_policy(str(candidate["policy_id"]), root)
        reason = _verify_candidate_lineage(root, candidate, candidate_policy, baseline)
        if reason:
            return {"status": "held", "reason": reason}
        return {"status": "candidate", "policy_id": candidate["policy_id"]}
    except (store.DistillationStoreError, DistillationError, KeyError):
        pass
    training = materialize_training_rows(root)
    rows = training["rows"]
    required_splits = {"train", "validation", "test"}
    if not rows or required_splits - {str(row.get("split") or "") for row in rows}:
        return {"status": "held", "reason": "training_lineage_incomplete"}
    offline_gate = _offline_training_gate(rows, config, root=root)
    active_rows, model_cohort = _active_training_cohort(
        rows,
        teacher_profile=config.teacher_profile,
        profile_contract_id=(
            _current_ox_profile_contract_id(root)
            if config.teacher_profile == OX_SINGLE_PROFILE
            else ""
        ),
    )
    if offline_gate != baseline.get("offline_training_gate"):
        return {"status": "held", "reason": "offline_gate_baseline_mismatch"}
    if offline_gate["passed"] is not True:
        return {"status": "held", "reason": "offline_training_gate"}
    try:
        _stable_pointer_read(root, "lkg")
        _stable_pointer_read(root, "active")
    except store.DistillationStoreError:
        return {"status": "held", "reason": "sealed_incumbent_missing"}
    policy = train_tiny_policy(active_rows)
    training_rows_sha256 = canonical_json.canonical_json_sha256_strict(active_rows)
    replay_id, _, _ = store.write_immutable(
        store.distillation_dir(root) / "locked-replays",
        {
            "kind": "locked-replay-input",
            "training_snapshot_id": training["artifact_id"],
            "training_rows": active_rows,
            "baseline_artifact_id": baseline["artifact_id"],
            "policy_sha256": _policy_payload_digest(policy),
            "training_rows_sha256": training_rows_sha256,
            "candidate_head": store.chain_head(
                store.distillation_dir(root) / "candidate-ledger.jsonl"
            )["head_sha256"],
            "profile_contract_id": _current_ox_profile_contract_id(root)
            if config.teacher_profile == OX_SINGLE_PROFILE
            else "",
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
            "raw_watermark": baseline["raw_watermark"],
            "label_chain_head": training["label_chain_head"],
            "feature_revision": training["feature_revision"],
            "split_plan_id": str(
                next(
                    iter({str(row.get("split_plan_id") or "") for row in active_rows}),
                    "",
                )
            ),
            "offline_gate_sha256": canonical_json.canonical_json_sha256_strict(
                offline_gate
            ),
            "training_rows_sha256": training_rows_sha256,
            "candidate_head": store.chain_head(
                store.distillation_dir(root) / "candidate-ledger.jsonl"
            )["head_sha256"],
            "profile_contract_id": _current_ox_profile_contract_id(root)
            if config.teacher_profile == OX_SINGLE_PROFILE
            else "",
        },
        root=root,
    )
    try:
        candidate = _stable_pointer_read(root, "candidate")
        reason = _verify_candidate_lineage(root, candidate, artifact, baseline)
    except (store.DistillationStoreError, DistillationError, KeyError):
        reason = "candidate_lineage_incomplete"
    if reason:
        return {"status": "held", "reason": reason}
    return {"status": "candidate", "policy_id": artifact["artifact_id"]}


def _shadow_replay_artifact_ids(
    root: Path,
    *,
    run_id: str,
    stage: str,
    cohort: str,
    candidate_id: str,
    incumbent_id: str,
    baseline_artifact_id: str,
) -> list[str]:
    """Select source IDs; the replay writer derives every row from each seal."""

    try:
        from chronovisor.recall import recall_distillation_rollout as rollout

        receipts = list(rollout._shadow_receipt_index(root).values())
        registry = rollout._validated_replay_registry_rows(root)
    except (DistillationError, store.DistillationStoreError, ValueError) as exc:
        raise DistillationError("replay source ledger is invalid") from exc
    registered = {
        str(row.get("shadow_observation_artifact_id"))
        for row in registry
        if isinstance(row, Mapping)
        and isinstance(row.get("shadow_observation_artifact_id"), str)
    }
    selected: list[str] = []
    for receipt in receipts:
        artifact_id = receipt.get("shadow_observation_artifact_id")
        if (
            receipt.get("kind") != "shadow-policy-observation"
            or not isinstance(artifact_id, str)
            or re.fullmatch(r"[0-9a-f]{64}", artifact_id) is None
            or artifact_id in registered
            or receipt.get("qualified_run_id") != run_id
            or receipt.get("stage") != stage
            or receipt.get("cohort") != cohort
            or receipt.get("policy_id") != candidate_id
            or receipt.get("incumbent_policy_id") != incumbent_id
            or receipt.get("baseline_artifact_id") != baseline_artifact_id
            or receipt.get("paired_eligible") is not True
        ):
            continue
        selected.append(artifact_id)
    return sorted(set(selected))


def _materialize_replay_observation(
    root: Path,
    *,
    rollout: Any,
    run_id: str,
    stage: str,
    cohort: str,
    candidate_id: str,
    incumbent_id: str,
    baseline_artifact_id: str,
    split_artifact_id: str,
) -> tuple[dict[str, Any], str]:
    """Publish a replay split/window through artifact-ID-only public APIs."""

    source_ids = _shadow_replay_artifact_ids(
        root,
        run_id=run_id,
        stage=stage,
        cohort=cohort,
        candidate_id=candidate_id,
        incumbent_id=incumbent_id,
        baseline_artifact_id=baseline_artifact_id,
    )
    if source_ids:
        split = rollout.write_locked_replay_input(
            root,
            shadow_observation_artifact_ids=source_ids,
            run_id=run_id,
            stage=stage,
            cohort=cohort,
            candidate_policy_id=candidate_id,
            baseline_policy_id=incumbent_id,
            baseline_artifact_id=baseline_artifact_id,
        )
        split_artifact_id = str(split.get("artifact_id") or "")
        observation = rollout.write_replay_observation(
            root,
            run_id=run_id,
            stage=stage,
            cohort=cohort,
            candidate_policy_id=candidate_id,
            baseline_policy_id=incumbent_id,
            baseline_artifact_id=baseline_artifact_id,
            split_artifact_id=split_artifact_id,
            shadow_observation_artifact_ids=source_ids,
        )
    else:
        raise DistillationError("replay observation unavailable without paired observations")
    if not isinstance(observation, Mapping):
        raise DistillationError("replay observation writer returned invalid artifact")
    artifact_id = observation.get("artifact_id")
    if not isinstance(artifact_id, str) or re.fullmatch(r"[0-9a-f]{64}", artifact_id) is None:
        raise DistillationError("replay observation identity is invalid")
    return dict(observation), split_artifact_id


def _operational_rollout_sources(
    root: Path,
    *,
    candidate_id: str,
    incumbent_id: str,
    baseline_artifact_id: str,
    cohort: str,
    qualified_run_id: str,
    stage_name: str,
) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    """Return producer-validated paired sources and their stable artifacts."""

    try:
        from chronovisor.recall import recall_distillation_rollout as rollout

        state = _read_worker_state(root)
        stage_started_at = str(state.get("stage_started_at") or "")
        context = rollout._replay_context(
            run_id=qualified_run_id,
            stage=stage_name,
            cohort=cohort,
            candidate_policy_id=candidate_id,
            baseline_policy_id=incumbent_id,
            baseline_artifact_id=baseline_artifact_id,
        )
        receipts = rollout._shadow_receipt_index(root)
    except (DistillationError, store.DistillationStoreError, ValueError):
        return []
    sources: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for artifact_id, receipt in receipts.items():
        if receipt.get("stage_started_at") != stage_started_at:
            continue
        try:
            rollout._shadow_row(root, artifact_id, context, receipt=receipt)
            artifact = rollout._shadow_observation(root, artifact_id)
        except (
            rollout.RolloutError,
            DistillationError,
            store.DistillationStoreError,
            ValueError,
        ):
            continue
        if receipt.get("paired_eligible") is True:
            sources.append((artifact_id, receipt, artifact))
    return sorted(sources, key=lambda source: source[0])


def _operational_rollout_source_ids(
    root: Path,
    *,
    candidate_id: str,
    incumbent_id: str,
    baseline_artifact_id: str,
    cohort: str,
    qualified_run_id: str,
    stage_name: str,
) -> list[str]:
    """Return only producer-validated paired source artifact IDs."""

    return [
        artifact_id
        for artifact_id, _receipt, _artifact in _operational_rollout_sources(
            root,
            candidate_id=candidate_id,
            incumbent_id=incumbent_id,
            baseline_artifact_id=baseline_artifact_id,
            cohort=cohort,
            qualified_run_id=qualified_run_id,
            stage_name=stage_name,
        )
    ]


def _operational_rollout_metrics(
    root: Path,
    candidate_id: str,
    incumbent_id: str,
    *,
    baseline_artifact_id: str,
    cohort: str,
    qualified_run_id: str | None = None,
    stage_name: str | None = None,
    source_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    missing = {
        "denominator": 0,
        "min_denominator": 500,
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

    def evidence_valid(
        value: object,
        artifact: Mapping[str, Any],
        *,
        stage_name: str,
        run_id: str,
        cohort: str,
        host: str,
    ) -> bool:
        if not isinstance(value, Mapping) or set(value) != _OPERATIONAL_EVIDENCE_KEYS:
            return False
        if any(
            not isinstance(value[key], bool)
            for key in _OPERATIONAL_EVIDENCE_KEYS
            - {
                "deadline_ms",
                "candidate_score_ms",
                "live_latency_ms",
                "producer",
                "stage",
                "run_id",
                "cohort",
                "host",
                "pair_id",
                "candidate_decision_sha256",
                "baseline_decision_sha256",
                "candidate_pool_sha256",
                "baseline_pool_sha256",
                "candidate_feature_snapshot_sha256",
                "baseline_feature_snapshot_sha256",
                "candidate_feature_bytes_sha256",
                "baseline_feature_bytes_sha256",
                "feature_snapshot_sha256",
                "feature_parity",
            }
        ):
            return False
        producer = value.get("producer")
        try:
            hashes = shadow_observation_hashes(
                artifact.get("candidate_feature_snapshot", []),
                artifact.get("baseline_feature_snapshot", []),
                artifact.get("candidate_pool_refs", []),
                artifact.get("baseline_pool_refs", []),
                selected_candidate_ids=artifact.get("selected_candidate_ids", []),
                baseline_selected_candidate_ids=artifact.get(
                    "incumbent_selected_candidate_ids", []
                ),
            )
        except (DistillationError, TypeError, ValueError):
            return False
        return (
            isinstance(value["deadline_ms"], int)
            and not isinstance(value["deadline_ms"], bool)
            and 1 <= value["deadline_ms"] <= 1_200
            and isinstance(value["candidate_score_ms"], int)
            and not isinstance(value["candidate_score_ms"], bool)
            and 0 <= value["candidate_score_ms"] <= 60_000
            and isinstance(value["live_latency_ms"], int)
            and not isinstance(value["live_latency_ms"], bool)
            and 0 <= value["live_latency_ms"] <= 60_000
            and isinstance(producer, Mapping)
            and set(producer) == {"name", "version", "synthetic_fixture"}
            and producer.get("name") == SHADOW_PRODUCER_NAME
            and producer.get("version") == SHADOW_PRODUCER_VERSION
            and producer.get("synthetic_fixture") is False
            and value.get("stage") == stage_name
            and value.get("run_id") == run_id
            and value.get("cohort") == cohort
            and value.get("host") == host
            and bool(stage_name)
            and bool(run_id)
            and bool(cohort)
            and bool(host)
            and all(
                value.get(key) == expected
                for key, expected in hashes.items()
                if key != "feature_parity"
            )
            and value.get("feature_parity") is hashes["feature_parity"]
        )

    try:
        state = _read_worker_state(root)
    except store.DistillationStoreError:
        state = {}
    stage = stage_name or str(state.get("status") or "")
    stage_started_at = str(state.get("stage_started_at") or "")
    qualified_run_id = qualified_run_id or str(state.get("stage_run_id") or "")
    metrics = {
        name: dict(missing)
        for name in (
            "coverage_abstain",
            "latency_timeout",
            "cohort_delta",
            "feature_parity",
        )
    }
    try:
        from chronovisor.recall import recall_distillation_rollout as rollout
        validated_sources = _operational_rollout_sources(
            root,
            candidate_id=candidate_id,
            incumbent_id=incumbent_id,
            baseline_artifact_id=baseline_artifact_id,
            cohort=cohort,
            qualified_run_id=qualified_run_id,
            stage_name=stage,
        )
    except (DistillationError, store.DistillationStoreError, ValueError):
        validated_sources = []
    validated_source_ids = [source[0] for source in validated_sources]
    if source_ids is not None:
        provided_source_ids = list(source_ids)
        if (
            isinstance(source_ids, (str, bytes, bytearray))
            or any(not isinstance(value, str) for value in provided_source_ids)
            or len(set(provided_source_ids)) != len(provided_source_ids)
            or provided_source_ids != validated_source_ids
        ):
            return metrics
        sources_by_id = {
            artifact_id: (receipt, artifact)
            for artifact_id, receipt, artifact in validated_sources
            if artifact_id in set(provided_source_ids)
        }
    else:
        sources_by_id = {
            artifact_id: (receipt, artifact)
            for artifact_id, receipt, artifact in validated_sources
        }
    pairs: list[dict[str, Any]] = []
    for artifact_id in validated_source_ids:
        source = sources_by_id.get(artifact_id)
        if source is None:
            continue
        receipt, artifact = source
        try:
            rollout._shadow_row(
                root,
                artifact_id,
                rollout._replay_context(
                    run_id=qualified_run_id,
                    stage=stage,
                    cohort=cohort,
                    candidate_policy_id=candidate_id,
                    baseline_policy_id=incumbent_id,
                    baseline_artifact_id=baseline_artifact_id,
                ),
                receipt=receipt,
            )
        except (rollout.RolloutError, DistillationError, ValueError):
            continue
        observation = artifact.get("runtime_observation")
        candidate_selected = artifact.get("selected_candidate_ids")
        incumbent_selected = artifact.get("incumbent_selected_candidate_ids")
        features = artifact.get("candidate_feature_snapshot")
        evidence = artifact.get("operational_evidence")
        if (
            artifact.get("stage") != stage
            or artifact.get("stage_started_at") != stage_started_at
            or artifact.get("qualified_run_id") != qualified_run_id
            or not isinstance(observation, Mapping)
            or not isinstance(candidate_selected, list)
            or not isinstance(incumbent_selected, list)
            or not isinstance(features, list)
            or receipt.get("operational_evidence_sha256")
            != canonical_json.canonical_json_sha256_strict(evidence or {})
            or not evidence_valid(
                evidence,
                artifact,
                stage_name=stage,
                run_id=qualified_run_id,
                cohort=cohort,
                host=str(artifact.get("host") or ""),
            )
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
                "candidate_abstained": bool(evidence.get("candidate_abstained")),
                "incumbent_abstained": bool(evidence.get("baseline_abstained")),
                "latency_ms": latency_ms,
                "timed_out": timed_out,
                "observed_at": artifact.get("observed_at"),
                "evidence": evidence,
            }
        )
    if pairs:
        try:
            observed = sorted(
                datetime.fromisoformat(str(row["observed_at"]).replace("Z", "+00:00"))
                for row in pairs
            )
        except (TypeError, ValueError):
            return metrics
        if (
            any(item.tzinfo is None for item in observed)
            or observed[-1] - observed[0] < timedelta(days=7)
        ):
            return metrics
        candidate_coverage = sum(row["candidate_covered"] for row in pairs) / len(pairs)
        incumbent_coverage = sum(row["incumbent_covered"] for row in pairs) / len(pairs)
        evidence = [row["evidence"] for row in pairs]
        operational_ok = all(
            row["resource_ok"]
            and row["integrity_ok"]
            and not row["negative_veto"]
            for row in evidence
        )
        quality_ok = sum(row["candidate_quality"] for row in evidence) >= sum(
            row["baseline_quality"] for row in evidence
        )
        anchor_ok = sum(row["candidate_anchor_retained"] for row in evidence) >= sum(
            row["baseline_anchor_retained"] for row in evidence
        )
        abstain_ok = (
            sum(row["candidate_abstained"] for row in pairs)
            - sum(row["incumbent_abstained"] for row in pairs)
        ) / len(pairs) <= 0.02
        coverage_ok = float(
            operational_ok
            and quality_ok
            and anchor_ok
            and abstain_ok
            and candidate_coverage >= incumbent_coverage * 0.90
        )
        metrics["coverage_abstain"] = gate(len(pairs), coverage_ok, 1.0)
        metrics["cohort_delta"] = gate(len(pairs), coverage_ok, 1.0)
        ordered = sorted(row["latency_ms"] for row in pairs)
        p95 = ordered[math.ceil(len(ordered) * 0.95) - 1]
        timeouts = sum(row["timed_out"] for row in pairs)
        p50 = ordered[math.ceil(len(ordered) * 0.50) - 1]
        latency_ok = float(
            p95 <= 180
            and p50 <= 400
            and p95 < 900
            and _wilson_upper(timeouts, len(pairs)) <= 0.03
            and all(row["deadline_ms"] <= 1_200 for row in evidence)
        )
        metrics["latency_timeout"] = gate(len(pairs), latency_ok, 1.0)
        metrics["feature_parity"] = gate(
            len(pairs), float(operational_ok and all(not row["negative_veto"] for row in evidence)), 1.0
        )
    return metrics


def _authenticated_negative_vetoes(root: Path, policy_id: str) -> int:
    rows = _read_chain(store.distillation_dir(root) / "negative-veto-receipts.jsonl")
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
        candidate_id = str(_stable_pointer_read(root, "candidate")["policy_id"])
        incumbent_id = str(_stable_pointer_read(root, "active")["policy_id"])
        policy = _load_policy(candidate_id, root)
    except (KeyError, store.DistillationStoreError, DistillationError):
        return {"status": "held", "reason": "rollout_identity_unavailable"}
    if candidate_id == incumbent_id:
        return {"status": "held", "reason": "rollout_policy_identity_collision"}
    vetoes = _authenticated_negative_vetoes(root, candidate_id)
    if vetoes:
        veto_run_id = canonical_json.canonical_json_sha256_strict(
            {
                "kind": "authenticated-negative-veto-run-v1",
                "policy_id": candidate_id,
                "veto_head": store.chain_head(
                    store.distillation_dir(root) / "negative-veto-receipts.jsonl"
                )["head_sha256"],
            }
        )
        result = rollout.rollback_to_lkg(
            root, veto_run_id, "authenticated_negative_veto"
        )
        return {"status": "rolled_back", "negative_vetoes": vetoes, **result}
    run_id = str(state.get("stage_run_id") or "")
    if re.fullmatch(r"[0-9a-f]{64}", run_id) is None:
        return {"status": "held", "reason": "rollout_run_identity_unavailable"}
    lineage = (
        policy.get("lineage") if isinstance(policy.get("lineage"), Mapping) else {}
    )
    cohort = str(lineage.get("model_cohort_sha256") or "")
    if not cohort:
        return {"status": "held", "reason": "rollout_cohort_unavailable"}
    baseline_artifact_id = str(baseline.get("artifact_id") or "")
    if re.fullmatch(r"[0-9a-f]{64}", baseline_artifact_id) is None:
        return {"status": "held", "reason": "rollout_baseline_unavailable"}
    if lineage.get("baseline_artifact_id") != baseline_artifact_id:
        return {"status": "held", "reason": "rollout_baseline_mismatch"}
    operational_source_ids = _operational_rollout_source_ids(
        root,
        candidate_id=candidate_id,
        incumbent_id=incumbent_id,
        baseline_artifact_id=baseline_artifact_id,
        cohort=cohort,
        qualified_run_id=run_id,
        stage_name=str(state.get("status") or ""),
    )
    measured_metrics = _operational_rollout_metrics(
        root,
        candidate_id,
        incumbent_id,
        baseline_artifact_id=baseline_artifact_id,
        cohort=cohort,
        qualified_run_id=run_id,
        stage_name=str(state.get("status") or ""),
        source_ids=operational_source_ids,
    )
    split_hint = str(lineage.get("locked_replay_id") or "")
    try:
        replay_observation, split_sha256 = _materialize_replay_observation(
            root,
            rollout=rollout,
            run_id=run_id,
            stage=str(state.get("status") or ""),
            cohort=cohort,
            candidate_id=candidate_id,
            incumbent_id=incumbent_id,
            baseline_artifact_id=baseline_artifact_id,
            split_artifact_id=split_hint,
        )
    except (DistillationError, rollout.RolloutError, store.DistillationStoreError):
        return {"status": "held", "reason": "replay_observation_unavailable"}
    pair_count = replay_observation.get("pair_count")
    if isinstance(pair_count, bool) or not isinstance(pair_count, int) or pair_count < 0:
        return {"status": "held", "reason": "replay_observation_invalid"}
    replay_gate = {
        "denominator": pair_count,
        "min_denominator": 500,
        "min_days": 7,
        "ci_lower": 1.0 if pair_count >= 500 else 0.0,
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
    raw_watermark = str(baseline.get("raw_watermark") or "")
    if re.fullmatch(r"[0-9a-f]{64}", raw_watermark) is None:
        return {"status": "held", "reason": "rollout_watermark_unavailable"}
    evaluation_payload = {
        "kind": "automatic-closed-metrics",
        "run_id": run_id,
        "candidate_policy_id": candidate_id,
        "baseline_artifact_id": baseline_artifact_id,
        "raw_watermark": raw_watermark,
        "baseline_policy_id": incumbent_id,
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
        "operational_metrics_sha256": canonical_json.canonical_json_sha256_strict(
            measured_metrics
        ),
        "operational_source_sha256": canonical_json.canonical_json_sha256_strict(
            operational_source_ids
        ),
        "replay_observation_artifact_id": str(
            replay_observation.get("artifact_id") or ""
        ),
    }
    if re.fullmatch(
        r"[0-9a-f]{64}", evaluation_payload["replay_observation_artifact_id"]
    ) is None:
        return {"status": "held", "reason": "replay_observation_invalid"}
    evaluation_id, _, _ = store.write_immutable(
        store.distillation_dir(root) / "evaluations",
        evaluation_payload,
        schema=rollout.EVALUATION_SCHEMA,
    )
    try:
        result = rollout.evaluate_and_advance(
            root,
            {"run_id": run_id, "evaluation_artifact_id": evaluation_id},
        )
    except rollout.RolloutError:
        return {"status": "held", "reason": "automatic_evaluation_invalid"}
    return {
        **result,
        "replay_observation_artifact_id": evaluation_payload[
            "replay_observation_artifact_id"
        ],
        "evaluation_artifact_id": evaluation_id,
    }


def publish_policy(
    policy: Mapping[str, Any],
    *,
    lineage: Mapping[str, Any],
    root: Path | None = None,
) -> dict[str, Any]:
    root = root or CHRONOVISOR_ROOT
    training_snapshot_id = str(lineage.get("training_snapshot_id") or "")
    if not training_snapshot_id:
        raise DistillationError("candidate lineage is incomplete")
    required = (
        "locked_replay_id",
        "baseline_artifact_id",
        "model_cohort_sha256",
        "raw_watermark",
        "label_chain_head",
        "split_plan_id",
        "offline_gate_sha256",
        "training_rows_sha256",
        "candidate_head",
    )
    if (
        any(
            re.fullmatch(r"[0-9a-f]{64}", str(lineage.get(name) or "")) is None
            for name in ("training_snapshot_id", *required)
        )
        or lineage.get("feature_revision") != TEXT_FEATURE_REVISION
    ):
        raise DistillationError("candidate lineage is incomplete")
    try:
        snapshot = store.read_sealed(
            store.distillation_dir(root)
            / "training-snapshots"
            / f"{training_snapshot_id}.json",
            schema="chronovisor.recall-distill-training.v1",
        )
        baseline = store.read_sealed(
            store.distillation_dir(root)
            / "baselines"
            / f"{lineage['baseline_artifact_id']}.json",
            schema=BASELINE_SCHEMA,
        )
        replay = store.read_sealed(
            store.distillation_dir(root)
            / "locked-replays"
            / f"{lineage['locked_replay_id']}.json",
            schema="chronovisor.recall-distill-locked-replay.v1",
        )
        split_plan = _read_split_plan_artifact(root, str(lineage["split_plan_id"]))
    except (DistillationError, store.DistillationStoreError) as exc:
        raise DistillationError("candidate sealed lineage is unavailable") from exc
    rows = snapshot.get("rows")
    if (
        snapshot.get("artifact_id") != training_snapshot_id
        or snapshot.get("label_chain_head") != lineage["label_chain_head"]
        or store.chain_head(store.distillation_dir(root) / "label-ledger.jsonl")[
            "head_sha256"
        ]
        != lineage["label_chain_head"]
        or store.chain_head(store.distillation_dir(root) / "candidate-ledger.jsonl")[
            "head_sha256"
        ]
        != lineage["candidate_head"]
        or snapshot.get("feature_revision") != TEXT_FEATURE_REVISION
        or not isinstance(rows, list)
        or not rows
        or any(
            not isinstance(row, Mapping)
            or not _materialized_row_integrity(row, root=root, split_plan=split_plan)
            for row in rows
        )
        or {str(row.get("split") or "") for row in rows if isinstance(row, Mapping)}
        < {"train", "validation", "test"}
        or baseline.get("raw_watermark") != lineage["raw_watermark"]
        or baseline.get("hard_floor", {}).get("p5_allowed") is not True
        or baseline.get("offline_training_gate", {}).get("passed") is not True
        or canonical_json.canonical_json_sha256_strict(
            baseline.get("offline_training_gate")
        )
        != lineage["offline_gate_sha256"]
        or replay.get("training_snapshot_id") != training_snapshot_id
        or replay.get("baseline_artifact_id") != lineage["baseline_artifact_id"]
        or replay.get("model_cohort_sha256") != lineage["model_cohort_sha256"]
        or replay.get("split_revision") != "grouped-rolling-v1"
        or replay.get("offline_gate_sha256") != lineage["offline_gate_sha256"]
        or split_plan.get("feature_revision") != TEXT_FEATURE_REVISION
        or split_plan.get("model_cohort_sha256") != lineage["model_cohort_sha256"]
        or replay.get("training_rows_sha256") != lineage["training_rows_sha256"]
        or replay.get("candidate_head") != lineage["candidate_head"]
        or replay.get("profile_contract_id") != lineage.get("profile_contract_id", "")
        or canonical_json.canonical_json_sha256_strict(replay.get("training_rows"))
        != lineage["training_rows_sha256"]
        or replay.get("policy_sha256") != _policy_payload_digest(policy)
        or not isinstance(replay.get("training_rows"), list)
        or not replay["training_rows"]
        or any(
            canonical_json.canonical_json_sha256_strict(row)
            not in {
                canonical_json.canonical_json_sha256_strict(snapshot_row)
                for snapshot_row in rows
                if isinstance(snapshot_row, Mapping)
            }
            for row in replay["training_rows"]
            if isinstance(row, Mapping)
        )
        or any(not isinstance(row, Mapping) for row in replay["training_rows"])
        or not any(row.get("split") == "train" for row in replay["training_rows"])
        or not any(row.get("split") == "validation" for row in replay["training_rows"])
        or not any(
            row.get("split") == "test" and row.get("locked_test_read_only") is True
            for row in replay["training_rows"]
        )
        or train_tiny_policy(replay["training_rows"]) != dict(policy)
        or baseline.get("frozen_contract", {}).get("hard_floors")
        != _canonical_hard_floors()
    ):
        raise DistillationError("candidate training lineage is incomplete")
    with store._locked(store.distillation_dir(root) / "rollout.lock"):
        policy_id, _, artifact = store.write_immutable(
            store.distillation_dir(root) / "policies",
            {"kind": "tiny-logistic-policy", **policy, "lineage": dict(lineage)},
            schema=POLICY_SCHEMA,
        )
        reason = _verify_candidate_lineage(
            root, {"policy_id": policy_id}, artifact, baseline
        )
        if reason:
            raise DistillationError("candidate training lineage is incomplete")
        try:
            active = _stable_pointer_read(root, "active")
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
            max(_source_epoch(rows[index]) for index in indexes),
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


def _read_split_plan_artifact(root: Path, plan_id: str) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{64}", plan_id) is None:
        raise DistillationError("split plan id is invalid")
    artifact = store.read_sealed(
        store.distillation_dir(root) / "split-plans" / f"{plan_id}.json",
        schema=SPLIT_PLAN_SCHEMA,
    )
    assignments = artifact.get("assignments")
    if (
        artifact.get("artifact_id") != plan_id
        or not isinstance(assignments, Mapping)
        or any(
            value not in {"train", "validation", "test", "embargo"}
            for value in assignments.values()
        )
    ):
        raise DistillationError("split plan artifact is invalid")
    return artifact


def _read_split_plan(root: Path) -> dict[str, Any]:
    pointer = store.read_sealed(
        store.distillation_dir(root) / "split-plan.json",
        schema=store.DISTILLATION_SCHEMA,
    )
    plan_id = str(pointer.get("split_plan_id") or "")
    return _read_split_plan_artifact(root, plan_id)


def _scheduling_split_plan_id(plan: Mapping[str, Any]) -> str:
    """Return the frozen work-plan receipt, never a growth-only pointer."""

    current = str(plan.get("artifact_id") or "")
    frozen = str(plan.get("scheduling_split_plan_id") or current)
    if re.fullmatch(r"[0-9a-f]{64}", frozen) is None:
        raise DistillationError("split plan scheduling receipt is invalid")
    return frozen


def _scheduling_age_bands(root: Path, plan: Mapping[str, Any]) -> Mapping[str, str]:
    """Read the age receipt bound to the immutable work-plan identity."""

    frozen = _read_split_plan_artifact(root, _scheduling_split_plan_id(plan))
    bands = frozen.get("age_bands")
    # Legacy sealed plans predate age receipts.  Their callers use the
    # deterministic UTC fallback; newly created plans must carry this field.
    if bands is None:
        return {}
    if not isinstance(bands, Mapping) or any(
        not isinstance(rally_id, str)
        or band not in {"old-history", "recent", "locked-test"}
        for rally_id, band in bands.items()
    ):
        raise DistillationError("split plan age receipt is invalid")
    return {str(rally_id): str(band) for rally_id, band in bands.items()}


def _ensure_split_plan(
    root: Path,
    rallies: Sequence[Mapping[str, Any]],
    *,
    raw_watermark: str,
    model_cohort_sha256: str,
) -> dict[str, Any]:
    assignments = grouped_rolling_split(rallies)
    rally_ids = {str(rally["rally_id"]) for rally in rallies}
    try:
        prior = _read_split_plan(root)
    except (DistillationError, store.DistillationStoreError, KeyError):
        prior = {}
    if (
        prior.get("feature_revision") == TEXT_FEATURE_REVISION
        and prior.get("split_revision") == "grouped-rolling-v1"
        and prior.get("model_cohort_sha256") == model_cohort_sha256
        and isinstance(prior.get("assignments"), Mapping)
    ):
        prior_assignments = {
            str(rally_id): str(value)
            for rally_id, value in prior["assignments"].items()
        }
        if not set(prior_assignments).issubset(rally_ids):
            raise DistillationError("split plan rally set regressed")
        # A growth receipt preserves every existing assignment and embargoes
        # the new source.  Its ``scheduling_split_plan_id`` remains the prior
        # immutable plan, so Workset payloads for already-known work do not
        # acquire mutable provenance merely because the source watermark grew.
        assignments = {
            **prior_assignments,
            **{
                rally_id: "embargo" for rally_id in rally_ids - prior_assignments.keys()
            },
        }
        scheduling_split_plan_id = str(
            prior.get("scheduling_split_plan_id") or prior.get("artifact_id") or ""
        )
    else:
        scheduling_split_plan_id = ""
    plan_id, _, artifact = store.write_immutable(
        store.distillation_dir(root) / "split-plans",
        {
            "kind": "fixed-chronological-group-split",
            "raw_watermark": raw_watermark,
            "feature_revision": TEXT_FEATURE_REVISION,
            "model_cohort_sha256": model_cohort_sha256,
            "split_revision": "grouped-rolling-v1",
            "assignments": assignments,
            "age_boundary_utc": _source_age_boundary(rallies),
            "age_bands": _source_age_bands(rallies, assignments=assignments),
            **(
                {"scheduling_split_plan_id": scheduling_split_plan_id}
                if scheduling_split_plan_id
                else {}
            ),
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
    for row in _read_chain(store.distillation_dir(root) / "label-ledger.jsonl"):
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
        receipt_rows = _read_chain(
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
        from chronovisor.recall import recall_distillation_rollout as rollout

        shadow_receipts = list(rollout._shadow_receipt_index(root).values())
    except (DistillationError, store.DistillationStoreError, ValueError):
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
        labels = _read_chain(store.distillation_dir(root) / "label-ledger.jsonl")
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
    _training_snapshot: Mapping[str, Any] | None = None,
    _profile_contract_id: str = "",
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
    training_snapshot = (
        _training_snapshot
        if _training_snapshot is not None
        else materialize_training_rows(root)
    )
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
    teacher_contract = {
        "routes": list(TEACHER_ROLES),
        "local_only": True,
        "max_input_bytes": config.max_input_bytes,
        "max_candidates": config.max_candidates,
        "max_load_skew": 0.10,
        "order_bias_max": 0.05,
    }
    if config.teacher_profile == OX_SINGLE_PROFILE and config.ox_enabled:
        if re.fullmatch(r"[0-9a-f]{64}", _profile_contract_id):
            try:
                profile_contract = _read_ox_profile_contract(root, _profile_contract_id)
                if not profile_contract:
                    raise DistillationError("OX profile contract is invalid")
            except store.DistillationStoreError as exc:
                raise DistillationError("OX profile contract is unavailable") from exc
        else:
            profile_contract = _ensure_ox_profile_contract(root, config)
        teacher_contract = {
            "profile": OX_SINGLE_PROFILE,
            "cohort": OX_SINGLE_COHORT,
            "profile_contract_id": profile_contract["artifact_id"],
            "routes": [OX_TEACHER_ROLE],
            "local_only": False,
            "free_only": True,
            "max_input_bytes": config.max_input_bytes,
            "max_candidates": config.max_candidates,
        }
    elif config.teacher_profile == OX_SINGLE_PROFILE:
        teacher_contract = {
            "profile": OX_SINGLE_PROFILE,
            "cohort": OX_SINGLE_COHORT,
            "routes": [OX_TEACHER_ROLE],
            "local_only": False,
            "free_only": True,
            "enabled": False,
        }
    label_projection = store.label_health_projection(
        store.distillation_dir(root) / store.LABEL_LEDGER_FILE,
        repair=True,
    )
    label_counts = label_projection["counts"]
    payload = {
        "kind": "privacy-safe-baseline",
        "raw_watermark": committed_raw_watermark(raw_dir),
        "label_chain_head": label_projection["label_chain_head"],
        "label_records": label_projection["label_records"],
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
            "teacher_only_labels": label_counts["teacher_only"],
            "verified_truth_labels": label_counts["verified_truth"],
            "probe_pairs": offline_gate["probe"]["pairs"],
            "counterfactual_pairs": offline_gate["counterfactual_pairs"],
            "probe_label_rows": label_counts["probe_not_truth"],
            "locked_test_probe_pairs": offline_gate["probe"]["pairs"],
            "locked_test_counterfactual_pairs": offline_gate["counterfactual_pairs"],
        },
        "offline_training_gate": offline_gate,
        "hard_floor": {
            "p5_allowed": not reasons,
            "reasons": reasons,
        },
        "metrics": metrics,
        "frozen_contract": {
            "hard_floors": _canonical_hard_floors(),
            "rally_revision": "rally-v1",
            "assignment_revision": ASSIGNMENT_REVISION,
            "probe_revision": PROBE_REVISION,
            "probe_rate": 0.15,
            "feature_revision": TEXT_FEATURE_REVISION,
            "feature_whitelist": list(FAST_FEATURE_KEYS),
            "closed_predicates": sorted(CLOSED_PREDICATES),
            "teacher": teacher_contract,
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
    context: list[str] = []
    payload = {
        "schema": "chronovisor.recall-distill-teacher-input.v1",
        "rally_id": rally["rally_id"],
        "candidate_id": candidate["candidate_id"],
        "query": query,
        "context": context,
        "candidate": candidate_text,
    }
    if (
        not query
        or not candidate_text
        or len(canonical_json.canonical_json_bytes_strict(payload)) > max_input_bytes
    ):
        return None
    for ref in reversed(rally.get("context_refs", [])):
        context.insert(0, texts.get(str(ref["semantic_sha256"]), ""))
        if len(canonical_json.canonical_json_bytes_strict(payload)) > max_input_bytes:
            context.pop(0)
            break
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
    workset_status: Mapping[str, int] | None = None
    profile_stopped: bool = False
    profile_contract_id: str = ""
    ramp_cap: int | None = None
    ramp_valid_receipts: int | None = None
    ramp_provider_attempts: int | None = None
    last_durable_progress: Mapping[str, Any] | None = None
    source_binding: Mapping[str, str] | None = None


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


def _ox_ramp_state(state: Mapping[str, Any], max_inflight: int) -> tuple[int, int, int]:
    if state.get("ox_ramp_request_revision") != OX_RAMP_REQUEST_REVISION:
        return 1, 0, 0
    cap = state.get("ox_ramp_cap", 1)
    receipts = state.get("ox_ramp_valid_receipts", 0)
    attempts = state.get("ox_ramp_provider_attempts")
    allowed_caps = tuple(sorted({min(value, max_inflight) for value in (1, 2, 5, 10)}))
    if attempts is None and receipts == 0:
        attempts = 0
    if (
        isinstance(cap, bool)
        or not isinstance(cap, int)
        or cap not in allowed_caps
        or isinstance(receipts, bool)
        or not isinstance(receipts, int)
        or receipts < 0
        or isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or attempts < receipts
    ):
        # A legacy receipt count has no trustworthy denominator.  Reset rather
        # than assuming every recorded receipt was a successful first attempt.
        return 1, 0, 0
    return cap, receipts, attempts


def _next_ox_ramp_cap(current_cap: int, max_inflight: int) -> int:
    for cap in sorted({min(value, max_inflight) for value in (1, 2, 5, 10)}):
        if cap > current_cap:
            return cap
    return current_cap


def _previous_ox_ramp_cap(current_cap: int, max_inflight: int) -> int:
    return max(
        (
            cap
            for cap in {min(value, max_inflight) for value in (1, 2, 5, 10)}
            if cap <= current_cap // 2
        ),
        default=1,
    )


def _advance_ox_ramp(
    *,
    cap: int,
    valid_receipts: int,
    provider_attempts: int,
    valid_results: int,
    actual_attempts: int,
    rate_limited: bool,
    stopped: bool,
    max_inflight: int,
) -> tuple[int, int, int]:
    """Apply the OX-only quality gate after deep response validation."""
    final_cap = max(min(value, max_inflight) for value in (1, 2, 5, 10))
    if (
        cap == final_cap
        and valid_receipts >= OX_RAMP_RECEIPTS_PER_CAP
        and valid_receipts * 100 >= provider_attempts * 95
    ):
        return cap, valid_receipts, provider_attempts
    if rate_limited:
        return _previous_ox_ramp_cap(cap, max_inflight), 0, 0
    if stopped:
        return cap, valid_receipts, provider_attempts + actual_attempts
    receipts = valid_receipts + valid_results
    attempts = provider_attempts + actual_attempts
    if receipts < OX_RAMP_RECEIPTS_PER_CAP or receipts * 100 < attempts * 95:
        return cap, receipts, attempts
    next_cap = _next_ox_ramp_cap(cap, max_inflight)
    if next_cap == cap:
        return cap, receipts, attempts
    return next_cap, 0, 0


def _ox_prepare_tasks(
    *,
    config: DistillationConfig,
    snapshots: Mapping[str, Mapping[str, Any]],
    rally_by_id: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, Any],
    split_plan_id: str,
    profile_contract_id: str,
    candidate_indexed: bool,
    candidate_state: Mapping[str, Any],
    age_bands: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    tasks: dict[str, dict[str, Any]] = {}
    work_items: list[dict[str, Any]] = []
    age_bands = dict(
        age_bands
        or _source_age_bands(list(rally_by_id.values()), assignments=assignments)
    )

    def add_task(
        rally_id: str,
        snapshot: Mapping[str, Any],
        rally: Mapping[str, Any],
        candidate: Mapping[str, Any],
        assignment: Mapping[str, Any],
        *,
        priority: int = 0,
        register_item: bool = True,
        temporal_override: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            return None
        age_band = age_bands.get(rally_id, "old-history")
        source = {
            "rally_id": rally_id,
            "candidate_id": candidate_id,
            "snapshot_sha256": snapshot.get("snapshot_sha256", ""),
            "query_sha256": rally.get("query_sha256", ""),
            "candidate_text_sha256": candidate.get("text_sha256", ""),
            "context_sha256": [
                ref.get("semantic_sha256", "")
                for ref in rally.get("context_refs", [])
                if isinstance(ref, Mapping)
            ],
            **({"assignment": dict(assignment)} if assignment.get("probe") else {}),
        }
        payload_digest = canonical_json.canonical_json_sha256_strict(source)
        work_id = canonical_json.canonical_json_sha256_strict(
            {
                "kind": "ox-teacher-label-v1",
                "profile": OX_SINGLE_PROFILE,
                "cohort": OX_SINGLE_COHORT,
                "route": "opencode-go/ox-alpha-free",
                "profile_contract_id": profile_contract_id,
                "payload_digest": payload_digest,
            }
        )
        temporal = {
            "as_of": str(rally.get("as_of") or ""),
            "group_id": str(rally.get("session_cluster_id") or ""),
            "split": str(assignments.get(rally_id) or "embargo"),
            "split_plan_id": split_plan_id,
        }
        if temporal_override is not None:
            temporal = dict(temporal_override)
        task = {
            "rally": rally,
            "candidate": candidate,
            "assignment": dict(assignment),
            "temporal": temporal,
            "probe_batch_id": str(assignment.get("probe_batch_id") or ""),
            "work_id": work_id,
            "payload_digest": payload_digest,
            "payload_source": source,
        }
        tasks[work_id] = task
        if not register_item:
            return task
        work_items.append(
            {
                "work_id": work_id,
                "kind": "ox",
                "payload_ref": f"candidate-snapshot:{rally_id}:{candidate_id}",
                "payload_digest": payload_digest,
                "priority": max(priority, _age_band_priority(age_band)),
                "temporal_split": temporal,
                "provenance": {
                    "profile": OX_SINGLE_PROFILE,
                    "cohort": OX_SINGLE_COHORT,
                    "route": "opencode-go/ox-alpha-free",
                    "teacher_role": OX_TEACHER_ROLE,
                    "profile_contract_id": profile_contract_id,
                    "probe": assignment.get("probe") is True,
                    **{
                        key: assignment[key]
                        for key in (
                            "probe_revision",
                            "repeat_pair_id",
                            "fixed_repeat",
                            "order_swap",
                            "blind_order",
                            "probe_batch_id",
                            "order_variant",
                            "candidate_position",
                        )
                        if key in assignment
                    },
                },
            }
        )
        return task

    eligible_probe_rallies: list[
        tuple[str, Mapping[str, Any], Mapping[str, Any], list[Mapping[str, Any]]]
    ] = []
    for rally_id, snapshot in snapshots.items():
        rally = rally_by_id.get(rally_id)
        if rally is None or assignments.get(rally_id) != "test":
            continue
        candidates = [
            candidate
            for candidate in snapshot.get("candidates", [])
            if isinstance(candidate, Mapping)
            and str(candidate.get("candidate_id") or "")
        ]
        if len(candidates) >= 2:
            eligible_probe_rallies.append((rally_id, snapshot, rally, candidates))
    probe_rallies = sorted(
        eligible_probe_rallies,
        key=lambda item: (_source_epoch(item[2]), item[0]),
    )[: config.hard_floor_probe_pairs]
    for rally_id, snapshot in sorted(
        snapshots.items(),
        key=lambda item: (_source_epoch(rally_by_id.get(item[0], {})), item[0]),
    ):
        rally = rally_by_id.get(rally_id)
        if rally is None or (
            split_plan_id
            and assignments.get(rally_id) not in {"train", "validation", "test"}
        ):
            continue
        candidates = [
            candidate
            for candidate in snapshot.get("candidates", [])
            if isinstance(candidate, Mapping)
        ]
        selected = candidates[:3]
        if candidates[3:]:
            selected.append(candidates[-1])
        for candidate in selected:
            add_task(
                rally_id,
                snapshot,
                rally,
                candidate,
                {
                    "revision": "single-teacher-v1",
                    "owner": OX_TEACHER_ROLE,
                    "probe": False,
                    "routes": [OX_TEACHER_ROLE],
                },
            )
    for rally_id, snapshot, rally, candidates in probe_rallies:
        first, second = sorted(
            candidates, key=lambda candidate: str(candidate["candidate_id"])
        )[:2]
        ordered_variants = (("a_first", (first, second)), ("b_first", (second, first)))
        for order_variant, (blind_order, ordered_candidates) in enumerate(
            ordered_variants, start=1
        ):
            probe_batch_id = canonical_json.canonical_json_sha256_strict(
                {
                    "revision": OX_PROBE_REVISION,
                    "split_plan_id": split_plan_id,
                    "rally_id": rally_id,
                    "blind_order": blind_order,
                }
            )
            for position, candidate in enumerate(ordered_candidates):
                candidate_id = str(candidate["candidate_id"])
                repeat_pair_id = canonical_json.canonical_json_sha256_strict(
                    {
                        "revision": OX_PROBE_REVISION,
                        "split_plan_id": split_plan_id,
                        "rally_id": rally_id,
                        "candidate_id": candidate_id,
                    }
                )
                add_task(
                    rally_id,
                    snapshot,
                    rally,
                    candidate,
                    {
                        "revision": "single-teacher-v1",
                        "owner": OX_TEACHER_ROLE,
                        "probe": True,
                        "routes": [OX_TEACHER_ROLE],
                        "probe_revision": OX_PROBE_REVISION,
                        "repeat_pair_id": repeat_pair_id,
                        "fixed_repeat": True,
                        "order_swap": True,
                        "blind_order": blind_order,
                        "probe_batch_id": probe_batch_id,
                        "order_variant": order_variant,
                        "candidate_position": position,
                    },
                    priority=100,
                )
    watermark: Any = canonical_json.canonical_json_sha256_strict(
        {"work_ids": sorted(tasks)}
    )
    if candidate_indexed:
        watermark = {
            "candidate_records": int(candidate_state["record_count"]),
            "candidate_head": str(candidate_state["head_sha256"]),
            "split_plan_id": split_plan_id,
            "probe_revision": OX_PROBE_REVISION,
        }
    return {
        "tasks": tasks,
        "work_items": work_items,
        "watermark": watermark,
        "add_task": add_task,
    }


def _ox_restore_claims(
    *,
    root: Path,
    candidate_path: Path,
    catalog: Any,
    candidate_indexed: bool,
    claims: list[Any],
    tasks: dict[str, dict[str, Any]],
    add_task: Callable[..., dict[str, Any] | None],
    workset: Any,
    rally_by_id: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, Any],
    split_plan_id: str,
    profile_contract_id: str,
) -> list[Any]:
    def assignment_from_claim(claim: Any) -> Mapping[str, Any] | None:
        provenance = claim.provenance
        if not isinstance(provenance, Mapping):
            return None
        if (
            provenance.get("profile") != OX_SINGLE_PROFILE
            or provenance.get("cohort") != OX_SINGLE_COHORT
            or provenance.get("route") != "opencode-go/ox-alpha-free"
            or provenance.get("teacher_role") != OX_TEACHER_ROLE
            or provenance.get("profile_contract_id") != profile_contract_id
        ):
            return None
        if provenance.get("probe") is not True:
            return {
                "revision": "single-teacher-v1",
                "owner": OX_TEACHER_ROLE,
                "probe": False,
                "routes": [OX_TEACHER_ROLE],
            }
        required = {
            "probe_revision",
            "repeat_pair_id",
            "fixed_repeat",
            "order_swap",
            "blind_order",
            "probe_batch_id",
            "order_variant",
            "candidate_position",
        }
        if (
            not required.issubset(provenance)
            or provenance.get("probe_revision") != OX_PROBE_REVISION
            or provenance.get("fixed_repeat") is not True
            or provenance.get("order_swap") is not True
            or provenance.get("blind_order") not in {"a_first", "b_first"}
            or provenance.get("order_variant") not in {1, 2}
            or provenance.get("candidate_position") not in {0, 1}
            or any(
                not isinstance(provenance.get(key), str) or not provenance[key]
                for key in ("repeat_pair_id", "probe_batch_id")
            )
        ):
            return None
        return {
            "revision": "single-teacher-v1",
            "owner": OX_TEACHER_ROLE,
            "probe": True,
            "routes": [OX_TEACHER_ROLE],
            **{key: provenance[key] for key in required},
        }

    def claim_reference(claim: Any) -> tuple[str, str] | None:
        parts = str(claim.payload_ref).split(":")
        if len(parts) != 3 or parts[0] != "candidate-snapshot":
            return None
        if not parts[1] or not parts[2]:
            return None
        return parts[1], parts[2]

    if not candidate_indexed or not claims:
        return claims
    missing = [claim for claim in claims if claim.work_id not in tasks]
    invalid_claims = [claim for claim in missing if claim_reference(claim) is None]
    requested_rallies = {
        reference[0]
        for claim in missing
        if (reference := claim_reference(claim)) is not None
    }
    resolved_snapshots: Mapping[str, Mapping[str, Any]] = {}
    if requested_rallies:
        try:
            resolved_snapshots = catalog.read_candidate_snapshots(
                root, candidate_path, requested_rallies
            )
        except catalog.CatalogError as exc:
            raise DistillationError(
                "OX claim candidate snapshot is unavailable"
            ) from exc
    for claim in missing:
        reference = claim_reference(claim)
        if reference is None:
            continue
        rally_id, candidate_id = reference
        rally = rally_by_id.get(rally_id)
        claimed_snapshot = resolved_snapshots.get(rally_id)
        restored_assignment = assignment_from_claim(claim)
        temporal = claim.temporal_split
        restored_candidate: Mapping[str, Any] | None = None
        if isinstance(claimed_snapshot, Mapping):
            candidate_rows = claimed_snapshot.get("candidates", [])
            if isinstance(candidate_rows, list):
                restored_candidate = next(
                    (
                        row
                        for row in candidate_rows
                        if isinstance(row, Mapping)
                        and str(row.get("candidate_id") or "") == candidate_id
                    ),
                    None,
                )
        if (
            rally is None
            or not isinstance(claimed_snapshot, Mapping)
            or not isinstance(restored_candidate, Mapping)
            or restored_assignment is None
            or not isinstance(temporal, Mapping)
            or set(temporal) != {"as_of", "group_id", "split", "split_plan_id"}
            or temporal.get("as_of") != rally.get("as_of")
            or temporal.get("group_id") != rally.get("session_cluster_id")
            or temporal.get("split") != str(assignments.get(rally_id) or "embargo")
            or temporal.get("split_plan_id") != split_plan_id
            or (
                split_plan_id
                and temporal.get("split") not in {"train", "validation", "test"}
            )
        ):
            invalid_claims.append(claim)
            continue
        restored = add_task(
            rally_id,
            claimed_snapshot,
            rally,
            restored_candidate,
            restored_assignment,
            register_item=False,
            temporal_override=temporal,
        )
        if (
            restored is None
            or restored["work_id"] != claim.work_id
            or restored["payload_digest"] != claim.payload_digest
        ):
            invalid_claims.append(claim)
    if invalid_claims:
        invalid_ids = {claim.work_id for claim in invalid_claims}
        workset.commit(
            [claim for claim in claims if claim.work_id in invalid_ids],
            [
                {
                    "status": "quarantined" if claim.attempt >= 3 else "retry",
                    "error_class": "remote_payload_rejected",
                }
                for claim in claims
                if claim.work_id in invalid_ids
            ],
        )
        claims = [claim for claim in claims if claim.work_id not in invalid_ids]
    return claims


def _ox_resolve_claim_payloads(
    *,
    claims: Sequence[Any],
    tasks: Mapping[str, dict[str, Any]],
    texts: Mapping[str, str],
    config: DistillationConfig,
    workset: Any,
) -> tuple[list[Any], list[tuple[Any, dict[str, str]]], list[Any], bool]:
    # BEGIN payload resolution block
    payload_failures: list[tuple[Any, dict[str, str]]] = []
    local_payload_rejects: list[Any] = []
    claim_integrity_failure = False
    resolved_claims: list[Any] = []
    prefetch = getattr(texts, "prefetch", None)
    if callable(prefetch):
        hashes: list[str] = []
        for claim in claims:
            task = tasks.get(claim.work_id)
            if task is None:
                continue
            hashes.extend(
                (
                    str(task["rally"].get("query_sha256") or ""),
                    str(task["candidate"].get("text_sha256") or ""),
                    *(
                        str(ref.get("semantic_sha256") or "")
                        for ref in task["rally"].get("context_refs", [])
                        if isinstance(ref, Mapping)
                    ),
                )
            )
        prefetch(hashes)
    for claim in claims:
        task = tasks.get(claim.work_id)
        if task is None:
            claim_integrity_failure = True
            payload_failures.append(
                (
                    claim,
                    {
                        "status": "quarantined" if claim.attempt >= 3 else "retry",
                        "error_class": "remote_payload_rejected",
                    },
                )
            )
            continue
        payload = _teacher_payload(
            task["rally"],
            task["candidate"],
            texts,
            max_input_bytes=config.max_input_bytes,
        )
        if payload is None:
            local_payload_rejects.append(claim)
            payload_failures.append(
                (
                    claim,
                    {
                        "status": "quarantined",
                        "error_class": "remote_payload_rejected",
                    },
                )
            )
            continue
        task["input"] = {
            "candidate_id": task["candidate"]["candidate_id"],
            "rally_id": task["rally"]["rally_id"],
            "query": payload["query"],
            "context": payload["context"],
            "evidence": payload["candidate"],
        }
        resolved_claims.append(claim)
    failed_probe_batches = {
        str(task.get("probe_batch_id") or "")
        for claim in local_payload_rejects
        if (task := tasks.get(claim.work_id)) is not None
        and str(task.get("probe_batch_id") or "")
    }
    if failed_probe_batches:
        for claim in resolved_claims[:]:
            if (
                str(tasks[claim.work_id].get("probe_batch_id") or "")
                in failed_probe_batches
            ):
                payload_failures.append(
                    (
                        claim,
                        {
                            "status": "quarantined",
                            "error_class": "remote_payload_rejected",
                        },
                    )
                )
                resolved_claims.remove(claim)
    if payload_failures:
        workset.commit(
            [claim for claim, _outcome in payload_failures],
            [outcome for _claim, outcome in payload_failures],
        )
    return (
        resolved_claims,
        payload_failures,
        local_payload_rejects,
        claim_integrity_failure,
    )


def _ox_batch_payload(
    tasks: Mapping[str, Mapping[str, Any]], current: Sequence[Any]
) -> dict[str, Any]:
    return {
        "schema": "chronovisor.recall-distill-teacher-batch.v1",
        "candidates": [tasks[claim.work_id]["input"] for claim in current],
    }


def _ox_response_metadata_matches(
    tasks: Mapping[str, Mapping[str, Any]],
    claims: Sequence[Any],
    response: Mapping[str, Any],
    *,
    max_input_bytes: int,
) -> bool:
    expected_metadata = ox_alpha_response_metadata(
        _ox_batch_payload(tasks, claims), max_input_bytes=max_input_bytes
    )
    return expected_metadata is not None and not any(
        response.get(key) != value for key, value in expected_metadata.items()
    )


def _ox_prepare_batches(
    *,
    claims: Sequence[Any],
    tasks: Mapping[str, Mapping[str, Any]],
    workset: Any,
    claim_limit: int,
    preflight: Callable[[Mapping[str, Any]], bool] | None,
    payload_scan_remaining: int | None,
) -> tuple[list[list[Any]], int]:
    batches: list[list[Any]] = []
    probe_groups: dict[str, list[Any]] = defaultdict(list)
    normal_claims: list[Any] = []
    for claim in claims:
        probe_batch_id = str(tasks[claim.work_id].get("probe_batch_id") or "")
        if probe_batch_id:
            probe_groups[probe_batch_id].append(claim)
        else:
            normal_claims.append(claim)
    incomplete_probe_outcomes: list[tuple[Any, dict[str, str]]] = []
    for probe_batch_id, group in probe_groups.items():
        ordered = sorted(
            group,
            key=lambda claim: int(
                tasks[claim.work_id]["assignment"].get("candidate_position", -1)
            ),
        )
        assignments_for_batch = [
            tasks[claim.work_id]["assignment"] for claim in ordered
        ]
        blind_orders = {
            assignment.get("blind_order") for assignment in assignments_for_batch
        }
        probe_candidate_ids = {
            str(tasks[claim.work_id]["candidate"].get("candidate_id") or "")
            for claim in ordered
        }
        valid_probe_batch = (
            len(ordered) == 2
            and len(probe_candidate_ids) == 2
            and len(blind_orders) == 1
            and blind_orders <= {"a_first", "b_first"}
            and all(
                assignment.get("probe") is True
                and assignment.get("probe_batch_id") == probe_batch_id
                and assignment.get("fixed_repeat") is True
                and assignment.get("order_swap") is True
                and assignment.get("probe_revision") == OX_PROBE_REVISION
                for assignment in assignments_for_batch
            )
            and [
                assignment.get("candidate_position")
                for assignment in assignments_for_batch
            ]
            == [0, 1]
            and len(
                canonical_json.canonical_json_bytes_strict(
                    _ox_batch_payload(tasks, ordered)
                )
            )
            <= 12_000
        )
        if valid_probe_batch:
            batches.append(ordered)
            continue
        incomplete_probe_outcomes.extend(
            (
                claim,
                {
                    "status": "quarantined",
                    "error_class": "incomplete_probe_pair",
                },
            )
            for claim in ordered
        )
    if incomplete_probe_outcomes:
        workset.commit(
            [claim for claim, _outcome in incomplete_probe_outcomes],
            [outcome for _claim, outcome in incomplete_probe_outcomes],
        )
        if not batches and not normal_claims:
            remaining = (
                OX_PREFLIGHT_SCAN_CLAIM_BUDGET
                if payload_scan_remaining is None
                else payload_scan_remaining
            ) - len(incomplete_probe_outcomes)
            if remaining > 0:
                return [], len(incomplete_probe_outcomes)
    batch: list[Any] = []
    candidate_ids: set[str] = set()
    oversize: list[tuple[Any, dict[str, str]]] = []
    for claim in normal_claims:
        task = tasks[claim.work_id]
        candidate_id = str(task["candidate"]["candidate_id"])
        proposed = [*batch, claim]
        request = _ox_batch_payload(tasks, proposed)
        if (
            candidate_id in candidate_ids
            or len(proposed) > 16
            or len(canonical_json.canonical_json_bytes_strict(request)) > 12_000
        ):
            if batch:
                batches.append(batch)
            single_request = _ox_batch_payload(tasks, [claim])
            if len(canonical_json.canonical_json_bytes_strict(single_request)) > 12_000:
                oversize.append(
                    (
                        claim,
                        {
                            "status": "quarantined",
                            "error_class": "remote_payload_rejected",
                        },
                    )
                )
                batch = []
                candidate_ids = set()
            else:
                batch = [claim]
                candidate_ids = {candidate_id}
        else:
            batch = proposed
            candidate_ids.add(candidate_id)
    if batch:
        batches.append(batch)
    if oversize:
        workset.commit(
            [claim for claim, _outcome in oversize],
            [outcome for _claim, outcome in oversize],
        )
    if claim_limit == 1:
        single_request_batches: list[list[Any]] = []
        for current in batches:
            if all(
                tasks[claim.work_id]["assignment"].get("probe") for claim in current
            ):
                single_request_batches.append(current)
            else:
                single_request_batches.extend([claim] for claim in current)
        batches = single_request_batches
    preflight_rejected: list[list[Any]] = []
    if callable(preflight):
        accepted: list[list[Any]] = []

        def ready_for_egress(current: Sequence[Any]) -> bool:
            try:
                return preflight(_ox_batch_payload(tasks, current)) is True
            except Exception:
                return False

        for current in batches:
            if ready_for_egress(current):
                accepted.append(current)
            elif len(current) == 1 or all(
                tasks[claim.work_id]["assignment"].get("probe") for claim in current
            ):
                preflight_rejected.append(current)
            else:
                for claim in current:
                    target = [claim]
                    (
                        accepted if ready_for_egress(target) else preflight_rejected
                    ).append(target)
        batches = accepted
    if preflight_rejected:
        rejected_claims = [claim for batch in preflight_rejected for claim in batch]
        workset.commit(
            rejected_claims,
            [
                {"status": "quarantined", "error_class": "remote_payload_rejected"}
                for _claim in rejected_claims
            ],
        )
    if claim_limit == 1 and len(batches) > 1:
        workset.release_unattempted(
            [claim for current in batches[1:] for claim in current]
        )
        batches = batches[:1]
    if not batches:
        rejected_count = len(oversize) + sum(map(len, preflight_rejected))
        remaining = (
            OX_PREFLIGHT_SCAN_CLAIM_BUDGET
            if payload_scan_remaining is None
            else payload_scan_remaining
        ) - rejected_count
        if rejected_count and remaining > 0:
            return [], rejected_count
    return batches, 0


def _ox_workset_progress(
    workset: Any,
    *,
    candidate_state: Mapping[str, Any],
    label_state: Mapping[str, Any],
    profile_contract_id: str,
    split_plan_id: str,
) -> dict[str, Any]:
    prior = workset.progress() or {}
    cursor = prior.get("cursor", {})
    provenance = prior.get("provenance", {})
    epoch = cursor.get("revision_epoch", 0) if isinstance(cursor, Mapping) else 0
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise DistillationError("OX workset progress is invalid")
    current_provenance = {
        "profile": OX_SINGLE_PROFILE,
        "profile_contract_id": profile_contract_id,
        "probe_revision": OX_PROBE_REVISION,
        "split_plan_id": split_plan_id,
    }
    if (
        prior
        and prior.get("progress_kind") != "ox-label-v2"
        and provenance != current_provenance
    ):
        epoch += 1
    return {
        "cursor": {
            "candidate_count": int(candidate_state.get("record_count") or 0),
            "label_count": int(label_state.get("records") or 0),
            "revision_epoch": epoch,
        },
        "ledger_heads": {
            "candidate": str(candidate_state.get("head_sha256") or ""),
            "labels": str(label_state.get("head_sha256") or ""),
        },
        "provenance": current_provenance,
        "progress_kind": "ox-workset-v2",
    }


def _ox_dispatch_and_commit(
    *,
    root: Path,
    claims: Sequence[Any],
    batches: Sequence[Sequence[Any]],
    ramp_cap: int,
    ramp_valid_receipts: int,
    ramp_provider_attempts: int,
    teacher: Teacher,
    tasks: Mapping[str, Mapping[str, Any]],
    config: DistillationConfig,
    workset: Any,
    profile_contract_id: str,
    source_binding: Mapping[str, str],
    split_plan_id: str,
    candidate_state: Mapping[str, Any],
    structural_verifier: Callable[
        [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], str | None
    ],
    label_path: Path,
) -> _TeacherBatchResult:
    # Dispatch one fixed-cap wave at a time.  The generic dispatcher remains a
    # transport primitive; only this OX layer can promote a cap after deep
    # identity, digest, and label validation below.
    from chronovisor.recall.recall_distillation_dispatcher import dispatch_claimed_work

    try:
        contract = _ensure_ox_profile_contract(
            root, config, source_binding=source_binding
        )
        if contract["artifact_id"] != profile_contract_id:
            raise DistillationError("OX profile contract changed before dispatch")
        contract_expiry = _ox_expiry(contract.get("expires_at"))
    except DistillationError:
        workset.release_unattempted(claims)
        return _TeacherBatchResult(
            deferred=True,
            workset_status=workset.status("ox"),
            profile_contract_id=profile_contract_id,
            last_durable_progress=workset.progress(),
        )

    results: list[Any] = []
    dispatched_batches = 0
    metadata_drift = False
    for start in range(0, len(batches), ramp_cap):
        wave = batches[start : start + ramp_cap]
        # A claim may have waited in the workset while an earlier wave ran.
        # Recheck at the actual egress boundary, not just when the run began.
        try:
            _ox_expiry(config.ox_expires_at)
        except DistillationError:
            workset.release_unattempted(claims)
            return _TeacherBatchResult(
                deferred=True,
                workset_status=workset.status("ox"),
                profile_contract_id=profile_contract_id,
                last_durable_progress=workset.progress(),
            )
        wave_results = dispatch_claimed_work(
            wave,
            lambda current: teacher.evaluate(_ox_batch_payload(tasks, current)),
            max_inflight=ramp_cap,
            max_retries=0 if config.teacher_claim_limit == 1 else 2,
            min_valid_results_per_cap=OX_RAMP_RECEIPTS_PER_CAP,
            initial_cap=ramp_cap,
            initial_valid_results=0,
            valid_result_count=lambda _response: 0,
        )
        results.extend(wave_results)
        dispatched_batches += len(wave)
        if any(
            result.status == "ok"
            and (
                not isinstance(result.value, Mapping)
                or not _ox_response_metadata_matches(
                    tasks,
                    result.work,
                    result.value,
                    max_input_bytes=config.max_input_bytes,
                )
            )
            for result in wave_results
        ):
            metadata_drift = True
            break
        if any(
            result.rate_limited or result.status == "stopped" for result in wave_results
        ):
            break
    if dispatched_batches < len(batches):
        workset.release_unattempted(
            [claim for batch in batches[dispatched_batches:] for claim in batch]
        )
    expected_identity = OX_ALPHA_FIXED_IDENTITY["route_identity"]
    stopped = False
    deferred = False
    records: list[dict[str, Any]] = []
    completed_claims: list[Any] = []
    outcomes: dict[str, dict[str, str]] = {}
    valid_provider_results = 0
    if metadata_drift:
        drift_claims = [claim for result in results for claim in result.work]
        if drift_claims:
            workset.commit(
                drift_claims,
                [
                    {"status": "retry", "error_class": "route_model_drift"}
                    for _claim in drift_claims
                ],
            )
        ramp_cap, ramp_valid_receipts, ramp_provider_attempts = _advance_ox_ramp(
            cap=ramp_cap,
            valid_receipts=ramp_valid_receipts,
            provider_attempts=ramp_provider_attempts,
            valid_results=0,
            actual_attempts=sum(result.attempts for result in results),
            rate_limited=any(result.rate_limited for result in results),
            stopped=True,
            max_inflight=config.teacher_max_inflight,
        )
        _append_ox_event(
            root,
            "ox-failure-receipts.jsonl",
            {
                "kind": "ox-provider-failure",
                    "profile_contract_id": profile_contract_id,
                    **source_binding,
                    "request_revision": OX_RAMP_REQUEST_REVISION,
                    "expires_at": contract_expiry,
                    "captured_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "category": "model_drift",
                "status": "hard_stop",
                "attempts": sum(result.attempts for result in results),
                "work_ids": [str(claim.work_id) for claim in drift_claims],
                "attempts_by_work": {
                    str(claim.work_id): int(claim.attempt) for claim in drift_claims
                },
                "provider_receipts": {
                    str(claim.work_id): expected_ox_provider_request_sha256(
                        profile_contract_id=profile_contract_id,
                        payload_digest=str(claim.payload_digest),
                        work_id=str(claim.work_id),
                        expires_at=contract_expiry,
                    )
                    for claim in drift_claims
                },
            },
        )
        return _TeacherBatchResult(
            model_calls=sum(result.attempts for result in results),
            deferred=True,
            workset_status=workset.status("ox"),
            profile_stopped=True,
            profile_contract_id=profile_contract_id,
            ramp_cap=ramp_cap,
            ramp_valid_receipts=ramp_valid_receipts,
            ramp_provider_attempts=ramp_provider_attempts,
            last_durable_progress=workset.progress(),
            source_binding=source_binding,
        )
    if not _ox_source_binding_matches(teacher, source_binding):
        workset.release_unattempted(claims)
        _append_ox_event(
            root,
            "ox-failure-receipts.jsonl",
            {
                "kind": "ox-provider-failure",
                    "profile_contract_id": profile_contract_id,
                    **source_binding,
                    "request_revision": OX_RAMP_REQUEST_REVISION,
                    "expires_at": contract_expiry,
                    "captured_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "category": "source_drift",
                "status": "hard_stop",
                "attempts": sum(result.attempts for result in results),
                "work_ids": [str(claim.work_id) for claim in claims],
                "attempts_by_work": {
                    str(claim.work_id): int(claim.attempt) for claim in claims
                },
                "provider_receipts": {
                    str(claim.work_id): expected_ox_provider_request_sha256(
                        profile_contract_id=profile_contract_id,
                        payload_digest=str(claim.payload_digest),
                        work_id=str(claim.work_id),
                        expires_at=contract_expiry,
                    )
                    for claim in claims
                },
            },
        )
        return _TeacherBatchResult(
            model_calls=sum(result.attempts for result in results),
            deferred=True,
            workset_status=workset.status("ox"),
            profile_stopped=True,
            profile_contract_id=profile_contract_id,
            last_durable_progress=workset.progress(),
            source_binding=source_binding,
        )
    for result in results:
        batch_claims = result.work
        if result.status != "ok" or not isinstance(result.value, Mapping):
            category = str(result.category or "teacher_failure")
            stage = getattr(result.error, "stage", None)
            error_class = (
                f"{category}.{stage}"
                if category == "invalid_response"
                and isinstance(stage, str)
                and re.fullmatch(r"[a-z][a-z0-9_]{0,31}", stage) is not None
                else category
            )
            stopped = stopped or result.status == "stopped"
            deferred = True
            for claim in batch_claims:
                outcomes[claim.work_id] = {
                    "status": (
                        "quarantined"
                        if category == "remote_payload_rejected"
                        or (category == "invalid_response" and claim.attempt >= 3)
                        else "retry"
                    ),
                    "error_class": error_class,
                }
            continue
        response = result.value
        response_metadata = response
        labels = response.get("labels")
        safe_labels = validate_ox_alpha_labels(
            labels,
            tuple(
                str(tasks[claim.work_id]["candidate"]["candidate_id"])
                for claim in batch_claims
            ),
        )
        valid = safe_labels is not None
        labels_by_id = {
            str(label["candidate_id"]): label for label in safe_labels or []
        }
        if not valid:
            deferred = True
            for claim in batch_claims:
                status = "quarantined" if claim.attempt >= 3 else "retry"
                outcomes[claim.work_id] = {
                    "status": status,
                    "error_class": "invalid_teacher_output",
                }
            continue
        valid_provider_results += 1
        for claim in batch_claims:
            task = tasks[claim.work_id]
            candidate = task["candidate"]
            rally = task["rally"]
            label_response = labels_by_id[str(candidate["candidate_id"])]
            predicate = structural_verifier(rally, candidate, label_response)
            records.append(
                {
                    "kind": "teacher-label",
                    "status": "completed",
                    "work_id": claim.work_id,
                    "payload_digest": claim.payload_digest,
                    "payload_source": task["payload_source"],
                    "rally_id": rally["rally_id"],
                    "candidate_id": candidate["candidate_id"],
                    "route": "opencode-go/ox-alpha-free",
                    "teacher_role": OX_TEACHER_ROLE,
                    "profile": OX_SINGLE_PROFILE,
                    "cohort": OX_SINGLE_COHORT,
                    "profile_contract_id": profile_contract_id,
                    **source_binding,
                    "expires_at": contract_expiry,
                    "identity_revision": OX_ALPHA_FIXED_IDENTITY["revision"],
                    "request_revision": OX_RAMP_REQUEST_REVISION,
                    "ramp_cap": ramp_cap,
                    "attempt_count": claim.attempt,
                    "route_identity": dict(expected_identity),
                    "route_digest": response_metadata["_route_digest"],
                    "model_digest": response_metadata["_model_digest"],
                    "prompt_sha256": response_metadata["_prompt_digest"],
                    "schema_sha256": response_metadata["_schema_digest"],
                    "test_only": response_metadata.get("_test_only") is True,
                    "provider_response_request_sha256": expected_ox_provider_request_sha256(
                        profile_contract_id=profile_contract_id,
                        payload_digest=claim.payload_digest,
                        work_id=claim.work_id,
                        expires_at=contract_expiry,
                    ),
                    "provider_request_sha256": expected_ox_provider_request_sha256(
                        profile_contract_id=profile_contract_id,
                        payload_digest=claim.payload_digest,
                        work_id=claim.work_id,
                        expires_at=contract_expiry,
                    ),
                    "request_sha256": expected_ox_request_sha256(
                        profile_contract_id=profile_contract_id,
                        payload_digest=claim.payload_digest,
                    ),
                    "assignment": task["assignment"],
                    "assignment_revision": str(
                        task["assignment"].get("revision") or ""
                    ),
                    **task["temporal"],
                    **_teacher_label(
                        label_response,
                        verified_predicate=predicate
                        if predicate in CLOSED_PREDICATES
                        else None,
                    ),
                }
            )
            completed_claims.append(claim)
    try:
        # Recheck after HTTP and immediately before appending the ledger.  No
        # transcript or raw provider body has been persisted at this point.
        _ox_expiry(config.ox_expires_at)
        contract = _ensure_ox_profile_contract(
            root, config, source_binding=source_binding
        )
        if contract["artifact_id"] != profile_contract_id:
            raise DistillationError("OX profile contract changed before commit")
    except DistillationError:
        # Every result is still uncommitted here, including failed HTTP
        # outcomes.  Returning only successes leaves retry leases stranded.
        workset.release_unattempted(claims)
        return _TeacherBatchResult(
            deferred=True,
            workset_status=workset.status("ox"),
            profile_contract_id=profile_contract_id,
            last_durable_progress=workset.progress(),
        )
    appended = store.append_chain_batch(label_path, records)
    appended_by_work = {str(row["work_id"]): row for row in appended}
    for claim in completed_claims:
        row = appended_by_work[claim.work_id]
        outcomes[claim.work_id] = {
            "status": "completed",
            "completion_ref": f"label-ledger:{row['record_sha256']}",
            "completion_digest": str(row["record_sha256"]),
        }
    active_claims = [claim for claim in claims if claim.work_id in outcomes]
    if active_claims:
        try:
            _ox_expiry(config.ox_expires_at)
        except DistillationError:
            # The label append was sealed only after the preceding check; do
            # not turn an expired lease into completed work.  Reconciliation
            # will safely finish it on a future valid run.
            workset.release_unattempted(claims)
            return _TeacherBatchResult(
                labels_written=len(appended),
                model_calls=sum(result.attempts for result in results),
                deferred=True,
                workset_status=workset.status("ox"),
                profile_contract_id=profile_contract_id,
                last_durable_progress=workset.progress(),
            )
        progress: Mapping[str, Any] | None = None
        if appended:
            label_state = store.chain_head(label_path)
            progress = _ox_workset_progress(
                workset,
                candidate_state=candidate_state,
                label_state=label_state,
                profile_contract_id=profile_contract_id,
                split_plan_id=split_plan_id,
            )
        workset.commit(
            active_claims,
            [outcomes[claim.work_id] for claim in active_claims],
            progress=progress,
        )
    attempts = sum(result.attempts for result in results)
    previous_cap = ramp_cap
    previous_receipts = ramp_valid_receipts
    previous_attempts = ramp_provider_attempts
    rate_limited = any(result.rate_limited for result in results)
    ramp_cap, ramp_valid_receipts, ramp_provider_attempts = _advance_ox_ramp(
        cap=ramp_cap,
        valid_receipts=ramp_valid_receipts,
        provider_attempts=ramp_provider_attempts,
        valid_results=valid_provider_results,
        actual_attempts=attempts,
        rate_limited=rate_limited,
        stopped=stopped,
        max_inflight=config.teacher_max_inflight,
    )
    captured_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    # A stage receipt is only evidence of an actual completed stage.  Cap 10
    # has no following transition, so it is emitted when its own gate closes.
    final_cap = max(min(cap, config.teacher_max_inflight) for cap in (1, 2, 5, 10))
    final_stage_completed = (
        previous_cap == final_cap
        and ramp_cap == final_cap
        and ramp_valid_receipts >= OX_RAMP_RECEIPTS_PER_CAP
        and ramp_valid_receipts * 100 >= ramp_provider_attempts * 95
    )
    final_receipt_emitted = any(
        row.get("profile_contract_id") == profile_contract_id
        and row.get("cap") == final_cap
        and all(row.get(key) == value for key, value in source_binding.items())
        for row in store.read_chain(
            store.distillation_dir(root) / "ox-ramp-receipts.jsonl"
        )
    )
    if ramp_cap > previous_cap or (final_stage_completed and not final_receipt_emitted):
        stage_cap = previous_cap if ramp_cap > previous_cap else final_cap
        stage_labels = [
            str(row.get("work_id") or "")
            for row in store.read_chain(label_path)
            if row.get("profile_contract_id") == profile_contract_id
            and row.get("ramp_cap") == stage_cap
            and row.get("status") == "completed"
        ]
        _append_ox_event(
            root,
            "ox-ramp-receipts.jsonl",
            {
                "kind": "ox-ramp-stage",
                "profile_contract_id": profile_contract_id,
                **source_binding,
                "request_revision": OX_RAMP_REQUEST_REVISION,
                "expires_at": contract_expiry,
                "cap": stage_cap,
                "valid_receipts": (
                    previous_receipts + valid_provider_results
                    if ramp_cap > previous_cap
                    else ramp_valid_receipts
                ),
                "attempts": (
                    previous_attempts + attempts
                    if ramp_cap > previous_cap
                    else ramp_provider_attempts
                ),
                "work_ids": stage_labels,
                "label_count": len(appended),
                "label_head_sha256": store.chain_head(label_path)["head_sha256"],
                "captured_at": captured_at,
            },
        )
    for result in results:
        category = str(result.category or "")
        if result.rate_limited:
            event = {
                "category": "429",
                "before_cap": previous_cap,
                "after_cap": ramp_cap,
                "status": "deferred",
            }
        elif category in {
            "http_5xx",
            "timeout",
            "http_402",
            "paid_fallback",
            "payment_required",
            "model_unavailable",
            "route_model_drift",
        }:
            event = {
                "category": {
                    "http_5xx": "5xx",
                    "timeout": "timeout",
                    "http_402": "402",
                    "paid_fallback": "paid",
                    "payment_required": "paid",
                    "model_unavailable": "model_drift",
                    "route_model_drift": "model_drift",
                }[category],
                "attempts": result.attempts,
                "bounded": category in {"http_5xx", "timeout"},
                "status": (
                    "hard_stop"
                    if category
                    in {
                        "http_402",
                        "paid_fallback",
                        "payment_required",
                        "model_unavailable",
                        "route_model_drift",
                    }
                    else "deferred"
                ),
            }
        else:
            continue
        _append_ox_event(
            root,
            "ox-failure-receipts.jsonl",
            {
                "kind": "ox-provider-failure",
                "profile_contract_id": profile_contract_id,
                **source_binding,
                "request_revision": OX_RAMP_REQUEST_REVISION,
                "expires_at": contract_expiry,
                "captured_at": captured_at,
                "work_ids": [str(claim.work_id) for claim in result.work],
                "attempts": result.attempts,
                "attempts_by_work": {
                    str(claim.work_id): int(claim.attempt) for claim in result.work
                },
                "provider_receipts": {
                    str(claim.work_id): expected_ox_provider_request_sha256(
                        profile_contract_id=profile_contract_id,
                        payload_digest=str(claim.payload_digest),
                        work_id=str(claim.work_id),
                        expires_at=contract_expiry,
                    )
                    for claim in result.work
                },
                **event,
            },
        )
    return _TeacherBatchResult(
        labels_written=len(appended),
        model_calls=sum(result.attempts for result in results),
        deferred=deferred,
        workset_status=workset.status("ox"),
        profile_stopped=stopped,
        profile_contract_id=profile_contract_id,
        ramp_cap=ramp_cap,
        ramp_valid_receipts=ramp_valid_receipts,
        ramp_provider_attempts=ramp_provider_attempts,
        last_durable_progress=workset.progress(),
        source_binding=source_binding,
    )


def _run_ox_teacher_batch(
    *,
    root: Path,
    config: DistillationConfig,
    teachers: Mapping[str, Teacher],
    snapshots: Mapping[str, Mapping[str, Any]],
    rally_by_id: Mapping[str, Mapping[str, Any]],
    texts: Mapping[str, str],
    label_path: Path,
    label_rows: Sequence[Mapping[str, Any]],
    candidate_indexed: bool = False,
    _payload_scan_remaining: int | None = None,
    structural_verifier: Callable[
        [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], str | None
    ],
) -> _TeacherBatchResult:
    """Advance, claim, dispatch, append, and commit OX work in that order."""

    from chronovisor.recall.recall_distillation_workset import DistillationWorkset

    teacher = teachers.get(OX_TEACHER_ROLE)
    workset = DistillationWorkset(
        store.distillation_dir(root) / "ox-workset.sqlite3", migrate=False
    )
    if teacher is None or teacher.local:
        return _TeacherBatchResult(
            deferred=True,
            workset_status=workset.status("ox"),
            last_durable_progress=workset.progress(),
        )
    try:
        source_binding = _ox_teacher_source_binding(teacher)
        profile_contract = _ensure_ox_profile_contract(
            root, config, source_binding=source_binding
        )
        profile_contract_id = str(profile_contract["artifact_id"])
        profile_contract_expiry = _ox_expiry(profile_contract.get("expires_at"))
    except DistillationError:
        return _TeacherBatchResult(
            deferred=True,
            workset_status=workset.status("ox"),
            last_durable_progress=workset.progress(),
        )
    try:
        worker_state = _read_worker_state(root)
        if worker_state.get("ox_profile_stopped") is True:
            return _TeacherBatchResult(
                deferred=True,
                workset_status=workset.status("ox"),
                last_durable_progress=workset.progress(),
                profile_stopped=True,
                profile_contract_id=profile_contract_id,
            )
    except store.DistillationStoreError:
        worker_state = {}
    ramp_source = (
        worker_state
        if worker_state.get("kind") == "worker-state"
        and worker_state.get("ox_profile_contract_id") == profile_contract_id
        and worker_state.get("ox_ramp_request_revision") == OX_RAMP_REQUEST_REVISION
        else {}
    )
    ramp_cap, ramp_valid_receipts, ramp_provider_attempts = _ox_ramp_state(
        ramp_source, config.teacher_max_inflight
    )
    try:
        split_plan = _read_split_plan(root)
        assignments = split_plan["assignments"]
        split_plan_id = _scheduling_split_plan_id(split_plan)
        age_bands = _scheduling_age_bands(root, split_plan)
    except (DistillationError, store.DistillationStoreError, KeyError):
        assignments = {}
        split_plan_id = ""
        age_bands = {}

    candidate_path = store.distillation_dir(root) / "candidate-ledger.jsonl"
    candidate_state: Mapping[str, Any] = {}
    if candidate_indexed:
        from chronovisor.recall import recall_distillation_catalog as catalog

        try:
            catalog.sync_candidate_index(root, candidate_path)
            candidate_state = catalog.candidate_index_state(root)
            previous_watermark = workset.watermark()
            prior_records = 0
            if previous_watermark is not None:
                if not isinstance(previous_watermark, Mapping):
                    raise DistillationError("OX workset watermark is invalid")
                prior_records = previous_watermark.get("candidate_records", 0)
                prior_head = str(previous_watermark.get("candidate_head") or "")
                if isinstance(prior_records, bool) or not isinstance(
                    prior_records, int
                ):
                    raise DistillationError("OX workset candidate watermark is invalid")
                if prior_records < 0 or prior_records > int(
                    candidate_state["record_count"]
                ):
                    raise DistillationError("OX workset candidate watermark regressed")
                if prior_records == int(candidate_state["record_count"]):
                    current_head = str(candidate_state.get("head_sha256") or "")
                    if prior_head and prior_head != current_head:
                        raise DistillationError("OX candidate index head conflicts")
                if (
                    str(previous_watermark.get("split_plan_id") or "") != split_plan_id
                    or str(previous_watermark.get("probe_revision") or "")
                    != OX_PROBE_REVISION
                ):
                    prior_records = 0
            snapshots = catalog.read_candidate_snapshots(
                root,
                candidate_path,
                catalog.candidate_rally_ids(root, after_seq=prior_records),
            )
        except catalog.CatalogError as exc:
            raise DistillationError("OX candidate index is unavailable") from exc

    prepared = _ox_prepare_tasks(
        config=config,
        snapshots=snapshots,
        rally_by_id=rally_by_id,
        assignments=assignments,
        split_plan_id=split_plan_id,
        profile_contract_id=profile_contract_id,
        candidate_indexed=candidate_indexed,
        candidate_state=candidate_state,
        age_bands=age_bands if isinstance(age_bands, Mapping) else None,
    )
    tasks = prepared["tasks"]
    work_items = prepared["work_items"]
    watermark = prepared["watermark"]
    add_task = prepared["add_task"]
    label_state = store.chain_head(label_path)
    workset.advance(
        work_items,
        watermark,
        progress=_ox_workset_progress(
            workset,
            candidate_state=candidate_state,
            label_state=label_state,
            profile_contract_id=profile_contract_id,
            split_plan_id=split_plan_id,
        ),
    )
    # Prepared tasks already contain candidate and probe work.
    claim_limit = config.teacher_claim_limit
    if (
        isinstance(claim_limit, bool)
        or not isinstance(claim_limit, int)
        or not 1 <= claim_limit <= 500
    ):
        raise DistillationError("teacher_claim_limit must be between 1 and 500")
    preflight = getattr(teacher, "accepts_egress_payload", None)
    scan_limit = claim_limit
    if claim_limit == 1 and callable(preflight):
        scan_limit = min(
            OX_PREFLIGHT_SCAN_CLAIM_BUDGET,
            _payload_scan_remaining
            if _payload_scan_remaining is not None
            else OX_PREFLIGHT_SCAN_CLAIM_BUDGET,
        )
    claims = list(
        workset.claim(
            "ox",
            scan_limit,
            OX_TEACHER_ROLE,
            7200,
        )
    )
    probe_claim_counts: dict[str, int] = defaultdict(int)
    for claim in claims:
        probe_batch_id = str(claim.provenance.get("probe_batch_id") or "")
        if probe_batch_id:
            probe_claim_counts[probe_batch_id] += 1
    # A blind-order request is an atomic pair, including after a restart where
    # no delta work items were rebuilt before pending claims were leased.
    if any(count == 1 for count in probe_claim_counts.values()):
        claims.extend(workset.claim("ox", 1, OX_TEACHER_ROLE, 7200))
    for receipt in workset.recent_transition_receipts(limit=3):
        if receipt["operation"] != "claim_reclaim":
            continue
        details = receipt["details"]
        _append_ox_event(
            root,
            "ox-lease-recovery-receipts.jsonl",
            {
                "kind": "ox-lease-reclaim",
                "profile_contract_id": profile_contract_id,
                **source_binding,
                "request_revision": OX_RAMP_REQUEST_REVISION,
                "expires_at": profile_contract_expiry,
                "workset_receipt_generation": receipt["generation"],
                "workset_receipt_sha256": receipt["receipt_sha256"],
                "reclaimed": details["count"],
                "leased_after": workset.status("ox")["leased"],
                "captured_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            },
            unique_key="workset_receipt_sha256",
        )
    existing = {
        str(row.get("work_id")): row
        for row in label_rows
        if row.get("kind") == "teacher-label"
        and row.get("status") == "completed"
        and row.get("route") == "opencode-go/ox-alpha-free"
        and row.get("teacher_role") == OX_TEACHER_ROLE
        and row.get("profile") == OX_SINGLE_PROFILE
        and row.get("cohort") == OX_SINGLE_COHORT
        and row.get("profile_contract_id") == profile_contract_id
    }
    reconciled = [claim for claim in claims if claim.work_id in existing]
    if reconciled:
        label_state = store.chain_head(label_path)
        workset.commit(
            reconciled,
            [
                {
                    "status": "completed",
                    "completion_ref": f"label-ledger:{existing[claim.work_id]['record_sha256']}",
                    "completion_digest": str(existing[claim.work_id]["record_sha256"]),
                }
                for claim in reconciled
            ],
            progress=_ox_workset_progress(
                workset,
                candidate_state=candidate_state,
                label_state=label_state,
                profile_contract_id=profile_contract_id,
                split_plan_id=split_plan_id,
            ),
        )
    claims = [claim for claim in claims if claim not in reconciled]
    claims = _ox_restore_claims(
        root=root,
        candidate_path=candidate_path,
        catalog=catalog if candidate_indexed else None,
        candidate_indexed=candidate_indexed,
        claims=claims,
        tasks=tasks,
        add_task=add_task,
        workset=workset,
        rally_by_id=rally_by_id,
        assignments=assignments,
        split_plan_id=split_plan_id,
        profile_contract_id=profile_contract_id,
    )
    # Restore and provenance validation are delegated to _ox_restore_claims.
    # payload resolution delegated to _ox_resolve_claim_payloads
    (
        resolved_claims,
        payload_failures,
        local_payload_rejects,
        claim_integrity_failure,
    ) = _ox_resolve_claim_payloads(
        claims=claims,
        tasks=tasks,
        texts=texts,
        config=config,
        workset=workset,
    )
    claims = resolved_claims
    if not claims:
        remaining = (
            OX_PREFLIGHT_SCAN_CLAIM_BUDGET
            if _payload_scan_remaining is None
            else _payload_scan_remaining
        ) - len(payload_failures)
        if local_payload_rejects and not claim_integrity_failure and remaining > 0:
            return _run_ox_teacher_batch(
                root=root,
                config=config,
                teachers=teachers,
                snapshots=snapshots,
                rally_by_id=rally_by_id,
                texts=texts,
                label_path=label_path,
                label_rows=label_rows,
                candidate_indexed=candidate_indexed,
                _payload_scan_remaining=remaining,
                structural_verifier=structural_verifier,
            )
        return _TeacherBatchResult(
            workset_status=workset.status("ox"),
            last_durable_progress=workset.progress(),
        )

    batches, rescan_count = _ox_prepare_batches(
        claims=claims,
        tasks=tasks,
        workset=workset,
        claim_limit=claim_limit,
        preflight=preflight,
        payload_scan_remaining=_payload_scan_remaining,
    )
    # Probe/normal grouping is handled by _ox_prepare_batches.
    if not batches:
        remaining = (
            OX_PREFLIGHT_SCAN_CLAIM_BUDGET
            if _payload_scan_remaining is None
            else _payload_scan_remaining
        ) - rescan_count
        if rescan_count and remaining > 0:
            return _run_ox_teacher_batch(
                root=root,
                config=config,
                teachers=teachers,
                snapshots=snapshots,
                rally_by_id=rally_by_id,
                texts=texts,
                label_path=label_path,
                label_rows=label_rows,
                candidate_indexed=candidate_indexed,
                _payload_scan_remaining=remaining,
                structural_verifier=structural_verifier,
            )
        return _TeacherBatchResult(
            deferred=True,
            workset_status=workset.status("ox"),
            last_durable_progress=workset.progress(),
        )

    return _ox_dispatch_and_commit(
        root=root,
        claims=claims,
        batches=batches,
        ramp_cap=ramp_cap,
        ramp_valid_receipts=ramp_valid_receipts,
        ramp_provider_attempts=ramp_provider_attempts,
        teacher=teacher,
        tasks=tasks,
        config=config,
        workset=workset,
        profile_contract_id=profile_contract_id,
        source_binding=source_binding,
        split_plan_id=split_plan_id,
        candidate_state=candidate_state,
        structural_verifier=structural_verifier,
        label_path=label_path,
    )


def _run_teacher_batch(
    *,
    root: Path,
    raw_dir: Path | None = None,
    config: DistillationConfig,
    teachers: Mapping[str, Teacher],
    snapshots: Mapping[str, Mapping[str, Any]],
    rally_by_id: Mapping[str, Mapping[str, Any]],
    texts: Mapping[str, str],
    label_path: Path,
    label_rows: Sequence[Mapping[str, Any]],
    candidate_indexed: bool = False,
    structural_verifier: Callable[
        [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], str | None
    ],
) -> _TeacherBatchResult:
    if config.teacher_profile == OX_SINGLE_PROFILE:
        return _run_ox_teacher_batch(
            root=root,
            config=config,
            teachers=teachers,
            snapshots=snapshots,
            rally_by_id=rally_by_id,
            texts=texts,
            label_path=label_path,
            label_rows=label_rows,
            candidate_indexed=candidate_indexed,
            structural_verifier=structural_verifier,
        )
    return _run_local_teacher_batch(
        root=root,
        raw_dir=raw_dir,
        config=config,
        teachers=teachers,
        snapshots=snapshots,
        rally_by_id=rally_by_id,
        texts=texts,
        label_path=label_path,
        label_rows=label_rows,
        structural_verifier=structural_verifier,
    )


def _local_work_id(payload_digest: str, route: str) -> str:
    digest = canonical_json.canonical_json_sha256_strict(
        {
            "kind": "local-teacher",
            "profile": LOCAL_TRIAD_PROFILE,
            "payload_digest": payload_digest,
            "route": route,
            "assignment_revision": ASSIGNMENT_REVISION,
        }
    )
    return f"local-teacher-{digest}"


def _local_workset_watermark(
    *,
    root: Path,
    raw_dir: Path,
    label_path: Path,
    snapshots: Mapping[str, Mapping[str, Any]],
    progress_kind: str = "local-teacher-v1",
) -> dict[str, Any]:
    candidate_path = store.distillation_dir(root) / "candidate-ledger.jsonl"
    candidate_state = store.chain_head(candidate_path)
    label_state = store.chain_head(label_path)
    try:
        split_plan = _read_split_plan(root)
        split_plan_id = _scheduling_split_plan_id(split_plan)
    except (DistillationError, store.DistillationStoreError):
        split_plan_id = ""
    return {
        "progress_kind": progress_kind,
        "raw_watermark": committed_raw_watermark(raw_dir),
        "candidate_head": str(candidate_state.get("head_sha256") or ""),
        "label_head": str(label_state.get("head_sha256") or ""),
        "candidate_count": int(candidate_state.get("records") or 0),
        "label_count": int(label_state.get("records") or 0),
        "split_plan_id": split_plan_id,
        "assignment_revision": ASSIGNMENT_REVISION,
        "probe_revision": PROBE_REVISION,
    }


def _local_workset_progress(watermark: Mapping[str, Any]) -> dict[str, Any]:
    """Payload-free lineage shared by local teacher and counterfactual work."""

    return {
        "cursor": {
            "candidate_count": int(watermark["candidate_count"]),
            "label_count": int(watermark["label_count"]),
        },
        "ledger_heads": {
            "candidate": str(watermark["candidate_head"]),
            "labels": str(watermark["label_head"]),
        },
        "provenance": {
            "assignment_revision": str(watermark["assignment_revision"]),
            "probe_revision": str(watermark["probe_revision"]),
            "split_plan_id": str(watermark["split_plan_id"]),
        },
        "progress_kind": "local-workset-v2",
    }


def _advance_local_workset(
    workset: Any, items: Sequence[Mapping[str, Any]], watermark: Mapping[str, Any]
) -> None:
    previous = workset.watermark()
    if isinstance(previous, Mapping):
        for count_key, head_key in (
            ("candidate_count", "candidate_head"),
            ("label_count", "label_head"),
        ):
            before_count = previous.get(count_key)
            before_head = previous.get(head_key)
            after_count = watermark[count_key]
            after_head = watermark[head_key]
            if not isinstance(before_count, int) or before_count < 0:
                raise DistillationError("local workset watermark is invalid")
            if (
                after_count < before_count
                or (after_count == before_count and before_head != after_head)
                or (after_count > before_count and not after_head)
            ):
                raise DistillationError("local workset watermark regressed")
    workset.advance(items, watermark, progress=_local_workset_progress(watermark))


def _prepare_local_teacher_work(
    *,
    snapshots: Mapping[str, Mapping[str, Any]],
    rally_by_id: Mapping[str, Mapping[str, Any]],
    split_assignments: Mapping[str, Any],
    split_plan_id: str,
    age_bands: Mapping[str, str] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    tasks: dict[str, dict[str, Any]] = {}
    work_items: list[dict[str, Any]] = []
    age_bands = dict(
        age_bands
        or _source_age_bands(list(rally_by_id.values()), assignments=split_assignments)
    )
    for rally_id, snapshot in sorted(
        snapshots.items(),
        key=lambda item: (_source_epoch(rally_by_id.get(item[0], {})), item[0]),
    ):
        rally = rally_by_id.get(rally_id)
        if rally is None or (
            split_plan_id
            and split_assignments.get(rally_id) not in {"train", "validation", "test"}
        ):
            continue
        candidates = list(snapshot.get("candidates", []))
        selected = candidates[:3]
        if candidates[3:]:
            selected.append(candidates[-1])
        selected_ids: set[str] = set()
        for candidate in selected:
            if not isinstance(candidate, Mapping):
                continue
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id:
                continue
            if candidate_id in selected_ids:
                continue
            selected_ids.add(candidate_id)
            assignment = teacher_assignment(rally_id, candidate_id)
            for route in assignment["routes"]:
                payload_digest = canonical_json.canonical_json_sha256_strict(
                    {
                        "rally_id": rally_id,
                        "candidate_id": candidate_id,
                        "route": route,
                        "candidate_sha256": str(candidate.get("text_sha256") or ""),
                        "query_sha256": str(rally.get("query_sha256") or ""),
                        "context_sha256": [
                            str(ref.get("semantic_sha256") or "")
                            for ref in rally.get("context_refs", [])
                            if isinstance(ref, Mapping)
                        ],
                        "snapshot_sha256": str(snapshot.get("snapshot_sha256") or ""),
                        "profile": LOCAL_TRIAD_PROFILE,
                    }
                )
                work_id = _local_work_id(payload_digest, str(route))
                tasks[work_id] = {
                    "rally": rally,
                    "candidate": candidate,
                    "assignment": assignment,
                    "route": str(route),
                }
                work_items.append(
                    {
                        "work_id": work_id,
                        "kind": f"local-teacher:{route}",
                        "payload_ref": f"candidate-snapshot:{rally_id}:{candidate_id}",
                        "payload_digest": payload_digest,
                        "priority": (
                            100
                            + _age_band_priority(age_bands.get(rally_id, "old-history"))
                            if assignment["probe"]
                            else _age_band_priority(
                                age_bands.get(rally_id, "old-history")
                            )
                        ),
                        "temporal_split": {
                            "as_of": str(rally.get("as_of") or ""),
                            "group_id": str(rally.get("session_cluster_id") or ""),
                            "split": str(split_assignments.get(rally_id) or "embargo"),
                            "split_plan_id": split_plan_id,
                        },
                        "provenance": {
                            "assignment_revision": ASSIGNMENT_REVISION,
                            "probe_revision": PROBE_REVISION,
                            "route": str(route),
                        },
                    }
                )
    return tasks, work_items


def _run_local_teacher_route(
    *,
    workset: Any,
    route: str,
    config: DistillationConfig,
    teachers: Mapping[str, Teacher],
    tasks: Mapping[str, Mapping[str, Any]],
    texts: Mapping[str, str],
    root: Path,
    raw_dir: Path,
    label_path: Path,
    snapshots: Mapping[str, Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    structural_verifier: Callable[
        [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], str | None
    ],
) -> _TeacherBatchResult | None:
    claims = list(
        workset.claim(
            f"local-teacher:{route}",
            min(16, config.teacher_claim_limit),
            str(route),
            300,
        )
    )
    if not claims:
        return None
    completed_by_work = {
        str(row["work_id"]): row
        for row in label_rows
        if row.get("kind") == "teacher-label" and row.get("work_id")
    }
    legacy_completed = {
        (
            str(row.get("rally_id")),
            str(row.get("candidate_id")),
            str(row.get("route")),
        ): row
        for row in label_rows
        if row.get("kind") == "teacher-label"
    }
    reconciled_rows: dict[str, Mapping[str, Any]] = {}
    for claim in claims:
        row = completed_by_work.get(claim.work_id)
        task = tasks.get(claim.work_id)
        if row is None and task is not None:
            legacy = legacy_completed.get(
                (
                    str(task["rally"]["rally_id"]),
                    str(task["candidate"]["candidate_id"]),
                    route,
                )
            )
            if (
                legacy is not None
                and legacy.get("payload_digest") == claim.payload_digest
            ):
                row = legacy
        if row is not None and isinstance(row.get("record_sha256"), str):
            reconciled_rows[claim.work_id] = row
    reconciled = [claim for claim in claims if claim.work_id in reconciled_rows]
    if reconciled:
        workset.commit(
            reconciled,
            [
                {
                    "status": "completed",
                    "completion_ref": f"label-ledger:{reconciled_rows[claim.work_id]['record_sha256']}",
                    "completion_digest": str(
                        reconciled_rows[claim.work_id]["record_sha256"]
                    ),
                }
                for claim in reconciled
            ],
        )
    claims = [claim for claim in claims if claim not in reconciled]
    active = [claim for claim in claims if claim.work_id in tasks]
    missing = [claim for claim in claims if claim.work_id not in tasks]
    if missing:
        workset.commit(
            missing,
            [
                {"status": "quarantined", "error_class": "payload_missing"}
                for _ in missing
            ],
        )
    if not active:
        return None
    unique_active: list[Any] = []
    duplicate_claims: list[Any] = []
    batch_candidate_ids: set[str] = set()
    for claim in active:
        candidate_id = str(tasks[claim.work_id]["candidate"]["candidate_id"])
        if candidate_id in batch_candidate_ids:
            duplicate_claims.append(claim)
            continue
        batch_candidate_ids.add(candidate_id)
        unique_active.append(claim)
    if duplicate_claims:
        workset.release_unattempted(duplicate_claims)
    batch: list[dict[str, Any]] = []
    for claim in unique_active:
        task = tasks[claim.work_id]
        payload = _teacher_payload(
            task["rally"],
            task["candidate"],
            texts,
            max_input_bytes=config.max_input_bytes,
        )
        if payload is None:
            workset.commit(
                [claim], [{"status": "quarantined", "error_class": "payload_missing"}]
            )
            continue
        batch.append(
            {
                "claim": claim,
                "task": task,
                "input": {
                    "candidate_id": task["candidate"]["candidate_id"],
                    "rally_id": task["rally"]["rally_id"],
                    "query": payload["query"],
                    "context": payload["context"],
                    "evidence": payload["candidate"],
                },
            }
        )
    if not batch:
        return _TeacherBatchResult(workset_status=workset.status(include_timing=True))
    worker_input = {
        "schema": "chronovisor.recall-distill-teacher-batch.v1",
        "candidates": [item["input"] for item in batch],
    }
    while len(canonical_json.canonical_json_bytes_strict(worker_input)) > min(
        config.max_input_bytes, 12_000
    ):
        dropped = batch.pop()
        workset.release_unattempted([dropped["claim"]])
        worker_input["candidates"] = [item["input"] for item in batch]
    if not batch:
        return _TeacherBatchResult(
            deferred=True, workset_status=workset.status(include_timing=True)
        )
    try:
        response = teachers[route].evaluate(worker_input)
        labels = response.get("labels")
        if (
            not isinstance(labels, list)
            or len(labels) != len(batch)
            or not all(isinstance(label, Mapping) for label in labels)
            or {
                str(label.get("candidate_id"))
                for label in labels
                if isinstance(label, Mapping)
            }
            != {str(item["task"]["candidate"]["candidate_id"]) for item in batch}
            or isinstance(teachers[route], _WorkerTeacher)
            and (
                response.get("_route_identity") != teachers[route].expected_route
                or response.get("_model_digest") != teachers[route].expected_digest
            )
        ):
            raise DistillationError("teacher batch response coverage is invalid")
    except DistillationDeferred as exc:
        if not exc.attempted and exc.failure_class in {
            "capacity_unavailable",
            "resource_busy",
            "foreground_preempted",
            "cancelled",
            "deferred",
        }:
            workset.release_unattempted([item["claim"] for item in batch])
            return _TeacherBatchResult(
                deferred=True, workset_status=workset.status(include_timing=True)
            )
        outcomes = [
            {
                "status": "quarantined" if item["claim"].attempt >= 3 else "retry",
                "error_class": "teacher_timeout",
                "retry_after_seconds": 0 if item["claim"].attempt >= 3 else 60,
            }
            for item in batch
        ]
        workset.commit([item["claim"] for item in batch], outcomes)
        return _TeacherBatchResult(
            deferred=True,
            model_calls=1,
            workset_status=workset.status(include_timing=True),
        )
    except OSError:
        outcomes = [
            {
                "status": "quarantined" if item["claim"].attempt >= 3 else "retry",
                "error_class": "teacher_transport",
                "retry_after_seconds": 0,
            }
            for item in batch
        ]
        workset.commit([item["claim"] for item in batch], outcomes)
        return _TeacherBatchResult(
            deferred=True,
            model_calls=1,
            workset_status=workset.status(include_timing=True),
        )
    except (TimeoutError, DistillationError):
        outcomes = [
            {
                "status": "quarantined" if item["claim"].attempt >= 3 else "retry",
                "error_class": "invalid_teacher_output",
                "retry_after_seconds": 0 if item["claim"].attempt >= 3 else 60,
            }
            for item in batch
        ]
        workset.commit([item["claim"] for item in batch], outcomes)
        return _TeacherBatchResult(
            deferred=True,
            model_calls=1,
            workset_status=workset.status(include_timing=True),
        )
    labels_by_id = {
        str(label["candidate_id"]): label
        for label in labels
        if isinstance(label, Mapping)
    }
    if any(
        label.get("verdict") not in RELEVANCE_LABELS | UTILITY_LABELS
        for label in labels_by_id.values()
    ):
        workset.commit(
            [item["claim"] for item in batch],
            [
                {
                    "status": "quarantined" if item["claim"].attempt >= 3 else "retry",
                    "error_class": "invalid_teacher_output",
                    "retry_after_seconds": 0 if item["claim"].attempt >= 3 else 60,
                }
                for item in batch
            ],
        )
        return _TeacherBatchResult(
            deferred=True,
            model_calls=1,
            workset_status=workset.status(include_timing=True),
        )
    records: list[dict[str, Any]] = []
    for item in batch:
        task = item["task"]
        candidate = task["candidate"]
        label_response = labels_by_id[str(candidate["candidate_id"])]
        predicate = structural_verifier(task["rally"], candidate, label_response)
        records.append(
            {
                "kind": "teacher-label",
                "status": "completed",
                "work_id": item["claim"].work_id,
                "payload_digest": item["claim"].payload_digest,
                "rally_id": task["rally"]["rally_id"],
                "candidate_id": candidate["candidate_id"],
                "route": route,
                "teacher_profile": config.teacher_profile,
                "profile_contract_id": "",
                "expires_at": "",
                "identity_revision": "local-teacher-v1",
                "request_revision": "local-teacher-v1",
                "assignment_revision": ASSIGNMENT_REVISION,
                "route_identity": response.get("_route_identity", {}),
                "model_digest": response.get("_model_digest", ""),
                "assignment": task["assignment"],
                **_teacher_label(
                    label_response,
                    verified_predicate=predicate
                    if predicate in CLOSED_PREDICATES
                    else None,
                ),
            }
        )
    appended = store.append_chain_batch(label_path, records)
    _advance_local_workset(
        workset,
        [],
        _local_workset_watermark(
            root=root, raw_dir=raw_dir, label_path=label_path, snapshots=snapshots
        ),
    )
    appended_by_work = {str(row["work_id"]): row for row in appended}
    workset.commit(
        [item["claim"] for item in batch],
        [
            {
                "status": "completed",
                "completion_ref": f"label-ledger:{appended_by_work[item['claim'].work_id]['record_sha256']}",
                "completion_digest": str(
                    appended_by_work[item["claim"].work_id]["record_sha256"]
                ),
            }
            for item in batch
        ],
    )
    return _TeacherBatchResult(
        labels_written=len(appended),
        model_calls=1,
        workset_status=workset.status(include_timing=True),
    )


def _run_local_teacher_batch(
    *,
    root: Path,
    raw_dir: Path | None = None,
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
    """Run one durable local teacher batch; bodies stay in the ledgers."""
    from chronovisor.recall.recall_distillation_workset import DistillationWorkset

    workset = DistillationWorkset(
        store.distillation_dir(root) / "local-workset.sqlite3"
    )
    raw_dir = raw_dir or root / "raw"  # compatibility for direct test callers
    try:
        split_plan = _read_split_plan(root)
        split_plan_id = _scheduling_split_plan_id(split_plan)
        split_assignments = split_plan.get("assignments", {})
        age_bands = _scheduling_age_bands(root, split_plan)
    except (DistillationError, store.DistillationStoreError, KeyError):
        split_plan_id = ""
        split_assignments = {}
        age_bands = {}
    tasks, work_items = _prepare_local_teacher_work(
        snapshots=snapshots,
        rally_by_id=rally_by_id,
        split_assignments=split_assignments,
        split_plan_id=split_plan_id,
        age_bands=age_bands if isinstance(age_bands, Mapping) else None,
    )
    _advance_local_workset(
        workset,
        work_items,
        _local_workset_watermark(
            root=root, raw_dir=raw_dir, label_path=label_path, snapshots=snapshots
        ),
    )
    for route in _ordered_teacher_routes(
        {task["route"]: [task] for task in tasks.values() if task["route"] in teachers},
        label_rows,
    ):
        result = _run_local_teacher_route(
            workset=workset,
            route=route,
            config=config,
            teachers=teachers,
            tasks=tasks,
            texts=texts,
            root=root,
            raw_dir=raw_dir,
            label_path=label_path,
            snapshots=snapshots,
            label_rows=label_rows,
            structural_verifier=structural_verifier,
        )
        if result is not None:
            return result
    return _TeacherBatchResult(workset_status=workset.status(include_timing=True))


def _prepare_counterfactual_work(
    *,
    root: Path,
    snapshots: Mapping[str, Mapping[str, Any]],
    rally_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], str]]:
    try:
        split_plan = _read_split_plan(root)
        split_plan_id = _scheduling_split_plan_id(split_plan)
        split_assignments = split_plan.get("assignments", {})
        age_bands = _scheduling_age_bands(root, split_plan)
    except (DistillationError, store.DistillationStoreError, KeyError):
        split_plan_id = ""
        split_assignments = {}
        age_bands = {}
    age_bands = (
        dict(age_bands)
        if isinstance(age_bands, Mapping)
        else _source_age_bands(
            list(rally_by_id.values()), assignments=split_assignments
        )
    )
    items: list[dict[str, Any]] = []
    keys: dict[tuple[str, str], str] = {}
    for rally_id, snapshot in sorted(
        snapshots.items(),
        key=lambda item: (_source_epoch(rally_by_id.get(item[0], {})), item[0]),
    ):
        rally = rally_by_id.get(rally_id)
        if (
            rally is None
            or not rally.get("actual_answer_refs")
            or (
                split_plan_id
                and split_assignments.get(rally_id)
                not in {"train", "validation", "test"}
            )
        ):
            continue
        candidates = [
            candidate
            for candidate in snapshot.get("candidates", [])[:3]
            if isinstance(candidate, Mapping)
        ]
        exposure = rally.get("exposure_receipts", [{}])
        exposure = exposure[0] if isinstance(exposure, list) and exposure else {}
        if isinstance(exposure, Mapping) and exposure.get("exposure_artifact_id"):
            try:
                artifact = store.read_sealed(
                    store.distillation_dir(root)
                    / "exposures"
                    / f"{exposure['exposure_artifact_id']}.json",
                    schema="chronovisor.recall-exact-exposure.v1",
                )
                candidates.extend(
                    candidate
                    for candidate in artifact.get("candidate_refs", [])
                    if isinstance(candidate, Mapping)
                )
                candidates.extend(
                    candidate
                    for candidate in artifact.get("candidate_pool_refs", [])
                    if isinstance(candidate, Mapping)
                    and candidate.get("selected") is False
                )
            except (KeyError, store.DistillationStoreError):
                pass
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id or (rally_id, candidate_id) in keys:
                continue
            payload_digest = canonical_json.canonical_json_sha256_strict(
                {
                    "kind": "local-counterfactual",
                    "revision": "a0-a1-v1",
                    "profile": LOCAL_TRIAD_PROFILE,
                    "rally_id": rally_id,
                    "candidate_id": candidate_id,
                    "candidate_sha256": str(
                        candidate.get("text_sha256")
                        or candidate.get("page_content_sha256")
                        or candidate.get("content_sha256")
                        or ""
                    ),
                    "snapshot_sha256": str(snapshot.get("snapshot_sha256") or ""),
                    "exposure_artifact_id": str(
                        exposure.get("exposure_artifact_id") or ""
                    ),
                    "actual_answer_refs": rally.get("actual_answer_refs", []),
                }
            )
            work_id = (
                "local-counterfactual-"
                + canonical_json.canonical_json_sha256_strict(
                    {
                        "kind": "local-counterfactual",
                        "profile": LOCAL_TRIAD_PROFILE,
                        "payload_digest": payload_digest,
                    }
                )
            )
            keys[(rally_id, candidate_id)] = work_id
            items.append(
                {
                    "work_id": work_id,
                    "kind": "local-counterfactual",
                    "payload_ref": f"candidate-snapshot:{rally_id}:{candidate_id}",
                    "payload_digest": payload_digest,
                    "priority": (
                        100 * int("page_content_sha256" in candidate)
                        + _age_band_priority(age_bands.get(rally_id, "old-history"))
                    ),
                    "temporal_split": {
                        "as_of": str(rally.get("as_of") or ""),
                        "group_id": str(rally.get("session_cluster_id") or ""),
                        "split": str(split_assignments.get(rally_id) or "embargo"),
                        "split_plan_id": split_plan_id,
                    },
                    "provenance": {
                        "kind": "counterfactual",
                        "revision": "a0-a1-v1",
                    },
                }
            )
    return items, keys


def _reconcile_counterfactual_claims(
    *,
    workset: Any,
    claims: Sequence[Any],
    keys: Mapping[tuple[str, str], str],
    label_rows: Sequence[Mapping[str, Any]],
) -> list[Any]:
    completed = {
        str(row.get("work_id")): row
        for row in label_rows
        if row.get("route") == "counterfactual" and row.get("work_id")
    }
    legacy_completed = {
        (str(row.get("rally_id")), str(row.get("candidate_id"))): row
        for row in label_rows
        if row.get("route") == "counterfactual"
    }
    reconciled_rows: dict[str, Mapping[str, Any]] = {}
    for claim in claims:
        row = completed.get(claim.work_id)
        if row is None:
            key = next(
                (key for key, work_id in keys.items() if work_id == claim.work_id),
                None,
            )
            legacy = legacy_completed.get(key) if key is not None else None
            if (
                legacy is not None
                and legacy.get("payload_digest") == claim.payload_digest
            ):
                row = legacy
        if row is not None and isinstance(row.get("record_sha256"), str):
            reconciled_rows[claim.work_id] = row
    reconciled = [claim for claim in claims if claim.work_id in reconciled_rows]
    if reconciled:
        workset.commit(
            reconciled,
            [
                {
                    "status": "completed",
                    "completion_ref": f"label-ledger:{reconciled_rows[claim.work_id]['record_sha256']}",
                    "completion_digest": str(
                        reconciled_rows[claim.work_id]["record_sha256"]
                    ),
                }
                for claim in reconciled
            ],
        )
    return [claim for claim in claims if claim not in reconciled]


def _counterfactual_snapshot_inputs(
    *,
    root: Path,
    rally: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    texts: Mapping[str, str],
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    dict[str, dict[str, Any]],
    dict[str, str],
    list[dict[str, Any]],
]:
    receipts = rally.get("exposure_receipts")
    exposure = (
        receipts[0]
        if isinstance(receipts, list) and receipts and isinstance(receipts[0], Mapping)
        else {}
    )
    exposure_id = str(exposure.get("exposure_artifact_id") or "")
    # Counterfactual labels are evidence about a concrete prior exposure.  A
    # missing, malformed, or unverifiable receipt is not a degraded input: it
    # is a resumable precondition failure.  In particular, never let a
    # snapshot-only fallback manufacture a blind-comparison training row.
    if re.fullmatch(r"[0-9a-f]{64}", exposure_id) is None:
        raise DistillationDeferred(
            "sealed counterfactual exposure receipt is unavailable",
            failure_class="counterfactual_exposure_unavailable",
            attempted=False,
        )
    exposure_artifact: Mapping[str, Any] = {}
    try:
        exposure_artifact = store.read_sealed(
            store.distillation_dir(root) / "exposures" / f"{exposure_id}.json",
            schema="chronovisor.recall-exact-exposure.v1",
        )
    except (KeyError, store.DistillationStoreError):
        raise DistillationDeferred(
            "sealed counterfactual exposure receipt is unavailable",
            failure_class="counterfactual_exposure_unavailable",
            attempted=False,
        ) from None
    if str(exposure_artifact.get("artifact_id") or "") != exposure_id:
        raise DistillationDeferred(
            "sealed counterfactual exposure receipt is invalid",
            failure_class="counterfactual_exposure_unavailable",
            attempted=False,
        )
    exact_candidates = exposure_artifact.get("candidate_refs", [])
    raw_feature_rows = [
        *(
            exposure_artifact.get("candidate_feature_snapshot", [])
            if isinstance(exposure_artifact.get("candidate_feature_snapshot", []), list)
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
        candidate_id_raw = item.get("candidate_id")
        candidate_id = str(candidate_id_raw or "")
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
                "candidate_id": candidate_id_raw,
                "text_sha256": item.get("content_sha256"),
                "ref": evidence_refs[0]
                if isinstance(evidence_refs, list) and evidence_refs
                else {"structural": {}},
                "rendered_context": rendered,
            }
        )
    return (
        exposure,
        exposure_artifact,
        exact_features,
        original_evidence,
        removal_candidates,
    )


def _counterfactual_additions(
    *,
    snapshot: Mapping[str, Any],
    exposure_artifact: Mapping[str, Any],
    original_evidence: Mapping[str, str],
    removal_candidates: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
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
        if str(candidate["candidate_id"]) not in original_evidence
        and str(candidate["candidate_id"])
        not in {str(item["candidate_id"]) for item in live_additions}
    ]
    return [*removal_candidates, *live_additions, *historical_additions]


def _compare_and_commit_counterfactual(
    *,
    workset: Any,
    claim: Any,
    counterfactual: CounterfactualGenerator,
    payload: Mapping[str, Any],
    label_path: Path,
    root: Path,
    raw_dir: Path,
    snapshots: Mapping[str, Mapping[str, Any]],
    exposure: Mapping[str, Any],
    exact_features: Mapping[str, Mapping[str, Any]],
    rally_id: str,
    candidate: Mapping[str, Any],
    mode: str,
    config: DistillationConfig,
) -> _CounterfactualBlockResult:
    response: Mapping[str, Any] = {}
    try:
        response = counterfactual.compare(payload)
        generator_digest = str(response.get("generator_model_digest") or "")
        judge_digest = str(response.get("judge_model_digest") or "")
        verdict = str(response.get("verdict") or "uncertain")
        blind_orders = response.get("blind_orders")
        generator_route_identity = response.get("generator_route_identity")
        judge_route_identity = response.get("judge_route_identity")
        local_routes = (
            isinstance(generator_route_identity, Mapping)
            and isinstance(judge_route_identity, Mapping)
            and generator_route_identity.get("location") == "local"
            and judge_route_identity.get("location") == "local"
            and bool(generator_route_identity.get("provider"))
            and bool(judge_route_identity.get("provider"))
            and bool(generator_route_identity.get("model"))
            and bool(judge_route_identity.get("model"))
            and dict(generator_route_identity) != dict(judge_route_identity)
        )
        if (
            response.get("order_agreement") is not True
            or re.fullmatch(r"[0-9a-f]{64}", str(response.get("a0_sha256") or ""))
            is None
            or re.fullmatch(r"[0-9a-f]{64}", str(response.get("a1_sha256") or ""))
            is None
            or re.fullmatch(r"[0-9a-f]{64}", generator_digest) is None
            or re.fullmatch(r"[0-9a-f]{64}", judge_digest) is None
            or generator_digest == judge_digest
            or verdict not in UTILITY_LABELS
            or not isinstance(blind_orders, list)
            or set(blind_orders) != {"a0_first", "a1_first"}
            or len(blind_orders) != 2
            or not local_routes
            or isinstance(counterfactual, _WorkerCounterfactual)
            and (
                dict(generator_route_identity)
                != counterfactual.routes["recall.distill.answer_generator"]
                or generator_digest
                != counterfactual.digests["recall.distill.answer_generator"]
                or dict(judge_route_identity)
                != counterfactual.routes["recall.distill.utility_judge"]
                or judge_digest
                != counterfactual.digests["recall.distill.utility_judge"]
            )
        ):
            raise DistillationError("counterfactual response is invalid")
        label = adjudicate_label(
            verdict,
            closed_predicate=None,
            reason=str(response.get("reason") or "")[:500],
            dimension="answer_utility",
        )
    except DistillationDeferred as exc:
        if not exc.attempted and exc.failure_class in {
            "capacity_unavailable",
            "resource_busy",
            "foreground_preempted",
            "cancelled",
            "deferred",
        }:
            workset.release_unattempted([claim])
            return _CounterfactualBlockResult(pending=True, deferred=True)
        workset.commit(
            [claim],
            [
                {
                    "status": "quarantined" if claim.attempt >= 3 else "retry",
                    "error_class": "counterfactual_timeout",
                    "retry_after_seconds": 0 if claim.attempt >= 3 else 60,
                }
            ],
        )
        return _CounterfactualBlockResult(pending=True, deferred=True)
    except OSError:
        workset.commit(
            [claim],
            [
                {
                    "status": "quarantined" if claim.attempt >= 3 else "retry",
                    "error_class": "counterfactual_transport",
                    "retry_after_seconds": 0,
                }
            ],
        )
        return _CounterfactualBlockResult(pending=True, deferred=True)
    except (TimeoutError, DistillationError):
        workset.commit(
            [claim],
            [
                {
                    "status": "quarantined" if claim.attempt >= 3 else "retry",
                    "error_class": "counterfactual_invalid",
                    "retry_after_seconds": 0 if claim.attempt >= 3 else 60,
                }
            ],
        )
        return _CounterfactualBlockResult(pending=True, deferred=True)
    candidate_id_raw = candidate["candidate_id"]
    candidate_id = str(candidate_id_raw)
    ox_contract_id = (
        _current_ox_profile_contract_id(root)
        if config.teacher_profile == OX_SINGLE_PROFILE
        else ""
    )
    if config.teacher_profile == OX_SINGLE_PROFILE and not ox_contract_id:
        workset.release_unattempted([claim])
        return _CounterfactualBlockResult(pending=True, deferred=True)
    appended = store.append_chain(
        label_path,
        {
            "kind": "counterfactual-label",
            "work_id": claim.work_id,
            "payload_digest": claim.payload_digest,
            "rally_id": rally_id,
            "candidate_id": candidate_id_raw,
            "route": "counterfactual",
            "assignment": {
                "revision": ASSIGNMENT_REVISION,
                "kind": "counterfactual",
            },
            "split_plan_id": str(claim.temporal_split.get("split_plan_id") or ""),
            "as_of": str(claim.temporal_split.get("as_of") or ""),
            "group_id": str(claim.temporal_split.get("group_id") or ""),
            "counterfactual_producer": "chronovisor-local-blind-v1",
            "counterfactual_revision": "two-order-locked-v1",
            # This producer is intentionally local; only its stable route
            # fingerprints are retained, never generated text or provider body.
            "profile": config.teacher_profile,
            "cohort": OX_SINGLE_COHORT
            if config.teacher_profile == OX_SINGLE_PROFILE
            else LOCAL_TRIAD_PROFILE,
            "profile_contract_id": ox_contract_id,
            "identity_revision": "local-blind-counterfactual-v1",
            "request_revision": "local-blind-counterfactual-v1",
            "assignment_revision": ASSIGNMENT_REVISION,
            "expires_at": config.ox_expires_at
            if config.teacher_profile == OX_SINGLE_PROFILE
            else "",
            "mode": mode,
            "exposure_artifact_id": str(exposure.get("exposure_artifact_id") or ""),
            "a0_sha256": response.get("a0_sha256", ""),
            "a1_sha256": response.get("a1_sha256", ""),
            "blind_orders": response.get("blind_orders", []),
            "order_agreement": response.get("order_agreement", False),
            "generator_route_identity": response.get("generator_route_identity", {}),
            "generator_model_digest": response.get("generator_model_digest", ""),
            "judge_route_identity": response.get("judge_route_identity", {}),
            "judge_model_digest": response.get("judge_model_digest", ""),
            "model_digest": response.get("generator_model_digest", ""),
            **(
                {"features": exact_features[candidate_id]}
                if candidate_id in exact_features
                else {}
            ),
            **label,
        },
    )
    _advance_local_workset(
        workset,
        [],
        _local_workset_watermark(
            root=root,
            raw_dir=raw_dir,
            label_path=label_path,
            snapshots=snapshots,
            progress_kind="local-counterfactual-v1",
        ),
    )
    workset.commit(
        [claim],
        [
            {
                "status": "completed",
                "completion_ref": f"label-ledger:{appended['record_sha256']}",
                "completion_digest": str(appended["record_sha256"]),
            }
        ],
    )
    return _CounterfactualBlockResult(pending=True, written=1, model_calls=1)


def _run_counterfactual_claim(
    *,
    workset: Any,
    claim: Any,
    target: tuple[str, str],
    root: Path,
    raw_dir: Path,
    config: DistillationConfig,
    counterfactual: CounterfactualGenerator,
    snapshots: Mapping[str, Mapping[str, Any]],
    rally_by_id: Mapping[str, Mapping[str, Any]],
    texts: Mapping[str, str],
    label_path: Path,
) -> _CounterfactualBlockResult:
    for rally_id, snapshot in sorted(
        snapshots.items(),
        key=lambda item: (_source_epoch(rally_by_id.get(item[0], {})), item[0]),
    ):
        rally = rally_by_id.get(rally_id)
        if rally is None or not rally.get("actual_answer_refs"):
            continue
        try:
            (
                exposure,
                exposure_artifact,
                exact_features,
                original_evidence,
                removal_candidates,
            ) = _counterfactual_snapshot_inputs(
                root=root, rally=rally, snapshot=snapshot, texts=texts
            )
        except DistillationDeferred:
            if target[0] == rally_id:
                # Receipt acquisition is outside this claim.  Preserve the
                # claim and its attempt budget so a later sealed exposure can
                # resume it without creating any label or training row.
                workset.release_unattempted([claim])
                return _CounterfactualBlockResult(pending=True, deferred=True)
            continue
        if exposure and set(original_evidence) != {
            str(candidate_id) for candidate_id in exposure["candidate_ids"]
        }:
            if target[0] == rally_id:
                workset.commit(
                    [claim],
                    [
                        {
                            "status": "quarantined" if claim.attempt >= 3 else "retry",
                            "error_class": "counterfactual_exposure_mismatch",
                            "retry_after_seconds": 0 if claim.attempt >= 3 else 60,
                        }
                    ],
                )
                return _CounterfactualBlockResult(pending=True, deferred=True)
            continue
        candidates = _counterfactual_additions(
            snapshot=snapshot,
            exposure_artifact=exposure_artifact,
            original_evidence=original_evidence,
            removal_candidates=removal_candidates,
        )
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            if (rally_id, candidate_id) != target:
                continue
            candidate_text = candidate.get("rendered_context") or texts.get(
                str(candidate.get("text_sha256") or ""), ""
            )
            if not isinstance(candidate_text, str) or not candidate_text:
                workset.commit(
                    [claim],
                    [{"status": "quarantined", "error_class": "payload_missing"}],
                )
                return _CounterfactualBlockResult(pending=True)
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
                workset.commit(
                    [claim],
                    [{"status": "quarantined", "error_class": "payload_invalid"}],
                )
                return _CounterfactualBlockResult(pending=True)
            return _compare_and_commit_counterfactual(
                workset=workset,
                claim=claim,
                counterfactual=counterfactual,
                payload=payload,
                label_path=label_path,
                root=root,
                raw_dir=raw_dir,
                snapshots=snapshots,
                exposure=exposure,
                exact_features=exact_features,
                rally_id=rally_id,
                candidate=candidate,
                mode=mode,
                config=config,
            )
    workset.commit(
        [claim],
        [
            {
                "status": "quarantined" if claim.attempt >= 3 else "retry",
                "error_class": "counterfactual_payload_missing",
                "retry_after_seconds": 0 if claim.attempt >= 3 else 60,
            }
        ],
    )
    return _CounterfactualBlockResult(pending=True, deferred=True)


def _run_counterfactual_block(
    *,
    execute: bool,
    root: Path,
    raw_dir: Path | None = None,
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
    from chronovisor.recall.recall_distillation_workset import DistillationWorkset

    workset = DistillationWorkset(
        store.distillation_dir(root) / "local-workset.sqlite3"
    )
    raw_dir = raw_dir or root / "raw"  # compatibility for direct test callers
    items, keys = _prepare_counterfactual_work(
        root=root, snapshots=snapshots, rally_by_id=rally_by_id
    )
    _advance_local_workset(
        workset,
        items,
        _local_workset_watermark(
            root=root,
            raw_dir=raw_dir,
            label_path=label_path,
            snapshots=snapshots,
            progress_kind="local-counterfactual-v1",
        ),
    )
    pending = bool(workset.status("local-counterfactual")["backlog"])
    if not execute:
        return _CounterfactualBlockResult(pending=pending)
    claims = _reconcile_counterfactual_claims(
        workset=workset,
        claims=list(workset.claim("local-counterfactual", 1, "counterfactual", 300)),
        keys=keys,
        label_rows=label_rows,
    )
    if not claims:
        return _CounterfactualBlockResult(
            pending=bool(workset.status("local-counterfactual")["backlog"])
        )
    claim = claims[0]
    target = next(
        (key for key, work_id in keys.items() if work_id == claim.work_id), None
    )
    if target is None:
        workset.commit(
            [claim], [{"status": "quarantined", "error_class": "payload_missing"}]
        )
        return _CounterfactualBlockResult(pending=True)
    return _run_counterfactual_claim(
        workset=workset,
        claim=claim,
        target=target,
        root=root,
        raw_dir=raw_dir,
        config=config,
        counterfactual=counterfactual,
        snapshots=snapshots,
        rally_by_id=rally_by_id,
        texts=texts,
        label_path=label_path,
    )


def _source_age_bands(
    rallies: Sequence[Mapping[str, Any]],
    *,
    assignments: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Classify source rows against a stable source-watermark UTC boundary."""

    parsed: dict[str, datetime] = {}
    for rally in rallies:
        rally_id = str(rally.get("rally_id") or "")
        value = str(rally.get("as_of") or "")
        try:
            parsed[rally_id] = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            ).astimezone(UTC)
        except ValueError:
            continue
    newest = max(parsed.values(), default=None)
    cutoff = newest - timedelta(days=7) if newest is not None else None
    result: dict[str, str] = {}
    for rally in rallies:
        rally_id = str(rally.get("rally_id") or "")
        if rally.get("locked_test_read_only") is True or (
            assignments is not None and assignments.get(rally_id) == "test"
        ):
            result[rally_id] = "locked-test"
        elif (
            cutoff is not None
            and parsed.get(rally_id, datetime.min.replace(tzinfo=UTC)) >= cutoff
        ):
            result[rally_id] = "recent"
        else:
            result[rally_id] = "old-history"
    return result


def _source_age_boundary(rallies: Sequence[Mapping[str, Any]]) -> str:
    values: list[datetime] = []
    for rally in rallies:
        try:
            values.append(
                datetime.fromisoformat(
                    str(rally.get("as_of") or "").replace("Z", "+00:00")
                ).astimezone(UTC)
            )
        except ValueError:
            continue
    return max(values).isoformat().replace("+00:00", "Z") if values else ""


def _source_epoch(rally: Mapping[str, Any]) -> float:
    value = str(rally.get("as_of") or "")
    try:
        return (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            .astimezone(UTC)
            .timestamp()
        )
    except ValueError:
        # Historical unit fixtures use a monotonic raw microsecond surrogate.
        # Production source rows are RFC3339, so this is only a deterministic
        # compatibility representation and never a wall-clock fallback.
        try:
            return float(value) if value and value.isdecimal() else float("-inf")
        except ValueError:
            return float("-inf")


def _age_aware_backfill(
    *, rallies: Sequence[Mapping[str, Any]], profile: str
) -> list[dict[str, Any]]:
    """Round-robin fixed UTC bands; source time, not wall-clock, defines age."""

    ordered = sorted(
        (dict(rally) for rally in rallies),
        key=lambda rally: (_source_epoch(rally), str(rally.get("rally_id") or "")),
    )
    age_bands = _source_age_bands(ordered)
    bands: dict[str, list[dict[str, Any]]] = {
        "old-history": [],
        "recent": [],
        "locked-test": [],
    }
    boundary = _source_age_boundary(ordered)
    for rally in ordered:
        band = age_bands[str(rally.get("rally_id") or "")]
        bands[band].append(
            {**rally, "_backfill_band": band, "_age_boundary_utc": boundary}
        )
    schedule = (
        ("old-history", "recent", "locked-test")
        if profile == OX_SINGLE_PROFILE
        else ("recent", "locked-test", "old-history")
    )
    result: list[dict[str, Any]] = []
    while any(bands.values()):
        for band in schedule:
            if bands[band]:
                result.append(bands[band].pop(0))
    return result


def _age_band_priority(band: str) -> int:
    return {"locked-test": 30, "recent": 20, "old-history": 10}.get(band, 0)


def _capture_candidate_snapshots(
    *,
    root: Path,
    raw_dir: Path,
    config: DistillationConfig,
    rallies: Sequence[Mapping[str, Any]],
    texts: Mapping[str, str],
    index_path: Path,
    label_rows: Sequence[Mapping[str, Any]],
    cold_start: bool,
    deadline: float,
) -> _CandidateCaptureResult:
    candidate_path = store.distillation_dir(root) / "candidate-ledger.jsonl"
    snapshots: dict[str, Mapping[str, Any]]
    if config.teacher_profile == OX_SINGLE_PROFILE:
        from chronovisor.recall import recall_distillation_catalog as catalog

        try:
            catalog.sync_candidate_index(root, candidate_path)
            known_rally_ids = catalog.candidate_rally_ids(root)
            label_rally_ids = {
                str(row.get("rally_id") or "")
                for row in label_rows
                if isinstance(row.get("rally_id"), str) and row.get("rally_id")
            }
            snapshots = {
                rally_id: snapshot
                for rally_id, snapshot in catalog.read_candidate_snapshots(
                    root, candidate_path, label_rally_ids
                ).items()
            }
        except catalog.CatalogError as exc:
            raise DistillationError("OX candidate index is unavailable") from exc
    else:
        snapshots = {
            str(row["rally_id"]): row["snapshot"]
            for row in _read_chain(candidate_path)
            if isinstance(row.get("snapshot"), dict)
        }
        known_rally_ids = set(snapshots)
    candidate_limit = 100 if cold_start else config.chunk_size
    planned = [rally for rally in rallies if rally["rally_id"] not in known_rally_ids]
    planned = _age_aware_backfill(rallies=planned, profile=config.teacher_profile)
    split_plan: Mapping[str, Any] = {}
    if cold_start:
        training = materialize_training_rows(
            root,
            _rallies=rallies,
            _snapshots=snapshots,
            _label_rows=label_rows,
        )
        _, cohort = _active_training_cohort(
            training["rows"],
            teacher_profile=config.teacher_profile,
            profile_contract_id=(
                _current_ox_profile_contract_id(root)
                if config.teacher_profile == OX_SINGLE_PROFILE
                else ""
            ),
        )
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
                _source_epoch(rally),
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
                "backfill_revision": "age-bands-v1",
                "backfill_profile": config.teacher_profile,
                "backfill_band": str(rally.get("_backfill_band") or "old-history"),
                "age_boundary_utc": str(rally.get("_age_boundary_utc") or ""),
                "snapshot": snapshot,
            }
        )
        snapshots[str(rally["rally_id"])] = snapshot
        work.append(rally)
    store.append_chain_batch(candidate_path, payloads)
    if config.teacher_profile == OX_SINGLE_PROFILE and payloads:
        try:
            catalog.sync_candidate_index(root, candidate_path)
        except catalog.CatalogError as exc:
            raise DistillationError("OX candidate index cannot advance") from exc
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


def _prepare_distillation_chunk(
    *,
    root: Path | None,
    raw_dir: Path | None,
    config_path: Path | None,
    teachers: Mapping[str, Teacher] | None,
    counterfactual: CounterfactualGenerator | None,
    structural_verifier: (
        Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], str | None]
        | None
    ),
    dry_run: bool,
    cold_start: bool,
    max_elapsed_seconds: int,
) -> dict[str, Any]:
    root = root or CHRONOVISOR_ROOT
    raw_dir = raw_dir or root / "raw"
    config = load_distillation_config(config_path)
    deadline = time.monotonic() + max_elapsed_seconds
    if not config.enabled:
        return {"early": {"status": "disabled", "processed": 0}}
    ox_teacher = (teachers or {}).get(OX_TEACHER_ROLE)
    ox_source_binding: Mapping[str, str] | None = None
    if config.teacher_profile == OX_SINGLE_PROFILE and config.ox_enabled and ox_teacher:
        # A mapping-shaped method on a caller-supplied teacher is not an
        # authority for remote-label provenance.  Only the closed adapter
        # identity can replace the installed production source binding.
        ox_source_binding = _ox_teacher_source_binding(ox_teacher)
    ox_profile_contract_id = (
        str(
            _ensure_ox_profile_contract(root, config, source_binding=ox_source_binding)[
                "artifact_id"
            ]
        )
        if config.teacher_profile == OX_SINGLE_PROFILE and config.ox_enabled
        else ""
    )
    try:
        scheduler_state = _read_worker_state(root)
    except store.DistillationStoreError:
        scheduler_state = {}
    teacher_model_calls = int(scheduler_state.get("teacher_model_calls", 0))
    counterfactual_model_calls = int(
        scheduler_state.get("counterfactual_model_calls", 0)
    )
    # The catalog owns Raw projection and legacy bootstrap. This worker consumes
    # the current text-free projection and resolves text only on demand.
    from chronovisor.recall import recall_distillation_catalog as catalog

    catalog.advance(raw_dir, root, config.max_input_bytes)
    rallies = catalog.rallies(root)
    if teachers is None:
        teachers, default_counterfactual = _default_workers(
            config,
            teacher_deadline_ms=120_000 if cold_start else 60_000,
            counterfactual_deadline_ms=45_000 if cold_start else 60_000,
        )
        if counterfactual is None:
            counterfactual = default_counterfactual
    structural_verifier = structural_verifier or _default_structural_verifier
    ledger_path = store.distillation_dir(root) / "rally-manifest.jsonl"
    existing = {
        manifest.get("rally_id")
        for row in _read_chain(ledger_path)
        if isinstance((manifest := row.get("manifest")), dict)
    }
    manifest_limit = 500 if cold_start else config.chunk_size
    pending = [row for row in rallies if row["rally_id"] not in existing][
        :manifest_limit
    ]
    local_teachers = bool(teachers) and all(
        role in teachers and teachers[role].local for role in TEACHER_ROLES
    )
    ox_teacher_available = (
        config.teacher_profile == OX_SINGLE_PROFILE
        and config.ox_enabled
        and config.ox_free_only
        and set(teachers or {}) == {OX_TEACHER_ROLE}
        and teachers is not None
        and teachers[OX_TEACHER_ROLE].local is False
    )
    teachers_available = local_teachers or ox_teacher_available
    if dry_run:
        return {
            "early": {
                "status": "dry_run",
                "pending": len(pending),
                "teachers_available": teachers_available,
            }
        }
    store.append_chain_batch(
        ledger_path,
        ({"kind": "rally-manifest", "manifest": rally} for rally in pending),
    )
    index_path = catalog.historical_index_path(root)
    index_sha256 = catalog.sync_historical_index(raw_dir, root)
    texts = catalog.CatalogTextCache(raw_dir, root)
    rally_by_id = {str(rally["rally_id"]): rally for rally in rallies}
    candidate_path = store.distillation_dir(root) / "candidate-ledger.jsonl"
    label_path = store.distillation_dir(root) / "label-ledger.jsonl"
    label_rows = _read_chain(label_path)
    capture = _capture_candidate_snapshots(
        root=root,
        raw_dir=raw_dir,
        config=config,
        rallies=rallies,
        texts=texts,
        index_path=index_path,
        label_rows=label_rows,
        cold_start=cold_start,
        deadline=deadline,
    )
    snapshots = capture.snapshots
    candidate_work = capture.work
    split_plan = capture.split_plan
    deadline_deferred = capture.deadline_deferred
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
        raw_dir=raw_dir,
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
    minimum_model_seconds = 220 if prefer_counterfactual else 130
    model_work_available = teachers_available or counterfactual_probe.pending
    model_deferred = deadline_deferred or (
        model_work_available
        and cold_start
        and deadline - time.monotonic() < minimum_model_seconds
    )
    return {
        "root": root,
        "raw_dir": raw_dir,
        "config": config,
        "deadline": deadline,
        "ox_profile_contract_id": ox_profile_contract_id,
        "scheduler_state": scheduler_state,
        "teacher_model_calls": teacher_model_calls,
        "counterfactual_model_calls": counterfactual_model_calls,
        "catalog": catalog,
        "rallies": rallies,
        "teachers": teachers,
        "counterfactual": counterfactual,
        "structural_verifier": structural_verifier,
        "ledger_path": ledger_path,
        "existing": existing,
        "pending": pending,
        "teachers_available": teachers_available,
        "ox_teacher_available": ox_teacher_available,
        "index_sha256": index_sha256,
        "texts": texts,
        "rally_by_id": rally_by_id,
        "candidate_path": candidate_path,
        "label_path": label_path,
        "label_rows": label_rows,
        "snapshots": snapshots,
        "candidate_work": candidate_work,
        "split_plan": split_plan,
        "model_snapshots": model_snapshots,
        "counterfactual_probe": counterfactual_probe,
        "prefer_counterfactual": prefer_counterfactual,
        "model_deferred": model_deferred,
        "cold_start": cold_start,
        "config_path": config_path,
    }


def _prepare_distillation_training(
    *,
    root: Path,
    raw_dir: Path,
    config_path: Path | None,
    config: DistillationConfig,
    catalog: Any,
    candidate_path: Path,
    label_path: Path,
    rallies: Sequence[Mapping[str, Any]],
    snapshots: Mapping[str, Mapping[str, Any]],
    split_plan: Mapping[str, Any],
    scheduler_state: Mapping[str, Any],
    existing: set[Any],
    pending: Sequence[Mapping[str, Any]],
    candidate_work: Sequence[Mapping[str, Any]],
    teachers_available: bool,
    model_deferred: bool,
    cold_start: bool,
    profile_contract_id: str,
) -> dict[str, Any]:
    current_labels = _read_chain(label_path)
    training_snapshots: Mapping[str, Mapping[str, Any]] = snapshots
    candidate_index_state: Mapping[str, Any] = {}
    if config.teacher_profile == OX_SINGLE_PROFILE:
        try:
            catalog.sync_candidate_index(root, candidate_path)
            candidate_index_state = catalog.candidate_index_state(root)
            training_snapshots = {
                rally_id: snapshot
                for rally_id, snapshot in catalog.read_candidate_snapshots(
                    root,
                    candidate_path,
                    {
                        str(row.get("rally_id") or "")
                        for row in current_labels
                        if isinstance(row.get("rally_id"), str) and row.get("rally_id")
                    },
                ).items()
            }
        except catalog.CatalogError as exc:
            raise DistillationError("OX training snapshots are unavailable") from exc
    training_snapshot = materialize_training_rows(
        root,
        _rallies=rallies,
        _snapshots=training_snapshots,
        _label_rows=current_labels,
    )
    if cold_start:
        _, current_cohort = _active_training_cohort(
            training_snapshot["rows"],
            teacher_profile=config.teacher_profile,
            profile_contract_id=(
                _current_ox_profile_contract_id(root)
                if config.teacher_profile == OX_SINGLE_PROFILE
                else ""
            ),
        )
        split_plan = _ensure_split_plan(
            root,
            rallies,
            raw_watermark=committed_raw_watermark(raw_dir),
            model_cohort_sha256=current_cohort["cohort_sha256"],
        )
        training_snapshot = materialize_training_rows(
            root,
            _rallies=rallies,
            _snapshots=training_snapshots,
            _label_rows=current_labels,
        )
    baseline = preflight(
        raw_dir=raw_dir,
        root=root,
        config_path=config_path,
        _rallies=rallies,
        _training_snapshot=training_snapshot,
        _profile_contract_id=profile_contract_id,
    )
    gate_baseline = _matching_p5_baseline(root, baseline) or baseline
    p5_allowed = bool(
        gate_baseline["hard_floor"]["p5_allowed"]
    ) and _has_canonical_hard_floors(config)
    manifest_backlog = max(0, len(rallies) - len(existing) - len(pending))
    candidate_backlog = max(
        0,
        len(rallies)
        - (
            int(candidate_index_state["record_count"])
            if candidate_index_state
            else len(snapshots)
        ),
    )
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
        local_teachers=teachers_available,
        model_deferred=model_deferred,
        gate_baseline=gate_baseline,
        promotion=promotion,
    )
    return {
        "current_labels": current_labels,
        "training_snapshots": training_snapshots,
        "candidate_index_state": candidate_index_state,
        "training_snapshot": training_snapshot,
        "split_plan": split_plan,
        "baseline": baseline,
        "gate_baseline": gate_baseline,
        "p5_allowed": p5_allowed,
        "manifest_backlog": manifest_backlog,
        "candidate_backlog": candidate_backlog,
        "cold_start_pending": cold_start_pending,
        "split_plan_id": split_plan_id,
        "bootstrap": bootstrap,
        "promotion": promotion,
        "rollout_evaluation": rollout_evaluation,
        "transition": transition,
    }


def _persist_distillation_chunk(
    *,
    setup: Mapping[str, Any],
    training: Mapping[str, Any],
    teacher_result: _TeacherBatchResult,
    ox_workset: Mapping[str, Any],
    local_workset: Mapping[str, Any],
    ox_ramp_fields: Mapping[str, Any],
    counterfactual_written: int,
    teacher_model_calls: int,
    counterfactual_model_calls: int,
    model_deferred: bool,
) -> dict[str, Any]:
    root = setup["root"]
    config = setup["config"]
    scheduler_state = setup["scheduler_state"]
    index_sha256 = setup["index_sha256"]
    ledger_path = setup["ledger_path"]
    candidate_path = setup["candidate_path"]
    pending = setup["pending"]
    candidate_work = setup["candidate_work"]
    teachers_available = setup["teachers_available"]
    ox_profile_contract_id = (
        teacher_result.profile_contract_id or setup["ox_profile_contract_id"]
    )
    counterfactual = setup["counterfactual"]
    baseline = training["baseline"]
    gate_baseline = training["gate_baseline"]
    transition = training["transition"]
    split_plan_id = training["split_plan_id"]
    bootstrap = training["bootstrap"]
    promotion = training["promotion"]
    rollout_evaluation = training["rollout_evaluation"]
    p5_allowed = training["p5_allowed"]
    cold_start_pending = training["cold_start_pending"]
    manifest_backlog = training["manifest_backlog"]
    candidate_backlog = training["candidate_backlog"]
    manifest_head = store.chain_head(ledger_path)["head_sha256"]
    candidate_head = store.chain_head(candidate_path)["head_sha256"]
    label_path = setup["label_path"]
    label_head = store.chain_head(label_path)["head_sha256"]
    ox_event_heads: Mapping[str, Mapping[str, Any]] = {}
    ox_quality_gate_id = ""
    ox_projection: Mapping[str, Any] = {}
    ox_source_binding: Mapping[str, str] = {}
    if config.teacher_profile == OX_SINGLE_PROFILE:
        ox_event_heads = _ox_event_heads(root)
        ox_source_binding = _ox_contract_source_binding(root, ox_profile_contract_id)
        if ox_profile_contract_id and not ox_source_binding:
            raise DistillationError("OX profile contract source binding is unavailable")
        authoritative_ox_gate = _offline_training_gate(
            training["training_snapshot"]["rows"], config, root=root
        )
        ox_projection = _ox_event_projection(
            root,
            profile_contract_id=ox_profile_contract_id,
            source_binding=ox_source_binding,
            workset=ox_workset,
            label_path=label_path,
            authoritative_gate=authoritative_ox_gate,
        )
        quality = ox_projection.get("quality_gates")
        quality = quality if isinstance(quality, Mapping) else {}
        reasons = quality.get("reasons")
        reasons = (
            sorted(str(reason) for reason in reasons)
            if isinstance(reasons, list)
            else ["quality_gate_unavailable"]
        )
        ox_quality_gate_id, _, _ = store.write_immutable(
            store.distillation_dir(root) / "ox-quality-gates",
            {
                "kind": "ox-quality-gate",
                "profile_contract_id": ox_profile_contract_id,
                **ox_source_binding,
                "label_head_sha256": label_head,
                "passed": quality.get("passed") is True,
                "reasons": reasons,
                "quality_gates": dict(quality),
                **ox_event_heads,
            },
            schema="chronovisor.recall-distill-ox-quality-gate.v1",
        )
    runtime_identity: Mapping[str, Any] = {}
    if config.teacher_profile == OX_SINGLE_PROFILE:
        runtime_identity = _r4_runtime_identity_projection(
            root,
            config_path=setup.get("config_path"),
            source_binding=ox_source_binding,
            profile_contract_id=ox_profile_contract_id,
            candidate_path=candidate_path,
            label_path=label_path,
        )
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
            "labels_written": teacher_result.labels_written,
            "ox_workset": ox_workset,
            "local_workset": local_workset,
            "ox_profile_contract_id": ox_profile_contract_id,
            "ox_profile_stopped": teacher_result.profile_stopped,
            "ox_quality_gate_id": ox_quality_gate_id,
            "profile_contract_id": ox_profile_contract_id,
            **ox_source_binding,
            **ox_projection,
            "runtime_identity": runtime_identity,
            **ox_event_heads,
            **ox_ramp_fields,
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
            "labels_written": teacher_result.labels_written,
            "ox_workset": ox_workset,
            "local_workset": local_workset,
            "ox_profile_contract_id": ox_profile_contract_id,
            "ox_profile_stopped": teacher_result.profile_stopped,
            "ox_quality_gate_id": ox_quality_gate_id,
            "profile_contract_id": ox_profile_contract_id,
            **ox_source_binding,
            **ox_projection,
            "runtime_identity": runtime_identity,
            **ox_event_heads,
            **ox_ramp_fields,
            "counterfactuals_written": counterfactual_written,
            "teacher_model_calls": teacher_model_calls,
            "counterfactual_model_calls": counterfactual_model_calls,
            "cold_start_pending": cold_start_pending,
            "cold_start_lane_turn": int(scheduler_state.get("cold_start_lane_turn", 0))
            + int(setup["cold_start"] and not model_deferred),
            "split_plan_id": split_plan_id,
            "manifest_backlog": manifest_backlog,
            "candidate_backlog": candidate_backlog,
            "promotion_status": promotion["status"],
            "promotion_reason": promotion.get("reason", ""),
            "incumbent_policy_id": bootstrap["artifact_id"],
            "rollout_evaluation_status": rollout_evaluation["status"],
            "hold_reason": transition["hold_reason"],
            "capture_only_reasons": gate_baseline["hard_floor"]["reasons"]
            + (
                []
                if teachers_available
                else [
                    "ox_teacher_unavailable"
                    if config.teacher_profile == OX_SINGLE_PROFILE
                    else "local_teachers_unavailable"
                ]
            ),
            "last_success_at": transition["last_success_at"],
            "error_code": (
                "ox_teacher_stopped"
                if teacher_result.profile_stopped
                else "worker_deferred"
                if model_deferred
                else ""
            ),
        },
    )
    return {
        "status": transition["worker_status"],
        "processed": len(pending),
        "p5_allowed": p5_allowed,
        "teachers_available": teachers_available,
        "counterfactual_available": bool(counterfactual and counterfactual.local),
        "candidate_snapshots": len(candidate_work),
        "labels_written": teacher_result.labels_written,
        "ox_workset": ox_workset,
        "local_workset": local_workset,
        "ox_profile_contract_id": ox_profile_contract_id,
        "ox_profile_stopped": teacher_result.profile_stopped,
        "ox_quality_gate_id": ox_quality_gate_id,
        "profile_contract_id": ox_profile_contract_id,
        **ox_source_binding,
        **ox_projection,
        "runtime_identity": runtime_identity,
        **ox_event_heads,
        **ox_ramp_fields,
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
    setup = _prepare_distillation_chunk(
        root=root,
        raw_dir=raw_dir,
        config_path=config_path,
        teachers=teachers,
        counterfactual=counterfactual,
        structural_verifier=structural_verifier,
        dry_run=dry_run,
        cold_start=cold_start,
        max_elapsed_seconds=max_elapsed_seconds,
    )
    if "early" in setup:
        return dict(setup["early"])
    root = setup["root"]
    raw_dir = setup["raw_dir"]
    config = setup["config"]
    deadline = setup["deadline"]
    ox_profile_contract_id = setup["ox_profile_contract_id"]
    scheduler_state = setup["scheduler_state"]
    teacher_model_calls = setup["teacher_model_calls"]
    counterfactual_model_calls = setup["counterfactual_model_calls"]
    catalog = setup["catalog"]
    rallies = setup["rallies"]
    teachers = setup["teachers"]
    counterfactual = setup["counterfactual"]
    structural_verifier = setup["structural_verifier"]
    existing = setup["existing"]
    pending = setup["pending"]
    teachers_available = setup["teachers_available"]
    ox_teacher_available = setup["ox_teacher_available"]
    texts = setup["texts"]
    rally_by_id = setup["rally_by_id"]
    candidate_path = setup["candidate_path"]
    label_path = setup["label_path"]
    label_rows = setup["label_rows"]
    snapshots = setup["snapshots"]
    candidate_work = setup["candidate_work"]
    split_plan = setup["split_plan"]
    model_snapshots = setup["model_snapshots"]
    counterfactual_probe = setup["counterfactual_probe"]
    prefer_counterfactual = setup["prefer_counterfactual"]
    cold_start = setup["cold_start"]
    config_path = setup["config_path"]
    model_deferred = setup["model_deferred"]
    teacher_result = _TeacherBatchResult()
    if teachers_available and not prefer_counterfactual and not model_deferred:
        teacher_result = _run_teacher_batch(
            root=root,
            raw_dir=raw_dir,
            config=config,
            teachers=teachers or {},
            snapshots=model_snapshots,
            rally_by_id=rally_by_id,
            texts=texts,
            label_path=label_path,
            label_rows=label_rows,
            candidate_indexed=ox_teacher_available,
            structural_verifier=structural_verifier,
        )
    labels_written = teacher_result.labels_written
    teacher_model_calls += teacher_result.model_calls
    model_deferred = model_deferred or teacher_result.deferred
    ox_workset = dict(teacher_result.workset_status or {})
    local_workset: dict[str, Any] = {}
    if config.teacher_profile == LOCAL_TRIAD_PROFILE:
        from chronovisor.recall.recall_distillation_workset import DistillationWorkset

        local_workset = DistillationWorkset(
            store.distillation_dir(root) / "local-workset.sqlite3"
        ).status(include_timing=True)
        ox_workset = {}
    ox_ramp_fields: dict[str, Any] = {}
    if config.teacher_profile == OX_SINGLE_PROFILE:
        ramp_source: Mapping[str, Any] = (
            scheduler_state
            if scheduler_state.get("kind") == "worker-state"
            and scheduler_state.get("ox_profile_contract_id") == ox_profile_contract_id
            and scheduler_state.get("ox_ramp_request_revision")
            == OX_RAMP_REQUEST_REVISION
            else {}
        )
        if (
            teacher_result.ramp_cap is not None
            or teacher_result.ramp_valid_receipts is not None
            or teacher_result.ramp_provider_attempts is not None
        ):
            ramp_source = {
                "ox_ramp_cap": teacher_result.ramp_cap,
                "ox_ramp_valid_receipts": teacher_result.ramp_valid_receipts,
                "ox_ramp_provider_attempts": teacher_result.ramp_provider_attempts,
                "ox_ramp_request_revision": OX_RAMP_REQUEST_REVISION,
            }
        (
            ox_ramp_cap,
            ox_ramp_valid_receipts,
            ox_ramp_provider_attempts,
        ) = _ox_ramp_state(ramp_source, config.teacher_max_inflight)
        ox_ramp_fields = {
            "ox_ramp_cap": ox_ramp_cap,
            "ox_ramp_valid_receipts": ox_ramp_valid_receipts,
            "ox_ramp_provider_attempts": ox_ramp_provider_attempts,
            "ox_ramp_request_revision": OX_RAMP_REQUEST_REVISION,
            **(
                dict(teacher_result.source_binding)
                if teacher_result.source_binding is not None
                else _ox_contract_source_binding(root, ox_profile_contract_id)
            ),
        }
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
            raw_dir=raw_dir,
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
    if config.teacher_profile == LOCAL_TRIAD_PROFILE:
        from chronovisor.recall.recall_distillation_workset import DistillationWorkset

        local_workset = DistillationWorkset(
            store.distillation_dir(root) / "local-workset.sqlite3"
        ).status(include_timing=True)
    training = _prepare_distillation_training(
        root=root,
        raw_dir=raw_dir,
        config_path=config_path,
        config=config,
        catalog=catalog,
        candidate_path=candidate_path,
        label_path=label_path,
        rallies=rallies,
        snapshots=snapshots,
        split_plan=split_plan,
        scheduler_state=scheduler_state,
        existing=existing,
        pending=pending,
        candidate_work=candidate_work,
        teachers_available=teachers_available,
        model_deferred=model_deferred,
        cold_start=cold_start,
        profile_contract_id=ox_profile_contract_id,
    )
    return _persist_distillation_chunk(
        setup=setup,
        training=training,
        teacher_result=teacher_result,
        ox_workset=ox_workset,
        local_workset=local_workset,
        ox_ramp_fields=ox_ramp_fields,
        counterfactual_written=counterfactual_written,
        teacher_model_calls=teacher_model_calls,
        counterfactual_model_calls=counterfactual_model_calls,
        model_deferred=model_deferred,
    )


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


def _timeout_workset_statuses(root: Path) -> dict[str, dict[str, Any]]:
    """Return only durable queue boundaries; timeout must not hide a bad queue."""

    from chronovisor.recall.recall_distillation_workset import DistillationWorkset
    from chronovisor.recall.recall_runtime import RecallWallClockTimeout

    statuses: dict[str, dict[str, Any]] = {}
    for name, filename in (
        ("ox_workset", "ox-workset.sqlite3"),
        ("local_workset", "local-workset.sqlite3"),
    ):
        path = store.distillation_dir(root) / filename
        try:
            if not path.exists():
                statuses[name] = {"observation": "missing"}
            else:
                statuses[name] = {
                    "observation": "available",
                    **DistillationWorkset(path).status(include_timing=True),
                }
        except RecallWallClockTimeout:
            statuses[name] = {"observation": "unavailable"}
        except Exception:
            statuses[name] = {"observation": "unavailable"}
    return statuses


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
                **_timeout_workset_statuses(resolved_root),
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
    migrate_parser = subcommands.add_parser("migrate-config")
    migrate_parser.add_argument("--config", type=Path)
    migrate_parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = preflight(
            raw_dir=args.raw_dir or args.root / "raw",
            root=args.root,
            config_path=args.config,
            runtime_commit=args.runtime_commit,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif args.command == "migrate-config":
        print(
            json.dumps(
                migrate_distillation_config(args.config, apply=args.apply),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
