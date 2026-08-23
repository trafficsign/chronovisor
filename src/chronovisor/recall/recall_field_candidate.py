"""Shadow/canary candidate lane for Stateful Recall Field."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from chronovisor.core import semantic_client
from chronovisor.core.durable_state import exclusive_text_file_lock
from chronovisor.core.runtime_config import load_search_embedding_config
from chronovisor.core.store import CHRONOVISOR_ROOT, find_page
from chronovisor.recall.recall_field_schema import (
    RecallFieldConfig,
    load_recall_field_config,
)

CANDIDATE_TRACE_FILE = (
    CHRONOVISOR_ROOT / "runtime" / "recall-field" / "candidate-trace.jsonl"
)
PROMOTION_ARTIFACT = CHRONOVISOR_ROOT / "runtime" / "recall-field" / "promotion.json"
MIN_PROMOTION_TRACES = 100
MIN_PROMOTION_SESSIONS = 20


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value.casefold()
    )


def _strict_utc_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        return None
    return parsed


def selected_for_canary(session_hash: str, config: RecallFieldConfig) -> bool:
    """Select sessions deterministically; never use raw session IDs."""

    if config.mode not in {"candidate", "active"} or not session_hash:
        return False
    percent = config.canary_percent
    if config.mode == "active" and percent <= 0:
        percent = 100
    if percent <= 0:
        return False
    bucket = (
        int(
            hashlib.sha256(f"recall-field:{session_hash}".encode()).hexdigest()[:8],
            16,
        )
        % 100
    )
    return bucket < percent


def effective_rollout(config: RecallFieldConfig) -> RecallFieldConfig:
    """Resolve autonomous rollout state without mutating the user config."""

    if not config.auto_promote or config.mode in {"off", "shadow"}:
        return config
    try:
        from dataclasses import replace

        from chronovisor.recall.recall_growth import automatic_rollout

        mode, percent = automatic_rollout(enabled=True)
        return replace(config, mode=mode, canary_percent=percent)
    except Exception:
        return config


def authority_allowed(path: Path | None = None) -> bool:
    """Require a sealed, non-degrading promotion artifact for Field authority."""

    artifact = path or PROMOTION_ARTIFACT
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 4
        or payload.get("status") != "passed"
    ):
        return False
    expected_sha = str(payload.get("snapshot_sha256") or "")
    unsigned = {
        key: value for key, value in payload.items() if key != "snapshot_sha256"
    }
    if expected_sha != _canonical_sha256(unsigned):
        return False
    metrics = payload.get("metrics")
    confidence = payload.get("confidence_evidence")
    answer = payload.get("answer_evaluation")
    locked_answer = payload.get("locked_answer_evaluation")
    retrieval = payload.get("retrieval_locked_e2e")
    artifact_set = payload.get("answer_artifact_set")
    generated_at = payload.get("generated_at")
    expires_at = payload.get("expires_at")
    generated_dt = _strict_utc_datetime(generated_at)
    expires_dt = _strict_utc_datetime(expires_at)
    now = datetime.now(UTC)
    fresh = bool(
        generated_dt is not None
        and expires_dt is not None
        and generated_dt < expires_dt
        and expires_dt - generated_dt <= timedelta(hours=24)
        and generated_dt <= now < expires_dt
    )
    if not fresh:
        return False

    from chronovisor.recall.recall_answer_eval import (
        builtin_field_environment_identity,
        validate_answer_artifact_set,
        validate_answer_outcome_artifact,
        validate_locked_answer_artifact,
    )
    from chronovisor.recall.recall_confidence import (
        cluster_rate_wilson_interval,
        manifest_sha256,
    )
    from chronovisor.recall.recall_growth import (
        _candidate_growth_metrics,
        _file_sha256,
        _validated_candidate_trace_rows,
        retrieval_locked_e2e_status,
    )

    current_environment = builtin_field_environment_identity()
    transition = payload.get("lkg_policy_transition")
    if (
        payload.get("environment_epoch") != current_environment
        or not isinstance(transition, dict)
        or transition.get("base_artifact_sha256")
        != current_environment.get("lkg_base_artifact_sha256")
        or transition.get("base_snapshot_sha256")
        != current_environment.get("lkg_base_snapshot_sha256")
        or transition.get("activated_artifact_sha256")
        != current_environment.get("lkg_base_artifact_sha256")
        or transition.get("evaluated_effective_config_sha256")
        != current_environment.get("effective_field_config_sha256")
        or transition.get("activated") is not False
        or not _valid_sha(transition.get("base_policy_sha256"))
        or not _valid_sha(transition.get("candidate_result_policy_sha256"))
    ):
        return False
    sources = payload.get("source_artifacts")
    candidate_source = payload.get("candidate_trace")
    if not isinstance(sources, dict) or not isinstance(candidate_source, dict):
        return False
    resolved: dict[str, Path] = {}
    for name, value in sources.items():
        # Recall/pull logs are diagnostic sources.  Used receipts must not be
        # consulted or freshness-gated by the Field authority check.
        if name in {"recall_log", "pull_log"}:
            continue
        if not isinstance(value, dict) or not str(value.get("path") or ""):
            return False
        source_path = Path(str(value["path"])).expanduser().resolve(strict=False)
        if _file_sha256(source_path) != value.get("file_sha256"):
            return False
        resolved[name] = source_path
    candidate_path = Path(str(candidate_source.get("path") or "")).expanduser().resolve(strict=False)
    candidate_rows, candidate_error = _validated_candidate_trace_rows(candidate_path)
    if candidate_error or not candidate_rows:
        return False
    promoted_head = str(candidate_source.get("head_sha256") or "")
    promoted_count = candidate_source.get("cumulative_eligible_trace_count")
    prefix_end = next(
        (
            index
            for index, row in enumerate(candidate_rows)
            if row.get("record_sha256") == promoted_head
        ),
        -1,
    )
    if (
        prefix_end < 0
        or candidate_rows[prefix_end].get("cumulative_eligible_trace_count")
        != promoted_count
    ):
        return False
    candidate_prefix = candidate_rows[: prefix_end + 1]
    live_candidate = _candidate_growth_metrics(candidate_prefix)
    manual_check = retrieval_locked_e2e_status(resolved["manual94"])
    train_check = validate_answer_outcome_artifact(
        resolved["train_answer"],
        required_split="train",
        episode_file=resolved["answer_episodes"],
        review_ledger_file=resolved["answer_reviews"],
        execution_ledger_file=resolved["answer_executions"],
        adapter_registry=resolved["answer_registry"],
    )
    locked_check = validate_locked_answer_artifact(
        resolved["locked_answer"],
        episode_file=resolved["answer_episodes"],
        review_ledger_file=resolved["answer_reviews"],
        execution_ledger_file=resolved["answer_executions"],
        adapter_registry=resolved["answer_registry"],
    )
    set_check = validate_answer_artifact_set(
        train=resolved["train_answer"],
        locked=resolved["locked_answer"],
        episode_file=resolved["answer_episodes"],
        review_ledger_file=resolved["answer_reviews"],
        execution_ledger_file=resolved["answer_executions"],
        adapter_registry=resolved["answer_registry"],
    )
    if not (
        manual_check.get("passed")
        and train_check.get("passed")
        and locked_check.get("passed")
        and set_check.get("passed")
        and manual_check.get("environment_epoch_sha256")
        == locked_check.get("environment_epoch_sha256")
        and retrieval == manual_check
        and answer == train_check
        and locked_answer == locked_check
        and artifact_set == set_check
    ):
        return False
    train_point = train_check.get("point")
    if isinstance(train_point, bool) or not isinstance(train_point, int | float):
        return False
    expected_live_metrics = {
        "stable_traces": int(live_candidate["quality_window_stable_traces"]),
        "stable_sessions": int(live_candidate["quality_window_stable_sessions"]),
        "coverage_evidence_traces": int(live_candidate["coverage_evidence_traces"]),
        "coverage_evidence_sessions": int(live_candidate["coverage_evidence_sessions"]),
        "commit_evidence_traces": int(live_candidate["commit_evidence_traces"]),
        "commit_evidence_sessions": int(live_candidate["commit_evidence_sessions"]),
        "paired_latency_traces": int(live_candidate["paired_latency_traces"]),
        "paired_latency_sessions": int(live_candidate["paired_latency_sessions"]),
        "validated_confidence_traces": int(
            live_candidate["validated_confidence_traces"]
        ),
        "incomplete_quality_traces": int(
            live_candidate["incomplete_quality_traces"]
        ),
        "teacher_commit_coverage": float(live_candidate["teacher_commit_coverage"]),
        "precision_delta_points": round(float(train_point) * 100.0, 6),
        "recall_delta_points": round(float(train_point) * 100.0, 6),
        "over_4s": int(live_candidate["over_4s"]),
        "fallback_rate": float(live_candidate["fallback_rate"]),
        "full_search_rate": float(live_candidate["full_search_rate"]),
    }
    if (
        metrics != expected_live_metrics
        or not isinstance(confidence, dict)
        or confidence.get("candidate") != live_candidate["confidence"]
    ):
        return False

    def confidence_summary_valid(
        bundle: object,
        thresholds: Mapping[str, float],
    ) -> bool:
        if not isinstance(bundle, dict):
            return False
        cases = bundle.get("cases")
        if not isinstance(cases, list) or not cases:
            return False
        for case in cases:
            if not isinstance(case, dict):
                return False
            unsigned_case = {key: value for key, value in case.items() if key != "case_sha256"}
            if case.get("case_sha256") != _canonical_sha256(unsigned_case):
                return False
        if bundle.get("manifest_sha256") != manifest_sha256(
            sorted(str(case["case_sha256"]) for case in cases)
        ) or bundle.get("samples") != len(cases):
            return False
        for key, threshold in thresholds.items():
            if bundle.get(key) != cluster_rate_wilson_interval(
                cases,
                value_key=key,
                success_threshold=threshold,
            ):
                return False
        cluster_counts = {
            int(bundle[key].get("clusters") or 0) for key in thresholds
        }
        return len(cluster_counts) == 1 and bundle.get("clusters") == next(iter(cluster_counts))

    def number(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    def confidence_bound(
        value: Any,
        *,
        methods: set[str],
        lower_floor: float,
    ) -> bool:
        if not isinstance(value, dict) or value.get("valid") is not True:
            return False
        point = number(value.get("point"))
        lower = number(value.get("lower"))
        upper = number(value.get("upper"))
        return bool(
            value.get("method") in methods
            and point is not None
            and lower is not None
            and upper is not None
            and lower <= point <= upper
            and lower >= lower_floor
        )

    state_path = artifact.parent / "growth-state.json"
    try:
        state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        not isinstance(state_payload, dict)
        or state_payload.get("generated_at") != generated_at
        or state_payload.get("snapshot_sha256")
        != payload.get("growth_state_snapshot_sha256")
        or state_payload.get("snapshot_sha256")
        != _canonical_sha256(
            {
                key: value
                for key, value in state_payload.items()
                if key != "snapshot_sha256"
            }
        )
    ):
        return False

    try:
        candidate_confidence = confidence["candidate"]
        answer_confidence = confidence["answer_reward"]
        precision_delta = number(metrics.get("precision_delta_points"))
        recall_delta = number(metrics.get("recall_delta_points"))
        teacher_commit = number(metrics.get("teacher_commit_coverage"))
        answer_point = number(answer_confidence.get("point"))
        answer_lower = number(answer_confidence.get("lower"))
        answer_upper = number(answer_confidence.get("upper"))
        answer_point_floor = number(answer_confidence.get("point_floor"))
        answer_lower_floor = number(answer_confidence.get("lower_floor"))
        expected_answer_confidence = {
            "valid": train_check.get("passed") is True,
            "method": str(train_check.get("method") or ""),
            "confidence": train_check.get("confidence"),
            "seed": train_check.get("seed"),
            "manifest_sha256": str(train_check.get("manifest_sha256") or ""),
            "samples": int(train_check.get("samples") or 0),
            "clusters": int(train_check.get("distinct_clusters") or 0),
            "point": train_check.get("point"),
            "lower": train_check.get("lower"),
            "upper": train_check.get("upper"),
            "point_floor": 0.02,
            "lower_floor": 0.0,
        }
        return bool(
            isinstance(metrics, dict)
            and isinstance(confidence, dict)
            and confidence_summary_valid(
                candidate_confidence,
                {
                    "teacher_coverage": 0.99,
                    "field_precision": 0.90,
                    "commit_coverage": 0.99,
                },
            )
            and answer_confidence == expected_answer_confidence
            and isinstance(answer, dict)
            and answer.get("passed") is True
            and isinstance(locked_answer, dict)
            and locked_answer.get("passed") is True
            and isinstance(retrieval, dict)
            and retrieval.get("passed") is True
            and retrieval.get("examples") == 94
            and len(str(retrieval.get("manifest_sha256") or "")) == 64
            and isinstance(artifact_set, dict)
            and artifact_set.get("passed") is True
            and len(str(artifact_set.get("split_manifest_sha256") or "")) == 64
            and artifact_set.get("split_manifest_sha256")
            == answer.get("split_manifest_sha256")
            == locked_answer.get("split_manifest_sha256")
            and int(metrics.get("stable_traces") or 0) >= MIN_PROMOTION_TRACES
            and int(metrics.get("stable_sessions") or 0) >= MIN_PROMOTION_SESSIONS
            and int(metrics.get("coverage_evidence_traces") or 0)
            >= MIN_PROMOTION_TRACES
            and int(metrics.get("coverage_evidence_sessions") or 0)
            >= MIN_PROMOTION_SESSIONS
            and int(metrics.get("commit_evidence_traces") or 0) >= MIN_PROMOTION_TRACES
            and int(metrics.get("commit_evidence_sessions") or 0)
            >= MIN_PROMOTION_SESSIONS
            and int(metrics.get("paired_latency_traces") or 0) >= MIN_PROMOTION_TRACES
            and int(metrics.get("paired_latency_sessions") or 0)
            >= MIN_PROMOTION_SESSIONS
            and int(metrics.get("incomplete_quality_traces") or 0) == 0
            and int(metrics.get("validated_confidence_traces") or 0)
            == promoted_count
            and teacher_commit is not None
            and teacher_commit >= 0.99
            and precision_delta is not None
            and precision_delta >= -1.0
            and recall_delta is not None
            and recall_delta >= -1.0
            and int(metrics.get("over_4s") or 0) == 0
            and confidence_bound(
                candidate_confidence["teacher_coverage"],
                methods={"connected-cluster-wilson-score"},
                lower_floor=0.95,
            )
            and confidence_bound(
                candidate_confidence["commit_coverage"],
                methods={"connected-cluster-wilson-score"},
                lower_floor=0.95,
            )
            and confidence_bound(
                candidate_confidence["field_precision"],
                methods={"connected-cluster-wilson-score"},
                lower_floor=0.85,
            )
            and int(candidate_confidence.get("samples") or 0) >= MIN_PROMOTION_TRACES
            and int(candidate_confidence.get("clusters") or 0)
            >= MIN_PROMOTION_SESSIONS
            and answer_confidence.get("valid") is True
            and answer_confidence.get("method")
            == "connected-cluster-bootstrap-percentile"
            and answer_confidence.get("confidence") == 0.95
            and answer_confidence.get("seed") == 1729
            and int(answer_confidence.get("samples") or 0) >= 20
            and int(answer_confidence.get("clusters") or 0) >= 20
            and answer_point is not None
            and answer_lower is not None
            and answer_upper is not None
            and answer_point_floor is not None
            and answer_lower_floor is not None
            and answer_lower <= answer_point <= answer_upper
            and answer_point >= answer_point_floor
            and answer_lower >= answer_lower_floor
            and all(
                len(str(item.get("manifest_sha256") or "")) == 64
                for item in (
                    candidate_confidence,
                    answer_confidence,
                )
            )
            and expected_sha
        )
    except (KeyError, TypeError, ValueError):
        return False


def _verify(
    query: str,
    page_ids: list[str],
    *,
    timeout_ms: int,
) -> tuple[list[Any], dict[str, Any]]:
    started = time.perf_counter()
    if not page_ids:
        return [], {"status": "fallback", "reason": "empty_field"}
    config = load_search_embedding_config()
    try:
        results = semantic_client.verify(
            query,
            page_ids[:30],
            config=config,
            timeout_ms=max(25, timeout_ms),
        )
    except Exception as exc:
        return [], {
            "status": "fallback",
            "reason": type(exc).__name__,
            "latency_ms": int(round((time.perf_counter() - started) * 1_000)),
        }
    return results, {
        "status": "verified",
        "candidate_count": len(page_ids[:30]),
        "verified_count": len(results),
        "latency_ms": int(round((time.perf_counter() - started) * 1_000)),
    }


def run_candidate_teacher_pair(
    *,
    query: str,
    field_turn: dict[str, Any],
    teacher_search: Callable[[], tuple[list[Any], str]],
    timeout_ms: int,
    config: RecallFieldConfig | None = None,
    certificate_boundary_enabled: bool = True,
) -> tuple[list[Any], str, dict[str, Any]]:
    """Run the authoritative teacher, then verify Field within the remaining budget.

    The returned ranking remains teacher-owned unless a sealed promotion
    artifact exists. Candidate mode therefore cannot bypass certificates.
    """

    cfg = effective_rollout(config or load_recall_field_config())
    session = str(field_turn.get("session_hash") or "")
    page_ids = [
        str(page_id)
        for page_id in field_turn.get("candidate_page_ids", [])
        if isinstance(page_id, str) and page_id
    ][:30]
    fallback = field_turn.get("full_search_fallback") is True
    selected = selected_for_canary(session, cfg)
    if not selected or fallback or not page_ids:
        teacher_started = time.perf_counter()
        results, mode = teacher_search()
        teacher_latency_ms = int(round((time.perf_counter() - teacher_started) * 1_000))
        teacher_ids = [str(row.page_id) for row in results[:30]]
        return (
            results,
            mode,
            {
                "status": "fallback",
                "reason": (
                    "not_selected"
                    if not selected
                    else "topic_reset"
                    if fallback
                    else "empty_field"
                ),
                "teacher_count": len(results),
                "teacher_page_ids": teacher_ids,
                "field_page_ids": page_ids,
                "missed_page_ids": [
                    page_id for page_id in teacher_ids if page_id not in set(page_ids)
                ],
                "teacher_top30_coverage": round(
                    len(set(teacher_ids) & set(page_ids))
                    / max(1, len(set(teacher_ids))),
                    6,
                ),
                "full_search_required": True,
                "quality_eligible": False,
                "field_attempted": False,
                "field_verified": False,
                "field_latency_ms": None,
                "teacher_latency_ms": teacher_latency_ms,
                "authority": "teacher",
            },
        )

    pair_started = time.perf_counter()
    teacher_results, teacher_mode = teacher_search()
    teacher_elapsed_ms = (time.perf_counter() - pair_started) * 1_000
    teacher_latency_ms = int(round(teacher_elapsed_ms))
    remaining_ms = max(0, int(timeout_ms - teacher_elapsed_ms))
    field_attempted = remaining_ms >= 25
    if field_attempted:
        verified, verify_meta = _verify(
            query,
            page_ids,
            timeout_ms=remaining_ms,
        )
    else:
        verified, verify_meta = (
            [],
            {
                "status": "fallback",
                "reason": "budget_exhausted",
            },
        )

    teacher_ids = [str(row.page_id) for row in teacher_results[:30]]
    field_ids = [str(row.page_id) for row in verified[:30]]
    overlap = len(set(teacher_ids) & set(field_ids))
    coverage = overlap / max(1, len(set(teacher_ids)))
    verify_completed = verify_meta.get("status") == "verified"
    if not verify_completed or not verified:
        fallback_reason = (
            "empty_verified_field"
            if verify_completed
            else str(verify_meta.get("reason") or "field_verification_failed")
        )
        return (
            teacher_results,
            teacher_mode,
            {
                **verify_meta,
                "status": "fallback",
                "reason": fallback_reason,
                "authority": "teacher",
                "field_page_ids": field_ids,
                "teacher_page_ids": teacher_ids,
                "teacher_top30_coverage": round(coverage, 6),
                "missed_page_ids": [
                    page_id for page_id in teacher_ids if page_id not in set(field_ids)
                ],
                "rollback": cfg.mode == "active",
                "rollback_reason": (
                    "field_verification_empty"
                    if verify_completed
                    else "field_verification_failed"
                ),
                "full_search_required": True,
                "quality_eligible": field_attempted,
                "field_attempted": field_attempted,
                "field_verified": verify_completed,
                "field_latency_ms": (
                    int(verify_meta.get("latency_ms") or 0) if field_attempted else None
                ),
                "teacher_latency_ms": teacher_latency_ms,
                "effective_mode": "shadow",
            },
        )
    promoted = bool(
        cfg.mode == "active" and certificate_boundary_enabled and authority_allowed()
    )
    return (
        verified if promoted else teacher_results,
        "field-active" if promoted else teacher_mode,
        {
            **verify_meta,
            "status": "active" if promoted else "observed",
            "authority": "field" if promoted else "teacher",
            "field_page_ids": field_ids,
            "teacher_page_ids": teacher_ids,
            "teacher_top30_coverage": round(coverage, 6),
            "missed_page_ids": [
                page_id for page_id in teacher_ids if page_id not in set(field_ids)
            ],
            "rollback": bool(cfg.mode == "active" and not promoted),
            "rollback_reason": (
                "certificate_boundary_disabled"
                if cfg.mode == "active" and not certificate_boundary_enabled
                else "promotion_artifact_missing_or_failed"
                if cfg.mode == "active" and not promoted
                else ""
            ),
            "full_search_required": False,
            "quality_eligible": True,
            "field_attempted": True,
            "field_verified": True,
            "field_latency_ms": int(verify_meta.get("latency_ms") or 0),
            "teacher_latency_ms": teacher_latency_ms,
            "effective_mode": "active" if promoted else "shadow",
        },
    )


def append_candidate_trace(
    *,
    session_hash: str,
    prompt: str,
    observer: dict[str, Any],
    committed_page_ids: list[str],
    latency_ms: int,
    path: Path = CANDIDATE_TRACE_FILE,
) -> dict[str, Any]:
    """Persist privacy-safe Field/teacher disagreement evidence."""

    field_ids = [
        str(value)
        for value in observer.get("field_page_ids", [])
        if isinstance(value, str)
    ]
    committed = list(
        dict.fromkeys(page_id for page_id in committed_page_ids if page_id)
    )
    commit_coverage = (
        len(set(field_ids) & set(committed)) / len(committed) if committed else 1.0
    )
    all_pages = list(
        dict.fromkeys(
            [
                *field_ids,
                *[
                    str(value)
                    for value in observer.get("teacher_page_ids", [])
                    if isinstance(value, str)
                ],
                *committed,
            ]
        )
    )
    content_hashes: dict[str, str] = {}
    page_uids: dict[str, str] = {}
    from chronovisor.recall.recall_runtime import page_uid_for_id

    for page_id in all_pages:
        path_value = find_page(page_id)
        try:
            digest = (
                hashlib.sha256(path_value.read_bytes()).hexdigest()
                if path_value is not None
                else ""
            )
        except OSError:
            digest = ""
        if digest:
            content_hashes[page_id] = digest
            uid = page_uid_for_id(page_id)
            if uid:
                page_uids[page_id] = uid
    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    record = {
        "schema_version": 3,
        "ts_epoch": round(time.time(), 3),
        "session_hash": session_hash,
        "prompt_sha256": prompt_sha,
        "query_sha256": prompt_sha,
        "content_sha256": _canonical_sha256(content_hashes) if content_hashes else "",
        "page_content_sha256": content_hashes,
        "page_uids": page_uids,
        "cluster_nodes": list(
            dict.fromkeys(
                [f"session:{session_hash}", f"query:{prompt_sha}"]
                + [f"page:{page_id}" for page_id in sorted(all_pages)]
                + [f"uid:{page_uids[page_id]}" for page_id in sorted(all_pages) if page_id in page_uids]
                + [
                    f"content:{content_hashes[page_id]}"
                    for page_id in sorted(all_pages)
                    if page_id in content_hashes
                ]
            )
        ),
        "status": str(observer.get("status") or ""),
        "authority": str(observer.get("authority") or "teacher"),
        "fallback_reason": str(observer.get("reason") or ""),
        "full_search_required": observer.get("full_search_required") is True,
        "quality_eligible": observer.get("quality_eligible") is True,
        "field_attempted": observer.get("field_attempted") is True,
        "field_verified": observer.get("field_verified") is True,
        "teacher_top30_coverage": float(observer.get("teacher_top30_coverage") or 0.0),
        "teacher_commit_coverage": round(commit_coverage, 6),
        "field_page_ids": field_ids,
        "teacher_page_ids": list(observer.get("teacher_page_ids") or []),
        "committed_page_ids": committed,
        "missed_page_ids": list(observer.get("missed_page_ids") or []),
        "latency_ms": max(0, int(latency_ms)),
        "field_latency_ms": (
            max(0, int(observer["field_latency_ms"]))
            if isinstance(observer.get("field_latency_ms"), int | float)
            else None
        ),
        "teacher_latency_ms": (
            max(0, int(observer["teacher_latency_ms"]))
            if isinstance(observer.get("teacher_latency_ms"), int | float)
            else None
        ),
        "over_4s": latency_ms > 4_000,
        "rollback": observer.get("rollback") is True,
        "rollback_reason": str(observer.get("rollback_reason") or ""),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    from chronovisor.recall.recall_growth import _candidate_confidence_case

    with exclusive_text_file_lock(lock_path):
        previous_sha = "0" * 64
        cumulative_eligible = 0
        try:
            previous_rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            previous = previous_rows[-1] if previous_rows else {}
            if isinstance(previous, dict):
                previous_sha = str(previous.get("record_sha256") or previous_sha)
                cumulative_eligible = int(
                    previous.get("cumulative_eligible_trace_count") or 0
                )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            previous_sha = "0" * 64
            cumulative_eligible = 0
        record["previous_record_sha256"] = previous_sha
        record["cumulative_eligible_trace_count"] = cumulative_eligible + int(
            _candidate_confidence_case(record) is not None
        )
        record["record_sha256"] = _canonical_sha256(record)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.chmod(path, 0o600)
    return record
