"""Knowledge health KPIs for dashboard and CLI."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from chronovisor.core.index_store import canonical_document_paths, get_store
from chronovisor.core.jsonl import count_jsonl, iter_jsonl, read_jsonl
from chronovisor.core.store import CHRONOVISOR_ROOT, RAW_DIR
from chronovisor.decision.decision_authority import semantic_authority_shape_error
from chronovisor.decision.semantic_hold import (
    LOCAL_SEMANTIC_NO_QUORUM,
    canonical_sha256,
    persisted_semantic_no_quorum_hold,
)

INGEST_LIVENESS_TTL_SECONDS = 5 * 60
DISTILLATION_PROGRESS_STALE_SECONDS = 24 * 60 * 60
DISTILLATION_EXPIRY_WARNING_SECONDS = 7 * 24 * 60 * 60
NATIVE_LUNA_SUPERVISOR_SCHEMA = "chronovisor.recall-luna-supervisor.v1"
NATIVE_LUNA_SUPERVISOR_HEARTBEAT_STALE_SECONDS = 15 * 60
NATIVE_LUNA_SUPERVISOR_PROGRESS_STALE_SECONDS = 30 * 60
_NATIVE_LUNA_SUPERVISOR_STATUSES = {
    "starting",
    "running",
    "retrying",
    "idle",
    "disabled",
}
_NATIVE_LUNA_SUPERVISOR_UNSAFE_KEYS = {
    "child_pid",
    "error",
    "last_error_code",
    "last_exit_code",
    "path",
    "pid",
    "reason",
    "updated_at",
}


def _jsonl_count(path: Path) -> int:
    return count_jsonl(path)


def _read_jsonl(path: Path, *, limit: int = 500) -> list[dict[str, Any]]:
    return read_jsonl(path, limit=limit)


def _recall_distillation_fallback(
    status: str, *, alert: bool, error: str = ""
) -> dict[str, Any]:
    return {
        "status": status,
        "worker_status": status,
        "rollout_status": status,
        "rollout_percent": 0.0,
        "active_policy_id": None,
        "candidate_policy_id": None,
        "lkg_policy_id": None,
        "feature_revision": "",
        "teacher_only": 0,
        "verified_truth": 0,
        "probe_not_truth": 0,
        "paired_denominator": 0,
        "hold_reason": "",
        "alert": alert,
        **({"error": error} if error else {}),
    }


def recall_distillation_kpi(root: Path | None = None) -> dict[str, Any]:
    """Expose the distillation worker's public, privacy-safe status.

    The distillation snapshot owns pointer validation. The health classifier
    adds actionable stop reasons without reading private label text.
    """

    try:
        from chronovisor.recall.recall_distillation_store import snapshot
    except ImportError:
        return _recall_distillation_fallback("unavailable", alert=False)
    try:
        value = snapshot(root or CHRONOVISOR_ROOT)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return _recall_distillation_fallback("unavailable", alert=False)
    if not isinstance(value, dict):
        return _recall_distillation_fallback(
            "invalid", alert=True, error="invalid_snapshot"
        )
    status = str(value.get("status") or "invalid")
    state = value.get("worker_status", value.get("state"))
    worker_status = str(
        state.get("status") if isinstance(state, dict) else state or status
    )
    rollout_status = str(value.get("rollout_status") or "")
    rollout = value.get("rollout")
    rollout_percent = float(rollout) if isinstance(rollout, (int, float)) else 0.0
    return {
        **value,
        "worker_status": worker_status,
        "rollout_status": rollout_status,
        "rollout_percent": rollout_percent,
        "alert": status in {"stale", "tampered", "invalid", "error"},
    }


def _distillation_config(root: Path) -> dict[str, Any]:
    """Read only the public distillation switches needed for health alerts."""

    from chronovisor.core import runtime_config

    config_path = root / "config.toml"
    if not config_path.exists():
        config_path = runtime_config.CONFIG_FILE
    data = runtime_config.load_toml_file(config_path)
    recall = data.get("recall") if isinstance(data, dict) else None
    table = recall.get("distillation") if isinstance(recall, dict) else None
    table = table if isinstance(table, dict) else {}
    enabled_value = table.get("enabled")
    enabled = enabled_value is True
    override = os.environ.get("CHRONOVISOR_RECALL_DISTILLATION")
    if override is not None:
        normalized = override.strip().lower()
        if normalized in {"0", "false", "no", "off"}:
            enabled = False
        elif normalized in {"1", "true", "yes", "on"}:
            enabled = True
        else:
            return {
                "enabled": enabled,
                "capture_enabled": table.get("capture_enabled") is True,
                "teacher_profile": str(table.get("teacher_profile") or ""),
                "native_luna_enabled": table.get("native_luna_enabled") is True,
                "ox_enabled": table.get("ox_enabled") is True,
                "ox_expires_at": table.get("ox_expires_at"),
                "config_error": "invalid_enabled_override",
            }
    return {
        "enabled": enabled,
        "capture_enabled": table.get("capture_enabled") is True,
        "teacher_profile": str(table.get("teacher_profile") or ""),
        "native_luna_enabled": table.get("native_luna_enabled") is True,
        "ox_enabled": table.get("ox_enabled") is True,
        "ox_expires_at": table.get("ox_expires_at"),
        "config_error": "" if isinstance(enabled_value, (bool, type(None))) else "invalid_enabled",
    }


def _distillation_state(root: Path) -> dict[str, Any]:
    """Read sealed state metadata without exposing private ledgers or text."""

    try:
        from chronovisor.recall import recall_distillation_store as store

        return store.read_sealed(
            store.distillation_dir(root) / store.STATE_FILE,
            schema=store.DISTILLATION_SCHEMA,
        )
    except Exception:
        return {}


def _parse_utc_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _native_luna_supervisor_counts(value: object) -> dict[str, Any] | None:
    """Keep supervisor bundle metadata to non-negative integer counts only."""

    if not isinstance(value, dict):
        return None
    counts: dict[str, Any] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key
            or key.lower() in _NATIVE_LUNA_SUPERVISOR_UNSAFE_KEYS
        ):
            return None
        if isinstance(item, bool):
            return None
        if isinstance(item, int):
            if item < 0:
                return None
            counts[key] = item
            continue
        nested = _native_luna_supervisor_counts(item)
        if nested is None:
            return None
        counts[key] = nested
    return counts


def _native_luna_supervisor_health(
    root: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    """Read the native Luna supervisor state without exposing private fields."""

    path = root / "runtime" / "recall-luna-bulk" / "supervisor-status.json"
    if not path.exists():
        return None, []

    invalid = {
        "status": "invalid",
        "completed_total": 0,
        "pending_total": 0,
        "bundles": {},
        "last_progress_at": None,
    }
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return invalid, [
            {
                "category": "human_wait",
                "reason": "executor_state_invalid",
                "message": "人の判断待ち: native Luna executor の状態を確認できません",
            }
        ]
    if not isinstance(raw, dict) or raw.get("schema") != NATIVE_LUNA_SUPERVISOR_SCHEMA:
        return invalid, [
            {
                "category": "human_wait",
                "reason": "executor_state_invalid",
                "message": "人の判断待ち: native Luna executor の状態を確認できません",
            }
        ]

    status = raw.get("status")
    updated_at = _parse_utc_timestamp(raw.get("updated_at"))
    last_progress_value = raw.get("last_progress_at")
    last_progress_at = (
        _parse_utc_timestamp(last_progress_value)
        if isinstance(last_progress_value, str) and last_progress_value
        else None
    )
    if (
        last_progress_value is not None
        and last_progress_value != ""
        and last_progress_at is None
    ):
        return invalid, [
            {
                "category": "human_wait",
                "reason": "executor_state_invalid",
                "message": "人の判断待ち: native Luna executor の状態を確認できません",
            }
        ]
    completed_total = raw.get("completed_total")
    pending_total = raw.get("pending_total")
    bundles = _native_luna_supervisor_counts(raw.get("bundles", {}))
    if (
        not isinstance(status, str)
        or status not in _NATIVE_LUNA_SUPERVISOR_STATUSES
        or updated_at is None
        or isinstance(completed_total, bool)
        or not isinstance(completed_total, int)
        or completed_total < 0
        or isinstance(pending_total, bool)
        or not isinstance(pending_total, int)
        or pending_total < 0
        or bundles is None
    ):
        return invalid, [
            {
                "category": "human_wait",
                "reason": "executor_state_invalid",
                "message": "人の判断待ち: native Luna executor の状態を確認できません",
            }
        ]

    diagnostic = {
        "status": status,
        "completed_total": completed_total,
        "pending_total": pending_total,
        "bundles": bundles,
        "last_progress_at": last_progress_at.isoformat()
        if last_progress_at is not None
        else None,
    }
    alerts: list[dict[str, str]] = []

    def add(category: str, reason: str, message: str) -> None:
        alerts.append({"category": category, "reason": reason, "message": message})

    if status == "retrying" and bool(str(raw.get("last_error_code") or "").strip()):
        add(
            "human_wait",
            "native_luna_executor_error",
            "人の判断待ち: native Luna executor が retrying です",
        )
    if status == "disabled":
        add(
            "human_wait",
            "executor_disabled",
            "人の判断待ち: native Luna executor が無効です",
        )

    now = datetime.now(UTC)
    heartbeat_age = max(0, int((now - updated_at).total_seconds()))
    progress_age = (
        max(0, int((now - last_progress_at).total_seconds()))
        if last_progress_at is not None
        else None
    )
    stalled = status != "disabled" and (
        heartbeat_age > NATIVE_LUNA_SUPERVISOR_HEARTBEAT_STALE_SECONDS
        or (
            pending_total > 0
            and progress_age is not None
            and progress_age > NATIVE_LUNA_SUPERVISOR_PROGRESS_STALE_SECONDS
        )
    )
    if stalled:
        add(
            "data_wait",
            "native_luna_executor_stalled",
            "データ待ち: native Luna executor の進捗が stale です",
        )
    return diagnostic, alerts


def _distillation_contract_expiry(root: Path, config: dict[str, Any]) -> tuple[str, bool]:
    """Return a contract expiry and whether an expected sealed contract exists."""

    expiry = config.get("ox_expires_at")
    teacher_profile = str(config.get("teacher_profile") or "")
    native_profile = teacher_profile == "native-luna-single-v1"
    native_required = native_profile or config.get("native_luna_enabled") is True
    ox_required = bool(config.get("ox_enabled")) or expiry not in {None, ""}
    if native_required:
        contract_path = (
            root
            / "runtime"
            / "recall-distillation"
            / "native-luna-profile-contract.json"
        )
    elif ox_required:
        contract_path = (
            root / "runtime" / "recall-distillation" / "ox-profile-contract.json"
        )
    else:
        return str(expiry or ""), False
    if not contract_path.exists():
        return str(expiry or ""), False
    if native_required:
        # The native lane owns strict pointer, immutable-artifact, and closed
        # identity validation in its small contract module.  Do not accept the
        # generic legacy sealed-state reader here: a native contract must prove
        # its fixed profile identity and immutable artifact binding.
        try:
            from chronovisor.recall import recall_distillation_native as native
            contract = native.read_contract(root)
        except Exception:
            return "", False
        # ``read_contract`` fails closed with an empty mapping for a missing,
        # mismatched, or tampered pointer.  Do not mistake that sentinel for a
        # valid no-expiry contract.
        if not isinstance(contract, dict) or not contract:
            return "", False
        contract_expiry = contract.get("expires_at")
        if isinstance(contract_expiry, str):
            expiry = contract_expiry
        return str(expiry or ""), True
    try:
        from chronovisor.recall import recall_distillation_store as store

        contract = store.read_sealed(
            contract_path,
            schema=store.DISTILLATION_SCHEMA,
        )
        contract_expiry = contract.get("expires_at")
        # An empty expires_at explicitly means that the contract has no expiry.
        if isinstance(contract_expiry, str):
            expiry = contract_expiry
    except Exception:
        return "", False
    return str(expiry or ""), True


def _distillation_alerts(
    snapshot: dict[str, Any],
    *,
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Classify a stopped worker without weakening any distillation gate."""

    config = _distillation_config(root)
    state = _distillation_state(root)
    alerts: list[dict[str, str]] = []

    def add(category: str, reason: str, message: str) -> None:
        if not any(item["reason"] == reason for item in alerts):
            alerts.append(
                {"category": category, "reason": reason, "message": message}
            )

    if config.get("config_error"):
        add(
            "human_wait",
            str(config["config_error"]),
            "人の判断待ち: 蒸留設定を確認してください",
        )

    enabled = bool(config.get("enabled"))
    capture_enabled = bool(config.get("capture_enabled"))
    if not enabled and capture_enabled:
        add(
            "human_wait",
            "enabled_false",
            "人の判断待ち: 蒸留が無効です (enabled=false)",
        )

    teacher_profile = str(config.get("teacher_profile") or "")
    native_profile = teacher_profile == "native-luna-single-v1"
    native_luna_enabled = config.get("native_luna_enabled") is True
    native_luna_supervisor: dict[str, Any] | None = None
    if enabled and native_profile:
        native_luna_supervisor, supervisor_alerts = _native_luna_supervisor_health(root)
        alerts.extend(supervisor_alerts)
    if enabled and not teacher_profile:
        add(
            "human_wait",
            "teacher_config_missing",
            "人の判断待ち: 教師設定がありません",
        )
    if enabled and native_luna_enabled and not native_profile:
        add(
            "human_wait",
            "native_luna_profile_mismatch",
            "人の判断待ち: native Luna の教師 profile が一致しません",
        )
    if enabled and native_profile and not native_luna_enabled:
        add(
            "human_wait",
            "native_luna_disabled",
            "人の判断待ち: native Luna 教師が無効です (native_luna_enabled=false)",
        )

    error_code = str(snapshot.get("error_code") or state.get("error_code") or "")
    worker_status = str(
        snapshot.get("worker_status")
        or snapshot.get("status")
        or state.get("worker_status")
        or state.get("status")
        or ""
    )
    state_reasons: list[object] = []
    for source in (snapshot, state):
        reasons = source.get("capture_only_reasons")
        if isinstance(reasons, list):
            state_reasons.extend(reasons)
    state_reason_set = {str(reason) for reason in state_reasons}
    teacher_stopped_reasons = {
        "ox_teacher_stopped",
        "teacher_stopped",
        "native_luna_stopped",
    }
    teacher_missing_reasons = {
        "local_teachers_unavailable",
        "ox_teacher_unavailable",
        "teacher_unavailable",
        "teacher_missing",
        "native_luna_unavailable",
        "native_luna_teacher_unavailable",
    }
    contract_missing_reasons = {
        "profile_contract_unavailable",
    }
    contract_expired_reasons = {"profile_contract_expired"}
    if error_code in teacher_stopped_reasons or state_reason_set & teacher_stopped_reasons:
        add(
            "human_wait",
            "teacher_stopped",
            "人の判断待ち: 教師モデルが停止しています",
        )
    if error_code in teacher_missing_reasons or state_reason_set & teacher_missing_reasons:
        add(
            "human_wait",
            "teacher_missing",
            "人の判断待ち: 教師が利用できません",
        )
    if error_code in contract_missing_reasons or state_reason_set & contract_missing_reasons:
        add(
            "human_wait",
            "teacher_contract_missing",
            "人の判断待ち: 教師の契約がありません",
        )
    if error_code in contract_expired_reasons or state_reason_set & contract_expired_reasons:
        add(
            "human_wait",
            "teacher_contract_expired",
            "人の判断待ち: 教師の契約が期限切れです",
        )
    if snapshot.get("status") in {"unavailable", "tampered", "invalid", "error"}:
        add(
            "human_wait",
            "distillation_state_unavailable",
            "人の判断待ち: 蒸留の状態を確認できません",
        )

    expiry, contract_present = _distillation_contract_expiry(root, config)
    contract_required = (
        native_profile
        or native_luna_enabled
        or bool(config.get("ox_enabled"))
        or expiry not in {None, ""}
    )
    if enabled and contract_required and not contract_present:
        add(
            "human_wait",
            "teacher_contract_missing",
            "人の判断待ち: 教師の契約がありません",
        )
    expires_at = _parse_utc_timestamp(expiry)
    if expires_at is not None:
        remaining = (expires_at - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            add(
                "human_wait",
                "teacher_contract_expired",
                "人の判断待ち: 教師の契約が期限切れです",
            )
        elif remaining <= DISTILLATION_EXPIRY_WARNING_SECONDS:
            add(
                "human_wait",
                "teacher_contract_expiry_warning",
                "人の判断待ち: 教師の契約がまもなく期限切れです",
            )

    hold_reason = str(snapshot.get("hold_reason") or state.get("hold_reason") or "")
    if enabled and hold_reason and not alerts:
        add(
            "data_wait",
            hold_reason,
            f"データ待ち: 蒸留の受入条件が未達です ({hold_reason})",
        )

    progress_values: list[dict[str, Any]] = []
    for source in (snapshot, state):
        progress = source.get("progress")
        if isinstance(progress, dict):
            progress_values.append(progress)
    last_label_value: object = snapshot.get("last_label_at") or state.get(
        "last_label_at"
    )
    if not last_label_value:
        for progress in progress_values:
            last_label_value = progress.get("last_label_at")
            if last_label_value:
                break
    last_label_at = _parse_utc_timestamp(last_label_value)
    last_success_value = snapshot.get("last_success_at") or state.get(
        "last_success_at"
    )
    # A production profile's last_success_at can advance for capture-only or
    # maintenance work.  Once labeling is configured, only the explicit label
    # boundary proves teacher progress; do not present the older success time
    # as a fresh label watermark.  Legacy states without a configured profile
    # retain the conservative fallback for backward compatibility.
    legacy_success_at = _parse_utc_timestamp(last_success_value)
    production_labeling = enabled and bool(
        str(config.get("teacher_profile") or "")
        or str(snapshot.get("teacher_profile") or "")
        or str(state.get("teacher_profile") or "")
    )
    last_success = (
        last_label_at
        if production_labeling
        else last_label_at or legacy_success_at
    )
    progress_source = (
        "last_label_at"
        if last_label_at is not None
        else "missing"
        if production_labeling
        else "last_success_at_legacy"
        if legacy_success_at is not None
        else "missing"
    )
    progress_age = None
    if last_success is not None:
        progress_age = max(0, int((datetime.now(UTC) - last_success).total_seconds()))
    progress_stale = bool(enabled and progress_age is not None and progress_age > DISTILLATION_PROGRESS_STALE_SECONDS)
    if enabled and progress_stale:
        add(
            "data_wait",
            "durable_progress_stale",
            "データ待ち: 蒸留のdurable progressが stale です",
        )
    if enabled and not last_success and worker_status not in {"ready", "idle"}:
        add(
            "data_wait",
            "durable_progress_missing",
            "データ待ち: 蒸留のdurable progressがありません",
        )

    diagnostic = {
        **snapshot,
        "enabled": enabled,
        "capture_enabled": capture_enabled,
        "teacher_profile": str(config.get("teacher_profile") or ""),
        "native_luna_enabled": native_luna_enabled,
        "teacher_contract_expires_at": expiry or None,
        "last_label_at": last_label_at.isoformat() if last_label_at else None,
        "progress_source": progress_source,
        "progress_age_seconds": progress_age,
        "progress_stale": progress_stale,
        "alert": bool(snapshot.get("alert")) or bool(alerts),
        "alerts": alerts,
        "reason_codes": [item["reason"] for item in alerts],
        "wait_category": alerts[0]["category"] if alerts else "",
        "detail": alerts[0]["message"] if alerts else "",
        "stop_reason": alerts[0] if alerts else None,
    }
    if native_luna_supervisor is not None:
        diagnostic["native_luna_supervisor"] = native_luna_supervisor
    return diagnostic, alerts


def recall_distillation_health(root: Path | None = None) -> dict[str, Any]:
    """Return the lightweight distillation diagnosis used by health and init."""

    effective_root = root or CHRONOVISOR_ROOT
    snapshot = (
        recall_distillation_kpi() if root is None else recall_distillation_kpi(root)
    )
    diagnostic, _ = _distillation_alerts(snapshot, root=effective_root)
    return diagnostic


def summary_coverage() -> dict[str, Any]:
    store = get_store()
    store.refresh()
    metas = store.all_pages_meta(include_system=False)
    knowledge = [m for m in metas if m.get("page_type") != "reference"]
    with_summary = 0
    with_questions = 0
    typed: dict[str, int] = {}
    sensitivity: dict[str, int] = {}
    for meta in knowledge:
        typed[str(meta.get("page_type") or "knowledge")] = (
            typed.get(str(meta.get("page_type") or "knowledge"), 0) + 1
        )
        tier = str(meta.get("sensitivity") or "normal")
        sensitivity[tier] = sensitivity.get(tier, 0) + 1
        full = store.meta(str(meta.get("page_id", ""))) or {}
        if isinstance(full.get("summary"), str) and full["summary"].strip():
            with_summary += 1
        questions = full.get("recall_questions")
        if isinstance(questions, list) and questions:
            with_questions += 1
    total = len(knowledge)
    return {
        "knowledge_pages": total,
        "summary_pages": with_summary,
        "recall_question_pages": with_questions,
        "summary_coverage": (with_summary / total) if total else 0.0,
        "recall_question_coverage": (with_questions / total) if total else 0.0,
        "page_types": typed,
        "sensitivity": sensitivity,
    }


def semantic_index_kpi() -> dict[str, Any]:
    """Report search-index coverage without loading the model or scanning vectors."""

    from chronovisor.core.runtime_config import load_search_embedding_config

    config = load_search_embedding_config()
    if not config.enabled or config.rollout_mode == "off":
        return {
            "status": "inactive",
            "execution_mode": "disabled" if not config.enabled else "service",
            "rollout_mode": config.rollout_mode,
            "enabled": config.enabled,
        }
    from chronovisor.core.semantic_index import semantic_index_status
    from chronovisor.core.semantic_jobs import job_status

    jobs = job_status()
    dead = int((jobs.get("counts") or {}).get("dead") or 0)
    socket_ready = Path(config.socket).expanduser().is_socket()
    service_file = CHRONOVISOR_ROOT / "runtime" / "semantic-service-status.json"
    try:
        service = json.loads(service_file.read_text(encoding="utf-8"))
        if not isinstance(service, dict):
            service = {}
    except (OSError, json.JSONDecodeError):
        service = {}
    routes = service.get("routes")
    route_identity = (
        routes.get("search.semantic.foreground")
        if isinstance(routes, dict)
        else None
    )
    if not (
        isinstance(route_identity, dict)
        and set(route_identity) == {"role", "provider", "model", "location"}
        and all(isinstance(value, str) and value for value in route_identity.values())
    ):
        route_identity = None
    index = (
        semantic_index_status(expected_route=route_identity)
        if route_identity is not None
        else {
            "status": "invalid",
            "generation_id": "",
            "coverage": 0.0,
            "error": "semantic_failure",
        }
    )
    coverage = float(index.get("coverage") or 0.0)
    service_age = max(
        0.0, datetime.now(UTC).timestamp()
        - float(service.get("observed_at_epoch") or 0.0)
    )
    service_fresh = bool(service) and service_age <= 30
    service_pid = int(service.get("pid") or 0)
    service_process_alive = False
    if service_pid > 0:
        try:
            os.kill(service_pid, 0)
            service_process_alive = True
        except OSError:
            pass
    generation_matches = (
        bool(index.get("generation_id"))
        and service.get("generation_id") == index.get("generation_id")
    )
    ready = (
        index.get("status") == "ok"
        and coverage >= 0.999
        and dead == 0
        and socket_ready
        and service_fresh
        and service_process_alive
        and generation_matches
        and service.get("ready") is True
    )
    return {
        "status": "ok" if ready else "alert",
        "enabled": True,
        "execution_mode": "service",
        "rollout_mode": config.rollout_mode,
        "route": route_identity or {},
        "model": str((route_identity or {}).get("model") or ""),
        "revision": config.revision,
        "socket_ready": socket_ready,
        "service_fresh": service_fresh,
        "service_process_alive": service_process_alive,
        "generation_matches": generation_matches,
        "service_age_seconds": round(service_age, 3) if service else None,
        "service": service,
        "coverage": coverage,
        "index": index,
        "jobs": jobs,
    }


def _raw_host(path: Path) -> str:
    name = path.name.lower()
    if "codex" in name:
        return "codex"
    if "claude" in name:
        return "claude"
    if "cowork" in name:
        return "cowork"
    return "unknown"


def capture_kpi() -> dict[str, Any]:
    from chronovisor.core.raw_store import RawStore

    raws = tuple(RawStore(RAW_DIR).iter_units()) if RAW_DIR.exists() else ()
    artifact_dir = CHRONOVISOR_ROOT / "runtime" / "raw-projections" / "artifacts"
    artifact_paths = tuple(artifact_dir.glob("*.md")) if artifact_dir.exists() else ()
    by_host: dict[str, int] = {}
    for unit in raws:
        host = _raw_host(Path(unit.raw_id))
        by_host[host] = by_host.get(host, 0) + 1
    for path in artifact_paths:
        host = _raw_host(path)
        by_host[host] = by_host.get(host, 0) + 1

    raw_names = {unit.raw_id for unit in raws} | {path.name for path in artifact_paths}
    claim_rows = 0
    claimed_raws: set[str] = set()
    for row in iter_jsonl(
        CHRONOVISOR_ROOT / "claims" / "claims.jsonl", limit=100_000
    ):
        claim_rows += 1
        source_raw = row.get("source_raw")
        if isinstance(source_raw, str) and source_raw in raw_names:
            claimed_raws.add(source_raw)
    covered = raw_names & claimed_raws
    return {
        "raw_files": len(raws) + len(artifact_paths),
        "claimed_raw_files": len(covered),
        "claim_coverage": (
            len(covered) / (len(raws) + len(artifact_paths))
            if raws or artifact_paths
            else None
        ),
        "raw_by_host": by_host,
        "claim_rows": claim_rows,
    }


def latest_memory_integrity() -> dict[str, Any]:
    path = CHRONOVISOR_ROOT / "eval" / "memory-integrity-latest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "missing", "path": str(path)}
    if not isinstance(payload, dict):
        return {"status": "invalid", "path": str(path)}
    return {
        "status": payload.get("status", "ok"),
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "total": payload.get("total", 0),
        "passed": payload.get("passed", 0),
        "missed": payload.get("missed", 0),
        "capture_rate": payload.get("capture_rate"),
        "by_host": payload.get("by_host", {}),
    }


def cofire_kpi() -> dict[str, Any]:
    path = CHRONOVISOR_ROOT / "recall" / "cofire.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "missing", "path": str(path), "nodes": 0, "edges": 0}
    if not isinstance(payload, dict):
        return {"status": "invalid", "path": str(path), "nodes": 0, "edges": 0}
    return {
        "status": payload.get("status", "ok"),
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "episodes": payload.get("episodes", 0),
        "nodes": payload.get("nodes", 0),
        "edges": payload.get("edges", 0),
    }


def prefetch_kpi() -> dict[str, Any]:
    path = CHRONOVISOR_ROOT / "recall" / "prefetch.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "missing", "path": str(path), "buckets": 0, "tokens": 0}
    if not isinstance(payload, dict):
        return {"status": "invalid", "path": str(path), "buckets": 0, "tokens": 0}
    buckets = payload.get("buckets")
    tokens = payload.get("tokens")
    return {
        "status": payload.get("status", "ok"),
        "path": str(path),
        "generated_at": payload.get("generated_at"),
        "episodes": payload.get("episodes", 0),
        "buckets": (
            int(payload.get("bucket_count") or 0)
            if payload.get("storage") == "sqlite"
            else len(buckets) if isinstance(buckets, dict) else 0
        ),
        "tokens": (
            int(payload.get("token_count") or 0)
            if payload.get("storage") == "sqlite"
            else len(tokens) if isinstance(tokens, dict) else 0
        ),
    }


def derived_memory_kpi() -> dict[str, Any]:
    retention_path = CHRONOVISOR_ROOT / "recall" / "retention.json"
    try:
        retention = json.loads(retention_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        retention = {}
    retention_counts = retention.get("counts") if isinstance(retention, dict) else {}
    stable_pages = canonical_document_paths(
        CHRONOVISOR_ROOT / "pages", require_stable=True
    )
    return {
        "claims": _jsonl_count(CHRONOVISOR_ROOT / "claims" / "claims-index.jsonl"),
        "golden": _jsonl_count(CHRONOVISOR_ROOT / "recall" / "search-golden.jsonl"),
        "distill_rows": _jsonl_count(CHRONOVISOR_ROOT / "distill" / "wiki-qa.jsonl"),
        "retention_pages": int(retention_counts.get("pages") or 0)
        if isinstance(retention_counts, dict)
        else 0,
        "deprecation_candidates": int(
            retention_counts.get("deprecation_candidates") or 0
        )
        if isinstance(retention_counts, dict)
        else 0,
        "hubs": sum(
            path.parent == (CHRONOVISOR_ROOT / "pages" / "hubs").resolve()
            for path in stable_pages
        ),
    }


def read_back_kpi() -> dict[str, Any]:
    run_path = CHRONOVISOR_ROOT / "runtime" / "ingest-read-back-runs.jsonl"
    rows = _read_jsonl(run_path, limit=200)
    cohort = "all_ingest_runs"
    if not rows:
        rows = _read_jsonl(
            CHRONOVISOR_ROOT / "runtime" / "ingest-read-back-failures.jsonl", limit=200
        )
        cohort = "legacy_failure_bearing_runs_only"
    failures = 0
    checked = 0
    latest: dict[str, Any] | None = None
    for row in rows:
        checked += int(row.get("checked") or 0)
        failed_rows = row.get("failed")
        if isinstance(failed_rows, list):
            failures += len(failed_rows)
        latest = row
    passed = max(0, checked - failures)
    ledger_path = CHRONOVISOR_ROOT / "runtime" / "ingest-read-back-repair.json"
    try:
        from chronovisor.core.durable_state import canonical_bytes
        from chronovisor.ingest.read_back_integrity import verify_prior_prefix

        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        observed = ledger.get("view_sha256") if isinstance(ledger, dict) else None
        unsigned = (
            {key: value for key, value in ledger.items() if key != "view_sha256"}
            if isinstance(ledger, dict)
            else {}
        )
        expected = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
        cursor = ledger.get("source_cursor") if isinstance(ledger, dict) else None
        source_path = CHRONOVISOR_ROOT / "runtime" / "ingest-read-back-failures.jsonl"
        ledger_integrity = {
            "status": (
                "ok"
                if observed == expected and verify_prior_prefix(source_path, cursor)
                else "invalid"
            ),
            "schema_version": ledger.get("schema_version")
            if isinstance(ledger, dict)
            else None,
            "source_cursor": cursor,
        }
    except FileNotFoundError:
        ledger_integrity = {"status": "missing"}
    except Exception as exc:
        ledger_integrity = {"status": "invalid", "error": str(exc)}
    return {
        "checked": checked,
        "passed": passed,
        "failures": failures,
        "pass_rate": (passed / checked) if checked else None,
        "latest": latest,
        "cohort": cohort,
        "derived_view_integrity": ledger_integrity,
    }


def autonomy_hardening_kpi() -> dict[str, Any]:
    from chronovisor.core.durable_state import DurableStateError, read_sealed_json
    from chronovisor.core.managed_hold import ManagedHoldStore
    from chronovisor.decision.quality_guard import quality_snapshot
    from chronovisor.ops.deadman import inspect_heartbeat
    from chronovisor.recall.provisional_recall import snapshot as provisional_snapshot

    runtime = CHRONOVISOR_ROOT / "runtime"
    autonomy = CHRONOVISOR_ROOT / "autonomy"
    managed = ManagedHoldStore(runtime / "managed-holds" / "state.json")
    try:
        managed_snapshot = managed.snapshot()
    except Exception as exc:
        managed_snapshot = {"status": "invalid", "error": str(exc)}
    try:
        provisional = provisional_snapshot(chronovisor_root=CHRONOVISOR_ROOT)
    except Exception as exc:
        provisional = {"status": "invalid", "error": str(exc)}
    try:
        quality = quality_snapshot(runtime / "quality")
    except Exception as exc:
        quality = {"status": "invalid", "error": str(exc)}
    try:
        deadman_threshold = read_sealed_json(autonomy / "observer-threshold-state.json")
    except DurableStateError:
        deadman_threshold = {"status": "unavailable"}
    artifacts = runtime / "decision-artifacts"
    try:
        artifact_count = sum(1 for _path in artifacts.glob("[0-9a-f][0-9a-f]/*.json"))
    except OSError:
        artifact_count = 0
    return {
        "decision_artifacts": {
            "count": artifact_count,
            "replay_definition": "sealed_execution_fingerprint",
        },
        "deadman": {
            "main": inspect_heartbeat(
                autonomy / "watchdog-heartbeat.json",
                expected_role="main_watchdog",
                max_age_seconds=20 * 60,
            ),
            "observer": inspect_heartbeat(
                autonomy / "observer-heartbeat.json",
                expected_role="independent_observer",
                max_age_seconds=10 * 60,
            ),
            "threshold": deadman_threshold,
        },
        "quality": quality,
        "managed_holds": managed_snapshot,
        "provisional_recall": provisional,
        "frontier_semantic_audit_allowed": False,
    }


def recall_feedback_kpi() -> dict[str, Any]:
    rows = _read_jsonl(CHRONOVISOR_ROOT / "recall" / "feedback.jsonl", limit=1000)
    counts: dict[str, int] = {}
    for row in rows:
        kind = str(row.get("kind") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    used = counts.get("injection_used", 0)
    ignored = counts.get("injection_ignored", 0) + counts.get("false-positive", 0)
    denominator = used + ignored
    return {
        "samples": len(rows),
        "counts": counts,
        "precision_proxy": (used / denominator) if denominator else None,
        "missed_candidates": counts.get("missed_candidate", 0)
        + counts.get("missed", 0),
    }


_CONTENT_CORRECTION_SEMANTIC_HOLD_KIND = "content_correction_semantic_no_quorum"
_CONTENT_CORRECTION_DECISION_LANES = frozenset(
    {"content_correction_classification", "content_correction_review"}
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_terminal_semantic_defer(item: dict[str, Any]) -> bool:
    """Separate a safe semantic non-decision from an operational quarantine."""

    common_hold = persisted_semantic_no_quorum_hold(item)
    if common_hold is not None:
        decision_lane = str(common_hold.get("lane") or "")
        result = item.get("result")
        return bool(
            str(item.get("status") or "") == "quarantined"
            and str(item.get("last_failure_class") or "") == LOCAL_SEMANTIC_NO_QUORUM
            and isinstance(result, dict)
            and result.get("terminal_reason") == "semantic_no_quorum"
            and item.get("quarantine_reason") == f"semantic_no_quorum:{decision_lane}"
        )

    result = item.get("result")
    legacy_hold = (
        result.get("legacy_semantic_hold") if isinstance(result, dict) else None
    )
    if isinstance(legacy_hold, dict):
        expected_fields = {
            "schema_version",
            "kind",
            "lane",
            "epoch",
            "epoch_sha256",
            "authority",
            "authority_sha256",
            "migrated_from",
            "hold_sha256",
        }
        unsigned = dict(legacy_hold)
        unsigned.pop("hold_sha256", None)
        try:
            legacy_valid = bool(
                set(legacy_hold) == expected_fields
                and legacy_hold.get("schema_version") == 1
                and legacy_hold.get("kind")
                == "legacy_local_semantic_no_quorum_fail_closed"
                and isinstance(legacy_hold.get("lane"), str)
                and isinstance(legacy_hold.get("epoch"), dict)
                and isinstance(legacy_hold.get("authority"), dict)
                and isinstance(legacy_hold.get("migrated_from"), dict)
                and semantic_authority_shape_error(
                    legacy_hold["authority"],
                    lane=str(legacy_hold.get("lane") or ""),
                )
                is None
                and legacy_hold.get("epoch_sha256")
                == canonical_sha256(legacy_hold["epoch"])
                and legacy_hold.get("authority_sha256")
                == canonical_sha256(legacy_hold["authority"])
                and legacy_hold.get("hold_sha256") == canonical_sha256(unsigned)
            )
        except (TypeError, ValueError):
            legacy_valid = False
        if legacy_valid:
            decision_lane = str(legacy_hold["lane"])
            return bool(
                item.get("status") == "quarantined"
                and item.get("last_failure_class") == LOCAL_SEMANTIC_NO_QUORUM
                and item.get("quarantine_reason")
                == f"semantic_no_quorum_legacy:{decision_lane}"
                and result.get("terminal_reason") == "semantic_no_quorum"
            )

    if (
        str(item.get("status") or "") != "quarantined"
        or str(item.get("lane") or "") != "content_correction"
        or str(item.get("last_failure_class") or "") != "local_semantic_no_quorum"
        or not str(item.get("quarantine_reason") or "").startswith(
            "semantic_no_quorum:"
        )
    ):
        return False
    result = item.get("result")
    if not isinstance(result, dict):
        return False
    semantic_hold = result.get("semantic_hold")
    if (
        result.get("terminal_reason") != "semantic_no_quorum"
        or not isinstance(semantic_hold, dict)
        or semantic_hold.get("kind") != _CONTENT_CORRECTION_SEMANTIC_HOLD_KIND
    ):
        return False
    decision_lane = semantic_hold.get("decision_lane")
    evidence_hashes = semantic_hold.get("page_evidence_hashes")
    return (
        decision_lane in _CONTENT_CORRECTION_DECISION_LANES
        and item.get("quarantine_reason") == f"semantic_no_quorum:{decision_lane}"
        and semantic_hold.get("input_hash") == item.get("input_hash")
        and _is_sha256(semantic_hold.get("input_hash"))
        and _is_sha256(semantic_hold.get("proposal_sha256"))
        and isinstance(evidence_hashes, dict)
        and all(
            isinstance(page_id, str) and _is_sha256(page_hash)
            for page_id, page_hash in evidence_hashes.items()
        )
        and isinstance(semantic_hold.get("authority"), dict)
        and bool(semantic_hold["authority"])
    )


def convergence_kpi() -> dict[str, Any]:
    path = CHRONOVISOR_ROOT / "runtime" / "convergence" / "state.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status": "missing",
            "path": str(path),
            "items": 0,
            "by_status": {},
            "by_lane": {},
        }
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, dict):
        return {
            "status": "invalid",
            "path": str(path),
            "items": 0,
            "by_status": {},
            "by_lane": {},
        }
    by_status: dict[str, int] = {}
    by_lane: dict[str, dict[str, int]] = {}
    now = datetime.now(UTC)
    actionable_dates: list[datetime] = []
    expired_running = 0
    for item in items.values():
        if not isinstance(item, dict):
            continue
        status = (
            "semantic_deferred"
            if _is_terminal_semantic_defer(item)
            else str(item.get("status") or "unknown")
        )
        lane = str(item.get("lane") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        lane_counts = by_lane.setdefault(lane, {})
        lane_counts[status] = lane_counts.get(status, 0) + 1
        if status in {
            "pending_local",
            "local_retry",
            "pending_frontier",
            "frontier_retry",
            "local_running",
            "frontier_running",
        }:
            with contextlib.suppress(ValueError):
                actionable_dates.append(
                    datetime.fromisoformat(
                        str(item.get("created_at") or "")
                    ).astimezone(UTC)
                )
        if status in {"local_running", "frontier_running"}:
            try:
                expires = datetime.fromisoformat(
                    str(item.get("lease_expires_at") or "")
                ).astimezone(UTC)
                expired_running += int(expires <= now)
            except ValueError:
                expired_running += 1
    events = _read_jsonl(path.with_name("events.jsonl"), limit=100000)
    cutoff = now - timedelta(hours=24)
    recent = []
    for event in events:
        try:
            ts = datetime.fromisoformat(str(event.get("ts") or "")).astimezone(
                UTC
            )
        except ValueError:
            continue
        if ts >= cutoff:
            recent.append(event)
    arrivals = sum(event.get("event") == "candidate_merged" for event in recent)
    completions = sum(
        event.get("event") in {"completed", "candidate_completed"} for event in recent
    )
    oldest = min(actionable_dates) if actionable_dates else None
    return {
        "status": "ok",
        "path": str(path),
        "items": sum(by_status.values()),
        "by_status": dict(sorted(by_status.items())),
        "by_lane": {
            lane: dict(sorted(counts.items()))
            for lane, counts in sorted(by_lane.items())
        },
        "actionable": sum(
            count
            for status, count in by_status.items()
            if status
            in {
                "pending_local",
                "local_retry",
                "pending_frontier",
                "frontier_retry",
                "local_running",
                "frontier_running",
            }
        ),
        "quarantined": by_status.get("quarantined", 0),
        "semantic_deferred": by_status.get("semantic_deferred", 0),
        "human_required": by_status.get("human_required", 0),
        "expired_running": expired_running,
        "oldest_actionable_at": oldest.isoformat(timespec="seconds")
        if oldest
        else None,
        "oldest_actionable_age_hours": round((now - oldest).total_seconds() / 3600, 2)
        if oldest
        else 0.0,
        "arrivals_24h": arrivals,
        "completions_24h": completions,
        "net_growth_24h": arrivals - completions,
    }


def capture_pipeline_kpi() -> dict[str, Any]:
    from chronovisor.core.background_jobs import snapshot as job_snapshot

    sweeper_path = CHRONOVISOR_ROOT / "runtime" / "session-sweeper-latest.json"
    try:
        sweeper = json.loads(sweeper_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        sweeper = {"status": "missing", "pending": None, "processed": 0}
    return {"background_jobs": job_snapshot(), "session_sweeper": sweeper}


def _timestamp_age_seconds(value: object) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    now = datetime.now(timestamp.tzinfo) if timestamp.tzinfo else datetime.now()
    return max(0.0, (now - timestamp).total_seconds())


def _liveness_failure(
    payload: dict[str, Any], *, authority_blocked: bool
) -> object:
    active_failure = payload.get("active_failure")
    if active_failure is not None:
        return active_failure
    error = payload.get("error")
    authority = payload.get("mutation_authority")
    if authority is None:
        authority = payload.get("authority_preflight")
    if not error and isinstance(authority, dict):
        error = authority.get("error")
    if authority_blocked and error:
        return {
            "kind": "authority",
            "message": str(error),
            "timestamp": payload.get("observed_at"),
        }
    return None


def ingest_liveness_kpi() -> dict[str, Any]:
    path = CHRONOVISOR_ROOT / "runtime" / "ingest-liveness.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status": "unknown",
            "path": str(path),
            "alert": False,
            "stale": False,
            "pending_raws": None,
        }
    if not isinstance(payload, dict):
        return {
            "status": "invalid",
            "path": str(path),
            "alert": False,
            "stale": False,
        }
    runtime_state = payload.get("runtime_state")
    if runtime_state is None:
        runtime_state = payload.get("status")
    waiting = runtime_state in {
        "waiting_for_ingest_runtime",
        "waiting_for_ollama",
    }
    authority = payload.get("mutation_authority")
    if authority is None:
        authority = payload.get("authority_preflight")
    mutation_ready = payload.get("mutation_ready")
    if not isinstance(mutation_ready, bool):
        if isinstance(authority, dict) and isinstance(authority.get("ok"), bool):
            mutation_ready = authority["ok"]
        elif isinstance(payload.get("authority_available"), bool):
            mutation_ready = payload["authority_available"]
        else:
            mutation_ready = None
    authority_blocked = (
        runtime_state == "blocked_by_decision_authority" or mutation_ready is False
    )
    pending = int(payload.get("pending_raws") or 0)
    age = _timestamp_age_seconds(payload.get("observed_at"))
    liveness_fresh = age is not None and age <= INGEST_LIVENESS_TTL_SECONDS
    stale = age is not None and not liveness_fresh
    active_failure = _liveness_failure(payload, authority_blocked=authority_blocked)
    last_failure = payload.get("last_failure")
    if last_failure is None:
        last_failure = active_failure
    last_failure_at = payload.get("last_failure_at")
    if last_failure_at is None and isinstance(last_failure, dict):
        last_failure_at = last_failure.get("timestamp")
    last_failure_age = _timestamp_age_seconds(last_failure_at)
    alert = authority_blocked or (waiting and pending > 0) or stale
    return {
        **payload,
        "status": "alert" if alert else "ok",
        "runtime_status": runtime_state,
        "runtime_state": runtime_state,
        "mutation_ready": mutation_ready,
        "mutation_authority": authority,
        "active_failure": active_failure,
        "last_failure": last_failure,
        "last_failure_at": last_failure_at,
        "last_failure_age_seconds": round(last_failure_age, 3)
        if last_failure_age is not None
        else None,
        "liveness": {
            "observed_at": payload.get("observed_at"),
            "age_seconds": round(age, 3) if age is not None else None,
            "fresh": liveness_fresh if age is not None else None,
            "stale": stale,
            "ttl_seconds": INGEST_LIVENESS_TTL_SECONDS,
        },
        "stale": stale,
        "alert": alert,
        "path": str(path),
    }


def runtime_status_kpi() -> dict[str, Any]:
    """Expose current runtime authority without treating failure history as live."""

    from chronovisor.core.runtime_status import read_status

    payload = read_status()
    mutation_ready = payload.get("mutation_ready")
    authority_blocked = mutation_ready is False
    return {
        **payload,
        "status": "alert" if authority_blocked else "ok",
        "authority_blocked": authority_blocked,
        "alert": authority_blocked,
    }


def research_kpi(*, limit: int = 200) -> dict[str, Any]:
    """Bounded summary of durable research traces for dashboard/alerts."""

    from chronovisor.search.research_config import load_research_config

    root = CHRONOVISOR_ROOT / "runtime" / "research"
    runs_root = root / "runs"
    try:
        summaries = sorted(
            runs_root.glob("*/summary.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[: max(1, limit)]
    except OSError:
        summaries = []
    runs: list[dict[str, Any]] = []
    totals = {
        "runs": 0,
        "completed": 0,
        "terminal": 0,
        "first_pass_malformed": 0,
        "repair_turns": 0,
        "invalid_action_executions": 0,
        "actions": 0,
        "observations": 0,
        "supported_claims": 0,
        "contradicted_claims": 0,
        "unknown_claims": 0,
        "observation_bytes": 0,
    }
    stop_reasons: dict[str, int] = {}
    provider_counts: dict[str, int] = {}
    cache_counts: dict[str, int] = {}
    traced_actions = 0
    traced_observations = 0
    for index, path in enumerate(summaries):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(summary, dict):
            continue
        totals["runs"] += 1
        status = str(summary.get("status") or "terminal")
        totals["completed" if status == "completed" else "terminal"] += 1
        for key in (
            "first_pass_malformed",
            "repair_turns",
            "invalid_action_executions",
            "actions",
            "observations",
        ):
            totals[key] += int(summary.get(key) or 0)
        usage = summary.get("usage") if isinstance(summary.get("usage"), dict) else {}
        totals["observation_bytes"] += int(usage.get("observation_bytes") or 0)
        reason = str(summary.get("stop_reason") or "unknown")
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
        for claim in summary.get("claims", []):
            if not isinstance(claim, dict):
                continue
            claim_status = str(claim.get("status") or "unknown")
            key = f"{claim_status}_claims"
            if key in totals:
                totals[key] += 1
        events_path = path.with_name("events.jsonl")
        for event in _read_jsonl(events_path, limit=2_000) if index < 20 else ():
            if event.get("kind") == "action":
                traced_actions += 1
            elif event.get("kind") == "observation":
                traced_observations += 1
                metadata = (
                    event.get("metadata")
                    if isinstance(event.get("metadata"), dict)
                    else {}
                )
                provider = str(metadata.get("provider") or "")
                cache = str(metadata.get("cache") or "")
                if provider:
                    provider_counts[provider] = provider_counts.get(provider, 0) + 1
                if cache:
                    cache_counts[cache] = cache_counts.get(cache, 0) + 1
        runs.append(
            {
                "research_run_id": summary.get("research_run_id"),
                "status": status,
                "stop_reason": reason,
                "elapsed_ms": summary.get("elapsed_ms"),
                "usage": usage,
                "first_pass_malformed": summary.get("first_pass_malformed", 0),
                "repair_turns": summary.get("repair_turns", 0),
            }
        )
    config = load_research_config()
    claim_total = (
        totals["supported_claims"]
        + totals["contradicted_claims"]
        + totals["unknown_claims"]
    )
    return {
        "status": "ok",
        "enabled": config.enabled,
        "mode": config.mode,
        "kill_switches": {
            "agent": not config.enabled,
            "web": not (config.web.adapter_enabled and config.web.live_egress_enabled),
            "compaction": not config.compaction.enabled,
            "consolidation": not config.consolidation_enabled,
        },
        "totals": totals,
        "claim_coverage": (
            totals["supported_claims"] / claim_total if claim_total else 0.0
        ),
        "decision_trace_coverage": (
            min(1.0, traced_observations / traced_actions)
            if traced_actions
            else 1.0
        ),
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "providers": dict(sorted(provider_counts.items())),
        "cache": dict(sorted(cache_counts.items())),
        "active": (root / "active-research.json").exists(),
        "recent": runs[:20],
    }


def _queue_status_counts(path: Path, field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in _read_jsonl(path, limit=100000):
        status = str(row.get(field) or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


_LINT_ACTIVE_STATUSES = {
    "pending_local",
    "local_retry",
    "pending_frontier",
    "frontier_retry",
    "local_running",
    "frontier_running",
}

QUEUE_PROGRESS_STALE_SECONDS = 24 * 60 * 60
_SEARCH_LABEL_REVIEW_STATUSES = {"pending_review", "unknown"}
_SEARCH_LABEL_FRONTIER_STATUSES = {
    "pending_frontier_review",
    "frontier_retry",
    "frontier_uncertain",
}
_RAW_REPLAY_ACTIVE_STATUSES = {"pending", "retry", "retry_wait"}
_LINT_LOCAL_AUTHORITY_STATUSES = {
    "local_route_provenance_invalid",
    "local_route_uncertain",
    "local_route_lease_expired",
    "local_lease_expired",
    "lease_expired",
}


def _lint_queue_kpi(queue_path: Path, convergence_path: Path) -> dict[str, int]:
    """Count current unresolved lint issues instead of append-only history.

    The lint queue is a current detector snapshot, but a row remains present
    after the convergence lane has observed, routed, rejected, or applied it.
    Counting every JSONL row therefore turns completed monitor/review work into
    a permanent backlog alert.  ``issue_key`` is the detector's stable
    identity, so active convergence wins, a terminal convergence record marks
    the issue handled, and a row with no state is still genuinely unprocessed.
    """

    rows = _read_jsonl(queue_path, limit=100000)
    issue_keys: set[str] = set()
    missing_identity = 0
    for row in rows:
        issue_key = str(row.get("issue_key") or "")
        if issue_key:
            issue_keys.add(issue_key)
        else:
            missing_identity += 1

    statuses: dict[str, set[str]] = {}
    try:
        payload = json.loads(convergence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    items = payload.get("items") if isinstance(payload, dict) else None
    if isinstance(items, dict):
        for item in items.values():
            if not isinstance(item, dict) or item.get("lane") != "lint_repair":
                continue
            metadata = (
                item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            )
            issue_key = str(metadata.get("issue_key") or "")
            if issue_key:
                statuses.setdefault(issue_key, set()).add(
                    str(item.get("status") or "unknown")
                )

    active = 0
    handled = 0
    untracked = missing_identity
    for issue_key in issue_keys:
        issue_statuses = statuses.get(issue_key, set())
        if issue_statuses & _LINT_ACTIVE_STATUSES:
            active += 1
        elif issue_statuses:
            handled += 1
        else:
            untracked += 1
    return {
        "total": len(rows),
        "unique": len(issue_keys) + missing_identity,
        "actionable": active + untracked,
        "active": active,
        "untracked": untracked,
        "handled": handled,
    }


def _queue_timestamp(value: object) -> datetime | None:
    """Parse queue timestamps, treating legacy naive values as UTC."""

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _queue_group_progress(
    rows: list[dict[str, Any]],
    *,
    status_field: str,
    statuses: set[str],
    created_fields: tuple[str, ...],
    progress_fields: tuple[str, ...],
) -> dict[str, Any]:
    selected = [
        row for row in rows if str(row.get(status_field) or "unknown") in statuses
    ]
    status_counts: dict[str, int] = {}
    for row in selected:
        status = str(row.get(status_field) or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    created: list[datetime] = []
    progress: list[datetime] = []
    for row in selected:
        for field in created_fields:
            timestamp = _queue_timestamp(row.get(field))
            if timestamp is not None:
                created.append(timestamp)
                break
        for field in progress_fields:
            timestamp = _queue_timestamp(row.get(field))
            if timestamp is not None:
                progress.append(timestamp)
                break
    oldest = min(created) if created else None
    last_progress = max(progress) if progress else None
    now = datetime.now(UTC)
    oldest_age = (now - oldest).total_seconds() if oldest else None
    progress_age = (now - last_progress).total_seconds() if last_progress else None
    stale = bool(
        selected
        and oldest_age is not None
        and oldest_age > QUEUE_PROGRESS_STALE_SECONDS
        and (progress_age is None or progress_age > QUEUE_PROGRESS_STALE_SECONDS)
    )
    return {
        "active": len(selected),
        "status_counts": dict(sorted(status_counts.items())),
        "oldest_at": oldest.isoformat() if oldest else None,
        "oldest_age_seconds": round(max(0.0, oldest_age), 3)
        if oldest_age is not None
        else None,
        "last_progress_at": last_progress.isoformat() if last_progress else None,
        "last_progress_age_seconds": round(max(0.0, progress_age), 3)
        if progress_age is not None
        else None,
        "stale": stale,
    }


def _queue_stall_alert(
    *,
    queue: str,
    lane: str,
    diagnostic: dict[str, Any],
    category: str,
    reason: str,
    message: str,
) -> dict[str, Any]:
    return {
        "component": f"queues.{queue}",
        "category": category,
        "reason": reason,
        "message": message,
        "queue": queue,
        "lane": lane,
        "active": diagnostic["active"],
        "statuses": diagnostic["status_counts"],
        "oldest_at": diagnostic["oldest_at"],
        "last_progress_at": diagnostic["last_progress_at"],
        "oldest_age_seconds": diagnostic["oldest_age_seconds"],
        "last_progress_age_seconds": diagnostic["last_progress_age_seconds"],
    }


def _queue_health_snapshot(root: Path) -> dict[str, Any]:
    """Return privacy-safe queue progress and stall diagnostics.

    Queue presence alone is not an alert: a lane is stale only when its active
    rows have exceeded the progress TTL.  Quarantined and terminal rows remain
    visible as historical counters and never make the current health red.
    """

    recall_root = root / "recall"
    review_root = root / "review"
    runtime_root = root / "runtime"
    label_path = recall_root / "search-label-queue.jsonl"
    convergence_path = runtime_root / "convergence" / "state.json"
    replay_path = review_root / "raw-replay-queue.jsonl"
    label_rows = _read_jsonl(label_path, limit=100000)
    label_counts = _queue_status_counts(label_path, "queue_status")
    alerts: list[dict[str, Any]] = []
    search_summary: dict[str, Any] = {
        "status_counts": label_counts,
        "active": sum(label_counts.get(status, 0) for status in (
            _SEARCH_LABEL_REVIEW_STATUSES | _SEARCH_LABEL_FRONTIER_STATUSES
        )),
        "lanes": {},
    }
    for lane, statuses in (
        ("review", _SEARCH_LABEL_REVIEW_STATUSES),
        ("frontier", _SEARCH_LABEL_FRONTIER_STATUSES),
    ):
        diagnostic = _queue_group_progress(
            label_rows,
            status_field="queue_status",
            statuses=statuses,
            created_fields=("preregistered_at", "ts"),
            progress_fields=("reviewed_at", "last_attempt_at"),
        )
        search_summary["lanes"][lane] = diagnostic
        if not diagnostic["stale"]:
            continue
        # A frontier lane is explicitly delegated to authority review.  Keep
        # that distinction visible even when an older row lacks the optional
        # decision_authority_error marker.
        authority_wait = lane == "frontier" or any(
            isinstance(row.get("decision_authority_error"), str)
            and row.get("decision_authority_error")
            for row in label_rows
            if str(row.get("queue_status") or "unknown") in statuses
        )
        if authority_wait:
            alerts.append(
                _queue_stall_alert(
                    queue="search_labels",
                    lane=lane,
                    diagnostic=diagnostic,
                    category="human_wait",
                    reason="search_labels_authority_wait",
                    message="人の判断待ち: search labels の frontier authority が停滞しています",
                )
            )
        else:
            alerts.append(
                _queue_stall_alert(
                    queue="search_labels",
                    lane=lane,
                    diagnostic=diagnostic,
                    category="data_wait",
                    reason="search_labels_late_lane",
                    message="データ待ち: search labels の lane が進んでいません",
                )
            )

    try:
        convergence_payload = json.loads(convergence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        convergence_payload = {}
    convergence_items = (
        convergence_payload.get("items")
        if isinstance(convergence_payload, dict)
        else None
    )
    lint_rows = [
        dict(item)
        for item in (
            convergence_items.values()
            if isinstance(convergence_items, dict)
            else ()
        )
        if isinstance(item, dict) and item.get("lane") == "lint_repair"
    ]
    lint_summary: dict[str, Any] = {
        "status_counts": {},
        "active": 0,
        "quarantined": 0,
        "lanes": {},
    }
    for row in lint_rows:
        status = str(row.get("status") or "unknown")
        lint_summary["status_counts"][status] = (
            lint_summary["status_counts"].get(status, 0) + 1
        )
    lint_summary["quarantined"] = lint_summary["status_counts"].get("quarantined", 0)
    for lane, statuses in (
        (
            "local",
            {
                "pending_local",
                "local_retry",
                "local_running",
                *_LINT_LOCAL_AUTHORITY_STATUSES,
            },
        ),
        ("frontier", {"pending_frontier", "frontier_retry", "frontier_running"}),
    ):
        diagnostic = _queue_group_progress(
            lint_rows,
            status_field="status",
            statuses=statuses,
            created_fields=("created_at", "updated_at"),
            # Updated-at changes when a row is reclassified or leased; an
            # actual attempt is the durable progress boundary for this lane.
            progress_fields=("last_attempt_at",),
        )
        lint_summary["lanes"][lane] = diagnostic
        lint_summary["active"] += diagnostic["active"]
        if not diagnostic["stale"]:
            continue
        # Frontier work always requires the authority lane.  Local rows that
        # explicitly report route provenance, uncertainty, or an expired lease
        # are also human-wait states; a plain pending_local row remains a
        # data-wait scheduler stall.
        authority_wait = lane == "frontier" or any(
            row.get("human_required") is True
            or any(
                marker in str(row.get(field) or "").casefold()
                for field in ("last_failure_class", "quarantine_reason")
                for marker in ("authority", "lease", "provenance", "uncertain")
            )
            or str(row.get("status") or "unknown") in _LINT_LOCAL_AUTHORITY_STATUSES
            for row in lint_rows
            if str(row.get("status") or "unknown") in statuses
        )
        alerts.append(
            _queue_stall_alert(
                queue="lint_repair",
                lane=lane,
                diagnostic=diagnostic,
                category="human_wait" if authority_wait else "data_wait",
                reason=(
                    "lint_repair_authority_wait"
                    if authority_wait
                    else "lint_repair_late_lane"
                ),
                message=(
                    "人の判断待ち: lint repair の frontier authority が停滞しています"
                    if authority_wait
                    else "データ待ち: lint repair の lane が進んでいません"
                ),
            )
        )

    replay_rows = _read_jsonl(replay_path, limit=100000)
    replay_counts = _queue_status_counts(replay_path, "status")
    replay_summary: dict[str, Any] = {
        "status_counts": replay_counts,
        "active": sum(replay_counts.get(status, 0) for status in _RAW_REPLAY_ACTIVE_STATUSES),
    }
    replay_diagnostic = _queue_group_progress(
        replay_rows,
        status_field="status",
        statuses=_RAW_REPLAY_ACTIVE_STATUSES,
        created_fields=("created_at", "updated_at"),
        progress_fields=("updated_at", "last_attempt_at", "started_at"),
    )
    replay_summary.update(replay_diagnostic)
    if replay_diagnostic["stale"]:
        alerts.append(
            _queue_stall_alert(
                queue="raw_replay",
                lane="replay",
                diagnostic=replay_diagnostic,
                category="data_wait",
                reason="raw_replay_late_lane",
                message="データ待ち: raw replay の lane が進んでいません",
            )
        )

    sweeper_path = runtime_root / "session-sweeper-latest.json"
    try:
        sweeper = json.loads(sweeper_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        sweeper = {}
    sweeper_summary = {
        "status": str(sweeper.get("status") or "missing")
        if isinstance(sweeper, dict)
        else "invalid",
        "pending": sweeper.get("pending") if isinstance(sweeper, dict) else None,
        "last_progress_at": sweeper.get("ts") if isinstance(sweeper, dict) else None,
    }
    if isinstance(sweeper, dict) and sweeper:
        sweeper_age = _queue_timestamp(sweeper.get("ts"))
        pending = sweeper.get("pending")
        sweeper_stale = (
            sweeper_age is not None
            and (datetime.now(UTC) - sweeper_age).total_seconds()
            > QUEUE_PROGRESS_STALE_SECONDS
        )
        if sweeper.get("status") not in {"ok", "clean"} or (
            isinstance(pending, int)
            and pending > 0
            and (sweeper_stale or sweeper_age is None)
        ):
            alerts.append(
                _queue_stall_alert(
                    queue="session_sweeper",
                    lane="sweeper",
                    diagnostic={
                        "active": int(pending or 0),
                        "status_counts": {str(sweeper.get("status") or "missing"): 1},
                        "oldest_at": sweeper.get("ts"),
                        "last_progress_at": sweeper.get("ts"),
                        "oldest_age_seconds": (
                            round((datetime.now(UTC) - sweeper_age).total_seconds(), 3)
                            if sweeper_age
                            else None
                        ),
                        "last_progress_age_seconds": (
                            round((datetime.now(UTC) - sweeper_age).total_seconds(), 3)
                            if sweeper_age
                            else None
                        ),
                    },
                    category="human_wait",
                    reason="session_sweeper_attention",
                    message="人の判断待ち: session sweeper の現在状態を確認してください",
                )
            )

    return {
        "alert": bool(alerts),
        "alerts": alerts,
        "search_labels": search_summary,
        "lint_repair": lint_summary,
        "raw_replay": replay_summary,
        "session_sweeper": sweeper_summary,
    }


def queue_health_snapshot(root: Path | None = None) -> dict[str, Any]:
    """Return queue progress and stall diagnostics without queue payloads."""

    return _queue_health_snapshot(root or CHRONOVISOR_ROOT)


def queue_health_alerts(root: Path | None = None) -> list[dict[str, Any]]:
    """Return only current queue alerts for lightweight session bootstrap."""

    return list(queue_health_snapshot(root).get("alerts", []))


def health_snapshot() -> dict[str, Any]:
    from chronovisor.core.runtime_config import runtime_identity
    from chronovisor.recall.librarian_status import build_librarian_status

    coverage = summary_coverage()
    duplicate_queue = CHRONOVISOR_ROOT / "review" / "duplicate-candidates.jsonl"
    lint_queue = CHRONOVISOR_ROOT / "review" / "lint-repair-queue.jsonl"
    golden = CHRONOVISOR_ROOT / "recall" / "search-golden.jsonl"
    label_queue = CHRONOVISOR_ROOT / "recall" / "search-label-queue.jsonl"
    raw_replay_queue = CHRONOVISOR_ROOT / "review" / "raw-replay-queue.jsonl"
    label_statuses = _queue_status_counts(label_queue, "queue_status")
    replay_statuses = _queue_status_counts(raw_replay_queue, "status")
    lint_queue_status = _lint_queue_kpi(
        lint_queue,
        CHRONOVISOR_ROOT / "runtime" / "convergence" / "state.json",
    )
    ingest_liveness = ingest_liveness_kpi()
    runtime_status = runtime_status_kpi()
    semantic_index = semantic_index_kpi()
    recall_distillation = recall_distillation_health()
    queue_health = queue_health_snapshot()
    overall_alert = (
        bool(ingest_liveness.get("alert"))
        or bool(runtime_status.get("alert"))
        or semantic_index.get("status") == "alert"
        or bool(recall_distillation.get("alert"))
        or recall_distillation.get("status") == "alert"
        or bool(queue_health.get("alert"))
        or queue_health.get("status") == "alert"
    )
    alerts: list[dict[str, Any]] = []
    for item in recall_distillation.get("alerts", []):
        if isinstance(item, dict):
            alerts.append({"component": "recall_distillation", **item})
    for item in queue_health.get("alerts", []):
        if isinstance(item, dict):
            alerts.append(dict(item))
    return {
        "status": "alert" if overall_alert else "ok",
        "alerts": alerts,
        "runtime": runtime_identity(),
        "coverage": coverage,
        "capture": capture_kpi(),
        "memory_integrity": latest_memory_integrity(),
        "cofire": cofire_kpi(),
        "prefetch": prefetch_kpi(),
        "derived": derived_memory_kpi(),
        "read_back": read_back_kpi(),
        "autonomy_hardening": autonomy_hardening_kpi(),
        "recall_feedback": recall_feedback_kpi(),
        "recall_distillation": recall_distillation,
        "convergence": convergence_kpi(),
        "capture_pipeline": capture_pipeline_kpi(),
        "ingest_liveness": ingest_liveness,
        "runtime_status": runtime_status,
        "semantic_index": semantic_index,
        "librarian": build_librarian_status(CHRONOVISOR_ROOT),
        "research": research_kpi(),
        "queues": {
            "duplicate_candidates": _jsonl_count(duplicate_queue),
            "lint_repair": lint_queue_status["actionable"],
            "lint_repair_total": lint_queue_status["total"],
            "lint_repair_active": lint_queue_status["active"],
            "lint_repair_untracked": lint_queue_status["untracked"],
            "lint_repair_handled": lint_queue_status["handled"],
            "search_golden": _jsonl_count(golden),
            "search_labels": _jsonl_count(label_queue),
            "search_labels_pending": sum(
                count
                for status, count in label_statuses.items()
                if status
                in {
                    "unknown",
                    "pending_review",
                    "pending_frontier_review",
                    "frontier_retry",
                    "frontier_uncertain",
                }
            ),
            "raw_replay": _jsonl_count(raw_replay_queue),
            "raw_replay_pending": sum(
                count
                for status, count in replay_statuses.items()
                if status in {"unknown", "pending", "failed", "retry", "retry_wait"}
            ),
            "search_label_statuses": label_statuses,
            "raw_replay_statuses": replay_statuses,
            "health": queue_health,
        },
        "paths": {
            "duplicate_queue": str(duplicate_queue),
            "lint_queue": str(lint_queue),
            "search_golden": str(golden),
            "search_label_queue": str(label_queue),
            "raw_replay_queue": str(raw_replay_queue),
        },
    }
