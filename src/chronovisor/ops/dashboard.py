"""Local web dashboard for Chronovisor ingest observability."""

from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import getpass
import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import secrets
import socket
import subprocess
import threading
import time
from datetime import date, datetime, timedelta
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse

import httpx

from chronovisor.core.ollama import OLLAMA_URL, embedding_model, ingest_model
from chronovisor.core.runtime_config import (
    load_decision_router_config,
    load_reranker_config,
    load_search_embedding_config,
    runtime_identity,
)
from chronovisor.core.sealed_artifact_decoder import schema_matches
from chronovisor.core.store import CHRONOVISOR_ROOT, LOG_FILE, init_chronovisor
from chronovisor.decision.decision_router import resolve_router_policy
from chronovisor.ingest import orchestrator
from chronovisor.ops import runtime_status
from chronovisor.ops.convergence import is_human_required_result
from chronovisor.ops.cortex import (
    CortexEventCursor,
    build_cortex_field_projection,
    build_cortex_graph,
    build_cortex_relation_details,
    websocket_accept,
    websocket_text_frame,
)
from chronovisor.ops.dashboard_http import (
    _file_response,
    _json_response,
    _send_security_headers,
)
from chronovisor.ops.dashboard_static import STATIC_DIR, _resolve_static_path
from chronovisor.ops.health import health_snapshot
from chronovisor.ops.model_lab import snapshot as model_lab_snapshot
from chronovisor.recall import recall_runtime
from chronovisor.recall.recall_auditor import load_audit_policy
from chronovisor.recall.recall_improvement import configured_models

DASHBOARD_ACCESS_COOKIE = "chronovisor_dashboard_access"
DASHBOARD_ACCESS_QUERY = "access_token"
DASHBOARD_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{32,128}$")
DASHBOARD_SESSION_MAX_AGE_SECONDS = 365 * 24 * 60 * 60
DASHBOARD_CREDENTIAL_VERSION = 1
DASHBOARD_PASSWORD_ALGORITHM = "scrypt"
DASHBOARD_PASSWORD_SCRYPT_N = 2**14
DASHBOARD_PASSWORD_SCRYPT_R = 8
DASHBOARD_PASSWORD_SCRYPT_P = 1
DASHBOARD_PASSWORD_DKLEN = 32
DASHBOARD_LOGIN_ATTEMPT_LIMIT = 5
DASHBOARD_LOGIN_ATTEMPT_WINDOW_SECONDS = 5 * 60
_LOCK_TYPE = type(threading.Lock())
LOG_LINE_RE = re.compile(r"^- \[(?P<time>[^\]]+)\] (?P<message>.*)$")
RAW_DATE_RE = re.compile(r"(?:^|[^0-9])(?P<stamp>20\d{6})(?:[^0-9]|$)")
SEMANTIC_PROJECTION_CHILD_RE = re.compile(
    r"^semantic-(?P<projection>[0-9a-f]{64})-child-[0-9]{8}-[0-9a-f]{64}\.md$"
)
LOG_PAGE_CHANGE_RE = re.compile(
    r"^- \[(?P<time>[^\]]+)\] ingest \| (?P<kind>created|updated) (?P<page>.+)$"
)
SELF_HEAL_PENDING_STATUSES = {
    "pending_local_repair",
    "local_repairing",
    "pending_frontier",
    "frontier_running",
    "frontier_retry",
    "frontier_preflight_failed",
    "pending_frontier_review",
    "repair_deferred",
}
SELF_HEAL_FAILED_STATUSES = {
    "local_repair_failed",
    "frontier_rejected",
    "frontier_quarantined",
    "human_required",
}
FRONTIER_ACTIVITY_STALE_SECONDS = 6 * 60 * 60
LOCAL_CONSENSUS_ACTIVITY_STALE_SECONDS = 60 * 60
MODEL_ACTIVITY_STALE_SECONDS = 6 * 60 * 60
MODEL_ACTIVITY_VISIBLE_SECONDS = 1.25
ACTIVE_BATCH_STAGES = {
    "batch",
    "raw",
    "triage",
    "generate",
    "authorization",
    "local-consensus-review",
    "local-regenerate",
    "frontier-review",
    # Compatibility with cached status written by releases before routine
    # ingest convergence was relabeled as a local-consensus operation.
    "frontier-regenerate",
    "apply",
}
DECISION_ROUTER_DASHBOARD_CACHE_SECONDS = 15.0
# Live runtime status is overlaid at response time; expensive cold aggregates
# only need this bounded active refresh cadence.
SNAPSHOT_ACTIVE_CACHE_SECONDS = 30.0
SNAPSHOT_IDLE_CACHE_SECONDS = 60.0
SNAPSHOT_FINGERPRINT_AUDIT_SECONDS = 1.0
SAVE_HISTORY_SEGMENT_DETAIL_DAYS = 30
SAVE_HISTORY_MAX_SEGMENTS_PER_DAY = 64
DASHBOARD_MATERIALIZED_SCHEMA = "chronovisor.ops.dashboard-component.v1"
DASHBOARD_COMPONENT_AUDIT_SECONDS = 300.0
DASHBOARD_HEALTH_AUDIT_SECONDS = 60.0
DASHBOARD_LOCAL_CONSENSUS_AUDIT_SECONDS = 300.0
PROCESSING_ACTIVITY_POLL_SECONDS = 0.25
PROCESSING_ACTIVITY_HEARTBEAT_SECONDS = 10.0
PROCESSING_ACTIVITY_AUDIT_SECONDS = 1.0
PROCESSING_ACTIVITY_RECENT_AUDIT_SECONDS = PROCESSING_ACTIVITY_POLL_SECONDS
_PROCESS_IDENTITY_MATCH = "match"
_PROCESS_IDENTITY_MISMATCH = "mismatch"
_PROCESS_IDENTITY_UNAVAILABLE = "unavailable"
_DECISION_ROUTER_CACHE_LOCK = threading.Lock()
_DECISION_ROUTER_CACHE: dict[str, Any] = {
    "key": None,
    "expires_at": 0.0,
    "config": None,
}
_SNAPSHOT_CACHE_LOCK = threading.Lock()
_SNAPSHOT_BUILD_LOCK = threading.Lock()
_SNAPSHOT_CACHE: dict[str, Any] = {
    "built_at": 0.0,
    "fingerprint": None,
    "snapshot": None,
    "refreshing": False,
}
_SNAPSHOT_FINGERPRINT_LOCK = threading.Lock()
_SNAPSHOT_FINGERPRINT_CONDITION = threading.Condition(_SNAPSHOT_FINGERPRINT_LOCK)
_SNAPSHOT_FINGERPRINT_CACHE: dict[str, Any] = {
    "source": None,
    "fingerprint": None,
    "audited_at": 0.0,
    "probing": False,
    "generation": 0,
    "probe_count": 0,
    "cache_hits": 0,
    "coalesced": 0,
    "error_count": 0,
}
_PROCESSING_ACTIVITY_CACHE_LOCK = threading.Lock()
_PROCESSING_ACTIVITY_CACHE_CONDITION = threading.Condition(
    _PROCESSING_ACTIVITY_CACHE_LOCK
)
_PROCESSING_ACTIVITY_CACHE: dict[str, Any] = {
    "source": None,
    "snapshot": None,
    "audited_at": 0.0,
    "refreshing": False,
    "build_count": 0,
    "cache_hits": 0,
    "coalesced": 0,
    "error_count": 0,
    "last_build_duration_ms": 0.0,
    "last_error": None,
}
_MATERIALIZED_COMPONENT_LOCK = threading.RLock()
_MATERIALIZED_COMPONENTS: dict[tuple[str, str], dict[str, Any]] = {}
_MATERIALIZED_COMPONENT_REFRESHING: set[tuple[str, str]] = set()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _materialized_component_path(name: str) -> Path:
    if re.fullmatch(r"[a-z0-9-]+", name) is None:
        raise ValueError("dashboard component name is invalid")
    return CHRONOVISOR_ROOT / "runtime" / "dashboard-materialized" / f"{name}.json"


def _path_identity(path: Path) -> tuple[Any, ...]:
    try:
        stat = path.stat()
    except OSError:
        return (str(path), "missing")
    return (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _component_source_fingerprint(
    name: str,
    paths: list[Path],
    *,
    identities: list[str] | None = None,
) -> str:
    payload = {
        "schema": DASHBOARD_MATERIALIZED_SCHEMA,
        "component": name,
        "paths": [_path_identity(path) for path in sorted(set(paths))],
        "identities": sorted(set(identities or [])),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _read_materialized_component(name: str) -> dict[str, Any] | None:
    path = _materialized_component_path(name)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or not schema_matches(payload.get("schema"), DASHBOARD_MATERIALIZED_SCHEMA)
        or not isinstance(payload.get("fingerprint"), str)
        or isinstance(payload.get("built_at_epoch"), bool)
        or not isinstance(payload.get("built_at_epoch"), (int, float))
        or not isinstance(payload.get("value"), dict)
        or not isinstance(payload.get("value_sha256"), str)
    ):
        return None
    observed = hashlib.sha256(_canonical_json_bytes(payload["value"])).hexdigest()
    return payload if hmac.compare_digest(observed, payload["value_sha256"]) else None


def _write_materialized_component(
    name: str,
    *,
    fingerprint: str,
    value: dict[str, Any],
    built_at_epoch: float,
) -> dict[str, Any]:
    from chronovisor.core.link_fix import atomic_write

    path = _materialized_component_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": DASHBOARD_MATERIALIZED_SCHEMA,
        "component": name,
        "fingerprint": fingerprint,
        "built_at_epoch": built_at_epoch,
        "value_sha256": hashlib.sha256(_canonical_json_bytes(value)).hexdigest(),
        "value": value,
    }
    atomic_write(path, json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    with contextlib.suppress(OSError):
        path.chmod(0o600)
    return payload


def _materialized_component(
    name: str,
    *,
    fingerprint: str,
    builder: Any,
    audit_seconds: float = DASHBOARD_COMPONENT_AUDIT_SECONDS,
) -> dict[str, Any]:
    """Reuse an integrity-bound derived view until inputs change or audit is due."""

    cache_key = (str(CHRONOVISOR_ROOT), name)
    now = time.time()
    stale_value: dict[str, Any] | None = None
    start_refresh = False
    with _MATERIALIZED_COMPONENT_LOCK:
        payload = _MATERIALIZED_COMPONENTS.get(cache_key)
        if payload is None:
            payload = _read_materialized_component(name)
            if payload is not None:
                _MATERIALIZED_COMPONENTS[cache_key] = payload
        if payload is not None:
            age = max(0.0, now - float(payload.get("built_at_epoch") or 0.0))
            if payload.get("fingerprint") == fingerprint and age < audit_seconds:
                value = payload.get("value")
                if isinstance(value, dict):
                    return value
            value = payload.get("value")
            if isinstance(value, dict):
                stale_value = value
                start_refresh = cache_key not in _MATERIALIZED_COMPONENT_REFRESHING
                if start_refresh:
                    _MATERIALIZED_COMPONENT_REFRESHING.add(cache_key)

        if stale_value is not None:
            if start_refresh:
                threading.Thread(
                    target=_refresh_materialized_component,
                    kwargs={
                        "name": name,
                        "cache_key": cache_key,
                        "fingerprint": fingerprint,
                        "builder": builder,
                    },
                    name=f"chronovisor-dashboard-{name}",
                    daemon=True,
                ).start()
            return stale_value

        value = builder()
        if not isinstance(value, dict):
            raise TypeError(
                f"dashboard component {name} returned {type(value).__name__}"
            )
        try:
            payload = _write_materialized_component(
                name,
                fingerprint=fingerprint,
                value=value,
                built_at_epoch=now,
            )
        except OSError:
            payload = {
                "schema": DASHBOARD_MATERIALIZED_SCHEMA,
                "fingerprint": fingerprint,
                "built_at_epoch": now,
                "value_sha256": hashlib.sha256(
                    _canonical_json_bytes(value)
                ).hexdigest(),
                "value": value,
            }
        _MATERIALIZED_COMPONENTS[cache_key] = payload
        return value


def _refresh_materialized_component(
    *,
    name: str,
    cache_key: tuple[str, str],
    fingerprint: str,
    builder: Any,
) -> None:
    try:
        value = builder()
        if not isinstance(value, dict):
            raise TypeError(
                f"dashboard component {name} returned {type(value).__name__}"
            )
        now = time.time()
        try:
            payload = _write_materialized_component(
                name,
                fingerprint=fingerprint,
                value=value,
                built_at_epoch=now,
            )
        except OSError:
            payload = {
                "schema": DASHBOARD_MATERIALIZED_SCHEMA,
                "fingerprint": fingerprint,
                "built_at_epoch": now,
                "value_sha256": hashlib.sha256(
                    _canonical_json_bytes(value)
                ).hexdigest(),
                "value": value,
            }
        with _MATERIALIZED_COMPONENT_LOCK:
            _MATERIALIZED_COMPONENTS[cache_key] = payload
        with _SNAPSHOT_CACHE_LOCK:
            _SNAPSHOT_CACHE["fingerprint"] = None
    except Exception:
        pass
    finally:
        with _MATERIALIZED_COMPONENT_LOCK:
            _MATERIALIZED_COMPONENT_REFRESHING.discard(cache_key)


def _ollama_snapshot() -> dict[str, Any]:
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/ps", timeout=1.5)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", [])
        if not isinstance(models, list):
            models = []
        return {"available": True, "models": models}
    except Exception as exc:
        return {"available": False, "models": [], "error": str(exc)}


def _ollama_tags_snapshot() -> dict[str, Any]:
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", [])
        if not isinstance(models, list):
            models = []
        return {"available": True, "models": models}
    except Exception as exc:
        return {"available": False, "models": [], "error": str(exc)}


def _model_name(row: dict[str, Any] | None) -> str:
    if not isinstance(row, dict):
        return ""
    for key in ("model", "name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _numeric_bytes(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        return max(0, int(value))
    return 0


def _add_model_role(roles: dict[str, set[str]], model: str | None, role: str) -> None:
    if not isinstance(model, str):
        return
    model = model.strip()
    if not model:
        return
    roles.setdefault(model, set()).add(role)


def _decision_router_cache_key(config: Any) -> tuple[Any, ...]:
    nominated = str(getattr(config, "adoption_artifact", "") or "").strip()
    artifact_identity: tuple[Any, ...] = ()
    if nominated:
        path = Path(nominated).expanduser()
        try:
            stat = path.stat()
            artifact_identity = (str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns)
        except OSError:
            artifact_identity = (str(path), "missing")
    return (repr(config), *artifact_identity)


def _resolved_decision_router_config() -> Any:
    """Resolve adopted model roles without revalidating the corpus every poll."""

    configured = load_decision_router_config()
    key = _decision_router_cache_key(configured)
    now = time.monotonic()
    with _DECISION_ROUTER_CACHE_LOCK:
        if (
            _DECISION_ROUTER_CACHE.get("key") == key
            and float(_DECISION_ROUTER_CACHE.get("expires_at") or 0.0) > now
            and _DECISION_ROUTER_CACHE.get("config") is not None
        ):
            return _DECISION_ROUTER_CACHE["config"]
        resolved = resolve_router_policy(configured).config
        resolved_at = time.monotonic()
        _DECISION_ROUTER_CACHE.update(
            {
                "key": key,
                "expires_at": (resolved_at + DECISION_ROUTER_DASHBOARD_CACHE_SECONDS),
                "config": resolved,
            }
        )
        return resolved


def _configured_model_roles() -> dict[str, set[str]]:
    roles: dict[str, set[str]] = {}
    _add_model_role(roles, ingest_model(), "ingest")

    try:
        audit_policy = load_audit_policy()
        if getattr(audit_policy, "enabled", True):
            _add_model_role(roles, getattr(audit_policy, "model", ""), "audit")
    except Exception:
        pass

    try:
        recall_policy = recall_runtime.load_policy()
        if getattr(recall_policy, "judge_mode", "auto") != "off":
            _add_model_role(roles, getattr(recall_policy, "judge_model", ""), "gate")
        if getattr(recall_policy, "rewrite_enabled", False):
            _add_model_role(
                roles, getattr(recall_policy, "rewrite_model", ""), "rewrite"
            )
    except Exception:
        pass

    try:
        for model in configured_models(None):
            _add_model_role(roles, model, "improve")
    except Exception:
        pass

    try:
        decision_router = _resolved_decision_router_config()
        _add_model_role(roles, decision_router.primary_model, "decision-primary")
        _add_model_role(
            roles,
            decision_router.challenger_model,
            "decision-challenger",
        )
        _add_model_role(
            roles,
            decision_router.tie_break_model,
            "decision-tie-break",
        )
    except Exception:
        pass

    with contextlib.suppress(Exception):
        _add_model_role(roles, embedding_model(), "embed")

    try:
        search_embedding = load_search_embedding_config()
        if search_embedding.enabled and search_embedding.backend == "nemotron_service":
            _add_model_role(roles, search_embedding.model, "search-embed")
    except Exception:
        pass

    try:
        reranker = load_reranker_config()
        if reranker.enabled:
            _add_model_role(roles, reranker.model, "rerank")
    except Exception:
        pass

    return roles


def _role_sort_key(role: str) -> int:
    order = {
        "ingest": 0,
        "audit": 1,
        "improve": 2,
        "gate": 3,
        "rewrite": 4,
        "decision-primary": 5,
        "decision-challenger": 6,
        "decision-tie-break": 7,
        "embed": 8,
        "search-embed": 9,
        "rerank": 10,
    }
    return order.get(role, 99)


def _resolve_model_name(
    name: str,
    installed_by_name: dict[str, dict[str, Any]],
    running_by_name: dict[str, dict[str, Any]],
) -> str:
    candidates = [name]
    if ":" not in name:
        candidates.append(f"{name}:latest")
    if name.endswith(":latest"):
        candidates.append(name.removesuffix(":latest"))
    for candidate in candidates:
        if candidate in running_by_name or candidate in installed_by_name:
            return candidate
    return name


def _external_configured_model(name: str, roles: set[str]) -> bool:
    return bool(
        roles <= {"rerank", "search-embed"}
        and roles
        and "/" in name
        and not name.startswith("hf.co/")
    )


def _model_status_snapshot(ollama: dict[str, Any] | None = None) -> dict[str, Any]:
    running_snapshot = ollama or _ollama_snapshot()
    installed_snapshot = _ollama_tags_snapshot()
    running_models = running_snapshot.get("models", [])
    installed_models = installed_snapshot.get("models", [])
    if not isinstance(running_models, list):
        running_models = []
    if not isinstance(installed_models, list):
        installed_models = []

    running_by_name = {
        _model_name(row): row for row in running_models if _model_name(row)
    }
    installed_by_name = {
        _model_name(row): row for row in installed_models if _model_name(row)
    }
    roles_by_name: dict[str, set[str]] = {}
    for name, roles in _configured_model_roles().items():
        resolved = _resolve_model_name(name, installed_by_name, running_by_name)
        roles_by_name.setdefault(resolved, set()).update(roles)
    names = sorted(set(running_by_name) | set(roles_by_name))
    unused_installed = set(installed_by_name) - set(names)
    rows: list[dict[str, Any]] = []
    missing = 0
    external = 0
    loaded_size = 0
    installed_size = 0

    for name in names:
        installed = installed_by_name.get(name)
        running = running_by_name.get(name)
        roles = roles_by_name.get(name, set())
        configured = bool(roles)
        installed_size += _numeric_bytes(installed.get("size") if installed else 0)
        loaded_size += _numeric_bytes(
            (running.get("size_vram") if isinstance(running, dict) else 0)
            or (running.get("size") if isinstance(running, dict) else 0)
        )
        is_external = (
            configured
            and not installed
            and not running
            and _external_configured_model(name, roles)
        )
        if running:
            status = "loaded"
        elif installed:
            status = "ready"
        elif is_external:
            status = "external"
            external += 1
        elif configured:
            status = "missing"
            missing += 1
        else:
            status = "unknown"
        details = {}
        capabilities: list[str] = []
        for source in (installed, running):
            if isinstance(source, dict):
                if isinstance(source.get("details"), dict):
                    details = dict(source["details"])
                if isinstance(source.get("capabilities"), list):
                    capabilities = [str(item) for item in source["capabilities"]]
        rows.append(
            {
                "name": name,
                "status": status,
                "installed": installed is not None,
                "running": running is not None,
                "configured": configured,
                "roles": sorted(roles, key=_role_sort_key),
                "size_bytes": _numeric_bytes(
                    installed.get("size") if installed else None
                ),
                "loaded_size_bytes": _numeric_bytes(
                    (running.get("size_vram") if isinstance(running, dict) else None)
                    or (running.get("size") if isinstance(running, dict) else None)
                ),
                "context_length": running.get("context_length")
                if isinstance(running, dict)
                else details.get("context_length"),
                "expires_at": running.get("expires_at")
                if isinstance(running, dict)
                else None,
                "modified_at": installed.get("modified_at")
                if isinstance(installed, dict)
                else None,
                "digest": (
                    (installed.get("digest") if isinstance(installed, dict) else None)
                    or (running.get("digest") if isinstance(running, dict) else None)
                ),
                "processor": running.get("processor")
                if isinstance(running, dict)
                else None,
                "details": details,
                "capabilities": capabilities,
            }
        )

    status_order = {"loaded": 0, "missing": 1, "ready": 2, "external": 3, "unknown": 4}
    rows.sort(
        key=lambda row: (
            0 if row["configured"] else 1,
            status_order.get(str(row["status"]), 9),
            str(row["name"]),
        )
    )
    return {
        "available": bool(
            running_snapshot.get("available") or installed_snapshot.get("available")
        ),
        "running_available": bool(running_snapshot.get("available")),
        "installed_available": bool(installed_snapshot.get("available")),
        "error": running_snapshot.get("error") or installed_snapshot.get("error"),
        "models": rows,
        "summary": {
            "installed": sum(1 for row in rows if row["installed"]),
            "loaded": len(running_by_name),
            "configured": len(roles_by_name),
            "missing": missing,
            "external": external,
            "all_installed": len(installed_by_name),
            "unused_installed": len(unused_installed),
            "installed_size_bytes": installed_size,
            "loaded_size_bytes": loaded_size,
        },
    }


def _result_count(result: dict[str, Any], key: str, fallback: Any = None) -> Any:
    value = result.get(key, fallback)
    if isinstance(value, list):
        return len(value)
    return value


def _drain_history(limit: int = 200) -> list[dict[str, Any]]:
    logs_dir = CHRONOVISOR_ROOT / "logs"
    records: list[dict[str, Any]] = []
    for path in sorted(logs_dir.glob("ingest-drain-*.jsonl"))[-10:]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in lines:
            try:
                data = json.loads(line)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            result = data.get("result")
            if not isinstance(result, dict):
                result = {}

            records.append(
                {
                    "timestamp": data.get("timestamp"),
                    "kind": "drain_batch",
                    "pending_before": data.get("pending_before"),
                    "pending_after": data.get("pending_after"),
                    "files_processed": data.get("files_processed"),
                    "files_attempted": _result_count(result, "files_attempted"),
                    "files_deferred": _result_count(result, "files_deferred", 0),
                    "files_continued": _result_count(result, "files_continued", 0),
                    "files_failed": _result_count(result, "files_failed"),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                    "batch": data.get("batch"),
                }
            )
    return records[-limit:]


def _batch_metric_identity(row: dict[str, Any]) -> tuple[Any, ...] | None:
    """Identify one batch emitted to both runtime metrics and drain history."""

    if row.get("kind") not in {"batch", "drain_batch"}:
        return None
    return tuple(
        row.get(key)
        for key in (
            "pending_before",
            "pending_after",
            "files_attempted",
            "files_processed",
            "files_deferred",
            "files_continued",
            "files_failed",
            "elapsed_seconds",
        )
    )


def _merge_metric_history(
    runtime_metrics: list[dict[str, Any]],
    drain_metrics: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Merge metric sources without counting the same completed batch twice.

    The orchestrator writes the canonical runtime metric first.  The drain
    wrapper records the same result a few seconds later, so timestamp-only UI
    deduplication cannot reliably recognize the duplicate.
    """

    runtime_batch_identities = {
        identity
        for row in runtime_metrics
        if (identity := _batch_metric_identity(row)) is not None
    }
    unique_drain_metrics = [
        row
        for row in drain_metrics
        if _batch_metric_identity(row) not in runtime_batch_identities
    ]
    merged = sorted(
        runtime_metrics + unique_drain_metrics,
        key=lambda item: str(item.get("timestamp") or ""),
    )
    return merged[-limit:]


def _recent_log_events(limit: int = 80) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in runtime_status.tail_text_lines(LOG_FILE, limit=limit):
        match = LOG_LINE_RE.match(line)
        if not match:
            continue
        message = match.group("message")
        events.append(
            {
                "timestamp": match.group("time"),
                "level": runtime_status.classify_log_message(message),
                "message": message,
                "source": "log.md",
            }
        )
    return events


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _read_jsonl_file(path: Path, limit: int) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            data = json.loads(line)
        except Exception:
            continue
        if isinstance(data, dict):
            rows.append(data)
    return rows


def _shorten(value: object, limit: int = 180) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}..."


def _basename(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value).name


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _date_from_value(value: object) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _raw_file_date(path: Path) -> date | None:
    match = RAW_DATE_RE.search(path.name)
    if match:
        try:
            return datetime.strptime(match.group("stamp"), "%Y%m%d").date()
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).date()
    except OSError:
        return None


def _raw_source_label(filename: str) -> str:
    lower = filename.lower()
    if SEMANTIC_PROJECTION_CHILD_RE.fullmatch(lower):
        return "projection"
    if "claude-code" in lower:
        return "claude-code"
    if "codex" in lower:
        return "codex"
    if lower.startswith("vestige-"):
        return "vestige"
    if "ingest" in lower:
        return "ingest"
    return "manual"


def _knowledge_category_label(category: str) -> str:
    special = {
        "ai": "AI",
        "bom": "BOM",
        "cad": "CAD",
        "jt": "JT",
        "jttok": "JTTOK",
        "qmk": "QMK",
        "vis": "VIS",
    }
    if category in special:
        return special[category]
    return category.replace("-", " ").replace("_", " ").title()


def _knowledge_mix_snapshot() -> dict[str, Any]:
    pages_dir = CHRONOVISOR_ROOT / "pages"
    categories: dict[str, dict[str, Any]] = {}
    if pages_dir.exists():
        for path in pages_dir.rglob("*.md"):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(pages_dir)
                page_bytes = path.stat().st_size
            except Exception:
                continue
            category = rel.parts[0] if len(rel.parts) > 1 else "root"
            row = categories.setdefault(
                category,
                {
                    "id": category,
                    "label": _knowledge_category_label(category),
                    "pages": 0,
                    "bytes": 0,
                    "samples": [],
                },
            )
            row["pages"] += 1
            row["bytes"] += page_bytes
            _add_sample(row, "samples", str(rel), limit=4)

    total_pages = sum(int(row["pages"]) for row in categories.values())
    total_bytes = sum(int(row["bytes"]) for row in categories.values())
    rows = sorted(
        categories.values(), key=lambda row: (-int(row["bytes"]), str(row["label"]))
    )
    for row in rows:
        row["share"] = (int(row["bytes"]) / total_bytes) if total_bytes else 0.0

    return {
        "total_pages": total_pages,
        "total_bytes": total_bytes,
        "categories": rows,
        "top": rows[:8],
        "paths": {
            "pages_dir": str(pages_dir),
        },
    }


def _new_save_day(day: date) -> dict[str, Any]:
    return {
        "date": day.isoformat(),
        "raw_saved": 0,
        "raw_bytes": 0,
        "processed_bytes": 0,
        "pending_bytes": 0,
        "deferred_bytes": 0,
        "failed_bytes": 0,
        "raw_segments": [],
        "processed": 0,
        "attempted": 0,
        "succeeded": 0,
        "deferred": 0,
        "continued": 0,
        "failed": 0,
        "pages_created": 0,
        "pages_updated": 0,
        "sources": {},
        "raw_samples": [],
        "page_samples": [],
    }


def _compact_raw_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bound dashboard payload while preserving exact byte/status totals."""

    ordered = sorted(segments, key=lambda item: str(item.get("name") or ""))
    if len(ordered) <= SAVE_HISTORY_MAX_SEGMENTS_PER_DAY:
        return ordered

    grouped: dict[str, dict[str, Any]] = {}
    for segment in ordered:
        status = str(segment.get("status") or "pending")
        aggregate = grouped.setdefault(
            status,
            {
                "name": "",
                "bytes": 0,
                "status": status,
                "source": "aggregate",
                "count": 0,
            },
        )
        aggregate["bytes"] += _int_value(segment.get("bytes"))
        aggregate["count"] += 1
    compacted: list[dict[str, Any]] = []
    for status in ("processed", "pending", "deferred", "failed"):
        aggregate = grouped.pop(status, None)
        if aggregate is None:
            continue
        count = int(aggregate.pop("count"))
        aggregate["name"] = f"{count} {status} Raw saves"
        compacted.append(aggregate)
    for status, aggregate in sorted(grouped.items()):
        count = int(aggregate.pop("count"))
        aggregate["name"] = f"{count} {status} Raw saves"
        compacted.append(aggregate)
    return compacted


def _add_sample(row: dict[str, Any], key: str, value: str, limit: int = 6) -> None:
    samples = row.setdefault(key, [])
    if isinstance(samples, list) and value not in samples and len(samples) < limit:
        samples.append(value)


def _operational_deferred_raw_statuses(raw_paths: list[Path]) -> dict[str, str]:
    """Return active queue holds keyed by immutable raw filename."""

    from chronovisor.decision.failure_supervisor import operational_deferred_raw_files

    return operational_deferred_raw_files(raw_paths)


def _semantic_deferred_raw_names(raw_paths: list[Path]) -> list[str]:
    """Return active semantic holds without including operational repairs."""

    deferred = _operational_deferred_raw_statuses(raw_paths)
    return sorted(
        raw_file
        for raw_file, status in deferred.items()
        if status == "semantic_no_quorum"
    )


def _projection_parent_name(
    raw_dir: Path,
    source_parent: dict[str, Any],
    *,
    raw_store: Any | None = None,
    verified_archives: set[Path] | None = None,
) -> str | None:
    """Resolve one manifest parent without trusting its path-like fields."""

    receipt = source_parent.get("receipt")
    if not isinstance(receipt, dict):
        return None
    host = receipt.get("host")
    session_key = receipt.get("session_key")
    after_line = receipt.get("after_line")
    until_line = receipt.get("until_line")
    idempotency_key = receipt.get("idempotency_key")
    expected_sha256 = source_parent.get("raw_sha256")
    if (
        host not in {"codex", "claude-code"}
        or not isinstance(session_key, str)
        or re.fullmatch(r"[0-9a-f]{24}", session_key) is None
        or isinstance(after_line, bool)
        or not isinstance(after_line, int)
        or after_line < 0
        or isinstance(until_line, bool)
        or not isinstance(until_line, int)
        or until_line <= after_line
        or not isinstance(idempotency_key, str)
        or re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256)) is None
    ):
        return None
    expected_idempotency_key = f"{host}-{session_key}-from{after_line}-to{until_line}"
    if idempotency_key != expected_idempotency_key:
        return None
    parent_name = f"save-{expected_idempotency_key}.md"
    try:
        if raw_store is None:
            from chronovisor.raw.raw_store import RawStore

            raw_store = RawStore(raw_dir)
        unit = raw_store.resolve(parent_name)
        if unit is None:
            return None
        if unit.storage == "legacy_archive" and unit.sha256 is not None:
            # Verify the compressed archive object once, then use the member
            # digest bound into that verified manifest. Reopening each member
            # separately repeatedly decompressed the same tar stream.
            from chronovisor.raw.legacy_archive import verify_legacy_manifest

            member = unit.archive_member
            archive_path = getattr(member, "archive_path", None)
            manifest_path = getattr(member, "manifest_path", None)
            if not isinstance(archive_path, Path) or not isinstance(
                manifest_path, Path
            ):
                return None
            verified = verified_archives if verified_archives is not None else set()
            if archive_path not in verified:
                verify_legacy_manifest(manifest_path, full=False)
                verified.add(archive_path)
            observed_sha256 = unit.sha256
        else:
            # Flat files and Raw v2 ranges still receive byte-level validation.
            observed_sha256 = hashlib.sha256(raw_store.read_bytes(unit)).hexdigest()
    except (OSError, ValueError):
        return None
    return parent_name if observed_sha256 == expected_sha256 else None


def _projection_parent_raw_names_by_child(
    raw_dir: Path,
    child_names: set[str],
) -> dict[str, set[str]]:
    """Resolve projection children to verified lossless saved raws."""

    from chronovisor.raw.raw_semantic_projection import verify_projection_bundle
    from chronovisor.raw.raw_store import RawStore

    raw_store = RawStore(raw_dir)
    verified_archives: set[Path] = set()
    children_by_projection: dict[str, set[str]] = {}
    for child_name in child_names:
        match = SEMANTIC_PROJECTION_CHILD_RE.fullmatch(child_name)
        if match is not None:
            children_by_projection.setdefault(match.group("projection"), set()).add(
                child_name
            )
    parents_by_child: dict[str, set[str]] = {}
    for projection_id, requested_children in children_by_projection.items():
        manifest_name = f"semantic-{projection_id}.manifest.json"
        manifest_candidates = (
            raw_dir / manifest_name,
            raw_dir.parent
            / "runtime"
            / "raw-projections"
            / "artifacts"
            / manifest_name,
        )
        manifest_path = next(
            (path for path in manifest_candidates if path.is_file()),
            manifest_candidates[0],
        )
        try:
            manifest = verify_projection_bundle(manifest_path)
        except Exception:
            continue
        manifest_children = manifest.get("children")
        if not isinstance(manifest_children, list):
            continue
        matched_children = requested_children.intersection(
            str(row.get("filename"))
            for row in manifest_children
            if isinstance(row, dict) and isinstance(row.get("filename"), str)
        )
        if not matched_children:
            continue
        source = manifest.get("source")
        source_parents = source.get("parents") if isinstance(source, dict) else None
        if not isinstance(source_parents, list):
            continue
        parent_names: set[str] = set()
        for source_parent in source_parents:
            if not isinstance(source_parent, dict):
                continue
            parent_name = _projection_parent_name(
                raw_dir,
                source_parent,
                raw_store=raw_store,
                verified_archives=verified_archives,
            )
            if parent_name is not None:
                parent_names.add(parent_name)
        for child_name in matched_children:
            if parent_names:
                parents_by_child[child_name] = set(parent_names)
    return parents_by_child


def _projection_save_states(
    raw_dir: Path,
    raw_files: dict[str, dict[str, Any]],
    raw_paths: list[Path],
    processed_raw_names: set[str],
    *,
    deferred_statuses: dict[str, str] | None = None,
) -> tuple[set[str], set[str]]:
    """Return active semantic-deferred and unresolved-pending saved raws."""

    active_deferred = (
        deferred_statuses
        if deferred_statuses is not None
        else _operational_deferred_raw_statuses(raw_paths)
    )
    semantic_deferred_raws = {
        name
        for name, reason in active_deferred.items()
        if reason == "semantic_no_quorum"
    }
    saved_names = set(raw_files)
    child_names = {
        path.name
        for path in raw_paths
        if SEMANTIC_PROJECTION_CHILD_RE.fullmatch(path.name) is not None
    }
    unresolved_children = child_names - processed_raw_names
    relevant_children = unresolved_children | (semantic_deferred_raws & child_names)
    parents_by_child = _projection_parent_raw_names_by_child(
        raw_dir,
        relevant_children,
    )
    semantic_deferred_saves = semantic_deferred_raws & saved_names
    pending_saves: set[str] = set()
    for child_name, parent_names in parents_by_child.items():
        if child_name in semantic_deferred_raws:
            semantic_deferred_saves.update(parent_names & saved_names)
        else:
            pending_saves.update(parent_names & saved_names)
    pending_saves -= semantic_deferred_saves
    return semantic_deferred_saves, pending_saves


def _save_history_snapshot(
    days: int = 371,
    today: date | None = None,
    *,
    raw_paths: list[Path] | None = None,
    deferred_statuses: dict[str, str] | None = None,
) -> dict[str, Any]:
    end = today or datetime.now().date()
    start = end - timedelta(days=max(1, days) - 1)
    rows = {
        (start + timedelta(days=offset)).isoformat(): _new_save_day(
            start + timedelta(days=offset)
        )
        for offset in range((end - start).days + 1)
    }

    raw_dir = CHRONOVISOR_ROOT / "raw"
    raw_files: dict[str, dict[str, Any]] = {}
    raw_status: dict[str, str] = {}
    raw_entries: list[tuple[Path, str, int, date | None]] = []
    effective_raw_paths: list[Path] = []
    if raw_paths is not None:
        effective_raw_paths = list(raw_paths)
        raw_entries = [
            (path, path.name, path.stat().st_size, _raw_file_date(path))
            for path in effective_raw_paths
        ]
    elif raw_dir.exists():
        from chronovisor.raw.raw_store import RawStore

        store = RawStore(raw_dir)
        reference_dir = raw_dir.parent / "runtime" / "raw-projections" / "parents"
        for unit in store.iter_units():
            logical_path = (
                unit.path
                if unit.storage == "legacy_file"
                else store.materialize_ingest(unit, reference_dir)
            )
            effective_raw_paths.append(logical_path)
            captured = None
            if unit.captured_at:
                try:
                    captured = datetime.fromisoformat(unit.captured_at).date()
                except ValueError:
                    captured = None
            raw_entries.append(
                (
                    logical_path,
                    unit.raw_id,
                    unit.length,
                    captured or _raw_file_date(logical_path),
                )
            )
        artifact_dir = raw_dir.parent / "runtime" / "raw-projections" / "artifacts"
        if artifact_dir.exists():
            effective_raw_paths.extend(sorted(artifact_dir.glob("*.md")))
    if raw_dir.exists():
        for path, raw_name, raw_bytes, raw_date in raw_entries:
            # Projection children are generated processing artifacts.  The
            # original lossless parent is already counted as the save, so
            # including children would double-count bytes and invent a
            # "manual" user save on the projection date.  Queue cardinality
            # remains visible through the canonical pending counter.
            if SEMANTIC_PROJECTION_CHILD_RE.fullmatch(raw_name.lower()):
                continue
            if raw_date is None or raw_date < start or raw_date > end:
                continue
            source = _raw_source_label(raw_name)
            raw_files[raw_name] = {
                "date": raw_date.isoformat(),
                "bytes": raw_bytes,
                "source": source,
            }
            row = rows[raw_date.isoformat()]
            row["raw_saved"] += 1
            row["raw_bytes"] += raw_bytes
            row["sources"][source] = row["sources"].get(source, 0) + 1
            _add_sample(row, "raw_samples", path.name)

    logs_dir = CHRONOVISOR_ROOT / "logs"
    if logs_dir.exists():
        for path in sorted(logs_dir.glob("ingest-drain-*.jsonl")):
            for record in _read_jsonl_file(path, limit=50_000):
                record_date = _date_from_value(record.get("timestamp"))
                if record_date is None or record_date < start or record_date > end:
                    continue
                row = rows[record_date.isoformat()]
                result = (
                    record.get("result")
                    if isinstance(record.get("result"), dict)
                    else {}
                )
                attempted_files = (
                    result.get("files_attempted")
                    if isinstance(result.get("files_attempted"), list)
                    else []
                )
                processed_files = (
                    result.get("files_processed")
                    if isinstance(result.get("files_processed"), list)
                    else []
                )
                deferred_files = (
                    result.get("files_deferred")
                    if isinstance(result.get("files_deferred"), list)
                    else []
                )
                continued_files = (
                    result.get("files_continued")
                    if isinstance(result.get("files_continued"), list)
                    else []
                )
                per_raw = (
                    result.get("per_raw")
                    if isinstance(result.get("per_raw"), list)
                    else []
                )

                processed = _int_value(record.get("files_processed")) or len(
                    processed_files
                )
                if per_raw:
                    succeeded = sum(
                        1
                        for item in per_raw
                        if isinstance(item, dict) and item.get("succeeded") is True
                    )
                    attempted = len(per_raw)
                    deferred = sum(
                        1
                        for item in per_raw
                        if isinstance(item, dict)
                        and (
                            item.get("deferred") is True
                            or (
                                isinstance(item.get("supervision"), dict)
                                and item["supervision"].get("terminal_deferred") is True
                            )
                        )
                    )
                    continued = sum(
                        1
                        for item in per_raw
                        if isinstance(item, dict) and item.get("continued") is True
                    )
                    failed = sum(
                        1
                        for item in per_raw
                        if isinstance(item, dict)
                        and item.get("succeeded") is False
                        and item.get("deferred") is not True
                        and item.get("continued") is not True
                        and not (
                            isinstance(item.get("supervision"), dict)
                            and item["supervision"].get("terminal_deferred") is True
                        )
                    )
                else:
                    attempted = len(attempted_files) or processed
                    succeeded = processed
                    deferred = len(deferred_files)
                    continued = len(continued_files)
                    failed = max(0, attempted - succeeded - deferred - continued)

                row["processed"] += processed
                row["attempted"] += attempted
                row["succeeded"] += succeeded
                row["deferred"] += deferred
                row["continued"] += continued
                row["failed"] += failed
                for filename in processed_files:
                    if isinstance(filename, str):
                        raw_status[filename] = "processed"
                if per_raw:
                    for item in per_raw:
                        if not isinstance(item, dict):
                            continue
                        filename = item.get("filename") or item.get("raw_file")
                        source_files = item.get("source_files")
                        status_filenames = (
                            [name for name in source_files if isinstance(name, str)]
                            if isinstance(source_files, list)
                            else []
                        )
                        if (
                            isinstance(filename, str)
                            and filename not in status_filenames
                        ):
                            status_filenames.append(filename)
                        item_deferred = item.get("deferred") is True or (
                            isinstance(item.get("supervision"), dict)
                            and item["supervision"].get("terminal_deferred") is True
                        )
                        item_continued = item.get("continued") is True
                        if item.get("succeeded") is True:
                            for status_filename in status_filenames:
                                raw_status[status_filename] = "processed"
                        elif item_deferred:
                            for status_filename in status_filenames:
                                if raw_status.get(status_filename) != "processed":
                                    # A historical defer is not a failed save.
                                    # The live authority projection below turns
                                    # it into deferred while active; after that
                                    # authority changes, it returns to pending
                                    # until a later attempt establishes a new
                                    # terminal state.
                                    raw_status[status_filename] = "deferred"
                        elif item_continued:
                            for status_filename in status_filenames:
                                if raw_status.get(status_filename) != "processed":
                                    raw_status[status_filename] = "continued"
                        else:
                            for status_filename in status_filenames:
                                if raw_status.get(status_filename) != "processed":
                                    raw_status[status_filename] = "failed"
                else:
                    processed_names = {
                        name for name in processed_files if isinstance(name, str)
                    }
                    deferred_names = {
                        name for name in deferred_files if isinstance(name, str)
                    }
                    continued_names = {
                        name for name in continued_files if isinstance(name, str)
                    }
                    for filename in deferred_names:
                        if raw_status.get(filename) != "processed":
                            raw_status[filename] = "deferred"
                    for filename in attempted_files:
                        if (
                            isinstance(filename, str)
                            and filename not in processed_names
                            and filename not in deferred_names
                            and filename not in continued_names
                            and raw_status.get(filename) != "processed"
                        ):
                            raw_status[filename] = "failed"
                for filename in processed_files[:3]:
                    if isinstance(filename, str):
                        _add_sample(row, "raw_samples", filename)

    # Drain logs are an event history, not the canonical queue state. They can
    # be rotated or miss an older successful retry, which previously made an
    # already-processed raw appear as pending in the Save Load chart. Reconcile
    # with the orchestrator state after reading the logs. Canonical processed
    # state wins over an older failed attempt for the same immutable raw.
    orchestrator_state = _read_json_file(CHRONOVISOR_ROOT / ".orchestrator_state.json") or {}
    processed_raw_files = orchestrator_state.get("processed_raw_files")
    processed_raw_names = (
        {filename for filename in processed_raw_files if isinstance(filename, str)}
        if isinstance(processed_raw_files, list)
        else set()
    )
    if isinstance(processed_raw_files, list):
        for filename in processed_raw_files:
            if isinstance(filename, str) and filename in raw_files:
                raw_status[filename] = "processed"

    semantic_deferred, projection_pending = _projection_save_states(
        raw_dir,
        raw_files,
        effective_raw_paths,
        processed_raw_names,
        deferred_statuses=deferred_statuses,
    )

    try:
        log_lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    except Exception:
        log_lines = []
    for line in log_lines:
        match = LOG_PAGE_CHANGE_RE.match(line)
        if not match:
            continue
        changed_date = _date_from_value(match.group("time"))
        if changed_date is None or changed_date < start or changed_date > end:
            continue
        row = rows[changed_date.isoformat()]
        kind = match.group("kind")
        if kind == "created":
            row["pages_created"] += 1
        else:
            row["pages_updated"] += 1
        _add_sample(row, "page_samples", f"{kind} {match.group('page')}")

    days_list: list[dict[str, Any]] = []
    totals = {
        "raw_saved": 0,
        "raw_bytes": 0,
        "processed_bytes": 0,
        "pending_bytes": 0,
        "deferred_bytes": 0,
        "failed_bytes": 0,
        "processed": 0,
        "attempted": 0,
        "succeeded": 0,
        "deferred": 0,
        "continued": 0,
        "failed": 0,
        "pages_created": 0,
        "pages_updated": 0,
        "days_with_saves": 0,
    }
    source_totals: dict[str, int] = {}
    segment_detail_start = end - timedelta(
        days=min(max(1, days), SAVE_HISTORY_SEGMENT_DETAIL_DAYS) - 1
    )
    for filename, meta in raw_files.items():
        row = rows.get(str(meta["date"]))
        if not row:
            continue
        raw_bytes = int(meta["bytes"])
        status = raw_status.get(filename)
        if filename in semantic_deferred:
            row["deferred_bytes"] += raw_bytes
            status = "deferred"
        elif filename in projection_pending:
            row["pending_bytes"] += raw_bytes
            status = "pending"
        elif status == "processed":
            row["processed_bytes"] += raw_bytes
        elif status == "failed":
            row["failed_bytes"] += raw_bytes
        else:
            row["pending_bytes"] += raw_bytes
            status = "pending"
        # The year heatmap and detail card use day aggregates.  Only the
        # 30-day load chart needs per-Raw hover regions, so do not ship a year
        # of segment names that the client never reads.
        if date.fromisoformat(str(meta["date"])) >= segment_detail_start:
            segments = row.setdefault("raw_segments", [])
            if isinstance(segments, list):
                segments.append(
                    {
                        "name": filename,
                        "bytes": raw_bytes,
                        "status": status,
                        "source": meta.get("source") or _raw_source_label(filename),
                    }
                )
    for row in rows.values():
        sources = row.get("sources") if isinstance(row.get("sources"), dict) else {}
        for source, count in sources.items():
            source_totals[source] = source_totals.get(source, 0) + int(count)
        row["sources"] = [
            {"name": source, "count": count}
            for source, count in sorted(
                sources.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        raw_segments = row.get("raw_segments")
        if isinstance(raw_segments, list):
            row["raw_segments"] = _compact_raw_segments(raw_segments)
        if (
            row["raw_saved"]
            or row["processed"]
            or row["pages_created"]
            or row["pages_updated"]
        ):
            totals["days_with_saves"] += 1
        for key in (
            "raw_saved",
            "raw_bytes",
            "processed_bytes",
            "pending_bytes",
            "deferred_bytes",
            "failed_bytes",
            "processed",
            "attempted",
            "succeeded",
            "deferred",
            "continued",
            "failed",
            "pages_created",
            "pages_updated",
        ):
            totals[key] += row[key]
        days_list.append(row)

    recent = [
        row
        for row in days_list
        if row["raw_saved"]
        or row["processed"]
        or row["pages_created"]
        or row["pages_updated"]
    ][-14:]
    return {
        "range": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "days": len(days_list),
        },
        "days": days_list,
        "recent": recent,
        "totals": totals,
        "sources": [
            {"name": source, "count": count}
            for source, count in sorted(
                source_totals.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "paths": {
            "raw_dir": str(raw_dir),
            "drain_logs": str(logs_dir),
            "log_file": str(LOG_FILE),
        },
    }


def _failure_detail(
    packet: dict[str, Any] | None, record: dict[str, Any] | None = None
) -> dict[str, Any]:
    packet = packet or {}
    record = record or {}
    return {
        "failure_id": record.get("failure_id") or packet.get("failure_id"),
        "raw_file": record.get("raw_file") or packet.get("raw_file"),
        "failure_class": record.get("failure_class") or packet.get("failure_class"),
        "fingerprint": record.get("fingerprint") or packet.get("fingerprint"),
        "attempts": packet.get("attempts"),
        "packet_status": packet.get("status"),
        "packet_path": packet.get("_path"),
    }


def _retry_detail(action: dict[str, Any]) -> dict[str, Any] | None:
    retry = action.get("retry")
    if not isinstance(retry, dict):
        return None
    return {
        "triggered": retry.get("triggered"),
        "files_attempted": retry.get("files_attempted"),
        "files_processed": retry.get("files_processed"),
        "elapsed_seconds": retry.get("elapsed_seconds"),
    }


def _self_heal_packet_index() -> dict[str, dict[str, Any]]:
    packets_dir = CHRONOVISOR_ROOT / "runtime" / "failures" / "packets"
    packets: dict[str, dict[str, Any]] = {}
    if not packets_dir.exists():
        return packets
    for path in sorted(packets_dir.glob("*.json")):
        packet = _read_json_file(path)
        if not packet:
            continue
        if (
            packet.get("failure_class") == "ingest.semantic_no_quorum"
            or packet.get("status") == "superseded_semantic_defer"
        ):
            continue
        failure_id = packet.get("failure_id")
        if isinstance(failure_id, str):
            packets[failure_id] = {**packet, "_path": str(path)}
    return packets


def _frontier_preflight_snapshot() -> dict[str, Any]:
    """Read the durable repair guard without starting Codex or any subprocess.

    The legacy function/response name is retained for dashboard API
    compatibility.  A real Codex preflight belongs inside an admitted repair
    attempt, immediately before that one process starts.
    """
    checked_at = datetime.now().isoformat(timespec="seconds")
    try:
        from chronovisor.decision.frontier_guard import FrontierGuard

        inspection = FrontierGuard(
            CHRONOVISOR_ROOT / "runtime" / "frontier-repair"
        ).inspect(dry_run=True)
        state = inspection.state
        incidents = state.get("incidents") if isinstance(state, dict) else {}
        incidents = incidents if isinstance(incidents, dict) else {}
        active_id = state.get("active_incident_id") if isinstance(state, dict) else None
        started = sum(
            1
            for incident in incidents.values()
            if isinstance(incident, dict) and incident.get("started_at")
        )
        summary = {
            "ok": True,
            "checked_at": checked_at,
            "mode": "on_demand_only",
            "state": "active" if active_id else "standby",
            "active_incident_id": active_id,
            "incidents_started": started,
            "would_abandon": list(inspection.would_abandon),
            "subprocess_checked": False,
        }
    except Exception as exc:
        summary = {
            "ok": False,
            "checked_at": checked_at,
            "mode": "guard_state_unreadable",
            "state": "blocked",
            "subprocess_checked": False,
            "error": str(exc),
        }
    return summary


def _frontier_activity_snapshot() -> dict[str, Any]:
    active_dir = CHRONOVISOR_ROOT / "runtime" / "frontier-reviews" / "active"
    records: list[dict[str, Any]] = []
    now = datetime.now()
    if active_dir.exists():
        for path in sorted(active_dir.glob("*.json")):
            record = _read_json_file(path)
            if not record:
                continue
            started_raw = record.get("started_at")
            try:
                started = (
                    datetime.fromisoformat(started_raw)
                    if isinstance(started_raw, str)
                    else None
                )
            except ValueError:
                started = None
            age_seconds = (
                max(
                    0.0,
                    (
                        (datetime.now(started.tzinfo) if started.tzinfo else now)
                        - started
                    ).total_seconds(),
                )
                if started is not None
                else None
            )
            pid = record.get("pid")
            stale = not _job_process_identity_matches(pid, started_raw) or (
                age_seconds is not None
                and age_seconds > FRONTIER_ACTIVITY_STALE_SECONDS
            )
            if stale:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
                continue
            records.append({**record, "elapsed_seconds": age_seconds})
    records.sort(key=lambda row: str(row.get("started_at") or ""))
    return {
        "active": bool(records),
        "count": len(records),
        "reviews": records,
        "latest": records[-1] if records else None,
    }


def _activity_age_seconds(started_raw: object) -> float | None:
    if not isinstance(started_raw, str):
        return None
    try:
        started = datetime.fromisoformat(started_raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    now = datetime.now(started.tzinfo) if started.tzinfo else datetime.now()
    return max(0.0, (now - started).total_seconds())


def _local_consensus_activities() -> list[dict[str, Any]]:
    """Read only live redacted markers, without scanning consensus history."""

    active_dir = CHRONOVISOR_ROOT / "runtime" / "local-consensus" / "active"
    activities: list[dict[str, Any]] = []
    if active_dir.exists():
        for path in sorted(active_dir.glob("*.json")):
            row = _read_json_file(path)
            pid = row.get("pid") if row else None
            age_seconds = _activity_age_seconds(row.get("started_at")) if row else None
            stale = (
                not row
                or not _job_process_identity_matches(pid, row.get("started_at"))
                or age_seconds is None
                or age_seconds > LOCAL_CONSENSUS_ACTIVITY_STALE_SECONDS
            )
            if stale:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
                continue
            activities.append(
                {
                    "request_sha256": row.get("request_sha256"),
                    "role": row.get("role"),
                    "model": row.get("model"),
                    "phase": row.get("phase"),
                    "attempt": row.get("attempt"),
                    "started_at": row.get("started_at"),
                    "updated_at": row.get("updated_at"),
                    "pid": pid,
                    "thread_id": row.get("thread_id"),
                    "elapsed_seconds": age_seconds,
                }
            )
    activities.sort(key=lambda row: str(row.get("started_at") or ""))
    return activities


def _model_activities() -> list[dict[str, Any]]:
    """Read direct Ollama calls that do not have a consensus marker."""

    root = CHRONOVISOR_ROOT / "runtime" / "model-activity"
    active_dir = root / "active"
    activities: list[dict[str, Any]] = []
    if active_dir.exists():
        for path in sorted(active_dir.glob("*.json")):
            row = _read_json_file(path)
            pid = row.get("pid") if row else None
            age_seconds = _activity_age_seconds(row.get("started_at")) if row else None
            stale = (
                not row
                or row.get("schema_version") != 1
                or not _job_process_identity_matches(pid, row.get("started_at"))
                or age_seconds is None
                or age_seconds > MODEL_ACTIVITY_STALE_SECONDS
            )
            if stale:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
                continue
            activities.append(
                {
                    "activity_id": row.get("activity_id"),
                    "pipeline": row.get("pipeline"),
                    "component": row.get("component"),
                    "caller": row.get("caller"),
                    "operation": row.get("operation"),
                    "model": row.get("model"),
                    "started_at": row.get("started_at"),
                    "updated_at": row.get("updated_at"),
                    "pid": pid,
                    "thread_id": row.get("thread_id"),
                    "elapsed_seconds": age_seconds,
                    "recent": False,
                }
            )
    recent_dir = root / "recent"
    if recent_dir.exists():
        for path in sorted(recent_dir.glob("*.json")):
            row = _read_json_file(path)
            finished_age = (
                _activity_age_seconds(row.get("finished_at")) if row else None
            )
            if (
                not row
                or row.get("schema_version") != 1
                or finished_age is None
                or finished_age > MODEL_ACTIVITY_VISIBLE_SECONDS
            ):
                continue
            activities.append(
                {
                    "activity_id": row.get("activity_id"),
                    "pipeline": row.get("pipeline"),
                    "component": row.get("component"),
                    "caller": row.get("caller"),
                    "operation": row.get("operation"),
                    "model": row.get("model"),
                    "started_at": row.get("started_at"),
                    "updated_at": row.get("updated_at"),
                    "finished_at": row.get("finished_at"),
                    "pid": row.get("pid"),
                    "thread_id": row.get("thread_id"),
                    "elapsed_seconds": _activity_age_seconds(row.get("started_at")),
                    "recent": True,
                }
            )
    activities.sort(key=lambda row: str(row.get("started_at") or ""))
    return activities


_PROCESSING_LANES: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    (
        "ingest",
        "Ingest",
        (
            ("raw", "Raw"),
            ("triage", "Triage"),
            ("generate", "Generate"),
            ("consensus", "Consensus"),
            ("apply", "Apply"),
        ),
    ),
    (
        "recall",
        "Recall",
        (
            ("search", "Search"),
            ("rerank", "Rerank"),
            ("primary", "Primary"),
            ("challenger", "Challenger"),
            ("tie_break", "Tie-break"),
            ("commit", "Commit"),
        ),
    ),
    (
        "audit",
        "Audit",
        (
            ("select", "Select"),
            ("inspect", "Inspect"),
            ("consensus", "Consensus"),
            ("report", "Report"),
        ),
    ),
    (
        "improve",
        "Improve",
        (
            ("discover", "Discover"),
            ("generate", "Generate"),
            ("verify", "Verify"),
            ("apply", "Apply"),
        ),
    ),
    (
        "repair",
        "Repair",
        (
            ("detect", "Detect"),
            ("local_fix", "Local fix"),
            ("verify", "Verify"),
            ("escalate", "Escalate"),
        ),
    ),
    (
        "typed_graph",
        "Typed Graph",
        (
            ("discover", "Discover"),
            ("extract", "Extract"),
            ("verify", "Verify"),
            ("consolidate", "Consolidate"),
            ("evaluate", "Evaluate"),
            ("promote", "Promote"),
        ),
    ),
)


def _processing_pipeline_for_role(role: object) -> str:
    normalized = str(role or "").casefold()
    if normalized.startswith(("relation_", "entity_merge", "recall_rubric")):
        return "typed_graph"
    if normalized.startswith("recall"):
        return "recall"
    if normalized.startswith("ingest"):
        return "ingest"
    if normalized.startswith(("model_eval", "autonomy", "orphan_link")):
        return "improve"
    if "repair" in normalized:
        return "repair"
    return "audit"


def _processing_consensus_step(pipeline: str, role: object, phase: object) -> str:
    normalized_role = str(role or "").casefold()
    normalized_phase = str(phase or "trigger").casefold()
    if pipeline == "typed_graph":
        return "verify" if normalized_phase in {"validate", "vote"} else "extract"
    if pipeline == "recall":
        if normalized_role.endswith(":tie_break"):
            return "tie_break"
        if normalized_role.endswith(":challenger"):
            return "challenger"
        return "primary"
    if pipeline == "ingest":
        return "consensus"
    if pipeline == "improve":
        return "verify" if normalized_phase in {"validate", "vote"} else "generate"
    if pipeline == "repair":
        return "verify" if normalized_phase in {"validate", "vote"} else "local_fix"
    return "consensus" if normalized_phase in {"validate", "vote"} else "inspect"


def _processing_model_step(pipeline: str, operation: object) -> str:
    if pipeline == "typed_graph":
        return "verify" if operation in {"verify", "judge"} else "extract"
    if pipeline == "recall":
        if operation == "rerank":
            return "rerank"
        if operation == "search":
            return "search"
        return "primary"
    if pipeline == "ingest":
        return "generate"
    if pipeline == "improve":
        return "generate"
    if pipeline == "repair":
        return "local_fix"
    return "inspect"


def _processing_component_label(component: object) -> str:
    value = str(component or "model worker").rsplit(".", 1)[-1]
    for suffix in ("_worker", "_runtime"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return value.replace("_", " ").strip().title() or "Model Worker"


def _processing_step_rows(
    definitions: tuple[tuple[str, str], ...], current_step: str | None
) -> list[dict[str, str]]:
    keys = [key for key, _label in definitions]
    active_index = keys.index(current_step) if current_step in keys else -1
    return [
        {
            "key": key,
            "label": label,
            "status": (
                "done"
                if active_index >= 0 and index < active_index
                else "active"
                if index == active_index
                else "pending"
            ),
        }
        for index, (key, label) in enumerate(definitions)
    ]


def _build_processing_activity_snapshot() -> dict[str, Any]:
    """Return a cheap live projection for the dashboard processing lanes."""

    cached_status = runtime_status.read_status()
    pending_value = cached_status.get("pending")
    pending = (
        int(pending_value)
        if isinstance(pending_value, int) and not isinstance(pending_value, bool)
        else 0
    )
    status = _canonicalize_runtime_status(
        cached_status,
        orchestrator._load_state(),
        pending=max(0, pending),
    )
    _mark_batch_activity(status)
    activities = _local_consensus_activities()
    model_activities = _model_activities()
    frontier_reviews = _frontier_activity_snapshot()
    frontier_repair = _frontier_repair_snapshot(limit=8)
    typed_graph = _typed_graph_dashboard_snapshot()

    active_by_lane: dict[str, dict[str, Any]] = {}

    llm = status.get("llm") if isinstance(status.get("llm"), dict) else {}
    batch = status.get("batch") if isinstance(status.get("batch"), dict) else {}
    ingest_active = bool(
        status.get("current_job_id")
        or status.get("current_raw")
        or llm.get("active") is True
        or batch.get("active") is True
    )
    if ingest_active:
        raw_stage = str(status.get("stage") or status.get("current_op") or "raw")
        stage_aliases = {
            "batch": "raw",
            "raw": "raw",
            "triage": "triage",
            "generate": "generate",
            "local-regenerate": "generate",
            "frontier-regenerate": "generate",
            "authorization": "consensus",
            "local-consensus-review": "consensus",
            "frontier-review": "consensus",
            "locked": "consensus",
            "apply": "apply",
        }
        active_by_lane["ingest"] = {
            "current_step": stage_aliases.get(raw_stage, "raw"),
            "model": str(llm.get("model") or ingest_model()),
            "role": str(status.get("current_op") or raw_stage),
            "started_at": llm.get("started_at") or status.get("updated_at"),
            "updated_at": llm.get("updated_at") or status.get("updated_at"),
            "work_item": status.get("current_raw") or status.get("current_job_id"),
            "active_jobs": 1,
        }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for activity in activities:
        grouped.setdefault(
            _processing_pipeline_for_role(activity.get("role")), []
        ).append(activity)
    for pipeline, rows in grouped.items():
        latest = rows[-1]
        current_step = _processing_consensus_step(
            pipeline,
            latest.get("role"),
            latest.get("phase"),
        )
        active_by_lane[pipeline] = {
            "current_step": current_step,
            "model": latest.get("model"),
            "role": latest.get("role"),
            "phase": latest.get("phase"),
            "started_at": latest.get("started_at"),
            "updated_at": latest.get("updated_at"),
            "work_item": latest.get("request_sha256"),
            "active_jobs": len(rows),
        }

    consensus_calls = {
        (activity.get("pid"), activity.get("thread_id"), activity.get("model"))
        for activity in activities
    }
    legacy_consensus_calls = {
        (activity.get("pid"), activity.get("model"))
        for activity in activities
        if activity.get("thread_id") is None
    }
    direct_grouped: dict[str, list[dict[str, Any]]] = {}
    for activity in model_activities:
        if activity.get("component") == "chronovisor.decision.local_structured":
            continue
        exact_call = (
            activity.get("pid"),
            activity.get("thread_id"),
            activity.get("model"),
        )
        legacy_call = (activity.get("pid"), activity.get("model"))
        if exact_call in consensus_calls or legacy_call in legacy_consensus_calls:
            continue
        pipeline = str(activity.get("pipeline") or "audit")
        if pipeline not in {key for key, _label, _steps in _PROCESSING_LANES}:
            pipeline = "audit"
        direct_grouped.setdefault(pipeline, []).append(activity)
    for pipeline, rows in direct_grouped.items():
        if pipeline in grouped:
            continue
        live_rows = [row for row in rows if not row.get("recent")]
        if pipeline in active_by_lane and not live_rows:
            continue
        visible_rows = live_rows or rows
        latest = visible_rows[-1]
        active_by_lane[pipeline] = {
            "current_step": _processing_model_step(pipeline, latest.get("operation")),
            "model": latest.get("model"),
            "role": _processing_component_label(latest.get("component")),
            "phase": latest.get("caller") or latest.get("operation"),
            "started_at": latest.get("started_at"),
            "updated_at": latest.get("updated_at"),
            "work_item": latest.get("activity_id"),
            "active_jobs": len(visible_rows),
            "recent": not live_rows,
        }

    repair_active = bool(
        frontier_reviews.get("active") or frontier_repair.get("active")
    )
    if repair_active:
        repair_row = (
            frontier_reviews.get("latest")
            or frontier_repair.get("active_incident")
            or {}
        )
        repair_status = str(repair_row.get("status") or repair_row.get("phase") or "")
        repair_step = "escalate" if "frontier" in repair_status else "local_fix"
        active_by_lane["repair"] = {
            "current_step": repair_step,
            "model": repair_row.get("model"),
            "role": repair_row.get("kind") or repair_row.get("component") or "repair",
            "phase": repair_status or None,
            "started_at": repair_row.get("started_at") or repair_row.get("reserved_at"),
            "updated_at": repair_row.get("updated_at") or repair_row.get("started_at"),
            "work_item": repair_row.get("incident_id"),
            "active_jobs": int(frontier_reviews.get("count") or 1),
        }

    lanes: list[dict[str, Any]] = []
    for key, label, steps in _PROCESSING_LANES:
        active = active_by_lane.get(key)
        current_step = str((active or {}).get("current_step") or "") or None
        lanes.append(
            {
                "key": key,
                "label": label,
                "state": "active" if active else "idle",
                "current_step": current_step,
                "model": (active or {}).get("model"),
                "role": (active or {}).get("role"),
                "phase": (active or {}).get("phase"),
                "started_at": (active or {}).get("started_at"),
                "updated_at": (active or {}).get("updated_at"),
                "work_item": (active or {}).get("work_item"),
                "active_jobs": int((active or {}).get("active_jobs") or 0),
                "recent": bool((active or {}).get("recent")),
                "detail": (
                    _typed_graph_lane_detail(typed_graph)
                    if key == "typed_graph"
                    else "waiting for work"
                ),
                "steps": _processing_step_rows(steps, current_step),
            }
        )

    stable = {
        "active_count": sum(1 for row in lanes if row["state"] == "active"),
        "lanes": lanes,
    }
    revision = hashlib.sha256(_canonical_json_bytes(stable)).hexdigest()
    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "revision": revision,
        **stable,
    }


def _processing_activity_source_fingerprint() -> tuple[Any, ...]:
    """Probe only fixed live identities used by the processing-lane builder."""

    paths = (
        runtime_status.STATUS_FILE,
        orchestrator.STATE_FILE,
        CHRONOVISOR_ROOT / "runtime" / "local-consensus" / "active",
        CHRONOVISOR_ROOT / "runtime" / "model-activity" / "active",
        CHRONOVISOR_ROOT / "runtime" / "model-activity" / "recent",
        CHRONOVISOR_ROOT / "runtime" / "frontier-reviews" / "active",
        CHRONOVISOR_ROOT / "runtime" / "frontier-repair" / "state.json",
        CHRONOVISOR_ROOT / "runtime" / "frontier-repair" / "events.jsonl",
        CHRONOVISOR_ROOT / "runtime" / "typed-graph" / "status.json",
        CHRONOVISOR_ROOT / "runtime" / "typed-graph" / "promotion.json",
        CHRONOVISOR_ROOT / "runtime" / "recall-rubric" / "status.json",
    )
    callables = (
        id(_build_processing_activity_snapshot),
        id(runtime_status.read_status),
        id(orchestrator._load_state),
        id(_canonicalize_runtime_status),
        id(_local_consensus_activities),
        id(_model_activities),
        id(_frontier_activity_snapshot),
        id(_frontier_repair_snapshot),
        id(_typed_graph_dashboard_snapshot),
    )
    return (
        str(CHRONOVISOR_ROOT),
        callables,
        *(_path_identity(path) for path in paths),
    )


def _processing_activity_cache_metrics_locked() -> dict[str, Any]:
    return {
        "build_count": int(_PROCESSING_ACTIVITY_CACHE.get("build_count") or 0),
        "cache_hits": int(_PROCESSING_ACTIVITY_CACHE.get("cache_hits") or 0),
        "coalesced": int(_PROCESSING_ACTIVITY_CACHE.get("coalesced") or 0),
        "error_count": int(_PROCESSING_ACTIVITY_CACHE.get("error_count") or 0),
        "last_build_duration_ms": round(
            float(
                _PROCESSING_ACTIVITY_CACHE.get("last_build_duration_ms") or 0.0
            ),
            3,
        ),
        "last_error": _PROCESSING_ACTIVITY_CACHE.get("last_error"),
        "refreshing": bool(_PROCESSING_ACTIVITY_CACHE.get("refreshing")),
        "audit_seconds": PROCESSING_ACTIVITY_AUDIT_SECONDS,
        "recent_audit_seconds": PROCESSING_ACTIVITY_RECENT_AUDIT_SECONDS,
    }


def _processing_activity_cache_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    with _PROCESSING_ACTIVITY_CACHE_LOCK:
        metrics = _processing_activity_cache_metrics_locked()
    dashboard_value = snapshot.get("_dashboard")
    dashboard_state = dashboard_value if isinstance(dashboard_value, dict) else {}
    return {
        **snapshot,
        "_dashboard": {**dashboard_state, "activity_cache": metrics},
    }


def _refresh_processing_activity_cache(
    source: tuple[Any, ...],
) -> dict[str, Any]:
    """Build one lane snapshot outside the cache lock and publish atomically."""

    started_at = time.monotonic()
    build_source = source
    for _attempt in range(2):
        try:
            snapshot = _build_processing_activity_snapshot()
            completed_source = _processing_activity_source_fingerprint()
        except Exception as exc:
            duration_ms = max(0.0, (time.monotonic() - started_at) * 1000)
            try:
                current_source = _processing_activity_source_fingerprint()
            except Exception:
                current_source = None
            with _PROCESSING_ACTIVITY_CACHE_CONDITION:
                fallback = _PROCESSING_ACTIVITY_CACHE.get("snapshot")
                _PROCESSING_ACTIVITY_CACHE.update(
                    {
                        # Suppress a failure storm for this exact source epoch;
                        # the next source change or bounded audit retries.
                        "source": (
                            build_source if current_source == build_source else None
                        ),
                        "audited_at": time.monotonic(),
                        "refreshing": False,
                        "error_count": int(
                            _PROCESSING_ACTIVITY_CACHE.get("error_count") or 0
                        )
                        + 1,
                        "last_build_duration_ms": duration_ms,
                        "last_error": exc.__class__.__name__,
                    }
                )
                _PROCESSING_ACTIVITY_CACHE_CONDITION.notify_all()
            if isinstance(fallback, dict):
                return fallback
            raise

        duration_ms = max(0.0, (time.monotonic() - started_at) * 1000)
        with _PROCESSING_ACTIVITY_CACHE_CONDITION:
            fallback = _PROCESSING_ACTIVITY_CACHE.get("snapshot")
            if completed_source == build_source:
                _PROCESSING_ACTIVITY_CACHE.update(
                    {
                        "source": build_source,
                        "snapshot": snapshot,
                        "audited_at": time.monotonic(),
                        "refreshing": False,
                        "build_count": int(
                            _PROCESSING_ACTIVITY_CACHE.get("build_count") or 0
                        )
                        + 1,
                        "last_build_duration_ms": duration_ms,
                        "last_error": None,
                    }
                )
                _PROCESSING_ACTIVITY_CACHE_CONDITION.notify_all()
                return snapshot
            if isinstance(fallback, dict):
                # Do not publish a snapshot assembled across source epochs.
                # The next 250ms poll observes source=None and starts clean.
                _PROCESSING_ACTIVITY_CACHE.update(
                    {
                        "source": None,
                        "audited_at": 0.0,
                        "refreshing": False,
                        "last_build_duration_ms": duration_ms,
                    }
                )
                _PROCESSING_ACTIVITY_CACHE_CONDITION.notify_all()
                return fallback
        # A cold cache has no safe fallback. Retry once against the completed
        # epoch before exposing any snapshot to a client.
        build_source = completed_source

    error = RuntimeError("processing activity source changed during two builds")
    with _PROCESSING_ACTIVITY_CACHE_CONDITION:
        fallback = _PROCESSING_ACTIVITY_CACHE.get("snapshot")
        _PROCESSING_ACTIVITY_CACHE.update(
            {
                "source": None,
                "audited_at": 0.0,
                "refreshing": False,
                "error_count": int(
                    _PROCESSING_ACTIVITY_CACHE.get("error_count") or 0
                )
                + 1,
                "last_build_duration_ms": max(
                    0.0, (time.monotonic() - started_at) * 1000
                ),
                "last_error": error.__class__.__name__,
            }
        )
        _PROCESSING_ACTIVITY_CACHE_CONDITION.notify_all()
    if isinstance(fallback, dict):
        return fallback
    raise error


def _processing_activity_snapshot() -> dict[str, Any]:
    """Share live lane derivation across clients while preserving 250ms polls."""

    source = _processing_activity_source_fingerprint()
    start_async = False
    build_synchronously = False
    snapshot: dict[str, Any] | None = None
    while snapshot is None and not build_synchronously:
        with _PROCESSING_ACTIVITY_CACHE_CONDITION:
            now = time.monotonic()
            cached = _PROCESSING_ACTIVITY_CACHE.get("snapshot")
            source_matches = _PROCESSING_ACTIVITY_CACHE.get("source") == source
            audited_at = float(
                _PROCESSING_ACTIVITY_CACHE.get("audited_at") or 0.0
            )
            lanes = cached.get("lanes") if isinstance(cached, dict) else []
            has_recent_lane = isinstance(lanes, list) and any(
                isinstance(lane, dict) and lane.get("recent") is True
                for lane in lanes
            )
            audit_seconds = (
                PROCESSING_ACTIVITY_RECENT_AUDIT_SECONDS
                if has_recent_lane
                else PROCESSING_ACTIVITY_AUDIT_SECONDS
            )
            if (
                isinstance(cached, dict)
                and source_matches
                and now - audited_at < audit_seconds
            ):
                _PROCESSING_ACTIVITY_CACHE["cache_hits"] = int(
                    _PROCESSING_ACTIVITY_CACHE.get("cache_hits") or 0
                ) + 1
                snapshot = cached
                continue
            if _PROCESSING_ACTIVITY_CACHE.get("refreshing"):
                _PROCESSING_ACTIVITY_CACHE["coalesced"] = int(
                    _PROCESSING_ACTIVITY_CACHE.get("coalesced") or 0
                ) + 1
                if isinstance(cached, dict):
                    _PROCESSING_ACTIVITY_CACHE["cache_hits"] = int(
                        _PROCESSING_ACTIVITY_CACHE.get("cache_hits") or 0
                    ) + 1
                    snapshot = cached
                    continue
                _PROCESSING_ACTIVITY_CACHE_CONDITION.wait()
                source = _processing_activity_source_fingerprint()
                continue
            _PROCESSING_ACTIVITY_CACHE["refreshing"] = True
            if isinstance(cached, dict) and source_matches:
                _PROCESSING_ACTIVITY_CACHE["cache_hits"] = int(
                    _PROCESSING_ACTIVITY_CACHE.get("cache_hits") or 0
                ) + 1
                snapshot = cached
                start_async = True
            else:
                # A real source identity change is the live-data path: one
                # caller builds synchronously while concurrent clients keep
                # receiving the last successful snapshot. Only time-based
                # audits use stale-while-refresh.
                build_synchronously = True

    if start_async:
        threading.Thread(
            target=_refresh_processing_activity_cache,
            args=(source,),
            name="chronovisor-dashboard-activity-refresh",
            daemon=True,
        ).start()
    elif build_synchronously:
        snapshot = _refresh_processing_activity_cache(source)
    if not isinstance(snapshot, dict):
        raise RuntimeError("processing activity cache did not produce a snapshot")
    return _processing_activity_cache_payload(snapshot)


def _typed_graph_dashboard_snapshot() -> dict[str, Any]:
    """Cheap browser-safe summary for Observatory and the processing lane."""

    status_value = _read_json_file(
        CHRONOVISOR_ROOT / "runtime" / "typed-graph" / "status.json"
    )
    status: dict[str, Any] = status_value if isinstance(status_value, dict) else {}
    promotion_value = _read_json_file(
        CHRONOVISOR_ROOT / "runtime" / "typed-graph" / "promotion.json"
    )
    promotion: dict[str, Any] = (
        promotion_value if isinstance(promotion_value, dict) else {}
    )
    rubric_value = _read_json_file(
        CHRONOVISOR_ROOT / "runtime" / "recall-rubric" / "status.json"
    )
    rubric: dict[str, Any] = rubric_value if isinstance(rubric_value, dict) else {}
    builder_value = status.get("builder")
    builder: dict[str, Any] = builder_value if isinstance(builder_value, dict) else {}
    consensus_value = status.get("consensus")
    consensus: dict[str, Any] = (
        consensus_value if isinstance(consensus_value, dict) else {}
    )
    return {
        "mode": str(status.get("mode") or "shadow"),
        "engineering_complete": status.get("engineering_complete") is True,
        "engineering_gates": status.get("engineering_gates") or {},
        "authority_mature": status.get("authority_mature") is True,
        "relation_counts": status.get("relation_counts") or {},
        "communities": int(status.get("communities") or 0),
        "builder": {
            "status": str(builder.get("status") or "not_started"),
            "changed_pages": int(builder.get("changed_pages") or 0),
            "queued_pages": int(builder.get("queued_pages") or 0),
            "remaining_pages": int(builder.get("remaining_pages") or 0),
            "queue_overflow": int(builder.get("queue_overflow") or 0),
            "model": str(builder.get("model") or "gemma4:26b"),
        },
        "consensus": {
            "status": str(consensus.get("status") or "not_started"),
            "verified": int(consensus.get("verified") or 0),
            "held": int(consensus.get("held") or 0),
            "disagreement": int(consensus.get("disagreement") or 0),
        },
        "rubric": {
            "status": str(rubric.get("status") or "builtin"),
            "gates": rubric.get("gates") or {},
            "samples": int(rubric.get("samples") or 0),
            "judge_metrics": rubric.get("judge_metrics") or {},
        },
        "rubric_gold": status.get("rubric_gold") or {},
        "entities": status.get("entities") or {},
        "community_summary": status.get("community_summary") or {},
        "evaluation": status.get("evaluation") or {},
        "four_arm": status.get("four_arm") or {},
        "authority": status.get("authority") or {},
        "rollout": {
            "mode": str(promotion.get("mode") or "shadow"),
            "canary_percent": int(promotion.get("canary_percent") or 0),
            "reason": str(promotion.get("reason") or "not_evaluated")[:160],
            "gates": promotion.get("gates") or {},
            "sample_count": int(promotion.get("sample_count") or 0),
            "sample_unit": str(promotion.get("sample_unit") or ""),
        },
        "external_model_calls": int(status.get("external_model_calls") or 0),
    }


def _typed_graph_lane_detail(value: dict[str, Any]) -> str:
    counts = value.get("relation_counts")
    counts = counts if isinstance(counts, dict) else {}
    verified = int(counts.get("verified") or 0)
    authoritative = int(counts.get("authoritative") or 0)
    rollout = value.get("rollout")
    rollout = rollout if isinstance(rollout, dict) else {}
    state = (
        "authority mature" if value.get("authority_mature") else "collecting authority"
    )
    return (
        f"{state} · verified {verified} · authoritative {authoritative} · "
        f"canary {int(rollout.get('canary_percent') or 0)}% · external 0"
    )


def _safe_typed_graph_snapshot() -> dict[str, Any]:
    return _safe_snapshot_component(
        "typed_graph",
        _typed_graph_dashboard_snapshot,
        {
            "mode": "shadow",
            "engineering_complete": False,
            "authority_mature": False,
            "relation_counts": {},
            "external_model_calls": 0,
        },
    )


def _empty_local_consensus_summary() -> dict[str, Any]:
    sessions = {
        "total": 0,
        "ok": 0,
        "first_pass_valid": 0,
        "repaired": 0,
        "repair_turns": 0,
        "failures": {},
    }
    decisions = {
        "total": 0,
        "agreed": 0,
        "pair_agreement": 0,
        "tie_break_used": 0,
        "unresolved_quarantine": 0,
        "conservative_veto_fired": 0,
        "conservative_veto_bypassed_by_lane_policy": 0,
        "dissent_effect_classes": {},
        "model_conservative_vote_rates": {},
    }
    return {
        "schema_version": 3,
        "retained_records": 0,
        "routine_records": 0,
        "sessions": sessions,
        "decisions": decisions,
        "evaluation": {
            "records": 0,
            "sessions": dict(sessions),
            "decisions": dict(decisions),
        },
        "roles": {},
    }


_DECISION_TRACE_ROLES = ("primary", "challenger", "tie_break")
_DECISION_TRACE_PHASES = ("trigger", "load", "context", "generate", "validate", "vote")
_DECISION_TRACE_EVENT_LIMIT = 64


def _decision_trace_role(value: object) -> tuple[str, str, bool]:
    role = str(value or "structured")
    if role in _DECISION_TRACE_ROLES:
        return "routine", role, True
    for lane in _DECISION_TRACE_ROLES:
        suffix = f":{lane}"
        if role.endswith(suffix):
            return role[: -len(suffix)] or "routine", lane, True
    return role, "primary", False


def _decision_trace_models() -> dict[str, str]:
    try:
        config = _resolved_decision_router_config()
        return {
            "primary": str(config.primary_model),
            "challenger": str(config.challenger_model),
            "tie_break": str(config.tie_break_model),
        }
    except Exception:
        return {role: "not configured" for role in _DECISION_TRACE_ROLES}


def _decision_trace_context_tokens() -> int | None:
    try:
        return int(_resolved_decision_router_config().num_ctx)
    except Exception:
        return None


def _decision_trace_steps(
    state: str,
    *,
    phase: str | None = None,
) -> list[dict[str, str]]:
    phase = "generate" if phase == "repair" else str(phase or "trigger")
    current_index = (
        _DECISION_TRACE_PHASES.index(phase) if phase in _DECISION_TRACE_PHASES else 0
    )
    labels = {
        "trigger": "Trigger",
        "load": "Load",
        "context": "Context",
        "generate": "Generate",
        "validate": "Validate",
        "vote": "Vote",
    }
    steps: list[dict[str, str]] = []
    for index, key in enumerate(_DECISION_TRACE_PHASES):
        if state == "done":
            status = "done"
        elif state == "error":
            status = "done" if index < 4 else "error" if index == 4 else "skipped"
        elif state == "skipped":
            status = "skipped"
        elif state == "active":
            status = (
                "done"
                if index < current_index
                else "active"
                if index == current_index
                else "pending"
            )
        else:
            status = "pending"
        steps.append({"key": key, "label": labels[key], "status": status})
    return steps


def _decision_trace_events(
    rows: list[dict[str, Any]],
    *,
    request_sha256: str,
) -> list[dict[str, Any]]:
    """Return a bounded, redacted, ordered transition stream for one request."""

    phase_labels = {
        "trigger": "Triggered",
        "load": "Loading model",
        "context": "Building context",
        "generate": "Generating",
        "repair": "Repairing JSON",
        "validate": "Validating",
        "vote": "Vote ready",
        "decision": "Decision sealed",
    }
    overall_keys = {
        "trigger": "dispatch",
        "load": "dispatch",
        "context": "generate",
        "generate": "generate",
        "repair": "generate",
        "validate": "validate",
        "vote": "quorum",
        "decision": "decision",
    }
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if str(row.get("request_sha256") or "") != request_sha256:
            continue
        event_id = str(row.get("event_id") or "")
        timestamp = str(row.get("timestamp") or "")
        phase = str(row.get("phase") or "")
        kind = str(row.get("kind") or "phase")
        status = str(row.get("status") or "active")
        if (
            not event_id
            or event_id in seen
            or phase not in {*_DECISION_TRACE_PHASES, "repair", "decision"}
            or kind not in {"phase", "session", "decision", "decision_artifact_replay"}
            or status not in {"active", "done", "error"}
        ):
            continue
        seen.add(event_id)
        role = str(row.get("role") or "structured")
        _base, lane, explicit_lane = _decision_trace_role(role)
        if phase == "decision" and not explicit_lane:
            lane_value: str | None = None
        else:
            lane_value = lane
        label = phase_labels[phase]
        if kind == "session":
            label = "Vote accepted" if status == "done" else "Vote rejected"
        elif kind == "decision_artifact_replay":
            label = "Artifact replayed"
        elif kind == "decision":
            label = "Decision approved" if status == "done" else "Decision held"
        attempt = row.get("attempt")
        events.append(
            {
                "event_id": event_id[:80],
                "timestamp": timestamp[:64],
                "kind": kind,
                "lane": lane_value,
                "model": str(row.get("model") or "")[:160],
                "phase": phase,
                "status": status,
                "attempt": (
                    int(attempt)
                    if isinstance(attempt, int) and not isinstance(attempt, bool)
                    else 0
                ),
                "label": label,
                "overall_key": overall_keys[phase],
            }
        )
    return events[-_DECISION_TRACE_EVENT_LIMIT:]


def _decision_trace_outcome(
    decision: dict[str, Any] | None,
    *,
    trace_state: str,
    task_role: str,
) -> dict[str, str]:
    """Explain a redacted decision without exposing prompts or vote payloads."""

    source_state = (
        "Raw retained" if task_role.startswith("ingest") else "Input retained"
    )
    if trace_state == "agreed":
        if (decision or {}).get("conservative_veto_bypassed_by_lane_policy") is True:
            dissent_effect = str(
                (decision or {}).get("dissent_effect_class") or "unclassifiable"
            )
            return {
                "kind": "approved",
                "reason": "Lane policy bypassed conservative veto",
                "data": f"Dissent effect: {dissent_effect}",
                "next": "Mutation may proceed",
                "code": "conservative_veto_bypassed_by_lane_policy",
            }
        return {
            "kind": "approved",
            "reason": "Safe local quorum reached",
            "data": "Decision artifact sealed",
            "next": "Mutation may proceed",
            "code": "local_quorum_agreed",
        }
    if trace_state == "ready":
        return {
            "kind": "ready",
            "reason": "Structured result validated",
            "data": source_state,
            "next": "Ready for the caller",
            "code": "structured_result_ready",
        }
    if trace_state == "active":
        return {
            "kind": "active",
            "reason": "Local decision in progress",
            "data": source_state,
            "next": "Mutation stays locked",
            "code": "local_decision_active",
        }
    if trace_state != "quarantined":
        return {
            "kind": "idle",
            "reason": "Waiting for local work",
            "data": "No active decision",
            "next": "Starts automatically",
            "code": "local_decision_idle",
        }

    reason = str((decision or {}).get("quarantine_reason") or "")
    failure_class = str((decision or {}).get("failure_class") or "")
    semantic_reasons = {
        "local_models_did_not_reach_two_vote_quorum": "Valid models disagreed",
        "mutating_local_majority_vetoed_by_conservative_vote": (
            "Conservative vote blocked mutation"
        ),
    }
    quality_reasons = {
        "fewer_than_two_valid_local_votes": "Too few valid model votes",
        "primary_and_challenger_invalid": "Primary pair returned invalid votes",
    }
    if reason in semantic_reasons:
        return {
            "kind": "semantic_hold",
            "reason": semantic_reasons[reason],
            "data": source_state,
            "next": "Recheck after model or policy change",
            "code": reason,
        }
    if reason in quality_reasons:
        return {
            "kind": "quality_hold",
            "reason": quality_reasons[reason],
            "data": source_state,
            "next": "Retry after model or runtime change",
            "code": reason,
        }

    resource_failure = failure_class == "local_resource_quarantined" or any(
        marker in reason
        for marker in (
            "runner_does_not_fit",
            "runner_no_longer_fits",
            "verify_initial_runner_eviction",
            "verify_primary_runner_eviction",
            "verify_challenger_runner_eviction",
            "verify_pair_runner_eviction",
            "verify_tie_break_runner_eviction",
        )
    )
    if resource_failure:
        return {
            "kind": "operational_hold",
            "reason": "Model memory could not be verified",
            "data": source_state,
            "next": "Retry when capacity recovers",
            "code": reason or failure_class or "local_resource_quarantined",
        }
    if failure_class == "context_window_exceeded" or "context" in reason:
        return {
            "kind": "operational_hold",
            "reason": "Request exceeds context capacity",
            "data": source_state,
            "next": "Retry with a compatible context plan",
            "code": reason or failure_class,
        }
    return {
        "kind": "operational_hold",
        "reason": "Local decision could not finish safely",
        "data": source_state,
        "next": "Retry after runtime conditions change",
        "code": reason or failure_class or "local_decision_quarantined",
    }


def _decision_trace_snapshot(
    activities: list[dict[str, Any]],
    history: list[dict[str, Any]],
    latest_decision: dict[str, Any] | None,
    trace_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Project redacted consensus telemetry into a stable three-lane trace."""

    active_request = str(
        (activities[-1] if activities else {}).get("request_sha256") or ""
    )
    decision_request = str((latest_decision or {}).get("request_sha256") or "")
    latest_session = next(
        (row for row in reversed(history) if row.get("kind") == "session"),
        None,
    )
    latest_event = (trace_events or [])[-1] if trace_events else None
    request_sha256 = (
        active_request
        or decision_request
        or str((latest_session or {}).get("request_sha256") or "")
        or str((latest_event or {}).get("request_sha256") or "")
    )
    related = [row for row in history if row.get("request_sha256") == request_sha256]
    decision = next(
        (
            row
            for row in reversed(related)
            if row.get("kind") in {"decision", "decision_artifact_replay"}
        ),
        None,
    )
    active_rows = [
        row for row in activities if row.get("request_sha256") == request_sha256
    ]
    session_rows = [row for row in related if row.get("kind") == "session"]

    task_role = "idle"
    if active_rows:
        task_role = _decision_trace_role(active_rows[-1].get("role"))[0]
    elif decision:
        task_role = str(
            decision.get("role") or decision.get("decision_lane") or "routine"
        )
    elif latest_session:
        task_role = _decision_trace_role(latest_session.get("role"))[0]

    quorum_flow = bool(
        decision
        or any(_decision_trace_role(row.get("role"))[2] for row in active_rows)
        or any(_decision_trace_role(row.get("role"))[2] for row in session_rows)
    )
    models = _decision_trace_models()
    decision_models = decision.get("models") if isinstance(decision, dict) else None
    if isinstance(decision_models, list):
        for lane, model in zip(_DECISION_TRACE_ROLES, decision_models, strict=False):
            if isinstance(model, str) and model:
                models[lane] = model

    active_by_lane: dict[str, dict[str, Any]] = {}
    for row in active_rows:
        _base, lane, _explicit = _decision_trace_role(row.get("role"))
        active_by_lane[lane] = row
        if isinstance(row.get("model"), str) and row.get("model"):
            models[lane] = str(row["model"])
    session_by_lane: dict[str, dict[str, Any]] = {}
    for row in session_rows:
        _base, lane, _explicit = _decision_trace_role(row.get("role"))
        session_by_lane[lane] = row
        if isinstance(row.get("model"), str) and row.get("model"):
            models[lane] = str(row["model"])

    artifact_replay = bool(
        decision and decision.get("kind") == "decision_artifact_replay"
    )
    lanes: list[dict[str, Any]] = []
    lane_labels = {
        "primary": "Primary",
        "challenger": "Challenger",
        "tie_break": "Tie-break",
    }
    for index, lane in enumerate(_DECISION_TRACE_ROLES):
        active = active_by_lane.get(lane)
        session = session_by_lane.get(lane)
        state = "pending"
        result = "Waiting"
        detail = "Not started"
        phase: str | None = None
        if artifact_replay:
            if lane == "tie_break":
                state, result, detail = (
                    "skipped",
                    "Not required",
                    "Sealed quorum proof replayed",
                )
            else:
                state, result, detail = "done", "Proof replay", "0 live model calls"
        elif active is not None:
            state = "active"
            phase = str(active.get("phase") or "generate")
            attempt = int(active.get("attempt") or 0)
            result = "JSON repair" if phase == "repair" else phase.title()
            detail = f"attempt {attempt + 1} · {int(float(active.get('elapsed_seconds') or 0))}s elapsed"
        elif session is not None:
            if bool(session.get("ok")):
                state = "done"
                repairs = int(session.get("repair_turns") or 0)
                result = "Valid after repair" if repairs else "Valid first pass"
                detail = f"{repairs} repair turn{'s' if repairs != 1 else ''}"
            else:
                state = "error"
                result = "Invalid vote"
                detail = str(session.get("failure_class") or "validation failed")
        elif not quorum_flow and lane != "primary":
            state, result, detail = (
                "skipped",
                "Not required",
                "Single-model structured task",
            )
        elif (
            decision is not None
            and lane == "tie_break"
            and not decision.get("tie_break_used")
        ):
            state, result, detail = (
                "skipped",
                "Not required",
                "Primary pair resolved quorum",
            )
        elif decision is not None and index < int(decision.get("valid_votes") or 0):
            state, result, detail = (
                "done",
                "Valid vote",
                "Recovered from decision audit",
            )
        lanes.append(
            {
                "key": lane,
                "label": lane_labels[lane],
                "model": models[lane],
                "state": state,
                "result": result,
                "detail": detail,
                "phase": phase,
                "steps": _decision_trace_steps(state, phase=phase),
            }
        )

    decision_status = str((decision or {}).get("status") or "")
    if active_rows:
        trace_state = "active"
        active_lane = next(
            (lane for lane in lanes if lane["state"] == "active"), lanes[0]
        )
        summary = f"{active_lane['label']} · {active_lane['result']}"
    elif decision_status == "agreed":
        trace_state = "agreed"
        if artifact_replay:
            summary = "Canonical artifact replay · 0 model calls"
        elif decision and decision.get("pair_agreement"):
            summary = "2/2 pair agreement"
        elif decision and decision.get("tie_break_used"):
            summary = (
                "2/3 quorum · conservative veto bypassed by lane policy"
                if decision.get("conservative_veto_bypassed_by_lane_policy")
                else "2/3 quorum after tie-break"
            )
        else:
            summary = "Local quorum agreed"
    elif decision_status == "quarantined":
        trace_state = "quarantined"
        summary = "No safe quorum · quarantined"
    elif session_rows:
        trace_state = "ready"
        summary = "Structured result ready"
    else:
        trace_state = "idle"
        summary = "No local decision yet"

    active_phases = {
        "generate"
        if row.get("phase") == "repair"
        else str(row.get("phase") or "trigger")
        for row in active_rows
    }
    generating = bool(active_phases & {"trigger", "load", "context", "generate"})
    validating = bool(active_phases) and not generating and "validate" in active_phases
    voting = bool(active_phases) and not generating and not validating
    completed = decision_status in {"agreed", "quarantined"} or (
        not quorum_flow and bool(session_rows)
    )
    quorum_status = (
        "done"
        if decision_status == "agreed" or (not quorum_flow and bool(session_rows))
        else "error"
        if decision_status == "quarantined"
        else "active"
        if voting or (bool(session_rows) and not active_rows)
        else "pending"
    )
    overall = [
        {
            "key": "packet",
            "label": "Packet",
            "status": "done" if request_sha256 else "pending",
        },
        {
            "key": "dispatch",
            "label": "Dispatch",
            "status": "done" if active_rows or session_rows or decision else "pending",
        },
        {
            "key": "generate",
            "label": "Generate",
            "status": "active"
            if generating
            else "done"
            if session_rows or completed
            else "pending",
        },
        {
            "key": "validate",
            "label": "Validate",
            "status": "active"
            if validating
            else "done"
            if not generating and (session_rows or completed)
            else "pending",
        },
        {"key": "quorum", "label": "Quorum", "status": quorum_status},
        {
            "key": "artifact",
            "label": "Artifact",
            "status": "done"
            if decision_status == "agreed"
            or artifact_replay
            or (not quorum_flow and session_rows)
            else "skipped"
            if decision_status == "quarantined"
            else "pending",
        },
        {
            "key": "decision",
            "label": "Decision",
            "status": "done"
            if decision_status == "agreed" or (not quorum_flow and session_rows)
            else "error"
            if decision_status == "quarantined"
            else "active"
            if voting and not quorum_flow
            else "pending",
        },
    ]
    updated_at = (
        (active_rows[-1] if active_rows else {}).get("updated_at")
        or (active_rows[-1] if active_rows else {}).get("started_at")
        or (decision or {}).get("timestamp")
        or (session_rows[-1] if session_rows else {}).get("timestamp")
    )
    outcome = _decision_trace_outcome(
        decision,
        trace_state=trace_state,
        task_role=task_role,
    )
    events = _decision_trace_events(
        trace_events or [],
        request_sha256=request_sha256,
    )
    return {
        "state": trace_state,
        "active": bool(active_rows),
        "request_sha256": request_sha256 or None,
        "task_role": task_role,
        "summary": summary,
        "started_at": (active_rows[0] if active_rows else {}).get("started_at"),
        "updated_at": updated_at,
        "context_tokens": _decision_trace_context_tokens(),
        "quorum_flow": quorum_flow,
        "artifact_replay": artifact_replay,
        "outcome": outcome,
        "overall": overall,
        "lanes": lanes,
        "events": events,
        "event_count": len(events),
    }


def _local_consensus_snapshot(limit: int = 40) -> dict[str, Any]:
    """Return live local review truth plus a redacted bounded audit tail."""

    root = CHRONOVISOR_ROOT / "runtime" / "local-consensus"
    activities = _local_consensus_activities()

    summary = _read_json_file(root / "summary.json") or _empty_local_consensus_summary()
    history: list[dict[str, Any]] = []
    for row in _read_jsonl_file(root / "audit.jsonl", limit=max(1, limit)):
        kind = row.get("kind")
        if kind == "session":
            history.append(
                {
                    key: row.get(key)
                    for key in (
                        "kind",
                        "timestamp",
                        "request_sha256",
                        "role",
                        "model",
                        "ok",
                        "first_pass_valid",
                        "repaired",
                        "repair_turns",
                        "failure_class",
                    )
                }
            )
        elif kind == "decision":
            history.append(
                {
                    key: row.get(key)
                    for key in (
                        "kind",
                        "timestamp",
                        "request_sha256",
                        "role",
                        "status",
                        "failure_class",
                        "quarantine_reason",
                        "pair_agreement",
                        "tie_break_used",
                        "unresolved_quarantine",
                        "vote_count",
                        "valid_votes",
                        "first_pass_valid_votes",
                        "repaired_votes",
                        "repair_turns",
                        "models",
                        "decision_lane",
                        "pair_safe_resolution_without_tie",
                        "signature_majority_resolution",
                        "safe_policy_resolution",
                        "conservative_veto_fired",
                        "conservative_veto_bypassed_by_lane_policy",
                        "dissent_effect_class",
                    )
                }
            )
        elif kind == "decision_artifact_replay":
            history.append(
                {
                    key: row.get(key)
                    for key in (
                        "kind",
                        "timestamp",
                        "request_sha256",
                        "role",
                        "decision_lane",
                        "status",
                        "model_invocations",
                    )
                }
            )
    trace_events = _read_jsonl_file(
        root / "trace-events.jsonl",
        limit=max(_DECISION_TRACE_EVENT_LIMIT * 4, limit * 4),
    )
    latest_decision = next(
        (
            row
            for row in reversed(history)
            if row.get("kind") in {"decision", "decision_artifact_replay"}
        ),
        None,
    )
    decision_trace = _decision_trace_snapshot(
        activities,
        history,
        latest_decision,
        trace_events,
    )
    return {
        "active": bool(activities),
        "count": len(activities),
        "activities": activities,
        "latest": activities[-1] if activities else None,
        "summary": summary,
        "latest_decision": latest_decision,
        "decision_trace": decision_trace,
        "history": history,
    }


def _frontier_repair_snapshot(limit: int = 40) -> dict[str, Any]:
    """Expose the exceptional repair ledger without leaking incident payloads."""

    root = CHRONOVISOR_ROOT / "runtime" / "frontier-repair"
    state_path = root / "state.json"
    state = _read_json_file(state_path) or {}
    incidents_raw = state.get("incidents")
    incidents = incidents_raw if isinstance(incidents_raw, dict) else {}
    counts: dict[str, int] = {}
    recent: list[dict[str, Any]] = []
    starts_24h = 0
    cutoff = datetime.now().astimezone() - timedelta(hours=24)
    for incident_id, raw in incidents.items():
        if not isinstance(raw, dict):
            continue
        status = str(raw.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
        started_at = raw.get("started_at")
        if isinstance(started_at, str):
            try:
                started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                compare_cutoff = (
                    cutoff.astimezone(started.tzinfo)
                    if started.tzinfo
                    else cutoff.replace(tzinfo=None)
                )
                if started >= compare_cutoff:
                    starts_24h += 1
            except ValueError:
                pass
        evidence = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}
        recent.append(
            {
                "incident_id": incident_id,
                "status": status,
                "component": evidence.get("component"),
                "failure_class": evidence.get("failure_class"),
                "fingerprint_key": raw.get("fingerprint_key"),
                "reserved_at": raw.get("reserved_at"),
                "started_at": started_at,
                "finished_at": raw.get("finished_at"),
                "lease_expires_at": raw.get("lease_expires_at"),
                "owner_pid": raw.get("owner_pid"),
                "owner_process_started_at": raw.get("owner_process_started_at"),
                "pid": raw.get("pid"),
            }
        )
    recent.sort(
        key=lambda row: str(
            row.get("started_at")
            or row.get("reserved_at")
            or row.get("finished_at")
            or ""
        )
    )
    recent = recent[-max(1, limit) :]

    active_id = state.get("active_incident_id")
    active_row = next(
        (row for row in recent if row.get("incident_id") == active_id),
        None,
    )
    active_status = active_row.get("status") if active_row else None
    owner_alive = bool(
        active_row
        and active_status in {"reserved", "started"}
        and runtime_status._pid_is_alive(active_row.get("owner_pid"))
    )
    owner_identity_status = (
        _process_start_identity_status(
            active_row.get("owner_pid"),
            active_row.get("owner_process_started_at"),
        )
        if active_row and active_status in {"reserved", "started"}
        else None
    )
    lease_expired = bool(
        active_row
        and active_status in {"reserved", "started"}
        and _timestamp_is_expired(active_row.get("lease_expires_at"))
    )
    active = bool(
        active_row
        and active_status in {"reserved", "started"}
        and owner_alive
        and not lease_expired
        and owner_identity_status
        in {_PROCESS_IDENTITY_MATCH, _PROCESS_IDENTITY_UNAVAILABLE}
    )
    if active_row is not None:
        active_row = {
            **active_row,
            "owner_alive": owner_alive,
            "owner_identity_status": owner_identity_status,
            "lease_expired": lease_expired,
            "elapsed_seconds": _activity_age_seconds(
                active_row.get("started_at") or active_row.get("reserved_at")
            ),
        }

    events = []
    for row in _read_jsonl_file(root / "events.jsonl", limit=max(1, limit)):
        events.append(
            {
                key: row.get(key)
                for key in (
                    "sequence",
                    "timestamp",
                    "event",
                    "incident_id",
                    "fingerprint_key",
                    "outcome",
                    "reason",
                    "prior_status",
                    "stale_recovery",
                )
            }
        )
    return {
        "available": state_path.exists(),
        "active": active,
        "active_incident": active_row,
        "stale_active_incident": bool(active_row and not active),
        "summary": {
            "total": len(incidents),
            "starts_24h": starts_24h,
            "counts": counts,
        },
        "recent": recent,
        "events": events,
    }


def _last_self_heal_check(limit: int = 400) -> dict[str, Any] | None:
    logs_dir = CHRONOVISOR_ROOT / "logs"
    if not logs_dir.exists():
        return None
    for path in reversed(sorted(logs_dir.glob("ingest-drain-*.jsonl"))[-14:]):
        for record in reversed(_read_jsonl_file(path, limit=limit)):
            self_heal = record.get("self_heal")
            if not isinstance(self_heal, dict):
                continue
            results = self_heal.get("results")
            return {
                "timestamp": record.get("timestamp"),
                "status": self_heal.get("status"),
                "packets_seen": self_heal.get("packets_seen"),
                "results": len(results) if isinstance(results, list) else 0,
                "log_file": str(path),
            }
    return None


def _self_heal_watch_snapshot(packets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    pending: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for packet in packets.values():
        status = str(packet.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        item = {
            "failure_id": packet.get("failure_id"),
            "raw_file": packet.get("raw_file"),
            "failure_class": packet.get("failure_class"),
            "status": status,
            "updated_at": packet.get("updated_at") or packet.get("created_at"),
        }
        if status in SELF_HEAL_PENDING_STATUSES:
            pending.append(item)
        elif status in SELF_HEAL_FAILED_STATUSES:
            failed.append(item)

    return {
        "last_checked": _last_self_heal_check(),
        "packets": {
            "total": len(packets),
            "pending": len(pending),
            "failed": len(failed),
            "status_counts": status_counts,
            "pending_samples": pending[-5:],
            "failed_samples": failed[-5:],
        },
        "frontier_preflight": _frontier_preflight_snapshot(),
    }


def _local_repair_summary(
    record: dict[str, Any], packet: dict[str, Any] | None
) -> dict[str, Any]:
    decision = (
        record.get("decision") if isinstance(record.get("decision"), dict) else {}
    )
    action = record.get("action") if isinstance(record.get("action"), dict) else {}
    retry = action.get("retry") if isinstance(action.get("retry"), dict) else {}
    restore = action.get("restore") if isinstance(action.get("restore"), dict) else {}
    alias = action.get("alias") if isinstance(action.get("alias"), dict) else {}

    requested = decision.get("requested_page_id") or alias.get("requested")
    target = decision.get("target_page_id") or alias.get("target")
    details: list[str] = []
    if requested and target:
        details.append(f"alias {requested} -> {target}")
    elif decision.get("action"):
        details.append(str(decision["action"]))
    if retry.get("files_processed"):
        details.append("retry ok")
    elif retry:
        details.append("retry ran")

    raw_file = (
        record.get("raw_file")
        or (packet or {}).get("raw_file")
        or _basename(restore.get("target"))
        or _basename(restore.get("source"))
    )
    return {
        "timestamp": record.get("timestamp")
        or (packet or {}).get("updated_at")
        or (packet or {}).get("created_at"),
        "failure_id": record.get("failure_id"),
        "state": "resolved",
        "level": "success",
        "title": "Local repair applied",
        "detail": _shorten(
            "; ".join(details) or decision.get("reason") or "local repair applied"
        ),
        "raw_file": raw_file,
        "failure_class": record.get("failure_class")
        or (packet or {}).get("failure_class"),
        "resolution": "local",
        "action": decision.get("action"),
        "source": decision.get("source"),
        "retry_success": bool(retry.get("files_processed")),
        "details": {
            "failure": _failure_detail(packet, record),
            "decision": {
                "status": decision.get("status"),
                "action": decision.get("action"),
                "confidence": decision.get("confidence"),
                "requested_page_id": decision.get("requested_page_id"),
                "target_page_id": decision.get("target_page_id"),
                "source": decision.get("source"),
                "reason": decision.get("reason"),
            },
            "action": {
                "alias": alias or None,
                "restore": restore or None,
                "retry": _retry_detail(action),
            },
        },
    }


def _frontier_summary(
    record: dict[str, Any], packet: dict[str, Any] | None
) -> dict[str, Any]:
    frontier = (
        record.get("frontier") if isinstance(record.get("frontier"), dict) else {}
    )
    local_decision = (
        record.get("decision") if isinstance(record.get("decision"), dict) else {}
    )
    decision = frontier.get("decision") or "unknown"
    rescue_status = frontier.get("rescue_status")
    human_required = is_human_required_result(frontier)
    if decision == "approved":
        state = "resolved"
        level = "success"
        title = "Consensus approved"
    elif human_required:
        state = "failed"
        level = "error"
        title = "Human required"
    elif rescue_status == "pending_frontier_review":
        state = "pending"
        level = "warn"
        title = "Consensus review pending"
    elif rescue_status == "frontier_preflight_failed":
        state = "pending"
        level = "warn"
        title = "Consensus preflight failed"
    else:
        level = "warn" if decision in {"needs_retry", "quarantined"} else "error"
        state = "pending" if decision == "needs_retry" else "failed"
        title = f"Consensus {decision}"
    raw_file = record.get("raw_file") or (packet or {}).get("raw_file")
    commit = frontier.get("commit")
    detail = frontier.get("summary") or decision
    if commit:
        detail = f"{detail}; commit {str(commit)[:8]}"
    return {
        "timestamp": record.get("timestamp")
        or (packet or {}).get("updated_at")
        or (packet or {}).get("created_at"),
        "failure_id": record.get("failure_id"),
        "state": state,
        "level": level,
        "title": title,
        "detail": _shorten(detail),
        "raw_file": raw_file,
        "failure_class": record.get("failure_class")
        or (packet or {}).get("failure_class"),
        "resolution": "frontier",
        "action": decision,
        "source": "frontier",
        "retry_success": False,
        "details": {
            "failure": _failure_detail(packet, record),
            "local_decision": local_decision or None,
            "frontier": {
                "decision": frontier.get("decision"),
                "summary": frontier.get("summary"),
                "tests_run": frontier.get("tests_run"),
                "commit": frontier.get("commit"),
                "committed": frontier.get("committed"),
                "pushed": frontier.get("pushed"),
                "risk": frontier.get("risk"),
                "notes": frontier.get("notes"),
                "rescue_status": rescue_status,
                "human_required": human_required,
                "frontier_failure": frontier.get("frontier_failure"),
                "rescue_attempt": frontier.get("rescue_attempt"),
                "access_repair": frontier.get("access_repair"),
            },
            "human_notification": record.get("human_notification")
            or (packet or {}).get("human_notification"),
            "pending_frontier_review_path": (
                record.get("pending_frontier_review_path")
                or (packet or {}).get("pending_frontier_review_path")
            ),
        },
    }


def _packet_summary(packet: dict[str, Any]) -> dict[str, Any]:
    status = str(packet.get("status") or "unknown")
    state = (
        "pending"
        if status in SELF_HEAL_PENDING_STATUSES
        else "failed"
        if status in SELF_HEAL_FAILED_STATUSES
        else "resolved"
    )
    level = (
        "warn"
        if status == "repair_deferred"
        else "info"
        if state == "pending"
        else "error"
        if state == "failed"
        else "success"
    )
    title = {
        "pending_local_repair": "Repair queued",
        "local_repairing": "Local repair running",
        "pending_frontier": "Consensus queued",
        "frontier_running": "Consensus running",
        "frontier_retry": "Consensus retry needed",
        "frontier_preflight_failed": "Consensus preflight failed",
        "pending_frontier_review": "Consensus review pending",
        "repair_deferred": "Frontier repair deferred",
        "human_required": "Human required",
        "local_repair_failed": "Local repair failed",
        "frontier_rejected": "Consensus rejected",
        "frontier_quarantined": "Consensus quarantined",
    }.get(status, status.replace("_", " "))
    return {
        "timestamp": packet.get("updated_at") or packet.get("created_at"),
        "failure_id": packet.get("failure_id"),
        "state": state,
        "level": level,
        "title": title,
        "detail": _shorten(packet.get("error") or packet.get("fingerprint") or status),
        "raw_file": packet.get("raw_file"),
        "failure_class": packet.get("failure_class"),
        "resolution": None,
        "action": packet.get("status"),
        "source": "packet",
        "retry_success": False,
        "details": {
            "failure": _failure_detail(packet),
            "error": packet.get("error"),
            "requested_page_id": packet.get("requested_page_id"),
            "similar_existing_pages": packet.get("similar_existing_pages"),
            "local_decision": packet.get("local_decision"),
            "frontier_result": packet.get("frontier_result"),
            "human_notification": packet.get("human_notification"),
        },
    }


def _self_heal_snapshot(limit: int = 12) -> dict[str, Any]:
    failures_dir = CHRONOVISOR_ROOT / "runtime" / "failures"
    packets = _self_heal_packet_index()
    registry = _read_jsonl_file(failures_dir / "failure-registry.jsonl", limit=200)
    seen_failure_ids: set[str] = set()
    history: list[dict[str, Any]] = []

    for record in registry:
        if record.get("failure_class") == "ingest.semantic_no_quorum":
            continue
        failure_id = record.get("failure_id")
        packet = packets.get(failure_id) if isinstance(failure_id, str) else None
        if isinstance(failure_id, str):
            seen_failure_ids.add(failure_id)
        if record.get("resolution") == "frontier":
            history.append(_frontier_summary(record, packet))
        else:
            history.append(_local_repair_summary(record, packet))

    for failure_id, packet in packets.items():
        if failure_id not in seen_failure_ids:
            history.append(_packet_summary(packet))

    history.sort(key=lambda item: str(item.get("timestamp") or ""))
    history = history[-limit:]
    counts = {
        "resolved": sum(1 for item in history if item.get("state") == "resolved"),
        "pending": sum(1 for item in history if item.get("state") == "pending"),
        "failed": sum(1 for item in history if item.get("state") == "failed"),
        "frontier": sum(1 for item in history if item.get("resolution") == "frontier"),
        "human_required": sum(
            1 for item in history if item.get("title") == "Human required"
        ),
        "pending_frontier_review": sum(
            1 for item in history if item.get("title") == "Consensus review pending"
        ),
    }
    latest = history[-1] if history else None
    status = "quiet"
    if latest:
        status = str(latest.get("state") or "unknown")
    return {
        "status": status,
        "latest": latest,
        "history": history,
        "counts": counts,
        "watch": _self_heal_watch_snapshot(packets),
        "paths": {
            "failures_dir": str(failures_dir),
            "registry_file": str(failures_dir / "failure-registry.jsonl"),
        },
    }


def _latency_percentile(sorted_values: list[float], ratio: float) -> float | None:
    if not sorted_values:
        return None
    index = max(0, math.ceil(len(sorted_values) * ratio) - 1)
    return sorted_values[min(index, len(sorted_values) - 1)]


def _recall_eval_history(limit: int = 12) -> list[dict[str, Any]]:
    eval_dir = CHRONOVISOR_ROOT / "runtime" / "eval"
    if not eval_dir.exists():
        return []
    history: list[dict[str, Any]] = []
    for path in sorted(eval_dir.glob("*.json"))[-limit:]:
        data = _read_json_file(path)
        metrics = (data or {}).get("metrics")
        if not isinstance(metrics, dict):
            continue
        history.append(
            {
                "file": path.name,
                "examples": metrics.get("examples"),
                "recall_at_1": metrics.get("recall_at_1"),
                "recall_at_3": metrics.get("recall_at_3"),
                "waste_injection_rate": metrics.get("waste_injection_rate"),
                "latency_ms": metrics.get("latency_ms"),
                "policy": (data or {}).get("policy"),
            }
        )
    return history


def _recall_snapshot(limit: int = 400) -> dict[str, Any]:
    recall_dir = CHRONOVISOR_ROOT / "recall"
    rows = _read_jsonl_file(recall_dir / "recall-log.jsonl", limit=limit)

    decisions = {"none": 0, "search": 0, "read": 0}
    latencies: list[float] = []
    judge_used = 0
    rewrite_used = 0
    errors = 0
    for row in rows:
        decision = str(row.get("decision") or "")
        if decision in decisions:
            decisions[decision] += 1
        latency = row.get("latency_ms")
        if isinstance(latency, int | float):
            latencies.append(float(latency))
        if row.get("used_judge"):
            judge_used += 1
        features = row.get("evidence_features")
        if isinstance(features, dict):
            try:
                if float(features.get("rewrite_confidence") or 0.0) > 0:
                    rewrite_used += 1
            except (TypeError, ValueError):
                pass
        if row.get("error") or row.get("status") == "error":
            errors += 1
    latencies.sort()

    pulls = _read_jsonl_file(recall_dir / "pull-log.jsonl", limit=200)
    pull_counts: dict[str, int] = {}
    for pull in pulls:
        kind = str(pull.get("type") or "unknown")
        pull_counts[kind] = pull_counts.get(kind, 0) + 1

    calibration = _read_json_file(recall_dir / "calibration.json")
    calibration_history = _read_jsonl_file(
        recall_dir / "calibration-history.jsonl", limit=8
    )

    recent: list[dict[str, Any]] = []
    for row in rows[-12:]:
        recent.append(
            {
                "timestamp": row.get("ts"),
                "decision": row.get("decision"),
                "latency_ms": row.get("latency_ms"),
                "host": row.get("host"),
                "pages": len(row.get("pages") or []),
                "used_judge": bool(row.get("used_judge")),
                "preview": _shorten(row.get("prompt_preview"), 90),
            }
        )

    evals = _recall_eval_history()
    latest_eval = evals[-1] if evals else None

    return {
        "decisions": decisions,
        "samples": len(rows),
        "latency_ms": {
            "p50": _latency_percentile(latencies, 0.50),
            "p95": _latency_percentile(latencies, 0.95),
            "max": latencies[-1] if latencies else None,
        },
        "judge_used": judge_used,
        "rewrite_used": rewrite_used,
        "errors": errors,
        "pulls": {"total": len(pulls), "counts": pull_counts},
        "recent": recent,
        "evals": evals,
        "latest_eval": latest_eval,
        "calibration": {
            "current": calibration,
            "history": calibration_history,
            "last_applied": calibration_history[-1] if calibration_history else None,
        },
        "field": build_cortex_field_projection(
            CHRONOVISOR_ROOT,
            event_limit=128,
        ),
        "paths": {
            "recall_log": str(recall_dir / "recall-log.jsonl"),
            "pull_log": str(recall_dir / "pull-log.jsonl"),
            "calibration_file": str(recall_dir / "calibration.json"),
        },
    }


def _recall_improvement_snapshot() -> dict[str, Any]:
    try:
        from chronovisor.recall.recall_improvement import improvement_snapshot

        return improvement_snapshot()
    except Exception as exc:
        return {
            "status": "error",
            "error": exc.__class__.__name__,
            "active": None,
            "latest": None,
            "history": [],
            "counts": {},
        }


def _model_lab_snapshot() -> dict[str, Any]:
    try:
        return model_lab_snapshot()
    except Exception as exc:
        return {
            "status": "error",
            "error": exc.__class__.__name__,
            "policy": {"roles": {}},
            "candidates": [],
            "history": [],
        }


def _mark_batch_activity(status: dict[str, Any]) -> None:
    """Distinguish a retained completed batch from a currently running one."""

    batch = status.get("batch")
    if not isinstance(batch, dict):
        return
    annotated = dict(batch)
    annotated["active"] = bool(
        batch.get("total")
        and status.get("state") == "running"
        and status.get("stage") in ACTIVE_BATCH_STAGES
        and status.get("current_job_id")
    )
    status["batch"] = annotated


def _process_started_at(pid: object) -> datetime | None:
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "lstart="],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        value = result.stdout.strip()
        if not value:
            return None
        return datetime.strptime(value, "%a %b %d %H:%M:%S %Y")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _job_process_identity_matches(
    pid: object,
    job_started_at: object,
) -> bool:
    """Reject a live-but-reused PID left in durable orchestrator state."""

    if not runtime_status._pid_is_alive(pid):
        return False
    if not isinstance(job_started_at, str):
        return False
    try:
        job_start = datetime.fromisoformat(job_started_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    if job_start.tzinfo is not None:
        job_start = job_start.astimezone().replace(tzinfo=None)
    process_start = _process_started_at(pid)
    if process_start is None:
        return False
    # The orchestrator records current_job_started_at after this process was
    # launched. A process that started later is necessarily a reused PID.
    return process_start <= job_start


def _process_start_identity_status(
    pid: object,
    expected_started_at: object,
) -> str:
    """Classify exact process identity while preserving transient uncertainty."""

    if not runtime_status._pid_is_alive(pid) or not isinstance(
        expected_started_at, str
    ):
        return _PROCESS_IDENTITY_MISMATCH
    try:
        expected = datetime.fromisoformat(expected_started_at.replace("Z", "+00:00"))
    except ValueError:
        return _PROCESS_IDENTITY_MISMATCH
    if expected.tzinfo is not None:
        expected = expected.astimezone().replace(tzinfo=None)
    observed = _process_started_at(pid)
    if observed is None:
        return _PROCESS_IDENTITY_UNAVAILABLE
    if observed == expected:
        return _PROCESS_IDENTITY_MATCH
    return _PROCESS_IDENTITY_MISMATCH


def _timestamp_is_expired(value: object) -> bool:
    """Match the guard's lease semantics; missing or invalid legacy values survive."""

    if not isinstance(value, str) or not value:
        return False
    try:
        expires_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    now = datetime.now().astimezone()
    if expires_at.tzinfo is not None:
        now = now.astimezone(expires_at.tzinfo)
    else:
        now = now.replace(tzinfo=None)
    return expires_at <= now


def _canonicalize_runtime_status(
    status: dict[str, Any],
    orch_state: dict[str, Any],
    *,
    pending: int,
) -> dict[str, Any]:
    """Project durable orchestrator truth onto the dashboard status cache."""
    canonical = dict(status)
    job_id = orch_state.get("current_job_id")
    job_pid = orch_state.get("current_job_pid")
    try:
        from chronovisor.ingest.orchestrator import ingest_process_lease_is_held

        lease_active = ingest_process_lease_is_held(job_pid)
    except Exception:
        lease_active = False
    job_active = bool(
        job_id
        and lease_active
        and _job_process_identity_matches(
            job_pid,
            orch_state.get("current_job_started_at"),
        )
    )
    canonical["pending"] = pending

    if job_active:
        same_job = canonical.get("current_job_id") == job_id
        canonical["current_job_id"] = job_id
        canonical["current_job_pid"] = job_pid
        canonical["state"] = "running"
        if not same_job or canonical.get("stage") not in ACTIVE_BATCH_STAGES | {
            "locked"
        }:
            canonical["stage"] = "batch"
        canonical["updated_at"] = (
            canonical.get("updated_at")
            or orch_state.get("current_job_started_at")
            or orch_state.get("last_ingest")
        )
        return canonical

    canonical["state"] = "idle"
    canonical["stage"] = "waiting" if pending else "idle"
    canonical["current_raw"] = None
    canonical["current_op"] = None
    canonical["current_job_id"] = None
    canonical["current_job_pid"] = None
    canonical.pop("review_kind", None)
    llm = canonical.get("llm")
    if isinstance(llm, dict):
        canonical["llm"] = {**llm, "active": False}
    return canonical


def _safe_snapshot_component(
    name: str,
    builder: Any,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep one failed component from taking down the snapshot endpoint."""
    try:
        value = builder()
        if isinstance(value, dict):
            return value
        raise TypeError(
            f"snapshot component returned {type(value).__name__}, expected dict"
        )
    except Exception as exc:
        return {
            **(fallback or {}),
            "status": "error",
            "component": name,
            "error_class": exc.__class__.__name__,
            "error": str(exc),
        }


def _dashboard_glob_files(patterns: list[Path], *, limit: int = 0) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        try:
            matches = sorted(pattern.parent.glob(pattern.name))
        except OSError:
            continue
        files.extend(path for path in matches if path.is_file())
    if limit > 0:
        return files[-limit:]
    return files


def _dashboard_rglob_files(root: Path, pattern: str, *, limit: int = 0) -> list[Path]:
    try:
        files = sorted(path for path in root.rglob(pattern) if path.is_file())
    except OSError:
        return []
    if limit > 0:
        return files[-limit:]
    return files


def _deferred_materialization_fingerprint(raw_paths: list[Path]) -> str:
    failure_root = CHRONOVISOR_ROOT / "runtime" / "failures"
    paths = [
        CHRONOVISOR_ROOT / "config.toml",
        failure_root / "state.json",
        CHRONOVISOR_ROOT / "runtime" / "model-lab" / "active-policy.json",
        CHRONOVISOR_ROOT / "runtime" / "model-lab" / "state.json",
    ]
    paths.extend(_dashboard_glob_files([failure_root / "packets" / "*.json"]))
    paths.extend(
        _dashboard_rglob_files(
            CHRONOVISOR_ROOT / "runtime" / "decision-artifacts", "*.json"
        )
    )
    return _component_source_fingerprint(
        "deferred-statuses",
        paths,
        identities=[path.name for path in raw_paths],
    )


def _local_consensus_materialization_fingerprint() -> str:
    root = CHRONOVISOR_ROOT / "runtime" / "local-consensus"
    paths = [
        CHRONOVISOR_ROOT / "config.toml",
        root / "summary.json",
        root / "audit.jsonl",
        root / "trace-events.jsonl",
        CHRONOVISOR_ROOT / "runtime" / "model-lab" / "active-policy.json",
    ]
    paths.extend(_dashboard_glob_files([root / "active" / "*.json"]))
    paths.extend(
        _dashboard_rglob_files(
            CHRONOVISOR_ROOT / "runtime" / "decision-artifacts", "*.json", limit=32
        )
    )
    return _component_source_fingerprint("local-consensus", paths)


def _save_history_materialization_fingerprint(raw_paths: list[Path]) -> str:
    paths = [
        LOG_FILE,
        orchestrator.STATE_FILE,
        CHRONOVISOR_ROOT / "runtime" / "failures" / "state.json",
    ]
    paths.extend(
        _dashboard_glob_files(
            [
                CHRONOVISOR_ROOT / "logs" / "ingest-drain-*.jsonl",
                CHRONOVISOR_ROOT / "runtime" / "failures" / "packets" / "*.json",
                CHRONOVISOR_ROOT
                / "runtime"
                / "raw-projections"
                / "artifacts"
                / "semantic-*.manifest.json",
            ]
        )
    )
    return _component_source_fingerprint(
        "save-history",
        paths,
        identities=[path.name for path in raw_paths],
    )


def _health_materialization_fingerprint(raw_paths: list[Path]) -> str:
    paths = [
        CHRONOVISOR_ROOT / "config.toml",
        CHRONOVISOR_ROOT / "pages",
        CHRONOVISOR_ROOT / "claims",
        CHRONOVISOR_ROOT / "recall",
        CHRONOVISOR_ROOT / "review",
        CHRONOVISOR_ROOT / "eval",
        CHRONOVISOR_ROOT / "distill",
        CHRONOVISOR_ROOT / "autonomy",
        CHRONOVISOR_ROOT / "runtime" / "managed-holds" / "state.json",
        CHRONOVISOR_ROOT / "runtime" / "provisional-recall" / "index.json",
        CHRONOVISOR_ROOT / "runtime" / "quality" / "probe-latest.json",
        CHRONOVISOR_ROOT / "runtime" / "ingest-liveness.json",
        CHRONOVISOR_ROOT / "runtime" / "ingest-read-back-runs.jsonl",
        CHRONOVISOR_ROOT / "runtime" / "ingest-read-back-failures.jsonl",
        CHRONOVISOR_ROOT / "runtime" / "ingest-read-back-repair.json",
    ]
    paths.extend(
        _dashboard_glob_files(
            [
                CHRONOVISOR_ROOT / "claims" / "*.jsonl",
                CHRONOVISOR_ROOT / "recall" / "*.json",
                CHRONOVISOR_ROOT / "recall" / "*.jsonl",
                CHRONOVISOR_ROOT / "review" / "*.jsonl",
                CHRONOVISOR_ROOT / "eval" / "*.json",
                CHRONOVISOR_ROOT / "distill" / "*.jsonl",
                CHRONOVISOR_ROOT / "autonomy" / "*.json",
            ]
        )
    )
    paths.extend(
        _dashboard_rglob_files(
            CHRONOVISOR_ROOT / "runtime" / "research" / "runs",
            "summary.json",
            limit=50,
        )
    )
    runtime = runtime_identity()
    runtime_cache_identity = hashlib.sha256(
        _canonical_json_bytes(
            {
                "commit_id": runtime.get("commit_id"),
                "module_path": runtime.get("module_path"),
                "package_version": runtime.get("package_version"),
            }
        )
    ).hexdigest()
    return _component_source_fingerprint(
        "health",
        paths,
        identities=[runtime_cache_identity, *(path.name for path in raw_paths)],
    )


def _model_status_materialization_fingerprint(ollama: dict[str, Any]) -> str:
    paths = [
        CHRONOVISOR_ROOT / "config.toml",
        CHRONOVISOR_ROOT / "runtime" / "model-lab" / "active-policy.json",
        CHRONOVISOR_ROOT / "runtime" / "model-lab" / "state.json",
    ]
    model_identity = hashlib.sha256(
        _canonical_json_bytes(ollama.get("models") or [])
    ).hexdigest()
    return _component_source_fingerprint(
        "model-status",
        paths,
        identities=[model_identity],
    )


def build_fast_snapshot() -> dict[str, Any]:
    """Return the live status shell without scanning Raw or audit history."""

    init_chronovisor()
    from chronovisor.librarian.librarian_status import build_librarian_status

    status = runtime_status.read_status()
    if not isinstance(status, dict):
        status = {}
    return {
        "status": status,
        "events": runtime_status.read_events(limit=40)[-40:],
        "metrics": runtime_status.read_metrics(limit=60)[-60:],
        "local_consensus": status.get("local_consensus") or {},
        "frontier_repair": status.get("frontier_repair") or {},
        "ollama": {},
        "model_status": {},
        "self_heal": {},
        "recall": {},
        "recall_improvement": {},
        "model_lab": {},
        "typed_graph": _typed_graph_dashboard_snapshot(),
        "save_history": {},
        "knowledge_mix": {},
        "librarian": _safe_snapshot_component(
            "librarian",
            lambda: build_librarian_status(CHRONOVISOR_ROOT),
            {
                "state": "BLOCKED",
                "detail": "librarian status unavailable",
                "progress": {},
                "queue": {},
                "flow": {},
                "recent_receipts": [],
            },
        ),
        "health": {},
        "_dashboard": {"detail_state": "loading"},
    }


def build_snapshot() -> dict[str, Any]:
    init_chronovisor()
    from chronovisor.librarian.librarian_status import build_librarian_status

    cached_status = runtime_status.read_status()
    orch_state = orchestrator._load_state()
    from chronovisor.raw.raw_store import RawStore

    raw_dir = CHRONOVISOR_ROOT / "raw"
    raw_store = RawStore(raw_dir)
    reference_dir = CHRONOVISOR_ROOT / "runtime" / "raw-projections" / "parents"
    raw_paths = sorted(
        unit.path
        if unit.storage == "legacy_file"
        else raw_store.materialize_ingest(unit, reference_dir)
        for unit in raw_store.iter_units()
    )
    artifact_dir = CHRONOVISOR_ROOT / "runtime" / "raw-projections" / "artifacts"
    if artifact_dir.exists():
        # A migrated legacy archive can contain a semantic child that is also
        # present in the projection artifact store. Queue identity is the Raw
        # basename, so count it once and prefer the directly readable artifact.
        paths_by_raw_id = {path.name: path for path in raw_paths}
        paths_by_raw_id.update(
            {path.name: path for path in sorted(artifact_dir.glob("*.md"))}
        )
        raw_paths = sorted(paths_by_raw_id.values(), key=lambda path: path.name)
    deferred_view = _materialized_component(
        "deferred-statuses",
        fingerprint=_deferred_materialization_fingerprint(raw_paths),
        builder=lambda: {
            "statuses": _operational_deferred_raw_statuses(raw_paths),
        },
    )
    cached_deferred = deferred_view.get("statuses")
    deferred_statuses = (
        {
            str(name): str(reason)
            for name, reason in cached_deferred.items()
            if isinstance(name, str) and isinstance(reason, str)
        }
        if isinstance(cached_deferred, dict)
        else _operational_deferred_raw_statuses(raw_paths)
    )
    processed_raw_files = orch_state.get("processed_raw_files")
    processed_raw_names = (
        {name for name in processed_raw_files if isinstance(name, str)}
        if isinstance(processed_raw_files, list)
        else set()
    )
    from chronovisor.raw.raw_replay import is_raw_retracted

    pending = sum(
        1
        for raw_path in raw_paths
        if raw_path.name not in processed_raw_names
        and raw_path.name not in deferred_statuses
        and not is_raw_retracted(raw_path)
    )
    semantic_deferred_names = sorted(
        raw_file
        for raw_file, reason in deferred_statuses.items()
        if reason == "semantic_no_quorum"
    )
    operational_deferred_names = sorted(
        raw_file
        for raw_file, reason in deferred_statuses.items()
        if reason != "semantic_no_quorum"
    )
    status = _canonicalize_runtime_status(cached_status, orch_state, pending=pending)
    status["semantic_deferred"] = {
        "count": len(semantic_deferred_names),
        "samples": semantic_deferred_names[:5],
    }
    status["operational_deferred"] = {
        "count": len(operational_deferred_names),
        "samples": operational_deferred_names[:5],
    }
    status["raw_outstanding"] = pending + len(deferred_statuses)

    local_consensus = _safe_snapshot_component(
        "local_consensus",
        lambda: _materialized_component(
            "local-consensus",
            fingerprint=_local_consensus_materialization_fingerprint(),
            builder=_local_consensus_snapshot,
            audit_seconds=DASHBOARD_LOCAL_CONSENSUS_AUDIT_SECONDS,
        ),
        {
            "active": False,
            "count": 0,
            "activities": [],
            "summary": _empty_local_consensus_summary(),
            "history": [],
        },
    )
    frontier_activity = _safe_snapshot_component(
        "frontier_process_activity",
        _frontier_activity_snapshot,
        {"active": False, "count": 0, "reviews": [], "latest": None},
    )
    frontier_repair = _safe_snapshot_component(
        "frontier_repair",
        _frontier_repair_snapshot,
        {
            "available": False,
            "active": False,
            "summary": {"total": 0, "starts_24h": 0, "counts": {}},
            "recent": [],
            "events": [],
        },
    )
    frontier_repair = {
        **frontier_repair,
        "active": bool(
            frontier_repair.get("active") or frontier_activity.get("active")
        ),
        "process_activity": frontier_activity,
    }
    status["local_consensus"] = local_consensus
    status["frontier_repair"] = frontier_repair
    # Compatibility for older dashboard clients. This is repair-plane activity,
    # never a routine semantic-review tier.
    status["frontier_review"] = frontier_activity
    active_llm = isinstance(status.get("llm"), dict) and status["llm"].get("active")
    local_review_active = bool(local_consensus.get("active"))
    repair_active = bool(frontier_repair.get("active"))
    if (
        (local_review_active or repair_active)
        and not status.get("current_job_id")
        and not status.get("current_raw")
        and not active_llm
    ):
        status["state"] = "running"
        status["stage"] = "review"
        latest_review = (
            local_consensus.get("latest")
            if local_review_active
            else frontier_activity.get("latest")
            or frontier_repair.get("active_incident")
        ) or {}
        latest_role = str(latest_review.get("role") or "")
        if local_review_active:
            status["review_kind"] = (
                "local_model_eval"
                if latest_role.startswith("model_eval:")
                else "local_consensus"
            )
        else:
            status["review_kind"] = "frontier_repair"
        status["updated_at"] = latest_review.get("started_at") or status.get(
            "updated_at"
        )
    _mark_batch_activity(status)

    runtime_metrics = runtime_status.read_metrics(limit=240)
    drain_metrics = _drain_history(limit=240)
    metrics = _merge_metric_history(runtime_metrics, drain_metrics, limit=240)
    metrics.append(
        {
            "timestamp": runtime_status.now_iso(),
            "kind": "current",
            "pending_after": pending,
            "files_processed": 0,
            "files_deferred": 0,
            "files_continued": 0,
            "files_failed": 0,
        }
    )
    events = (runtime_status.read_events(limit=120) + _recent_log_events(limit=80))[
        -160:
    ]
    ollama = _ollama_snapshot()
    model_status = _safe_snapshot_component(
        "model_status",
        lambda: _materialized_component(
            "model-status",
            fingerprint=_model_status_materialization_fingerprint(ollama),
            builder=lambda: _model_status_snapshot(ollama),
            audit_seconds=DASHBOARD_HEALTH_AUDIT_SECONDS,
        ),
        {"available": False, "models": [], "summary": {}},
    )
    from chronovisor.core.runtime_config import runtime_identity
    from chronovisor.decision.decision_policy import decision_policy_snapshot

    decision_policies = _safe_snapshot_component(
        "decision_policies",
        decision_policy_snapshot,
        {"lanes": {}, "counts": {"off": 0, "shadow": 0, "enabled": 0}},
    )
    status["decision_policies"] = decision_policies
    from chronovisor.raw.raw_archive import archive_status

    raw_archive = _safe_snapshot_component(
        "raw_archive",
        lambda: archive_status(raw_dir),
        {
            "logical_units": len(raw_paths),
            "open_segments": 0,
            "sealed_segments": 0,
            "legacy_archives": 0,
            "unsealed_bytes": 0,
        },
    )

    return {
        "runtime": runtime_identity(),
        "status": status,
        "decision_policies": decision_policies,
        "local_consensus": local_consensus,
        "frontier_repair": frontier_repair,
        "frontier_review": frontier_activity,
        "orchestrator": {
            "last_ingest": orch_state.get("last_ingest"),
            "last_lint": orch_state.get("last_lint"),
            "triage_failure_count": orch_state.get("triage_failure_count", 0),
        },
        "raw_archive": raw_archive,
        "ollama": ollama,
        "model_status": model_status,
        "librarian": _safe_snapshot_component(
            "librarian",
            lambda: build_librarian_status(CHRONOVISOR_ROOT),
            {
                "state": "BLOCKED",
                "detail": "librarian status unavailable",
                "progress": {},
                "queue": {},
                "flow": {},
                "recent_receipts": [],
            },
        ),
        "events": events,
        "metrics": metrics,
        "self_heal": _safe_snapshot_component(
            "self_heal",
            _self_heal_snapshot,
            {"history": [], "counts": {}, "watch": {}},
        ),
        "recall": _safe_snapshot_component(
            "recall",
            _recall_snapshot,
            {"recent": [], "evals": [], "errors": 0},
        ),
        "recall_improvement": _safe_snapshot_component(
            "recall_improvement",
            _recall_improvement_snapshot,
            {"active": None, "latest": None, "history": [], "counts": {}},
        ),
        "model_lab": _safe_snapshot_component(
            "model_lab",
            _model_lab_snapshot,
            {"policy": {"roles": {}}, "candidates": [], "history": []},
        ),
        "typed_graph": _safe_typed_graph_snapshot(),
        "save_history": _safe_snapshot_component(
            "save_history",
            lambda: _materialized_component(
                "save-history",
                fingerprint=_save_history_materialization_fingerprint(raw_paths),
                builder=lambda: _save_history_snapshot(
                    raw_paths=raw_paths,
                    deferred_statuses=deferred_statuses,
                ),
            ),
            {"days": [], "recent": [], "totals": {}, "sources": []},
        ),
        "knowledge_mix": _safe_snapshot_component(
            "knowledge_mix",
            _knowledge_mix_snapshot,
            {"total_pages": 0, "total_bytes": 0, "categories": [], "top": []},
        ),
        "health": _safe_snapshot_component(
            "health",
            lambda: _materialized_component(
                "health",
                fingerprint=_health_materialization_fingerprint(raw_paths),
                builder=health_snapshot,
                audit_seconds=DASHBOARD_HEALTH_AUDIT_SECONDS,
            ),
        ),
        "paths": {
            "chronovisor_root": str(CHRONOVISOR_ROOT),
            "status_file": str(runtime_status.STATUS_FILE),
            "events_file": str(runtime_status.EVENTS_FILE),
            "metrics_file": str(runtime_status.METRICS_FILE),
            "local_consensus": str(CHRONOVISOR_ROOT / "runtime" / "local-consensus"),
            "frontier_repair": str(CHRONOVISOR_ROOT / "runtime" / "frontier-repair"),
        },
    }


def _snapshot_fixed_source_paths() -> tuple[Path, ...]:
    # Directory identities are the immediate coarse signal. Append/in-place
    # cold inputs that do not move a directory entry are intentionally bounded
    # by the 30s active / 60s idle snapshot TTL instead of triggering rebuilds.
    # Local Consensus live files are also excluded here: processing activity
    # and /api/local-consensus remain live, while the aggregate snapshot is
    # deliberately bounded by the same active/idle TTLs.
    return (
        CHRONOVISOR_ROOT / "raw",
        CHRONOVISOR_ROOT / "pages",
        CHRONOVISOR_ROOT / "runtime" / "raw-projections" / "parents",
        CHRONOVISOR_ROOT / "runtime" / "raw-projections" / "artifacts",
        CHRONOVISOR_ROOT / "runtime" / "ingest-liveness.json",
        CHRONOVISOR_ROOT / "runtime" / "ingest-read-back-runs.jsonl",
        CHRONOVISOR_ROOT / "runtime" / "ingest-read-back-failures.jsonl",
        CHRONOVISOR_ROOT / "runtime" / "ingest-read-back-repair.json",
        CHRONOVISOR_ROOT / "runtime" / "librarian",
        CHRONOVISOR_ROOT / "runtime" / "librarian" / "events.jsonl",
        CHRONOVISOR_ROOT / "runtime" / "typed-graph" / "status.json",
        CHRONOVISOR_ROOT / "runtime" / "typed-graph" / "promotion.json",
        CHRONOVISOR_ROOT / "runtime" / "recall-rubric" / "status.json",
        CHRONOVISOR_ROOT / "runtime" / "failures" / "state.json",
        CHRONOVISOR_ROOT / "runtime" / "failures" / "failure-registry.jsonl",
        CHRONOVISOR_ROOT / "runtime" / "convergence" / "state.json",
        CHRONOVISOR_ROOT / "runtime" / "frontier-reviews" / "active",
        CHRONOVISOR_ROOT / "runtime" / "frontier-repair" / "state.json",
        CHRONOVISOR_ROOT / "runtime" / "frontier-repair" / "events.jsonl",
        CHRONOVISOR_ROOT / "runtime" / "recall-improvement" / "active-policy.json",
        CHRONOVISOR_ROOT / "runtime" / "recall-improvement" / "policy-registry.jsonl",
        CHRONOVISOR_ROOT / "runtime" / "recall-improvement" / "schedule-state.json",
        CHRONOVISOR_ROOT / "runtime" / "model-lab" / "active-policy.json",
        CHRONOVISOR_ROOT / "runtime" / "model-lab" / "state.json",
        CHRONOVISOR_ROOT / "runtime" / "model-lab" / "history.jsonl",
        CHRONOVISOR_ROOT / "runtime" / "model-lab" / "replay.jsonl",
        CHRONOVISOR_ROOT / "runtime" / "decision-artifacts",
        CHRONOVISOR_ROOT / "runtime" / "managed-holds" / "state.json",
        CHRONOVISOR_ROOT / "runtime" / "provisional-recall" / "index.json",
        CHRONOVISOR_ROOT / "runtime" / "quality" / "probe-latest.json",
        CHRONOVISOR_ROOT / "autonomy" / "watchdog-heartbeat.json",
        CHRONOVISOR_ROOT / "autonomy" / "observer-heartbeat.json",
        CHRONOVISOR_ROOT / "autonomy" / "observer-threshold-state.json",
        CHRONOVISOR_ROOT / "review" / "raw-replay-queue.jsonl",
        CHRONOVISOR_ROOT / "runtime" / "raw-replay-history.jsonl",
        CHRONOVISOR_ROOT / "runtime" / "raw-replay-completions.jsonl",
        CHRONOVISOR_ROOT / "recall" / "recall-log.jsonl",
        CHRONOVISOR_ROOT / "recall" / "pull-log.jsonl",
        CHRONOVISOR_ROOT / "recall" / "calibration.json",
        CHRONOVISOR_ROOT / "recall" / "calibration-history.jsonl",
        CHRONOVISOR_ROOT / "config.toml",
        LOG_FILE,
        orchestrator.STATE_FILE,
        runtime_status.EVENTS_FILE,
        runtime_status.METRICS_FILE,
    )


def _snapshot_runtime_epoch() -> tuple[Any, ...]:
    """Return stable live semantics, excluding high-frequency stage churn."""

    try:
        value = runtime_status.read_status()
    except Exception as exc:
        return ("unavailable", exc.__class__.__name__)
    status = value if isinstance(value, dict) else {}
    active = bool(
        status.get("state") == "running"
        or status.get("current_job_id")
        or status.get("current_raw")
    )
    return (
        "active" if active else "idle",
        str(status.get("current_job_id") or ""),
        str(status.get("current_raw") or ""),
    )


def _snapshot_dynamic_source_patterns() -> tuple[tuple[Path, int], ...]:
    return (
        (CHRONOVISOR_ROOT / "runtime" / "frontier-reviews" / "active" / "*.json", 8),
        (CHRONOVISOR_ROOT / "runtime" / "failures" / "packets" / "*.json", 24),
        (CHRONOVISOR_ROOT / "runtime" / "eval" / "*.json", 16),
        (CHRONOVISOR_ROOT / "runtime" / "recall-improvement" / "runs" / "*.json", 16),
        (CHRONOVISOR_ROOT / "runtime" / "quality" / "lanes" / "*.json", 32),
        (CHRONOVISOR_ROOT / "logs" / "ingest-drain-*.jsonl", 14),
    )


def _snapshot_source_probe_identity() -> tuple[Any, ...]:
    """Bounded O(1) identity used to reuse the expensive fingerprint scan."""

    dynamic_directories = tuple(
        dict.fromkeys(
            pattern.parent for pattern, _limit in _snapshot_dynamic_source_patterns()
        )
    )
    decision_artifacts = CHRONOVISOR_ROOT / "runtime" / "decision-artifacts"
    paths = (*_snapshot_fixed_source_paths(), *dynamic_directories, decision_artifacts)
    return (
        str(CHRONOVISOR_ROOT),
        id(build_snapshot),
        id(_build_snapshot_source_fingerprint),
        id(_snapshot_runtime_epoch),
        _snapshot_runtime_epoch(),
        *(_path_identity(path) for path in paths),
    )


def _build_snapshot_source_fingerprint() -> tuple[Any, ...]:
    """Scan bounded dynamic tails for a full dashboard source identity."""

    tracked = list(_snapshot_fixed_source_paths())
    # Existing append-only files do not change their parent directory mtime.
    # Bound each dynamic tail so the cache check stays cheap while still
    # observing standalone review activity that does not write status.json.
    for pattern, limit in _snapshot_dynamic_source_patterns():
        tracked.extend(sorted(pattern.parent.glob(pattern.name))[-limit:])
    tracked.extend(
        sorted((CHRONOVISOR_ROOT / "runtime" / "decision-artifacts").glob("*/*.json"))[
            -32:
        ]
    )
    identities: list[tuple[Any, ...]] = []
    for path in dict.fromkeys(tracked):
        try:
            stat = path.stat()
        except OSError:
            identities.append((str(path), "missing"))
            continue
        identities.append((str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns))
    # Tests and embedded callers may replace the builder without changing the
    # wiki paths. Binding the callable prevents a cached success from masking
    # the replacement (or its exception).
    return (id(build_snapshot), _snapshot_runtime_epoch(), *identities)


def _invalidate_snapshot_fingerprint_probe() -> None:
    with _SNAPSHOT_FINGERPRINT_CONDITION:
        while _SNAPSHOT_FINGERPRINT_CACHE.get("probing"):
            _SNAPSHOT_FINGERPRINT_CONDITION.wait()
        _SNAPSHOT_FINGERPRINT_CACHE.update(
            {"source": None, "fingerprint": None, "audited_at": 0.0}
        )


def _snapshot_source_fingerprint() -> tuple[Any, ...]:
    """Share one expensive source scan across concurrent snapshot requests."""

    source = _snapshot_source_probe_identity()
    while True:
        with _SNAPSHOT_FINGERPRINT_CONDITION:
            now = time.monotonic()
            cached = _SNAPSHOT_FINGERPRINT_CACHE.get("fingerprint")
            if (
                isinstance(cached, tuple)
                and _SNAPSHOT_FINGERPRINT_CACHE.get("source") == source
                and now - float(
                    _SNAPSHOT_FINGERPRINT_CACHE.get("audited_at") or 0.0
                )
                < SNAPSHOT_FINGERPRINT_AUDIT_SECONDS
            ):
                _SNAPSHOT_FINGERPRINT_CACHE["cache_hits"] = int(
                    _SNAPSHOT_FINGERPRINT_CACHE.get("cache_hits") or 0
                ) + 1
                return cached
            if _SNAPSHOT_FINGERPRINT_CACHE.get("probing"):
                _SNAPSHOT_FINGERPRINT_CACHE["coalesced"] = int(
                    _SNAPSHOT_FINGERPRINT_CACHE.get("coalesced") or 0
                ) + 1
                _SNAPSHOT_FINGERPRINT_CONDITION.wait()
                source = _snapshot_source_probe_identity()
                continue
            _SNAPSHOT_FINGERPRINT_CACHE["probing"] = True
            break

    try:
        fingerprint = _build_snapshot_source_fingerprint()
        completed_source = _snapshot_source_probe_identity()
    except Exception:
        with _SNAPSHOT_FINGERPRINT_CONDITION:
            _SNAPSHOT_FINGERPRINT_CACHE["probing"] = False
            _SNAPSHOT_FINGERPRINT_CACHE["error_count"] = int(
                _SNAPSHOT_FINGERPRINT_CACHE.get("error_count") or 0
            ) + 1
            _SNAPSHOT_FINGERPRINT_CONDITION.notify_all()
        raise
    with _SNAPSHOT_FINGERPRINT_CONDITION:
        _SNAPSHOT_FINGERPRINT_CACHE.update(
            {
                "source": completed_source,
                "fingerprint": fingerprint,
                "audited_at": time.monotonic(),
                "probing": False,
                "generation": int(
                    _SNAPSHOT_FINGERPRINT_CACHE.get("generation") or 0
                )
                + 1,
                "probe_count": int(
                    _SNAPSHOT_FINGERPRINT_CACHE.get("probe_count") or 0
                )
                + 1,
            }
        )
        _SNAPSHOT_FINGERPRINT_CONDITION.notify_all()
    return fingerprint


def _snapshot_is_active(snapshot: dict[str, Any]) -> bool:
    status = snapshot.get("status")
    if not isinstance(status, dict):
        return True
    batch = status.get("batch") if isinstance(status.get("batch"), dict) else {}
    llm = status.get("llm") if isinstance(status.get("llm"), dict) else {}
    local = (
        status.get("local_consensus")
        if isinstance(status.get("local_consensus"), dict)
        else {}
    )
    repair = (
        status.get("frontier_repair")
        if isinstance(status.get("frontier_repair"), dict)
        else {}
    )
    return bool(
        status.get("state") == "running"
        or batch.get("active") is True
        or llm.get("active") is True
        or local.get("active") is True
        or repair.get("active") is True
    )


def _build_snapshot_cache(
    fingerprint: tuple[Any, ...], observed_built_at: float
) -> dict[str, Any]:
    """Build one full snapshot and publish it atomically to the cache."""

    try:
        with _SNAPSHOT_BUILD_LOCK:
            with _SNAPSHOT_CACHE_LOCK:
                cached = _SNAPSHOT_CACHE.get("snapshot")
                if (
                    isinstance(cached, dict)
                    and _SNAPSHOT_CACHE.get("fingerprint") == fingerprint
                    and float(_SNAPSHOT_CACHE.get("built_at") or 0.0)
                    != observed_built_at
                ):
                    _SNAPSHOT_CACHE["refreshing"] = False
                    return cached
            snapshot = build_snapshot()
            # The build may have changed materialized files itself. Bypass the
            # short probe reuse so this stability check observes the true
            # post-build source epoch.
            _invalidate_snapshot_fingerprint_probe()
            post_build_fingerprint = _snapshot_source_fingerprint()
    except Exception:
        with _SNAPSHOT_CACHE_LOCK:
            _SNAPSHOT_CACHE["refreshing"] = False
        raise
    with _SNAPSHOT_CACHE_LOCK:
        _SNAPSHOT_CACHE.update(
            {
                "built_at": time.monotonic(),
                # A write during the multi-component build may have happened
                # after its component was read. Leave the cache deliberately
                # unmatched so the next request rebuilds from one stable
                # source epoch instead of serving false idle for the full TTL.
                "fingerprint": (
                    post_build_fingerprint
                    if post_build_fingerprint == fingerprint
                    else None
                ),
                "snapshot": snapshot,
                "refreshing": False,
            }
        )
    return snapshot


def _refresh_snapshot_cache(
    fingerprint: tuple[Any, ...], observed_built_at: float
) -> None:
    """Refresh a stale dashboard snapshot without blocking an HTTP request."""

    try:
        _build_snapshot_cache(fingerprint, observed_built_at)
    except Exception:
        # The last successful snapshot remains available. A later poll retries.
        return


def _cached_snapshot(*, allow_stale: bool = False) -> dict[str, Any]:
    """Single-flight expensive snapshots and reuse unchanged idle results."""

    # Probe before the snapshot cache lock. The fingerprint cache has its own
    # condition and no code path holds both locks, avoiding lock-order cycles.
    fingerprint = _snapshot_source_fingerprint()
    with _SNAPSHOT_CACHE_LOCK:
        now = time.monotonic()
        observed_built_at = float(_SNAPSHOT_CACHE.get("built_at") or 0.0)
        cached = _SNAPSHOT_CACHE.get("snapshot")
        if isinstance(cached, dict):
            max_age = (
                SNAPSHOT_ACTIVE_CACHE_SECONDS
                if _snapshot_is_active(cached)
                else SNAPSHOT_IDLE_CACHE_SECONDS
            )
            if (
                _SNAPSHOT_CACHE.get("fingerprint") == fingerprint
                and now - observed_built_at < max_age
            ):
                return cached
            if allow_stale:
                start_refresh = not bool(_SNAPSHOT_CACHE.get("refreshing"))
                if start_refresh:
                    _SNAPSHOT_CACHE["refreshing"] = True
                stale = {
                    **cached,
                    "_dashboard": {
                        "detail_state": "refreshing",
                        "stale": True,
                    },
                }
            else:
                start_refresh = False
                stale = None
        else:
            start_refresh = False
            stale = None

    if stale is not None:
        if start_refresh:
            threading.Thread(
                target=_refresh_snapshot_cache,
                args=(fingerprint, observed_built_at),
                name="chronovisor-dashboard-refresh",
                daemon=True,
            ).start()
        return stale
    return _build_snapshot_cache(fingerprint, observed_built_at)


_COLD_STATUS_DERIVED_KEYS = (
    "local_consensus",
    "frontier_repair",
    "frontier_review",
    "decision_policies",
    "semantic_deferred",
    "operational_deferred",
    "raw_outstanding",
)


def _snapshot_with_live_status(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Overlay cheap runtime truth without mutating cached cold aggregates."""

    cached_value = snapshot.get("status")
    cached_status = cached_value if isinstance(cached_value, dict) else {}
    pending_value = cached_status.get("pending")
    pending = (
        int(pending_value)
        if isinstance(pending_value, int) and not isinstance(pending_value, bool)
        else 0
    )
    dashboard_value = snapshot.get("_dashboard")
    dashboard_state = dashboard_value if isinstance(dashboard_value, dict) else {}
    try:
        live_value = runtime_status.read_status()
        live_status = live_value if isinstance(live_value, dict) else {}
        canonical_live = _canonicalize_runtime_status(
            live_status,
            orchestrator._load_state(),
            pending=max(0, pending),
        )
    except Exception as exc:
        return {
            **snapshot,
            "_dashboard": {
                **dashboard_state,
                "live_overlay": False,
                "live_overlay_error": exc.__class__.__name__,
            },
        }

    status = {**cached_status, **canonical_live}
    for key in _COLD_STATUS_DERIVED_KEYS:
        if key in cached_status:
            status[key] = cached_status[key]
    batch = status.get("batch")
    if isinstance(batch, dict):
        status["batch"] = dict(batch)
    llm = status.get("llm")
    if isinstance(llm, dict):
        status["llm"] = dict(llm)
    _mark_batch_activity(status)
    return {
        **snapshot,
        "status": status,
        "_dashboard": {**dashboard_state, "live_overlay": True},
    }


def _load_or_create_dashboard_token(path: Path) -> str:
    try:
        existing = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        existing = ""
    if existing:
        if DASHBOARD_TOKEN_RE.fullmatch(existing) is None:
            raise RuntimeError(f"dashboard access token is malformed: {path}")
        os.chmod(path, 0o600)
        return existing

    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        existing = path.read_text(encoding="utf-8").strip()
        if DASHBOARD_TOKEN_RE.fullmatch(existing) is None:
            raise RuntimeError(f"dashboard access token is malformed: {path}") from exc
        os.chmod(path, 0o600)
        return existing
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(token + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return token


def _rotate_dashboard_token(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return token


def _password_digest(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=DASHBOARD_PASSWORD_SCRYPT_N,
        r=DASHBOARD_PASSWORD_SCRYPT_R,
        p=DASHBOARD_PASSWORD_SCRYPT_P,
        dklen=DASHBOARD_PASSWORD_DKLEN,
        maxmem=64 * 1024 * 1024,
    )


def _write_dashboard_credentials(path: Path, username: str, password: str) -> None:
    username = username.strip()
    if not username or len(username) > 64:
        raise ValueError("dashboard username must contain 1 to 64 characters")
    if not password:
        raise ValueError("dashboard password must not be empty")

    salt = secrets.token_bytes(16)
    digest = _password_digest(password, salt)
    payload = {
        "version": DASHBOARD_CREDENTIAL_VERSION,
        "username": username,
        "password": {
            "algorithm": DASHBOARD_PASSWORD_ALGORITHM,
            "salt": base64.b64encode(salt).decode("ascii"),
            "digest": base64.b64encode(digest).decode("ascii"),
            "n": DASHBOARD_PASSWORD_SCRYPT_N,
            "r": DASHBOARD_PASSWORD_SCRYPT_R,
            "p": DASHBOARD_PASSWORD_SCRYPT_P,
            "dklen": DASHBOARD_PASSWORD_DKLEN,
        },
        "created_at": datetime.now().astimezone().isoformat(),
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _load_dashboard_credentials(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"dashboard credentials are unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"dashboard credentials are malformed: {path}")
    password = payload.get("password")
    if (
        payload.get("version") != DASHBOARD_CREDENTIAL_VERSION
        or not isinstance(payload.get("username"), str)
        or not isinstance(password, dict)
        or password.get("algorithm") != DASHBOARD_PASSWORD_ALGORITHM
    ):
        raise RuntimeError(f"dashboard credentials are malformed: {path}")
    try:
        salt = base64.b64decode(str(password["salt"]), validate=True)
        digest = base64.b64decode(str(password["digest"]), validate=True)
    except (KeyError, ValueError, binascii.Error) as exc:
        raise RuntimeError(f"dashboard credentials are malformed: {path}") from exc
    if len(salt) != 16 or len(digest) != DASHBOARD_PASSWORD_DKLEN:
        raise RuntimeError(f"dashboard credentials are malformed: {path}")
    expected_parameters = {
        "n": DASHBOARD_PASSWORD_SCRYPT_N,
        "r": DASHBOARD_PASSWORD_SCRYPT_R,
        "p": DASHBOARD_PASSWORD_SCRYPT_P,
        "dklen": DASHBOARD_PASSWORD_DKLEN,
    }
    if any(password.get(key) != value for key, value in expected_parameters.items()):
        raise RuntimeError(f"dashboard credentials are malformed: {path}")
    os.chmod(path, 0o600)
    return payload


def _dashboard_credentials_match(
    credentials: dict[str, Any] | None,
    username: str,
    password: str,
) -> bool:
    if not credentials:
        return False
    stored_password = credentials.get("password")
    if not isinstance(stored_password, dict):
        return False
    try:
        salt = base64.b64decode(str(stored_password["salt"]), validate=True)
        expected = base64.b64decode(str(stored_password["digest"]), validate=True)
        actual = _password_digest(password, salt)
    except (KeyError, ValueError, binascii.Error):
        return False
    stored_username = str(credentials.get("username") or "")
    return hmac.compare_digest(username, stored_username) and hmac.compare_digest(
        actual, expected
    )


def _normalized_client_ip(
    value: object,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        address = ipaddress.ip_address(str(value))
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        return address.ipv4_mapped
    return address


def _private_client_scope(value: object) -> str:
    address = _normalized_client_ip(value)
    if address is None:
        return "invalid"
    if address.is_loopback:
        return "loopback"
    if address.is_private or address.is_link_local:
        return "private"
    return "public"


def _dashboard_lan_hosts() -> list[str]:
    local_name = re.sub(
        r"[^A-Za-z0-9-]", "-", socket.gethostname().split(".")[0]
    ).strip("-")
    mdns_hosts = [f"{local_name}.local"] if local_name else []
    private_hosts: list[str] = []
    link_local_hosts: list[str] = []
    for interface in ("en0", "en1"):
        try:
            result = subprocess.run(
                ["/usr/sbin/ipconfig", "getifaddr", interface],
                capture_output=True,
                check=False,
                text=True,
                timeout=1,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        candidate = result.stdout.strip()
        address = _normalized_client_ip(candidate)
        if address is None or _private_client_scope(candidate) != "private":
            continue
        target = link_local_hosts if address.is_link_local else private_hosts
        if candidate not in target:
            target.append(candidate)

    # The share button copies the first URL. Prefer the directly routable LAN
    # address because mDNS is not available on every phone/PC and a link-local
    # address is usually the inactive interface on a multi-homed Mac.
    return [*private_hosts, *mdns_hosts, *link_local_hosts]


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "LLMWikiDashboard/0.1"

    def _access_token(self) -> str:
        return str(getattr(self.server, "lan_access_token", "") or "")

    def _credentials(self) -> dict[str, Any] | None:
        value = getattr(self.server, "dashboard_credentials", None)
        return value if isinstance(value, dict) else None

    def _is_loopback(self) -> bool:
        return _private_client_scope(self.client_address[0]) == "loopback"

    def _cookie_authorized(self, token: str) -> bool:
        try:
            cookies = SimpleCookie()
            cookies.load(self.headers.get("Cookie", ""))
            supplied = cookies.get(DASHBOARD_ACCESS_COOKIE)
        except Exception:
            return False
        return bool(supplied and hmac.compare_digest(supplied.value, token))

    def _set_access_cookie(self, token: str, *, max_age: int) -> None:
        value = token if max_age > 0 else ""
        self.send_header(
            "Set-Cookie",
            f"{DASHBOARD_ACCESS_COOKIE}={value}; Path=/; Max-Age={max_age}; "
            "HttpOnly; SameSite=Strict",
        )

    def _redirect_after_token(self, parsed: Any, token: str) -> None:
        clean_query = urlencode(
            [
                (key, value)
                for key, value in parse_qsl(parsed.query, keep_blank_values=True)
                if key != DASHBOARD_ACCESS_QUERY
            ]
        )
        location = parsed.path or "/"
        if clean_query:
            location += f"?{clean_query}"
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self._set_access_cookie(
            token,
            max_age=DASHBOARD_SESSION_MAX_AGE_SECONDS,
        )
        self.send_header("Cache-Control", "no-store")
        _send_security_headers(self)
        self.end_headers()

    def _basic_authorized(self, credentials: dict[str, Any] | None) -> bool:
        authorization = self.headers.get("Authorization", "")
        scheme, separator, encoded = authorization.partition(" ")
        if not separator or scheme.lower() != "basic" or not encoded:
            return False
        try:
            decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
        except (UnicodeDecodeError, ValueError, binascii.Error):
            return False
        username, separator, password = decoded.partition(":")
        if not separator:
            return False
        return _dashboard_credentials_match(credentials, username, password)

    def _login_attempt_state(self) -> tuple[threading.Lock, dict[str, list[float]]]:
        lock = getattr(self.server, "login_attempt_lock", None)
        attempts = getattr(self.server, "login_attempts", None)
        if not isinstance(lock, _LOCK_TYPE):
            lock = threading.Lock()
            self.server.login_attempt_lock = lock  # type: ignore[attr-defined]
        if not isinstance(attempts, dict):
            attempts = {}
            self.server.login_attempts = attempts  # type: ignore[attr-defined]
        return lock, attempts

    def _login_is_rate_limited(self) -> bool:
        lock, attempts = self._login_attempt_state()
        client = str(self.client_address[0])
        cutoff = time.monotonic() - DASHBOARD_LOGIN_ATTEMPT_WINDOW_SECONDS
        with lock:
            recent = [stamp for stamp in attempts.get(client, []) if stamp >= cutoff]
            attempts[client] = recent
            return len(recent) >= DASHBOARD_LOGIN_ATTEMPT_LIMIT

    def _record_login_failure(self) -> None:
        lock, attempts = self._login_attempt_state()
        client = str(self.client_address[0])
        cutoff = time.monotonic() - DASHBOARD_LOGIN_ATTEMPT_WINDOW_SECONDS
        with lock:
            recent = [stamp for stamp in attempts.get(client, []) if stamp >= cutoff]
            recent.append(time.monotonic())
            attempts[client] = recent

    def _clear_login_failures(self) -> None:
        lock, attempts = self._login_attempt_state()
        with lock:
            attempts.pop(str(self.client_address[0]), None)

    def _deny_basic_auth(
        self,
        status: HTTPStatus = HTTPStatus.UNAUTHORIZED,
        message: str = "Authentication required.",
    ) -> None:
        body = message.encode("utf-8")
        self.send_response(status)
        if status == HTTPStatus.UNAUTHORIZED:
            self.send_header(
                "WWW-Authenticate",
                'Basic realm="Chronovisor Dashboard", charset="UTF-8"',
            )
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        _send_security_headers(self)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _deny_remote(self, status: HTTPStatus, message: str) -> None:
        body = (
            "<!doctype html><meta charset='utf-8'><title>Dashboard access</title>"
            f"<p>{message}</p>"
        ).encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        _send_security_headers(self)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorize(self, parsed: Any) -> bool:
        scope = _private_client_scope(self.client_address[0])
        token = self._access_token()
        lan_enabled = bool(getattr(self.server, "lan_access_enabled", False))
        supplied = dict(parse_qsl(parsed.query, keep_blank_values=True)).get(
            DASHBOARD_ACCESS_QUERY,
            "",
        )
        if scope == "loopback":
            if (
                lan_enabled
                and token
                and supplied
                and hmac.compare_digest(supplied, token)
            ):
                self._redirect_after_token(parsed, token)
                return False
            return True
        if not lan_enabled:
            self._deny_remote(HTTPStatus.FORBIDDEN, "LAN access is disabled.")
            return False
        if scope != "private":
            self._deny_remote(
                HTTPStatus.FORBIDDEN, "Only private-network clients are allowed."
            )
            return False
        if token and supplied and hmac.compare_digest(supplied, token):
            self._redirect_after_token(parsed, token)
            return False
        if token and self._cookie_authorized(token):
            return True
        credentials = self._credentials()
        if credentials is None:
            self._deny_basic_auth(
                HTTPStatus.SERVICE_UNAVAILABLE,
                "Dashboard credentials are not configured on the Mac.",
            )
            return False
        if self._login_is_rate_limited():
            self._deny_basic_auth(
                HTTPStatus.TOO_MANY_REQUESTS,
                "Too many attempts. Try again in five minutes.",
            )
            return False
        authorization = self.headers.get("Authorization", "")
        if not authorization.lower().startswith("basic "):
            self._deny_basic_auth()
            return False
        if not self._basic_authorized(credentials):
            self._record_login_failure()
            if self._login_is_rate_limited():
                self._deny_basic_auth(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    "Too many attempts. Try again in five minutes.",
                )
            else:
                self._deny_basic_auth()
            return False
        self._clear_login_failures()
        if token:
            # Safari does not reliably reuse HTTP Basic credentials for every
            # HTML, static asset, API, and WebSocket request. Promote the first
            # successful login to the same scoped HttpOnly session used by the
            # recovery link so navigation never triggers another challenge.
            self._redirect_after_token(parsed, token)
            return False
        return True

    def _lan_access_response(self) -> None:
        if not self._is_loopback():
            _json_response(self, {"enabled": False}, status=HTTPStatus.FORBIDDEN)
            return
        enabled = bool(getattr(self.server, "lan_access_enabled", False))
        token = self._access_token()
        port = int(self.server.server_address[1])
        urls = (
            [
                f"http://{host}:{port}/?{DASHBOARD_ACCESS_QUERY}={token}"
                for host in _dashboard_lan_hosts()
            ]
            if enabled and token
            else []
        )
        _json_response(
            self,
            {
                "enabled": enabled,
                "urls": urls,
                "trusted_lan_only": True,
            },
        )

    def _cortex_graph_response(self) -> None:
        identity = runtime_identity()
        commit = str(identity.get("commit_id") or identity.get("expected_commit") or "")
        _json_response(
            self,
            build_cortex_graph(CHRONOVISOR_ROOT, commit=commit),
        )

    def _cortex_field_response(self, query: str) -> None:
        params = dict(parse_qsl(query, keep_blank_values=False))
        _json_response(
            self,
            build_cortex_field_projection(
                CHRONOVISOR_ROOT,
                session_hash=str(params.get("session") or ""),
            ),
        )

    def _cortex_relations_response(self, query: str) -> None:
        params = dict(parse_qsl(query, keep_blank_values=False))
        raw_keys = str(params.get("keys") or "")
        relation_keys: list[tuple[str, str, str]] = []
        if len(raw_keys) <= 12_000:
            try:
                values = json.loads(raw_keys)
            except (json.JSONDecodeError, TypeError, ValueError):
                values = []
            if isinstance(values, list):
                for value in values[:24]:
                    if (
                        not isinstance(value, list)
                        or len(value) != 3
                        or not isinstance(value[0], str)
                        or not isinstance(value[1], str)
                        or not isinstance(value[2], str)
                        or not value[0]
                        or not value[1]
                        or not value[2]
                        or len(value[0]) > 256
                        or len(value[1]) > 240
                        or len(value[2]) > 240
                    ):
                        continue
                    relation_keys.append((value[0], value[1], value[2]))
        _json_response(
            self,
            {
                "relations": build_cortex_relation_details(
                    CHRONOVISOR_ROOT, relation_keys
                )
            },
        )

    def _cortex_events_response(self, query: str) -> None:
        upgrade = self.headers.get("Upgrade", "").casefold()
        connection = {
            item.strip().casefold()
            for item in self.headers.get("Connection", "").split(",")
        }
        key = self.headers.get("Sec-WebSocket-Key", "")
        if upgrade != "websocket" or "upgrade" not in connection or not key:
            self.send_error(HTTPStatus.BAD_REQUEST, "WebSocket upgrade required")
            return
        try:
            accept = websocket_accept(key)
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid WebSocket key")
            return

        self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        self.close_connection = True

        params = dict(parse_qsl(query, keep_blank_values=False))
        cursor = CortexEventCursor(
            CHRONOVISOR_ROOT,
            recall_log=recall_runtime.RECALL_LOG_FILE,
            pull_log=recall_runtime.RECALL_PULL_LOG_FILE,
            activity_log=LOG_FILE,
            field_session=str(params.get("session") or ""),
            follow_field_sessions=params.get("follow") == "latest",
            after_seq=max(0, int(params.get("after_seq") or 0))
            if str(params.get("after_seq") or "").isdigit()
            else 0,
        )
        last_heartbeat = 0.0
        try:
            while True:
                payload = cursor.poll_payload()
                events = (
                    payload.get("events") if payload.get("type") == "events" else []
                )
                now = time.monotonic()
                if events or payload.get("type") in {"resync", "session_changed"}:
                    pass
                elif now - last_heartbeat >= 15:
                    payload = {"type": "heartbeat"}
                else:
                    time.sleep(0.25)
                    continue
                self.wfile.write(websocket_text_frame(payload))
                self.wfile.flush()
                last_heartbeat = now
                if not events:
                    time.sleep(0.25)
        except (BrokenPipeError, ConnectionError, OSError):
            return

    def _processing_activity_stream_response(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        _send_security_headers(self)
        self.end_headers()
        self.close_connection = True

        last_revision = ""
        last_heartbeat = 0.0
        try:
            self.wfile.write(b"retry: 1000\n\n")
            self.wfile.flush()
            while True:
                snapshot = _processing_activity_snapshot()
                revision = str(snapshot.get("revision") or "")
                now = time.monotonic()
                if revision != last_revision:
                    encoded = json.dumps(
                        snapshot,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    self.wfile.write(f"id: {revision}\n".encode("ascii"))
                    self.wfile.write(b"event: activity\n")
                    self.wfile.write(b"data: " + encoded + b"\n\n")
                    self.wfile.flush()
                    last_revision = revision
                    last_heartbeat = now
                elif now - last_heartbeat >= PROCESSING_ACTIVITY_HEARTBEAT_SECONDS:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    last_heartbeat = now
                time.sleep(PROCESSING_ACTIVITY_POLL_SECONDS)
        except (BrokenPipeError, ConnectionError, OSError):
            return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if not self._authorize(parsed):
            return
        try:
            if path == "/":
                _file_response(self, STATIC_DIR / "index.html")
            elif path == "/cortex":
                _file_response(self, STATIC_DIR / "cortex.html")
            elif path == "/api/lan-access":
                self._lan_access_response()
            elif path == "/api/fast-snapshot":
                _json_response(self, build_fast_snapshot())
            elif path == "/api/snapshot":
                _json_response(
                    self,
                    _snapshot_with_live_status(
                        _cached_snapshot(allow_stale=True)
                    ),
                )
            elif path == "/api/status":
                snapshot = _snapshot_with_live_status(
                    _cached_snapshot(allow_stale=True)
                )
                _json_response(
                    self,
                    {
                        "status": snapshot["status"],
                        "_dashboard": snapshot.get("_dashboard") or {},
                    },
                )
            elif path == "/api/local-consensus":
                _json_response(
                    self,
                    {
                        # Decision Trace is latency-sensitive and cheap to build.
                        # Never make it wait for the full dashboard materialization.
                        "local_consensus": _local_consensus_snapshot()
                    },
                )
            elif path == "/api/activity":
                _json_response(self, _processing_activity_snapshot())
            elif path == "/api/activity-stream":
                self._processing_activity_stream_response()
            elif path == "/api/frontier-repair":
                _json_response(
                    self,
                    {
                        "frontier_repair": _cached_snapshot(allow_stale=True)[
                            "frontier_repair"
                        ]
                    },
                )
            elif path == "/api/events":
                _json_response(
                    self,
                    {"events": _cached_snapshot(allow_stale=True)["events"]},
                )
            elif path == "/api/metrics":
                _json_response(
                    self,
                    {"metrics": _cached_snapshot(allow_stale=True)["metrics"]},
                )
            elif path == "/api/self-heal":
                _json_response(
                    self,
                    {"self_heal": _cached_snapshot(allow_stale=True)["self_heal"]},
                )
            elif path == "/api/recall":
                _json_response(
                    self,
                    {"recall": _cached_snapshot(allow_stale=True)["recall"]},
                )
            elif path == "/api/recall-improvement":
                _json_response(
                    self,
                    {
                        "recall_improvement": _cached_snapshot(allow_stale=True)[
                            "recall_improvement"
                        ]
                    },
                )
            elif path == "/api/model-lab":
                _json_response(
                    self,
                    {"model_lab": _cached_snapshot(allow_stale=True)["model_lab"]},
                )
            elif path == "/api/typed-graph":
                _json_response(
                    self,
                    {"typed_graph": _typed_graph_dashboard_snapshot()},
                )
            elif path == "/api/save-history":
                _json_response(
                    self,
                    {
                        "save_history": _cached_snapshot(allow_stale=True)[
                            "save_history"
                        ]
                    },
                )
            elif path == "/api/knowledge-mix":
                _json_response(
                    self,
                    {
                        "knowledge_mix": _cached_snapshot(allow_stale=True)[
                            "knowledge_mix"
                        ]
                    },
                )
            elif path == "/api/health":
                _json_response(
                    self,
                    {"health": _cached_snapshot(allow_stale=True)["health"]},
                )
            elif path == "/api/cortex/graph":
                self._cortex_graph_response()
            elif path == "/api/cortex/field":
                self._cortex_field_response(parsed.query)
            elif path == "/api/cortex/relations":
                self._cortex_relations_response(parsed.query)
            elif path == "/api/cortex/events":
                self._cortex_events_response(parsed.query)
            elif path == "/api/model-status":
                snapshot = _cached_snapshot(allow_stale=True)
                _json_response(
                    self,
                    {
                        "model_status": snapshot["model_status"],
                        "ollama": snapshot["ollama"],
                    },
                )
            elif path.startswith("/static/"):
                target = _resolve_static_path(STATIC_DIR, path)
                if target is None:
                    self.send_error(HTTPStatus.FORBIDDEN)
                    return
                _file_response(self, target)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            if path.startswith("/api/"):
                _json_response(
                    self,
                    {
                        "status": "error",
                        "error_class": exc.__class__.__name__,
                        "error": str(exc),
                    },
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            raise

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def serve(
    host: str,
    port: int,
    *,
    lan: bool = False,
    access_token_file: Path | None = None,
    credentials_file: Path | None = None,
) -> None:
    init_chronovisor()
    token_path = (
        access_token_file or CHRONOVISOR_ROOT / "runtime" / "dashboard-access-token"
    )
    credentials_path = (
        credentials_file or CHRONOVISOR_ROOT / "runtime" / "dashboard-credentials.json"
    )
    token = _load_or_create_dashboard_token(token_path) if lan else ""
    credentials = _load_dashboard_credentials(credentials_path) if lan else None
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    server.lan_access_enabled = lan  # type: ignore[attr-defined]
    server.lan_access_token = token  # type: ignore[attr-defined]
    server.dashboard_credentials = credentials  # type: ignore[attr-defined]
    server.login_attempt_lock = threading.Lock()  # type: ignore[attr-defined]
    server.login_attempts = {}  # type: ignore[attr-defined]
    print(f"Chronovisor dashboard: http://{host}:{port}")
    if lan:
        print(f"LAN access enabled with token file: {token_path}")
        print(f"LAN password credentials file: {credentials_path}")
    server.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Chronovisor local dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--lan",
        action="store_true",
        help="Bind for trusted private-network access and require authentication.",
    )
    parser.add_argument("--access-token-file", type=Path)
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument(
        "--set-credentials",
        action="store_true",
        help="Prompt for and store hashed dashboard credentials, then exit.",
    )
    parser.add_argument("--username", default="admin")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-dashboard`` command-line entry point."""
    args = build_parser().parse_args(argv)
    token_path = (
        args.access_token_file
        or CHRONOVISOR_ROOT / "runtime" / "dashboard-access-token"
    )
    credentials_path = (
        args.credentials_file
        or CHRONOVISOR_ROOT / "runtime" / "dashboard-credentials.json"
    )
    if args.set_credentials:
        password = getpass.getpass("Dashboard password: ")
        confirmation = getpass.getpass("Confirm dashboard password: ")
        if password != confirmation:
            raise SystemExit("dashboard passwords do not match")
        _write_dashboard_credentials(credentials_path, args.username, password)
        _rotate_dashboard_token(token_path)
        print(f"Dashboard credentials stored: {credentials_path}")
        print("Existing dashboard sessions and recovery links were revoked.")
        return 0
    host = "0.0.0.0" if args.lan and args.host == "127.0.0.1" else args.host
    serve(
        host,
        args.port,
        lan=args.lan,
        access_token_file=args.access_token_file,
        credentials_file=credentials_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
