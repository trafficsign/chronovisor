"""Local web dashboard for LLM Wiki ingest observability."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import threading
import time
from datetime import date, datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from llm_wiki_mcp import orchestrator, recall_runtime, runtime_status
from llm_wiki_mcp.convergence import is_human_required_result
from llm_wiki_mcp.decision_router import resolve_router_policy
from llm_wiki_mcp.health import health_snapshot
from llm_wiki_mcp.ollama import OLLAMA_URL, embedding_model, ingest_model
from llm_wiki_mcp.recall_auditor import load_audit_policy
from llm_wiki_mcp.recall_improvement import configured_models
from llm_wiki_mcp.runtime_config import (
    load_decision_router_config,
    load_reranker_config,
)
from llm_wiki_mcp.wiki import LOG_FILE, WIKI_ROOT, init_wiki

STATIC_DIR = Path(__file__).with_name("dashboard_static")
LOG_LINE_RE = re.compile(r"^- \[(?P<time>[^\]]+)\] (?P<message>.*)$")
RAW_DATE_RE = re.compile(r"(?:^|[^0-9])(?P<stamp>20\d{6})(?:[^0-9]|$)")
SEMANTIC_PROJECTION_CHILD_RE = re.compile(
    r"^semantic-[0-9a-f]{64}-child-[0-9]{8}-[0-9a-f]{64}\.md$"
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
_DECISION_ROUTER_CACHE_LOCK = threading.Lock()
_DECISION_ROUTER_CACHE: dict[str, Any] = {
    "key": None,
    "expires_at": 0.0,
    "config": None,
}


def _json_response(
    handler: BaseHTTPRequestHandler, data: Any, status: int = 200
) -> None:
    body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _file_response(handler: BaseHTTPRequestHandler, path: Path) -> None:
    if not path.exists() or not path.is_file():
        handler.send_error(HTTPStatus.NOT_FOUND)
        return
    content_type = {
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".svg": "image/svg+xml",
    }.get(path.suffix, "application/octet-stream")
    body = path.read_bytes()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


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

    try:
        _add_model_role(roles, embedding_model(), "embed")
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
        "rerank": 9,
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
    if roles == {"rerank"} and "/" in name and not name.startswith("hf.co/"):
        return True
    return False


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


def _drain_history(limit: int = 200) -> list[dict[str, Any]]:
    logs_dir = WIKI_ROOT / "logs"
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
            records.append(
                {
                    "timestamp": data.get("timestamp"),
                    "kind": "drain_batch",
                    "pending_before": data.get("pending_before"),
                    "pending_after": data.get("pending_after"),
                    "files_processed": data.get("files_processed"),
                    "batch": data.get("batch"),
                }
            )
    return records[-limit:]


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
    pages_dir = WIKI_ROOT / "pages"
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
        "failed_bytes": 0,
        "raw_segments": [],
        "processed": 0,
        "attempted": 0,
        "succeeded": 0,
        "failed": 0,
        "pages_created": 0,
        "pages_updated": 0,
        "sources": {},
        "raw_samples": [],
        "page_samples": [],
    }


def _add_sample(row: dict[str, Any], key: str, value: str, limit: int = 6) -> None:
    samples = row.setdefault(key, [])
    if isinstance(samples, list) and value not in samples and len(samples) < limit:
        samples.append(value)


def _save_history_snapshot(
    days: int = 371, today: date | None = None
) -> dict[str, Any]:
    end = today or datetime.now().date()
    start = end - timedelta(days=max(1, days) - 1)
    rows = {
        (start + timedelta(days=offset)).isoformat(): _new_save_day(
            start + timedelta(days=offset)
        )
        for offset in range((end - start).days + 1)
    }

    raw_dir = WIKI_ROOT / "raw"
    raw_files: dict[str, dict[str, Any]] = {}
    raw_status: dict[str, str] = {}
    if raw_dir.exists():
        for path in raw_dir.glob("*.md"):
            # Projection children are generated processing artifacts.  The
            # original lossless parent is already counted as the save, so
            # including children would double-count bytes and invent a
            # "manual" user save on the projection date.  Queue cardinality
            # remains visible through the canonical pending counter.
            if SEMANTIC_PROJECTION_CHILD_RE.fullmatch(path.name.lower()):
                continue
            raw_date = _raw_file_date(path)
            if raw_date is None or raw_date < start or raw_date > end:
                continue
            raw_bytes = path.stat().st_size
            source = _raw_source_label(path.name)
            raw_files[path.name] = {
                "date": raw_date.isoformat(),
                "bytes": raw_bytes,
                "source": source,
            }
            row = rows[raw_date.isoformat()]
            row["raw_saved"] += 1
            row["raw_bytes"] += raw_bytes
            row["sources"][source] = row["sources"].get(source, 0) + 1
            _add_sample(row, "raw_samples", path.name)

    logs_dir = WIKI_ROOT / "logs"
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
                    failed = sum(
                        1
                        for item in per_raw
                        if isinstance(item, dict) and item.get("succeeded") is False
                    )
                else:
                    attempted = len(attempted_files) or processed
                    succeeded = processed
                    failed = max(0, attempted - succeeded)

                row["processed"] += processed
                row["attempted"] += attempted
                row["succeeded"] += succeeded
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
                        if item.get("succeeded") is True:
                            for status_filename in status_filenames:
                                raw_status[status_filename] = "processed"
                        else:
                            for status_filename in status_filenames:
                                if raw_status.get(status_filename) != "processed":
                                    raw_status[status_filename] = "failed"
                else:
                    processed_names = {
                        name for name in processed_files if isinstance(name, str)
                    }
                    for filename in attempted_files:
                        if (
                            isinstance(filename, str)
                            and filename not in processed_names
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
    orchestrator_state = _read_json_file(WIKI_ROOT / ".orchestrator_state.json") or {}
    processed_raw_files = orchestrator_state.get("processed_raw_files")
    if isinstance(processed_raw_files, list):
        for filename in processed_raw_files:
            if isinstance(filename, str) and filename in raw_files:
                raw_status[filename] = "processed"

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
        "failed_bytes": 0,
        "processed": 0,
        "attempted": 0,
        "succeeded": 0,
        "failed": 0,
        "pages_created": 0,
        "pages_updated": 0,
        "days_with_saves": 0,
    }
    source_totals: dict[str, int] = {}
    for filename, meta in raw_files.items():
        row = rows.get(str(meta["date"]))
        if not row:
            continue
        raw_bytes = int(meta["bytes"])
        status = raw_status.get(filename)
        if status == "processed":
            row["processed_bytes"] += raw_bytes
        elif status == "failed":
            row["failed_bytes"] += raw_bytes
        else:
            row["pending_bytes"] += raw_bytes
            status = "pending"
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
            raw_segments.sort(key=lambda item: str(item.get("name") or ""))
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
            "failed_bytes",
            "processed",
            "attempted",
            "succeeded",
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
    packets_dir = WIKI_ROOT / "runtime" / "failures" / "packets"
    packets: dict[str, dict[str, Any]] = {}
    if not packets_dir.exists():
        return packets
    for path in sorted(packets_dir.glob("*.json")):
        packet = _read_json_file(path)
        if not packet:
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
        from llm_wiki_mcp.frontier_guard import FrontierGuard

        inspection = FrontierGuard(WIKI_ROOT / "runtime" / "frontier-repair").inspect(
            dry_run=True
        )
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
    active_dir = WIKI_ROOT / "runtime" / "frontier-reviews" / "active"
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
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
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
    }
    return {
        "schema_version": 2,
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


def _local_consensus_snapshot(limit: int = 40) -> dict[str, Any]:
    """Return live local review truth plus a redacted bounded audit tail."""

    root = WIKI_ROOT / "runtime" / "local-consensus"
    active_dir = root / "active"
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
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
                continue
            safe = {
                "request_sha256": row.get("request_sha256"),
                "role": row.get("role"),
                "model": row.get("model"),
                "started_at": row.get("started_at"),
                "pid": pid,
                "elapsed_seconds": age_seconds,
            }
            activities.append(safe)
    activities.sort(key=lambda row: str(row.get("started_at") or ""))

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
                    )
                }
            )
    latest_decision = next(
        (row for row in reversed(history) if row.get("kind") == "decision"),
        None,
    )
    return {
        "active": bool(activities),
        "count": len(activities),
        "activities": activities,
        "latest": activities[-1] if activities else None,
        "summary": summary,
        "latest_decision": latest_decision,
        "history": history,
    }


def _frontier_repair_snapshot(limit: int = 40) -> dict[str, Any]:
    """Expose the exceptional repair ledger without leaking incident payloads."""

    root = WIKI_ROOT / "runtime" / "frontier-repair"
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
                "owner_pid": raw.get("owner_pid"),
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
    if active_row is not None:
        active_row = {
            **active_row,
            "owner_alive": owner_alive,
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
        "active": owner_alive,
        "active_incident": active_row,
        "stale_active_incident": bool(active_row and not owner_alive),
        "summary": {
            "total": len(incidents),
            "starts_24h": starts_24h,
            "counts": counts,
        },
        "recent": recent,
        "events": events,
    }


def _last_self_heal_check(limit: int = 400) -> dict[str, Any] | None:
    logs_dir = WIKI_ROOT / "logs"
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
        title = "Frontier approved"
    elif human_required:
        state = "failed"
        level = "error"
        title = "Human required"
    elif rescue_status == "pending_frontier_review":
        state = "pending"
        level = "warn"
        title = "Frontier review pending"
    elif rescue_status == "frontier_preflight_failed":
        state = "pending"
        level = "warn"
        title = "Frontier preflight failed"
    else:
        level = "warn" if decision in {"needs_retry", "quarantined"} else "error"
        state = "pending" if decision == "needs_retry" else "failed"
        title = f"Frontier {decision}"
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
        "pending_frontier": "Frontier queued",
        "frontier_running": "Frontier running",
        "frontier_retry": "Frontier retry needed",
        "frontier_preflight_failed": "Frontier preflight failed",
        "pending_frontier_review": "Frontier review pending",
        "repair_deferred": "Frontier repair deferred",
        "human_required": "Human required",
        "local_repair_failed": "Local repair failed",
        "frontier_rejected": "Frontier rejected",
        "frontier_quarantined": "Frontier quarantined",
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
    failures_dir = WIKI_ROOT / "runtime" / "failures"
    packets = _self_heal_packet_index()
    registry = _read_jsonl_file(failures_dir / "failure-registry.jsonl", limit=200)
    seen_failure_ids: set[str] = set()
    history: list[dict[str, Any]] = []

    for record in registry:
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
            1 for item in history if item.get("title") == "Frontier review pending"
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
    eval_dir = WIKI_ROOT / "runtime" / "eval"
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
    recall_dir = WIKI_ROOT / "recall"
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
        "paths": {
            "recall_log": str(recall_dir / "recall-log.jsonl"),
            "pull_log": str(recall_dir / "pull-log.jsonl"),
            "calibration_file": str(recall_dir / "calibration.json"),
        },
    }


def _recall_improvement_snapshot() -> dict[str, Any]:
    try:
        from llm_wiki_mcp.recall_improvement import improvement_snapshot

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
        from llm_wiki_mcp.model_lab import snapshot

        return snapshot()
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
        from llm_wiki_mcp.orchestrator import ingest_process_lease_is_held

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


def build_snapshot() -> dict[str, Any]:
    init_wiki()
    cached_status = runtime_status.read_status()
    orch_state = orchestrator._load_state()
    pending = len(orchestrator.get_pending_raw_files())
    status = _canonicalize_runtime_status(cached_status, orch_state, pending=pending)

    local_consensus = _safe_snapshot_component(
        "local_consensus",
        _local_consensus_snapshot,
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
    metrics = sorted(
        runtime_metrics + drain_metrics,
        key=lambda row: str(row.get("timestamp") or ""),
    )[-240:]
    metrics.append(
        {
            "timestamp": runtime_status.now_iso(),
            "kind": "current",
            "pending_after": pending,
            "files_processed": 0,
            "files_failed": 0,
        }
    )
    events = (runtime_status.read_events(limit=120) + _recent_log_events(limit=80))[
        -160:
    ]
    ollama = _ollama_snapshot()
    model_status = _safe_snapshot_component(
        "model_status",
        lambda: _model_status_snapshot(ollama),
        {"available": False, "models": [], "summary": {}},
    )
    from llm_wiki_mcp.runtime_config import runtime_identity
    from llm_wiki_mcp.decision_policy import decision_policy_snapshot

    decision_policies = _safe_snapshot_component(
        "decision_policies",
        decision_policy_snapshot,
        {"lanes": {}, "counts": {"off": 0, "shadow": 0, "enabled": 0}},
    )
    status["decision_policies"] = decision_policies

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
        "ollama": ollama,
        "model_status": model_status,
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
        "save_history": _safe_snapshot_component(
            "save_history",
            _save_history_snapshot,
            {"days": [], "recent": [], "totals": {}, "sources": []},
        ),
        "knowledge_mix": _safe_snapshot_component(
            "knowledge_mix",
            _knowledge_mix_snapshot,
            {"total_pages": 0, "total_bytes": 0, "categories": [], "top": []},
        ),
        "health": _safe_snapshot_component("health", health_snapshot),
        "paths": {
            "wiki_root": str(WIKI_ROOT),
            "status_file": str(runtime_status.STATUS_FILE),
            "events_file": str(runtime_status.EVENTS_FILE),
            "metrics_file": str(runtime_status.METRICS_FILE),
            "local_consensus": str(WIKI_ROOT / "runtime" / "local-consensus"),
            "frontier_repair": str(WIKI_ROOT / "runtime" / "frontier-repair"),
        },
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "LLMWikiDashboard/0.1"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/":
                _file_response(self, STATIC_DIR / "index.html")
            elif path == "/api/snapshot":
                _json_response(self, build_snapshot())
            elif path == "/api/status":
                _json_response(self, {"status": build_snapshot()["status"]})
            elif path == "/api/local-consensus":
                _json_response(
                    self,
                    {"local_consensus": build_snapshot()["local_consensus"]},
                )
            elif path == "/api/frontier-repair":
                _json_response(
                    self,
                    {"frontier_repair": build_snapshot()["frontier_repair"]},
                )
            elif path == "/api/events":
                _json_response(self, {"events": build_snapshot()["events"]})
            elif path == "/api/metrics":
                _json_response(self, {"metrics": build_snapshot()["metrics"]})
            elif path == "/api/self-heal":
                _json_response(self, {"self_heal": build_snapshot()["self_heal"]})
            elif path == "/api/recall":
                _json_response(self, {"recall": build_snapshot()["recall"]})
            elif path == "/api/recall-improvement":
                _json_response(
                    self, {"recall_improvement": build_snapshot()["recall_improvement"]}
                )
            elif path == "/api/model-lab":
                _json_response(self, {"model_lab": build_snapshot()["model_lab"]})
            elif path == "/api/save-history":
                _json_response(self, {"save_history": build_snapshot()["save_history"]})
            elif path == "/api/knowledge-mix":
                _json_response(
                    self, {"knowledge_mix": build_snapshot()["knowledge_mix"]}
                )
            elif path == "/api/health":
                _json_response(self, {"health": build_snapshot()["health"]})
            elif path == "/api/model-status":
                snapshot = build_snapshot()
                _json_response(
                    self,
                    {
                        "model_status": snapshot["model_status"],
                        "ollama": snapshot["ollama"],
                    },
                )
            elif path.startswith("/static/"):
                rel = path.removeprefix("/static/").lstrip("/")
                target = (STATIC_DIR / rel).resolve()
                try:
                    target.relative_to(STATIC_DIR.resolve())
                except ValueError:
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


def serve(host: str, port: int) -> None:
    init_wiki()
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"LLM Wiki dashboard: http://{host}:{port}")
    server.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LLM Wiki local dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
