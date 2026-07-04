"""Local web dashboard for LLM Wiki ingest observability."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import date, datetime, timedelta
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from llm_wiki_mcp import orchestrator, runtime_status
from llm_wiki_mcp.ollama import OLLAMA_URL
from llm_wiki_mcp.wiki import LOG_FILE, WIKI_ROOT, init_wiki

STATIC_DIR = Path(__file__).with_name("dashboard_static")
LOG_LINE_RE = re.compile(r"^- \[(?P<time>[^\]]+)\] (?P<message>.*)$")
RAW_DATE_RE = re.compile(r"(?:^|[^0-9])(?P<stamp>20\d{6})(?:[^0-9]|$)")
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
}
SELF_HEAL_FAILED_STATUSES = {
    "local_repair_failed",
    "frontier_rejected",
    "frontier_quarantined",
    "human_required",
}
FRONTIER_PREFLIGHT_TTL_SECONDS = 300
_FRONTIER_PREFLIGHT_CACHE: dict[str, Any] | None = None
_FRONTIER_PREFLIGHT_CACHE_AT = 0.0


def _json_response(handler: BaseHTTPRequestHandler, data: Any, status: int = 200) -> None:
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
    if "claude-code" in lower:
        return "claude-code"
    if "codex" in lower:
        return "codex"
    if lower.startswith("vestige-"):
        return "vestige"
    if "ingest" in lower:
        return "ingest"
    return "manual"


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


def _save_history_snapshot(days: int = 371, today: date | None = None) -> dict[str, Any]:
    end = today or datetime.now().date()
    start = end - timedelta(days=max(1, days) - 1)
    rows = {
        (start + timedelta(days=offset)).isoformat(): _new_save_day(start + timedelta(days=offset))
        for offset in range((end - start).days + 1)
    }

    raw_dir = WIKI_ROOT / "raw"
    raw_files: dict[str, dict[str, Any]] = {}
    raw_status: dict[str, str] = {}
    if raw_dir.exists():
        for path in raw_dir.glob("*.md"):
            raw_date = _raw_file_date(path)
            if raw_date is None or raw_date < start or raw_date > end:
                continue
            raw_bytes = path.stat().st_size
            source = _raw_source_label(path.name)
            raw_files[path.name] = {"date": raw_date.isoformat(), "bytes": raw_bytes, "source": source}
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
                result = record.get("result") if isinstance(record.get("result"), dict) else {}
                attempted_files = result.get("files_attempted") if isinstance(result.get("files_attempted"), list) else []
                processed_files = result.get("files_processed") if isinstance(result.get("files_processed"), list) else []
                per_raw = result.get("per_raw") if isinstance(result.get("per_raw"), list) else []

                processed = _int_value(record.get("files_processed")) or len(processed_files)
                if per_raw:
                    succeeded = sum(
                        1 for item in per_raw if isinstance(item, dict) and item.get("succeeded") is True
                    )
                    attempted = len(per_raw)
                    failed = sum(
                        1 for item in per_raw if isinstance(item, dict) and item.get("succeeded") is False
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
                        if not isinstance(filename, str):
                            continue
                        if item.get("succeeded") is True:
                            raw_status[filename] = "processed"
                        elif raw_status.get(filename) != "processed":
                            raw_status[filename] = "failed"
                else:
                    processed_names = {name for name in processed_files if isinstance(name, str)}
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
            segments.append({
                "name": filename,
                "bytes": raw_bytes,
                "status": status,
                "source": meta.get("source") or _raw_source_label(filename),
            })
    for row in rows.values():
        sources = row.get("sources") if isinstance(row.get("sources"), dict) else {}
        for source, count in sources.items():
            source_totals[source] = source_totals.get(source, 0) + int(count)
        row["sources"] = [
            {"name": source, "count": count}
            for source, count in sorted(sources.items(), key=lambda item: (-item[1], item[0]))
        ]
        raw_segments = row.get("raw_segments")
        if isinstance(raw_segments, list):
            raw_segments.sort(key=lambda item: str(item.get("name") or ""))
        if row["raw_saved"] or row["processed"] or row["pages_created"] or row["pages_updated"]:
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
        row for row in days_list
        if row["raw_saved"] or row["processed"] or row["pages_created"] or row["pages_updated"]
    ][-14:]
    return {
        "range": {"start": start.isoformat(), "end": end.isoformat(), "days": len(days_list)},
        "days": days_list,
        "recent": recent,
        "totals": totals,
        "sources": [
            {"name": source, "count": count}
            for source, count in sorted(source_totals.items(), key=lambda item: (-item[1], item[0]))
        ],
        "paths": {
            "raw_dir": str(raw_dir),
            "drain_logs": str(logs_dir),
            "log_file": str(LOG_FILE),
        },
    }


def _failure_detail(packet: dict[str, Any] | None, record: dict[str, Any] | None = None) -> dict[str, Any]:
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
    global _FRONTIER_PREFLIGHT_CACHE, _FRONTIER_PREFLIGHT_CACHE_AT

    now = time.time()
    if (
        _FRONTIER_PREFLIGHT_CACHE is not None
        and now - _FRONTIER_PREFLIGHT_CACHE_AT < FRONTIER_PREFLIGHT_TTL_SECONDS
    ):
        return {**_FRONTIER_PREFLIGHT_CACHE, "cached": True}

    checked_at = datetime.now().isoformat(timespec="seconds")
    try:
        from llm_wiki_mcp.frontier_review import run_frontier_preflight

        result = run_frontier_preflight()
        codex = result.get("codex") if isinstance(result.get("codex"), dict) else {}
        failure = result.get("failure") if isinstance(result.get("failure"), dict) else None
        summary = {
            "ok": bool(result.get("ok")),
            "checked_at": checked_at,
            "cached": False,
            "codex_home": result.get("codex_home"),
            "codex_version_ok": bool((codex.get("version") or {}).get("ok")),
            "exec_help_ok": bool((codex.get("exec_help") or {}).get("ok")),
            "missing_exec_options": codex.get("missing_exec_options") or [],
            "adaptive_required": bool(codex.get("adaptive_required")),
            "failure": failure,
        }
    except Exception as exc:
        summary = {
            "ok": False,
            "checked_at": checked_at,
            "cached": False,
            "error": str(exc),
        }

    _FRONTIER_PREFLIGHT_CACHE = summary
    _FRONTIER_PREFLIGHT_CACHE_AT = now
    return summary


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


def _local_repair_summary(record: dict[str, Any], packet: dict[str, Any] | None) -> dict[str, Any]:
    decision = record.get("decision") if isinstance(record.get("decision"), dict) else {}
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
        "timestamp": record.get("timestamp") or (packet or {}).get("updated_at") or (packet or {}).get("created_at"),
        "failure_id": record.get("failure_id"),
        "state": "resolved",
        "level": "success",
        "title": "Local repair applied",
        "detail": _shorten("; ".join(details) or decision.get("reason") or "local repair applied"),
        "raw_file": raw_file,
        "failure_class": record.get("failure_class") or (packet or {}).get("failure_class"),
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


def _frontier_summary(record: dict[str, Any], packet: dict[str, Any] | None) -> dict[str, Any]:
    frontier = record.get("frontier") if isinstance(record.get("frontier"), dict) else {}
    local_decision = record.get("decision") if isinstance(record.get("decision"), dict) else {}
    decision = frontier.get("decision") or "unknown"
    rescue_status = frontier.get("rescue_status")
    human_required = bool(frontier.get("human_required"))
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
        "timestamp": record.get("timestamp") or (packet or {}).get("updated_at") or (packet or {}).get("created_at"),
        "failure_id": record.get("failure_id"),
        "state": state,
        "level": level,
        "title": title,
        "detail": _shorten(detail),
        "raw_file": raw_file,
        "failure_class": record.get("failure_class") or (packet or {}).get("failure_class"),
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
            "human_notification": record.get("human_notification") or (packet or {}).get("human_notification"),
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
    level = "info" if state == "pending" else "error" if state == "failed" else "success"
    title = {
        "pending_local_repair": "Repair queued",
        "local_repairing": "Local repair running",
        "pending_frontier": "Frontier queued",
        "frontier_running": "Frontier running",
        "frontier_retry": "Frontier retry needed",
        "frontier_preflight_failed": "Frontier preflight failed",
        "pending_frontier_review": "Frontier review pending",
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
        "human_required": sum(1 for item in history if item.get("title") == "Human required"),
        "pending_frontier_review": sum(1 for item in history if item.get("title") == "Frontier review pending"),
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
    calibration_history = _read_jsonl_file(recall_dir / "calibration-history.jsonl", limit=8)

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


def build_snapshot() -> dict[str, Any]:
    init_wiki()
    status = runtime_status.read_status()
    orch_state = orchestrator._load_state()
    pending = len(orchestrator.get_pending_raw_files())
    status["pending"] = pending
    status["current_job_id"] = status.get("current_job_id") or orch_state.get("current_job_id")
    status["current_job_pid"] = status.get("current_job_pid") or orch_state.get("current_job_pid")
    if not status.get("stage") and status.get("current_job_id"):
        status["stage"] = "running"
    if status.get("state") in (None, "unknown") and orch_state.get("current_job_id"):
        status["state"] = "running"
    elif status.get("state") in (None, "unknown"):
        status["state"] = "idle"
    if not status.get("updated_at"):
        status["updated_at"] = orch_state.get("current_job_started_at") or orch_state.get("last_ingest")

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
    events = (runtime_status.read_events(limit=120) + _recent_log_events(limit=80))[-160:]
    ollama = _ollama_snapshot()
    return {
        "status": status,
        "orchestrator": {
            "last_ingest": orch_state.get("last_ingest"),
            "last_lint": orch_state.get("last_lint"),
            "triage_failure_count": orch_state.get("triage_failure_count", 0),
        },
        "ollama": ollama,
        "events": events,
        "metrics": metrics,
        "self_heal": _self_heal_snapshot(),
        "recall": _recall_snapshot(),
        "save_history": _save_history_snapshot(),
        "paths": {
            "wiki_root": str(WIKI_ROOT),
            "status_file": str(runtime_status.STATUS_FILE),
            "events_file": str(runtime_status.EVENTS_FILE),
            "metrics_file": str(runtime_status.METRICS_FILE),
        },
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "LLMWikiDashboard/0.1"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            _file_response(self, STATIC_DIR / "index.html")
        elif path == "/api/snapshot":
            _json_response(self, build_snapshot())
        elif path == "/api/status":
            _json_response(self, {"status": build_snapshot()["status"]})
        elif path == "/api/events":
            _json_response(self, {"events": build_snapshot()["events"]})
        elif path == "/api/metrics":
            _json_response(self, {"metrics": build_snapshot()["metrics"]})
        elif path == "/api/self-heal":
            _json_response(self, {"self_heal": build_snapshot()["self_heal"]})
        elif path == "/api/recall":
            _json_response(self, {"recall": build_snapshot()["recall"]})
        elif path == "/api/save-history":
            _json_response(self, {"save_history": build_snapshot()["save_history"]})
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
