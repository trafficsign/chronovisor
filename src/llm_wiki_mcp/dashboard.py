"""Local web dashboard for LLM Wiki ingest observability."""

from __future__ import annotations

import argparse
import json
import re
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
            },
            "human_notification": record.get("human_notification") or (packet or {}).get("human_notification"),
        },
    }


def _packet_summary(packet: dict[str, Any]) -> dict[str, Any]:
    status = str(packet.get("status") or "unknown")
    pending_statuses = {
        "pending_local_repair",
        "local_repairing",
        "pending_frontier",
        "frontier_running",
        "frontier_retry",
        "frontier_preflight_failed",
        "pending_frontier_review",
    }
    failed_statuses = {
        "local_repair_failed",
        "frontier_rejected",
        "frontier_quarantined",
        "human_required",
    }
    state = "pending" if status in pending_statuses else "failed" if status in failed_statuses else "resolved"
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
        "paths": {
            "failures_dir": str(failures_dir),
            "registry_file": str(failures_dir / "failure-registry.jsonl"),
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
