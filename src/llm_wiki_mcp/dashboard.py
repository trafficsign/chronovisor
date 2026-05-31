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
