"""Runtime status files for local observability."""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.core.store import CHRONOVISOR_ROOT

RUNTIME_DIR = CHRONOVISOR_ROOT / "runtime"
STATUS_FILE = RUNTIME_DIR / "status.json"
EVENTS_FILE = RUNTIME_DIR / "events.jsonl"
METRICS_FILE = RUNTIME_DIR / "metrics.jsonl"
MAX_EVENTS = 2000
MAX_METRICS = 5000
STALE_RUNTIME_MAX_AGE_SECONDS = 30 * 60


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _parse_iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _pid_is_alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _age_seconds(value: object) -> float | None:
    timestamp = _parse_iso_datetime(value)
    if timestamp is None:
        return None
    return (datetime.now() - timestamp).total_seconds()


def _ensure_runtime_dir() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        if not path.exists():
            return dict(default)
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else dict(default)
    except Exception:
        return dict(default)


def _atomic_write_text(path: Path, text: str) -> None:
    _ensure_runtime_dir()
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, record: dict[str, Any], *, max_lines: int) -> None:
    _ensure_runtime_dir()
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except Exception:
        return
    if len(lines) > max_lines:
        _atomic_write_text(path, "\n".join(lines[-max_lines:]) + "\n")


def _default_status() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state": "unknown",
        "stage": None,
        "pending": None,
        "current_raw": None,
        "current_op": None,
        "current_job_id": None,
        "current_job_pid": None,
        "batch": None,
        "ollama": None,
        "llm": None,
        "last_event": None,
        "updated_at": None,
    }


def read_status() -> dict[str, Any]:
    return _read_json(STATUS_FILE, _default_status())


def write_status(fields: dict[str, Any]) -> dict[str, Any]:
    status = read_status()
    status.update(fields)
    status["schema_version"] = 1
    status["updated_at"] = now_iso()
    _atomic_write_text(
        STATUS_FILE,
        json.dumps(status, ensure_ascii=False, indent=2, default=str) + "\n",
    )
    return status


def append_event(level: str, message: str, **fields: Any) -> dict[str, Any]:
    record = {
        "timestamp": now_iso(),
        "level": level,
        "message": message,
        **fields,
    }
    _append_jsonl(EVENTS_FILE, record, max_lines=MAX_EVENTS)
    status_fields = {"last_event": record}
    if level in {"error", "warn"}:
        status_fields["last_problem"] = record
    write_status(status_fields)
    return record


def append_metric(kind: str, **fields: Any) -> dict[str, Any]:
    record = {
        "timestamp": now_iso(),
        "kind": kind,
        **fields,
    }
    _append_jsonl(METRICS_FILE, record, max_lines=MAX_METRICS)
    return record


def _read_jsonl(path: Path, limit: int) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").split("\n")
    except Exception:
        return []
    records: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            data = json.loads(line)
        except Exception:
            continue
        if isinstance(data, dict):
            records.append(data)
    return records


def read_events(limit: int = 100) -> list[dict[str, Any]]:
    return _read_jsonl(EVENTS_FILE, max(1, limit))


def read_metrics(limit: int = 300) -> list[dict[str, Any]]:
    return _read_jsonl(METRICS_FILE, max(1, limit))


def safe_write_status(**fields: Any) -> None:
    with contextlib.suppress(Exception):
        write_status(fields)


def safe_append_event(level: str, message: str, **fields: Any) -> None:
    with contextlib.suppress(Exception):
        append_event(level, message, **fields)


def safe_append_metric(kind: str, **fields: Any) -> None:
    with contextlib.suppress(Exception):
        append_metric(kind, **fields)


def reset_stale_runtime_status(
    *,
    max_age_seconds: int = STALE_RUNTIME_MAX_AGE_SECONDS,
) -> bool:
    """Clear a stale live status left by a killed/slept ingest process.

    Runtime status is just an observability cache. After a sleep/wake or
    process kill, the orchestrator lock can be reset while the dashboard still
    shows an old "running" / "LLM streaming" state. This function removes that
    stale live surface before a new drain cycle starts.
    """
    status = read_status()
    llm = status.get("llm") if isinstance(status.get("llm"), dict) else None
    active_llm = bool(llm and llm.get("active"))
    live_stage = status.get("stage") in {
        "batch",
        "raw",
        "triage",
        "generate",
        "apply",
        "locked",
    }
    looks_live = status.get("state") == "running" or live_stage or active_llm
    if not looks_live:
        return False

    pid = status.get("current_job_pid")
    if _pid_is_alive(pid):
        return False

    status_age = _age_seconds(status.get("updated_at"))
    llm_age = _age_seconds(llm.get("updated_at")) if llm else None
    has_dead_pid = isinstance(pid, int) and pid > 0
    pid_missing = pid is None
    aged_without_pid = (
        pid_missing
        and (llm_age is None or llm_age > max_age_seconds)
        and (status_age is None or status_age > max_age_seconds)
    )
    if not has_dead_pid and not aged_without_pid:
        return False

    write_status({
        "state": "idle",
        "stage": "waiting",
        "current_raw": None,
        "current_op": None,
        "current_job_id": None,
        "current_job_pid": None,
        "llm": None,
    })
    append_event(
        "warn",
        "runtime | cleared stale live status after drain startup",
        source="runtime",
        stale_pid=pid,
    )
    return True


def classify_log_message(message: str) -> str:
    text = message.lower()
    if "failed" in text or "error" in text or "parse failed" in text:
        return "error"
    if "partial" in text or "quarantined" in text or "converted to update" in text:
        return "warn"
    if "completed" in text or "created" in text or "updated" in text or "batch done" in text:
        return "success"
    return "info"


def tail_text_lines(path: Path, limit: int = 100) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    return lines[-max(1, limit):]


def compact_records(records: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return list(records)[-max(1, limit):]
