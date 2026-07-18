"""Cross-process sync-first admission for optional local research inference."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from llm_wiki_mcp.jsonl_write import append_jsonl_durable
from llm_wiki_mcp.wiki import WIKI_ROOT

RUNTIME_DIR = WIKI_ROOT / "runtime" / "research"
SYNC_DIR = RUNTIME_DIR / "sync-pending"
RESEARCH_LOCK = RUNTIME_DIR / "research-generation.lock"
ACTIVE_FILE = RUNTIME_DIR / "active-research.json"
SCHEDULER_LOG = RUNTIME_DIR / "scheduler.jsonl"


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _event(kind: str, **payload: Any) -> None:
    try:
        append_jsonl_durable(
            SCHEDULER_LOG,
            [{"ts": _iso(), "kind": kind, "pid": os.getpid(), **payload}],
        )
    except OSError:
        pass


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def _active_research() -> dict[str, Any] | None:
    try:
        payload = json.loads(ACTIVE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def sync_pending() -> bool:
    try:
        return any(path.is_file() for path in SYNC_DIR.iterdir())
    except OSError:
        return False


@dataclass(frozen=True)
class ForegroundReceipt:
    marker_id: str
    resource_wait_ms: int
    research_overlap: bool
    preempted: bool


@contextmanager
def foreground_lane(*, preempt_grace_ms: int = 250) -> Iterator[ForegroundReceipt]:
    """Announce foreground work and give a cancellable research child time to exit."""

    started = time.monotonic()
    marker_id = uuid.uuid4().hex
    SYNC_DIR.mkdir(parents=True, exist_ok=True)
    marker = SYNC_DIR / f"{os.getpid()}-{marker_id}.json"
    _atomic_json(marker, {"pid": os.getpid(), "marker_id": marker_id, "ts": _iso()})
    active = _active_research()
    overlap = active is not None
    deadline = started + max(0, preempt_grace_ms) / 1000.0
    while overlap and ACTIVE_FILE.exists() and time.monotonic() < deadline:
        time.sleep(0.005)
    preempted = overlap and not ACTIVE_FILE.exists()
    receipt = ForegroundReceipt(
        marker_id=marker_id,
        resource_wait_ms=round((time.monotonic() - started) * 1000),
        research_overlap=overlap,
        preempted=preempted,
    )
    _event("sync_enter", **asdict(receipt))
    try:
        yield receipt
    finally:
        try:
            marker.unlink()
        except OSError:
            pass
        _event("sync_exit", marker_id=marker_id)


@dataclass(frozen=True)
class ResearchAdmission:
    admitted: bool
    reason: str
    run_id: str
    resource_wait_ms: int = 0
    mode: str = "off"


class ResearchLease:
    def __init__(self, admission: ResearchAdmission, started: float) -> None:
        self.admission = admission
        self.started = started

    def cancelled(self) -> bool:
        return self.admission.admitted and sync_pending()


def capacity_proven() -> bool:
    return os.getenv("LLM_WIKI_RESEARCH_CAPACITY_PROVEN", "").casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


@contextmanager
def research_lane(
    run_id: str,
    *,
    enabled: bool,
    mode: str,
    purpose: str,
    needs_model: bool,
) -> Iterator[ResearchLease]:
    """Admit at most one low-priority research generation across processes."""

    started = time.monotonic()
    reason = ""
    if not enabled or mode == "off":
        reason = "research_disabled"
    elif needs_model and mode in {"auto", "shadow"} and not capacity_proven():
        reason = "protected_capacity_unproven"
    elif purpose not in {"explicit", "idle", "sleep", "trace", "shadow", "auto"}:
        reason = "invalid_purpose"
    elif sync_pending():
        reason = "sync_pending"

    handle = None
    if not reason:
        RESEARCH_LOCK.parent.mkdir(parents=True, exist_ok=True)
        handle = RESEARCH_LOCK.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            reason = "research_generation_busy"
            handle.close()
            handle = None
    admission = ResearchAdmission(
        admitted=not reason,
        reason=reason or "admitted",
        run_id=run_id,
        resource_wait_ms=round((time.monotonic() - started) * 1000),
        mode=mode,
    )
    lease = ResearchLease(admission, started)
    if not admission.admitted:
        _event("research_deferred", **asdict(admission), purpose=purpose)
        yield lease
        return

    _atomic_json(
        ACTIVE_FILE,
        {
            "run_id": run_id,
            "pid": os.getpid(),
            "started_at": _iso(),
            "mode": mode,
            "purpose": purpose,
            "needs_model": needs_model,
        },
    )
    _event("research_enter", **asdict(admission), purpose=purpose)
    try:
        yield lease
    finally:
        try:
            current = _active_research()
            if current is None or current.get("run_id") == run_id:
                ACTIVE_FILE.unlink(missing_ok=True)
        finally:
            if handle is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
        _event("research_exit", run_id=run_id, cancelled=lease.cancelled())


@dataclass(frozen=True)
class CancellableResult:
    status: str
    value: Any = None
    error: str = ""
    latency_ms: int = 0


def run_cancellable_command(
    command: list[str],
    stdin_text: str,
    lease: ResearchLease,
    *,
    timeout_seconds: float,
    poll_seconds: float = 0.01,
) -> CancellableResult:
    """Run a model worker subprocess and propagate sync cancellation."""

    started = time.monotonic()
    if not lease.admission.admitted:
        return CancellableResult("deferred", error=lease.admission.reason)
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.stdin is not None:
        process.stdin.write(stdin_text)
        process.stdin.close()
        process.stdin = None
    deadline = started + max(0.001, timeout_seconds)
    status = ""
    value: Any = None
    error = ""
    try:
        while process.poll() is None:
            if lease.cancelled():
                status = "cancelled"
                error = "cancelled for foreground sync"
                process.terminate()
                break
            if time.monotonic() >= deadline:
                status = "timeout"
                error = "research model call deadline exceeded"
                process.terminate()
                break
            time.sleep(poll_seconds)
        try:
            stdout, stderr = process.communicate(timeout=0.2)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=0.2)
        if not status:
            if process.returncode == 0:
                try:
                    value = json.loads(stdout)
                    status = "completed"
                except json.JSONDecodeError:
                    status = "error"
                    error = "research worker returned malformed JSON"
            else:
                status = "error"
                error = (stderr or stdout or f"research worker exited {process.returncode}")[-2000:]
    except Exception as exc:
        if process.poll() is None:
            process.kill()
        status = "error"
        error = f"{exc.__class__.__name__}: {exc}"
    result = CancellableResult(
        status=status,
        value=value,
        error=error,
        latency_ms=round((time.monotonic() - started) * 1000),
    )
    _event(
        "research_call_terminal",
        run_id=lease.admission.run_id,
        status=result.status,
        latency_ms=result.latency_ms,
        error=result.error,
    )
    return result
