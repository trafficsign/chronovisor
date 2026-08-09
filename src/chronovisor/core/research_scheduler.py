"""Cross-process sync-first admission for optional local research inference."""

from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.jsonl_write import append_jsonl_durable
from chronovisor.core.store import CHRONOVISOR_ROOT

RUNTIME_DIR = CHRONOVISOR_ROOT / "runtime" / "research"
SYNC_DIR = RUNTIME_DIR / "sync-pending"
RESEARCH_LOCK = RUNTIME_DIR / "research-generation.lock"
ACTIVE_FILE = RUNTIME_DIR / "active-research.json"
SCHEDULER_LOG = RUNTIME_DIR / "scheduler.jsonl"


def _iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _event(kind: str, **payload: Any) -> None:
    with suppress(OSError):
        append_jsonl_durable(
            SCHEDULER_LOG,
            [{"ts": _iso(), "kind": kind, "pid": os.getpid(), **payload}],
        )


def _atomic_ephemeral_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish runtime coordination state without putting fsync on sync latency."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
        os.replace(temporary, path)
    finally:
        with suppress(OSError):
            os.unlink(temporary)


def _active_research() -> dict[str, Any] | None:
    try:
        payload = json.loads(ACTIVE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _model_is_active(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    value = payload.get("model_active")
    if isinstance(value, bool):
        if not value:
            return False
        model_pid = payload.get("model_pid")
        if isinstance(model_pid, int) and model_pid > 1:
            try:
                os.kill(model_pid, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                pass
        return True
    # Treat pre-model-phase-schema records conservatively during rollout.
    return payload.get("needs_model") is True


def _set_model_phase(run_id: str, *, active: bool, model_pid: int | None) -> None:
    current = _active_research()
    if current is None or current.get("run_id") != run_id:
        return
    _atomic_ephemeral_json(
        ACTIVE_FILE,
        {
            **current,
            "model_active": active,
            "model_pid": model_pid,
        },
    )


def _preempt_requested(run_id: str) -> bool:
    current = _active_research()
    return bool(
        current is not None
        and current.get("run_id") == run_id
        and current.get("preempt_requested") is True
    )


def sync_pending() -> bool:
    try:
        markers = [path for path in SYNC_DIR.iterdir() if path.is_file()]
    except FileNotFoundError:
        return False
    except OSError:
        return True
    for marker in markers:
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            pid = payload["pid"]
            marker_id = payload["marker_id"]
            if (
                not isinstance(pid, int)
                or isinstance(pid, bool)
                or pid <= 0
                or not isinstance(marker_id, str)
                or not isinstance(payload["ts"], str)
                or not payload["ts"]
                or marker.name != f"{pid}-{marker_id}.json"
                or uuid.UUID(marker_id).hex != marker_id
            ):
                return True
            os.kill(pid, 0)
        except ProcessLookupError:
            try:
                marker.unlink()
            except OSError:
                return True
            continue
        except (OSError, KeyError, TypeError, ValueError):
            return True
        return True
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
    # This is an ephemeral cross-process cancellation marker, not durable
    # state. Visibility is the contract; fsync would put disk latency on every
    # prompt and violate the foreground wait budget under filesystem pressure.
    marker.write_text(
        json.dumps({"pid": os.getpid(), "marker_id": marker_id, "ts": _iso()}),
        encoding="utf-8",
    )
    active = _active_research()
    overlap = active is not None
    model_overlap = _model_is_active(active)
    preempt_signal_sent = False
    if model_overlap and active is not None:
        model_pid = active.get("model_pid")
        if isinstance(model_pid, int) and model_pid > 1:
            try:
                # The model worker is deliberately an isolated, stateless
                # child.  Signal it from the foreground process so preemption
                # cannot be delayed by a GPU-saturated research parent.
                _atomic_ephemeral_json(
                    ACTIVE_FILE,
                    {
                        **active,
                        "preempt_requested": True,
                        "preempt_marker_id": marker_id,
                        "preempt_requested_at": _iso(),
                    },
                )
                os.kill(model_pid, signal.SIGKILL)
                preempt_signal_sent = True
            except (ProcessLookupError, PermissionError):
                pass
    deadline = started + max(0, preempt_grace_ms) / 1000.0
    current = active
    while model_overlap and not preempt_signal_sent and time.monotonic() < deadline:
        current = _active_research()
        if not _model_is_active(current):
            break
        time.sleep(0.005)
    current = _active_research()
    preempted = model_overlap and (
        preempt_signal_sent or not _model_is_active(current)
    )
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
        with suppress(OSError):
            marker.unlink()
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
    return os.getenv("CHRONOVISOR_RESEARCH_CAPACITY_PROVEN", "").casefold() in {
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

    _atomic_ephemeral_json(
        ACTIVE_FILE,
        {
            "run_id": run_id,
            "pid": os.getpid(),
            "started_at": _iso(),
            "mode": mode,
            "purpose": purpose,
            "needs_model": needs_model,
            "model_active": False,
            "model_pid": None,
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
    if lease.cancelled():
        result = CancellableResult(
            "cancelled",
            error="cancelled for foreground sync",
        )
        _event(
            "research_call_terminal",
            run_id=lease.admission.run_id,
            status=result.status,
            latency_ms=result.latency_ms,
            error=result.error,
        )
        return result
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _set_model_phase(
        lease.admission.run_id,
        active=True,
        model_pid=process.pid,
    )
    input_fd = os.dup(process.stdin.fileno()) if process.stdin is not None else None
    if process.stdin is not None:
        process.stdin.close()
    process.stdin = None
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def feed_input() -> None:
        if input_fd is None:
            return
        try:
            payload = stdin_text.encode("utf-8")
            offset = 0
            while offset < len(payload):
                offset += os.write(input_fd, payload[offset:])
        except (BrokenPipeError, OSError):
            pass
        finally:
            with suppress(OSError):
                os.close(input_fd)

    # Import-heavy model workers may not read stdin immediately.  Feeding in a
    # daemon thread keeps the scheduler polling the foreground marker instead
    # of blocking on a full pipe before cancellation becomes observable.
    threading.Thread(target=feed_input, daemon=True).start()

    def drain_output(stream: Any, chunks: list[str]) -> None:
        if stream is None:
            return
        try:
            while value := stream.read(65_536):
                chunks.append(value)
        except (OSError, ValueError):
            pass

    # Model workers can return multi-megabyte JSON vectors.  Drain both pipes
    # while the child is alive; waiting for process exit before communicate()
    # deadlocks once either OS pipe buffer fills.
    stdout_thread = threading.Thread(
        target=drain_output,
        args=(process.stdout, stdout_chunks),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=drain_output,
        args=(process.stderr, stderr_chunks),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    deadline = started + max(0.001, timeout_seconds)
    status = ""
    value: Any = None
    error = ""
    try:
        while process.poll() is None:
            if lease.cancelled():
                status = "cancelled"
                error = "cancelled for foreground sync"
                # The worker is an isolated, stateless model-call child.  A
                # foreground prompt must not spend its 50 ms lease allowance
                # waiting for Python/HTTP graceful shutdown.
                process.kill()
                break
            if time.monotonic() >= deadline:
                status = "timeout"
                error = "research model call deadline exceeded"
                process.terminate()
                break
            time.sleep(poll_seconds)
        # A cooperative terminate is allowed only a small slice of the
        # foreground lease budget.  Some HTTP/model workers linger while
        # unwinding a cancelled request; kill them before they can consume the
        # 50 ms synchronous resource-wait allowance.
        settle_timeout = 0.025 if status in {"cancelled", "timeout"} else 0.2
        try:
            process.wait(timeout=settle_timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=0.05)
        stdout_thread.join(timeout=0.05)
        stderr_thread.join(timeout=0.05)
        stdout = "".join(stdout_chunks)
        stderr = "".join(stderr_chunks)
        if not status:
            if lease.cancelled() or _preempt_requested(
                lease.admission.run_id
            ):
                status = "cancelled"
                error = "cancelled for foreground sync"
            elif process.returncode == 0:
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
    finally:
        _set_model_phase(
            lease.admission.run_id,
            active=False,
            model_pid=None,
        )
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
