"""Durable execution ledger for detached hook work."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import exclusive_text_file_lock, fsync_directory
from chronovisor.core.llm_security import build_child_process_env
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.core.timeutil import utc_now as _now

SECRET_PATTERNS = [
    re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"(?i)(api[_-]?key['\"=\s:]+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)(token['\"=\s:]+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)(cookie['\"=\s:]+)[^\s\n;]{8,}"),
]


def redact_sensitive_text(text: str | None) -> str:
    redacted = text or ""
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(_redact_match, redacted)
    return redacted


def _redact_match(match: re.Match[str]) -> str:
    if match.groups():
        return f"{match.group(1)}[REDACTED]"
    return "[REDACTED]"


JOB_DIR = CHRONOVISOR_ROOT / "runtime" / "background-jobs"
STATE_FILE = JOB_DIR / "state.json"
LOCK_FILE = JOB_DIR / "state.lock"
MAX_ATTEMPTS = 5
MAX_TERMINAL_JOBS = 500
MAX_CANCELLATION_TOMBSTONES = 500
RETRYABLE_EXIT_CODE = 75
QUARANTINE_EXIT_CODE = 78
RECENT_QUARANTINE_HOURS = 24
ACTIVE_STATUSES = frozenset({"queued", "running", "retry_wait"})
TERMINAL_STATUSES = frozenset({"completed", "quarantined", "failed", "cancelled"})
_SESSION_ID_KEYS = frozenset(
    {"session_id", "sessionId", "conversation_id", "conversationId", "rollout_id"}
)
_SESSION_PATH_KEYS = frozenset(
    {"session_file", "sessionFile", "transcript_path", "transcriptPath"}
)
CAPTURE_MODULES = frozenset(
    {
        "chronovisor.hosts.codex_record",
        "chronovisor.hosts.claude_code_record",
        "chronovisor.raw.codex_record",
        "chronovisor.raw.claude_code_record",
    }
)
_JOB_ENV_NAMES = frozenset(
    {
        "CLAUDE_CODE_CHRONOVISOR_RECORD_ENABLED",
        "CODEX_CHRONOVISOR_RECORD_ENABLED",
    }
)


def _retire(job: dict[str, Any]) -> None:
    job.update(
        status="cancelled",
        reason="retired_module",
        next_retry_at=None,
        updated_at=_iso(),
        stdin="",
        on_success=[],
    )
    job.pop("owner_pid", None)


def _is_capture_job(name: str) -> bool:
    """Return whether a job captures a mutable host transcript by session."""

    return name in {"codex-save", "claude-code-save"}


def _fixed_job_env(env: Any) -> dict[str, str]:
    if not isinstance(env, dict):
        return {}
    return {
        str(name): str(value) for name, value in env.items() if name in _JOB_ENV_NAMES
    }


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat(timespec="seconds")


def _lock() -> contextlib.AbstractContextManager[Any]:
    return exclusive_text_file_lock(LOCK_FILE)


class CaptureJobStateError(RuntimeError):
    """The capture ledger needs repair before work can resume."""


def _load(*, quarantine: bool = True) -> dict[str, Any]:
    try:
        value = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("jobs"), dict):
            return value
    except FileNotFoundError:
        if not STATE_FILE.is_symlink() and not any(
            STATE_FILE.parent.glob(f"{STATE_FILE.name}.corrupt-*")
        ):
            return {"schema_version": 1, "jobs": {}}
    except OSError, UnicodeError, json.JSONDecodeError:
        pass
    if quarantine:
        suffix = _now().strftime("%Y%m%dT%H%M%S%fZ")
        try:
            STATE_FILE.rename(STATE_FILE.with_name(f"{STATE_FILE.name}.corrupt-{suffix}"))
            fsync_directory(STATE_FILE.parent)
        except OSError:
            pass  # An unmovable ledger is still preserved and blocks all writers.
    raise CaptureJobStateError("capture job state is corrupt or unreadable; repair required")


def _save(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".jobs-", suffix=".json", dir=STATE_FILE.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, STATE_FILE)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp)


def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except TypeError, ValueError, ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _prune_terminal(state: dict[str, Any]) -> int:
    """Keep active work and a bounded tail of terminal history."""
    jobs = state.get("jobs")
    if not isinstance(jobs, dict):
        return 0
    terminal = [
        (job_id, job)
        for job_id, job in jobs.items()
        if isinstance(job, dict) and str(job.get("status") or "") in TERMINAL_STATUSES
    ]
    terminal.sort(
        key=lambda item: str(
            item[1].get("updated_at") or item[1].get("created_at") or ""
        ),
        reverse=True,
    )
    removed = 0
    for job_id, _job in terminal[MAX_TERMINAL_JOBS:]:
        jobs.pop(job_id, None)
        removed += 1
    if removed:
        state["pruned_terminal_total"] = (
            int(state.get("pruned_terminal_total") or 0) + removed
        )
    return removed


def repair_stale(*, quarantine_pending: bool = False) -> dict[str, Any]:
    """Quarantine abandoned workers and optionally an incident backlog."""
    repaired = 0
    requeued = 0
    with _lock():
        state = _load()
        for job in state["jobs"].values():
            if not isinstance(job, dict):
                continue
            status = str(job.get("status") or "")
            if job.get("module") not in CAPTURE_MODULES and status in {
                "queued",
                "retry_wait",
            }:
                _retire(job)
                repaired += 1
                continue
            abandoned = status == "running" and not _pid_alive(job.get("owner_pid"))
            pending_incident = quarantine_pending and status in {"queued", "retry_wait"}
            if not abandoned and not pending_incident:
                continue
            capture_job = job.get("module") in CAPTURE_MODULES
            if abandoned and capture_job and not quarantine_pending:
                job["status"] = "queued"
                job["attempts"] = 0
                job["output_tail"] = "requeued after abandoned capture worker"
                requeued += 1
            else:
                job["status"] = "quarantined"
                job["exit_code"] = 1
                job["output_tail"] = "quarantined by background-job stale recovery"
            job["next_retry_at"] = None
            job["updated_at"] = _iso()
            job.pop("owner_pid", None)
            repaired += 1
        pruned = _prune_terminal(state)
        if repaired or pruned:
            _save(state)
    return {
        "status": "ok",
        "repaired": repaired,
        "requeued": requeued,
        "pruned": pruned,
        "quarantine_pending": quarantine_pending,
    }


def _find_payload_string(value: Any, keys: frozenset[str]) -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys and isinstance(child, str) and child.strip():
                return child.strip()
        for child in value.values():
            found = _find_payload_string(child, keys)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_payload_string(child, keys)
            if found:
                return found
    return None


def _capture_session_identity(stdin_text: str) -> str | None:
    try:
        payload = json.loads(stdin_text)
    except json.JSONDecodeError, TypeError:
        return None
    session_id = _find_payload_string(payload, _SESSION_ID_KEYS)
    if session_id:
        return f"id:{session_id}"
    session_path = _find_payload_string(payload, _SESSION_PATH_KEYS)
    if session_path:
        normalized = str(Path(session_path).expanduser().resolve(strict=False))
        return f"path:{normalized}"
    return None


def _dedupe_key(name: str, module: str, args: list[str], stdin_text: str) -> str:
    session_identity = (
        _capture_session_identity(stdin_text) if _is_capture_job(name) else None
    )
    payload = json.dumps(
        {
            "name": name,
            "module": module,
            "args": args,
            "session": session_identity,
            "stdin": "" if session_identity else stdin_text,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def enqueue_job(
    *,
    name: str,
    module: str,
    args: list[str],
    env: dict[str, str],
    stdin_text: str,
) -> dict[str, Any]:
    session_identity = (
        _capture_session_identity(stdin_text) if _is_capture_job(name) else None
    )
    dedupe = _dedupe_key(name, module, args, stdin_text)
    if _is_capture_job(name) and session_identity is None:
        # Without a stable identity, coalescing could collapse different
        # sessions that emitted the same minimal Stop payload.
        dedupe = f"unscoped-{uuid.uuid4().hex}"
    with _lock():
        state = _load()
        tombstones = state.get("cancellation_tombstones")
        tombstone = tombstones.get(dedupe) if isinstance(tombstones, dict) else None
        if isinstance(tombstone, dict):
            return {
                "job_id": tombstone.get("last_job_id"),
                "name": name,
                "module": module,
                "args": list(args),
                "dedupe_key": dedupe,
                "status": "cancelled",
                "enqueued": False,
                "coalesced": False,
                "cancelled": True,
                "cancellation_reason": tombstone.get("reason"),
                "cancelled_at": tombstone.get("cancelled_at"),
            }
        for existing in state["jobs"].values():
            if not isinstance(existing, dict):
                continue
            if (
                existing.get("dedupe_key") != dedupe
                or existing.get("status") not in ACTIVE_STATUSES
            ):
                continue
            existing["stdin"] = stdin_text
            existing["env"] = _fixed_job_env(env)
            existing["updated_at"] = _iso()
            existing["coalesced_count"] = int(existing.get("coalesced_count") or 0) + 1
            if existing.get("status") == "running":
                existing["rerun_requested"] = True
            else:
                existing["status"] = "queued"
                existing["attempts"] = 0
                existing["next_retry_at"] = None
            _prune_terminal(state)
            _save(state)
            return {**existing, "enqueued": False, "coalesced": True}
        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "name": name,
            "module": module,
            "args": list(args),
            "env": _fixed_job_env(env),
            "stdin": stdin_text,
            "dedupe_key": dedupe,
            "lane_key": name,
            "status": "queued",
            "attempts": 0,
            "created_at": _iso(),
            "updated_at": _iso(),
            "next_retry_at": None,
            "exit_code": None,
            "output_tail": "",
        }
        state["jobs"][job_id] = job
        _prune_terminal(state)
        _save(state)
        return {**job, "enqueued": True, "coalesced": False}


def _claim(job_id: str) -> dict[str, Any] | None:
    with _lock():
        state = _load()
        job = state["jobs"].get(job_id)
        if not isinstance(job, dict) or job.get("status") not in {
            "queued",
            "retry_wait",
        }:
            return None
        if job.get("module") not in CAPTURE_MODULES:
            _retire(job)
            _save(state)
            return dict(job)
        job.pop("on_success", None)
        retry_at = job.get("next_retry_at")
        if isinstance(retry_at, str):
            try:
                if datetime.fromisoformat(retry_at) > _now():
                    return None
            except ValueError:
                pass
        lane_key = str(job.get("lane_key") or job.get("name") or "")
        for other_id, other in state["jobs"].items():
            if other_id == job_id or not isinstance(other, dict):
                continue
            other_lane = str(other.get("lane_key") or other.get("name") or "")
            if (
                lane_key
                and other_lane == lane_key
                and other.get("status") == "running"
                and _pid_alive(other.get("owner_pid"))
            ):
                return None
        job["status"] = "running"
        job["attempts"] = int(job.get("attempts") or 0) + 1
        job["updated_at"] = _iso()
        job["owner_pid"] = os.getpid()
        _save(state)
        return dict(job)


def _finish(job_id: str, *, exit_code: int, output: str) -> dict[str, Any]:
    with _lock():
        state = _load()
        job = state["jobs"].get(job_id)
        if not isinstance(job, dict):
            raise KeyError(job_id)
        attempts = int(job.get("attempts") or 0)
        rerun_requested = bool(job.pop("rerun_requested", False))
        if job.get("status") == "cancelled":
            job["next_retry_at"] = None
            job["stdin"] = ""
        elif rerun_requested:
            job["status"] = "queued"
            job["attempts"] = 0
            job["next_retry_at"] = None
        elif exit_code == QUARANTINE_EXIT_CODE:
            job["status"] = "quarantined"
            job["next_retry_at"] = None
        elif exit_code == 0:
            job["status"] = "completed"
            job["next_retry_at"] = None
            job["stdin"] = ""
        elif attempts >= MAX_ATTEMPTS:
            job["status"] = "quarantined"
            job["next_retry_at"] = None
        else:
            delay = min(6 * 3600, 60 * (2 ** max(0, attempts - 1)))
            job["status"] = "retry_wait"
            job["next_retry_at"] = _iso(_now() + timedelta(seconds=delay))
        job["exit_code"] = exit_code
        job["output_tail"] = redact_sensitive_text(output)[-4000:]
        job["updated_at"] = _iso()
        job.pop("owner_pid", None)
        _prune_terminal(state)
        _save(state)
        return dict(job)


def run_job(job_id: str) -> dict[str, Any]:
    job = _claim(job_id)
    if job is None:
        return {"status": "not_due", "job_id": job_id}
    if job.get("reason") == "retired_module":
        return job
    env = build_child_process_env()
    env.update(_fixed_job_env(job.get("env", {})))
    env["CHRONOVISOR_ROOT"] = str(CHRONOVISOR_ROOT)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    cmd = [
        sys.executable,
        "-m",
        str(job["module"]),
        *[str(v) for v in job.get("args", [])],
    ]
    try:
        completed = subprocess.run(
            cmd,
            input=str(job.get("stdin") or ""),
            text=True,
            capture_output=True,
            timeout=1800,
            env=env,
        )
        output = (completed.stdout or "") + "\n" + (completed.stderr or "")
        result = _finish(job_id, exit_code=completed.returncode, output=output)
    except Exception as exc:
        result = _finish(job_id, exit_code=1, output=exc.__class__.__name__)
    print(
        json.dumps(
            {
                "job_id": job_id,
                "status": result["status"],
                "exit_code": result.get("exit_code"),
            },
            ensure_ascii=False,
        )
    )
    if result.get("output_tail"):
        print(result["output_tail"])
    return result


def retry_due(*, limit: int = 8) -> dict[str, Any]:
    repair_stale()
    with _lock():
        state = _load()
        now = _now()
        due: list[tuple[str, str]] = []
        for job_id, job in state["jobs"].items():
            if not isinstance(job, dict) or job.get("status") not in {
                "queued",
                "retry_wait",
            }:
                continue
            retry_at = job.get("next_retry_at")
            try:
                ready = not retry_at or datetime.fromisoformat(str(retry_at)) <= now
            except ValueError:
                ready = True
            if ready:
                due.append((str(job.get("created_at") or ""), job_id))
        due.sort()
        job_ids = [job_id for _created, job_id in due[: max(0, limit)]]
    results = [run_job(job_id) for job_id in job_ids]
    return {"status": "ok", "due": len(job_ids), "results": results}


def snapshot() -> dict[str, Any]:
    """Read capture job status without taking a writer lock or repairing state."""
    try:
        state = _load(quarantine=False)
    except CaptureJobStateError:
        return {
            "status": "unavailable",
            "error": "capture_job_state_corrupt",
            "by_status": {},
        }
    jobs = {
        key: job
        for key, job in state["jobs"].items()
        if isinstance(job, dict) and job.get("module") in CAPTURE_MODULES
    }
    counts: dict[str, int] = {}
    oldest_pending: str | None = None
    quarantined_24h = 0
    latest_quarantined_at: str | None = None
    quarantine_cutoff = _now() - timedelta(hours=RECENT_QUARANTINE_HOURS)
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        status = str(job.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
        if status in {"queued", "running", "retry_wait"}:
            created = str(job.get("created_at") or "")
            if created and (oldest_pending is None or created < oldest_pending):
                oldest_pending = created
        if status == "quarantined":
            timestamp = str(
                job.get("updated_at")
                or job.get("finished_at")
                or job.get("created_at")
                or ""
            )
            if timestamp and (
                latest_quarantined_at is None or timestamp > latest_quarantined_at
            ):
                latest_quarantined_at = timestamp
            try:
                quarantined_at = datetime.fromisoformat(timestamp)
            except ValueError:
                quarantined_at = None
            if quarantined_at is not None:
                if quarantined_at.tzinfo is None:
                    quarantined_at = quarantined_at.replace(tzinfo=UTC)
                if quarantined_at >= quarantine_cutoff:
                    quarantined_24h += 1
    return {
        "status": "ok",
        "items": len(jobs),
        "by_status": counts,
        "oldest_pending_at": oldest_pending,
        "quarantined_24h": quarantined_24h,
        "latest_quarantined_at": latest_quarantined_at,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-background-jobs`` command-line entry point."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("job_id")
    retry_p = sub.add_parser("retry")
    retry_p.add_argument("--limit", type=int, default=8)
    sub.add_parser("status")
    repair_p = sub.add_parser("repair")
    repair_p.add_argument("--quarantine-pending", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_job(args.job_id)
    elif args.command == "retry":
        result = retry_due(limit=max(0, args.limit))
    elif args.command == "repair":
        result = repair_stale(quarantine_pending=args.quarantine_pending)
    else:
        result = snapshot()
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
