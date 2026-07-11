"""Durable execution ledger for detached hook work."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from llm_wiki_mcp.frontier_review import redact_sensitive_text
from llm_wiki_mcp.wiki import WIKI_ROOT

JOB_DIR = WIKI_ROOT / "runtime" / "background-jobs"
STATE_FILE = JOB_DIR / "state.json"
LOCK_FILE = JOB_DIR / "state.lock"
MAX_ATTEMPTS = 5
MAX_TERMINAL_JOBS = 500
RETRYABLE_EXIT_CODE = 75
QUARANTINE_EXIT_CODE = 78
ACTIVE_STATUSES = frozenset({"queued", "running", "retry_wait"})
TERMINAL_STATUSES = frozenset({"completed", "quarantined", "failed", "cancelled"})
_SESSION_ID_KEYS = frozenset(
    {"session_id", "sessionId", "conversation_id", "conversationId", "rollout_id"}
)
_SESSION_PATH_KEYS = frozenset(
    {"session_file", "sessionFile", "transcript_path", "transcriptPath"}
)


def _is_capture_job(name: str) -> bool:
    """Return whether a job captures a mutable host transcript by session."""

    return name.endswith("-save") or name.endswith("-capture")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat(timespec="seconds")


@contextmanager
def _lock():
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load() -> dict[str, Any]:
    try:
        value = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {}
    if not isinstance(value, dict) or not isinstance(value.get("jobs"), dict):
        return {"schema_version": 1, "jobs": {}}
    return value


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
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, ProcessLookupError):
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
        key=lambda item: str(item[1].get("updated_at") or item[1].get("created_at") or ""),
        reverse=True,
    )
    removed = 0
    for job_id, _job in terminal[MAX_TERMINAL_JOBS:]:
        jobs.pop(job_id, None)
        removed += 1
    if removed:
        state["pruned_terminal_total"] = int(state.get("pruned_terminal_total") or 0) + removed
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
            abandoned = status == "running" and not _pid_alive(job.get("owner_pid"))
            pending_incident = quarantine_pending and status in {"queued", "retry_wait"}
            if not abandoned and not pending_incident:
                continue
            capture_job = _is_capture_job(str(job.get("name") or ""))
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
    except (json.JSONDecodeError, TypeError):
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
        for existing in state["jobs"].values():
            if not isinstance(existing, dict):
                continue
            if existing.get("dedupe_key") != dedupe or existing.get("status") not in ACTIVE_STATUSES:
                continue
            existing["stdin"] = stdin_text
            existing["env"] = dict(env)
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
            "env": dict(env),
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
        if not isinstance(job, dict) or job.get("status") not in {"queued", "retry_wait"}:
            return None
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
        if rerun_requested:
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
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in job.get("env", {}).items()})
    cmd = [sys.executable, "-m", str(job["module"]), *[str(v) for v in job.get("args", [])]]
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
        result = _finish(job_id, exit_code=1, output=f"{exc.__class__.__name__}: {exc}")
    print(json.dumps({"job_id": job_id, "status": result["status"], "exit_code": result.get("exit_code")}, ensure_ascii=False))
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
            if not isinstance(job, dict) or job.get("status") not in {"queued", "retry_wait"}:
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
    repair_stale()
    with _lock():
        jobs = _load()["jobs"]
    counts: dict[str, int] = {}
    oldest_pending: str | None = None
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        status = str(job.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
        if status in {"queued", "running", "retry_wait"}:
            created = str(job.get("created_at") or "")
            if created and (oldest_pending is None or created < oldest_pending):
                oldest_pending = created
    return {"status": "ok", "items": len(jobs), "by_status": counts, "oldest_pending_at": oldest_pending}


def main(argv: list[str] | None = None) -> int:
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
    if args.command != "run":
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
