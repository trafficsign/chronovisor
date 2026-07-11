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


def _dedupe_key(name: str, module: str, args: list[str], stdin_text: str) -> str:
    payload = json.dumps(
        {"name": name, "module": module, "args": args, "stdin": stdin_text},
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
    dedupe = _dedupe_key(name, module, args, stdin_text)
    with _lock():
        state = _load()
        for existing in state["jobs"].values():
            if existing.get("dedupe_key") == dedupe and existing.get("status") in {"queued", "running", "retry_wait"}:
                return dict(existing)
        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "name": name,
            "module": module,
            "args": list(args),
            "env": dict(env),
            "stdin": stdin_text,
            "dedupe_key": dedupe,
            "status": "queued",
            "attempts": 0,
            "created_at": _iso(),
            "updated_at": _iso(),
            "next_retry_at": None,
            "exit_code": None,
            "output_tail": "",
        }
        state["jobs"][job_id] = job
        _save(state)
        return dict(job)


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
        if exit_code == 0:
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
    with _lock():
        state = _load()
        now = _now()
        due: list[str] = []
        for job_id, job in state["jobs"].items():
            if not isinstance(job, dict) or job.get("status") not in {"queued", "retry_wait"}:
                continue
            retry_at = job.get("next_retry_at")
            try:
                ready = not retry_at or datetime.fromisoformat(str(retry_at)) <= now
            except ValueError:
                ready = True
            if ready:
                due.append(job_id)
        due = due[: max(0, limit)]
    results = [run_job(job_id) for job_id in due]
    return {"status": "ok", "due": len(due), "results": results}


def snapshot() -> dict[str, Any]:
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
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_job(args.job_id)
    elif args.command == "retry":
        result = retry_due(limit=max(0, args.limit))
    else:
        result = snapshot()
    if args.command != "run":
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
