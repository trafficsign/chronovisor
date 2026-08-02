"""Durable execution ledger for detached hook work."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import exclusive_text_file_lock
from chronovisor.core.module_paths import canonical_module_path
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.core.timeutil import utc_now as _now
from chronovisor.decision.frontier_review import redact_sensitive_text

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


def _is_capture_job(name: str) -> bool:
    """Return whether a job captures a mutable host transcript by session."""

    return name.endswith("-save") or name.endswith("-capture")




def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat(timespec="seconds")


_lock = partial(exclusive_text_file_lock, LOCK_FILE)


def _load() -> dict[str, Any]:
    try:
        value = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {}
    if not isinstance(value, dict) or not isinstance(value.get("jobs"), dict):
        return {"schema_version": 1, "jobs": {}}
    _canonicalize_state_module_paths(value)
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
        with contextlib.suppress(OSError):
            os.unlink(tmp)


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
            "module": canonical_module_path(module),
            "args": args,
            "session": session_identity,
            "stdin": "" if session_identity else stdin_text,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonicalize_state_module_paths(state: dict[str, Any]) -> None:
    """Upgrade durable job references without retaining importable shims."""

    jobs = state.get("jobs")
    if isinstance(jobs, dict):
        for job in jobs.values():
            if not isinstance(job, dict):
                continue
            module = canonical_module_path(str(job.get("module") or ""))
            job["module"] = module
            args = [str(value) for value in job.get("args", [])]
            job["dedupe_key"] = _dedupe_key(
                str(job.get("name") or ""),
                module,
                args,
                str(job.get("stdin") or ""),
            )
            followups = job.get("on_success")
            if isinstance(followups, list):
                for followup in followups:
                    if isinstance(followup, dict):
                        followup["module"] = canonical_module_path(
                            str(followup.get("module") or "")
                        )

    tombstones = state.get("cancellation_tombstones")
    if not isinstance(tombstones, dict):
        return
    canonical_tombstones: dict[str, Any] = {}
    for old_key, value in tombstones.items():
        if not isinstance(value, dict):
            continue
        module = canonical_module_path(str(value.get("module") or ""))
        value["module"] = module
        canonical_tombstones[str(old_key)] = value
        name = str(value.get("name") or "")
        if _is_capture_job(name) and "stdin" not in value:
            continue
        args = [str(item) for item in value.get("args", [])]
        key = _dedupe_key(
            name,
            module,
            args,
            str(value.get("stdin") or ""),
        )
        canonical_tombstones[key] = value
    state["cancellation_tombstones"] = canonical_tombstones


def enqueue_job(
    *,
    name: str,
    module: str,
    args: list[str],
    env: dict[str, str],
    stdin_text: str,
    on_success: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    module = canonical_module_path(module)
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
            if existing.get("dedupe_key") != dedupe or existing.get("status") not in ACTIVE_STATUSES:
                continue
            existing["stdin"] = stdin_text
            existing["env"] = dict(env)
            if on_success is not None:
                existing["on_success"] = list(on_success)
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
            "on_success": list(on_success or []),
        }
        state["jobs"][job_id] = job
        _prune_terminal(state)
        _save(state)
        return {**job, "enqueued": True, "coalesced": False}


def cancel_matching_jobs(
    *,
    name: str,
    module: str,
    args: list[str],
    reason: str,
    stdin_text: str = "",
) -> dict[str, Any]:
    """Cancel exact active ledger entries without terminating their process.

    Callers must first make the underlying work item durably non-executable.
    A worker may already have been claimed between packet and ledger commits;
    preserving ``cancelled`` in :func:`_finish` then prevents that harmless
    cached/no-op worker from resurrecting the queue entry.
    """

    module = canonical_module_path(module)
    normalized_args = [str(value) for value in args]
    normalized_reason = str(reason).strip() or "superseded"
    dedupe = _dedupe_key(name, module, normalized_args, stdin_text)
    cancelled: list[str] = []
    prior_statuses: dict[str, str] = {}
    matched_job_ids: list[str] = []
    with _lock():
        state = _load()
        for job_id, job in state["jobs"].items():
            if not isinstance(job, dict) or job.get("dedupe_key") != dedupe:
                continue
            matched_job_ids.append(str(job_id))
            status = str(job.get("status") or "")
            prior_statuses[str(job_id)] = status
            if status not in ACTIVE_STATUSES:
                continue
            job["status"] = "cancelled"
            job["cancelled_at"] = _iso()
            job["cancellation_reason"] = normalized_reason
            job["next_retry_at"] = None
            job["updated_at"] = _iso()
            job["stdin"] = ""
            job.pop("rerun_requested", None)
            cancelled.append(str(job_id))
        tombstones = state.setdefault("cancellation_tombstones", {})
        if not isinstance(tombstones, dict):
            tombstones = {}
            state["cancellation_tombstones"] = tombstones
        prior_tombstone = tombstones.get(dedupe)
        prior_tombstone = (
            prior_tombstone if isinstance(prior_tombstone, dict) else {}
        )
        tombstone = {
            "name": name,
            "module": module,
            "args": normalized_args,
            "cancelled_at": prior_tombstone.get("cancelled_at") or _iso(),
            "reason": normalized_reason,
            "last_job_id": (
                cancelled[-1]
                if cancelled
                else prior_tombstone.get("last_job_id")
                or (matched_job_ids[-1] if matched_job_ids else None)
            ),
        }
        tombstone_changed = prior_tombstone != tombstone
        tombstones[dedupe] = tombstone
        ordered_tombstones = sorted(
            tombstones.items(),
            key=lambda item: str(item[1].get("cancelled_at") or "")
            if isinstance(item[1], dict)
            else "",
            reverse=True,
        )
        for stale_key, _value in ordered_tombstones[MAX_CANCELLATION_TOMBSTONES:]:
            tombstones.pop(stale_key, None)
        pruned = _prune_terminal(state)
        if cancelled or pruned or tombstone_changed:
            _save(state)
    return {
        "status": "ok",
        "matched": len(prior_statuses),
        "cancelled": len(cancelled),
        "cancelled_job_ids": cancelled,
        "prior_statuses": prior_statuses,
        "dedupe_key": dedupe,
        "tombstoned": True,
    }


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


def _enqueue_followups_locked(
    state: dict[str, Any],
    *,
    parent_job_id: str,
    parent: dict[str, Any],
    output_statuses: set[str],
    output_by_status: dict[str, str],
) -> list[str]:
    specs = parent.get("on_success")
    if not isinstance(specs, list):
        return []
    parent_stdin = str(parent.get("stdin") or "")
    enqueued: list[str] = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        required_status = spec.get("when_output_status")
        required_statuses = spec.get("when_output_statuses")
        if (
            isinstance(required_status, str)
            and required_status not in output_statuses
        ):
            continue
        if isinstance(required_statuses, list) and not any(
            isinstance(value, str) and value in output_statuses
            for value in required_statuses
        ):
            continue
        stdin_text = parent_stdin
        if spec.get("stdin_from_output") is True:
            candidates = (
                [value for value in required_statuses if isinstance(value, str)]
                if isinstance(required_statuses, list)
                else [required_status]
                if isinstance(required_status, str)
                else sorted(output_statuses)
            )
            stdin_text = next(
                (output_by_status[value] for value in candidates if value in output_by_status),
                "",
            )
            if not stdin_text:
                continue
        name = str(spec.get("name") or "").strip()
        module = str(spec.get("module") or "").strip()
        module = canonical_module_path(module)
        args = [str(value) for value in spec.get("args", [])]
        env_value = spec.get("env")
        env = (
            {str(key): str(value) for key, value in env_value.items()}
            if isinstance(env_value, dict)
            else {}
        )
        if not name or not module:
            continue
        dedupe = _dedupe_key(name, module, args, stdin_text)
        existing_job_id = ""
        for candidate_id, candidate in state["jobs"].items():
            if not isinstance(candidate, dict):
                continue
            if (
                candidate.get("dedupe_key") == dedupe
                and candidate.get("status") in ACTIVE_STATUSES
            ):
                candidate["stdin"] = stdin_text
                candidate["env"] = env
                candidate["updated_at"] = _iso()
                candidate["coalesced_count"] = int(
                    candidate.get("coalesced_count") or 0
                ) + 1
                if candidate.get("status") == "running":
                    candidate["rerun_requested"] = True
                else:
                    candidate["status"] = "queued"
                    candidate["attempts"] = 0
                    candidate["next_retry_at"] = None
                existing_job_id = str(candidate_id)
                break
        if existing_job_id:
            enqueued.append(existing_job_id)
            continue
        followup_id = uuid.uuid4().hex
        state["jobs"][followup_id] = {
            "job_id": followup_id,
            "name": name,
            "module": module,
            "args": args,
            "env": env,
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
            "on_success": [],
            "parent_job_id": parent_job_id,
        }
        enqueued.append(followup_id)
    return enqueued


def _last_json_status(output: str) -> str | None:
    """Return the last object status emitted by a background command."""

    for line in reversed(output.splitlines()):
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("status"), str):
            return payload["status"]
    return None


def _last_json_status_line(output: str) -> tuple[str, str] | None:
    """Return the exact final JSON receipt line and its declared status."""

    for line in reversed(output.splitlines()):
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        status = payload.get("status") if isinstance(payload, dict) else None
        if isinstance(status, str):
            return status, line
    return None


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
            rerun_receipt = _last_json_status_line(output) if exit_code == 0 else None
            if rerun_receipt:
                rerun_status, receipt_line = rerun_receipt
                deferred = job.setdefault("deferred_success_statuses", [])
                if isinstance(deferred, list) and rerun_status not in deferred:
                    deferred.append(rerun_status)
                deferred_outputs = job.setdefault("deferred_success_outputs", {})
                if isinstance(deferred_outputs, dict):
                    deferred_outputs[rerun_status] = receipt_line
            job["status"] = "queued"
            job["attempts"] = 0
            job["next_retry_at"] = None
        elif exit_code == QUARANTINE_EXIT_CODE:
            job["status"] = "quarantined"
            job["next_retry_at"] = None
        elif exit_code == 0:
            job["status"] = "completed"
            job["next_retry_at"] = None
            output_statuses = {
                status
                for status in [
                    _last_json_status(output),
                    *(
                        job.get("deferred_success_statuses", [])
                        if isinstance(job.get("deferred_success_statuses"), list)
                        else []
                    ),
                ]
                if isinstance(status, str) and status
            }
            output_by_status = {
                str(key): str(value)
                for key, value in (
                    job.get("deferred_success_outputs", {}).items()
                    if isinstance(job.get("deferred_success_outputs"), dict)
                    else []
                )
            }
            current_receipt = _last_json_status_line(output)
            if current_receipt is not None:
                output_by_status[current_receipt[0]] = current_receipt[1]
            if not job.get("followups_enqueued_at"):
                followup_ids = _enqueue_followups_locked(
                    state,
                    parent_job_id=job_id,
                    parent=job,
                    output_statuses=output_statuses,
                    output_by_status=output_by_status,
                )
                if followup_ids:
                    job["followup_job_ids"] = followup_ids
                    job["followups_enqueued_at"] = _iso()
            job.pop("deferred_success_statuses", None)
            job.pop("deferred_success_outputs", None)
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
    cmd = [
        sys.executable,
        "-m",
        canonical_module_path(str(job["module"])),
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


def get_job(job_id: str) -> dict[str, Any] | None:
    """Return one durable job without exposing the full ledger."""

    repair_stale()
    with _lock():
        job = _load()["jobs"].get(job_id)
    return dict(job) if isinstance(job, dict) else None


def recent_jobs(*, limit: int = 10) -> list[dict[str, Any]]:
    repair_stale()
    with _lock():
        jobs = [dict(row) for row in _load()["jobs"].values() if isinstance(row, dict)]
    jobs.sort(key=lambda row: str(row.get("updated_at") or row.get("created_at") or ""), reverse=True)
    return jobs[: max(0, limit)]


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
    if args.command != "run":
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
