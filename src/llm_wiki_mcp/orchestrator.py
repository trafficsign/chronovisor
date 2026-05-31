"""Orchestrator - deterministic control flow for Ingest/Lint scheduling.

NOT an LLM. Pure code logic. Sonnet handles content structuring,
this module handles when to trigger it.
"""

import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from llm_wiki_mcp.wiki import RAW_DIR, WIKI_ROOT, LOG_FILE
from llm_wiki_mcp.ollama import is_available

# Config
INGEST_THRESHOLD = 5  # Trigger ingest after N raw files
LINT_INTERVAL_HOURS = 24  # Run lint every N hours

# State file
STATE_FILE = WIKI_ROOT / ".orchestrator_state.json"

# In-process lock so concurrent calls into run_pending_ingest can't both
# spawn an ingest thread for the same batch. The state file holds the
# cross-call truth (current_job_id), this lock just serializes the
# read-modify-write around it.
_INGEST_LOCK = threading.Lock()


def _load_state() -> dict:
    """Load orchestrator state."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "last_ingest": None,
        "last_lint": None,
        "processed_raw_files": [],
        "ollama_health": {"status": None, "checked_at": None},
        "current_job_id": None,
        "current_job_pid": None,
        "current_job_started_at": None,
        "triage_failure_count": 0,
    }


STALE_LOCK_MAX_AGE_SECONDS = 12 * 60 * 60


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


def _lock_is_fresh_in_live_process(state: dict) -> bool:
    started_at = _parse_iso_datetime(state.get("current_job_started_at"))
    if started_at is None:
        return False
    age = (datetime.now() - started_at).total_seconds()
    if age > STALE_LOCK_MAX_AGE_SECONDS:
        return False
    return _pid_is_alive(state.get("current_job_pid"))


def _clear_current_job(state: dict) -> None:
    state["current_job_id"] = None
    state["current_job_pid"] = None
    state["current_job_started_at"] = None


def reset_stale_lock() -> None:
    """Clear ``current_job_id`` on startup if the referenced job is gone.

    ``job_store`` is in-memory: after a process restart, any job_id persisted
    in the state file refers to a job that no longer exists, so the
    orchestrator would refuse to ever trigger ingest again. Call this once
    at server startup so a crash mid-ingest doesn't permanently brick the
    queue. The reservation sentinel ``__pending__`` is also cleared.
    """
    state = _load_state()
    cur = state.get("current_job_id")
    if not cur:
        return
    if _lock_is_fresh_in_live_process(state):
        return
    if cur == "__pending__":
        _clear_current_job(state)
        _save_state(state)
        return
    from llm_wiki_mcp.jobs import job_store  # local import to avoid cycle
    if job_store.get(cur) is None:
        _clear_current_job(state)
        _save_state(state)


def _save_state(state: dict) -> None:
    """Save orchestrator state."""
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def get_pending_raw_files() -> list[Path]:
    """Get raw files that haven't been processed yet."""
    state = _load_state()
    processed = set(state.get("processed_raw_files", []))
    pending = []
    for f in sorted(RAW_DIR.glob("*.md")):
        if f.name not in processed:
            pending.append(f)
    return pending


def should_ingest() -> tuple[bool, str]:
    """Check if ingest should be triggered. Returns (should_run, reason)."""
    pending = get_pending_raw_files()
    if len(pending) >= INGEST_THRESHOLD:
        return True, f"{len(pending)} pending raw files (threshold: {INGEST_THRESHOLD})"
    return False, f"Only {len(pending)} pending (threshold: {INGEST_THRESHOLD})"


def should_lint() -> tuple[bool, str]:
    """Check if lint should be triggered. Returns (should_run, reason)."""
    state = _load_state()
    last_lint = state.get("last_lint")

    if last_lint is None:
        return True, "Lint has never been run"

    last_lint_dt = datetime.fromisoformat(last_lint)
    hours_since = (datetime.now() - last_lint_dt).total_seconds() / 3600

    if hours_since >= LINT_INTERVAL_HOURS:
        return True, f"{hours_since:.1f} hours since last lint (threshold: {LINT_INTERVAL_HOURS}h)"
    return False, f"Only {hours_since:.1f}h since last lint (threshold: {LINT_INTERVAL_HOURS}h)"


def mark_raw_processed(filenames: list[str]) -> None:
    """Mark raw files as processed.

    Public, batch-level API used by quarantine and (legacy) full-batch
    ingest paths. Updates the processed set, refreshes ``last_ingest``,
    AND clears ``current_job_id`` + resets ``triage_failure_count``.
    The per-raw synchronous ingest loop in :func:`run_pending_ingest`
    does NOT call this — it uses :func:`_mark_one_raw_processed` so a
    single raw's success doesn't release the batch-wide in-flight slot.
    """
    state = _load_state()
    processed = set(state.get("processed_raw_files", []))
    processed.update(filenames)
    state["processed_raw_files"] = sorted(processed)
    state["last_ingest"] = datetime.now().isoformat()
    _clear_current_job(state)
    state["triage_failure_count"] = 0
    _save_state(state)


def _mark_one_raw_processed(filename: str) -> None:
    """Per-raw success mark — does NOT touch ``current_job_id``.

    Used by :func:`run_pending_ingest`'s synchronous serial loop. Each
    raw's ``on_complete`` callback marks just that file processed and
    clears the triage failure counter (success means the queue head is
    healthy), but leaves ``current_job_id`` intact so the batch-wide
    in-flight marker survives until the loop's outer ``finally`` clears
    it.
    """
    state = _load_state()
    processed = set(state.get("processed_raw_files", []))
    processed.add(filename)
    state["processed_raw_files"] = sorted(processed)
    state["last_ingest"] = datetime.now().isoformat()
    state["triage_failure_count"] = 0
    _save_state(state)


def _update_triage_failure_count(failed: bool, triage_failed: bool) -> None:
    """Per-raw triage counter update — does NOT touch ``current_job_id``.

    Mirror of :func:`_release_lock`'s counter logic without the lock-clear
    side effect, for use inside the per-raw loop where the batch holds the
    in-flight slot for its full duration.
    """
    if not failed:
        state = _load_state()
        state["triage_failure_count"] = 0
        _save_state(state)
    elif triage_failed:
        state = _load_state()
        state["triage_failure_count"] = state.get("triage_failure_count", 0) + 1
        _save_state(state)


def _release_lock(failed: bool = False, triage_failed: bool = False) -> None:
    """Clear the in-flight job marker after a job finishes.

    Counter semantics: only ``triage_failed`` increments
    ``triage_failure_count``. A triage failure means the *content* is
    unprocessable, so repeated occurrences justify quarantine. Generate
    parse errors, apply errors, or Ollama-unavailable are transient or
    per-op and must NOT push raws toward dead-letter.

    A successful run (failed=False) clears the counter so transient
    failures don't accumulate.
    """
    state = _load_state()
    _clear_current_job(state)
    if not failed:
        state["triage_failure_count"] = 0
    elif triage_failed:
        state["triage_failure_count"] = state.get("triage_failure_count", 0) + 1
    _save_state(state)


def _orch_log(message: str) -> None:
    """Best-effort log to LOG_FILE. Failures are swallowed so a wedged log
    file can't break the orchestrator loop.
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(LOG_FILE, "a") as f:
            f.write(f"\n- [{timestamp}] {message}")
    except Exception:
        pass


def mark_lint_complete() -> None:
    """Mark lint as completed."""
    state = _load_state()
    state["last_lint"] = datetime.now().isoformat()
    _save_state(state)


def get_ollama_status() -> dict:
    """Get Ollama status with caching."""
    available = is_available()
    return {
        "available": available,
        "processor": "ollama" if available else "sonnet",
    }


# How many consecutive triage failures we tolerate before quarantining the
# offending raws so the queue can keep moving.
TRIAGE_FAILURE_QUARANTINE_THRESHOLD = 3


def _quarantine_pending_raws(filenames: list[str]) -> Path:
    """Move raws that keep killing triage into a dead-letter folder.

    Marks them processed so the orchestrator stops retrying them, but keeps
    the source bytes on disk under ``raw/.dead-letter/`` so we can inspect
    or re-feed them after fixing the parser.
    """
    dead_letter_dir = RAW_DIR / ".dead-letter"
    dead_letter_dir.mkdir(exist_ok=True)
    moved: list[str] = []
    for name in filenames:
        src = RAW_DIR / name
        if not src.exists():
            continue
        dst = dead_letter_dir / name
        try:
            src.rename(dst)
            moved.append(name)
        except OSError:
            continue
    if moved:
        # Mark as processed so get_pending_raw_files() skips them even if a
        # future operation copies them back into raw/.
        mark_raw_processed(moved)
    return dead_letter_dir


def run_pending_ingest(force: bool = False) -> dict:
    """Run ingest on all pending raw files if threshold is met.

    Per-raw synchronous serial execution. Each pending raw is parsed for
    ``raw_keywords`` (with legacy ``keywords`` fallback), then ingested
    individually via :func:`run_ingest` with that metadata propagated as
    a side channel for downstream apply. Successful raws are marked
    individually so a single raw's failure doesn't block retry of the
    others — and so partial-batch progress survives a server crash.

    The ``_INGEST_LOCK`` (in-process) serializes concurrent calls; the
    ``current_job_id`` state slot is reserved at batch start and cleared
    in the outer ``finally`` so cross-process observers (and the startup
    ``reset_stale_lock``) still see the batch as in flight.

    Args:
        force: When True, bypass the ``INGEST_THRESHOLD`` check and trigger
            immediately as long as at least one raw is pending. Used by
            ``wiki_ingest`` to preserve its historical "ingest now" contract.

    Returns result dict with status and details. On a triggered batch the
    result includes per-raw entries (filename, job_id, succeeded) so
    callers can inspect partial outcomes without re-loading job state.
    """
    with _INGEST_LOCK:
        if force:
            pending_now = get_pending_raw_files()
            if not pending_now:
                return {"triggered": False, "reason": "no pending raws"}
            reason = f"force=True with {len(pending_now)} pending"
        else:
            should, reason = should_ingest()
            if not should:
                return {"triggered": False, "reason": reason}

        state = _load_state()
        if state.get("current_job_id"):
            return {
                "triggered": False,
                "reason": f"ingest job {state['current_job_id']} already in flight",
            }

        pending = get_pending_raw_files()

        # Limit batch size to avoid overwhelming LLM.
        MAX_BATCH = 10
        pending = pending[:MAX_BATCH]
        filenames = [f.name for f in pending]

        # Quarantine if this same head-of-queue keeps blowing up triage.
        if state.get("triage_failure_count", 0) >= TRIAGE_FAILURE_QUARANTINE_THRESHOLD:
            quarantine_dir = _quarantine_pending_raws(filenames)
            return {
                "triggered": False,
                "reason": (
                    f"quarantined {len(filenames)} raws after "
                    f"{state['triage_failure_count']} consecutive triage failures "
                    f"→ {quarantine_dir}"
                ),
                "quarantined": filenames,
            }

        # Reserve the batch-wide in-flight slot BEFORE entering the loop.
        # Cleared in the outer ``finally`` regardless of how we exit.
        reserved_state = _load_state()
        reserved_state["current_job_id"] = "__pending__"
        reserved_state["current_job_pid"] = os.getpid()
        reserved_state["current_job_started_at"] = datetime.now().isoformat()
        _save_state(reserved_state)

        # Lazy imports keep module-level cycles minimal and isolate test
        # patches that swap these out.
        from llm_wiki_mcp.ingest import run_ingest
        from llm_wiki_mcp.jobs import job_store
        from llm_wiki_mcp.frontmatter import parse as _frontmatter_parse

        per_raw: list[dict] = []
        job_ids: list[str] = []
        succeeded_filenames: list[str] = []
        batch_started = time.time()

        try:
            for raw_path in pending:
                fname = raw_path.name
                try:
                    raw_text = raw_path.read_text()
                except Exception as e:
                    _orch_log(f"orchestrator | failed to read raw {fname}: {e}")
                    per_raw.append({
                        "filename": fname,
                        "succeeded": False,
                        "error": f"read error: {e}",
                    })
                    continue

                # Extract raw_keywords from frontmatter, falling back to the
                # legacy ``keywords`` field for raws written before Phase 1.
                # Anything that isn't a list of strings is normalized to [].
                meta, _body = _frontmatter_parse(raw_text)
                raw_keywords = _coerce_str_list(meta.get("raw_keywords"))
                if raw_keywords is None:
                    raw_keywords = _coerce_str_list(meta.get("keywords")) or []

                processor = "ollama" if is_available() else "sonnet"
                job = job_store.create(processor=processor)
                job_ids.append(job.job_id)

                # Make the per-raw job_id observable to other processes
                # while it runs. The outer ``finally`` will clear it.
                visible_state = _load_state()
                visible_state["current_job_id"] = job.job_id
                visible_state["current_job_pid"] = os.getpid()
                if not visible_state.get("current_job_started_at"):
                    visible_state["current_job_started_at"] = datetime.now().isoformat()
                _save_state(visible_state)

                # Mutable flag the on_complete closure flips on success.
                # Wrapped in a list so the closure can mutate it without
                # ``nonlocal`` gymnastics across the loop iterations.
                raw_success_flag = [False]

                def _on_complete(name=fname, flag=raw_success_flag):
                    flag[0] = True
                    _mark_one_raw_processed(name)

                def _on_finally(failed: bool, triage_failed: bool):
                    _update_triage_failure_count(failed, triage_failed)

                try:
                    run_ingest(
                        raw_text,
                        job.job_id,
                        on_complete=_on_complete,
                        on_finally=_on_finally,
                        metadata={"raw_keywords": raw_keywords},
                    )
                except Exception as e:
                    # ``run_ingest`` already routes its own exceptions through
                    # job_store.update(FAILED) + on_finally; this catch is a
                    # belt-and-braces guard against a callback raising past
                    # the inner try/finally. Log and continue to the next raw
                    # so one bad raw can't strand the whole batch.
                    _orch_log(f"orchestrator | raw {fname} ingest exception: {e}")

                if raw_success_flag[0]:
                    succeeded_filenames.append(fname)

                per_raw.append({
                    "filename": fname,
                    "job_id": job.job_id,
                    "succeeded": raw_success_flag[0],
                })
        finally:
            release_state = _load_state()
            _clear_current_job(release_state)
            _save_state(release_state)

        elapsed = time.time() - batch_started
        _orch_log(
            f"orchestrator | batch done: {len(succeeded_filenames)}/{len(filenames)} "
            f"succeeded, {elapsed:.1f}s, jobs={len(job_ids)}"
        )

        return {
            "triggered": True,
            "reason": reason,
            "job_ids": job_ids,
            "files_attempted": filenames,
            "files_processed": succeeded_filenames,
            "per_raw": per_raw,
            "processor": get_ollama_status()["processor"],
            "elapsed_seconds": round(elapsed, 2),
        }


def _coerce_str_list(value: object) -> list[str] | None:
    """Return ``value`` as ``list[str]`` only if every element is a str.

    Anything else (None, str, dict, list with non-str items) returns
    ``None`` so the caller can fall through to the next source. We
    deliberately don't promote scalars to singleton lists — that would
    silently rewrite intent if a raw frontmatter accidentally wrote
    ``raw_keywords: foo`` instead of ``raw_keywords: [foo]``.
    """
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    return None


def run_lint_if_due() -> dict:
    """Run lint check + safe apply if due.

    Returns result dict with status and details.
    """
    should, reason = should_lint()
    if not should:
        return {"triggered": False, "reason": reason}

    from llm_wiki_mcp.lint import check, apply_safe_fixes

    issues = check()
    actions = apply_safe_fixes(issues)
    remaining = [i for i in issues if not i.get("auto_fixable")]

    mark_lint_complete()

    return {
        "triggered": True,
        "reason": reason,
        "total_issues": len(issues),
        "actions_taken": actions,
        "remaining_issues": len(remaining),
    }


def tick() -> dict:
    """Main orchestration tick. Call this periodically.

    Checks if ingest or lint should run, and triggers them if needed.
    Returns summary of what happened.
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "ollama": get_ollama_status(),
        "ingest": run_pending_ingest(),
        "lint": run_lint_if_due(),
    }
    return results
