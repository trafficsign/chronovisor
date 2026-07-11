"""Orchestrator - deterministic control flow for Ingest/Lint scheduling.

NOT an LLM. Pure code logic. Local Ollama models handle content structuring;
this module handles when to trigger them.
"""

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from llm_wiki_mcp.wiki import RAW_DIR, WIKI_ROOT, LOG_FILE
from llm_wiki_mcp.ollama import is_available
from llm_wiki_mcp import runtime_status

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
    """Get active raw files that haven't been processed yet.

    A raw with ``raw_status: retracted`` remains on disk as audit evidence,
    but is never offered to normal ingest.
    """
    from llm_wiki_mcp.raw_replay import is_raw_retracted

    state = _load_state()
    processed = set(state.get("processed_raw_files", []))
    pending = []
    for f in sorted(RAW_DIR.glob("*.md")):
        if f.name not in processed and not is_raw_retracted(f):
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


def _mark_raws_processed_preserving_lock(filenames: list[str]) -> None:
    """Mark one ingest unit's source files without releasing the batch lock."""
    state = _load_state()
    processed = set(state.get("processed_raw_files", []))
    processed.update(filenames)
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
    runtime_status.safe_append_event(
        runtime_status.classify_log_message(message),
        message,
        source="orchestrator",
    )


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
        "processor": "ollama" if available else "unavailable",
    }


# How many consecutive triage failures we tolerate before quarantining the
# offending raws so the queue can keep moving.
TRIAGE_FAILURE_QUARANTINE_THRESHOLD = 3


@dataclass(frozen=True)
class _PendingRawUnit:
    paths: tuple[Path, ...]
    content: str | None = None
    raw_keywords: tuple[str, ...] | None = None
    fragment_record_sha256: str | None = None

    @property
    def representative(self) -> Path:
        return self.paths[0]

    @property
    def filenames(self) -> list[str]:
        return [path.name for path in self.paths]


def _raw_ingest_input_limit() -> int:
    from llm_wiki_mcp.runtime_config import load_decision_router_config

    return max(1, load_decision_router_config().max_input_chars)


def _quarantine_capture_fragment_paths(
    paths: list[Path],
    *,
    record_sha256: str,
    reason: str,
    details: dict | None = None,
) -> dict:
    """Preserve a complete unsafe fragment set with a durable reason manifest."""

    dead_letter_dir = (
        RAW_DIR
        / ".dead-letter"
        / "raw-capture-fragments"
        / record_sha256
    )
    dead_letter_dir.mkdir(parents=True, exist_ok=True)
    moved: list[dict[str, str]] = []
    for src in paths:
        if not src.exists():
            continue
        dst = dead_letter_dir / src.name
        duplicate = 1
        while dst.exists():
            dst = dead_letter_dir / f"{src.name}.duplicate-{duplicate}"
            duplicate += 1
        try:
            src.rename(dst)
        except OSError:
            continue
        moved.append({"source": src.name, "preserved_as": dst.name})

    filenames = [row["source"] for row in moved]
    if filenames:
        _mark_raws_processed_preserving_lock(filenames)
    manifest = {
        "schema_version": 1,
        "kind": "raw_capture_fragment_quarantine",
        "created_at": datetime.now().isoformat(),
        "record_sha256": record_sha256,
        "reason": reason,
        "details": details or {},
        "files": moved,
    }
    manifest_path = dead_letter_dir / "manifest.json"
    temporary = dead_letter_dir / ".manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return {
        "record_sha256": record_sha256,
        "reason": reason,
        "files": filenames,
        "manifest": str(manifest_path),
    }


def _prepare_pending_raw_units(
    pending: list[Path],
) -> tuple[list[_PendingRawUnit], list[dict], list[dict]]:
    """Separate normal raws from validated, complete fragment transport sets."""

    from llm_wiki_mcp.raw_capture_fragments import (
        RawCaptureFragmentError,
        group_capture_fragments,
        parse_capture_fragment,
    )

    fragments = []
    fragment_paths: set[Path] = set()
    quarantined: list[dict] = []
    deferred: list[dict] = []
    for path in pending:
        try:
            fragment = parse_capture_fragment(path)
        except (OSError, UnicodeError, RawCaptureFragmentError) as exc:
            fragment_paths.add(path)
            quarantined.append(
                _quarantine_capture_fragment_paths(
                    [path],
                    record_sha256=f"malformed-{path.stem}",
                    reason="fragment_parse_error",
                    details={"error": f"{type(exc).__name__}: {exc}"},
                )
            )
            continue
        if fragment is not None:
            fragments.append(fragment)
            fragment_paths.add(path)

    units = [
        _PendingRawUnit(paths=(path,))
        for path in pending
        if path not in fragment_paths
    ]
    try:
        groups = group_capture_fragments(fragments)
    except RawCaptureFragmentError as exc:
        # Duplicate indices are not independently meaningful. Preserve every
        # claimed transport fragment rather than leaking any one into ingest.
        paths = sorted(fragment_paths, key=lambda path: path.name)
        if paths:
            quarantined.append(
                _quarantine_capture_fragment_paths(
                    paths,
                    record_sha256="malformed-duplicate-fragment-index",
                    reason="fragment_group_invalid",
                    details={"error": str(exc)},
                )
            )
        return sorted(units, key=lambda unit: unit.representative.name), quarantined, deferred

    input_limit = _raw_ingest_input_limit()
    for group in groups:
        if not group.complete:
            deferred.append(
                {
                    "record_sha256": group.identity.record_sha256,
                    "reason": "fragment_group_incomplete",
                    "missing_indices": list(group.missing_indices),
                    "files": [path.name for path in group.paths],
                }
            )
            continue
        try:
            record_text = group.assemble_text()
        except RawCaptureFragmentError as exc:
            quarantined.append(
                _quarantine_capture_fragment_paths(
                    list(group.paths),
                    record_sha256=group.identity.record_sha256,
                    reason="fragment_integrity_failure",
                    details={"error": str(exc)},
                )
            )
            continue
        ingest_content = group.ingest_content()
        if len(ingest_content) > input_limit:
            quarantined.append(
                _quarantine_capture_fragment_paths(
                    list(group.paths),
                    record_sha256=group.identity.record_sha256,
                    reason="reassembled_input_limit_exceeded",
                    details={
                        "record_chars": len(record_text),
                        "ingest_chars": len(ingest_content),
                        "record_bytes": group.identity.record_bytes,
                        "max_input_chars": input_limit,
                        "fragment_count": group.identity.fragment_count,
                    },
                )
            )
            continue
        host_label = "Claude Code" if group.identity.host == "claude-code" else "Codex"
        units.append(
            _PendingRawUnit(
                paths=tuple(sorted(group.paths, key=lambda path: path.name)),
                content=ingest_content,
                raw_keywords=(
                    host_label,
                    "transcript-delta",
                    "transcript-reassembled",
                ),
                fragment_record_sha256=group.identity.record_sha256,
            )
        )
    units.sort(key=lambda unit: unit.representative.name)
    return units, quarantined, deferred


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
                runtime_status.safe_write_status(
                    state="idle",
                    stage="idle",
                    pending=0,
                    current_raw=None,
                    current_op=None,
                    batch=None,
                    current_job_id=None,
                    current_job_pid=None,
                    ollama=get_ollama_status(),
                    llm=None,
                )
                return {"triggered": False, "reason": "no pending raws"}
            reason = f"force=True with {len(pending_now)} pending"
        else:
            should, reason = should_ingest()
            if not should:
                runtime_status.safe_write_status(
                    state="idle",
                    stage="waiting",
                    pending=len(get_pending_raw_files()),
                    current_raw=None,
                    current_op=None,
                    batch=None,
                    current_job_id=None,
                    current_job_pid=None,
                    ollama=get_ollama_status(),
                    llm=None,
                )
                return {"triggered": False, "reason": reason}

        state = _load_state()
        if state.get("current_job_id"):
            runtime_status.safe_write_status(
                state="running",
                stage="locked",
                pending=len(get_pending_raw_files()),
                current_job_id=state.get("current_job_id"),
                current_job_pid=state.get("current_job_pid"),
                ollama=get_ollama_status(),
            )
            return {
                "triggered": False,
                "reason": f"ingest job {state['current_job_id']} already in flight",
            }

        pending = get_pending_raw_files()
        pending_before_count = len(pending)
        units, fragment_quarantined, fragment_deferred = _prepare_pending_raw_units(
            pending
        )

        # Limit semantic work units, not transport fragments. A complete
        # fragment set is reconstructed and presented to ingest exactly once.
        MAX_BATCH = 10
        units = units[:MAX_BATCH]
        filenames = [name for unit in units for name in unit.filenames]

        if not units:
            pending_after_preflight = len(get_pending_raw_files())
            runtime_status.safe_write_status(
                state="idle",
                stage="waiting",
                pending=pending_after_preflight,
                current_raw=None,
                current_op=None,
                current_job_id=None,
                current_job_pid=None,
                ollama=get_ollama_status(),
                llm=None,
            )
            if fragment_quarantined:
                return {
                    "triggered": True,
                    "reason": "capture fragments quarantined before semantic ingest",
                    "job_ids": [],
                    "files_attempted": [],
                    "files_processed": [],
                    "files_quarantined": [
                        name
                        for row in fragment_quarantined
                        for name in row.get("files", [])
                    ],
                    "fragment_quarantined": fragment_quarantined,
                    "fragment_deferred": fragment_deferred,
                    "processor": get_ollama_status()["processor"],
                    "elapsed_seconds": 0.0,
                }
            return {
                "triggered": False,
                "reason": "capture fragment groups are incomplete",
                "fragment_quarantined": [],
                "fragment_deferred": fragment_deferred,
            }

        # Quarantine if this same head-of-queue keeps blowing up triage.
        if state.get("triage_failure_count", 0) >= TRIAGE_FAILURE_QUARANTINE_THRESHOLD:
            quarantine_dir = _quarantine_pending_raws(filenames)
            runtime_status.safe_append_event(
                "warn",
                f"orchestrator | quarantined {len(filenames)} raws",
                source="orchestrator",
                raw_files=filenames,
                quarantine_dir=str(quarantine_dir),
            )
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
        runtime_status.safe_write_status(
            state="running",
            stage="batch",
            pending=pending_before_count,
            current_raw=None,
            current_op=None,
            current_job_id="__pending__",
            current_job_pid=os.getpid(),
            batch={
                "started_at": datetime.now().isoformat(),
                "index": 0,
                "total": len(filenames),
                "succeeded": 0,
                "failed": 0,
                "files": filenames,
            },
            ollama=get_ollama_status(),
            llm=None,
        )

        succeeded_units = 0
        try:
            for raw_index, unit in enumerate(units, start=1):
                raw_path = unit.representative
                fname = raw_path.name
                source_filenames = unit.filenames
                try:
                    raw_text = (
                        unit.content
                        if unit.content is not None
                        else raw_path.read_text()
                    )
                except Exception as e:
                    _orch_log(f"orchestrator | failed to read raw {fname}: {e}")
                    per_raw.append({
                        "filename": fname,
                        "source_files": source_filenames,
                        "succeeded": False,
                        "error": f"read error: {e}",
                    })
                    continue

                # Extract raw_keywords from frontmatter, falling back to the
                # legacy ``keywords`` field for raws written before Phase 1.
                # Anything that isn't a list of strings is normalized to [].
                if unit.raw_keywords is not None:
                    raw_keywords = list(unit.raw_keywords)
                else:
                    meta, _body = _frontmatter_parse(raw_text)
                    raw_keywords = _coerce_str_list(meta.get("raw_keywords"))
                    if raw_keywords is None:
                        raw_keywords = _coerce_str_list(meta.get("keywords")) or []

                processor = "ollama" if is_available() else "unavailable"
                job = job_store.create(processor=processor)
                job_ids.append(job.job_id)
                runtime_status.safe_write_status(
                    state="running",
                    stage="raw",
                    pending=len(get_pending_raw_files()),
                    current_raw=fname,
                    current_op=None,
                    current_job_id=job.job_id,
                    current_job_pid=os.getpid(),
                    batch={
                        "started_at": datetime.fromtimestamp(batch_started).isoformat(),
                        "index": raw_index,
                        "total": len(units),
                        "succeeded": succeeded_units,
                        "failed": len(per_raw) - succeeded_units,
                        "files": filenames,
                    },
                    ollama=get_ollama_status(),
                    llm=None,
                )

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

                def _on_complete(names=source_filenames, flag=raw_success_flag):
                    flag[0] = True
                    _mark_raws_processed_preserving_lock(list(names))

                def _on_finally(failed: bool, triage_failed: bool):
                    _update_triage_failure_count(failed, triage_failed)

                try:
                    run_ingest(
                        raw_text,
                        job.job_id,
                        on_complete=_on_complete,
                        on_finally=_on_finally,
                        metadata={"raw_keywords": raw_keywords, "source_raw": fname},
                    )
                except Exception as e:
                    # ``run_ingest`` already routes its own exceptions through
                    # job_store.update(FAILED) + on_finally; this catch is a
                    # belt-and-braces guard against a callback raising past
                    # the inner try/finally. Log and continue to the next raw
                    # so one bad raw can't strand the whole batch.
                    _orch_log(f"orchestrator | raw {fname} ingest exception: {e}")

                supervision = None
                if raw_success_flag[0]:
                    succeeded_filenames.extend(source_filenames)
                    succeeded_units += 1
                    try:
                        from llm_wiki_mcp.failure_supervisor import reset_raw_failure

                        reset_raw_failure(fname)
                    except Exception as e:
                        _orch_log(
                            f"orchestrator | failure supervisor reset failed "
                            f"for {fname}: {e}"
                        )
                else:
                    try:
                        from llm_wiki_mcp.failure_supervisor import (
                            record_raw_failure,
                            result_to_dict,
                        )

                        job_record = job_store.get(job.job_id)
                        supervision = result_to_dict(
                            record_raw_failure(
                                raw_path=raw_path,
                                error=job_record.error if job_record else None,
                                job_id=job.job_id,
                                raw_text=raw_text,
                            )
                        )
                        if (
                            isinstance(supervision, dict)
                            and supervision.get("quarantined") is True
                            and len(source_filenames) > 1
                        ):
                            fragment_paths = [
                                path for path in unit.paths if path.exists()
                            ]
                            quarantined_representative = supervision.get(
                                "quarantine_path"
                            )
                            if isinstance(quarantined_representative, str):
                                quarantined_path = Path(quarantined_representative)
                                if quarantined_path.exists():
                                    fragment_paths.append(quarantined_path)
                            _quarantine_capture_fragment_paths(
                                fragment_paths,
                                record_sha256=(
                                    unit.fragment_record_sha256
                                    or f"failed-{raw_path.stem}"
                                ),
                                reason="reassembled_ingest_failure",
                                details={
                                    "failure_class": supervision.get("failure_class")
                                },
                            )
                    except Exception as e:
                        _orch_log(
                            f"orchestrator | failure supervisor failed "
                            f"for {fname}: {e}"
                        )

                raw_result = {
                    "filename": fname,
                    "source_files": source_filenames,
                    "job_id": job.job_id,
                    "succeeded": raw_success_flag[0],
                }
                if unit.fragment_record_sha256 is not None:
                    raw_result["reassembled_fragment_record_sha256"] = (
                        unit.fragment_record_sha256
                    )
                if supervision is not None:
                    raw_result["supervision"] = supervision
                per_raw.append(raw_result)
                runtime_status.safe_append_event(
                    "success" if raw_success_flag[0] else "warn",
                    (
                        f"orchestrator | raw {fname} "
                        + ("processed" if raw_success_flag[0] else "not processed")
                    ),
                    source="orchestrator",
                    raw_file=fname,
                    job_id=job.job_id,
                    supervision=supervision,
                )
        finally:
            release_state = _load_state()
            _clear_current_job(release_state)
            _save_state(release_state)

        elapsed = time.time() - batch_started
        _orch_log(
            f"orchestrator | batch done: {len(succeeded_filenames)}/{len(filenames)} "
            f"succeeded, {elapsed:.1f}s, jobs={len(job_ids)}"
        )
        pending_after = len(get_pending_raw_files())
        runtime_status.safe_append_metric(
            "batch",
            pending_before=pending_before_count,
            pending_after=pending_after,
            files_attempted=len(filenames),
            files_processed=len(succeeded_filenames),
            files_failed=len(filenames) - len(succeeded_filenames),
            elapsed_seconds=round(elapsed, 2),
            processor=get_ollama_status()["processor"],
        )
        runtime_status.safe_write_status(
            state="running" if pending_after else "idle",
            stage="waiting" if pending_after else "idle",
            pending=pending_after,
            current_raw=None,
            current_op=None,
            current_job_id=None,
            current_job_pid=None,
            batch={
                "started_at": datetime.fromtimestamp(batch_started).isoformat(),
                "index": len(units),
                "total": len(units),
                "succeeded": succeeded_units,
                "failed": len(units) - succeeded_units,
                "elapsed_seconds": round(elapsed, 2),
                "files": filenames,
            },
            ollama=get_ollama_status(),
            llm=None,
        )

        return {
            "triggered": True,
            "reason": reason,
            "job_ids": job_ids,
            "files_attempted": filenames,
            "files_processed": succeeded_filenames,
            "files_quarantined": [
                name
                for row in fragment_quarantined
                for name in row.get("files", [])
            ],
            "fragment_quarantined": fragment_quarantined,
            "fragment_deferred": fragment_deferred,
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


def run_lint_if_due(*, dry_run: bool = False) -> dict:
    """Run lint check + safe apply if due.

    Returns result dict with status and details.
    """
    should, reason = should_lint()
    if not should:
        return {"triggered": False, "reason": reason}

    from llm_wiki_mcp.lint import (
        apply_safe_fixes,
        check,
        summarize_issues,
        write_repair_queue,
    )

    issues = check()
    try:
        from llm_wiki_mcp.wiki_snapshot import snapshot_wiki
        snapshot = (
            {"status": "skipped", "reason": "dry_run"}
            if dry_run
            else snapshot_wiki("before scheduled lint auto-fix")
        )
    except Exception as exc:
        snapshot = {"status": "error", "error": str(exc)}
    actions = apply_safe_fixes(issues, dry_run=dry_run)
    remaining = [i for i in issues if not i.get("auto_fixable")]
    if dry_run:
        repair_queue = str(WIKI_ROOT / "review" / "lint-repair-queue.jsonl")
    else:
        try:
            repair_queue = str(write_repair_queue(remaining))
        except Exception:
            repair_queue = None

    if not dry_run:
        mark_lint_complete()

    return {
        "triggered": True,
        "reason": reason,
        "total_issues": len(issues),
        "summary": summarize_issues(issues),
        "wiki_snapshot": snapshot,
        "actions_taken": actions,
        "remaining_issues": len(remaining),
        "repair_queue": repair_queue,
        "dry_run": dry_run,
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
