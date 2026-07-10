"""Autonomous repair lane for ingest read-back failures.

The ingest pipeline records failures as append-only JSONL events.  This module
turns those events into an idempotent state machine: safe retrieval misses
become exact query hints, transient failures back off and eventually enter
quarantine, and only external access/billing failures require a human.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from llm_wiki_mcp import recall_hints, wiki
from llm_wiki_mcp.convergence import HUMAN_REQUIRED_FAILURE_CLASSES


FAILURE_FILE = wiki.WIKI_ROOT / "runtime" / "ingest-read-back-failures.jsonl"
LEDGER_FILE = wiki.WIKI_ROOT / "runtime" / "ingest-read-back-repair.json"
SCHEMA_VERSION = 1
TERMINAL_STATUSES = frozenset({"applied", "quarantined", "human_required"})
HUMAN_REQUIRED_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:401|403)\b",
        r"\bapi[-_ ]?key\b",
        r"\bauthentication\b",
        r"\bauthorization\b",
        r"\boauth\b",
        r"\bbilling\b",
        r"\bcredential(?:s)?\b",
        r"\bforbidden\b",
        r"\bkeychain\b",
        r"\bpayment required\b",
        r"\bquota\b",
        r"\bunauthori[sz]ed\b",
    )
)


def _canonical_reason(value: object) -> str:
    reason = str(value or "unknown").strip().casefold().replace("_", "-")
    return re.sub(r"\s+", "-", reason) or "unknown"


def failure_key(failure: dict[str, Any]) -> str:
    """Return a stable key independent of timestamp and result ordering."""
    canonical = {
        "page_id": str(failure.get("page_id") or "").strip().casefold(),
        "query": recall_hints.normalize_query_text(str(failure.get("query") or "")),
        "reason": _canonical_reason(failure.get("reason")),
    }
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"read-back-{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:24]}"


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return [], 0
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows, len(lines)


def _flatten_failures(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for row in rows:
        failures = row.get("failed")
        if not isinstance(failures, list):
            failures = [row] if "reason" in row or "error" in row else []
        observed_at = str(row.get("timestamp") or row.get("ts") or "")
        for raw_failure in failures:
            if not isinstance(raw_failure, dict):
                continue
            failure = dict(raw_failure)
            key = failure_key(failure)
            prior = aggregated.get(key)
            if prior is None:
                aggregated[key] = {
                    "failure_key": key,
                    "failure": failure,
                    "first_seen": observed_at,
                    "last_seen": observed_at,
                    "occurrences": 1,
                }
                continue
            prior["failure"] = failure
            prior["last_seen"] = observed_at or prior.get("last_seen", "")
            prior["occurrences"] = int(prior.get("occurrences") or 0) + 1
    return aggregated


def _empty_ledger() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "entries": {}}


def _load_ledger(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_ledger()
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), dict):
        return _empty_ledger()
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp = Path(handle.name)
            handle.write(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _as_utc(parsed)


def _merge_entries(
    ledger: dict[str, Any],
    observed: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    raw_entries = ledger.get("entries")
    entries = deepcopy(raw_entries) if isinstance(raw_entries, dict) else {}
    for key, aggregate in observed.items():
        existing = entries.get(key)
        if not isinstance(existing, dict):
            existing = {
                "failure_key": key,
                "status": "pending",
                "attempts": 0,
            }
        previous_occurrences = int(existing.get("occurrences") or 0)
        previous_last_seen = str(existing.get("last_seen") or "")
        existing["failure"] = aggregate["failure"]
        existing["first_seen"] = existing.get("first_seen") or aggregate.get("first_seen") or ""
        existing["last_seen"] = aggregate.get("last_seen") or existing.get("last_seen") or ""
        existing["occurrences"] = max(
            previous_occurrences,
            int(aggregate.get("occurrences") or 0),
        )
        if existing.get("status") == "applied":
            resolved_occurrences = int(existing.get("resolved_occurrences") or previous_occurrences)
            resolved_last_seen = str(existing.get("resolved_last_seen") or previous_last_seen)
            observed_time = _parse_time(aggregate.get("last_seen"))
            resolved_time = _parse_time(resolved_last_seen)
            newly_observed = (
                int(aggregate.get("occurrences") or 0) > resolved_occurrences
                or (
                    observed_time is not None
                    and (resolved_time is None or observed_time > resolved_time)
                )
            )
            if newly_observed:
                existing["status"] = "pending"
                existing["attempts"] = 0
                existing["reopen_count"] = int(existing.get("reopen_count") or 0) + 1
                existing["reopened_at"] = aggregate.get("last_seen") or datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                )
                existing.pop("next_attempt_at", None)
        entries[key] = existing
    return entries


def _human_required(failure: dict[str, Any]) -> bool:
    failure_class = str(failure.get("failure_class") or "")
    if failure_class in HUMAN_REQUIRED_FAILURE_CLASSES:
        return True
    text = " ".join(
        str(failure.get(field) or "")
        for field in ("reason", "error", "message", "detail")
    )
    return any(pattern.search(text) for pattern in HUMAN_REQUIRED_PATTERNS)


def _due(entry: dict[str, Any], *, now: datetime) -> bool:
    if str(entry.get("status") or "pending") == "pending":
        return True
    if str(entry.get("status") or "") != "retry_wait":
        return False
    next_attempt = _parse_time(entry.get("next_attempt_at"))
    return next_attempt is None or next_attempt <= now


def _schedule_retry(
    entry: dict[str, Any],
    *,
    now: datetime,
    max_attempts: int,
    retry_base_seconds: int,
    max_backoff_seconds: int,
    error: str = "",
) -> str:
    attempts = int(entry.get("attempts") or 0) + 1
    entry["attempts"] = attempts
    entry["last_attempt_at"] = now.isoformat(timespec="seconds")
    if error:
        entry["last_error"] = error[:500]
    if attempts >= max_attempts:
        entry["status"] = "quarantined"
        entry["quarantined_at"] = now.isoformat(timespec="seconds")
        entry.pop("next_attempt_at", None)
        return "quarantined"
    delay = min(max_backoff_seconds, retry_base_seconds * (2 ** max(0, attempts - 1)))
    entry["status"] = "retry_wait"
    entry["next_attempt_at"] = (now + timedelta(seconds=delay)).isoformat(timespec="seconds")
    return "retry_scheduled"


def _matching_hint(page_id: str, query: str, *, path: Path) -> dict[str, Any] | None:
    query_key = recall_hints.normalize_query_text(query)
    for hint in recall_hints.load_query_hints(path):
        hint_query_key = str(
            hint.get("query_key")
            or recall_hints.normalize_query_text(str(hint.get("query") or ""))
        )
        same_page = str(hint.get("page_id") or "").casefold() == page_id.casefold()
        if same_page and hint_query_key == query_key:
            return hint
    return None


def _target_page_exists(page_id: str) -> bool:
    return (
        recall_hints.wiki.find_page(page_id) is not None
        or (recall_hints.wiki.SYSTEM_DIR / f"{page_id}.md").exists()
    )


def _ensure_query_hint(
    entry: dict[str, Any],
    *,
    hints_file: Path,
) -> tuple[str, dict[str, Any]]:
    failure = entry["failure"]
    page_id = str(failure.get("page_id") or "").strip()
    query = str(failure.get("query") or "").strip()
    if not _target_page_exists(page_id):
        raise ValueError(f"query hint target page does not exist: {page_id!r}")
    existing = _matching_hint(page_id, query, path=hints_file)
    if existing is not None:
        return "already_present", existing
    hint = recall_hints.add_query_hint(
        page_id=page_id,
        query=query,
        signal="ingest read-back not-in-top-results",
        source="ingest-read-back-repair",
        normalize_key=str(entry.get("failure_key") or ""),
        path=hints_file,
        increment_existing=False,
    )
    return "applied", hint


def run_read_back_repair(
    *,
    failure_file: Path = FAILURE_FILE,
    ledger_file: Path = LEDGER_FILE,
    hints_file: Path | None = None,
    max_items: int = 20,
    dry_run: bool = False,
    now: datetime | None = None,
    max_attempts: int = 3,
    retry_base_seconds: int = 3600,
    max_backoff_seconds: int = 7 * 24 * 3600,
    budget: Any | None = None,
) -> dict[str, Any]:
    """Process a bounded batch of read-back failures.

    Dry runs read the source, ledger, and hint store but never create or modify
    any file.  A terminal ledger entry is never attempted again.  The exact
    hint lookup also protects against a crash after hint write but before the
    ledger's atomic replace.
    """
    now_utc = _as_utc(now)
    hints_path = hints_file or recall_hints.QUERY_HINTS_FILE
    rows, source_lines = _read_jsonl(failure_file)
    observed = _flatten_failures(rows)
    original_ledger = _load_ledger(ledger_file)
    entries = _merge_entries(original_ledger, observed)

    max_items = max(0, int(max_items))
    max_attempts = max(1, int(max_attempts))
    retry_base_seconds = max(1, int(retry_base_seconds))
    max_backoff_seconds = max(retry_base_seconds, int(max_backoff_seconds))
    candidates = [
        entry
        for entry in entries.values()
        if isinstance(entry, dict)
        and str(entry.get("status") or "pending") not in TERMINAL_STATUSES
        and _due(entry, now=now_utc)
    ]
    candidates.sort(
        key=lambda entry: (
            str(entry.get("first_seen") or ""),
            str(entry.get("failure_key") or ""),
        )
    )

    actions: list[dict[str, Any]] = []
    counts = {
        "applied": 0,
        "already_present": 0,
        "retry_scheduled": 0,
        "quarantined": 0,
        "human_required": 0,
        "budget_deferred": 0,
    }
    mutation_consumed = False
    for entry in candidates[:max_items]:
        failure = entry.get("failure") if isinstance(entry.get("failure"), dict) else {}
        key = str(entry.get("failure_key") or failure_key(failure))
        reason = _canonical_reason(failure.get("reason"))
        action = {
            "failure_key": key,
            "page_id": str(failure.get("page_id") or ""),
            "reason": reason,
        }

        if budget is not None and not dry_run:
            allowed, budget_reason = budget.consume("mutation")
            if not allowed:
                counts["budget_deferred"] += 1
                action["outcome"] = "budget_deferred"
                action["budget_reason"] = budget_reason
                actions.append(action)
                continue
            mutation_consumed = True

        if _human_required(failure):
            outcome = "human_required"
            entry["status"] = outcome
            entry["human_required_at"] = now_utc.isoformat(timespec="seconds")
        elif reason == "not-in-top-results":
            page_id = str(failure.get("page_id") or "").strip()
            query = str(failure.get("query") or "").strip()
            if not page_id or not query:
                outcome = _schedule_retry(
                    entry,
                    now=now_utc,
                    max_attempts=max_attempts,
                    retry_base_seconds=retry_base_seconds,
                    max_backoff_seconds=max_backoff_seconds,
                    error="not-in-top-results is missing page_id or query",
                )
            elif dry_run:
                if not _target_page_exists(page_id):
                    outcome = _schedule_retry(
                        entry,
                        now=now_utc,
                        max_attempts=max_attempts,
                        retry_base_seconds=retry_base_seconds,
                        max_backoff_seconds=max_backoff_seconds,
                        error=f"query hint target page does not exist: {page_id!r}",
                    )
                else:
                    outcome = (
                        "already_present"
                        if _matching_hint(page_id, query, path=hints_path)
                        else "applied"
                    )
                    entry["status"] = "applied"
            else:
                try:
                    outcome, hint = _ensure_query_hint(entry, hints_file=hints_path)
                    action["hint"] = hint
                    if outcome == "already_present" and int(entry.get("reopen_count") or 0) > 0:
                        outcome = _schedule_retry(
                            entry,
                            now=now_utc,
                            max_attempts=max_attempts,
                            retry_base_seconds=retry_base_seconds,
                            max_backoff_seconds=max_backoff_seconds,
                            error="read-back miss persisted after exact query hint was applied",
                        )
                    else:
                        entry["status"] = "applied"
                        entry["application"] = outcome
                        entry["applied_at"] = now_utc.isoformat(timespec="seconds")
                        entry["resolved_occurrences"] = int(entry.get("occurrences") or 0)
                        entry["resolved_last_seen"] = str(entry.get("last_seen") or "")
                        entry.pop("next_attempt_at", None)
                except Exception as exc:
                    if _human_required({"reason": reason, "error": str(exc)}):
                        outcome = "human_required"
                        entry["status"] = outcome
                        entry["last_error"] = str(exc)[:500]
                        entry["human_required_at"] = now_utc.isoformat(timespec="seconds")
                    else:
                        outcome = _schedule_retry(
                            entry,
                            now=now_utc,
                            max_attempts=max_attempts,
                            retry_base_seconds=retry_base_seconds,
                            max_backoff_seconds=max_backoff_seconds,
                            error=str(exc),
                        )
        else:
            outcome = _schedule_retry(
                entry,
                now=now_utc,
                max_attempts=max_attempts,
                retry_base_seconds=retry_base_seconds,
                max_backoff_seconds=max_backoff_seconds,
                error=str(failure.get("error") or reason),
            )

        counts[outcome] += 1
        dry_run_outcomes = {
            "applied": "would_apply",
            "already_present": "would_mark_applied",
            "retry_scheduled": "would_schedule_retry",
            "quarantined": "would_quarantine",
            "human_required": "would_require_human",
        }
        action["outcome"] = dry_run_outcomes[outcome] if dry_run else outcome
        actions.append(action)

    waiting = sum(
        1
        for entry in entries.values()
        if isinstance(entry, dict)
        and str(entry.get("status") or "") == "retry_wait"
        and not _due(entry, now=now_utc)
    )
    terminal = sum(
        1
        for entry in entries.values()
        if isinstance(entry, dict)
        and str(entry.get("status") or "") in TERMINAL_STATUSES
    )
    if not dry_run and (budget is None or mutation_consumed):
        ledger_payload = {
            "schema_version": SCHEMA_VERSION,
            "updated_at": now_utc.isoformat(timespec="seconds"),
            "source_file": str(failure_file),
            "entries": entries,
        }
        _atomic_write_json(ledger_file, ledger_payload)

    return {
        "status": "budget_deferred" if counts["budget_deferred"] else "ok",
        "dry_run": dry_run,
        "failure_file": str(failure_file),
        "ledger_file": str(ledger_file),
        "source_lines": source_lines,
        "source_records": len(rows),
        "observed_failures": sum(int(row.get("occurrences") or 0) for row in observed.values()),
        "unique_failures": len(entries),
        "eligible": len(candidates),
        "processed": len(actions) - counts["budget_deferred"],
        "deferred_by_limit": max(0, len(candidates) - max_items),
        "waiting_for_retry": waiting,
        "terminal": terminal,
        **counts,
        "actions": actions,
    }


repair_read_back_failures = run_read_back_repair
