"""Autonomous repair lane for ingest read-back failures.

The ingest pipeline records failures as append-only JSONL events.  This module
turns those events into an idempotent state machine: safe retrieval misses
become exact query hints, transient failures back off and eventually enter
quarantine, and only external access/billing failures require a human.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from chronovisor import recall_hints, store as chronovisor_store
from chronovisor.convergence import HUMAN_REQUIRED_FAILURE_CLASSES
from chronovisor import decision_authority
from chronovisor.durable_state import atomic_write_bytes, canonical_bytes
from chronovisor.frontmatter import parse as parse_frontmatter
from chronovisor.page_mutation import decision_authority_lock, chronovisor_mutation_lock
from chronovisor.runtime_config import runtime_repo_root
from chronovisor.semantic_hold import (
    LOCAL_SEMANTIC_NO_QUORUM,
    build_semantic_no_quorum_hold,
    canonical_sha256,
    frontier_failure_class,
    is_local_semantic_no_quorum,
    persisted_semantic_no_quorum_hold,
    semantic_no_quorum_hold_error,
)
from chronovisor.read_back_integrity import (
    scan_jsonl_prefix,
    verify_prior_prefix,
)


FAILURE_FILE = chronovisor_store.CHRONOVISOR_ROOT / "runtime" / "ingest-read-back-failures.jsonl"
LEDGER_FILE = chronovisor_store.CHRONOVISOR_ROOT / "runtime" / "ingest-read-back-repair.json"
SCHEMA_VERSION = 2
TERMINAL_STATUSES = frozenset({"applied", "rejected", "quarantined", "human_required"})
DEFAULT_QUARANTINE_COOLDOWN_SECONDS = 6 * 60 * 60
PROJECT_ROOT = runtime_repo_root()
READ_BACK_EVIDENCE_POLICY_MARKER = "LLM_WIKI_READ_BACK_EVIDENCE_POLICY=1"
READ_BACK_DECISION_LANE = "read_back_repair"
TARGET_PAGE_TITLE_MAX_CHARS = 500
TARGET_PAGE_EXCERPT_MAX_CHARS = 8_000
TARGET_PAGE_RECALL_QUESTIONS_MAX_ITEMS = 20
TARGET_PAGE_RECALL_QUESTION_MAX_CHARS = 500
READ_BACK_FRONTIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "confidence", "summary"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["approved", "rejected", "needs_retry"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
    },
}
AUTH_REQUIRED_PATTERN = re.compile(
    r"\b401\b.{0,20}\bunauthori[sz]ed\b"
    r"|\b(?:api[-_ ]?key|authentication|authorization|oauth|credentials?)\b"
    r".{0,40}\b(?:denied|expired|invalid|missing|required|revoked|unauthori[sz]ed)\b"
    r"|\b(?:denied|expired|invalid|missing|required|revoked|unauthori[sz]ed)\b"
    r".{0,40}\b(?:api[-_ ]?key|authentication|authorization|oauth|credentials?)\b",
    re.IGNORECASE,
)
BILLING_REQUIRED_PATTERN = re.compile(
    r"\b(?:billing|payment required|quota(?: exceeded)?)\b",
    re.IGNORECASE,
)
SECRET_PERMISSION_PATTERN = re.compile(
    r"(?:\b(?:keychain|secret store|secret service|credential store|credential helper)\b"
    r".{0,60}\b(?:access denied|denied|not permitted|permission|required permission)\b)"
    r"|(?:\b(?:access denied|denied|not permitted|permission|required permission)\b"
    r".{0,60}\b(?:keychain|secret store|secret service|credential store|credential helper)\b)",
    re.IGNORECASE,
)
TRANSIENT_OPERATIONAL_PATTERN = re.compile(
    r"\b(?:"
    r"temporary|temporarily|timeout|timed out|unavailable|overloaded|try again|"
    r"connection reset|connection refused|connection aborted|network|"
    r"rate limit(?:ed)?|too many requests|502|503|504"
    r")\b",
    re.IGNORECASE,
)
EXHAUSTED_QUERY_HINT_ERROR = (
    "read-back miss persisted after exact query hint was applied"
)
UNVERIFIABLE_QUERY_HINT_PATTERN = re.compile(
    r"(?:"
    r"available workspace evidence does not include the target page|"
    r"query hint target page does not exist|"
    r"target page no longer exists"
    r")",
    re.IGNORECASE,
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
    encoded = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return f"read-back-{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:24]}"


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    scan = scan_jsonl_prefix(path)
    return list(scan.records), scan.complete_lines


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
    except FileNotFoundError:
        return _empty_ledger()
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("read-back derived view is unreadable") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), dict):
        raise RuntimeError("read-back derived view is malformed")
    if payload.get("schema_version") == SCHEMA_VERSION:
        if "view_sha256" not in payload:
            # Pre-v2 ledgers were sometimes stamped with the running schema
            # constant by callers but contained only the durable entry map.
            # They have no derived-source cursor to authenticate, so accept
            # that exact legacy shape once and seal it on the next write.  A
            # cursor-bearing v2 view without its seal remains fail-closed.
            legacy_keys = {"schema_version", "entries"}
            if set(payload) - legacy_keys:
                raise RuntimeError("read-back derived view seal is missing")
            return payload
        observed = payload.get("view_sha256")
        unsigned = {key: value for key, value in payload.items() if key != "view_sha256"}
        expected = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
        if observed != expected:
            raise RuntimeError("read-back derived view seal mismatch")
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "view_sha256"}
    sealed = {
        **unsigned,
        "view_sha256": hashlib.sha256(canonical_bytes(unsigned)).hexdigest(),
    }
    atomic_write_bytes(path, canonical_bytes(sealed), backup=True)


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
        existing["first_seen"] = (
            existing.get("first_seen") or aggregate.get("first_seen") or ""
        )
        existing["last_seen"] = (
            aggregate.get("last_seen") or existing.get("last_seen") or ""
        )
        existing["occurrences"] = max(
            previous_occurrences,
            int(aggregate.get("occurrences") or 0),
        )
        terminal_status = str(existing.get("status") or "")
        if terminal_status in {"applied", "rejected"}:
            resolved_occurrences = int(
                existing.get("resolved_occurrences") or previous_occurrences
            )
            resolved_last_seen = str(
                existing.get("resolved_last_seen") or previous_last_seen
            )
            observed_time = _parse_time(aggregate.get("last_seen"))
            resolved_time = _parse_time(resolved_last_seen)
            newly_observed = int(
                aggregate.get("occurrences") or 0
            ) > resolved_occurrences or (
                observed_time is not None
                and (resolved_time is None or observed_time > resolved_time)
            )
            if newly_observed:
                existing["status"] = "pending"
                existing["attempts"] = 0
                existing["reopen_count"] = int(existing.get("reopen_count") or 0) + 1
                existing["reopened_at"] = aggregate.get("last_seen") or datetime.now(
                    timezone.utc
                ).isoformat(timespec="seconds")
                existing.pop("next_attempt_at", None)
                if terminal_status == "rejected":
                    existing.pop("frontier_review", None)
                    existing.pop("frontier_proposal_fingerprint", None)
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
    return bool(
        AUTH_REQUIRED_PATTERN.search(text)
        or BILLING_REQUIRED_PATTERN.search(text)
        or SECRET_PERMISSION_PATTERN.search(text)
    )


def _transient_operational_failure(failure: dict[str, Any]) -> bool:
    reason = _canonical_reason(failure.get("reason"))
    if reason not in {"search-error", "read-back-unavailable"}:
        return False
    text = " ".join(
        str(failure.get(field) or "") for field in ("error", "message", "detail")
    )
    return bool(TRANSIENT_OPERATIONAL_PATTERN.search(text))


def _should_queue_operational_self_heal(
    failure: dict[str, Any],
    entry: dict[str, Any],
) -> bool:
    if _transient_operational_failure(failure):
        return False
    if _canonical_reason(failure.get("reason")) == "not-in-top-results":
        last_error = str(entry.get("last_error") or "")
        if EXHAUSTED_QUERY_HINT_ERROR in last_error:
            return False
        if UNVERIFIABLE_QUERY_HINT_PATTERN.search(last_error):
            return False
    return True


def _due(entry: dict[str, Any], *, now: datetime) -> bool:
    if str(entry.get("status") or "pending") in {"pending", "frontier_approved"}:
        return True
    if str(entry.get("status") or "") != "retry_wait":
        return False
    next_attempt = _parse_time(entry.get("next_attempt_at"))
    return next_attempt is None or next_attempt <= now


def _resume_due_quarantines(
    entries: dict[str, dict[str, Any]],
    *,
    now: datetime,
    cooldown_seconds: int,
) -> int:
    """Reset non-human quarantine attempt budgets after a bounded cooldown."""
    resumed = 0
    for entry in entries.values():
        if (
            not isinstance(entry, dict)
            or str(entry.get("status") or "") != "quarantined"
        ):
            continue
        # A local semantic disagreement is an exact-epoch terminal hold, not
        # an operational quarantine.  Even a malformed/legacy marker fails
        # closed here; the lane may only reconsider it after reconstructing
        # and comparing the current epoch below.
        if _has_semantic_no_quorum_marker(entry):
            continue
        failure = entry.get("failure") if isinstance(entry.get("failure"), dict) else {}
        if _human_required(failure):
            entry["status"] = "human_required"
            entry.setdefault("human_required_at", now.isoformat(timespec="seconds"))
            continue
        retry_at = _parse_time(entry.get("quarantine_retry_at"))
        if retry_at is None:
            quarantined_at = _parse_time(
                entry.get("quarantined_at") or entry.get("last_attempt_at")
            )
            retry_at = (
                quarantined_at + timedelta(seconds=max(0, cooldown_seconds))
                if quarantined_at is not None
                else now
            )
        if retry_at > now:
            continue
        entry["status"] = "pending"
        entry["attempts"] = 0
        entry["resumed_at"] = now.isoformat(timespec="seconds")
        entry["quarantine_resume_count"] = (
            int(entry.get("quarantine_resume_count") or 0) + 1
        )
        entry.pop("next_attempt_at", None)
        entry.pop("quarantine_retry_at", None)
        resumed += 1
    return resumed


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
    entry["next_attempt_at"] = (now + timedelta(seconds=delay)).isoformat(
        timespec="seconds"
    )
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
        recall_hints.chronovisor_store.find_page(page_id) is not None
        or (recall_hints.chronovisor_store.SYSTEM_DIR / f"{page_id}.md").exists()
    )


def _target_page_path(page_id: str) -> Path | None:
    path = recall_hints.chronovisor_store.find_page(page_id)
    if path is not None:
        return path
    system_path = recall_hints.chronovisor_store.SYSTEM_DIR / f"{page_id}.md"
    return system_path if system_path.exists() else None


def _target_page_snapshot(page_id: str) -> dict[str, Any]:
    """Return bounded page evidence and a host-generated content binding."""

    path = _target_page_path(page_id)
    if path is None:
        return {
            "status": "missing",
            "content_hash": None,
            "title": None,
            "recall_questions": [],
            "body_excerpt": "",
            "body_truncated": False,
        }
    try:
        content = path.read_bytes()
    except OSError:
        return {
            "status": "unreadable",
            "content_hash": None,
            "title": None,
            "recall_questions": [],
            "body_excerpt": "",
            "body_truncated": False,
        }

    content_hash = hashlib.sha256(content).hexdigest()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return {
            "status": "unreadable",
            "content_hash": content_hash,
            "title": None,
            "recall_questions": [],
            "body_excerpt": "",
            "body_truncated": False,
        }

    meta, body = parse_frontmatter(text)
    raw_questions = meta.get("recall_questions")
    recall_questions = (
        [
            str(question)[:TARGET_PAGE_RECALL_QUESTION_MAX_CHARS]
            for question in raw_questions[:TARGET_PAGE_RECALL_QUESTIONS_MAX_ITEMS]
            if str(question).strip()
        ]
        if isinstance(raw_questions, list)
        else []
    )
    return {
        "status": "ok",
        "content_hash": content_hash,
        "title": str(meta.get("title") or "").strip()[:TARGET_PAGE_TITLE_MAX_CHARS]
        or None,
        "recall_questions": recall_questions,
        "body_excerpt": body[:TARGET_PAGE_EXCERPT_MAX_CHARS],
        "body_truncated": len(body) > TARGET_PAGE_EXCERPT_MAX_CHARS,
    }


def _target_page_hash(page_id: str) -> str:
    snapshot = _target_page_snapshot(page_id)
    content_hash = snapshot.get("content_hash")
    return str(content_hash) if content_hash else str(snapshot["status"])


def _query_hint_proposal(entry: dict[str, Any]) -> dict[str, Any]:
    failure = entry["failure"]
    page_id = str(failure.get("page_id") or "").strip()
    query = str(failure.get("query") or "").strip()
    target_snapshot = _target_page_snapshot(page_id)
    return {
        "kind": "query_hint",
        "failure_key": str(entry.get("failure_key") or failure_key(failure)),
        "page_id": page_id,
        "query": query,
        "query_key": recall_hints.normalize_query_text(query),
        "target_page_hash": target_snapshot.get("content_hash")
        or target_snapshot["status"],
        "target_snapshot": target_snapshot,
        "reason": "ingest read-back not-in-top-results",
    }


def _proposal_fingerprint(proposal: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(proposal), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _has_semantic_no_quorum_marker(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    if "semantic_hold" in value:
        return True
    if value.get("last_failure_class") == LOCAL_SEMANTIC_NO_QUORUM:
        return True
    return frontier_failure_class(value) == LOCAL_SEMANTIC_NO_QUORUM


def _query_hint_semantic_epoch(
    proposal: Mapping[str, Any],
    proposal_fingerprint: str,
) -> dict[str, Any]:
    """Return the redacted identity that may legitimately reopen a hold."""

    return {
        "ledger_schema_version": SCHEMA_VERSION,
        "proposal_sha256": proposal_fingerprint,
        "target_page_sha256": str(proposal.get("target_page_hash") or ""),
        "review_schema_sha256": canonical_sha256(READ_BACK_FRONTIER_SCHEMA),
    }


def _semantic_hold_state(
    entry: Mapping[str, Any],
    *,
    epoch: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    """Classify a durable marker without treating corruption as retryable."""

    if not _has_semantic_no_quorum_marker(entry):
        return "none", None
    hold = persisted_semantic_no_quorum_hold(entry, lane=READ_BACK_DECISION_LANE)
    if hold is None:
        return "malformed", None
    error = semantic_no_quorum_hold_error(
        hold,
        READ_BACK_DECISION_LANE,
        epoch=epoch,
        authority=authority,
    )
    if error is None:
        return "same", hold
    if error in {"semantic hold epoch changed", "semantic hold authority changed"}:
        history = entry.get("semantic_hold_history")
        if isinstance(history, list):
            for candidate in reversed(history):
                historical = persisted_semantic_no_quorum_hold(
                    candidate,
                    lane=READ_BACK_DECISION_LANE,
                    epoch=epoch,
                    authority=authority,
                )
                if historical is not None:
                    return "same", historical
        return "changed", hold
    return "malformed", None


def _apply_semantic_hold(
    entry: dict[str, Any],
    *,
    hold: Mapping[str, Any] | None,
    now: datetime,
    malformed: bool = False,
) -> None:
    entry["status"] = "quarantined"
    entry["last_failure_class"] = LOCAL_SEMANTIC_NO_QUORUM
    entry["last_error"] = (
        "malformed local semantic no-quorum hold; refusing resample"
        if malformed
        else "local semantic models did not reach a safe quorum"
    )
    entry["quarantined_at"] = entry.get("quarantined_at") or now.isoformat(
        timespec="seconds"
    )
    entry.pop("next_attempt_at", None)
    entry.pop("quarantine_retry_at", None)
    entry.pop("frontier_review", None)
    entry.pop("frontier_review_authority", None)
    entry.pop("frontier_proposal_fingerprint", None)
    if hold is not None:
        existing = persisted_semantic_no_quorum_hold(
            entry, lane=READ_BACK_DECISION_LANE
        )
        if existing is not None and existing.get("hold_sha256") != hold.get(
            "hold_sha256"
        ):
            history = [
                item
                for item in entry.get("semantic_hold_history", [])
                if isinstance(item, Mapping)
            ]
            if not any(
                item.get("hold_sha256") == existing.get("hold_sha256")
                for item in history
            ):
                history.append(existing)
            # Hold identities are durable evidence, not a retry cache.  Keep
            # every distinct epoch/authority so an A -> ... -> A transition
            # can restore the original terminal hold without another model
            # call, regardless of how many authorities were adopted between.
            entry["semantic_hold_history"] = history
        entry["semantic_hold"] = dict(hold)


def _normalize_frontier_review(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {
            "decision": "needs_retry",
            "confidence": 0.0,
            "summary": "frontier result is not an object",
            "valid": False,
        }
    review = dict(value)
    decision = review.get("decision")
    confidence = review.get("confidence")
    summary = review.get("summary")
    errors: list[str] = []
    if decision not in {"approved", "rejected", "needs_retry"}:
        errors.append("invalid decision")
        decision = "needs_retry"
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        errors.append("confidence must be numeric")
        confidence = 0.0
    else:
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            errors.append("confidence outside [0, 1]")
            confidence = 0.0
    if not isinstance(summary, str) or not summary.strip():
        errors.append("summary is required")
        summary = "frontier result is missing a summary"
    return {
        **review,
        "decision": decision if not errors else "needs_retry",
        "confidence": confidence,
        "summary": str(summary).strip(),
        "valid": not errors,
        "validation_errors": errors,
    }


def _current_query_hint_authority(
    *, reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve the exact lane epoch allowed to install a query hint.

    A callable reviewer is an explicit dependency-injection boundary used by
    tests and integrations.  Production reviews must instead bind the enabled
    lane, both contract manifests, adopted artifact digest, and model triplet.
    """

    return decision_authority.current_semantic_authority(
        READ_BACK_DECISION_LANE,
        injected_reviewer=reviewer is not None,
    )


def _query_hint_authority_error(
    review: object,
    authority: object,
) -> str | None:
    return decision_authority.semantic_verdict_authority_error(
        review,
        authority,
        lane=READ_BACK_DECISION_LANE,
    )


def _review_query_hint(
    proposal: dict[str, Any],
    *,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> dict[str, Any]:
    if reviewer is not None:
        return _normalize_frontier_review(reviewer(proposal))
    from chronovisor.decision_lane_prompts import build_read_back_repair_request
    from chronovisor.frontier_review import run_structured_review

    prompt, system = build_read_back_repair_request(
        proposal,
        evidence_policy_marker=READ_BACK_EVIDENCE_POLICY_MARKER,
    )
    return _normalize_frontier_review(
        run_structured_review(
            prompt,
            READ_BACK_FRONTIER_SCHEMA,
            repo_root=PROJECT_ROOT,
            decision_lane="read_back_repair",
            system=system,
        )
    )


def _ensure_query_hint(
    entry: dict[str, Any],
    *,
    hints_file: Path,
    expected_target_hash: str,
) -> tuple[str, dict[str, Any]]:
    failure = entry["failure"]
    page_id = str(failure.get("page_id") or "").strip()
    query = str(failure.get("query") or "").strip()
    if not _target_page_exists(page_id):
        raise ValueError(f"query hint target page does not exist: {page_id!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_target_hash):
        raise ValueError("query hint review has no valid target page hash")
    # Lock order is Wiki mutation -> query-hint lock (inside add_query_hint).
    # The recall-hints writer never acquires the Wiki lock while holding its
    # own lock, so this has no inverse order. Keeping the shared Wiki lock from
    # the hash check through the durable hint write closes the final TOCTOU
    # window against every cooperating page writer.
    with chronovisor_mutation_lock():
        existing = _matching_hint(page_id, query, path=hints_file)
        current_snapshot = _target_page_snapshot(page_id)
        current_hash = current_snapshot.get("content_hash")
        if (
            current_snapshot.get("status") != "ok"
            or current_hash != expected_target_hash
        ):
            raise ValueError("query hint target page changed after review")
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


def _target_meta_present(page_id: str) -> bool:
    from chronovisor.index_store import get_store

    store = get_store()
    store.refresh()
    return store.meta(page_id) is not None


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
    quarantine_cooldown_seconds: int = DEFAULT_QUARANTINE_COOLDOWN_SECONDS,
    budget: Any | None = None,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    frontier_confidence_threshold: float | None = None,
) -> dict[str, Any]:
    """Process a bounded batch of read-back failures.

    Dry runs read the source, ledger, and hint store but never create or modify
    any file. Human-required entries remain terminal; non-human quarantine
    resets its attempt budget after a bounded cooldown. The exact hint lookup
    also protects against a crash after hint write but before the ledger's
    atomic replace.
    """
    now_utc = _as_utc(now)
    hints_path = hints_file or recall_hints.QUERY_HINTS_FILE
    source_scan = scan_jsonl_prefix(failure_file)
    rows = list(source_scan.records)
    source_lines = source_scan.complete_lines
    source_cursor = source_scan.cursor(source_file=failure_file)
    if not source_scan.valid:
        return {
            "status": "source_integrity_error",
            "dry_run": dry_run,
            "failure_file": str(failure_file),
            "ledger_file": str(ledger_file),
            "source_cursor": source_cursor,
            "source_lines": source_lines,
            "source_records": len(rows),
            "processed": 0,
            "actions": [],
        }
    observed = _flatten_failures(rows)
    try:
        original_ledger = _load_ledger(ledger_file)
    except RuntimeError as exc:
        return {
            "status": "ledger_integrity_error",
            "dry_run": dry_run,
            "failure_file": str(failure_file),
            "ledger_file": str(ledger_file),
            "source_cursor": source_cursor,
            "error": str(exc),
            "processed": 0,
            "actions": [],
        }
    prior_cursor = original_ledger.get("source_cursor")
    if isinstance(prior_cursor, dict) and not verify_prior_prefix(
        failure_file,
        prior_cursor,
    ):
        return {
            "status": "source_history_rewritten",
            "dry_run": dry_run,
            "failure_file": str(failure_file),
            "ledger_file": str(ledger_file),
            "source_cursor": source_cursor,
            "processed": 0,
            "actions": [],
        }
    entries = _merge_entries(original_ledger, observed)
    last_persisted_sha256: str | None = None

    def persist_ledger() -> None:
        nonlocal last_persisted_sha256
        if not verify_prior_prefix(failure_file, source_cursor):
            raise RuntimeError(
                "canonical read-back source changed before derived view publication"
            )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "updated_at": now_utc.isoformat(timespec="seconds"),
            "source_file": str(failure_file),
            "source_cursor": source_cursor,
            "entries": entries,
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        if digest == last_persisted_sha256:
            return
        _atomic_write_json(ledger_file, payload)
        last_persisted_sha256 = digest

    max_items = max(0, int(max_items))
    max_attempts = max(1, int(max_attempts))
    retry_base_seconds = max(1, int(retry_base_seconds))
    max_backoff_seconds = max(retry_base_seconds, int(max_backoff_seconds))
    quarantine_cooldown_seconds = max(0, int(quarantine_cooldown_seconds))
    # Deprecated compatibility input. Consensus confidence is diagnostic only;
    # the schema-valid decision and deterministic read-back evidence authorize
    # the action.
    del frontier_confidence_threshold
    resumed_quarantined = _resume_due_quarantines(
        entries,
        now=now_utc,
        cooldown_seconds=quarantine_cooldown_seconds,
    )
    candidates = [
        entry
        for entry in entries.values()
        if isinstance(entry, dict)
        and (
            _has_semantic_no_quorum_marker(entry)
            or (
                str(entry.get("status") or "pending") not in TERMINAL_STATUSES
                and _due(entry, now=now_utc)
            )
        )
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
        "rejected": 0,
        "frontier_review": 0,
        "semantic_hold": 0,
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

        prepared_proposal: dict[str, Any] | None = None
        prepared_fingerprint: str | None = None
        prepared_authority: dict[str, Any] | None = None
        if _has_semantic_no_quorum_marker(entry):
            page_id = str(failure.get("page_id") or "").strip()
            query = str(failure.get("query") or "").strip()
            if (
                reason != "not-in-top-results"
                or not page_id
                or not query
                or not _target_page_exists(page_id)
            ):
                _apply_semantic_hold(entry, hold=None, now=now_utc, malformed=True)
                counts["semantic_hold"] += 1
                action["outcome"] = "semantic_hold_malformed"
                actions.append(action)
                continue
            prepared_proposal = _query_hint_proposal(entry)
            prepared_fingerprint = _proposal_fingerprint(prepared_proposal)
            prepared_authority, prepared_authority_error = (
                _current_query_hint_authority(reviewer=reviewer)
            )
            if prepared_authority_error is not None or prepared_authority is None:
                # Authority cannot be compared, so the previously durable hold
                # remains authoritative.  This is not an operational retry.
                existing_hold = persisted_semantic_no_quorum_hold(
                    entry, lane=READ_BACK_DECISION_LANE
                )
                _apply_semantic_hold(entry, hold=existing_hold, now=now_utc)
                counts["semantic_hold"] += 1
                action["outcome"] = "semantic_hold_authority_unavailable"
                actions.append(action)
                continue
            hold_state, existing_hold = _semantic_hold_state(
                entry,
                epoch=_query_hint_semantic_epoch(
                    prepared_proposal, prepared_fingerprint
                ),
                authority=prepared_authority,
            )
            if hold_state in {"same", "malformed"}:
                _apply_semantic_hold(
                    entry,
                    hold=existing_hold,
                    now=now_utc,
                    malformed=hold_state == "malformed",
                )
                counts["semantic_hold"] += 1
                action["outcome"] = (
                    "semantic_hold"
                    if hold_state == "same"
                    else "semantic_hold_malformed"
                )
                actions.append(action)
                continue

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
        elif reason == "empty-query":
            outcome = "rejected"
            entry["status"] = outcome
            entry["last_error"] = (
                "empty-query read-back failure has no repairable query"
            )
            entry["rejected_at"] = now_utc.isoformat(timespec="seconds")
            entry["resolved_occurrences"] = int(entry.get("occurrences") or 0)
            entry["resolved_last_seen"] = str(entry.get("last_seen") or "")
        elif reason == "missing-meta":
            page_id = str(failure.get("page_id") or "").strip()
            if not page_id:
                outcome = "rejected"
                entry["status"] = outcome
                entry["last_error"] = "missing-meta failure is missing page_id"
                entry["rejected_at"] = now_utc.isoformat(timespec="seconds")
                entry["resolved_occurrences"] = int(entry.get("occurrences") or 0)
                entry["resolved_last_seen"] = str(entry.get("last_seen") or "")
            elif not _target_page_exists(page_id):
                outcome = "rejected"
                entry["status"] = outcome
                entry["last_error"] = (
                    f"missing-meta target page no longer exists: {page_id!r}"
                )
                entry["rejected_at"] = now_utc.isoformat(timespec="seconds")
                entry["resolved_occurrences"] = int(entry.get("occurrences") or 0)
                entry["resolved_last_seen"] = str(entry.get("last_seen") or "")
            elif dry_run:
                outcome = "retry_scheduled"
            else:
                try:
                    meta_present = _target_meta_present(page_id)
                except Exception as exc:
                    outcome = _schedule_retry(
                        entry,
                        now=now_utc,
                        max_attempts=max_attempts,
                        retry_base_seconds=retry_base_seconds,
                        max_backoff_seconds=max_backoff_seconds,
                        error=f"missing-meta refresh failed: {exc}",
                    )
                else:
                    if meta_present:
                        outcome = "already_present"
                        entry["status"] = "applied"
                        entry["application"] = "metadata_present"
                        entry["applied_at"] = now_utc.isoformat(timespec="seconds")
                        entry["resolved_occurrences"] = int(
                            entry.get("occurrences") or 0
                        )
                        entry["resolved_last_seen"] = str(entry.get("last_seen") or "")
                        entry.pop("next_attempt_at", None)
                    else:
                        outcome = _schedule_retry(
                            entry,
                            now=now_utc,
                            max_attempts=max_attempts,
                            retry_base_seconds=retry_base_seconds,
                            max_backoff_seconds=max_backoff_seconds,
                            error="missing-meta target page still absent from index",
                        )
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
            elif not _target_page_exists(page_id):
                outcome = _schedule_retry(
                    entry,
                    now=now_utc,
                    max_attempts=max_attempts,
                    retry_base_seconds=retry_base_seconds,
                    max_backoff_seconds=max_backoff_seconds,
                    error=f"query hint target page does not exist: {page_id!r}",
                )
            elif dry_run:
                outcome = "frontier_review"
            else:
                proposal = prepared_proposal or _query_hint_proposal(entry)
                proposal_fingerprint = prepared_fingerprint or _proposal_fingerprint(
                    proposal
                )
                if prepared_authority is not None:
                    review_authority, authority_error = prepared_authority, None
                else:
                    review_authority, authority_error = _current_query_hint_authority(
                        reviewer=reviewer
                    )
                persisted_review = entry.get("frontier_review")
                persisted_authority = entry.get("frontier_review_authority")
                review = (
                    _normalize_frontier_review(persisted_review)
                    if isinstance(persisted_review, dict)
                    and entry.get("frontier_proposal_fingerprint")
                    == proposal_fingerprint
                    and authority_error is None
                    and decision_authority.compare_semantic_authority(
                        persisted_authority,
                        review_authority,
                        lane=READ_BACK_DECISION_LANE,
                    )
                    is None
                    and _query_hint_authority_error(
                        persisted_review,
                        persisted_authority,
                    )
                    is None
                    else None
                )
                if review is None:
                    entry.pop("frontier_review", None)
                    entry.pop("frontier_proposal_fingerprint", None)
                    entry.pop("frontier_review_authority", None)
                    if isinstance(persisted_review, dict):
                        action["frontier_review_stale"] = True
                    if authority_error is not None or review_authority is None:
                        outcome = _schedule_retry(
                            entry,
                            now=now_utc,
                            max_attempts=max_attempts,
                            retry_base_seconds=retry_base_seconds,
                            max_backoff_seconds=max_backoff_seconds,
                            error=authority_error
                            or "query hint review authority is missing",
                        )
                        action["outcome"] = outcome
                        counts[outcome] += 1
                        actions.append(action)
                        continue
                    if budget is not None:
                        allowed, budget_reason = budget.consume("frontier")
                        if not allowed:
                            counts["budget_deferred"] += 1
                            action["outcome"] = "budget_deferred"
                            action["budget_reason"] = budget_reason
                            actions.append(action)
                            continue
                    try:
                        review = _review_query_hint(proposal, reviewer=reviewer)
                    except Exception as exc:
                        review = {
                            "decision": "needs_retry",
                            "confidence": 0.0,
                            "summary": f"{exc.__class__.__name__}: {exc}",
                            "valid": False,
                        }
                    current_authority, current_authority_error = (
                        _current_query_hint_authority(reviewer=reviewer)
                    )
                    authority_change_error = (
                        current_authority_error
                        or decision_authority.compare_semantic_authority(
                            review_authority,
                            current_authority,
                            lane=READ_BACK_DECISION_LANE,
                        )
                        or _query_hint_authority_error(review, review_authority)
                    )
                    if authority_change_error is not None:
                        review = {
                            **review,
                            "decision": "needs_retry",
                            "summary": authority_change_error,
                            "valid": False,
                        }
                    action["frontier_reviewed"] = True
                else:
                    action["frontier_review_reused"] = True

                if is_local_semantic_no_quorum(review):
                    hold: dict[str, Any] | None = None
                    hold_error: str | None = None
                    try:
                        with decision_authority_lock():
                            current_authority, current_authority_error = (
                                _current_query_hint_authority(reviewer=reviewer)
                            )
                            hold_error = (
                                current_authority_error
                                or decision_authority.compare_semantic_authority(
                                    review_authority,
                                    current_authority,
                                    lane=READ_BACK_DECISION_LANE,
                                )
                            )
                            current_proposal = _query_hint_proposal(entry)
                            if (
                                hold_error is None
                                and _proposal_fingerprint(current_proposal)
                                != proposal_fingerprint
                            ):
                                hold_error = (
                                    "query hint proposal changed before semantic hold"
                                )
                            if hold_error is None:
                                assert review_authority is not None
                                hold = build_semantic_no_quorum_hold(
                                    READ_BACK_DECISION_LANE,
                                    _query_hint_semantic_epoch(
                                        proposal, proposal_fingerprint
                                    ),
                                    review_authority,
                                    review,
                                )
                                _apply_semantic_hold(entry, hold=hold, now=now_utc)
                                persist_ledger()
                    except (TypeError, ValueError) as exc:
                        hold_error = str(exc)
                    if hold is not None:
                        outcome = "semantic_hold"
                        counts[outcome] += 1
                        action["outcome"] = outcome
                        action["semantic_hold_sha256"] = hold["hold_sha256"]
                        actions.append(action)
                        continue
                    review = {
                        **review,
                        "decision": "needs_retry",
                        "summary": hold_error
                        or "semantic no-quorum hold provenance is invalid",
                        "valid": False,
                    }

                decision = review.get("decision")
                if decision == "rejected":
                    try:
                        # Rejection is terminal state, so it observes the same
                        # final authority boundary as an applied hint.
                        with decision_authority_lock():
                            current_authority, current_authority_error = (
                                _current_query_hint_authority(reviewer=reviewer)
                            )
                            terminal_authority_error = (
                                current_authority_error
                                or decision_authority.compare_semantic_authority(
                                    review_authority,
                                    current_authority,
                                    lane=READ_BACK_DECISION_LANE,
                                )
                                or _query_hint_authority_error(
                                    review,
                                    review_authority,
                                )
                            )
                            if terminal_authority_error is not None:
                                raise ValueError(terminal_authority_error)
                            outcome = "rejected"
                            entry["status"] = outcome
                            entry["frontier_review"] = review
                            entry["frontier_proposal_fingerprint"] = (
                                proposal_fingerprint
                            )
                            entry["frontier_review_authority"] = review_authority
                            entry.pop("semantic_hold", None)
                            entry.pop("semantic_hold_history", None)
                            entry.pop("last_failure_class", None)
                            entry["rejected_at"] = now_utc.isoformat(timespec="seconds")
                            entry["resolved_occurrences"] = int(
                                entry.get("occurrences") or 0
                            )
                            entry["resolved_last_seen"] = str(
                                entry.get("last_seen") or ""
                            )
                            persist_ledger()
                    except Exception as exc:
                        outcome = _schedule_retry(
                            entry,
                            now=now_utc,
                            max_attempts=max_attempts,
                            retry_base_seconds=retry_base_seconds,
                            max_backoff_seconds=max_backoff_seconds,
                            error=str(exc),
                        )
                elif decision != "approved":
                    outcome = _schedule_retry(
                        entry,
                        now=now_utc,
                        max_attempts=max_attempts,
                        retry_base_seconds=retry_base_seconds,
                        max_backoff_seconds=max_backoff_seconds,
                        error=str(review.get("summary") or "frontier needs retry"),
                    )
                else:
                    # Persist the exact local-consensus verdict before the ranking
                    # artifact changes. A crash can then reuse the verdict and
                    # the hint writer's exact lookup makes the operation
                    # idempotent.
                    entry["status"] = "frontier_approved"
                    entry["frontier_review"] = review
                    entry["frontier_proposal_fingerprint"] = proposal_fingerprint
                    entry["frontier_review_authority"] = review_authority
                    entry.pop("semantic_hold", None)
                    entry.pop("semantic_hold_history", None)
                    entry.pop("last_failure_class", None)
                    entry["frontier_approved_at"] = now_utc.isoformat(
                        timespec="seconds"
                    )
                    persist_ledger()
                    try:
                        # Adoption/config writers use the same outer lease.
                        # Keep the authority epoch stable through the exact
                        # target-page CAS and durable hint write.
                        with decision_authority_lock():
                            current_authority, current_authority_error = (
                                _current_query_hint_authority(reviewer=reviewer)
                            )
                            effect_authority_error = (
                                current_authority_error
                                or decision_authority.compare_semantic_authority(
                                    review_authority,
                                    current_authority,
                                    lane=READ_BACK_DECISION_LANE,
                                )
                                or _query_hint_authority_error(
                                    review,
                                    review_authority,
                                )
                            )
                            if effect_authority_error is not None:
                                raise ValueError(effect_authority_error)
                            outcome, hint = _ensure_query_hint(
                                entry,
                                hints_file=hints_path,
                                expected_target_hash=str(
                                    proposal["target_snapshot"].get("content_hash")
                                    or ""
                                ),
                            )
                            action["hint"] = hint
                            if (
                                outcome == "already_present"
                                and int(entry.get("reopen_count") or 0) > 0
                            ):
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
                                entry["applied_at"] = now_utc.isoformat(
                                    timespec="seconds"
                                )
                                entry["resolved_occurrences"] = int(
                                    entry.get("occurrences") or 0
                                )
                                entry["resolved_last_seen"] = str(
                                    entry.get("last_seen") or ""
                                )
                                entry.pop("next_attempt_at", None)
                            # The hint write and its terminal/retry ledger
                            # disposition are one authority-epoch commit.  A
                            # crash may leave only the durable approved review
                            # or only the exact idempotent hint, both of which
                            # are safely recoverable on the next pass.
                            persist_ledger()
                    except Exception as exc:
                        if _human_required({"reason": reason, "error": str(exc)}):
                            outcome = "human_required"
                            entry["status"] = outcome
                            entry["last_error"] = str(exc)[:500]
                            entry["human_required_at"] = now_utc.isoformat(
                                timespec="seconds"
                            )
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

        if (
            outcome == "quarantined"
            and not dry_run
            and not entry.get("self_heal_packet_path")
            and _should_queue_operational_self_heal(failure, entry)
        ):
            from chronovisor.failure_supervisor import queue_operational_failure

            packet_path = queue_operational_failure(
                failure_class="read_back.repeated_miss",
                fingerprint=f"read_back.repeated_miss:{key}",
                message=(
                    "ingest read-back repair exhausted its bounded attempts: "
                    + str(entry.get("last_error") or reason)
                ),
                evidence={"failure": failure, "ledger_entry": entry},
                attempts=int(entry.get("attempts") or max_attempts),
                label=f"read-back-{str(failure.get('page_id') or key)}",
            )
            entry["self_heal_packet_path"] = str(packet_path)
            entry["self_heal_queued_at"] = now_utc.isoformat(timespec="seconds")
            action["self_heal_packet_path"] = str(packet_path)
        elif outcome == "quarantined":
            skipped_reason = None
            if _transient_operational_failure(failure):
                skipped_reason = "transient_operational_failure"
            elif reason == "not-in-top-results" and EXHAUSTED_QUERY_HINT_ERROR in str(
                entry.get("last_error") or ""
            ):
                skipped_reason = "exhausted_query_hint"
            elif (
                reason == "not-in-top-results"
                and UNVERIFIABLE_QUERY_HINT_PATTERN.search(
                    str(entry.get("last_error") or "")
                )
            ):
                skipped_reason = "unverifiable_query_hint"
            if skipped_reason is not None:
                entry["self_heal_skipped_reason"] = skipped_reason
                action["self_heal_skipped_reason"] = skipped_reason

        counts[outcome] += 1
        dry_run_outcomes = {
            "applied": "would_apply",
            "already_present": "would_mark_applied",
            "retry_scheduled": "would_schedule_retry",
            "quarantined": "would_quarantine",
            "human_required": "would_require_human",
            "rejected": "would_reject",
            "frontier_review": "would_request_frontier",
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
    waiting_in_quarantine = sum(
        1
        for entry in entries.values()
        if isinstance(entry, dict) and str(entry.get("status") or "") == "quarantined"
    )
    terminal = sum(
        1
        for entry in entries.values()
        if isinstance(entry, dict)
        and str(entry.get("status") or "") in TERMINAL_STATUSES
    )
    source_projection_changed = prior_cursor != source_cursor
    if not dry_run and (
        (source_projection_changed and budget is None)
        or budget is None
        or mutation_consumed
        or resumed_quarantined
    ):
        persist_ledger()

    return {
        "status": "budget_deferred" if counts["budget_deferred"] else "ok",
        "dry_run": dry_run,
        "failure_file": str(failure_file),
        "ledger_file": str(ledger_file),
        "source_lines": source_lines,
        "source_records": len(rows),
        "source_cursor": source_cursor,
        "observed_failures": sum(
            int(row.get("occurrences") or 0) for row in observed.values()
        ),
        "unique_failures": len(entries),
        "eligible": len(candidates),
        "processed": len(actions) - counts["budget_deferred"],
        "deferred_by_limit": max(0, len(candidates) - max_items),
        "waiting_for_retry": waiting,
        "waiting_in_quarantine": waiting_in_quarantine,
        "resumed_quarantined": resumed_quarantined,
        "terminal": terminal,
        **counts,
        "actions": actions,
    }


repair_read_back_failures = run_read_back_repair
