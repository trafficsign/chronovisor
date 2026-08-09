"""Bounded convergence state for autonomous maintenance lanes.

The individual maintenance lanes (lint, duplicate review, raw replay, recall
review, and self-heal) produce different payloads, but they need the same
control-plane guarantees:

* stable identities across queue rebuilds;
* at-most-one active lease per item;
* bounded local and frontier attempts with exponential backoff;
* terminal quarantine instead of an infinite retry loop;
* a narrow, deterministic boundary for genuinely human-required failures;
* per-cycle cost budgets; and
* byte-for-byte read-only dry runs.

This module deliberately does not know how to repair an item.  Callers merge a
candidate, claim a local/frontier attempt, then record success or failure.  The
state file is a compact latest-state projection; ``events.jsonl`` is the audit
trail.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from chronovisor.core import store as chronovisor_store
from chronovisor.core.timeutil import ensure_utc as _utc_now
from chronovisor.decision.semantic_hold import (
    LOCAL_SEMANTIC_NO_QUORUM,
    build_semantic_no_quorum_hold,
    persisted_semantic_no_quorum_hold,
)

SCHEMA_VERSION = 1

Stage = Literal["local", "frontier"]
BudgetKind = Literal["local", "frontier", "mutation", "raw_bytes"]

TERMINAL_STATUSES = frozenset({"applied", "rejected", "quarantined", "human_required"})
LOCAL_STATUSES = frozenset({"pending_local", "local_retry", "local_running"})
FRONTIER_STATUSES = frozenset(
    {"pending_frontier", "frontier_retry", "frontier_running"}
)

# This allowlist is intentionally narrow.  A model saying "ask a human" is
# not enough to enter human_required; the failure must be an external access
# or capability boundary classified by deterministic code.
HUMAN_REQUIRED_FAILURE_CLASSES = frozenset(
    {
        "auth_required",
        "oauth_required",
        "quota_or_billing_required",
        "keychain_permission_required",
        "secret_store_permission_required",
    }
)


class ConvergenceError(RuntimeError):
    """Base error for convergence state operations."""


class ConvergenceStateError(ConvergenceError):
    """The persisted state is unreadable or violates the expected shape."""


class InvalidTransition(ConvergenceError):
    """A caller attempted a state transition that is not currently valid."""


@dataclass(frozen=True)
class RetryPolicy:
    """Retry and lease bounds shared by all convergence items."""

    max_local_attempts: int = 2
    max_frontier_attempts: int = 3
    local_base_delay_seconds: int = 300
    frontier_base_delay_seconds: int = 900
    max_delay_seconds: int = 86_400
    lease_seconds: int = 1_800

    def __post_init__(self) -> None:
        for name in (
            "max_local_attempts",
            "max_frontier_attempts",
            "local_base_delay_seconds",
            "frontier_base_delay_seconds",
            "max_delay_seconds",
            "lease_seconds",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")


class CycleBudget:
    """In-memory guard against runaway work during one autonomy cycle.

    Budget consumption is intentionally not persisted.  A new sleep cycle gets
    a new budget, while per-item attempt limits remain persisted by
    :class:`ConvergenceStore`.
    """

    _LIMIT_ATTRS = {
        "local": "max_local_calls",
        "frontier": "max_frontier_calls",
        "mutation": "max_mutations",
        "raw_bytes": "max_raw_bytes",
    }

    def __init__(
        self,
        *,
        max_local_calls: int = 50,
        max_frontier_calls: int = 8,
        max_mutations: int = 25,
        max_raw_bytes: int = 2_000_000,
        max_elapsed_seconds: float = 1_800.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        limits = {
            "max_local_calls": max_local_calls,
            "max_frontier_calls": max_frontier_calls,
            "max_mutations": max_mutations,
            "max_raw_bytes": max_raw_bytes,
        }
        if any(value < 0 for value in limits.values()):
            raise ValueError("cycle budget limits must be >= 0")
        if max_elapsed_seconds < 0:
            raise ValueError("max_elapsed_seconds must be >= 0")
        self.max_local_calls = max_local_calls
        self.max_frontier_calls = max_frontier_calls
        self.max_mutations = max_mutations
        self.max_raw_bytes = max_raw_bytes
        self.max_elapsed_seconds = float(max_elapsed_seconds)
        self._clock = clock
        self._started = clock()
        self._used: dict[str, int] = {kind: 0 for kind in self._LIMIT_ATTRS}
        self._lock = threading.Lock()

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started)

    @property
    def remaining_elapsed_seconds(self) -> float:
        return max(0.0, self.max_elapsed_seconds - self.elapsed_seconds)

    def _can_consume_unlocked(self, kind: BudgetKind, amount: int) -> tuple[bool, str]:
        if kind not in self._LIMIT_ATTRS:
            raise ValueError(f"unknown budget kind: {kind!r}")
        if amount < 0:
            raise ValueError("budget amount must be >= 0")
        if self.elapsed_seconds >= self.max_elapsed_seconds:
            return False, "elapsed_budget_exhausted"
        limit = int(getattr(self, self._LIMIT_ATTRS[kind]))
        if self._used[kind] + amount > limit:
            return False, f"{kind}_budget_exhausted"
        return True, "ok"

    def can_consume(self, kind: BudgetKind, amount: int = 1) -> tuple[bool, str]:
        with self._lock:
            return self._can_consume_unlocked(kind, amount)

    def consume(self, kind: BudgetKind, amount: int = 1) -> tuple[bool, str]:
        with self._lock:
            allowed, reason = self._can_consume_unlocked(kind, amount)
            if allowed:
                self._used[kind] += amount
            return allowed, reason

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            limits = {
                kind: int(getattr(self, attr))
                for kind, attr in self._LIMIT_ATTRS.items()
            }
            return {
                "elapsed_seconds": round(self.elapsed_seconds, 3),
                "max_elapsed_seconds": self.max_elapsed_seconds,
                "used": dict(self._used),
                "limits": limits,
                "remaining": {
                    kind: max(0, limits[kind] - self._used[kind])
                    for kind in self._LIMIT_ATTRS
                },
            }

    def slice(
        self,
        *,
        max_local_calls: int = 0,
        max_frontier_calls: int = 0,
        max_mutations: int = 0,
        max_raw_bytes: int = 0,
    ) -> CycleBudgetSlice:
        """Reserve a per-lane ceiling while charging every use to this parent."""

        return CycleBudgetSlice(
            self,
            max_local_calls=max_local_calls,
            max_frontier_calls=max_frontier_calls,
            max_mutations=max_mutations,
            max_raw_bytes=max_raw_bytes,
        )


class CycleBudgetSlice:
    """A lane-local cap backed by one shared parent cycle budget."""

    _LIMIT_ATTRS = CycleBudget._LIMIT_ATTRS

    def __init__(
        self,
        parent: CycleBudget,
        *,
        max_local_calls: int,
        max_frontier_calls: int,
        max_mutations: int,
        max_raw_bytes: int,
    ) -> None:
        limits = {
            "max_local_calls": max_local_calls,
            "max_frontier_calls": max_frontier_calls,
            "max_mutations": max_mutations,
            "max_raw_bytes": max_raw_bytes,
        }
        if any(value < 0 for value in limits.values()):
            raise ValueError("budget slice limits must be >= 0")
        self.parent = parent
        for name, value in limits.items():
            setattr(self, name, int(value))
        self._used: dict[str, int] = {kind: 0 for kind in self._LIMIT_ATTRS}
        self._lock = threading.Lock()

    @property
    def elapsed_seconds(self) -> float:
        return self.parent.elapsed_seconds

    @property
    def max_elapsed_seconds(self) -> float:
        return self.parent.max_elapsed_seconds

    @property
    def remaining_elapsed_seconds(self) -> float:
        return self.parent.remaining_elapsed_seconds

    def _local_check(self, kind: BudgetKind, amount: int) -> tuple[bool, str]:
        if kind not in self._LIMIT_ATTRS:
            raise ValueError(f"unknown budget kind: {kind!r}")
        if amount < 0:
            raise ValueError("budget amount must be >= 0")
        limit = int(getattr(self, self._LIMIT_ATTRS[kind]))
        if self._used[kind] + amount > limit:
            return False, f"{kind}_lane_budget_exhausted"
        return self.parent.can_consume(kind, amount)

    def can_consume(self, kind: BudgetKind, amount: int = 1) -> tuple[bool, str]:
        with self._lock:
            return self._local_check(kind, amount)

    def consume(self, kind: BudgetKind, amount: int = 1) -> tuple[bool, str]:
        with self._lock:
            allowed, reason = self._local_check(kind, amount)
            if not allowed:
                return False, reason
            allowed, reason = self.parent.consume(kind, amount)
            if allowed:
                self._used[kind] += amount
            return allowed, reason

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            limits = {
                kind: int(getattr(self, attr))
                for kind, attr in self._LIMIT_ATTRS.items()
            }
            return {
                "elapsed_seconds": round(self.elapsed_seconds, 3),
                "max_elapsed_seconds": self.max_elapsed_seconds,
                "used": dict(self._used),
                "limits": limits,
                "remaining": {
                    kind: max(0, limits[kind] - self._used[kind])
                    for kind in self._LIMIT_ATTRS
                },
                "parent": self.parent.snapshot(),
            }


def _canonicalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats cannot be used in convergence keys")
        return value
    if isinstance(value, bytes):
        return {"__bytes_sha256__": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_canonicalize(item) for item in value]
        return sorted(normalized, key=canonical_json)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize(item) for item in value]
    raise TypeError(f"unsupported convergence key value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return a deterministic JSON representation suitable for hashing."""

    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _semantic_hold_history(
    *values: object,
    lane: str | None = None,
) -> list[dict[str, Any]]:
    """Collect strict common holds in stable oldest-to-newest order.

    A resumed item temporarily stores its terminal result under
    ``resume_context``.  A later no-quorum result must carry every earlier
    exact hold forward so an A -> B -> A authority rollback can be restored
    without another model sample.  Only self-validating common holds are
    retained; malformed or lane-mismatched history is ignored fail-closed.
    """

    holds: list[dict[str, Any]] = []
    seen_digests: set[str] = set()
    visited: set[int] = set()

    def append_hold(value: object) -> None:
        hold = persisted_semantic_no_quorum_hold(value, lane=lane)
        if hold is None:
            return
        digest = str(hold.get("hold_sha256") or "")
        if not digest or digest in seen_digests:
            return
        seen_digests.add(digest)
        holds.append(hold)

    def visit(value: object) -> None:
        if not isinstance(value, Mapping):
            return
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)

        # Preserve previously accumulated history before the current hold so
        # the resulting list remains chronological across repeated resumes.
        nested_result = value.get("result")
        if isinstance(nested_result, Mapping):
            visit(nested_result)
        nested_context = value.get("resume_context")
        if isinstance(nested_context, Mapping):
            visit(nested_context)
        history = value.get("semantic_hold_history")
        if isinstance(history, Sequence) and not isinstance(
            history, (str, bytes, bytearray)
        ):
            for candidate in history:
                append_hold(candidate)
        append_hold(value)
        invalidated = value.get("invalidated_semantic_hold")
        if isinstance(invalidated, Mapping):
            append_hold(invalidated)

    for value in values:
        visit(value)
    return holds


def input_fingerprint(input_data: Any) -> str:
    return hashlib.sha256(canonical_json(input_data).encode("utf-8")).hexdigest()


def stable_item_key(
    lane: str,
    source_id: str,
    input_data: Any,
    *,
    resolver_version: str | int = "1",
) -> str:
    """Build a stable identity for one input at one resolver version."""

    if not lane.strip():
        raise ValueError("lane is required")
    if not source_id.strip():
        raise ValueError("source_id is required")
    payload = {
        "lane": lane.strip(),
        "source_id": source_id.strip(),
        "input_hash": input_fingerprint(input_data),
        "resolver_version": str(resolver_version),
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{lane.strip()}:{digest}"


def exponential_backoff_seconds(
    attempt: int,
    *,
    base_seconds: int,
    max_seconds: int,
) -> int:
    """Return capped exponential backoff for a one-based attempt count."""

    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    if base_seconds < 0 or max_seconds < 0:
        raise ValueError("backoff bounds must be >= 0")
    return min(max_seconds, base_seconds * (2 ** (attempt - 1)))


def is_human_required_failure(failure_class: str | None) -> bool:
    return bool(failure_class and failure_class in HUMAN_REQUIRED_FAILURE_CLASSES)


def frontier_failure_class(result: object) -> str | None:
    """Return the machine-classified failure class from a frontier result.

    ``human_required`` is deliberately *not* accepted as evidence.  It is a
    derived presentation field and may also be emitted by a model or an older
    worker.  Only the deterministic failure class crosses the external
    authority boundary.
    """

    if not isinstance(result, Mapping):
        return None
    direct = result.get("failure_class")
    if isinstance(direct, str) and direct:
        return direct
    for key in ("frontier_failure", "failure"):
        nested = result.get(key)
        if isinstance(nested, Mapping):
            failure_class = frontier_failure_class(nested)
            if failure_class:
                return failure_class
    return None


def is_human_required_result(result: object) -> bool:
    """Classify a frontier payload without trusting model-authored booleans."""

    return is_human_required_failure(frontier_failure_class(result))




def _iso(now: datetime | None = None) -> str:
    return _utc_now(now).isoformat(timespec="seconds")


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _utc_now(parsed)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        with suppress(OSError):
            tmp_path.unlink()
        raise


class ConvergenceStore:
    """Persistent latest-state projection for autonomous work items."""

    def __init__(
        self,
        state_file: Path | None = None,
        *,
        events_file: Path | None = None,
        lock_file: Path | None = None,
        policy: RetryPolicy | None = None,
    ) -> None:
        default_dir = chronovisor_store.CHRONOVISOR_ROOT / "runtime" / "convergence"
        self.state_file = state_file or (default_dir / "state.json")
        self.events_file = events_file or (self.state_file.parent / "events.jsonl")
        self.lock_file = lock_file or (self.state_file.parent / "state.lock")
        self.policy = policy or RetryPolicy()

    def _empty_state(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "items": {}}

    def _load_unlocked(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return self._empty_state()
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConvergenceStateError(
                f"cannot read convergence state: {exc}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), dict):
            raise ConvergenceStateError(
                "convergence state must contain an items object"
            )
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ConvergenceStateError(
                f"unsupported convergence schema: {payload.get('schema_version')!r}"
            )
        return payload

    @contextmanager
    def _exclusive_lock(self):
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_file.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _save_unlocked(self, state: dict[str, Any]) -> None:
        _atomic_write_json(self.state_file, state)

    def _append_event_unlocked(self, event: dict[str, Any]) -> None:
        self._append_events_unlocked([event])

    def _append_events_unlocked(self, events: Iterable[dict[str, Any]]) -> None:
        rows = list(events)
        if not rows:
            return
        self.events_file.parent.mkdir(parents=True, exist_ok=True)
        with self.events_file.open("a", encoding="utf-8") as handle:
            for event in rows:
                handle.write(
                    json.dumps(
                        event,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())

    def _event(
        self,
        *,
        key: str,
        name: str,
        now: datetime,
        previous_status: str | None,
        item: Mapping[str, Any],
        **extra: Any,
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "ts": _iso(now),
            "key": key,
            "lane": item.get("lane"),
            "source_id": item.get("source_id"),
            "event": name,
            "previous_status": previous_status,
            "status": item.get("status"),
            **extra,
        }

    def load(self) -> dict[str, Any]:
        """Return a defensive copy without creating files."""

        return copy.deepcopy(self._load_unlocked())

    def get(self, key: str) -> dict[str, Any] | None:
        item = self._load_unlocked()["items"].get(key)
        return copy.deepcopy(item) if isinstance(item, dict) else None

    def list_items(
        self,
        *,
        lane: str | None = None,
        statuses: Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        allowed = set(statuses) if statuses is not None else None
        items = []
        for item in self._load_unlocked()["items"].values():
            if not isinstance(item, dict):
                continue
            if lane is not None and item.get("lane") != lane:
                continue
            if allowed is not None and item.get("status") not in allowed:
                continue
            items.append(copy.deepcopy(item))
        return sorted(
            items, key=lambda item: (str(item.get("lane")), str(item.get("key")))
        )

    def merge_item(
        self,
        *,
        lane: str,
        source_id: str,
        input_data: Any,
        resolver_version: str | int = "1",
        metadata: Mapping[str, Any] | None = None,
        update_metadata: bool = True,
        supersede_eligible_keys: Iterable[str] | None = None,
        source_history_eligible_keys: Iterable[str] | None = None,
        source_history_required_keys: Iterable[str] | None = None,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Insert a candidate or preserve its existing progress.

        Re-merging the same stable key is a no-op unless observational
        ``metadata`` changed. Set ``update_metadata=False`` when the producer's
        evidence bundle is immutable after first capture. Attempts and valid
        terminal states are never reset; legacy ``human_required`` items whose
        failure class is outside the external-authority allowlist are reopened
        or quarantined without resetting attempt counts.  When
        ``supersede_eligible_keys`` is supplied, creation fails closed instead
        of retiring an active same-source key outside that allowlist.  The
        stricter ``source_history_eligible_keys`` also rejects terminal
        same-source keys outside its snapshot and performs that check under
        the same lock as creation. ``source_history_required_keys`` makes
        deletion or reassignment of a snapshotted key fail closed as well.
        """

        current_time = _utc_now(now)
        key = stable_item_key(
            lane,
            source_id,
            input_data,
            resolver_version=resolver_version,
        )
        fingerprint = input_fingerprint(input_data)
        normalized_metadata = _canonicalize(dict(metadata or {}))
        eligible_superseded = (
            {str(value) for value in supersede_eligible_keys}
            if supersede_eligible_keys is not None
            else None
        )
        eligible_source_history = (
            {str(value) for value in source_history_eligible_keys}
            if source_history_eligible_keys is not None
            else None
        )
        required_source_history = (
            {str(value) for value in source_history_required_keys}
            if source_history_required_keys is not None
            else None
        )

        def merge(
            state: dict[str, Any],
        ) -> tuple[
            dict[str, Any] | None,
            bool,
            bool,
            bool,
            list[tuple[str, str | None, dict[str, Any]]],
            list[str],
        ]:
            items = state["items"]
            observed_source_history = {
                str(previous_key)
                for previous_key, previous in items.items()
                if isinstance(previous, dict)
                and previous.get("lane") == lane.strip()
                and previous.get("source_id") == source_id.strip()
            }
            missing_history = sorted(
                (required_source_history or set()) - observed_source_history
            )
            blocked = [
                str(previous_key)
                for previous_key, previous in items.items()
                if isinstance(previous, dict)
                and previous.get("lane") == lane.strip()
                and previous.get("source_id") == source_id.strip()
                and (
                    (
                        eligible_source_history is not None
                        and str(previous_key) not in eligible_source_history
                    )
                    or (
                        previous.get("status") not in TERMINAL_STATUSES
                        and eligible_superseded is not None
                        and str(previous_key) not in eligible_superseded
                    )
                )
            ]
            blocked.extend(f"missing:{key}" for key in missing_history)
            if blocked:
                return None, False, False, False, [], sorted(set(blocked))
            existing = items.get(key)
            if isinstance(existing, dict):
                reclassified = existing.get(
                    "status"
                ) == "human_required" and not is_human_required_failure(
                    str(existing.get("last_failure_class") or "")
                )
                if reclassified:
                    attempts = int(existing.get("frontier_attempts") or 0)
                    exhausted = attempts >= self.policy.max_frontier_attempts
                    existing = {
                        **existing,
                        "status": "quarantined" if exhausted else "frontier_retry",
                        "human_required": False,
                        "next_attempt_at": None,
                        "lease_stage": None,
                        "lease_owner": None,
                        "lease_expires_at": None,
                        "quarantine_reason": (
                            "retry_exhausted:frontier" if exhausted else None
                        ),
                        "updated_at": _iso(current_time),
                    }
                changed = (
                    update_metadata and existing.get("metadata") != normalized_metadata
                )
                if changed:
                    existing = {
                        **existing,
                        "metadata": normalized_metadata,
                        "updated_at": _iso(current_time),
                    }
                if changed or reclassified:
                    items[key] = existing
                return (
                    existing,
                    False,
                    changed or reclassified,
                    reclassified,
                    [],
                    [],
                )
            retired: list[tuple[str, str | None, dict[str, Any]]] = []
            for previous_key, previous in list(items.items()):
                if (
                    not isinstance(previous, dict)
                    or previous.get("lane") != lane.strip()
                    or previous.get("source_id") != source_id.strip()
                    or previous.get("status") in TERMINAL_STATUSES
                ):
                    continue
                previous_status = str(previous.get("status") or "")
                previous = {
                    **previous,
                    "status": "rejected",
                    "next_attempt_at": None,
                    "lease_stage": None,
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "last_error": None,
                    "result": {
                        "reason": "superseded_by_new_input",
                        "replacement_key": key,
                    },
                    "updated_at": _iso(current_time),
                }
                items[previous_key] = previous
                retired.append((str(previous_key), previous_status, previous))
            item = {
                "schema_version": SCHEMA_VERSION,
                "key": key,
                "lane": lane.strip(),
                "source_id": source_id.strip(),
                "input_hash": fingerprint,
                "resolver_version": str(resolver_version),
                "metadata": normalized_metadata,
                "status": "pending_local",
                "local_attempts": 0,
                "frontier_attempts": 0,
                "next_attempt_at": None,
                "lease_stage": None,
                "lease_owner": None,
                "lease_expires_at": None,
                "last_error": None,
                "last_failure_class": None,
                "human_required": False,
                "quarantine_reason": None,
                "result": None,
                "created_at": _iso(current_time),
                "updated_at": _iso(current_time),
            }
            items[key] = item
            return item, True, True, False, retired, []

        if dry_run:
            state = self._load_unlocked()
            item, created, changed, reclassified, retired, blocked = merge(state)
            return {
                "created": created,
                "changed": changed,
                "reclassified_human_boundary": reclassified,
                "dry_run": True,
                "item": copy.deepcopy(item),
                "retired": [entry[0] for entry in retired],
                "blocked_by_out_of_scope": blocked,
            }

        with self._exclusive_lock():
            state = self._load_unlocked()
            item, created, changed, reclassified, retired, blocked = merge(state)
            if changed:
                self._save_unlocked(state)
                for retired_key, previous_status, retired_item in retired:
                    self._append_event_unlocked(
                        self._event(
                            key=retired_key,
                            name="candidate_superseded",
                            now=current_time,
                            previous_status=previous_status,
                            item=retired_item,
                            replacement_key=key,
                        )
                    )
                self._append_event_unlocked(
                    self._event(
                        key=key,
                        name=(
                            "candidate_merged"
                            if created
                            else "human_boundary_reclassified"
                            if reclassified
                            else "metadata_updated"
                        ),
                        now=current_time,
                        previous_status=(
                            None
                            if created
                            else "human_required"
                            if reclassified
                            else str(item.get("status"))
                        ),
                        item=item,
                    )
                )
            return {
                "created": created,
                "changed": changed,
                "reclassified_human_boundary": reclassified,
                "dry_run": False,
                "item": copy.deepcopy(item),
                "retired": [entry[0] for entry in retired],
                "blocked_by_out_of_scope": blocked,
            }

    def merge_items_atomically(
        self,
        candidates: Iterable[Mapping[str, Any]],
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Validate and merge distinct source snapshots in one transaction.

        Every candidate is projected against one in-memory state while the
        store lock is held. If any source-history guard blocks, the projection
        is discarded, so no earlier candidate or audit event can leak through.
        Successful batches use one atomic state replacement and one event
        append batch.
        """

        rows = [dict(candidate) for candidate in candidates]
        seen_sources: set[tuple[str, str]] = set()
        for candidate in rows:
            marker = (
                str(candidate.get("lane") or "").strip(),
                str(candidate.get("source_id") or "").strip(),
            )
            if not all(marker) or marker in seen_sources:
                raise ValueError(
                    "atomic merge candidates require distinct non-empty lane/source"
                )
            seen_sources.add(marker)

        current_time = _utc_now(now)
        with self._exclusive_lock():
            state = self._load_unlocked()
            original_state = copy.deepcopy(state)
            projection_store = copy.copy(self)
            projection_store._load_unlocked = lambda: state  # type: ignore[method-assign]
            projected_results: list[dict[str, Any]] = []
            blocked_by_key: dict[str, list[str]] = {}
            for candidate in rows:
                key = stable_item_key(
                    str(candidate["lane"]),
                    str(candidate["source_id"]),
                    candidate.get("input_data"),
                    resolver_version=candidate.get("resolver_version", "1"),
                )
                result = projection_store.merge_item(
                    lane=str(candidate["lane"]),
                    source_id=str(candidate["source_id"]),
                    input_data=candidate.get("input_data"),
                    resolver_version=candidate.get("resolver_version", "1"),
                    metadata=(
                        candidate.get("metadata")
                        if isinstance(candidate.get("metadata"), Mapping)
                        else None
                    ),
                    update_metadata=bool(candidate.get("update_metadata", True)),
                    supersede_eligible_keys=candidate.get("supersede_eligible_keys"),
                    source_history_eligible_keys=candidate.get(
                        "source_history_eligible_keys"
                    ),
                    source_history_required_keys=candidate.get(
                        "source_history_required_keys"
                    ),
                    now=current_time,
                    dry_run=True,
                )
                blockers = result.get("blocked_by_out_of_scope")
                if isinstance(blockers, list) and blockers:
                    blocked_by_key[key] = [str(value) for value in blockers]
                    return {
                        "committed": False,
                        "results": [],
                        "blocked_by_key": blocked_by_key,
                    }
                projected_results.append({**result, "dry_run": False})

            if any(bool(result.get("changed")) for result in projected_results):
                events: list[dict[str, Any]] = []
                for result in projected_results:
                    if not result.get("changed"):
                        continue
                    item = result.get("item")
                    if not isinstance(item, Mapping):
                        raise ConvergenceStateError(
                            "atomic merge projected a changed item without state"
                        )
                    key = str(item.get("key") or "")
                    for retired_key in result.get("retired") or []:
                        previous = original_state["items"].get(str(retired_key))
                        retired_item = state["items"].get(str(retired_key))
                        if not isinstance(retired_item, Mapping):
                            raise ConvergenceStateError(
                                "atomic merge retired item readback is missing"
                            )
                        events.append(
                            self._event(
                                key=str(retired_key),
                                name="candidate_superseded",
                                now=current_time,
                                previous_status=(
                                    str(previous.get("status") or "")
                                    if isinstance(previous, Mapping)
                                    else None
                                ),
                                item=retired_item,
                                replacement_key=key,
                            )
                        )
                    reclassified = bool(result.get("reclassified_human_boundary"))
                    events.append(
                        self._event(
                            key=key,
                            name=(
                                "candidate_merged"
                                if result.get("created")
                                else "human_boundary_reclassified"
                                if reclassified
                                else "metadata_updated"
                            ),
                            now=current_time,
                            previous_status=(
                                None
                                if result.get("created")
                                else "human_required"
                                if reclassified
                                else str(item.get("status") or "")
                            ),
                            item=item,
                        )
                    )
                self._save_unlocked(state)
                self._append_events_unlocked(events)
            return {
                "committed": True,
                "results": projected_results,
                "blocked_by_key": {},
            }

    def retire_absent_sources(
        self,
        *,
        lane: str,
        active_source_ids: Iterable[str],
        eligible_keys: Iterable[str] | None = None,
        reason: str = "source_no_longer_actionable",
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Terminalize nonterminal items missing from a complete producer inventory."""
        current_time = _utc_now(now)
        active = {str(source_id) for source_id in active_source_ids}
        eligible = (
            {str(key) for key in eligible_keys} if eligible_keys is not None else None
        )

        def retire(state: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
            retired: list[tuple[str, str, dict[str, Any]]] = []
            for key, item in list(state["items"].items()):
                if (
                    (eligible is not None and str(key) not in eligible)
                    or not isinstance(item, dict)
                    or item.get("lane") != lane
                    or item.get("status") in TERMINAL_STATUSES
                    or str(item.get("source_id") or "") in active
                ):
                    continue
                previous_status = str(item.get("status") or "")
                updated = {
                    **item,
                    "status": "rejected",
                    "next_attempt_at": None,
                    "lease_stage": None,
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "last_error": None,
                    "result": {"reason": reason},
                    "updated_at": _iso(current_time),
                }
                state["items"][key] = updated
                retired.append((str(key), previous_status, updated))
            return retired

        if dry_run:
            projected = retire(self._load_unlocked())
            return {"retired": [entry[0] for entry in projected], "dry_run": True}
        with self._exclusive_lock():
            state = self._load_unlocked()
            retired = retire(state)
            if retired:
                self._save_unlocked(state)
                for key, previous_status, item in retired:
                    self._append_event_unlocked(
                        self._event(
                            key=key,
                            name="source_retired",
                            now=current_time,
                            previous_status=previous_status,
                            item=item,
                            reason=reason,
                        )
                    )
            return {"retired": [entry[0] for entry in retired], "dry_run": False}

    def retire_stale(
        self,
        *,
        lane: str,
        eligible_keys: Iterable[str] | None = None,
        max_age_seconds: int = 7 * 24 * 60 * 60,
        reason: str = "stale_source_ttl",
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Bound orphaned producer state even when a source inventory is truncated."""
        current_time = _utc_now(now)
        threshold = current_time - timedelta(seconds=max(0, max_age_seconds))
        eligible = (
            {str(key) for key in eligible_keys} if eligible_keys is not None else None
        )

        def retire(state: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
            retired: list[tuple[str, str, dict[str, Any]]] = []
            for key, item in list(state["items"].items()):
                if (
                    (eligible is not None and str(key) not in eligible)
                    or not isinstance(item, dict)
                    or item.get("lane") != lane
                    or item.get("status") in TERMINAL_STATUSES
                ):
                    continue
                last_seen = _parse_iso(item.get("updated_at")) or _parse_iso(
                    item.get("created_at")
                )
                if last_seen is None or last_seen >= threshold:
                    continue
                previous_status = str(item.get("status") or "")
                updated = {
                    **item,
                    "status": "rejected",
                    "next_attempt_at": None,
                    "lease_stage": None,
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "last_error": None,
                    "result": {"reason": reason},
                    "updated_at": _iso(current_time),
                }
                state["items"][key] = updated
                retired.append((str(key), previous_status, updated))
            return retired

        if dry_run:
            projected = retire(self._load_unlocked())
            return {"retired": [entry[0] for entry in projected], "dry_run": True}
        with self._exclusive_lock():
            state = self._load_unlocked()
            retired = retire(state)
            if retired:
                self._save_unlocked(state)
                for key, previous_status, item in retired:
                    self._append_event_unlocked(
                        self._event(
                            key=key,
                            name="source_expired",
                            now=current_time,
                            previous_status=previous_status,
                            item=item,
                            reason=reason,
                        )
                    )
            return {"retired": [entry[0] for entry in retired], "dry_run": False}

    def merge_items(
        self,
        candidates: Iterable[Mapping[str, Any]],
        *,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> list[dict[str, Any]]:
        """Convenience wrapper for queue builders.

        Each candidate accepts the same keyword fields as :meth:`merge_item`.
        """

        return [
            self.merge_item(
                lane=str(candidate["lane"]),
                source_id=str(candidate["source_id"]),
                input_data=candidate.get("input_data"),
                resolver_version=candidate.get("resolver_version", "1"),
                metadata=candidate.get("metadata")
                if isinstance(candidate.get("metadata"), Mapping)
                else None,
                now=now,
                dry_run=dry_run,
            )
            for candidate in candidates
        ]

    def _lease_active(self, item: Mapping[str, Any], now: datetime) -> bool:
        expiry = _parse_iso(item.get("lease_expires_at"))
        return bool(expiry and expiry > now and item.get("lease_owner"))

    def _stage_statuses(self, stage: Stage) -> frozenset[str]:
        return LOCAL_STATUSES if stage == "local" else FRONTIER_STATUSES

    def _stage_attempt_field(self, stage: Stage) -> str:
        return "local_attempts" if stage == "local" else "frontier_attempts"

    def _stage_limit(self, stage: Stage) -> int:
        return (
            self.policy.max_local_attempts
            if stage == "local"
            else self.policy.max_frontier_attempts
        )

    def _clear_lease(self, item: dict[str, Any]) -> None:
        item["lease_stage"] = None
        item["lease_owner"] = None
        item["lease_expires_at"] = None

    def _exhausted_transition(
        self,
        item: dict[str, Any],
        *,
        stage: Stage,
        now: datetime,
        allow_frontier: bool = True,
    ) -> None:
        self._clear_lease(item)
        item["next_attempt_at"] = None
        if (
            stage == "local"
            and allow_frontier
            and self.policy.max_frontier_attempts > 0
        ):
            item["status"] = "pending_frontier"
        else:
            item["status"] = "quarantined"
            item["quarantine_reason"] = f"retry_exhausted:{stage}"
        item["updated_at"] = _iso(now)

    def reap_expired_leases(
        self,
        *,
        eligible_keys: Iterable[str] | None = None,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Return expired running items to a visible retry state."""
        current_time = _utc_now(now)
        eligible = (
            {str(key) for key in eligible_keys} if eligible_keys is not None else None
        )

        def project(state: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
            recovered: list[tuple[str, str, dict[str, Any]]] = []
            for key, item in state["items"].items():
                if eligible is not None and str(key) not in eligible:
                    continue
                if not isinstance(item, dict):
                    continue
                status = str(item.get("status") or "")
                if status not in {"local_running", "frontier_running"}:
                    continue
                expiry = _parse_iso(item.get("lease_expires_at"))
                if expiry is None or expiry > current_time:
                    continue
                stage = "local" if status.startswith("local_") else "frontier"
                item["status"] = f"{stage}_retry"
                item["next_attempt_at"] = _iso(current_time)
                item["last_error"] = "worker lease expired before completion"
                item["last_failure_class"] = "worker_lease_expired"
                self._clear_lease(item)
                item["updated_at"] = _iso(current_time)
                recovered.append((str(key), status, item))
            return recovered

        if dry_run:
            state = self._load_unlocked()
            recovered = project(state)
            return {
                "status": "ok",
                "recovered": len(recovered),
                "keys": [row[0] for row in recovered],
                "dry_run": True,
            }
        with self._exclusive_lock():
            state = self._load_unlocked()
            recovered = project(state)
            if recovered:
                self._save_unlocked(state)
                for key, previous, item in recovered:
                    self._append_event_unlocked(
                        self._event(
                            key=key,
                            name="expired_lease_reaped",
                            now=current_time,
                            previous_status=previous,
                            item=item,
                        )
                    )
            return {
                "status": "ok",
                "recovered": len(recovered),
                "keys": [row[0] for row in recovered],
                "dry_run": False,
            }

    def claim_attempt(
        self,
        key: str,
        stage: Stage,
        *,
        owner: str | None = None,
        lease_seconds: int | None = None,
        budget: CycleBudget | None = None,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Atomically lease and count one local/frontier attempt."""

        if stage not in {"local", "frontier"}:
            raise ValueError(f"unknown stage: {stage!r}")
        current_time = _utc_now(now)
        lease_for = (
            self.policy.lease_seconds if lease_seconds is None else lease_seconds
        )
        if lease_for < 0:
            raise ValueError("lease_seconds must be >= 0")
        worker = owner or f"{os.getpid()}:{uuid.uuid4().hex}"

        def project(
            state: dict[str, Any],
        ) -> tuple[dict[str, Any], bool, str, str | None]:
            item = state["items"].get(key)
            if not isinstance(item, dict):
                raise KeyError(key)
            previous_status = str(item.get("status") or "")
            if previous_status in TERMINAL_STATUSES:
                return item, False, "terminal", previous_status
            if previous_status not in self._stage_statuses(stage):
                return item, False, f"not_{stage}_pending", previous_status
            if previous_status.endswith("_running") and self._lease_active(
                item, current_time
            ):
                return item, False, "leased", previous_status
            due = _parse_iso(item.get("next_attempt_at"))
            if due and due > current_time:
                return item, False, "backoff", previous_status
            field = self._stage_attempt_field(stage)
            attempts = int(item.get(field) or 0)
            if attempts >= self._stage_limit(stage):
                self._exhausted_transition(item, stage=stage, now=current_time)
                return item, False, "retry_exhausted", previous_status
            item[field] = attempts + 1
            item["status"] = f"{stage}_running"
            item["next_attempt_at"] = None
            item["lease_stage"] = stage
            item["lease_owner"] = worker
            item["lease_expires_at"] = _iso(current_time + timedelta(seconds=lease_for))
            item["updated_at"] = _iso(current_time)
            return item, True, "claimed", previous_status

        if budget is not None:
            allowed, reason = budget.can_consume(stage)
            if not allowed:
                return {
                    "claimed": False,
                    "reason": reason,
                    "dry_run": dry_run,
                    "item": self.get(key),
                    "budget": budget.snapshot(),
                }

        if dry_run:
            state = self._load_unlocked()
            item, claimed, reason, _previous = project(state)
            return {
                "claimed": claimed,
                "reason": "dry_run" if claimed else reason,
                "dry_run": True,
                "owner": worker if claimed else None,
                "item": copy.deepcopy(item),
                "budget": budget.snapshot() if budget is not None else None,
            }

        with self._exclusive_lock():
            state = self._load_unlocked()
            item, claimed, reason, previous_status = project(state)
            changed = str(item.get("status")) != previous_status
            if claimed and budget is not None:
                consumed, budget_reason = budget.consume(stage)
                if not consumed:
                    persisted = self._load_unlocked()["items"].get(key)
                    return {
                        "claimed": False,
                        "reason": budget_reason,
                        "dry_run": False,
                        "owner": None,
                        "item": copy.deepcopy(persisted),
                        "budget": budget.snapshot(),
                    }
            if claimed or changed:
                self._save_unlocked(state)
                self._append_event_unlocked(
                    self._event(
                        key=key,
                        name="attempt_claimed" if claimed else "attempts_exhausted",
                        now=current_time,
                        previous_status=previous_status,
                        item=item,
                        stage=stage,
                        reason=reason,
                    )
                )
            return {
                "claimed": claimed,
                "reason": reason,
                "dry_run": False,
                "owner": worker if claimed else None,
                "item": copy.deepcopy(item),
                "budget": budget.snapshot() if budget is not None else None,
            }

    def _validate_owner(self, item: Mapping[str, Any], owner: str | None) -> None:
        lease_owner = item.get("lease_owner")
        if lease_owner is not None and (owner is None or lease_owner != owner):
            raise InvalidTransition("attempt lease is owned by another worker")
        if owner is not None and lease_owner != owner:
            raise InvalidTransition("attempt lease is not owned by this worker")

    def fail_attempt(
        self,
        key: str,
        stage: Stage,
        *,
        error: str,
        failure_class: str | None = None,
        owner: str | None = None,
        allow_frontier: bool = True,
        consume_attempt: bool = True,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Record a claimed failure and choose retry/escalate/quarantine."""

        if stage not in {"local", "frontier"}:
            raise ValueError(f"unknown stage: {stage!r}")
        current_time = _utc_now(now)

        def project(state: dict[str, Any]) -> tuple[dict[str, Any], str]:
            item = state["items"].get(key)
            if not isinstance(item, dict):
                raise KeyError(key)
            previous_status = str(item.get("status") or "")
            if previous_status != f"{stage}_running":
                raise InvalidTransition(
                    f"cannot fail {stage} attempt from status {previous_status!r}"
                )
            self._validate_owner(item, owner)
            item["last_error"] = str(error)[:4000]
            item["last_failure_class"] = failure_class
            item["updated_at"] = _iso(current_time)
            self._clear_lease(item)

            if not consume_attempt:
                field = self._stage_attempt_field(stage)
                item[field] = max(0, int(item.get(field) or 0) - 1)
                item["status"] = f"pending_{stage}"
                item["next_attempt_at"] = None
                return item, previous_status

            if is_human_required_failure(failure_class):
                item["status"] = "human_required"
                item["human_required"] = True
                item["next_attempt_at"] = None
                return item, previous_status

            attempts = int(item.get(self._stage_attempt_field(stage)) or 0)
            limit = self._stage_limit(stage)
            if attempts >= limit:
                self._exhausted_transition(
                    item,
                    stage=stage,
                    now=current_time,
                    allow_frontier=allow_frontier,
                )
                return item, previous_status

            base = (
                self.policy.local_base_delay_seconds
                if stage == "local"
                else self.policy.frontier_base_delay_seconds
            )
            delay = exponential_backoff_seconds(
                attempts,
                base_seconds=base,
                max_seconds=self.policy.max_delay_seconds,
            )
            item["status"] = f"{stage}_retry"
            item["next_attempt_at"] = _iso(current_time + timedelta(seconds=delay))
            return item, previous_status

        if dry_run:
            state = self._load_unlocked()
            item, _previous = project(state)
            return {"dry_run": True, "item": copy.deepcopy(item)}

        with self._exclusive_lock():
            state = self._load_unlocked()
            item, previous_status = project(state)
            self._save_unlocked(state)
            self._append_event_unlocked(
                self._event(
                    key=key,
                    name=(
                        "attempt_failed" if consume_attempt else "attempt_preempted"
                    ),
                    now=current_time,
                    previous_status=previous_status,
                    item=item,
                    stage=stage,
                    failure_class=failure_class,
                )
            )
            return {"dry_run": False, "item": copy.deepcopy(item)}

    def escalate(
        self,
        key: str,
        *,
        reason: str,
        owner: str | None = None,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Route a deterministic/local non-decision to the frontier lane."""

        current_time = _utc_now(now)

        def project(state: dict[str, Any]) -> tuple[dict[str, Any], str]:
            item = state["items"].get(key)
            if not isinstance(item, dict):
                raise KeyError(key)
            previous_status = str(item.get("status") or "")
            if previous_status in TERMINAL_STATUSES:
                raise InvalidTransition(
                    f"cannot escalate terminal status {previous_status!r}"
                )
            self._validate_owner(item, owner)
            self._clear_lease(item)
            item["last_error"] = str(reason)[:4000]
            item["updated_at"] = _iso(current_time)
            item["next_attempt_at"] = None
            if self.policy.max_frontier_attempts > 0:
                item["status"] = "pending_frontier"
            else:
                item["status"] = "quarantined"
                item["quarantine_reason"] = "frontier_disabled"
            return item, previous_status

        return self._persist_transition(
            key=key,
            name="escalated",
            now=current_time,
            dry_run=dry_run,
            project=project,
        )

    def return_to_local(
        self,
        key: str,
        *,
        reason: str,
        owner: str | None = None,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Return a claimed frontier item for a fresh bounded local proposal."""

        current_time = _utc_now(now)

        def project(state: dict[str, Any]) -> tuple[dict[str, Any], str]:
            item = state["items"].get(key)
            if not isinstance(item, dict):
                raise KeyError(key)
            previous_status = str(item.get("status") or "")
            if previous_status != "frontier_running":
                raise InvalidTransition(
                    f"cannot return to local from status {previous_status!r}"
                )
            self._validate_owner(item, owner)
            self._clear_lease(item)
            item["status"] = "pending_local"
            item["local_attempts"] = 0
            item["next_attempt_at"] = None
            item["last_error"] = str(reason)[:4000]
            item["updated_at"] = _iso(current_time)
            return item, previous_status

        return self._persist_transition(
            key=key,
            name="returned_to_local",
            now=current_time,
            dry_run=dry_run,
            project=project,
        )

    def resume_quarantined(
        self,
        key: str,
        *,
        stage: Stage = "frontier",
        reason: str = "autonomous_cooldown_elapsed",
        resume_context: Mapping[str, Any] | None = None,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Autonomously reopen a non-human quarantine after an external cooldown."""

        if stage not in {"local", "frontier"}:
            raise ValueError(f"unknown stage: {stage!r}")
        current_time = _utc_now(now)
        normalized_resume_context = (
            _canonicalize(dict(resume_context)) if resume_context is not None else None
        )
        resume_event_context = {
            field: normalized_resume_context[field]
            for field in (
                "decision_lane",
                "invalidated_hold_sha256",
                "expected_epoch_sha256",
            )
            if isinstance(normalized_resume_context, Mapping)
            and isinstance(normalized_resume_context.get(field), str)
        }

        def project(state: dict[str, Any]) -> tuple[dict[str, Any], str]:
            item = state["items"].get(key)
            if not isinstance(item, dict):
                raise KeyError(key)
            previous_status = str(item.get("status") or "")
            if previous_status != "quarantined":
                raise InvalidTransition(
                    f"cannot resume non-quarantined status {previous_status!r}"
                )
            if bool(item.get("human_required")):
                raise InvalidTransition("human-required item cannot be auto-resumed")
            self._clear_lease(item)
            item["status"] = f"pending_{stage}"
            item[self._stage_attempt_field(stage)] = 0
            if stage == "local":
                item["frontier_attempts"] = 0
            item["next_attempt_at"] = None
            item["quarantine_reason"] = None
            if normalized_resume_context is not None:
                effective_context = copy.deepcopy(normalized_resume_context)
                decision_lane = effective_context.get("decision_lane")
                history = _semantic_hold_history(
                    item.get("result"),
                    effective_context,
                    lane=(
                        decision_lane
                        if isinstance(decision_lane, str) and decision_lane
                        else None
                    ),
                )
                if history:
                    effective_context["semantic_hold_history"] = history
                item["result"] = {"resume_context": effective_context}
            else:
                item["result"] = None
            item["last_error"] = str(reason)[:4000]
            item["updated_at"] = _iso(current_time)
            return item, previous_status

        return self._persist_transition(
            key=key,
            name="quarantine_resumed",
            now=current_time,
            dry_run=dry_run,
            project=project,
            event_extra=resume_event_context,
        )

    def resume_due_quarantined(
        self,
        *,
        cooldown_seconds: int = 21_600,
        exclude_lanes: Iterable[str] = (),
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Reopen due non-human quarantines across convergence lanes."""

        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")
        current_time = _utc_now(now)
        excluded = set(exclude_lanes)
        results: list[dict[str, Any]] = []
        semantic_deferred = 0
        for item in self.list_items(statuses={"quarantined"}):
            key = str(item.get("key") or "")
            lane = str(item.get("lane") or "")
            failure_class = str(item.get("last_failure_class") or "")
            if (
                not key
                or lane in excluded
                or bool(item.get("human_required"))
                or is_human_required_failure(failure_class)
            ):
                continue
            # A local semantic split is a safe terminal non-decision, not an
            # outage.  Time cannot change its evidence or adopted authority,
            # so the generic cooldown worker must never turn it into another
            # model sample.  Strict common holds and incomplete legacy rows are
            # both fail-closed here; the owning lane may reopen a strict hold
            # only after validating a concrete epoch/authority change.
            if (
                failure_class == LOCAL_SEMANTIC_NO_QUORUM
                or persisted_semantic_no_quorum_hold(item) is not None
            ):
                semantic_deferred += 1
                continue
            updated_at = _parse_iso(item.get("updated_at"))
            if (
                updated_at is not None
                and (current_time - updated_at).total_seconds() < cooldown_seconds
            ):
                continue
            reason = str(item.get("quarantine_reason") or "")
            stage: Stage = "local" if reason.endswith(":local") else "frontier"
            try:
                transition = self.resume_quarantined(
                    key,
                    stage=stage,
                    reason="autonomous convergence quarantine cooldown elapsed",
                    now=current_time,
                    dry_run=dry_run,
                )
            except InvalidTransition as exc:
                results.append(
                    {
                        "key": key,
                        "lane": lane,
                        "status": "resume_skipped",
                        "error": str(exc),
                    }
                )
                continue
            resumed = transition.get("item") if isinstance(transition, dict) else {}
            results.append(
                {
                    "key": key,
                    "lane": lane,
                    "stage": stage,
                    "status": str((resumed or {}).get("status") or f"pending_{stage}"),
                }
            )
        return {
            "status": "ok",
            "cooldown_seconds": cooldown_seconds,
            "resumed": len(
                [row for row in results if row.get("status") != "resume_skipped"]
            ),
            "semantic_deferred": semantic_deferred,
            "results": results,
            "dry_run": dry_run,
        }

    def hold_semantic_no_quorum(
        self,
        key: str,
        *,
        lane: str,
        stage: Stage,
        review: Mapping[str, Any],
        epoch: Mapping[str, Any],
        authority: Mapping[str, Any],
        owner: str | None = None,
        error: str | None = None,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Persist one exact-epoch semantic non-decision without retrying.

        The claimed attempt has already been counted by :meth:`claim_attempt`.
        This transition deliberately preserves that count: lack of a local
        two-vote quorum is terminal for the exact decision epoch and must not
        consume the remaining retry window.
        """

        if stage not in {"local", "frontier"}:
            raise ValueError(f"unknown stage: {stage!r}")
        semantic_hold = build_semantic_no_quorum_hold(
            lane,
            epoch,
            authority,
            review,
        )
        current_time = _utc_now(now)
        summary = str(error or review.get("summary") or "local semantic no quorum")
        terminal_result = {
            "terminal_reason": "semantic_no_quorum",
            "semantic_hold": semantic_hold,
            "stage": stage,
        }

        def project(state: dict[str, Any]) -> tuple[dict[str, Any], str]:
            item = state["items"].get(key)
            if not isinstance(item, dict):
                raise KeyError(key)
            previous_status = str(item.get("status") or "")
            if previous_status == "quarantined":
                existing = persisted_semantic_no_quorum_hold(
                    item,
                    lane=lane,
                    epoch=epoch,
                    authority=authority,
                )
                if existing == semantic_hold:
                    return item, previous_status
                raise InvalidTransition(
                    "cannot replace a quarantined item with a different semantic hold"
                )
            if previous_status != f"{stage}_running":
                raise InvalidTransition(
                    f"cannot hold {stage} attempt from status {previous_status!r}"
                )
            self._validate_owner(item, owner)
            self._clear_lease(item)
            history = [
                hold
                for hold in _semantic_hold_history(item.get("result"), lane=lane)
                if hold.get("hold_sha256") != semantic_hold.get("hold_sha256")
            ]
            persisted_result = copy.deepcopy(terminal_result)
            if history:
                persisted_result["semantic_hold_history"] = history
            item["status"] = "quarantined"
            item["quarantine_reason"] = f"semantic_no_quorum:{lane}"
            item["last_error"] = summary[:4000]
            item["last_failure_class"] = LOCAL_SEMANTIC_NO_QUORUM
            item["result"] = persisted_result
            item["next_attempt_at"] = None
            item["human_required"] = False
            item["updated_at"] = _iso(current_time)
            return item, previous_status

        return self._persist_transition(
            key=key,
            name="semantic_no_quorum_held",
            now=current_time,
            dry_run=dry_run,
            project=project,
            event_extra={
                "decision_lane": lane,
                "hold_sha256": semantic_hold["hold_sha256"],
                "stage": stage,
            },
        )

    def restore_semantic_no_quorum_hold(
        self,
        key: str,
        *,
        lane: str,
        epoch: Mapping[str, Any],
        authority: Mapping[str, Any],
        owner: str | None = None,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any] | None:
        """Restore an invalidated hold if its exact epoch returned (ABA)."""

        item = self.get(key)
        if not isinstance(item, dict):
            return None
        result = item.get("result")
        context = result.get("resume_context") if isinstance(result, Mapping) else None
        if not isinstance(context, Mapping):
            return None
        invalidated = context.get("invalidated_semantic_hold")
        if not isinstance(invalidated, Mapping):
            return None
        strict_invalidated = persisted_semantic_no_quorum_hold(
            invalidated,
            lane=lane,
        )
        if strict_invalidated is None or context.get(
            "invalidated_hold_sha256"
        ) != strict_invalidated.get("hold_sha256"):
            return None

        history = _semantic_hold_history(result, context, lane=lane)
        strict_hold = next(
            (
                candidate
                for candidate in reversed(history)
                if persisted_semantic_no_quorum_hold(
                    candidate,
                    lane=lane,
                    epoch=epoch,
                    authority=authority,
                )
                is not None
            ),
            None,
        )
        if strict_hold is None:
            return None
        preserved_history = [
            candidate
            for candidate in history
            if candidate.get("hold_sha256") != strict_hold.get("hold_sha256")
        ]
        terminal_result: dict[str, Any] = {
            "terminal_reason": "semantic_no_quorum",
            "semantic_hold": strict_hold,
            "stage": str(item.get("lease_stage") or "frontier"),
        }
        if preserved_history:
            terminal_result["semantic_hold_history"] = preserved_history
        return self.quarantine(
            key,
            reason=f"semantic_no_quorum:{lane}",
            error="semantic hold epoch restored before reevaluation",
            failure_class=LOCAL_SEMANTIC_NO_QUORUM,
            result=terminal_result,
            owner=owner,
            now=now,
            dry_run=dry_run,
        )

    def complete(
        self,
        key: str,
        status: Literal["applied", "rejected"],
        *,
        result: Mapping[str, Any] | None = None,
        owner: str | None = None,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Mark the current input terminal without resetting future versions."""

        if status not in {"applied", "rejected"}:
            raise ValueError("complete status must be applied or rejected")
        current_time = _utc_now(now)
        normalized_result = _canonicalize(dict(result or {}))

        def project(state: dict[str, Any]) -> tuple[dict[str, Any], str]:
            item = state["items"].get(key)
            if not isinstance(item, dict):
                raise KeyError(key)
            previous_status = str(item.get("status") or "")
            if previous_status in TERMINAL_STATUSES:
                if previous_status == status:
                    return item, previous_status
                raise InvalidTransition(
                    f"cannot replace terminal status {previous_status!r} with {status!r}"
                )
            self._validate_owner(item, owner)
            self._clear_lease(item)
            item["status"] = status
            item["result"] = normalized_result
            item["next_attempt_at"] = None
            item["updated_at"] = _iso(current_time)
            return item, previous_status

        return self._persist_transition(
            key=key,
            name="completed",
            now=current_time,
            dry_run=dry_run,
            project=project,
        )

    def complete_many(
        self,
        keys: Iterable[str],
        status: Literal["applied", "rejected"],
        *,
        result: Mapping[str, Any] | None = None,
        replace_terminal_statuses: Iterable[str] = (),
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Complete many unleased items with one durable state replacement.

        This is intended for deterministic queue migrations. It avoids one
        full state rewrite and fsync per item while preserving an individual
        completion event for every transitioned key. Running/leased items are
        skipped. Terminal replacement is limited to explicitly allowlisted
        quarantine/human-boundary migrations; applied/rejected are immutable.
        """

        if status not in {"applied", "rejected"}:
            raise ValueError("complete status must be applied or rejected")
        normalized_keys = list(
            dict.fromkeys(str(key) for key in keys if isinstance(key, str) and key)
        )
        replace_terminal = set(replace_terminal_statuses)
        allowed_terminal_replacements = {"quarantined", "human_required"}
        if not replace_terminal.issubset(allowed_terminal_replacements):
            raise ValueError(
                "complete_many can only replace quarantined or human_required terminals"
            )
        normalized_result = _canonicalize(dict(result or {}))
        current_time = _utc_now(now)

        def project(
            state: dict[str, Any],
        ) -> tuple[list[tuple[str, str, dict[str, Any]]], dict[str, int]]:
            completed: list[tuple[str, str, dict[str, Any]]] = []
            skipped: dict[str, int] = {}
            for key in normalized_keys:
                item = state["items"].get(key)
                if not isinstance(item, dict):
                    skipped["missing"] = skipped.get("missing", 0) + 1
                    continue
                previous_status = str(item.get("status") or "")
                if (
                    previous_status in TERMINAL_STATUSES
                    and previous_status not in replace_terminal
                ):
                    skipped["terminal"] = skipped.get("terminal", 0) + 1
                    continue
                if previous_status.endswith("_running") or item.get("lease_owner"):
                    skipped["leased"] = skipped.get("leased", 0) + 1
                    continue
                self._clear_lease(item)
                item["status"] = status
                item["result"] = copy.deepcopy(normalized_result)
                item["next_attempt_at"] = None
                if previous_status in replace_terminal:
                    item["human_required"] = False
                    item["quarantine_reason"] = None
                item["updated_at"] = _iso(current_time)
                completed.append((key, previous_status, item))
            return completed, skipped

        if dry_run:
            completed, skipped = project(self._load_unlocked())
            return {
                "status": "ok",
                "dry_run": True,
                "requested": len(normalized_keys),
                "completed": len(completed),
                "skipped": sum(skipped.values()),
                "skipped_reasons": skipped,
            }
        with self._exclusive_lock():
            state = self._load_unlocked()
            completed, skipped = project(state)
            if completed:
                self._save_unlocked(state)
                self._append_events_unlocked(
                    self._event(
                        key=key,
                        name="completed",
                        now=current_time,
                        previous_status=previous_status,
                        item=item,
                    )
                    for key, previous_status, item in completed
                )
            return {
                "status": "ok",
                "dry_run": False,
                "requested": len(normalized_keys),
                "completed": len(completed),
                "skipped": sum(skipped.values()),
                "skipped_reasons": skipped,
            }

    def quarantine(
        self,
        key: str,
        *,
        reason: str,
        error: str | None = None,
        failure_class: str | None = None,
        result: Mapping[str, Any] | None = None,
        owner: str | None = None,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        current_time = _utc_now(now)
        normalized_result = _canonicalize(dict(result)) if result is not None else None

        def project(state: dict[str, Any]) -> tuple[dict[str, Any], str]:
            item = state["items"].get(key)
            if not isinstance(item, dict):
                raise KeyError(key)
            previous_status = str(item.get("status") or "")
            if (
                previous_status in TERMINAL_STATUSES
                and previous_status != "quarantined"
            ):
                raise InvalidTransition(
                    f"cannot quarantine terminal status {previous_status!r}"
                )
            self._validate_owner(item, owner)
            self._clear_lease(item)
            item["status"] = "quarantined"
            item["quarantine_reason"] = str(reason)[:4000]
            if error is not None:
                item["last_error"] = str(error)[:4000]
            if failure_class is not None:
                item["last_failure_class"] = str(failure_class)[:4000]
            if normalized_result is not None:
                item["result"] = normalized_result
            item["next_attempt_at"] = None
            item["updated_at"] = _iso(current_time)
            return item, previous_status

        return self._persist_transition(
            key=key,
            name="quarantined",
            now=current_time,
            dry_run=dry_run,
            project=project,
        )

    def resume_human_required(
        self,
        key: str,
        *,
        capability_fingerprint: str,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Resume after deterministic preflight proves an external fix.

        The non-empty capability fingerprint should describe the new auth/tool
        state.  This prevents a model from blindly clearing human_required.
        """

        if not capability_fingerprint.strip():
            raise ValueError("capability_fingerprint is required")
        current_time = _utc_now(now)

        def project(state: dict[str, Any]) -> tuple[dict[str, Any], str]:
            item = state["items"].get(key)
            if not isinstance(item, dict):
                raise KeyError(key)
            previous_status = str(item.get("status") or "")
            if previous_status != "human_required":
                raise InvalidTransition("only human_required items can resume")
            item["status"] = "pending_frontier"
            item["human_required"] = False
            item["capability_fingerprint"] = capability_fingerprint.strip()
            # A verified external capability change grants a fresh bounded
            # frontier window instead of resuming directly into quarantine.
            item["frontier_attempts"] = 0
            item["next_attempt_at"] = None
            item["updated_at"] = _iso(current_time)
            return item, previous_status

        return self._persist_transition(
            key=key,
            name="human_boundary_cleared",
            now=current_time,
            dry_run=dry_run,
            project=project,
        )

    def resume_due_human_required(
        self,
        *,
        capability_fingerprint: str,
        cooldown_seconds: int = 3_600,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Requeue due external-authority items after deterministic preflight.

        The human only repairs authentication, billing or secret-store state.
        Once a shared capability preflight succeeds, the next sleep cycle
        automatically gives each old terminal item a fresh bounded frontier
        window; no queue acknowledgement or content judgment is required.
        """

        fingerprint = capability_fingerprint.strip()
        if not fingerprint:
            return {
                "status": "preflight_not_ready",
                "resumed": 0,
                "results": [],
                "dry_run": dry_run,
            }
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")
        current_time = _utc_now(now)
        due: list[str] = []
        for item in self.list_items(statuses={"human_required"}):
            updated_at = _parse_iso(item.get("updated_at")) or _parse_iso(
                item.get("created_at")
            )
            if (
                updated_at is not None
                and (current_time - updated_at).total_seconds() < cooldown_seconds
            ):
                continue
            key = str(item.get("key") or "")
            if key:
                due.append(key)
        results = [
            self.resume_human_required(
                key,
                capability_fingerprint=fingerprint,
                now=current_time,
                dry_run=dry_run,
            )
            for key in due
        ]
        return {
            "status": "ok",
            "resumed": len(results),
            "results": results,
            "cooldown_seconds": cooldown_seconds,
            "dry_run": dry_run,
        }

    def _persist_transition(
        self,
        *,
        key: str,
        name: str,
        now: datetime,
        dry_run: bool,
        project: Callable[[dict[str, Any]], tuple[dict[str, Any], str]],
        event_extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if dry_run:
            state = self._load_unlocked()
            item, _previous = project(state)
            return {"dry_run": True, "item": copy.deepcopy(item)}
        with self._exclusive_lock():
            state = self._load_unlocked()
            before = copy.deepcopy(state["items"].get(key))
            item, previous_status = project(state)
            changed = item != before
            if changed:
                self._save_unlocked(state)
                self._append_event_unlocked(
                    self._event(
                        key=key,
                        name=name,
                        now=now,
                        previous_status=previous_status,
                        item=item,
                        **dict(event_extra or {}),
                    )
                )
            return {"dry_run": False, "item": copy.deepcopy(item)}


__all__ = [
    "CycleBudget",
    "CycleBudgetSlice",
    "ConvergenceError",
    "ConvergenceStateError",
    "ConvergenceStore",
    "HUMAN_REQUIRED_FAILURE_CLASSES",
    "InvalidTransition",
    "RetryPolicy",
    "TERMINAL_STATUSES",
    "canonical_json",
    "exponential_backoff_seconds",
    "frontier_failure_class",
    "input_fingerprint",
    "is_human_required_failure",
    "is_human_required_result",
    "stable_item_key",
]
