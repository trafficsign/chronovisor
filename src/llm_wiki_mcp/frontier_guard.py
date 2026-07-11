"""Durable admission control for exceptional frontier code repair.

Frontier execution is deliberately *not* a normal review tier.  This module
admits only repeatedly reproduced system-code failures after local repair has
already failed, then enforces a process-wide single flight and durable daily
budget.  Reservation and execution start are separate transitions: merely
inspecting or reserving an incident never spends frontier budget.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import socket
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4

from llm_wiki_mcp.wiki import WIKI_ROOT


SCHEMA_VERSION = 1
DEFAULT_GUARD_ROOT = WIKI_ROOT / "runtime" / "frontier-repair"
DEFAULT_WINDOW = timedelta(hours=24)
DEFAULT_LEASE = timedelta(hours=2)
TERMINAL_STATUSES = frozenset(
    {
        "succeeded",
        "failed",
        "quarantined",
        "human_required",
        "abandoned",
        "released",
    }
)
FINISH_OUTCOMES = TERMINAL_STATUSES - {"abandoned", "released"}

_FINGERPRINT_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,512}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HUMAN_BOUNDARY_TOKENS = frozenset(
    {
        "auth",
        "authentication",
        "authorization",
        "unauthorized",
        "oauth",
        "billing",
        "quota",
        "keychain",
        "credential",
    }
)
_HUMAN_BOUNDARY_MARKERS = (
    "auth_required",
    "authentication",
    "authentication_required",
    "authorization",
    "authorization_required",
    "oauth",
    "oauth_required",
    "billing",
    "quota",
    "keychain",
    "credential_permission",
    "secret_store",
    "secretstore",
)
_TRUSTED_PRODUCER_CONTRACTS = frozenset(
    {
        (
            "trusted_watchdog",
            "watchdog.health_snapshot",
            "system_health_snapshot_exception",
        ),
    }
)


class FrontierGuardError(RuntimeError):
    """Base class for guard failures."""


class EvidenceValidationError(ValueError):
    """The incident does not prove eligibility for frontier code repair."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


class PermitDenied(FrontierGuardError):
    """A valid incident was denied by a durable admission rule."""

    def __init__(
        self,
        reason: str,
        *,
        incident_id: str | None = None,
        retry_at: datetime | None = None,
    ):
        self.reason = reason
        self.incident_id = incident_id
        self.retry_at = _utc_now(retry_at) if retry_at is not None else None
        suffix = f" ({incident_id})" if incident_id else ""
        super().__init__(f"frontier repair denied: {reason}{suffix}")


class FrontierStateError(FrontierGuardError):
    """The durable guard state is unreadable or internally inconsistent."""


def repair_fingerprint(*parts: object) -> str:
    """Return a deterministic fingerprint for normalized failure evidence."""

    encoded = json.dumps(
        parts,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clean_optional(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_sequence(value: Sequence[object] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    values: Sequence[object] = (value,) if isinstance(value, str) else value
    return tuple(text for item in values if (text := str(item).strip()))


@dataclass(frozen=True)
class RepairIncidentEvidence:
    """Strict evidence envelope for the only permitted frontier role.

    ``distinct_inputs`` should contain stable input identifiers or hashes, not
    raw user content.  A count alone is intentionally insufficient: the guard
    needs proof that one malformed input is not being retried repeatedly.
    """

    component: str
    fingerprint: str
    failure_class: str
    occurrence_count: int
    distinct_inputs: tuple[str, ...]
    local_repair_attempts: int
    local_repair_evidence: tuple[str, ...]
    reproduction_command: tuple[str, ...] = ()
    failing_test: str | None = None
    reproduction_artifact: str | None = None
    all_local_models_unavailable: bool = False
    local_unavailability_artifact: str | None = None
    role: str = "code_repair"
    incident_kind: str = "system_code_repair"
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "component", str(self.component).strip())
        object.__setattr__(self, "fingerprint", str(self.fingerprint).strip())
        object.__setattr__(self, "failure_class", str(self.failure_class).strip())
        object.__setattr__(self, "role", str(self.role).strip())
        object.__setattr__(self, "incident_kind", str(self.incident_kind).strip())
        object.__setattr__(self, "distinct_inputs", _clean_sequence(self.distinct_inputs))
        object.__setattr__(
            self,
            "local_repair_evidence",
            _clean_sequence(self.local_repair_evidence),
        )
        object.__setattr__(
            self,
            "reproduction_command",
            _clean_sequence(self.reproduction_command),
        )
        object.__setattr__(self, "failing_test", _clean_optional(self.failing_test))
        object.__setattr__(
            self,
            "reproduction_artifact",
            _clean_optional(self.reproduction_artifact),
        )
        object.__setattr__(
            self,
            "local_unavailability_artifact",
            _clean_optional(self.local_unavailability_artifact),
        )
        object.__setattr__(self, "notes", copy.deepcopy(dict(self.notes)))
        self.validate()

    @property
    def fingerprint_key(self) -> str:
        """Opaque state key so readable legacy fingerprints remain supported."""

        return hashlib.sha256(self.fingerprint.encode("utf-8")).hexdigest()

    @property
    def incident_key(self) -> str:
        value = self.notes.get("incident_key") if isinstance(self.notes, Mapping) else None
        return str(value or "").strip()

    @property
    def incident_key_hash(self) -> str:
        return hashlib.sha256(self.incident_key.encode("utf-8")).hexdigest()

    @property
    def distinct_input_count(self) -> int:
        return len(set(self.distinct_inputs))

    def validate(self) -> None:
        errors: list[str] = []
        if self.role != "code_repair":
            errors.append("role must be code_repair")
        if self.incident_kind != "system_code_repair":
            errors.append("incident_kind must be system_code_repair")
        if not self.component:
            errors.append("component is required")
        if not self.fingerprint or not _FINGERPRINT_RE.fullmatch(self.fingerprint):
            errors.append("fingerprint must be a stable printable value (1-512 chars)")
        if not self.failure_class:
            errors.append("failure_class is required")
        producer = self.notes.get("producer") if isinstance(self.notes, Mapping) else None
        if (producer, self.component, self.failure_class) not in _TRUSTED_PRODUCER_CONTRACTS:
            errors.append(
                "evidence must come from an allowlisted trusted system-incident producer"
            )
        if not self.incident_key or not _FINGERPRINT_RE.fullmatch(self.incident_key):
            errors.append("trusted evidence requires a stable printable incident_key")
        normalized_failure = self.failure_class.lower().replace("-", "_").replace(" ", "_")
        failure_tokens = frozenset(normalized_failure.split("_"))
        if _HUMAN_BOUNDARY_TOKENS & failure_tokens or any(
            marker in normalized_failure for marker in _HUMAN_BOUNDARY_MARKERS
        ):
            errors.append(
                "auth, billing, quota, keychain, and credential failures are human boundaries"
            )
        if (
            isinstance(self.occurrence_count, bool)
            or not isinstance(self.occurrence_count, int)
            or self.occurrence_count < 0
        ):
            errors.append("occurrence_count must be a non-negative integer")
        if not isinstance(self.all_local_models_unavailable, bool):
            errors.append("all_local_models_unavailable must be a boolean")
        if (
            isinstance(self.occurrence_count, int)
            and not isinstance(self.occurrence_count, bool)
            and self.all_local_models_unavailable is not True
            and self.occurrence_count < 3
        ):
            errors.append("occurrence_count must be at least 3")
        if self.distinct_input_count < 2:
            errors.append("at least 2 distinct input identifiers are required")
        if (
            isinstance(self.local_repair_attempts, bool)
            or not isinstance(self.local_repair_attempts, int)
            or self.local_repair_attempts < 2
        ):
            errors.append("at least 2 local repair attempts are required")
        if (
            len(self.local_repair_evidence) != self.local_repair_attempts
            or len(set(self.local_repair_evidence)) != self.local_repair_attempts
            or any(not _SHA256_RE.fullmatch(item) for item in self.local_repair_evidence)
        ):
            errors.append(
                "each local repair attempt requires one unique SHA-256 evidence digest"
            )
        if self.all_local_models_unavailable is True and not self.local_unavailability_artifact:
            errors.append("all-local-unavailable requires a health-check artifact")
        if not (
            self.reproduction_command
            or self.failing_test
            or self.reproduction_artifact
        ):
            errors.append("a reproduction command, failing test, or artifact is required")
        for input_id in self.distinct_inputs:
            if len(input_id) > 512 or any(ord(char) < 32 for char in input_id):
                errors.append("distinct input identifiers must be printable and <=512 chars")
                break
        try:
            json.dumps(self.notes, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            errors.append("notes must be JSON serializable")
        if errors:
            raise EvidenceValidationError(errors)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "role": self.role,
            "incident_kind": self.incident_kind,
            "component": self.component,
            "fingerprint": self.fingerprint,
            "fingerprint_key": self.fingerprint_key,
            "failure_class": self.failure_class,
            "occurrence_count": self.occurrence_count,
            "distinct_inputs": list(self.distinct_inputs),
            "distinct_input_count": self.distinct_input_count,
            "local_repair_attempts": self.local_repair_attempts,
            "local_repair_evidence": list(self.local_repair_evidence),
            "reproduction": {
                "command": list(self.reproduction_command),
                "failing_test": self.failing_test,
                "artifact": self.reproduction_artifact,
            },
            "all_local_models_unavailable": self.all_local_models_unavailable,
            "local_unavailability_artifact": self.local_unavailability_artifact,
            "notes": copy.deepcopy(dict(self.notes)),
        }


@dataclass(frozen=True)
class GuardInspection:
    """A read snapshot plus projected stale recovery information."""

    state: Mapping[str, Any]
    would_abandon: tuple[str, ...]
    dry_run: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": copy.deepcopy(dict(self.state)),
            "would_abandon": list(self.would_abandon),
            "dry_run": self.dry_run,
        }


def _utc_now(value: datetime | None = None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        return current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _utc_now(parsed)


def _pid_alive(pid: object) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _default_state() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "revision": 0,
        "event_sequence": 0,
        "active_incident_id": None,
        "incidents": {},
        "fingerprints": {},
        "incident_keys": {},
    }


@dataclass
class RepairPermit:
    """Capability returned by :meth:`FrontierGuard.reserve` or ``permit``."""

    guard: "FrontierGuard" = field(repr=False)
    incident_id: str
    owner: str
    evidence: RepairIncidentEvidence
    _token: str = field(repr=False)

    @property
    def status(self) -> str:
        return self.guard.incident_status(self.incident_id)

    def start(
        self,
        *,
        pid: int | None = None,
        now: datetime | None = None,
        lease: timedelta | None = None,
    ) -> dict[str, Any]:
        """Spend budget immediately before the frontier subprocess starts."""

        return self.guard.start(
            self.incident_id,
            token=self._token,
            pid=os.getpid() if pid is None else pid,
            now=now,
            lease=lease,
        )

    def heartbeat(
        self,
        *,
        now: datetime | None = None,
        lease: timedelta | None = None,
    ) -> dict[str, Any]:
        return self.guard.heartbeat(
            self.incident_id,
            token=self._token,
            now=now,
            lease=lease,
        )

    def finish(
        self,
        outcome: str,
        *,
        details: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        return self.guard.finish(
            self.incident_id,
            token=self._token,
            outcome=outcome,
            details=details,
            now=now,
        )

    def abandon(
        self,
        reason: str,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        return self.guard.abandon(
            self.incident_id,
            token=self._token,
            reason=reason,
            now=now,
        )

    def release(self, *, now: datetime | None = None) -> dict[str, Any]:
        return self.guard.release(self.incident_id, token=self._token, now=now)


class FrontierGuard:
    """Atomic, durable frontier code-repair admission controller."""

    def __init__(
        self,
        root: Path | str = DEFAULT_GUARD_ROOT,
        *,
        fingerprint_cooldown: timedelta = DEFAULT_WINDOW,
        global_window: timedelta = DEFAULT_WINDOW,
        global_limit: int = 1,
        default_lease: timedelta = DEFAULT_LEASE,
    ) -> None:
        self.root = Path(root)
        self.state_file = self.root / "state.json"
        self.events_file = self.root / "events.jsonl"
        self.state_lock_file = self.root / "state.lock"
        self.run_lock_file = self.root / "single-flight.lock"
        self.fingerprint_cooldown = fingerprint_cooldown
        self.global_window = global_window
        self.global_limit = global_limit
        self.default_lease = default_lease
        if fingerprint_cooldown.total_seconds() <= 0:
            raise ValueError("fingerprint_cooldown must be positive")
        if global_window.total_seconds() <= 0:
            raise ValueError("global_window must be positive")
        if global_limit < 1:
            raise ValueError("global_limit must be at least 1")
        if default_lease.total_seconds() <= 0:
            raise ValueError("default_lease must be positive")

    def _read_state(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return _default_state()
        try:
            value = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrontierStateError(f"cannot read frontier guard state: {exc}") from exc
        if not isinstance(value, dict):
            raise FrontierStateError("frontier guard state must be an object")
        if value.get("schema_version") != SCHEMA_VERSION:
            raise FrontierStateError("unsupported frontier guard schema")
        if not isinstance(value.get("incidents"), dict):
            raise FrontierStateError("frontier guard incidents must be an object")
        if not isinstance(value.get("fingerprints"), dict):
            raise FrontierStateError("frontier guard fingerprints must be an object")
        value.setdefault("incident_keys", {})
        if not isinstance(value.get("incident_keys"), dict):
            raise FrontierStateError("frontier guard incident_keys must be an object")
        return value

    @contextmanager
    def _state_lock(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.state_lock_file, os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(fd, "a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _atomic_write_state(self, state: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.state_file.name}.", suffix=".tmp", dir=self.root
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.state_file)
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _append_events(self, events: Sequence[Mapping[str, Any]]) -> None:
        if not events:
            return
        fd = os.open(
            self.events_file,
            os.O_CREAT | os.O_APPEND | os.O_WRONLY,
            0o600,
        )
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            for event in events:
                handle.write(
                    json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())

    def _commit(
        self,
        state: dict[str, Any],
        events: list[dict[str, Any]],
        *,
        now: datetime,
    ) -> None:
        if not events:
            return
        for event in events:
            state["event_sequence"] = int(state.get("event_sequence") or 0) + 1
            event["sequence"] = state["event_sequence"]
            event.setdefault("timestamp", _timestamp(now))
        state["revision"] = int(state.get("revision") or 0) + 1
        state["updated_at"] = _timestamp(now)
        self._atomic_write_state(state)
        self._append_events(events)

    def _recover_stale(
        self,
        state: dict[str, Any],
        *,
        now: datetime,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        active_id = state.get("active_incident_id")
        incident = state.get("incidents", {}).get(active_id) if active_id else None
        if not isinstance(incident, dict) or incident.get("status") not in {"reserved", "started"}:
            if active_id is not None:
                state["active_incident_id"] = None
            return events
        expiry = _parse_timestamp(incident.get("lease_expires_at"))
        # ``pid`` may be the short-lived frontier child.  A child exiting is
        # not a crash: its live owner still needs a chance to persist finish.
        # Owner death is the crash signal; lease expiry covers a hung owner or
        # a child whose result was never collected.
        owner_pid = incident.get("owner_pid")
        reason: str | None = None
        if expiry is not None and expiry <= now:
            reason = "lease_expired"
        elif not _pid_alive(owner_pid):
            reason = "owner_process_exited"
        if reason is None:
            return events
        prior_status = str(incident["status"])
        incident.update(
            {
                "status": "abandoned",
                "abandoned_at": _timestamp(now),
                "finished_at": _timestamp(now),
                "abandon_reason": reason,
                "lease_expires_at": None,
            }
        )
        state["active_incident_id"] = None
        events.append(
            {
                "event": "incident_abandoned",
                "incident_id": active_id,
                "fingerprint_key": incident.get("fingerprint_key"),
                "prior_status": prior_status,
                "reason": reason,
                "stale_recovery": True,
            }
        )
        return events

    def _save_recovery_if_needed(
        self,
        state: dict[str, Any],
        events: list[dict[str, Any]],
        *,
        now: datetime,
    ) -> None:
        if events:
            self._commit(state, events, now=now)

    def _assert_budget(
        self,
        state: Mapping[str, Any],
        evidence: RepairIncidentEvidence,
        *,
        now: datetime,
    ) -> None:
        prior_incident = state.get("incident_keys", {}).get(evidence.incident_key_hash)
        if isinstance(prior_incident, dict) and prior_incident.get("started_at"):
            raise PermitDenied(
                "incident_already_started",
                incident_id=str(prior_incident.get("incident_id") or "") or None,
            )
        prior = state.get("fingerprints", {}).get(evidence.fingerprint_key)
        if isinstance(prior, dict):
            last_started = _parse_timestamp(prior.get("last_started_at"))
            if last_started is not None and now - last_started < self.fingerprint_cooldown:
                raise PermitDenied(
                    "fingerprint_cooldown",
                    incident_id=str(prior.get("last_incident_id") or "") or None,
                    retry_at=last_started + self.fingerprint_cooldown,
                )
        recent_starts: list[datetime] = []
        for incident in state.get("incidents", {}).values():
            if not isinstance(incident, dict):
                continue
            started_at = _parse_timestamp(incident.get("started_at"))
            if started_at is not None and now - started_at < self.global_window:
                recent_starts.append(started_at)
        if len(recent_starts) >= self.global_limit:
            recent_starts.sort()
            raise PermitDenied(
                "global_24h_budget",
                retry_at=recent_starts[-self.global_limit] + self.global_window,
            )

    @staticmethod
    def _event_for(
        name: str,
        incident: Mapping[str, Any],
        **details: Any,
    ) -> dict[str, Any]:
        return {
            "event": name,
            "incident_id": incident.get("incident_id"),
            "fingerprint_key": incident.get("fingerprint_key"),
            "owner": incident.get("owner"),
            **details,
        }

    def reserve(
        self,
        evidence: RepairIncidentEvidence,
        *,
        owner: str | None = None,
        owner_pid: int | None = None,
        now: datetime | None = None,
        lease: timedelta | None = None,
    ) -> RepairPermit:
        """Reserve the one global slot without consuming frontier budget."""

        evidence.validate()
        current = _utc_now(now)
        effective_lease = lease or self.default_lease
        if effective_lease.total_seconds() <= 0:
            raise ValueError("lease must be positive")
        effective_owner_pid = owner_pid or os.getpid()
        if effective_owner_pid <= 0:
            raise ValueError("owner_pid must be positive")
        effective_owner = (owner or f"{socket.gethostname()}:{effective_owner_pid}").strip()
        if not effective_owner:
            raise ValueError("owner is required")
        incident_id = uuid4().hex
        token = secrets.token_urlsafe(32)
        with self._state_lock():
            state = self._read_state()
            recovery = self._recover_stale(state, now=current)
            try:
                active_id = state.get("active_incident_id")
                if active_id:
                    active = state.get("incidents", {}).get(active_id)
                    retry_at = (
                        _parse_timestamp(active.get("lease_expires_at"))
                        if isinstance(active, dict)
                        else None
                    )
                    raise PermitDenied(
                        "active_incident",
                        incident_id=str(active_id),
                        retry_at=retry_at,
                    )
                self._assert_budget(state, evidence, now=current)
            except PermitDenied:
                self._save_recovery_if_needed(state, recovery, now=current)
                raise
            incident = {
                "incident_id": incident_id,
                "status": "reserved",
                "reserved_at": _timestamp(current),
                "started_at": None,
                "finished_at": None,
                "lease_expires_at": _timestamp(current + effective_lease),
                "owner": effective_owner,
                "owner_pid": effective_owner_pid,
                "pid": None,
                "permit_token_hash": hashlib.sha256(token.encode("utf-8")).hexdigest(),
                "fingerprint": evidence.fingerprint,
                "fingerprint_key": evidence.fingerprint_key,
                "incident_key_hash": evidence.incident_key_hash,
                "evidence": evidence.to_dict(),
            }
            state["incidents"][incident_id] = incident
            state["active_incident_id"] = incident_id
            recovery.append(self._event_for("incident_reserved", incident))
            self._commit(state, recovery, now=current)
        return RepairPermit(self, incident_id, effective_owner, evidence, token)

    def _owned_incident(
        self,
        state: Mapping[str, Any],
        incident_id: str,
        token: str,
    ) -> dict[str, Any]:
        incident = state.get("incidents", {}).get(incident_id)
        if not isinstance(incident, dict):
            raise PermitDenied("unknown_incident", incident_id=incident_id)
        expected = str(incident.get("permit_token_hash") or "")
        actual = hashlib.sha256(token.encode("utf-8")).hexdigest()
        if not expected or not hmac.compare_digest(expected, actual):
            raise PermitDenied("permit_owner_mismatch", incident_id=incident_id)
        return incident

    def start(
        self,
        incident_id: str,
        *,
        token: str,
        pid: int,
        now: datetime | None = None,
        lease: timedelta | None = None,
    ) -> dict[str, Any]:
        """Consume budget once, immediately before the frontier process starts."""

        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("pid must be a positive integer")
        current = _utc_now(now)
        effective_lease = lease or self.default_lease
        if effective_lease.total_seconds() <= 0:
            raise ValueError("lease must be positive")
        with self._state_lock():
            state = self._read_state()
            recovery = self._recover_stale(state, now=current)
            try:
                incident = self._owned_incident(state, incident_id, token)
                status = str(incident.get("status"))
                if status == "started" or incident.get("started_at"):
                    raise PermitDenied("incident_already_started", incident_id=incident_id)
                if status != "reserved":
                    raise PermitDenied(f"incident_not_startable:{status}", incident_id=incident_id)
                if state.get("active_incident_id") != incident_id:
                    raise PermitDenied("incident_not_active", incident_id=incident_id)
                evidence = RepairIncidentEvidence(
                    component=incident["evidence"]["component"],
                    fingerprint=incident["evidence"]["fingerprint"],
                    failure_class=incident["evidence"]["failure_class"],
                    occurrence_count=incident["evidence"]["occurrence_count"],
                    distinct_inputs=tuple(incident["evidence"]["distinct_inputs"]),
                    local_repair_attempts=incident["evidence"]["local_repair_attempts"],
                    local_repair_evidence=tuple(
                        incident["evidence"]["local_repair_evidence"]
                    ),
                    reproduction_command=tuple(
                        incident["evidence"]["reproduction"]["command"]
                    ),
                    failing_test=incident["evidence"]["reproduction"]["failing_test"],
                    reproduction_artifact=incident["evidence"]["reproduction"]["artifact"],
                    all_local_models_unavailable=incident["evidence"][
                        "all_local_models_unavailable"
                    ],
                    local_unavailability_artifact=incident["evidence"][
                        "local_unavailability_artifact"
                    ],
                    role=incident["evidence"]["role"],
                    incident_kind=incident["evidence"]["incident_kind"],
                    notes=incident["evidence"].get("notes") or {},
                )
                self._assert_budget(state, evidence, now=current)
            except PermitDenied:
                self._save_recovery_if_needed(state, recovery, now=current)
                raise
            incident.update(
                {
                    "status": "started",
                    "started_at": _timestamp(current),
                    "pid": pid,
                    "lease_expires_at": _timestamp(current + effective_lease),
                }
            )
            state["fingerprints"][incident["fingerprint_key"]] = {
                "last_incident_id": incident_id,
                "last_started_at": incident["started_at"],
            }
            state["incident_keys"][incident["incident_key_hash"]] = {
                "incident_id": incident_id,
                "started_at": incident["started_at"],
            }
            recovery.append(self._event_for("incident_started", incident, pid=pid))
            self._commit(state, recovery, now=current)
            return copy.deepcopy(incident)

    def heartbeat(
        self,
        incident_id: str,
        *,
        token: str,
        now: datetime | None = None,
        lease: timedelta | None = None,
    ) -> dict[str, Any]:
        current = _utc_now(now)
        effective_lease = lease or self.default_lease
        if effective_lease.total_seconds() <= 0:
            raise ValueError("lease must be positive")
        with self._state_lock():
            state = self._read_state()
            recovery = self._recover_stale(state, now=current)
            try:
                incident = self._owned_incident(state, incident_id, token)
                if incident.get("status") != "started":
                    raise PermitDenied("incident_not_running", incident_id=incident_id)
            except PermitDenied:
                self._save_recovery_if_needed(state, recovery, now=current)
                raise
            incident["lease_expires_at"] = _timestamp(current + effective_lease)
            incident["last_heartbeat_at"] = _timestamp(current)
            recovery.append(self._event_for("incident_heartbeat", incident))
            self._commit(state, recovery, now=current)
            return copy.deepcopy(incident)

    def finish(
        self,
        incident_id: str,
        *,
        token: str,
        outcome: str,
        details: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if outcome not in FINISH_OUTCOMES:
            raise ValueError(f"invalid frontier outcome: {outcome}")
        result = copy.deepcopy(dict(details or {}))
        try:
            json.dumps(result, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("details must be JSON serializable") from exc
        current = _utc_now(now)
        with self._state_lock():
            state = self._read_state()
            recovery = self._recover_stale(state, now=current)
            try:
                incident = self._owned_incident(state, incident_id, token)
                if incident.get("status") != "started":
                    raise PermitDenied(
                        f"incident_not_finishable:{incident.get('status')}",
                        incident_id=incident_id,
                    )
            except PermitDenied:
                self._save_recovery_if_needed(state, recovery, now=current)
                raise
            incident.update(
                {
                    "status": outcome,
                    "finished_at": _timestamp(current),
                    "lease_expires_at": None,
                    "result": result,
                }
            )
            if state.get("active_incident_id") == incident_id:
                state["active_incident_id"] = None
            recovery.append(
                self._event_for("incident_finished", incident, outcome=outcome)
            )
            self._commit(state, recovery, now=current)
            return copy.deepcopy(incident)

    def abandon(
        self,
        incident_id: str,
        *,
        token: str,
        reason: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = _utc_now(now)
        clean_reason = str(reason).strip()
        if not clean_reason:
            raise ValueError("abandon reason is required")
        with self._state_lock():
            state = self._read_state()
            recovery = self._recover_stale(state, now=current)
            try:
                incident = self._owned_incident(state, incident_id, token)
                if incident.get("status") not in {"reserved", "started"}:
                    raise PermitDenied(
                        f"incident_not_abandonable:{incident.get('status')}",
                        incident_id=incident_id,
                    )
            except PermitDenied:
                self._save_recovery_if_needed(state, recovery, now=current)
                raise
            prior_status = str(incident["status"])
            incident.update(
                {
                    "status": "abandoned",
                    "abandoned_at": _timestamp(current),
                    "finished_at": _timestamp(current),
                    "abandon_reason": clean_reason,
                    "lease_expires_at": None,
                }
            )
            if state.get("active_incident_id") == incident_id:
                state["active_incident_id"] = None
            recovery.append(
                self._event_for(
                    "incident_abandoned",
                    incident,
                    prior_status=prior_status,
                    reason=clean_reason,
                    stale_recovery=False,
                )
            )
            self._commit(state, recovery, now=current)
            return copy.deepcopy(incident)

    def release(
        self,
        incident_id: str,
        *,
        token: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Release an unstarted reservation without spending budget."""

        current = _utc_now(now)
        with self._state_lock():
            state = self._read_state()
            recovery = self._recover_stale(state, now=current)
            try:
                incident = self._owned_incident(state, incident_id, token)
                if incident.get("status") != "reserved":
                    raise PermitDenied(
                        f"incident_not_releasable:{incident.get('status')}",
                        incident_id=incident_id,
                    )
            except PermitDenied:
                self._save_recovery_if_needed(state, recovery, now=current)
                raise
            incident.update(
                {
                    "status": "released",
                    "finished_at": _timestamp(current),
                    "lease_expires_at": None,
                }
            )
            if state.get("active_incident_id") == incident_id:
                state["active_incident_id"] = None
            recovery.append(self._event_for("incident_released", incident))
            self._commit(state, recovery, now=current)
            return copy.deepcopy(incident)

    def incident_status(self, incident_id: str) -> str:
        # This is deliberately the persisted status, not a stale-recovery
        # projection.  Context-manager cleanup must still run when a synthetic
        # clock was supplied by a test or replay.  Call ``inspect`` explicitly
        # when the caller wants projected crash recovery.
        state = self._read_state()
        incident = state.get("incidents", {}).get(incident_id)
        if not isinstance(incident, Mapping):
            raise PermitDenied("unknown_incident", incident_id=incident_id)
        return str(incident.get("status"))

    def inspect(
        self,
        *,
        dry_run: bool = True,
        now: datetime | None = None,
    ) -> GuardInspection:
        """Inspect state; dry mode never creates, locks, writes, or appends."""

        current = _utc_now(now)
        if dry_run:
            state = self._read_state()
            projected = copy.deepcopy(state)
            events = self._recover_stale(projected, now=current)
            return GuardInspection(
                state=projected,
                would_abandon=tuple(
                    str(event["incident_id"])
                    for event in events
                    if event.get("event") == "incident_abandoned"
                ),
                dry_run=True,
            )
        with self._state_lock():
            state = self._read_state()
            events = self._recover_stale(state, now=current)
            self._commit(state, events, now=current)
            return GuardInspection(
                state=copy.deepcopy(state),
                would_abandon=tuple(
                    str(event["incident_id"])
                    for event in events
                    if event.get("event") == "incident_abandoned"
                ),
                dry_run=False,
            )

    @contextmanager
    def permit(
        self,
        evidence: RepairIncidentEvidence,
        *,
        owner: str | None = None,
        owner_pid: int | None = None,
        now: datetime | None = None,
        lease: timedelta | None = None,
    ) -> Iterator[RepairPermit]:
        """Hold the cross-process single-flight lock for one repair attempt."""

        self.root.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.run_lock_file, os.O_CREAT | os.O_RDWR, 0o600)
        handle = os.fdopen(fd, "a+")
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PermitDenied("global_single_flight") from exc
            reserved = self.reserve(
                evidence,
                owner=owner,
                owner_pid=owner_pid,
                now=now,
                lease=lease,
            )
            try:
                yield reserved
            except BaseException:
                status = reserved.status
                if status in {"reserved", "started"}:
                    reserved.abandon("permit_context_exception")
                raise
            else:
                status = reserved.status
                if status == "reserved":
                    reserved.release()
                elif status == "started":
                    reserved.abandon("permit_context_exited_without_finish")
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    acquire = permit
