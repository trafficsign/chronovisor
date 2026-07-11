"""Trusted producer for exceptional system-code repair incidents.

Routine model output, semantic disagreement, and content failures must never
reach the frontier repair plane.  This module therefore exposes one narrow
producer for an internal watchdog component: a reproducible exception from
``health_snapshot``.  Observations are durable, privacy-bounded, and admitted
only after independent recurrence plus two deterministic local rechecks.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from llm_wiki_mcp import wiki
from llm_wiki_mcp.frontier_guard import RepairIncidentEvidence, repair_fingerprint


SCHEMA_VERSION = 1
TRUSTED_HEALTH_COMPONENT = "watchdog.health_snapshot"
MIN_OCCURRENCES = 3
MIN_DISTINCT_IDENTITIES = 2
LOCAL_RECHECK_ATTEMPTS = 2
MAX_IDENTITIES = 64

HEALTH_REPRODUCTION_COMMAND = (
    "uv",
    "run",
    "python",
    "-c",
    "from llm_wiki_mcp.health import health_snapshot; health_snapshot()",
)
HEALTH_FAILING_TEST = "runtime:llm_wiki_mcp.health.health_snapshot"

_HUMAN_BOUNDARY_MARKERS = (
    "auth",
    "authentication",
    "authorization",
    "oauth",
    "billing",
    "quota",
    "keychain",
    "credential",
    "secret store",
    "secret-store",
)
_ROUTINE_MODEL_MARKERS = (
    "semantic disagreement",
    "content disagreement",
    "structured output",
    "structured-output",
    "model output",
    "model-output",
    "llm response",
    "json schema",
    "json-schema",
)
_ROUTINE_EXCEPTION_TYPES = frozenset(
    {
        "JSONDecodeError",
        "UnicodeDecodeError",
    }
)
_INTERNAL_EXCEPTION_TYPES = frozenset(
    {
        "AssertionError",
        "AttributeError",
        "ImportError",
        "ModuleNotFoundError",
        "NameError",
        "NotImplementedError",
        "TypeError",
    }
)
_INTERNAL_RUNTIME_MARKERS = (
    "adapter contract",
    "code invariant",
    "index invariant",
    "internal invariant",
    "unreachable state",
)
_OPERATIONAL_MARKERS = (
    "configuration",
    "connection",
    "database is locked",
    "disk",
    "duplicate page_id",
    "file exists",
    "file not found",
    "model unavailable",
    "network",
    "no space",
    "ollama",
    "permission",
    "read-only",
    "timeout",
)
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_HEX_RE = re.compile(r"\b[0-9a-f]{12,}\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\b\d+\b")
_PATH_RE = re.compile(r"(?:~|/)[^\s:'\"]+")
_SPACE_RE = re.compile(r"\s+")


class IncidentStateError(RuntimeError):
    """The trusted incident ledger is corrupt or internally inconsistent."""


@dataclass(frozen=True)
class SafeDiagnostic:
    """Persistable exception identity with no raw message or stack."""

    exception_type: str
    diagnostic_hash: str
    summary: str

    def to_dict(self) -> dict[str, str]:
        return {
            "exception_type": self.exception_type,
            "diagnostic_hash": self.diagnostic_hash,
            "summary": self.summary,
        }


def _utc_now(value: datetime | None = None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        return current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return _utc_now(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _normalized_exception_text(error: BaseException) -> str:
    """Normalize volatile details in memory before hashing; never persist it."""

    text = f"{error.__class__.__name__}:{error}"
    text = _UUID_RE.sub("<id>", text)
    text = _HEX_RE.sub("<hex>", text)
    text = _PATH_RE.sub("<path>", text)
    text = _NUMBER_RE.sub("<n>", text)
    return _SPACE_RE.sub(" ", text).strip().casefold()[:4096]


def safe_exception_diagnostic(error: BaseException) -> SafeDiagnostic:
    """Return a bounded diagnostic that cannot disclose the exception body."""

    exception_type = re.sub(r"[^A-Za-z0-9_.-]", "_", error.__class__.__name__)[:120]
    normalized = _normalized_exception_text(error)
    return SafeDiagnostic(
        exception_type=exception_type or "Exception",
        diagnostic_hash=_hash_text(normalized),
        summary="trusted health component raised a reproducible exception",
    )


def _classification(error: BaseException) -> str:
    normalized = _normalized_exception_text(error)
    if any(marker in normalized for marker in _HUMAN_BOUNDARY_MARKERS):
        return "human_boundary"
    if error.__class__.__name__ in _ROUTINE_EXCEPTION_TYPES:
        return "routine_model_or_data_error"
    if any(marker in normalized for marker in _ROUTINE_MODEL_MARKERS):
        return "routine_model_or_data_error"
    if isinstance(error, OSError) or any(marker in normalized for marker in _OPERATIONAL_MARKERS):
        return "operational_failure"
    if error.__class__.__name__ in _INTERNAL_EXCEPTION_TYPES:
        return "system_exception"
    if isinstance(error, RuntimeError) and any(
        marker in normalized for marker in _INTERNAL_RUNTIME_MARKERS
    ):
        return "system_exception"
    return "unclassified_failure"


def _default_state() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "incidents": {}}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _default_state()
    except OSError as exc:
        raise IncidentStateError(f"cannot read incident state: {exc.__class__.__name__}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise IncidentStateError("incident state is not valid JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != SCHEMA_VERSION
        or not isinstance(payload.get("incidents"), dict)
    ):
        raise IncidentStateError("incident state has an unsupported shape")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


@contextmanager
def _state_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _identity_hash(run_id: str, input_id: str | None) -> str:
    # Only the digest is persisted: run/session paths and user identifiers are
    # deliberately absent from the durable ledger and repair packet.
    return _hash_text(
        json.dumps(
            [str(run_id), str(input_id or run_id)],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def _repair_evidence_digest(
    *,
    attempt: int,
    action_id: str,
    diagnostic: SafeDiagnostic,
) -> str:
    return _hash_text(
        json.dumps(
            {
                "attempt": attempt,
                "action_id": action_id,
                "verification_exception_type": diagnostic.exception_type,
                "verification_diagnostic_hash": diagnostic.diagnostic_hash,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _reset_derived_index_singletons() -> None:
    from llm_wiki_mcp import index_store, search

    with index_store._store_lock:
        index_store._store = None
    with search._BM25_LOCK:
        search._BM25_SINGLETON = None


def _default_health_repair(attempt: int, *, dry_run: bool) -> Mapping[str, Any]:
    """Perform one bounded, reversible repair of derived health state."""

    from llm_wiki_mcp.decision_policy import resolve_decision_policy

    policy, mode, error = resolve_decision_policy("derived_index_rebuild")
    if (
        error is not None
        or policy is None
        or policy.kind != "validated_local"
        or mode != "enabled"
    ):
        raise RuntimeError(error or "derived index rebuild lane is disabled")

    if attempt == 1:
        if not dry_run:
            _reset_derived_index_singletons()
        return {
            "action_id": "reset-derived-index-singletons",
            "performed": not dry_run,
            "projected": dry_run,
        }
    if attempt != 2:
        raise ValueError("unsupported health repair attempt")

    from llm_wiki_mcp import index_store, search

    cache_paths = (
        index_store.PAGES_INDEX_FILE,
        index_store.BACKLINKS_INDEX_FILE,
        search._BM25_CACHE_FILE,
    )
    existing = [path for path in cache_paths if path.exists()]
    if not dry_run:
        quarantine = (
            wiki.WIKI_ROOT
            / "runtime"
            / "system-incidents"
            / "cache-quarantine"
            / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}-{os.getpid()}"
        )
        quarantine.mkdir(parents=True, exist_ok=False)
        for path in existing:
            os.replace(path, quarantine / path.name)
        _reset_derived_index_singletons()
    return {
        "action_id": "quarantine-derived-index-cache-and-rebuild",
        "performed": not dry_run,
        "projected": dry_run,
        "derived_cache_files": len(existing),
    }


def _run_local_repairs(
    runner: Callable[[int], Any],
    *,
    expected: SafeDiagnostic,
    repairer: Callable[..., Mapping[str, Any]] | None,
    dry_run: bool,
) -> tuple[bool, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    if repairer is None:
        return False, [
            {
                "attempt": 0,
                "status": "repair_unavailable",
                "action_id": None,
            }
        ]
    for attempt in range(1, LOCAL_RECHECK_ATTEMPTS + 1):
        try:
            raw_action = repairer(attempt, dry_run=dry_run)
        except Exception as exc:
            diagnostics.append(
                {
                    "attempt": attempt,
                    "status": "repair_failed",
                    "action_id": None,
                    "repair_error_type": exc.__class__.__name__,
                }
            )
            return False, diagnostics
        action = dict(raw_action) if isinstance(raw_action, Mapping) else {}
        action_id = str(action.get("action_id") or "").strip()
        action_performed = action.get("performed") is True
        action_projected = action.get("projected") is True
        if not action_id or (dry_run and not action_projected) or (
            not dry_run and not action_performed
        ):
            diagnostics.append(
                {
                    "attempt": attempt,
                    "status": "repair_not_performed",
                    "action_id": action_id or None,
                }
            )
            return False, diagnostics
        try:
            result = runner(attempt)
        except Exception as exc:
            diagnostic = safe_exception_diagnostic(exc)
            status = (
                "failed_after_repair"
                if (
                    diagnostic.exception_type == expected.exception_type
                    and diagnostic.diagnostic_hash == expected.diagnostic_hash
                )
                else "different_failure"
            )
            evidence_sha256 = _repair_evidence_digest(
                attempt=attempt,
                action_id=action_id,
                diagnostic=diagnostic,
            )
            diagnostics.append(
                {
                    "attempt": attempt,
                    "action_id": action_id,
                    "action_performed": action_performed,
                    "action_projected": action_projected,
                    "evidence_sha256": evidence_sha256,
                    "status": status,
                    **diagnostic.to_dict(),
                }
            )
            if status != "failed_after_repair":
                return False, diagnostics
            continue
        if result is False:
            diagnostics.append(
                {
                    "attempt": attempt,
                    "status": "different_failure",
                    "action_id": action_id,
                    "exception_type": "FailedRecheck",
                    "diagnostic_hash": _hash_text("deterministic recheck returned false"),
                    "summary": "trusted health component failed deterministic recheck",
                }
            )
            return False, diagnostics
        diagnostics.append(
            {"attempt": attempt, "status": "recovered", "action_id": action_id}
        )
        return False, diagnostics
    return (
        len(diagnostics) == LOCAL_RECHECK_ATTEMPTS
        and all(row.get("status") == "failed_after_repair" for row in diagnostics)
    ), diagnostics


class SystemIncidentSupervisor:
    """Durably supervise one allow-listed internal component."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        packet_dir: Path | None = None,
        clock: Callable[[], datetime] | None = None,
        enqueue: Callable[[Path], Mapping[str, Any]] | None = None,
    ) -> None:
        self.root = root or (wiki.WIKI_ROOT / "runtime" / "system-incidents")
        self.state_file = self.root / "state.json"
        self.lock_file = self.root / "state.lock"
        self.artifact_dir = self.root / "artifacts"
        self.packet_dir = packet_dir or (wiki.WIKI_ROOT / "runtime" / "failures" / "packets")
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._enqueue = enqueue

    def _enqueue_packet(self, packet_path: Path) -> Mapping[str, Any]:
        if self._enqueue is not None:
            return self._enqueue(packet_path)
        from llm_wiki_mcp.self_heal import enqueue_system_code_repair

        return enqueue_system_code_repair(packet_path)

    def observe_health_snapshot_exception(
        self,
        error: BaseException,
        *,
        run_id: str,
        input_id: str | None = None,
        runner: Callable[[int], Any],
        repairer: Callable[..., Mapping[str, Any]] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Observe and possibly promote a reproducible watchdog exception."""

        classification = _classification(error)
        original = safe_exception_diagnostic(error)
        if classification != "system_exception":
            return {
                "status": "excluded",
                "reason": classification,
                "component": TRUSTED_HEALTH_COMPONENT,
                **original.to_dict(),
                "dry_run": dry_run,
            }

        reproduced, rechecks = _run_local_repairs(
            runner,
            expected=original,
            repairer=repairer,
            dry_run=dry_run,
        )
        if not reproduced:
            local_status = (
                "recovered_locally"
                if any(row.get("status") == "recovered" for row in rechecks)
                else "not_reproduced_locally"
            )
            return {
                "status": local_status,
                "component": TRUSTED_HEALTH_COMPONENT,
                **original.to_dict(),
                "local_repairs": rechecks,
                "dry_run": dry_run,
            }

        now = _utc_now(self.clock())
        fingerprint = repair_fingerprint(
            TRUSTED_HEALTH_COMPONENT,
            original.exception_type,
            original.diagnostic_hash,
        )
        identity = _identity_hash(run_id, input_id)
        packet_name = f"system-code-{fingerprint[:32]}.json"
        packet_path = self.packet_dir / packet_name
        artifact_path = self.artifact_dir / f"{fingerprint}.json"

        if dry_run:
            state = copy.deepcopy(_load_json(self.state_file))
            result = self._apply_observation(
                state,
                fingerprint=fingerprint,
                identity=identity,
                diagnostic=original,
                rechecks=rechecks,
                packet_path=packet_path,
                artifact_path=artifact_path,
                now=now,
                persist=False,
            )
            result["status"] = "dry_run"
            result["projected_status"] = result.pop("observation_status")
            result["dry_run"] = True
            return result

        with _state_lock(self.lock_file):
            state = _load_json(self.state_file)
            result = self._apply_observation(
                state,
                fingerprint=fingerprint,
                identity=identity,
                diagnostic=original,
                rechecks=rechecks,
                packet_path=packet_path,
                artifact_path=artifact_path,
                now=now,
                persist=True,
            )
            _write_json_atomic(self.state_file, state)

        if result.pop("should_enqueue", False):
            try:
                queued = dict(self._enqueue_packet(packet_path))
            except Exception as exc:  # keep the packet durable for the next converge pass
                result["observation_status"] = "packet_enqueue_failed"
                result["enqueue_error_type"] = exc.__class__.__name__
                with _state_lock(self.lock_file):
                    state = _load_json(self.state_file)
                    incident = state["incidents"].get(fingerprint)
                    if isinstance(incident, dict):
                        incident["last_enqueue_error_type"] = exc.__class__.__name__
                        incident["last_enqueue_failed_at"] = _timestamp(
                            _utc_now(self.clock())
                        )
                        _write_json_atomic(self.state_file, state)
            else:
                result["enqueue"] = {
                    "job_id": queued.get("job_id"),
                    "enqueued": bool(queued.get("enqueued")),
                    "coalesced": bool(queued.get("coalesced")),
                }
                with _state_lock(self.lock_file):
                    state = _load_json(self.state_file)
                    incident = state["incidents"].get(fingerprint)
                    if isinstance(incident, dict):
                        incident["enqueue_job_id"] = queued.get("job_id")
                        incident["enqueued_at"] = _timestamp(_utc_now(self.clock()))
                        incident.pop("last_enqueue_error_type", None)
                        incident.pop("last_enqueue_failed_at", None)
                        _write_json_atomic(self.state_file, state)
        result["status"] = result.pop("observation_status")
        result["dry_run"] = False
        return result

    def _apply_observation(
        self,
        state: dict[str, Any],
        *,
        fingerprint: str,
        identity: str,
        diagnostic: SafeDiagnostic,
        rechecks: Sequence[Mapping[str, Any]],
        packet_path: Path,
        artifact_path: Path,
        now: datetime,
        persist: bool,
    ) -> dict[str, Any]:
        incidents = state["incidents"]
        incident = incidents.get(fingerprint)
        if not isinstance(incident, dict):
            incident = {
                "component": TRUSTED_HEALTH_COMPONENT,
                "fingerprint": fingerprint,
                "failure_class": "system_health_snapshot_exception",
                "summary": diagnostic.summary,
                "exception_type": diagnostic.exception_type,
                "diagnostic_hash": diagnostic.diagnostic_hash,
                "occurrence_count": 0,
                "distinct_inputs": [],
                "first_seen_at": _timestamp(now),
                "last_seen_at": None,
                "packet_path": None,
                "enqueue_job_id": None,
            }
            incidents[fingerprint] = incident

        incident["occurrence_count"] = int(incident.get("occurrence_count") or 0) + 1
        identities = [
            str(value)
            for value in incident.get("distinct_inputs", [])
            if isinstance(value, str)
        ]
        if identity not in identities:
            identities.append(identity)
        incident["distinct_inputs"] = identities[-MAX_IDENTITIES:]
        repair_evidence = tuple(
            str(row["evidence_sha256"])
            for row in rechecks
            if row.get("status") == "failed_after_repair"
            and isinstance(row.get("evidence_sha256"), str)
        )
        incident["local_repair_attempts"] = len(repair_evidence)
        incident["local_repair_evidence"] = list(repair_evidence)
        incident["last_local_repairs"] = [dict(row) for row in rechecks]
        incident["last_seen_at"] = _timestamp(now)

        occurrence_count = int(incident["occurrence_count"])
        distinct_inputs = tuple(incident["distinct_inputs"])
        eligible = (
            occurrence_count >= MIN_OCCURRENCES
            and len(set(distinct_inputs)) >= MIN_DISTINCT_IDENTITIES
        )
        should_enqueue = False
        observation_status = "observed"
        if eligible:
            evidence = RepairIncidentEvidence(
                component=TRUSTED_HEALTH_COMPONENT,
                fingerprint=fingerprint,
                failure_class="system_health_snapshot_exception",
                occurrence_count=occurrence_count,
                distinct_inputs=distinct_inputs,
                local_repair_attempts=len(repair_evidence),
                local_repair_evidence=repair_evidence,
                reproduction_command=HEALTH_REPRODUCTION_COMMAND,
                failing_test=HEALTH_FAILING_TEST,
                reproduction_artifact=str(artifact_path),
                notes={
                    "producer": "trusted_watchdog",
                    "incident_key": packet_path.stem,
                    "privacy": "no_raw_exception_or_stack",
                },
            )
            packet = self._packet_payload(
                evidence=evidence,
                diagnostic=diagnostic,
                packet_path=packet_path,
                artifact_path=artifact_path,
                now=now,
            )
            artifact = self._artifact_payload(
                evidence=evidence,
                diagnostic=diagnostic,
                rechecks=rechecks,
                now=now,
            )
            if packet_path.exists():
                existing = _load_packet(packet_path)
                if existing.get("fingerprint") != fingerprint:
                    raise IncidentStateError("system incident packet fingerprint collision")
                if persist and not incident.get("enqueued_at"):
                    # Packet creation and durable job enqueue are separate
                    # commits.  A crash or enqueue failure between them must
                    # remain retryable on the next watchdog observation.
                    observation_status = "packet_exists_enqueue_pending"
                    should_enqueue = True
                else:
                    observation_status = "packet_exists"
            elif persist:
                _write_json_atomic(artifact_path, artifact)
                _write_json_atomic(packet_path, packet)
                observation_status = "packet_created"
                should_enqueue = True
            else:
                observation_status = "would_create_packet"
            incident["packet_path"] = str(packet_path)
            incident["artifact_path"] = str(artifact_path)
            incident["eligible_at"] = incident.get("eligible_at") or _timestamp(now)

        return {
            "observation_status": observation_status,
            "component": TRUSTED_HEALTH_COMPONENT,
            "fingerprint": fingerprint,
            "occurrence_count": occurrence_count,
            "distinct_input_count": len(set(distinct_inputs)),
            "local_repair_attempts": len(repair_evidence),
            "local_repair_evidence": list(repair_evidence),
            "packet_path": str(packet_path) if eligible else None,
            "artifact_path": str(artifact_path) if eligible else None,
            "should_enqueue": should_enqueue,
            "diagnostic_hash": diagnostic.diagnostic_hash,
            "exception_type": diagnostic.exception_type,
        }

    @staticmethod
    def _packet_payload(
        *,
        evidence: RepairIncidentEvidence,
        diagnostic: SafeDiagnostic,
        packet_path: Path,
        artifact_path: Path,
        now: datetime,
    ) -> dict[str, Any]:
        failure_id = packet_path.stem
        return {
            "failure_id": failure_id,
            "created_at": _timestamp(now),
            "raw_file": None,
            "job_id": "trusted-watchdog",
            "failure_class": evidence.failure_class,
            "fingerprint": evidence.fingerprint,
            "attempts": evidence.occurrence_count,
            "error": diagnostic.summary,
            "diagnostic_hash": diagnostic.diagnostic_hash,
            "incident_kind": "system_code_repair",
            "repair_evidence": evidence.to_dict(),
            "local_repair_attempts": evidence.local_repair_attempts,
            "local_decision": {
                "status": "unresolved",
                "action": "none",
                "source": "trusted_system_incident_supervisor",
                "notes": "two distinct local repairs were performed and the same verifier still failed",
            },
            "frontier_attempts": 0,
            "reproduction_artifact": str(artifact_path),
            "status": "pending_frontier",
        }

    @staticmethod
    def _artifact_payload(
        *,
        evidence: RepairIncidentEvidence,
        diagnostic: SafeDiagnostic,
        rechecks: Sequence[Mapping[str, Any]],
        now: datetime,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "created_at": _timestamp(now),
            "component": evidence.component,
            "fingerprint": evidence.fingerprint,
            "failure_class": evidence.failure_class,
            "diagnostic": diagnostic.to_dict(),
            "occurrence_count": evidence.occurrence_count,
            "distinct_inputs": list(evidence.distinct_inputs),
            "local_repairs": [dict(row) for row in rechecks],
            "reproduction_command": list(evidence.reproduction_command),
            "failing_test": evidence.failing_test,
            "privacy": "raw exception messages, stack traces, and user data are not stored",
        }


def _load_packet(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IncidentStateError("existing system incident packet is unreadable") from exc
    if not isinstance(payload, dict):
        raise IncidentStateError("existing system incident packet has invalid shape")
    return payload


def supervise_health_snapshot_exception(
    error: BaseException,
    *,
    run_id: str,
    input_id: str | None = None,
    runner: Callable[[int], Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Production convenience wrapper used only by the autonomy watchdog."""

    return SystemIncidentSupervisor().observe_health_snapshot_exception(
        error,
        run_id=run_id,
        input_id=input_id,
        runner=runner,
        repairer=_default_health_repair,
        dry_run=dry_run,
    )
