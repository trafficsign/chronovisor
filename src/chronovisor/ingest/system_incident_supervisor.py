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
import hmac
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core import store as chronovisor_store
from chronovisor.core.timeutil import ensure_utc as _utc_now
from chronovisor.decision.frontier_guard import (
    RepairIncidentEvidence,
    repair_fingerprint,
)

SCHEMA_VERSION = 1
TRUSTED_HEALTH_COMPONENT = "watchdog.health_snapshot"
TRUSTED_OPERATIONAL_COMPONENT = "ingest.operational_runtime"
TRUSTED_OPERATIONAL_PRODUCER = "trusted_operational_failure_supervisor"
TRUSTED_OPERATIONAL_FAILURE_CLASS = "system_operational_failure"
TRUSTED_OPERATIONAL_JOB_ID = "trusted-operational-supervisor"
MIN_OCCURRENCES = 3
MIN_DISTINCT_IDENTITIES = 2
LOCAL_RECHECK_ATTEMPTS = 2
MAX_IDENTITIES = 64
MIN_OPERATIONAL_LOCAL_ATTEMPTS = 2
_OPERATIONAL_TERMINAL_STATUSES = frozenset({"local_quarantined"})
_OPERATIONAL_SUCCESS_STATUSES = frozenset({"frontier_approved", "local_repair_applied"})

HEALTH_REPRODUCTION_COMMAND = (
    "uv",
    "run",
    "python",
    "-c",
    "from chronovisor.ops.health import health_snapshot; health_snapshot()",
)
HEALTH_FAILING_TEST = "runtime:chronovisor.ops.health.health_snapshot"

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
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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
    if isinstance(error, OSError) or any(
        marker in normalized for marker in _OPERATIONAL_MARKERS
    ):
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
        raise IncidentStateError(
            f"cannot read incident state: {exc.__class__.__name__}"
        ) from exc
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
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        with suppress(OSError):
            tmp.unlink()


@contextmanager
def _state_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _source_packet_lock(lock_root: Path, packet_path: Path):
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"{packet_path.name}.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
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


def _safe_raw_name(value: object) -> str | None:
    if not isinstance(value, str) or not value or Path(value).name != value:
        return None
    if any(ord(char) < 32 for char in value):
        return None
    return value


def _resolved_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).expanduser().resolve(strict=False)


def _load_failure_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return {"failures": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("failures"), dict):
        return {"failures": {}}
    return payload


def _connected_raw_groups(
    groups: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    graph: dict[str, set[str]] = {}
    for group in groups:
        members = {
            name for value in group if (name := _safe_raw_name(value)) is not None
        }
        for member in members:
            graph.setdefault(member, set()).update(members - {member})
    components: list[tuple[str, ...]] = []
    unseen = set(graph)
    while unseen:
        pending = [min(unseen)]
        component: set[str] = set()
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(sorted(graph.get(current, ()) - component))
        unseen.difference_update(component)
        components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def _linked_operational_inputs(
    state: Mapping[str, Any],
    *,
    packet_path: Path,
    source_fingerprint: str,
    source_failure_class: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    failures = state.get("failures")
    if not isinstance(failures, Mapping):
        return (), ()
    expected_path = packet_path.expanduser().resolve(strict=False)
    matched: dict[str, Mapping[str, Any]] = {}
    for raw_file, entry in failures.items():
        safe_name = _safe_raw_name(raw_file)
        if safe_name is None or not isinstance(entry, Mapping):
            continue
        linked_path = _resolved_path(entry.get("packet_path"))
        if (
            linked_path != expected_path
            or entry.get("fingerprint") != source_fingerprint
            or entry.get("failure_class") != source_failure_class
            or entry.get("self_heal_queued") is not True
        ):
            continue
        matched[safe_name] = entry
    raw_files = tuple(sorted(matched))
    groups: list[tuple[str, ...]] = []
    for raw_file, entry in matched.items():
        related = entry.get("related_raw_files")
        members = {raw_file}
        if isinstance(related, Sequence) and not isinstance(related, (str, bytes)):
            members.update(
                safe_name
                for value in related
                if (safe_name := _safe_raw_name(value)) is not None
                and safe_name in matched
            )
        groups.append(tuple(sorted(members)))
    return raw_files, _connected_raw_groups(groups)


def _operational_local_evidence(packet: Mapping[str, Any]) -> tuple[str, ...]:
    raw = packet.get("operational_local_repair_evidence")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    evidence = tuple(
        str(value)
        for value in raw
        if isinstance(value, str) and _SHA256_RE.fullmatch(value)
    )
    if len(evidence) != len(raw) or len(set(evidence)) != len(evidence):
        return ()
    return evidence


def _verified_deterministic_reproduction(
    packet: Mapping[str, Any],
    *,
    supervisor_root: Path,
    source_packet_path: Path,
    source_failure_class: str,
    source_fingerprint: str,
) -> dict[str, Any] | None:
    receipt_path = _resolved_path(packet.get("deterministic_reproduction_receipt"))
    receipt_root = (supervisor_root / "reproduction-receipts").resolve(strict=False)
    artifact_root = (supervisor_root / "reproduction-artifacts").resolve(strict=False)
    if receipt_path is None or not receipt_path.is_relative_to(receipt_root):
        return None
    try:
        raw = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(raw, Mapping)
        or raw.get("schema_version") != 1
        or raw.get("producer") != "trusted_system_incident_supervisor"
        or raw.get("outcome") != "reproducibly_failed"
        or _resolved_path(raw.get("source_packet_path")) != source_packet_path
        or raw.get("source_failure_class") != source_failure_class
        or raw.get("source_fingerprint") != source_fingerprint
    ):
        return None
    artifact_path = _resolved_path(raw.get("artifact"))
    expected_sha = raw.get("artifact_sha256")
    if (
        artifact_path is None
        or not artifact_path.is_relative_to(artifact_root)
        or not isinstance(expected_sha, str)
        or _SHA256_RE.fullmatch(expected_sha) is None
    ):
        return None
    try:
        actual_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    except OSError:
        return None
    if not hmac.compare_digest(actual_sha, expected_sha):
        return None
    failing_test = str(raw.get("failing_test") or "").strip() or None
    if (
        failing_test is None
        or not failing_test.startswith("tests/")
        or any(ord(char) < 32 for char in failing_test)
    ):
        return None
    test_path_text = failing_test.split("::", 1)[0]
    test_path = Path(test_path_text)
    if (
        test_path.is_absolute()
        or ".." in test_path.parts
        or test_path.suffix != ".py"
        or any(character.isspace() for character in failing_test)
    ):
        return None
    # Reproduction receipts are data, not an authority to execute arbitrary
    # argv.  Derive the only permitted command from the validated pytest node
    # id.  A receipt that supplies a command must already equal this host-owned
    # form; otherwise reject it instead of copying attacker-controlled argv
    # into the frontier repair postcondition.
    command = ["uv", "run", "pytest", "-q", failing_test]
    supplied_command = raw.get("command")
    if supplied_command is not None and (
        not isinstance(supplied_command, Sequence)
        or isinstance(supplied_command, (str, bytes))
        or [str(value) for value in supplied_command] != command
    ):
        return None
    return {
        "evidence_sha256": actual_sha,
        "command": command,
        "failing_test": failing_test,
        "artifact": str(artifact_path),
        "receipt": str(receipt_path),
    }


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
    from chronovisor.search import index_store, search

    with index_store._store_lock:
        index_store._store = None
    with search._BM25_LOCK:
        search._BM25_SINGLETON = None


def _default_health_repair(attempt: int, *, dry_run: bool) -> Mapping[str, Any]:
    """Perform one bounded, reversible repair of derived health state."""

    from chronovisor.decision.decision_policy import resolve_decision_policy

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

    from chronovisor.search import index_store, search

    cache_paths = (
        index_store.PAGES_INDEX_FILE,
        index_store.BACKLINKS_INDEX_FILE,
        *search.lexical_cache_paths(),
    )
    existing = [path for path in cache_paths if path.exists()]
    if not dry_run:
        quarantine = (
            chronovisor_store.CHRONOVISOR_ROOT
            / "runtime"
            / "system-incidents"
            / "cache-quarantine"
            / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{os.getpid()}"
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
        if (
            not action_id
            or (dry_run and not action_projected)
            or (not dry_run and not action_performed)
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
                    "diagnostic_hash": _hash_text(
                        "deterministic recheck returned false"
                    ),
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
        failure_state_file: Path | None = None,
        clock: Callable[[], datetime] | None = None,
        enqueue: Callable[[Path], Mapping[str, Any]] | None = None,
    ) -> None:
        self.root = root or (chronovisor_store.CHRONOVISOR_ROOT / "runtime" / "system-incidents")
        self.state_file = self.root / "state.json"
        self.lock_file = self.root / "state.lock"
        self.artifact_dir = self.root / "artifacts"
        self.packet_dir = packet_dir or (
            chronovisor_store.CHRONOVISOR_ROOT / "runtime" / "failures" / "packets"
        )
        self.failure_state_file = failure_state_file or (
            chronovisor_store.CHRONOVISOR_ROOT / "runtime" / "failures" / "state.json"
        )
        self.clock = clock or (lambda: datetime.now(UTC))
        self._enqueue = enqueue

    def _enqueue_packet(self, packet_path: Path) -> Mapping[str, Any]:
        if self._enqueue is not None:
            return self._enqueue(packet_path)
        from chronovisor.ingest.self_heal import enqueue_system_code_repair

        return enqueue_system_code_repair(packet_path)

    def _operational_binding_error(
        self,
        packet_path: Path,
        packet: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> str | None:
        evidence = packet.get("repair_evidence")
        notes = evidence.get("notes") if isinstance(evidence, Mapping) else None
        fingerprint = packet.get("fingerprint")
        incident = (
            state.get("incidents", {}).get(fingerprint)
            if isinstance(state.get("incidents"), Mapping)
            else None
        )
        expected_packet_path = packet_path.expanduser().resolve(strict=False)
        if (
            packet.get("job_id") != TRUSTED_OPERATIONAL_JOB_ID
            or packet.get("failure_class") != TRUSTED_OPERATIONAL_FAILURE_CLASS
            or not isinstance(evidence, Mapping)
            or not isinstance(notes, Mapping)
            or notes.get("producer") != TRUSTED_OPERATIONAL_PRODUCER
            or evidence.get("component") != TRUSTED_OPERATIONAL_COMPONENT
            or evidence.get("failure_class") != TRUSTED_OPERATIONAL_FAILURE_CLASS
            or evidence.get("fingerprint") != fingerprint
            or packet.get("source_failure_class") != notes.get("source_failure_class")
            or packet.get("source_fingerprint") != notes.get("source_fingerprint")
            or expected_packet_path.parent != self.packet_dir.resolve(strict=False)
            or packet_path.name != f"system-operational-{str(fingerprint)[:32]}.json"
            or not isinstance(incident, Mapping)
        ):
            return "incident_packet_contract_mismatch"

        source_failure_class = packet.get("source_failure_class")
        source_fingerprint = packet.get("source_fingerprint")
        source_incident_epoch = packet.get("source_incident_epoch")
        legacy_epoch_binding = (
            "source_incident_epoch" not in packet
            and "source_incident_epoch" not in notes
            and "source_incident_epoch" not in incident
            and fingerprint
            == repair_fingerprint(
                TRUSTED_OPERATIONAL_COMPONENT,
                source_failure_class,
                source_fingerprint,
            )
        )
        current_epoch_binding = (
            isinstance(source_incident_epoch, str)
            and _SHA256_RE.fullmatch(source_incident_epoch) is not None
            and notes.get("source_incident_epoch") == source_incident_epoch
            and incident.get("source_incident_epoch") == source_incident_epoch
            and fingerprint
            == repair_fingerprint(
                TRUSTED_OPERATIONAL_COMPONENT,
                source_failure_class,
                source_fingerprint,
                source_incident_epoch,
            )
        )
        if not legacy_epoch_binding and not current_epoch_binding:
            return "incident_packet_contract_mismatch"

        source_paths = packet.get("source_packet_paths")
        raw_files = packet.get("raw_files")
        logical_groups = packet.get("logical_raw_groups")
        local_evidence = evidence.get("local_repair_evidence")
        if (
            not isinstance(source_paths, list)
            or not source_paths
            or not isinstance(raw_files, list)
            or not raw_files
            or not isinstance(logical_groups, list)
            or not logical_groups
            or not isinstance(local_evidence, list)
        ):
            return "incident_packet_links_missing"
        normalized_groups = _connected_raw_groups(
            [
                tuple(str(value) for value in group)
                for group in logical_groups
                if isinstance(group, Sequence) and not isinstance(group, (str, bytes))
            ]
        )
        expected_groups = [list(group) for group in normalized_groups]
        if (
            expected_groups != logical_groups
            or sorted({name for group in normalized_groups for name in group})
            != raw_files
        ):
            return "incident_logical_inputs_invalid"

        artifact_path = _resolved_path(packet.get("reproduction_artifact"))
        if artifact_path is None or not artifact_path.is_relative_to(
            self.artifact_dir.resolve(strict=False)
        ):
            return "incident_artifact_path_invalid"
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return "incident_artifact_unreadable"
        if not isinstance(artifact, Mapping):
            return "incident_artifact_invalid"
        if legacy_epoch_binding and "source_incident_epoch" in artifact:
            return "incident_artifact_binding_mismatch"

        reproduction = evidence.get("reproduction")
        reproduction_artifact = (
            reproduction.get("artifact") if isinstance(reproduction, Mapping) else None
        )
        expected_state = {
            "component": TRUSTED_OPERATIONAL_COMPONENT,
            "failure_class": TRUSTED_OPERATIONAL_FAILURE_CLASS,
            "source_failure_class": packet.get("source_failure_class"),
            "source_fingerprint": packet.get("source_fingerprint"),
            "source_incident_epoch": packet.get("source_incident_epoch"),
            "packet_path": str(expected_packet_path),
            "artifact_path": str(artifact_path),
            "source_packet_paths": source_paths,
            "raw_files": raw_files,
            "logical_raw_groups": logical_groups,
            "local_repair_attempts": evidence.get("local_repair_attempts"),
            "local_repair_evidence": local_evidence,
            "occurrence_count": evidence.get("occurrence_count"),
            "deterministic_reproduction_sha256": notes.get(
                "deterministic_reproduction_sha256"
            ),
        }
        if any(incident.get(key) != value for key, value in expected_state.items()):
            return "incident_state_binding_mismatch"
        expected_artifact = {
            "component": TRUSTED_OPERATIONAL_COMPONENT,
            "fingerprint": fingerprint,
            "failure_class": TRUSTED_OPERATIONAL_FAILURE_CLASS,
            "source_failure_class": packet.get("source_failure_class"),
            "source_fingerprint": packet.get("source_fingerprint"),
            "source_incident_epoch": packet.get("source_incident_epoch"),
            "source_packet_paths": source_paths,
            "raw_files": raw_files,
            "logical_raw_groups": logical_groups,
            "distinct_inputs": evidence.get("distinct_inputs"),
            "local_repair_attempts": evidence.get("local_repair_attempts"),
            "local_repair_evidence": local_evidence,
        }
        if any(artifact.get(key) != value for key, value in expected_artifact.items()):
            return "incident_artifact_binding_mismatch"
        deterministic_artifact = artifact.get("deterministic_reproduction")
        if not isinstance(deterministic_artifact, Mapping):
            return "incident_deterministic_reproduction_invalid"
        deterministic_payload = copy.deepcopy(dict(deterministic_artifact))
        deterministic_digest = (
            _hash_text(
                json.dumps(
                    deterministic_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            if deterministic_payload
            else None
        )
        if not deterministic_payload:
            return "incident_deterministic_reproduction_invalid"
        deterministic_verified = True
        deterministic_command = deterministic_payload.get("command")
        deterministic_failing_test = deterministic_payload.get("failing_test")
        if deterministic_command != [
            "uv",
            "run",
            "pytest",
            "-q",
            deterministic_failing_test,
        ]:
            return "incident_deterministic_reproduction_invalid"
        expected_reproduction = {
            "command": list(deterministic_command),
            "failing_test": deterministic_failing_test,
            "artifact": str(artifact_path),
        }
        if set(deterministic_payload) != {
            "evidence_sha256",
            "command",
            "failing_test",
            "artifact",
            "receipt",
        }:
            return "incident_deterministic_reproduction_invalid"
        if (
            reproduction != expected_reproduction
            or notes.get("deterministic_reproduction_verified")
            is not deterministic_verified
            or notes.get("deterministic_reproduction_evidence")
            != deterministic_payload.get("evidence_sha256")
            or notes.get("deterministic_reproduction_sha256") != deterministic_digest
            or incident.get("deterministic_reproduction_sha256") != deterministic_digest
        ):
            return "incident_deterministic_reproduction_binding_mismatch"
        if (
            reproduction_artifact != str(artifact_path)
            or packet.get("local_repair_attempts")
            != evidence.get("local_repair_attempts")
            or len(logical_groups) != evidence.get("occurrence_count")
            or len(logical_groups) != len(evidence.get("distinct_inputs") or ())
        ):
            return "incident_evidence_binding_mismatch"
        return None

    def validate_operational_incident_packet(
        self,
        packet_path: Path,
    ) -> dict[str, Any]:
        """Read back the supervisor state, artifact, and packet as one binding."""

        resolved = packet_path.expanduser().resolve(strict=False)
        packet = _load_packet(resolved)
        with _state_lock(self.lock_file):
            state = _load_json(self.state_file)
            error = self._operational_binding_error(resolved, packet, state)
        if error is not None:
            raise IncidentStateError(error)
        return {
            "status": "valid",
            "packet_path": str(resolved),
            "fingerprint": packet.get("fingerprint"),
        }

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
            except (
                Exception
            ) as exc:  # keep the packet durable for the next converge pass
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

    def observe_operational_failure_packet(
        self,
        source_packet_path: Path,
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Promote a terminal, cross-raw operational defect into repair plane.

        The source packet remains a routine self-heal packet.  This method only
        emits a separate trusted system incident after the failure-state ledger
        proves that the same stable fingerprint affected independent raws (or
        the packet carries a verified deterministic reproduction receipt) and
        at least two bounded local attempts were durably recorded.
        """

        source_path = source_packet_path.expanduser().resolve(strict=False)
        try:
            source = _load_packet(source_path)
        except IncidentStateError:
            return {
                "status": "excluded",
                "reason": "source_packet_invalid",
                "source_packet_path": str(source_path),
                "dry_run": dry_run,
            }

        try:
            from chronovisor.raw.failure_supervisor import (
                OPERATIONAL_SELF_HEAL_FAILURE_CLASSES,
            )
        except ImportError:
            OPERATIONAL_SELF_HEAL_FAILURE_CLASSES = frozenset()

        source_failure_class = str(source.get("failure_class") or "").strip()
        source_fingerprint = str(source.get("fingerprint") or "").strip()
        normalized_source_failure = source_failure_class.casefold().replace("-", "_")
        if source_failure_class not in OPERATIONAL_SELF_HEAL_FAILURE_CLASSES or any(
            marker in normalized_source_failure for marker in _HUMAN_BOUNDARY_MARKERS
        ):
            return {
                "status": "excluded",
                "reason": "source_failure_not_operational",
                "source_packet_path": str(source_path),
                "dry_run": dry_run,
            }
        if not source_fingerprint or any(ord(char) < 32 for char in source_fingerprint):
            return {
                "status": "excluded",
                "reason": "source_fingerprint_invalid",
                "source_packet_path": str(source_path),
                "dry_run": dry_run,
            }
        if source.get("status") not in _OPERATIONAL_TERMINAL_STATUSES:
            return {
                "status": "observed",
                "reason": "local_repair_not_terminal",
                "source_packet_path": str(source_path),
                "dry_run": dry_run,
            }

        local_attempts = int(source.get("local_repair_attempts") or 0)
        local_evidence = _operational_local_evidence(source)
        if (
            local_attempts < MIN_OPERATIONAL_LOCAL_ATTEMPTS
            or len(local_evidence) < MIN_OPERATIONAL_LOCAL_ATTEMPTS
            or len(local_evidence) > local_attempts
        ):
            return {
                "status": "observed",
                "reason": "local_repair_evidence_incomplete",
                "source_packet_path": str(source_path),
                "local_repair_attempts": local_attempts,
                "dry_run": dry_run,
            }

        failure_state = _load_failure_state(self.failure_state_file)
        raw_files, input_groups = _linked_operational_inputs(
            failure_state,
            packet_path=source_path,
            source_fingerprint=source_fingerprint,
            source_failure_class=source_failure_class,
        )
        deterministic = _verified_deterministic_reproduction(
            source,
            supervisor_root=self.root,
            source_packet_path=source_path,
            source_failure_class=source_failure_class,
            source_fingerprint=source_fingerprint,
        )
        # Cross-input clustering is useful operational telemetry, but it is not
        # executable proof.  Every operational incident must carry a
        # supervisor-owned receipt whose artifact and pytest node were read back
        # above.  This keeps two coincidentally similar raws from opening the
        # token-spending repair plane.
        if not raw_files or deterministic is None:
            return {
                "status": "observed",
                "reason": (
                    "linked_operational_input_missing"
                    if not raw_files
                    else "deterministic_reproduction_not_verified"
                ),
                "source_packet_path": str(source_path),
                "distinct_raw_count": len(input_groups),
                "linked_raw_file_count": len(raw_files),
                "local_repair_attempts": local_attempts,
                "cross_input_cluster_observed": len(input_groups)
                >= MIN_DISTINCT_IDENTITIES,
                "deterministic_reproduction_verified": False,
                "frontier_eligible": False,
                "dry_run": dry_run,
            }

        now = _utc_now(self.clock())
        has_linked_path = "system_incident_packet_path" in source
        has_linked_fingerprint = "system_incident_fingerprint" in source
        linked_path_value = source.get("system_incident_packet_path")
        linked_fingerprint = source.get("system_incident_fingerprint")
        linked_incident_reused = False
        if not has_linked_path and not has_linked_fingerprint:
            source_incident_epoch: str | None = _hash_text(
                json.dumps(
                    {
                        "source_packet_path": str(source_path),
                        "failure_id": source.get("failure_id"),
                        "created_at": source.get("created_at"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            fingerprint = repair_fingerprint(
                TRUSTED_OPERATIONAL_COMPONENT,
                source_failure_class,
                source_fingerprint,
                source_incident_epoch,
            )
            packet_path = (
                self.packet_dir / f"system-operational-{fingerprint[:32]}.json"
            )
            artifact_path = self.artifact_dir / f"{fingerprint}.json"
        else:
            if (
                not isinstance(linked_path_value, str)
                or not linked_path_value.strip()
                or not isinstance(linked_fingerprint, str)
                or not linked_fingerprint
            ):
                raise IncidentStateError("linked operational incident edge invalid")
            requested_path = Path(linked_path_value).expanduser()
            try:
                if not requested_path.is_absolute() or requested_path.is_symlink():
                    raise OSError("linked operational incident path is unsafe")
                packet_path = requested_path.resolve(strict=True)
                packet_root = self.packet_dir.expanduser().resolve(strict=True)
            except OSError as exc:
                raise IncidentStateError(
                    "linked operational incident path invalid"
                ) from exc
            if (
                packet_path.parent != packet_root
                or packet_path.suffix != ".json"
                or not packet_path.name.startswith("system-operational-")
            ):
                raise IncidentStateError("linked operational incident path invalid")
            linked_packet = _load_packet(packet_path)
            linked_sources = linked_packet.get("source_packet_paths")
            artifact_path = _resolved_path(linked_packet.get("reproduction_artifact"))
            if (
                linked_packet.get("fingerprint") != linked_fingerprint
                or linked_packet.get("source_failure_class") != source_failure_class
                or linked_packet.get("source_fingerprint") != source_fingerprint
                or not isinstance(linked_sources, list)
                or str(source_path) not in linked_sources
                or artifact_path is None
            ):
                raise IncidentStateError("linked operational incident binding invalid")
            fingerprint = linked_fingerprint
            source_incident_epoch = linked_packet.get("source_incident_epoch")
            linked_incident_reused = True
        apply_kwargs = {
            "fingerprint": fingerprint,
            "source_packet_path": source_path,
            "source_failure_class": source_failure_class,
            "source_fingerprint": source_fingerprint,
            "source_incident_epoch": source_incident_epoch,
            "raw_files": raw_files,
            "input_groups": input_groups,
            "local_evidence": local_evidence,
            "deterministic": deterministic,
            "packet_path": packet_path,
            "artifact_path": artifact_path,
            "now": now,
            "require_existing_packet": linked_incident_reused,
        }

        if dry_run:
            state = copy.deepcopy(_load_json(self.state_file))
            result = self._apply_operational_observation(
                state,
                persist=False,
                **apply_kwargs,
            )
            result["status"] = "dry_run"
            result["projected_status"] = result.pop("observation_status")
            result.pop("should_enqueue", None)
            result["linked_incident_reused"] = linked_incident_reused
            result["dry_run"] = True
            return result

        with _state_lock(self.lock_file):
            state = _load_json(self.state_file)
            result = self._apply_operational_observation(
                state,
                persist=True,
                **apply_kwargs,
            )
            _write_json_atomic(self.state_file, state)
        result["linked_incident_reused"] = linked_incident_reused

        if result.pop("should_enqueue", False):
            try:
                self.validate_operational_incident_packet(packet_path)
                queued = dict(self._enqueue_packet(packet_path))
            except Exception as exc:
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

    def _apply_operational_observation(
        self,
        state: dict[str, Any],
        *,
        fingerprint: str,
        source_packet_path: Path,
        source_failure_class: str,
        source_fingerprint: str,
        source_incident_epoch: str | None,
        raw_files: Sequence[str],
        input_groups: Sequence[Sequence[str]],
        local_evidence: Sequence[str],
        deterministic: Mapping[str, Any] | None,
        packet_path: Path,
        artifact_path: Path,
        now: datetime,
        persist: bool,
        require_existing_packet: bool,
    ) -> dict[str, Any]:
        incidents = state["incidents"]
        incident = incidents.get(fingerprint)
        if not isinstance(incident, dict):
            if require_existing_packet:
                raise IncidentStateError("linked operational incident state missing")
            incident = {
                "component": TRUSTED_OPERATIONAL_COMPONENT,
                "fingerprint": fingerprint,
                "failure_class": TRUSTED_OPERATIONAL_FAILURE_CLASS,
                "source_failure_class": source_failure_class,
                "source_fingerprint": source_fingerprint,
                "source_incident_epoch": source_incident_epoch,
                "first_seen_at": _timestamp(now),
                "packet_path": None,
                "enqueue_job_id": None,
                "source_packet_paths": [],
                "raw_files": [],
                "logical_raw_groups": [],
            }
            incidents[fingerprint] = incident
        if (
            incident.get("source_failure_class") != source_failure_class
            or incident.get("source_fingerprint") != source_fingerprint
            or incident.get("source_incident_epoch") != source_incident_epoch
        ):
            raise IncidentStateError("operational incident fingerprint collision")
        if require_existing_packet and not packet_path.exists():
            raise IncidentStateError("linked operational incident packet missing")
        if packet_path.exists():
            existing = _load_packet(packet_path)
            binding_error = self._operational_binding_error(
                packet_path,
                existing,
                state,
            )
            if binding_error is not None:
                raise IncidentStateError(binding_error)
            incident["last_seen_at"] = _timestamp(now)
            should_enqueue = bool(persist and not incident.get("enqueued_at"))
            return {
                "observation_status": (
                    "packet_exists_enqueue_pending"
                    if should_enqueue
                    else "packet_exists"
                ),
                "component": TRUSTED_OPERATIONAL_COMPONENT,
                "fingerprint": fingerprint,
                "source_fingerprint": source_fingerprint,
                "source_failure_class": source_failure_class,
                "source_incident_epoch": source_incident_epoch,
                "source_packet_path": str(source_packet_path),
                "raw_files": list(existing.get("raw_files") or ()),
                "logical_raw_groups": list(existing.get("logical_raw_groups") or ()),
                "distinct_raw_count": len(existing.get("logical_raw_groups") or ()),
                "linked_raw_file_count": len(existing.get("raw_files") or ()),
                "local_repair_attempts": existing.get("local_repair_attempts"),
                "packet_path": str(packet_path),
                "artifact_path": existing.get("reproduction_artifact"),
                "should_enqueue": should_enqueue,
            }
        if not isinstance(source_incident_epoch, str):
            raise IncidentStateError("operational incident epoch missing")

        source_paths = {
            str(value)
            for value in incident.get("source_packet_paths", [])
            if isinstance(value, str)
        }
        source_paths.add(str(source_packet_path))
        merged_raw_files = {
            str(value)
            for value in incident.get("raw_files", [])
            if _safe_raw_name(value) is not None
        }
        merged_raw_files.update(raw_files)
        prior_groups = [
            tuple(str(value) for value in group)
            for group in incident.get("logical_raw_groups", [])
            if isinstance(group, Sequence) and not isinstance(group, (str, bytes))
        ]
        logical_groups = _connected_raw_groups([*prior_groups, *input_groups])
        incident["source_packet_paths"] = sorted(source_paths)
        incident["raw_files"] = sorted(merged_raw_files)
        incident["logical_raw_groups"] = [list(group) for group in logical_groups]
        incident["occurrence_count"] = len(logical_groups)
        incident["local_repair_attempts"] = len(local_evidence)
        incident["local_repair_evidence"] = list(local_evidence)
        incident["last_seen_at"] = _timestamp(now)

        identity_hashes = tuple(
            _hash_text(
                "operational-input:"
                + json.dumps(group, ensure_ascii=False, separators=(",", ":"))
            )
            for group in logical_groups
        )
        deterministic_verified = deterministic is not None
        deterministic_payload = copy.deepcopy(dict(deterministic or {}))
        deterministic_digest = (
            _hash_text(
                json.dumps(
                    deterministic_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            if deterministic_payload
            else None
        )
        incident["deterministic_reproduction_sha256"] = deterministic_digest
        evidence = RepairIncidentEvidence(
            component=TRUSTED_OPERATIONAL_COMPONENT,
            fingerprint=fingerprint,
            failure_class=TRUSTED_OPERATIONAL_FAILURE_CLASS,
            occurrence_count=len(identity_hashes),
            distinct_inputs=identity_hashes,
            local_repair_attempts=len(local_evidence),
            local_repair_evidence=tuple(local_evidence),
            reproduction_command=tuple(
                deterministic.get("command", ()) if deterministic is not None else ()
            ),
            failing_test=(
                deterministic.get("failing_test")
                if deterministic is not None
                else f"runtime:{source_failure_class}"
            ),
            reproduction_artifact=str(artifact_path),
            notes={
                "producer": TRUSTED_OPERATIONAL_PRODUCER,
                "incident_key": packet_path.stem,
                "source_failure_class": source_failure_class,
                "source_fingerprint": source_fingerprint,
                "source_incident_epoch": source_incident_epoch,
                "deterministic_reproduction_verified": deterministic_verified,
                "deterministic_reproduction_evidence": (
                    deterministic.get("evidence_sha256")
                    if deterministic is not None
                    else None
                ),
                "deterministic_reproduction_sha256": deterministic_digest,
            },
        )
        packet = self._operational_packet_payload(
            evidence=evidence,
            source_packet_paths=tuple(sorted(source_paths)),
            raw_files=tuple(incident["raw_files"]),
            logical_raw_groups=logical_groups,
            packet_path=packet_path,
            artifact_path=artifact_path,
            now=now,
        )
        artifact = self._operational_artifact_payload(
            evidence=evidence,
            source_packet_paths=tuple(sorted(source_paths)),
            raw_files=tuple(incident["raw_files"]),
            logical_raw_groups=logical_groups,
            deterministic=deterministic,
            now=now,
        )

        should_enqueue = False
        if packet_path.exists():
            existing = _load_packet(packet_path)
            if (
                existing.get("fingerprint") != fingerprint
                or existing.get("source_fingerprint") != source_fingerprint
            ):
                raise IncidentStateError("operational incident packet collision")
            if persist and not incident.get("enqueued_at"):
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
            "component": TRUSTED_OPERATIONAL_COMPONENT,
            "fingerprint": fingerprint,
            "source_fingerprint": source_fingerprint,
            "source_failure_class": source_failure_class,
            "source_incident_epoch": source_incident_epoch,
            "source_packet_path": str(source_packet_path),
            "raw_files": list(incident["raw_files"]),
            "logical_raw_groups": [list(group) for group in logical_groups],
            "distinct_raw_count": len(logical_groups),
            "linked_raw_file_count": len(incident["raw_files"]),
            "local_repair_attempts": len(local_evidence),
            "packet_path": str(packet_path),
            "artifact_path": str(artifact_path),
            "should_enqueue": should_enqueue,
        }

    def sync_operational_incident_outcome(
        self,
        incident_packet_path: Path,
    ) -> dict[str, Any]:
        """Mirror a trusted incident outcome back to its deferred source packets."""

        incident_path = incident_packet_path.expanduser().resolve(strict=False)
        try:
            incident = _load_packet(incident_path)
        except IncidentStateError:
            return {"status": "excluded", "reason": "incident_packet_invalid"}
        evidence = incident.get("repair_evidence")
        notes = evidence.get("notes") if isinstance(evidence, Mapping) else None
        if (
            incident.get("job_id") != TRUSTED_OPERATIONAL_JOB_ID
            or incident.get("failure_class") != TRUSTED_OPERATIONAL_FAILURE_CLASS
            or not isinstance(notes, Mapping)
            or notes.get("producer") != TRUSTED_OPERATIONAL_PRODUCER
            or incident.get("fingerprint") != evidence.get("fingerprint")
            or incident.get("source_failure_class") != notes.get("source_failure_class")
            or incident.get("source_fingerprint") != notes.get("source_fingerprint")
            or incident.get("source_incident_epoch")
            != notes.get("source_incident_epoch")
        ):
            return {"status": "excluded", "reason": "not_operational_incident"}
        try:
            self.validate_operational_incident_packet(incident_path)
        except IncidentStateError as exc:
            return {
                "status": "attention",
                "reason": "incident_binding_invalid",
                "error": str(exc),
            }

        incident_status = str(incident.get("status") or "")
        source_fingerprint = str(incident.get("source_fingerprint") or "")
        source_failure_class = str(incident.get("source_failure_class") or "")
        source_paths = incident.get("source_packet_paths")
        if not isinstance(source_paths, Sequence) or isinstance(
            source_paths, (str, bytes)
        ):
            return {"status": "attention", "reason": "source_links_missing"}
        artifact_path = _resolved_path(incident.get("reproduction_artifact"))
        artifact_root = self.artifact_dir.resolve(strict=False)
        if artifact_path is None or not artifact_path.is_relative_to(artifact_root):
            return {"status": "attention", "reason": "incident_artifact_invalid"}
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return {"status": "attention", "reason": "incident_artifact_invalid"}
        if (
            not isinstance(artifact, Mapping)
            or artifact.get("fingerprint") != incident.get("fingerprint")
            or artifact.get("source_failure_class") != source_failure_class
            or artifact.get("source_fingerprint") != source_fingerprint
            or artifact.get("source_packet_paths") != list(source_paths)
            or artifact.get("raw_files") != incident.get("raw_files")
            or artifact.get("logical_raw_groups") != incident.get("logical_raw_groups")
            or artifact.get("local_repair_evidence")
            != evidence.get("local_repair_evidence")
        ):
            return {"status": "attention", "reason": "incident_artifact_mismatch"}

        updated = 0
        busy = 0
        invalid = 0
        success = incident_status in _OPERATIONAL_SUCCESS_STATUSES
        for value in source_paths:
            source_path = _resolved_path(value)
            if source_path is None:
                invalid += 1
                continue
            with _source_packet_lock(
                self.failure_state_file.parent / "locks", source_path
            ) as acquired:
                if not acquired:
                    busy += 1
                    continue
                try:
                    source = _load_packet(source_path)
                    current_incident = _load_packet(incident_path)
                except IncidentStateError:
                    invalid += 1
                    continue
                if (
                    source.get("fingerprint") != source_fingerprint
                    or source.get("failure_class") != source_failure_class
                    or current_incident.get("fingerprint")
                    != incident.get("fingerprint")
                    or current_incident.get("source_fingerprint")
                    != source_fingerprint
                    or current_incident.get("source_failure_class")
                    != source_failure_class
                    or current_incident.get("source_incident_epoch")
                    != incident.get("source_incident_epoch")
                ):
                    invalid += 1
                    continue
                incident_status = str(current_incident.get("status") or "")
                success = incident_status in _OPERATIONAL_SUCCESS_STATUSES
                source["system_incident_packet_path"] = str(incident_path)
                source["system_incident_fingerprint"] = current_incident.get(
                    "fingerprint"
                )
                source["system_incident_status"] = incident_status
                source["system_incident_synced_at"] = _timestamp(_utc_now(self.clock()))
                if success:
                    source["status"] = incident_status
                    source["repair_completed_at"] = _timestamp(_utc_now(self.clock()))
                    source["next_attempt_at"] = None
                    source["quarantined_at"] = None
                _write_json_atomic(source_path, source)
                updated += 1
        return {
            "status": "ok" if not invalid else "attention",
            "incident_status": incident_status,
            "repair_success": success,
            "updated_source_packets": updated,
            "busy_source_packets": busy,
            "invalid_source_packets": invalid,
        }

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
                    raise IncidentStateError(
                        "system incident packet fingerprint collision"
                    )
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

    @staticmethod
    def _operational_packet_payload(
        *,
        evidence: RepairIncidentEvidence,
        source_packet_paths: Sequence[str],
        raw_files: Sequence[str],
        logical_raw_groups: Sequence[Sequence[str]],
        packet_path: Path,
        artifact_path: Path,
        now: datetime,
    ) -> dict[str, Any]:
        return {
            "failure_id": packet_path.stem,
            "created_at": _timestamp(now),
            "raw_file": None,
            "raw_files": list(raw_files),
            "logical_raw_groups": [list(group) for group in logical_raw_groups],
            "job_id": TRUSTED_OPERATIONAL_JOB_ID,
            "failure_class": evidence.failure_class,
            "fingerprint": evidence.fingerprint,
            "attempts": evidence.occurrence_count,
            "error": "trusted operational defect persisted after bounded local repair",
            "incident_kind": "system_code_repair",
            "repair_evidence": evidence.to_dict(),
            "local_repair_attempts": evidence.local_repair_attempts,
            "local_decision": {
                "status": "unresolved",
                "action": "none",
                "source": "trusted_system_incident_supervisor",
                "notes": "routine local repair was terminal across independent raws",
            },
            "source_failure_class": evidence.notes.get("source_failure_class"),
            "source_fingerprint": evidence.notes.get("source_fingerprint"),
            "source_incident_epoch": evidence.notes.get("source_incident_epoch"),
            "source_packet_paths": list(source_packet_paths),
            "frontier_attempts": 0,
            "reproduction_artifact": str(artifact_path),
            "status": "pending_frontier",
        }

    @staticmethod
    def _operational_artifact_payload(
        *,
        evidence: RepairIncidentEvidence,
        source_packet_paths: Sequence[str],
        raw_files: Sequence[str],
        logical_raw_groups: Sequence[Sequence[str]],
        deterministic: Mapping[str, Any] | None,
        now: datetime,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "created_at": _timestamp(now),
            "component": evidence.component,
            "fingerprint": evidence.fingerprint,
            "failure_class": evidence.failure_class,
            "source_failure_class": evidence.notes.get("source_failure_class"),
            "source_fingerprint": evidence.notes.get("source_fingerprint"),
            "source_incident_epoch": evidence.notes.get("source_incident_epoch"),
            "source_packet_paths": list(source_packet_paths),
            "raw_files": list(raw_files),
            "logical_raw_groups": [list(group) for group in logical_raw_groups],
            "distinct_inputs": list(evidence.distinct_inputs),
            "local_repair_attempts": evidence.local_repair_attempts,
            "local_repair_evidence": list(evidence.local_repair_evidence),
            "deterministic_reproduction": copy.deepcopy(dict(deterministic or {})),
            "privacy": "raw contents and model output are excluded",
        }


def _load_packet(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IncidentStateError(
            "existing system incident packet is unreadable"
        ) from exc
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


def supervise_operational_failure_packet(
    packet_path: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Production bridge from routine operational repair into system repair."""

    return SystemIncidentSupervisor().observe_operational_failure_packet(
        packet_path,
        dry_run=dry_run,
    )


def sync_operational_incident_outcome(packet_path: Path) -> dict[str, Any]:
    """Production wrapper for durable incident-to-source status propagation."""

    return SystemIncidentSupervisor().sync_operational_incident_outcome(packet_path)


def validate_operational_incident_packet(packet_path: Path) -> dict[str, Any]:
    """Production wrapper for supervisor-owned repair evidence read-back."""

    return SystemIncidentSupervisor().validate_operational_incident_packet(packet_path)
