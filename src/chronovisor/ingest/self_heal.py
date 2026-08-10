"""Autonomous self-healing loop for Chronovisor failure packets."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from chronovisor.core import live_layout, reserved_documents, runtime_status
from chronovisor.core import store as chronovisor_store
from chronovisor.core.alias_store import add_alias
from chronovisor.core.background_jobs import start_self_heal_background
from chronovisor.core.page_mutation import decision_authority_lock
from chronovisor.core.self_heal_cancellation import (
    PACKET_CANCELLATION_SCHEMA_VERSION as PACKET_CANCELLATION_SCHEMA_VERSION,
)
from chronovisor.core.self_heal_cancellation import (
    PACKET_CANCELLATION_STATUS as PACKET_CANCELLATION_STATUS,
)
from chronovisor.core.self_heal_cancellation import (
    PACKET_CANCELLATION_STATUSES as _PACKET_CANCELLATION_STATUSES,
)
from chronovisor.core.self_heal_cancellation import (
    PACKET_SUCCESS_STATUSES,
    packet_cancellation_dir,
    packet_cancellation_path,
)
from chronovisor.core.self_heal_cancellation import (
    VERIFIED_LOCAL_REPAIR_CANCELLATION_STATUS as VERIFIED_LOCAL_REPAIR_CANCELLATION_STATUS,
)
from chronovisor.core.self_heal_cancellation import (
    read_json as _read_json,
)
from chronovisor.core.self_heal_cancellation import (
    read_packet_cancellation as _read_packet_cancellation,
)
from chronovisor.core.self_heal_cancellation import (
    request_packet_cancellation as request_packet_cancellation,
)
from chronovisor.core.self_heal_cancellation import (
    write_json as _write_json,
)
from chronovisor.decision.decision_authority import (
    compare_semantic_authority,
    current_semantic_authority,
    semantic_verdict_authority_error,
)
from chronovisor.decision.frontier_guard import (
    EvidenceValidationError,
    RepairIncidentEvidence,
)
from chronovisor.decision.local_repair import (
    LocalRepairDecision,
    is_review_budget_nonconvergence,
    propose_repair,
    semantic_hold_epoch,
)
from chronovisor.decision.semantic_hold import (
    LOCAL_SEMANTIC_NO_QUORUM,
    persisted_semantic_no_quorum_hold,
    semantic_no_quorum_hold_error,
)
from chronovisor.ingest.convergence import (
    is_human_required_failure,
    is_human_required_result,
)

start_background = start_self_heal_background
_packet_cancellation_dir = packet_cancellation_dir
_packet_cancellation_path = packet_cancellation_path
_PACKET_SUCCESS_STATUSES = PACKET_SUCCESS_STATUSES

SELF_HEAL_STATUSES = {
    "pending_local_repair",
    "local_repair_failed",
    "pending_frontier",
    "frontier_retry",
    "frontier_preflight_failed",
    "pending_frontier_review",
    "repair_deferred",
}

# A local semantic decision may quarantine an operational source packet while
# a code fix is being prepared.  That terminal scheduler state must not be
# re-enqueued by ``pending_packets()``, but the explicit verified-repair CAS is
# still allowed to release it.  True semantic defers remain excluded by the
# stronger class/terminal guards in the release boundary.
_VERIFIED_LOCAL_REPAIR_RELEASABLE_STATUSES = frozenset(
    {*SELF_HEAL_STATUSES, "local_quarantined"}
)

RUNNING_STATUSES = {
    "local_repairing",
    "frontier_running",
}

_SYSTEM_INCIDENT_PRESTART_STATUSES = frozenset(
    {
        "pending_frontier",
        "frontier_retry",
        "frontier_preflight_failed",
        "pending_frontier_review",
        "repair_deferred",
    }
)
_FULL_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_RAW_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_EXPECTED_GITHUB_REPOSITORY = "trafficsign/chronovisor"

DEFAULT_RUNNING_LEASE_SECONDS = 2 * 60 * 60
DEFAULT_QUARANTINE_RETRY_SECONDS = 6 * 60 * 60
DEFAULT_HUMAN_RECHECK_SECONDS = 60 * 60

HUMAN_REQUIRED_STATUSES = {
    "human_required",
}

PENDING_REVIEW_STATUSES = {
    "frontier_preflight_failed",
    "pending_frontier_review",
}

FRONTIER_ONLY_STATUSES = {
    "pending_frontier",
    "frontier_retry",
    "frontier_preflight_failed",
    "pending_frontier_review",
    "repair_deferred",
    "frontier_running",
}

MAC_NOTIFICATION_TITLE = "Chronovisor 自己修復"
MAC_NOTIFICATION_COOLDOWN_SECONDS = 3600
READ_BACK_TRANSIENT_REASONS = {"search-error", "read-back-unavailable"}
READ_BACK_TRANSIENT_PATTERN = re.compile(
    r"\b(?:"
    r"temporary|temporarily|timeout|timed out|unavailable|overloaded|try again|"
    r"connection reset|connection refused|connection aborted|network|"
    r"rate limit(?:ed)?|too many requests|502|503|504"
    r")\b",
    re.IGNORECASE,
)
READ_BACK_EXHAUSTED_QUERY_HINT_TEXT = (
    "read-back miss persisted after exact query hint was applied"
)
READ_BACK_UNVERIFIABLE_QUERY_HINT_PATTERN = re.compile(
    r"(?:"
    r"available workspace evidence does not include the target page|"
    r"query hint target page does not exist|"
    r"target page no longer exists"
    r")",
    re.IGNORECASE,
)

_TRUSTED_REPAIR_PACKET_CONTRACTS = {
    (
        "trusted_watchdog",
        "watchdog.health_snapshot",
        "system_health_snapshot_exception",
    ): "trusted-watchdog",
    (
        "trusted_operational_failure_supervisor",
        "ingest.operational_runtime",
        "system_operational_failure",
    ): "trusted-operational-supervisor",
}


def _repo_root() -> Path:
    from chronovisor.core.runtime_config import runtime_repo_root

    return runtime_repo_root()


def _failures_dir() -> Path:
    return chronovisor_store.CHRONOVISOR_ROOT / "runtime" / "failures"


def _packet_dir() -> Path:
    return _failures_dir() / "packets"




def _local_repair_dir() -> Path:
    return _failures_dir() / "local-repair"


def _frontier_queue_dir() -> Path:
    return _failures_dir() / "frontier-queue"


def _frontier_decision_dir() -> Path:
    return _failures_dir() / "frontier-decisions"


def _pending_frontier_review_dir() -> Path:
    return _failures_dir() / "pending-frontier-review"


def _applied_actions_dir() -> Path:
    return _failures_dir() / "applied-actions"


def _rejected_actions_dir() -> Path:
    return _failures_dir() / "rejected-actions"


def _registry_file() -> Path:
    return _failures_dir() / "failure-registry.jsonl"


def _notification_file() -> Path:
    return _failures_dir() / "notifications.json"




class _PacketCancellationRequested(RuntimeError):
    """Internal control signal carrying a durable cancellation result."""

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__(str(result.get("reason") or "packet cancellation requested"))
        self.result = result




def _apply_packet_cancellation(
    packet_path: Path,
    packet: dict[str, Any],
    cancellation: dict[str, Any],
) -> dict[str, Any]:
    """Persist terminal cancellation without trusting a stale worker snapshot."""

    try:
        current = _read_json(packet_path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        current = dict(packet)
    cancelled_at = datetime.now().isoformat()
    cancellation_status = str(cancellation.get("status") or "")
    if cancellation_status not in _PACKET_CANCELLATION_STATUSES:
        raise ValueError("packet cancellation status is not allowlisted")
    current.update(
        {
            "status": cancellation_status,
            "self_heal_queued": False,
            "next_attempt_at": None,
            "lease_owner": None,
            "lease_expires_at": None,
            "cancellation_requested_at": cancellation.get("requested_at"),
            "cancellation_observed_at": cancelled_at,
            "cancellation_reason": cancellation.get("reason"),
            "superseded_by_packet": cancellation.get("superseded_by_packet"),
            "updated_at": cancelled_at,
        }
    )
    _write_json(packet_path, current)
    packet.clear()
    packet.update(current)
    return {
        "packet": str(packet_path),
        "failure_id": current.get("failure_id"),
        "status": cancellation_status,
        "reason": cancellation.get("reason"),
        "superseded_by_packet": cancellation.get("superseded_by_packet"),
        "cancelled": True,
    }


def _raise_if_packet_cancelled(
    packet_path: Path,
    packet: dict[str, Any],
) -> None:
    cancellation = _read_packet_cancellation(packet_path, packet)
    if cancellation is None:
        return
    raise _PacketCancellationRequested(
        _apply_packet_cancellation(packet_path, packet, cancellation)
    )


def _canonical_read_back_reason(value: object) -> str:
    reason = str(value or "unknown").strip().casefold().replace("_", "-")
    return re.sub(r"\s+", "-", reason) or "unknown"


def _read_back_failure_from_packet(packet: dict[str, Any]) -> dict[str, Any]:
    preview = packet.get("raw_preview")
    if not isinstance(preview, str) or not preview.strip():
        return {}
    try:
        evidence = json.loads(preview)
    except json.JSONDecodeError:
        return {}
    if not isinstance(evidence, dict):
        return {}
    failure = evidence.get("failure")
    if isinstance(failure, dict):
        return failure
    ledger_entry = evidence.get("ledger_entry")
    if isinstance(ledger_entry, dict) and isinstance(ledger_entry.get("failure"), dict):
        return ledger_entry["failure"]
    return {}


def _is_transient_read_back_packet(packet: dict[str, Any]) -> bool:
    if packet.get("failure_class") != "read_back.repeated_miss":
        return False
    failure = _read_back_failure_from_packet(packet)
    reason = _canonical_read_back_reason(failure.get("reason"))
    if reason and reason != "unknown" and reason not in READ_BACK_TRANSIENT_REASONS:
        return False
    text = " ".join(
        str(value or "")
        for value in (
            failure.get("error"),
            failure.get("message"),
            failure.get("detail"),
            packet.get("error"),
        )
    )
    return bool(READ_BACK_TRANSIENT_PATTERN.search(text))


def _is_empty_query_read_back_packet(packet: dict[str, Any]) -> bool:
    if packet.get("failure_class") != "read_back.repeated_miss":
        return False
    failure = _read_back_failure_from_packet(packet)
    reason = _canonical_read_back_reason(failure.get("reason"))
    if reason == "empty-query":
        return True
    if reason != "unknown":
        return False
    return (
        "empty-query"
        in _read_back_packet_diagnostic_text(
            packet,
            failure,
        ).casefold()
    )


def _read_back_packet_diagnostic_text(
    packet: dict[str, Any],
    failure: dict[str, Any],
) -> str:
    preview = packet.get("raw_preview")
    evidence: dict[str, Any] = {}
    if isinstance(preview, str) and preview.strip():
        try:
            parsed = json.loads(preview)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            evidence = parsed
    ledger_entry = evidence.get("ledger_entry") if isinstance(evidence, dict) else {}
    return " ".join(
        str(value or "")
        for value in (
            packet.get("error"),
            failure.get("error"),
            failure.get("message"),
            failure.get("detail"),
            ledger_entry.get("last_error") if isinstance(ledger_entry, dict) else "",
        )
    )


def _is_exhausted_query_hint_read_back_packet(packet: dict[str, Any]) -> bool:
    if packet.get("failure_class") != "read_back.repeated_miss":
        return False
    failure = _read_back_failure_from_packet(packet)
    if _canonical_read_back_reason(failure.get("reason")) != "not-in-top-results":
        return False
    return READ_BACK_EXHAUSTED_QUERY_HINT_TEXT in _read_back_packet_diagnostic_text(
        packet,
        failure,
    )


def _is_unverifiable_query_hint_read_back_packet(packet: dict[str, Any]) -> bool:
    if packet.get("failure_class") != "read_back.repeated_miss":
        return False
    failure = _read_back_failure_from_packet(packet)
    if _canonical_read_back_reason(failure.get("reason")) != "not-in-top-results":
        return False
    return bool(
        READ_BACK_UNVERIFIABLE_QUERY_HINT_PATTERN.search(
            _read_back_packet_diagnostic_text(packet, failure)
        )
    )


def _frontier_nonconvergence_should_reenter_local(packet: dict[str, Any]) -> bool:
    """Compatibility wrapper for new and legacy bounded-review packets."""

    return is_review_budget_nonconvergence(packet)


def _retire_non_actionable_read_back_packet(
    packet_path: Path,
    packet: dict[str, Any],
    *,
    reason: str,
    summary: str,
    resolution: str,
    outcome_kind: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    result = {
        "packet": str(packet_path),
        "failure_id": packet.get("failure_id"),
        "status": "dry_run" if dry_run else "frontier_rejected",
        "reason": reason,
        "summary": summary,
    }
    if dry_run:
        result["projected_status"] = "frontier_rejected"
        return result

    frontier_result = {
        "decision": "rejected",
        "summary": summary,
        "human_required": False,
        "frontier_required": False,
    }
    action = {
        "action": "retire_non_actionable_read_back_packet",
        "reason": reason,
        "decision": packet.get("local_decision"),
        "frontier": frontier_result,
    }
    action_path = _save_action(packet_path, action, applied=False)
    _update_packet(
        packet_path,
        packet,
        status="frontier_rejected",
        frontier_result=frontier_result,
        frontier_status="not_required",
        rejected_action_path=str(action_path),
        transient_read_back_retired_at=datetime.now().isoformat(timespec="seconds"),
        next_attempt_at=None,
        frontier_error=None,
    )
    _append_registry(
        {
            "timestamp": datetime.now().isoformat(),
            "failure_id": packet.get("failure_id"),
            "raw_file": packet.get("raw_file"),
            "failure_class": packet.get("failure_class"),
            "fingerprint": packet.get("fingerprint"),
            "resolution": resolution,
            "decision": packet.get("local_decision"),
            "frontier": frontier_result,
            "action": action,
        }
    )
    runtime_status.safe_append_event(
        "warn",
        f"self-heal | retired non-actionable read-back packet for {packet.get('raw_file')}",
        source="self-heal",
        packet=str(packet_path),
        frontier_status="frontier_rejected",
        outcome_kind=outcome_kind,
    )
    result["frontier_result"] = frontier_result
    result["rejected_action_path"] = str(action_path)
    return result


def _retire_transient_read_back_packet(
    packet_path: Path,
    packet: dict[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    return _retire_non_actionable_read_back_packet(
        packet_path,
        packet,
        reason="transient_read_back_operational_failure",
        summary=(
            "transient read-back search/model outage is handled by the read-back "
            "repair retry quarantine; no code or frontier repair is applicable"
        ),
        resolution="transient_read_back_rejected",
        outcome_kind="transient_read_back_retired",
        dry_run=dry_run,
    )


def _retire_empty_query_read_back_packet(
    packet_path: Path,
    packet: dict[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    return _retire_non_actionable_read_back_packet(
        packet_path,
        packet,
        reason="empty_query_read_back_failure",
        summary=(
            "empty-query read-back failures have no repairable query or raw "
            "mutation; read-back repair rejects them locally and frontier is "
            "not applicable"
        ),
        resolution="empty_query_read_back_rejected",
        outcome_kind="empty_query_read_back_retired",
        dry_run=dry_run,
    )


def _retire_exhausted_query_hint_read_back_packet(
    packet_path: Path,
    packet: dict[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    return _retire_non_actionable_read_back_packet(
        packet_path,
        packet,
        reason="exhausted_read_back_query_hint",
        summary=(
            "read-back repair already applied the exact frontier-approved query "
            "hint and the miss persisted; self-heal has no raw/code mutation to apply"
        ),
        resolution="exhausted_read_back_query_hint_rejected",
        outcome_kind="exhausted_read_back_query_hint_retired",
        dry_run=dry_run,
    )


def _retire_unverifiable_query_hint_read_back_packet(
    packet_path: Path,
    packet: dict[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    return _retire_non_actionable_read_back_packet(
        packet_path,
        packet,
        reason="unverifiable_read_back_query_hint",
        summary=(
            "read-back repair could not verify the missing target page or "
            "page-specific query evidence; self-heal has no safe code/raw "
            "mutation to apply"
        ),
        resolution="unverifiable_read_back_query_hint_rejected",
        outcome_kind="unverifiable_read_back_query_hint_retired",
        dry_run=dry_run,
    )


@contextmanager
def _packet_lock(packet_path: Path):
    """Acquire a non-blocking process lock for one failure packet."""

    lock_dir = _failures_dir() / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{packet_path.name}.lock"
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


def _verified_runtime_github_source(value: object) -> str:
    """Return one canonical GitHub SSH repository or fail closed."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("repair_runtime_source_unavailable")
    source = value.strip()
    if source.startswith("git+"):
        source = source[4:]
    parsed = urlsplit(source)
    repository = parsed.path.strip("/")
    if repository.endswith(".git"):
        repository = repository[:-4]
    if (
        parsed.scheme != "ssh"
        or parsed.hostname != "github.com"
        or parsed.username != "git"
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
        or repository != _EXPECTED_GITHUB_REPOSITORY
    ):
        raise ValueError("repair_runtime_source_not_exact_github_vcs")
    return f"ssh://git@github.com/{repository}"


def _verified_uv_archive_root(value: object) -> Path:
    """Resolve one runtime path to its exact immutable uv archive root."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("repair_runtime_archive_unavailable")
    try:
        resolved = Path(value).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError("repair_runtime_archive_unavailable") from exc
    indices = [
        index for index, part in enumerate(resolved.parts) if part == "archive-v0"
    ]
    if not indices:
        raise ValueError("repair_runtime_not_uv_archive")
    index = indices[-1]
    if index + 1 >= len(resolved.parts) or resolved.parts[index + 1] in {"", ".", ".."}:
        raise ValueError("repair_runtime_not_uv_archive")
    return Path(*resolved.parts[: index + 2])


def _verified_local_repair_git_state(expected_commit: str) -> dict[str, Any]:
    """Bind a repair ACK to the clean, pushed, executing revision."""

    expected = expected_commit.strip().casefold()
    if _FULL_GIT_SHA_RE.fullmatch(expected) is None:
        raise ValueError("repair_commit_must_be_full_git_sha")

    repo_root = _repo_root().expanduser().resolve(strict=True)

    def git_output(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"repair_git_verification_failed:{args[0]}")
        return completed.stdout.strip()

    head_commit = git_output("rev-parse", "--verify", "HEAD^{commit}").casefold()
    if head_commit != expected:
        raise ValueError("repair_commit_not_current_head")
    origin_main_commit = git_output(
        "rev-parse", "--verify", "origin/main^{commit}"
    ).casefold()
    if origin_main_commit != expected:
        raise ValueError("repair_commit_not_pushed_origin_main")
    if git_output("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("repair_checkout_has_uncommitted_tracked_changes")

    from chronovisor.core.runtime_config import runtime_identity

    identity = runtime_identity()
    if not isinstance(identity, dict):
        raise ValueError("repair_runtime_identity_invalid")
    runtime_commit = identity.get("commit_id")
    if (
        not isinstance(runtime_commit, str)
        or _FULL_GIT_SHA_RE.fullmatch(runtime_commit.strip().casefold()) is None
    ):
        raise ValueError("repair_runtime_commit_unavailable")
    runtime_commit = runtime_commit.strip().casefold()
    if runtime_commit != expected:
        raise ValueError("repair_commit_not_executing_runtime")

    expected_runtime_commit = identity.get("expected_commit")
    if (
        not isinstance(expected_runtime_commit, str)
        or _FULL_GIT_SHA_RE.fullmatch(expected_runtime_commit.strip().casefold())
        is None
    ):
        raise ValueError("repair_runtime_expected_commit_unavailable")
    expected_runtime_commit = expected_runtime_commit.strip().casefold()
    if expected_runtime_commit != expected:
        raise ValueError("repair_runtime_expected_commit_mismatch")
    if identity.get("drift") is not False:
        raise ValueError("repair_runtime_drift_not_false")

    archive_root = _verified_uv_archive_root(identity.get("archive_path"))
    module_path = identity.get("module_path")
    try:
        resolved_module_path = Path(str(module_path)).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError("repair_runtime_module_unavailable") from exc
    if not resolved_module_path.is_relative_to(archive_root):
        raise ValueError("repair_runtime_module_outside_archive")

    runtime_source = _verified_runtime_github_source(identity.get("runtime_source"))
    direct_url = identity.get("direct_url")
    if not isinstance(direct_url, dict):
        raise ValueError("repair_runtime_direct_url_invalid")
    direct_source = _verified_runtime_github_source(direct_url.get("url"))
    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, dict) or vcs_info.get("vcs") != "git":
        raise ValueError("repair_runtime_direct_url_not_git")
    direct_commit = vcs_info.get("commit_id")
    if (
        not isinstance(direct_commit, str)
        or _FULL_GIT_SHA_RE.fullmatch(direct_commit.strip().casefold()) is None
    ):
        raise ValueError("repair_runtime_direct_url_commit_unavailable")
    direct_commit = direct_commit.strip().casefold()
    if direct_commit != expected:
        raise ValueError("repair_runtime_direct_url_commit_mismatch")
    if direct_source != runtime_source:
        raise ValueError("repair_runtime_source_direct_url_mismatch")

    return {
        "git_commit_sha": expected,
        "checkout_head_sha": head_commit,
        "origin_main_sha": origin_main_commit,
        "runtime_commit_sha": runtime_commit,
        "runtime_expected_commit_sha": expected_runtime_commit,
        "runtime_archive_root": str(archive_root),
        "runtime_module_path": str(resolved_module_path),
        "runtime_source": runtime_source,
        "runtime_direct_url": direct_source,
        "runtime_drift": False,
    }


def _operational_release_refusal(
    packet_path: Path,
    reason: str,
    *,
    packet: dict[str, Any] | None = None,
    **details: Any,
) -> dict[str, Any]:
    return {
        "packet": str(packet_path),
        "failure_id": packet.get("failure_id") if packet is not None else None,
        "status": "refused",
        "accepted": False,
        "reason": reason,
        **details,
    }


def _normalize_expected_raw_manifest(
    values: Mapping[str, str] | Sequence[str],
) -> dict[str, str]:
    """Normalize API/CLI filename=sha256 evidence into one exact group CAS."""

    if isinstance(values, Mapping):
        pairs = list(values.items())
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        pairs = []
        for value in values:
            if not isinstance(value, str) or "=" not in value:
                raise ValueError("expected_raw_sha256_malformed")
            filename, digest = value.split("=", 1)
            pairs.append((filename, digest))
    else:
        raise ValueError("expected_raw_sha256_malformed")
    if not pairs:
        raise ValueError("expected_raw_sha256_required")

    manifest: dict[str, str] = {}
    for raw_filename, raw_digest in pairs:
        filename = str(raw_filename).strip()
        digest = str(raw_digest).strip().casefold()
        if (
            not filename
            or Path(filename).name != filename
            or filename in {".", ".."}
            or _RAW_SHA256_RE.fullmatch(digest) is None
        ):
            raise ValueError("expected_raw_sha256_malformed")
        if filename in manifest:
            raise ValueError("expected_raw_sha256_duplicate_filename")
        manifest[filename] = digest
    return dict(sorted(manifest.items()))


def _inspect_linked_operational_incident(
    source_packet_path: Path,
    source_packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one source-to-system-incident edge and its execution state."""

    linked_fields = (
        "system_incident_packet_path",
        "system_incident_fingerprint",
        "system_incident_status",
    )
    if not any(field in source_packet for field in linked_fields):
        return {"accepted": True, "linked": False}
    linked_value = source_packet.get("system_incident_packet_path")
    if linked_value is None:
        return {"accepted": False, "reason": "linked_system_incident_path_invalid"}
    if not isinstance(linked_value, str) or not linked_value.strip():
        return {"accepted": False, "reason": "linked_system_incident_path_invalid"}
    requested = Path(linked_value).expanduser()
    try:
        if requested.is_symlink():
            raise ValueError("linked_system_incident_symlink")
        incident_path = requested.resolve(strict=True)
        packet_root = _packet_dir().expanduser().resolve(strict=True)
    except (OSError, ValueError):
        return {"accepted": False, "reason": "linked_system_incident_unavailable"}
    if (
        incident_path == source_packet_path
        or incident_path.parent != packet_root
        or incident_path.suffix != ".json"
        or not incident_path.name.startswith("system-operational-")
    ):
        return {"accepted": False, "reason": "linked_system_incident_path_invalid"}

    try:
        from chronovisor.ingest.system_incident_supervisor import (
            TRUSTED_OPERATIONAL_FAILURE_CLASS,
            TRUSTED_OPERATIONAL_JOB_ID,
            validate_operational_incident_packet,
        )

        validate_operational_incident_packet(incident_path)
        incident = _read_json(incident_path)
    except Exception:
        return {"accepted": False, "reason": "linked_system_incident_binding_invalid"}
    linked_fingerprint = source_packet.get("system_incident_fingerprint")
    source_paths = incident.get("source_packet_paths")
    if (
        incident.get("incident_kind") != "system_code_repair"
        or incident.get("job_id") != TRUSTED_OPERATIONAL_JOB_ID
        or incident.get("failure_class") != TRUSTED_OPERATIONAL_FAILURE_CLASS
        or not isinstance(linked_fingerprint, str)
        or not linked_fingerprint
        or incident.get("fingerprint") != linked_fingerprint
        or incident.get("source_failure_class") != source_packet.get("failure_class")
        or incident.get("source_fingerprint") != source_packet.get("fingerprint")
        or not isinstance(source_paths, list)
        or str(source_packet_path) not in source_paths
    ):
        return {"accepted": False, "reason": "linked_system_incident_binding_invalid"}

    status = str(incident.get("status") or "")
    try:
        frontier_attempts = int(incident.get("frontier_attempts") or 0)
    except (TypeError, ValueError):
        return {"accepted": False, "reason": "linked_system_incident_binding_invalid"}
    frontier_result = incident.get("frontier_result")
    execution_started = bool(
        isinstance(frontier_result, Mapping)
        and frontier_result.get("execution_started") is True
    )
    base = {
        "linked": True,
        "incident_path": str(incident_path),
        "incident_fingerprint": incident.get("fingerprint"),
        "observed_status": status,
        "frontier_attempts": frontier_attempts,
        "execution_started": execution_started,
    }
    if status == VERIFIED_LOCAL_REPAIR_CANCELLATION_STATUS:
        cancellation = _read_packet_cancellation(incident_path, incident)
        if (
            cancellation is None
            or cancellation.get("status") != VERIFIED_LOCAL_REPAIR_CANCELLATION_STATUS
            or cancellation.get("superseded_by_packet") != str(source_packet_path)
        ):
            return {
                "accepted": False,
                "reason": "linked_system_incident_cancellation_invalid",
                **base,
            }
        return {"accepted": True, "state": "superseded", **base}
    if status == "frontier_running" or frontier_attempts > 0 or execution_started:
        return {
            "accepted": False,
            "reason": "linked_system_incident_already_started",
            **base,
        }
    if status not in _SYSTEM_INCIDENT_PRESTART_STATUSES:
        return {
            "accepted": False,
            "reason": "linked_system_incident_terminal",
            **base,
        }
    return {"accepted": True, "state": "prestart", **base}


def _prepare_linked_incident_for_verified_release(
    source_packet_path: Path,
    source_packet: Mapping[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    """Durably close an unstarted incident before releasing its source raws."""

    inspected = _inspect_linked_operational_incident(
        source_packet_path,
        source_packet,
    )
    if not inspected.get("accepted") or not inspected.get("linked"):
        return inspected
    if dry_run:
        return {
            **inspected,
            "would_supersede": inspected.get("state") == "prestart",
            "would_cancel_background_job": True,
        }

    incident_path = Path(str(inspected["incident_path"]))
    with _packet_lock(incident_path) as acquired:
        if not acquired:
            return {
                "accepted": False,
                "reason": "linked_system_incident_busy",
                **{
                    key: value
                    for key, value in inspected.items()
                    if key not in {"accepted", "reason"}
                },
            }
        inspected = _inspect_linked_operational_incident(
            source_packet_path,
            source_packet,
        )
        if not inspected.get("accepted"):
            return inspected
        if inspected.get("state") == "prestart":
            incident = _read_json(incident_path)
            cancellation = request_packet_cancellation(
                incident_path,
                reason="verified local repair superseded unstarted system incident",
                superseded_by_packet=source_packet_path,
                cancellation_status=VERIFIED_LOCAL_REPAIR_CANCELLATION_STATUS,
            )
            if not cancellation.get("accepted"):
                return {
                    "accepted": False,
                    "reason": "linked_system_incident_cancellation_refused",
                    **{
                        key: value
                        for key, value in inspected.items()
                        if key not in {"accepted", "reason"}
                    },
                }
            _apply_packet_cancellation(incident_path, incident, cancellation)
            inspected = _inspect_linked_operational_incident(
                source_packet_path,
                source_packet,
            )
            if not inspected.get("accepted") or inspected.get("state") != "superseded":
                return {
                    "accepted": False,
                    "reason": "linked_system_incident_cancellation_not_durable",
                    "incident_path": str(incident_path),
                }

        try:
            from chronovisor.core.background_jobs import cancel_matching_jobs

            background = cancel_matching_jobs(
                name="system-code-repair",
                module="chronovisor.ops.self_heal",
                args=["--packet", str(incident_path), "--enable-frontier-repair"],
                reason="verified local repair superseded system incident",
            )
        except Exception as exc:
            return {
                "accepted": False,
                "reason": "linked_system_incident_job_cancel_failed",
                "incident_path": str(incident_path),
                "error_type": exc.__class__.__name__,
            }
    return {
        **inspected,
        "superseded": True,
        "background_job_cancellation": background,
    }


def _release_operational_failure_after_local_repair_unlocked(
    packet_path: Path,
    *,
    affected_group: tuple[tuple[str, dict[str, Any]], ...],
    expected_status: str,
    expected_failure_class: str,
    expected_fingerprint: str,
    expected_raw_manifest: Mapping[str, str],
    repair_commit: str,
    reason: str,
    verification_command: str | None,
    verification_result: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    try:
        loaded = _read_json(packet_path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return _operational_release_refusal(packet_path, "packet_unreadable")
    if not isinstance(loaded, dict):
        return _operational_release_refusal(packet_path, "packet_invalid")
    packet = loaded

    def refuse(reason_code: str, **details: Any) -> dict[str, Any]:
        return _operational_release_refusal(
            packet_path,
            reason_code,
            packet=packet,
            **details,
        )

    from chronovisor.ingest.failure_supervisor import (
        SEMANTIC_NO_QUORUM_FAILURE_CLASS,
        verified_projection_child_bytes,
    )

    if (
        packet.get("failure_class") == SEMANTIC_NO_QUORUM_FAILURE_CLASS
        or packet.get("terminal_deferred") is True
        or packet.get("defer_reason") == "semantic_no_quorum"
    ):
        return refuse("semantic_defer_not_releasable")
    if packet.get("failure_class") != expected_failure_class:
        return refuse("failure_class_mismatch")
    if packet.get("fingerprint") != expected_fingerprint:
        return refuse("fingerprint_mismatch")
    if not _is_operational_source_packet(packet):
        return refuse("failure_class_not_operational")
    if _read_packet_cancellation(packet_path, packet) is not None:
        return refuse("packet_cancellation_requested")

    status = str(packet.get("status") or "")
    existing_receipt = packet.get("verified_local_repair")
    normalized_reason = reason.strip()
    command = str(verification_command or "").strip()
    verification = str(verification_result or "").strip()
    normalized_repair_commit = repair_commit.strip().casefold()
    expected_manifest_rows = [
        {"filename": filename, "sha256": digest}
        for filename, digest in sorted(expected_raw_manifest.items())
    ]
    if status == "local_repair_applied":
        receipt_matches = bool(
            isinstance(existing_receipt, dict)
            and existing_receipt.get("failure_id") == packet.get("failure_id")
            and existing_receipt.get("packet_raw_file") == packet.get("raw_file")
            and existing_receipt.get("expected_status") == expected_status
            and existing_receipt.get("failure_class") == expected_failure_class
            and existing_receipt.get("fingerprint") == expected_fingerprint
            and existing_receipt.get("expected_raw_manifest") == expected_manifest_rows
            and existing_receipt.get("git_commit_sha") == normalized_repair_commit
            and existing_receipt.get("reason") == normalized_reason
            and existing_receipt.get("verification_command") == (command or None)
            and existing_receipt.get("verification_result") == (verification or None)
        )
        if receipt_matches:
            linked_incident = _prepare_linked_incident_for_verified_release(
                packet_path,
                packet,
                dry_run=dry_run,
            )
            if not linked_incident.get("accepted"):
                return refuse(
                    str(
                        linked_incident.get("reason")
                        or "linked_system_incident_refused"
                    ),
                    linked_system_incident=linked_incident,
                )
            return {
                "packet": str(packet_path),
                "failure_id": packet.get("failure_id"),
                "status": "dry_run" if dry_run else "local_repair_applied",
                "projected_status": "local_repair_applied" if dry_run else None,
                "accepted": True,
                "cached": True,
                "verified_local_repair": existing_receipt,
                "linked_system_incident": linked_incident,
            }
        return refuse("completed_packet_repair_evidence_mismatch")

    if status != expected_status:
        return refuse(
            "packet_status_mismatch",
            expected_status=expected_status,
            observed_status=status,
        )
    if status in RUNNING_STATUSES:
        return refuse("packet_already_running")
    if status not in _VERIFIED_LOCAL_REPAIR_RELEASABLE_STATUSES:
        return refuse("packet_status_not_releasable")

    if not affected_group:
        return refuse("failure_state_group_missing")
    if any(
        entry.get("failure_class") != expected_failure_class
        or entry.get("fingerprint") != expected_fingerprint
        for _raw_file, entry in affected_group
    ):
        return refuse("failure_state_group_mismatch")

    observed_group_names = {raw_file for raw_file, _entry in affected_group}
    expected_group_names = set(expected_raw_manifest)
    if observed_group_names != expected_group_names:
        return refuse(
            "failure_state_group_manifest_mismatch",
            expected_raw_files=sorted(expected_group_names),
            observed_raw_files=sorted(observed_group_names),
        )

    affected_raws: list[dict[str, Any]] = []
    legacy_state_raws: list[str] = []
    try:
        from chronovisor.core.raw_store import RawStore

        raw_store = RawStore(chronovisor_store.RAW_DIR)
        for raw_file, entry in affected_group:
            if Path(raw_file).name != raw_file:
                raise ValueError("affected_raw_invalid")
            unit = raw_store.resolve(raw_file)
            raw = (
                raw_store.read_bytes(unit)
                if unit is not None
                else verified_projection_child_bytes(
                    raw_file,
                    artifact_dir=(
                        chronovisor_store.CHRONOVISOR_ROOT
                        / "runtime"
                        / "raw-projections"
                        / "artifacts"
                    ),
                )
            )
            digest = hashlib.sha256(raw).hexdigest()
            expected_digest = expected_raw_manifest[raw_file]
            if digest != expected_digest:
                return refuse(
                    "expected_raw_sha256_mismatch",
                    raw_file=raw_file,
                    expected_sha256=expected_digest,
                    observed_sha256=digest,
                )

            stored_digest = entry.get("raw_sha256")
            stored_bytes = entry.get("raw_bytes")
            has_stored_digest = "raw_sha256" in entry
            has_stored_bytes = "raw_bytes" in entry
            if has_stored_digest != has_stored_bytes:
                return refuse("failure_state_raw_binding_invalid", raw_file=raw_file)
            binding_source = "failure_state"
            if not has_stored_digest:
                binding_source = "expected_manifest_legacy"
                legacy_state_raws.append(raw_file)
            else:
                if (
                    not isinstance(stored_digest, str)
                    or _RAW_SHA256_RE.fullmatch(stored_digest) is None
                    or isinstance(stored_bytes, bool)
                    or not isinstance(stored_bytes, int)
                    or stored_bytes < 0
                ):
                    return refuse(
                        "failure_state_raw_binding_invalid", raw_file=raw_file
                    )
                if stored_digest != expected_digest or stored_bytes != len(raw):
                    return refuse(
                        "failure_state_raw_binding_mismatch",
                        raw_file=raw_file,
                        state_sha256=stored_digest,
                        expected_sha256=expected_digest,
                        state_bytes=stored_bytes,
                        observed_bytes=len(raw),
                    )
            affected_raws.append(
                {
                    "filename": raw_file,
                    "sha256": digest,
                    "bytes": len(raw),
                    "binding_source": binding_source,
                }
            )
    except (OSError, ValueError):
        return refuse("affected_raw_evidence_unavailable")

    if not normalized_reason:
        return refuse("repair_reason_required")
    if bool(command) != bool(verification):
        return refuse("verification_command_and_result_must_be_provided_together")
    if max(len(normalized_reason), len(command), len(verification)) > 8000:
        return refuse("repair_evidence_field_too_large")
    try:
        git_state = _verified_local_repair_git_state(repair_commit)
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
        return refuse(str(exc))
    evidence = {
        "failure_id": packet.get("failure_id"),
        "expected_status": expected_status,
        "failure_class": expected_failure_class,
        "fingerprint": expected_fingerprint,
        "packet_raw_file": packet.get("raw_file"),
        "reason": normalized_reason,
        **git_state,
        "affected_raw_scope": "fingerprint_group",
        "affected_raws": affected_raws,
        "expected_raw_manifest": expected_manifest_rows,
        "legacy_state_raws": sorted(legacy_state_raws),
        "verification_command": command or None,
        "verification_result": verification or None,
    }

    linked_incident = _prepare_linked_incident_for_verified_release(
        packet_path,
        packet,
        dry_run=dry_run,
    )
    if not linked_incident.get("accepted"):
        return refuse(
            str(linked_incident.get("reason") or "linked_system_incident_refused"),
            linked_system_incident=linked_incident,
        )
    evidence["linked_system_incident"] = (
        linked_incident if linked_incident.get("linked") else None
    )

    if dry_run:
        return {
            "packet": str(packet_path),
            "failure_id": packet.get("failure_id"),
            "status": "dry_run",
            "projected_status": "local_repair_applied",
            "accepted": True,
            "cached": False,
            "verified_local_repair": evidence,
            "linked_system_incident": linked_incident,
        }

    applied_at = datetime.now().isoformat()
    evidence["recorded_at"] = applied_at
    packet_updates: dict[str, Any] = {
        "status": "local_repair_applied",
        "self_heal_queued": False,
        "next_attempt_at": None,
        "frontier_status": "not_required",
        "verified_local_repair": evidence,
        "verified_local_repair_applied_at": applied_at,
    }
    active_semantic_hold = packet.get("semantic_hold")
    if status == "local_quarantined":
        packet_updates.update(
            {
                "terminal_reason": None,
                "quarantined_at": None,
            }
        )
    if active_semantic_hold is not None:
        packet_updates.update(
            {
                "semantic_hold": None,
                "semantic_hold_history": _semantic_hold_history_with(
                    packet,
                    active_semantic_hold,
                ),
                "invalidated_semantic_hold": active_semantic_hold,
                "semantic_hold_invalidated_at": applied_at,
            }
        )
    _update_packet(
        packet_path,
        packet,
        **packet_updates,
    )
    return {
        "packet": str(packet_path),
        "failure_id": packet.get("failure_id"),
        "status": "local_repair_applied",
        "accepted": True,
        "cached": False,
        "verified_local_repair": evidence,
        "linked_system_incident": linked_incident,
    }


def release_operational_failure_after_local_repair(
    packet_path: Path,
    *,
    expected_status: str,
    expected_failure_class: str,
    expected_fingerprint: str,
    expected_raw_sha256: Mapping[str, str] | Sequence[str],
    repair_commit: str,
    reason: str,
    verification_command: str | None = None,
    verification_result: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Release immutable raws only after a guarded operational repair ACK.

    Expected status, class, fingerprint, and the complete filename/SHA group are
    one compare-and-swap guard. Semantic no-quorum packets and terminal outcomes
    are deliberately never writable by this boundary. Dry runs do not create
    the packet lock or mutate any file.
    """

    requested = packet_path.expanduser()
    try:
        if requested.is_symlink():
            raise ValueError("packet_symlink_not_allowed")
        resolved = requested.resolve(strict=True)
        packet_root = _packet_dir().expanduser().resolve(strict=True)
    except (OSError, ValueError) as exc:
        reason_code = str(exc) if str(exc).startswith("packet_") else "packet_not_found"
        return _operational_release_refusal(requested, reason_code)
    if resolved.parent != packet_root or resolved.suffix != ".json":
        return _operational_release_refusal(
            resolved,
            "packet_outside_failure_packet_dir",
        )

    normalized_status = expected_status.strip()
    if not normalized_status:
        return _operational_release_refusal(resolved, "expected_status_required")
    try:
        expected_raw_manifest = _normalize_expected_raw_manifest(expected_raw_sha256)
    except ValueError as exc:
        return _operational_release_refusal(resolved, str(exc))

    kwargs = {
        "expected_status": normalized_status,
        "expected_failure_class": expected_failure_class.strip(),
        "expected_fingerprint": expected_fingerprint.strip(),
        "expected_raw_manifest": expected_raw_manifest,
        "repair_commit": repair_commit.strip(),
        "reason": reason,
        "verification_command": verification_command,
        "verification_result": verification_result,
        "dry_run": dry_run,
    }
    from chronovisor.ingest.failure_supervisor import (
        lock_operational_failure_group,
        operational_failure_group_snapshot,
    )

    if dry_run:
        kwargs["affected_group"] = operational_failure_group_snapshot(resolved)
        return _release_operational_failure_after_local_repair_unlocked(
            resolved, **kwargs
        )
    with _packet_lock(resolved) as acquired:
        if not acquired:
            return _operational_release_refusal(
                resolved,
                "packet_already_running",
                observed_status="busy",
            )
        with lock_operational_failure_group(resolved) as affected_group:
            kwargs["affected_group"] = affected_group
            try:
                return _release_operational_failure_after_local_repair_unlocked(
                    resolved, **kwargs
                )
            except _PacketCancellationRequested as exc:
                return _operational_release_refusal(
                    resolved,
                    "packet_cancellation_requested",
                    cancellation=exc.result,
                )


def _append_registry(record: dict[str, Any]) -> None:
    path = _registry_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _read_notification_state() -> dict[str, Any]:
    path = _notification_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"notifications": {}}
    if not isinstance(data, dict) or not isinstance(data.get("notifications"), dict):
        return {"notifications": {}}
    return data


def _write_notification_state(state: dict[str, Any]) -> None:
    _write_json(_notification_file(), state)


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _comparable_datetimes(left: datetime, right: datetime) -> tuple[datetime, datetime]:
    """Normalize legacy naive timestamps against newer aware timestamps."""

    if left.tzinfo is not None and right.tzinfo is None:
        left = left.replace(tzinfo=None)
    elif left.tzinfo is None and right.tzinfo is not None:
        right = right.replace(tzinfo=None)
    return left, right


def _running_lease_seconds() -> int:
    try:
        return max(
            0,
            int(
                os.environ.get(
                    "CHRONOVISOR_SELF_HEAL_RUNNING_LEASE_SECONDS",
                    DEFAULT_RUNNING_LEASE_SECONDS,
                )
            ),
        )
    except ValueError:
        return DEFAULT_RUNNING_LEASE_SECONDS


def _env_seconds(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _terminal_resume_kind(
    packet: dict[str, Any],
    *,
    now: datetime | None = None,
) -> str | None:
    current = now or datetime.now()
    status = str(packet.get("status") or "")
    if status == "human_required":
        if not is_human_required_result(packet.get("frontier_result")):
            return "legacy_nonhuman"
        started = _parse_iso(
            packet.get("human_required_at") or packet.get("updated_at")
        )
        cooldown = _env_seconds(
            "CHRONOVISOR_HUMAN_REQUIRED_RECHECK_SECONDS",
            DEFAULT_HUMAN_RECHECK_SECONDS,
        )
        kind = "external_authority_recheck"
    elif status == "frontier_quarantined":
        started = _parse_iso(packet.get("quarantined_at") or packet.get("updated_at"))
        cooldown = _env_seconds(
            "CHRONOVISOR_CONVERGENCE_QUARANTINE_RETRY_SECONDS",
            DEFAULT_QUARANTINE_RETRY_SECONDS,
        )
        kind = "quarantine_cooldown"
    else:
        return None
    if started is None:
        return None
    started, comparable_now = _comparable_datetimes(started, current)
    return kind if (comparable_now - started).total_seconds() >= cooldown else None


def _resume_terminal_packet(
    packet_path: Path,
    packet: dict[str, Any],
    *,
    now: datetime | None = None,
    dry_run: bool = False,
) -> str | None:
    kind = _terminal_resume_kind(packet, now=now)
    if kind is None:
        return None
    if dry_run:
        return kind
    field = (
        "human_recheck_count"
        if kind == "external_authority_recheck"
        else "quarantine_reopen_count"
    )
    _update_packet(
        packet_path,
        packet,
        status="frontier_retry",
        frontier_attempts=0,
        self_heal_attempts=0,
        next_attempt_at=None,
        terminal_resume_kind=kind,
        terminal_resumed_at=(now or datetime.now()).isoformat(timespec="seconds"),
        **{field: int(packet.get(field) or 0) + 1},
    )
    return kind


def _lease_updates(owner: str, *, now: datetime | None = None) -> dict[str, Any]:
    current = now or datetime.now()
    return {
        "lease_owner": owner,
        "lease_expires_at": (
            current + timedelta(seconds=_running_lease_seconds())
        ).isoformat(timespec="seconds"),
    }


def _running_lease_expired(
    packet: dict[str, Any],
    *,
    now: datetime | None = None,
) -> bool:
    current = now or datetime.now()
    expiry = _parse_iso(packet.get("lease_expires_at"))
    if expiry is None:
        started = _parse_iso(packet.get("updated_at")) or _parse_iso(
            packet.get("last_attempt_at")
        )
        if started is None:
            return True
        expiry = started + timedelta(seconds=_running_lease_seconds())
    expiry, current = _comparable_datetimes(expiry, current)
    return expiry <= current


def _notification_cooldown_seconds() -> int:
    try:
        return int(
            os.environ.get(
                "CHRONOVISOR_MAC_NOTIFICATION_COOLDOWN_SECONDS",
                MAC_NOTIFICATION_COOLDOWN_SECONDS,
            )
        )
    except ValueError:
        return MAC_NOTIFICATION_COOLDOWN_SECONDS


def _notification_key(packet: dict[str, Any], frontier_result: dict[str, Any]) -> str:
    failure = (
        frontier_result.get("frontier_failure")
        if isinstance(frontier_result.get("frontier_failure"), dict)
        else {}
    )
    return ":".join(
        str(part or "unknown")
        for part in (
            packet.get("fingerprint")
            or packet.get("failure_id")
            or packet.get("raw_file"),
            failure.get("failure_class") or frontier_result.get("rescue_status"),
        )
    )


def _human_notification_body(
    packet: dict[str, Any], frontier_result: dict[str, Any]
) -> str:
    failure = (
        frontier_result.get("frontier_failure")
        if isinstance(frontier_result.get("frontier_failure"), dict)
        else {}
    )
    failure_class = failure.get("failure_class") or "frontier"
    if failure_class in {"auth_required", "oauth_required"}:
        return "Codex の認証が切れている可能性があります。ログイン確認が必要です。"
    if failure_class == "quota_or_billing_required":
        return "Codex の quota または billing の確認が必要です。"
    if failure_class == "keychain_permission_required":
        return "Codex の Keychain アクセス許可が必要です。"
    if failure_class == "secret_store_permission_required":
        return "Codex の認証情報ストアへのアクセス許可が必要です。"
    raw_file = packet.get("raw_file")
    if isinstance(raw_file, str) and raw_file:
        return f"自己修復に人間の確認が必要です: {Path(raw_file).name}"
    return "自己修復に人間の確認が必要です。"


def _send_mac_notification(title: str, body: str) -> dict[str, Any]:
    if os.environ.get("CHRONOVISOR_MAC_NOTIFICATIONS", "1") in {"0", "false", "False"}:
        return {"sent": False, "reason": "disabled"}
    osascript = "/usr/bin/osascript"
    if not Path(osascript).exists():
        found = shutil.which("osascript")
        if found is None:
            return {"sent": False, "reason": "osascript not found"}
        osascript = found
    script = f"display notification {json.dumps(body)} with title {json.dumps(title)}"
    try:
        completed = subprocess.run(
            [osascript, "-e", script],
            text=True,
            capture_output=True,
            timeout=5,
        )
    except Exception as exc:
        return {"sent": False, "reason": str(exc)}
    return {
        "sent": completed.returncode == 0,
        "exit_code": completed.returncode,
        "stderr": (completed.stderr or "")[-500:],
    }


def maybe_notify_human_required(
    packet: dict[str, Any],
    frontier_result: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not is_human_required_result(frontier_result):
        return {"sent": False, "reason": "not human required"}
    now = now or datetime.now()
    key = _notification_key(packet, frontier_result)
    state = _read_notification_state()
    notifications = state.setdefault("notifications", {})
    prior = notifications.get(key) if isinstance(notifications.get(key), dict) else {}
    prior_at = _parse_iso(prior.get("last_notified_at"))
    cooldown = _notification_cooldown_seconds()
    if prior_at is not None and prior_at.tzinfo is not None and now.tzinfo is None:
        prior_at = prior_at.replace(tzinfo=None)
    if prior_at is not None and prior_at.tzinfo is None and now.tzinfo is not None:
        now = now.replace(tzinfo=None)
    if prior_at is not None and (now - prior_at).total_seconds() < cooldown:
        return {
            "sent": False,
            "reason": "cooldown",
            "key": key,
            "last_notified_at": prior_at.isoformat(),
            "cooldown_seconds": cooldown,
        }
    body = _human_notification_body(packet, frontier_result)
    delivery = _send_mac_notification(MAC_NOTIFICATION_TITLE, body)
    record = {
        "last_notified_at": now.isoformat(),
        "title": MAC_NOTIFICATION_TITLE,
        "body": body,
        "delivery": delivery,
    }
    notifications[key] = record
    _write_notification_state(state)
    return {"key": key, **record}


def _update_packet(path: Path, packet: dict[str, Any], **updates: Any) -> None:
    _raise_if_packet_cancelled(path, packet)
    next_status = updates.get("status", packet.get("status"))
    if next_status not in RUNNING_STATUSES:
        updates.setdefault("lease_owner", None)
        updates.setdefault("lease_expires_at", None)
    packet.update(updates)
    packet["updated_at"] = datetime.now().isoformat()
    _write_json(path, packet)


def pending_packets(
    *,
    now: datetime | None = None,
    lock_authority: bool = True,
) -> list[Path]:
    if not _packet_dir().exists():
        return []
    current = now or datetime.now()
    out: list[Path] = []
    for path in sorted(_packet_dir().glob("*.json")):
        try:
            packet = _read_json(path)
        except Exception:
            continue
        if _read_packet_cancellation(path, packet) is not None:
            continue
        next_attempt = _parse_iso(packet.get("next_attempt_at"))
        due = True
        if next_attempt is not None:
            next_attempt, comparable_now = _comparable_datetimes(next_attempt, current)
            due = next_attempt <= comparable_now
        status = packet.get("status")
        if status in SELF_HEAL_STATUSES and due:
            out.append(path)
        elif status == "local_quarantined":
            _exact_hold, stale_hold = _current_local_semantic_hold(
                packet,
                lock_authority=lock_authority,
            )
            if stale_hold is not None:
                out.append(path)
        elif packet.get("incident_kind") == "system_code_repair" and (
            int(packet.get("frontier_attempts") or 0) > 0
            or (
                isinstance(packet.get("frontier_result"), dict)
                and packet["frontier_result"].get("execution_started") is True
            )
        ):
            # A started repair incident owns exactly one subscription session
            # forever.  Generic quarantine cooldowns must not resurrect it.
            continue
        elif _terminal_resume_kind(packet, now=current) is not None:
            # Non-human quarantines reopen after cooldown. Genuine external
            # authority boundaries are periodically rechecked so fixing auth,
            # billing or keychain state never requires a queue-side manual ack.
            out.append(path)
        elif status in RUNNING_STATUSES and _running_lease_expired(packet, now=current):
            out.append(path)
    return out


def _raw_candidate_paths(packet: dict[str, Any]) -> list[Path]:
    raw_file = packet.get("raw_file")
    if not isinstance(raw_file, str) or not raw_file:
        return []
    return [
        _failures_dir() / "quarantined-raw" / raw_file,
        chronovisor_store.RAW_DIR / raw_file,
    ]


def _restore_quarantined_raw(
    packet: dict[str, Any], *, dry_run: bool
) -> dict[str, Any]:
    candidates = _raw_candidate_paths(packet)
    if not candidates:
        return {"restored": False, "reason": "packet has no raw_file"}
    quarantine_path, raw_path = candidates[0], candidates[1]
    if raw_path.exists():
        return {
            "restored": False,
            "reason": "raw already pending",
            "path": str(raw_path),
        }
    if not quarantine_path.exists():
        return {"restored": False, "reason": "quarantined raw not found"}
    if dry_run:
        return {
            "restored": False,
            "dry_run": True,
            "source": str(quarantine_path),
            "target": str(raw_path),
        }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(quarantine_path), str(raw_path))
    return {"restored": True, "source": str(quarantine_path), "target": str(raw_path)}


def _retry_ingest(*, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"triggered": False, "dry_run": True}
    from chronovisor.ingest import orchestrator

    return orchestrator.run_pending_ingest(force=True)


def apply_local_decision(
    packet: dict[str, Any],
    decision: LocalRepairDecision,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Apply a whitelisted local repair action."""

    if _is_operational_source_packet(packet):
        raise ValueError(
            "operational source packets require the guarded system-incident lane"
        )

    if decision.action == "resolve_update_target":
        if not decision.requested_page_id or not decision.target_page_id:
            raise ValueError("resolve_update_target requires requested and target ids")
        if not dry_run:
            add_alias(
                decision.requested_page_id,
                decision.target_page_id,
                source=packet.get("failure_id"),
            )
        restore = _restore_quarantined_raw(packet, dry_run=dry_run)
        retry = _retry_ingest(dry_run=dry_run)
        return {
            "action": decision.action,
            "alias": {
                "requested": decision.requested_page_id,
                "target": decision.target_page_id,
                "dry_run": dry_run,
            },
            "restore": restore,
            "retry": retry,
        }

    if decision.action == "retry_raw":
        restore = _restore_quarantined_raw(packet, dry_run=dry_run)
        retry = _retry_ingest(dry_run=dry_run)
        return {"action": decision.action, "restore": restore, "retry": retry}

    if decision.action == "quarantine_raw":
        return {"action": decision.action, "kept_quarantined": True}

    raise ValueError(
        "local action is not directly applicable and requires the guarded "
        f"system-repair lane: {decision.action}"
    )


def _local_decision_from_payload(
    payload: dict[str, Any] | None,
) -> LocalRepairDecision | None:
    if not isinstance(payload, dict):
        return None
    try:
        return LocalRepairDecision(
            status=str(payload["status"]),
            action=str(payload["action"]),
            confidence=float(payload["confidence"]),
            reason=str(payload["reason"]),
            requested_page_id=(
                str(payload["requested_page_id"])
                if payload.get("requested_page_id") is not None
                else None
            ),
            target_page_id=(
                str(payload["target_page_id"])
                if payload.get("target_page_id") is not None
                else None
            ),
            notes=str(payload["notes"]) if payload.get("notes") is not None else None,
            source=str(payload.get("source") or "persisted"),
            authority=(
                dict(payload["authority"])
                if isinstance(payload.get("authority"), dict)
                else None
            ),
            decision_policy=(
                dict(payload["decision_policy"])
                if isinstance(payload.get("decision_policy"), dict)
                else None
            ),
            local_consensus=(
                dict(payload["local_consensus"])
                if isinstance(payload.get("local_consensus"), dict)
                else None
            ),
            semantic_hold=(
                dict(payload["semantic_hold"])
                if isinstance(payload.get("semantic_hold"), dict)
                else None
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _packet_semantic_hold(packet: Mapping[str, Any]) -> dict[str, Any] | None:
    hold = persisted_semantic_no_quorum_hold(packet, "local_repair")
    if hold is not None:
        return hold
    local_decision = packet.get("local_decision")
    return persisted_semantic_no_quorum_hold(local_decision, "local_repair")


def _packet_semantic_holds(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return every valid historical hold without trusting packet metadata."""

    candidates: list[object] = [
        packet,
        packet.get("local_decision"),
        packet.get("invalidated_semantic_hold"),
    ]
    history = packet.get("semantic_hold_history")
    if isinstance(history, list):
        candidates.extend(history)
    holds: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        hold = persisted_semantic_no_quorum_hold(candidate, "local_repair")
        if hold is None:
            continue
        digest = str(hold["hold_sha256"])
        if digest in seen:
            continue
        seen.add(digest)
        holds.append(hold)
    return holds


def _semantic_hold_history_with(
    packet: Mapping[str, Any],
    *new_holds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Append strict holds once so authority A -> B -> A never re-samples A."""

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    history = packet.get("semantic_hold_history")
    candidates: list[object] = list(history) if isinstance(history, list) else []
    candidates.extend(
        [
            packet,
            packet.get("local_decision"),
            packet.get("invalidated_semantic_hold"),
            *new_holds,
        ]
    )
    for candidate in candidates:
        hold = persisted_semantic_no_quorum_hold(candidate, "local_repair")
        if hold is None:
            continue
        digest = str(hold["hold_sha256"])
        if digest in seen:
            continue
        seen.add(digest)
        ordered.append(hold)
    return ordered


def _current_local_semantic_hold(
    packet: dict[str, Any],
    *,
    lock_authority: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return (exact current hold, resumable stale hold).

    Malformed or legacy incomplete no-quorum records deliberately return
    ``(None, None)`` and remain fail-closed.  Only a structurally valid hold
    whose concrete evidence or adopted authority changed may reopen.
    """

    holds = _packet_semantic_holds(packet)
    if not holds:
        return None, None
    epoch = semantic_hold_epoch(packet)
    authority_epoch = decision_authority_lock() if lock_authority else nullcontext()
    with authority_epoch:
        authority, authority_error = current_semantic_authority("local_repair")
        if authority_error is not None or authority is None:
            return None, None
        errors = [
            semantic_no_quorum_hold_error(
                hold,
                "local_repair",
                epoch=epoch,
                authority=authority,
            )
            for hold in holds
        ]
    for hold, error in zip(holds, errors, strict=True):
        if error is None:
            return hold, None
    active_hold = _packet_semantic_hold(packet)
    if active_hold is not None:
        active_error = semantic_no_quorum_hold_error(
            active_hold,
            "local_repair",
            epoch=epoch,
            authority=authority,
        )
        if active_error in {
            "semantic hold epoch changed",
            "semantic hold authority changed",
        }:
            return None, active_hold
    return None, None


def _restore_invalidated_local_semantic_hold(
    packet_path: Path,
    packet: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any] | None:
    old_holds = _packet_semantic_holds(packet)
    if not old_holds:
        return None
    epoch = semantic_hold_epoch(packet)
    authority_epoch = nullcontext() if dry_run else decision_authority_lock()
    with authority_epoch:
        authority, authority_error = current_semantic_authority("local_repair")
        if authority_error is not None or authority is None:
            return None
        old_hold = next(
            (
                hold
                for hold in old_holds
                if semantic_no_quorum_hold_error(
                    hold,
                    "local_repair",
                    epoch=epoch,
                    authority=authority,
                )
                is None
            ),
            None,
        )
        if old_hold is None:
            return None
        if (
            semantic_no_quorum_hold_error(
                old_hold,
                "local_repair",
                epoch=epoch,
                authority=authority,
            )
            is not None
        ):
            return None
        result = {
            "packet": str(packet_path),
            "failure_id": packet.get("failure_id"),
            "status": "dry_run" if dry_run else "local_quarantined",
            "projected_status": "local_quarantined",
            "terminal_reason": "semantic_no_quorum",
            "restored_semantic_hold": True,
            "semantic_hold": old_hold,
        }
        if dry_run:
            return result
        _update_packet(
            packet_path,
            packet,
            status="local_quarantined",
            semantic_hold=old_hold,
            semantic_hold_history=_semantic_hold_history_with(packet, old_hold),
            invalidated_semantic_hold=None,
            terminal_reason="semantic_no_quorum",
            last_failure_class=LOCAL_SEMANTIC_NO_QUORUM,
            next_attempt_at=None,
            quarantined_at=datetime.now().isoformat(timespec="seconds"),
        )
    return result


def _local_decision_authority_error(decision: LocalRepairDecision) -> str | None:
    """Validate a local-consensus decision against the current effect epoch."""

    if decision.source != "local_consensus":
        return None
    if not isinstance(decision.authority, dict):
        return "local repair decision authority is missing"
    # Include the full LOCAL_REPAIR_SCHEMA action.  Passing only the audit
    # envelope would make canonical action/hash verification fail closed for
    # every real local-consensus repair.
    review = decision.to_dict()
    verdict_error = semantic_verdict_authority_error(
        review,
        decision.authority,
        lane="local_repair",
    )
    if verdict_error is not None:
        return verdict_error
    current, current_error = current_semantic_authority("local_repair")
    if current is None or current_error is not None:
        return current_error or "local repair decision authority is unavailable"
    return compare_semantic_authority(
        decision.authority,
        current,
        lane="local_repair",
    )


@contextmanager
def _local_decision_effect(decision: LocalRepairDecision | None):
    """Hold the shared authority epoch across one semantic state transition."""

    if decision is None or decision.source != "local_consensus":
        yield
        return
    with decision_authority_lock():
        error = _local_decision_authority_error(decision)
        if error is not None:
            raise RuntimeError(error)
        yield


def _save_local_decision(packet_path: Path, decision: LocalRepairDecision) -> Path:
    path = _local_repair_dir() / packet_path.name
    _write_json(path, decision.to_dict())
    return path


def _save_action(packet_path: Path, action: dict[str, Any], *, applied: bool) -> Path:
    target_dir = _applied_actions_dir() if applied else _rejected_actions_dir()
    path = target_dir / packet_path.name
    _write_json(path, action)
    return path


def _queue_frontier(
    packet_path: Path, packet: dict[str, Any], decision: dict[str, Any] | None
) -> Path:
    target = _frontier_queue_dir() / packet_path.name
    payload = {
        "queued_at": datetime.now().isoformat(),
        "packet_path": str(packet_path),
        "packet": packet,
        "local_decision": decision,
    }
    _write_json(target, payload)
    return target


def _trusted_repair_packet_job_id(
    packet: dict[str, Any],
    evidence_payload: dict[str, Any],
) -> str | None:
    notes = evidence_payload.get("notes")
    if not isinstance(notes, dict):
        return None
    contract = (
        notes.get("producer"),
        evidence_payload.get("component"),
        evidence_payload.get("failure_class"),
    )
    expected_job_id = _TRUSTED_REPAIR_PACKET_CONTRACTS.get(contract)
    if expected_job_id is None or packet.get("job_id") != expected_job_id:
        return None
    if packet.get("failure_class") != evidence_payload.get("failure_class"):
        return None
    if contract[0] == "trusted_operational_failure_supervisor" and (
        packet.get("source_failure_class") != notes.get("source_failure_class")
        or packet.get("source_fingerprint") != notes.get("source_fingerprint")
        or not isinstance(packet.get("source_packet_paths"), list)
        or not packet.get("source_packet_paths")
        or not isinstance(packet.get("raw_files"), list)
        or not packet.get("raw_files")
    ):
        return None
    return expected_job_id


def _is_operational_source_packet(packet: dict[str, Any]) -> bool:
    try:
        from chronovisor.ingest.failure_supervisor import (
            OPERATIONAL_SELF_HEAL_FAILURE_CLASSES,
        )
    except ImportError:
        return False
    return packet.get("failure_class") in OPERATIONAL_SELF_HEAL_FAILURE_CLASSES


def _operational_attempt_evidence(
    packet: dict[str, Any],
    decision: dict[str, Any],
    attempt: int,
) -> str:
    payload = {
        "attempt": attempt,
        "failure_class": packet.get("failure_class"),
        "fingerprint": packet.get("fingerprint"),
        "decision": decision,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _next_operational_attempt_evidence(
    packet: dict[str, Any],
    decision: dict[str, Any],
    attempt: int,
) -> list[str] | None:
    if not _is_operational_source_packet(packet):
        return None
    current = packet.get("operational_local_repair_evidence")
    evidence = (
        [
            value
            for value in current
            if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
        ]
        if isinstance(current, list)
        else []
    )
    receipt = _operational_attempt_evidence(packet, decision, attempt)
    if receipt not in evidence:
        evidence.append(receipt)
    return evidence[-16:]


def _promote_operational_source_packet(
    packet_path: Path,
    packet: dict[str, Any],
) -> dict[str, Any] | None:
    if not _is_operational_source_packet(packet):
        return None
    try:
        from chronovisor.ingest.system_incident_supervisor import (
            supervise_operational_failure_packet,
        )

        incident = supervise_operational_failure_packet(packet_path)
    except Exception as exc:
        runtime_status.safe_append_event(
            "warn",
            f"self-heal | operational incident bridge failed: {exc.__class__.__name__}",
            source="self-heal",
            packet=str(packet_path),
            outcome_kind="operational_incident_bridge_failed",
        )
        return {
            "status": "attention",
            "reason": "operational_incident_bridge_failed",
            "error_type": exc.__class__.__name__,
        }
    incident_path = incident.get("packet_path")
    if isinstance(incident_path, str) and incident_path:
        _update_packet(
            packet_path,
            packet,
            system_incident_packet_path=incident_path,
            system_incident_fingerprint=incident.get("fingerprint"),
            system_incident_status=incident.get("status"),
        )
    return incident


def _sync_system_incident_outcome(packet_path: Path) -> dict[str, Any] | None:
    try:
        from chronovisor.ingest.system_incident_supervisor import (
            sync_operational_incident_outcome,
        )

        result = sync_operational_incident_outcome(packet_path)
    except Exception as exc:
        runtime_status.safe_append_event(
            "warn",
            f"self-heal | operational incident sync failed: {exc.__class__.__name__}",
            source="self-heal",
            packet=str(packet_path),
            outcome_kind="operational_incident_sync_failed",
        )
        return None
    return result if result.get("reason") != "not_operational_incident" else None


def _repair_incident_evidence(packet: dict[str, Any]) -> RepairIncidentEvidence:
    """Return the strict repair-plane evidence carried by ``packet``.

    ``enable_frontier`` is an operator capability switch, not an admission
    decision.  Admission additionally requires an explicit system-code
    incident and a complete evidence envelope.  Routine semantic, content,
    and structured-output failures intentionally have neither field.
    """

    if packet.get("incident_kind") != "system_code_repair":
        raise EvidenceValidationError(
            ["incident_kind must explicitly be system_code_repair"]
        )
    payload = packet.get("repair_evidence")
    if not isinstance(payload, dict):
        raise EvidenceValidationError(["repair_evidence object is required"])

    required = {
        "component",
        "fingerprint",
        "failure_class",
        "occurrence_count",
        "distinct_inputs",
        "local_repair_attempts",
        "local_repair_evidence",
    }
    missing = sorted(key for key in required if key not in payload)
    if missing:
        raise EvidenceValidationError(
            [f"repair_evidence is missing required fields: {', '.join(missing)}"]
        )

    if _trusted_repair_packet_job_id(packet, payload) is None:
        raise EvidenceValidationError(
            ["packet was not emitted by an allowlisted trusted incident producer"]
        )

    packet_fingerprint = packet.get("fingerprint")
    evidence_fingerprint = payload.get("fingerprint")
    if packet_fingerprint != evidence_fingerprint:
        raise EvidenceValidationError(
            ["repair_evidence fingerprint must match the packet fingerprint"]
        )
    if packet.get("local_repair_attempts") != payload.get("local_repair_attempts"):
        raise EvidenceValidationError(
            [
                "repair_evidence local_repair_attempts must match the "
                "persisted packet attempt count"
            ]
        )

    reproduction = payload.get("reproduction")
    reproduction_payload = reproduction if isinstance(reproduction, dict) else {}
    distinct_inputs = payload.get("distinct_inputs")
    reproduction_command = payload.get(
        "reproduction_command",
        reproduction_payload.get("command"),
    )
    return RepairIncidentEvidence(
        component=payload["component"],
        fingerprint=evidence_fingerprint,
        failure_class=payload["failure_class"],
        occurrence_count=payload["occurrence_count"],
        distinct_inputs=(
            tuple(distinct_inputs)
            if isinstance(distinct_inputs, (list, tuple))
            else distinct_inputs
        ),
        local_repair_attempts=payload["local_repair_attempts"],
        local_repair_evidence=(
            tuple(payload.get("local_repair_evidence") or ())
            if isinstance(payload.get("local_repair_evidence"), (list, tuple))
            else payload.get("local_repair_evidence")
        ),
        reproduction_command=(
            tuple(reproduction_command)
            if isinstance(reproduction_command, (list, tuple))
            else reproduction_command
        ),
        failing_test=payload.get(
            "failing_test",
            reproduction_payload.get("failing_test"),
        ),
        reproduction_artifact=payload.get(
            "reproduction_artifact",
            reproduction_payload.get("artifact"),
        ),
        all_local_models_unavailable=payload.get(
            "all_local_models_unavailable",
            False,
        ),
        local_unavailability_artifact=payload.get("local_unavailability_artifact"),
        role=payload.get("role", "code_repair"),
        incident_kind=payload.get("incident_kind", "system_code_repair"),
        notes=payload.get("notes") or {},
    )


def _frontier_eligibility(
    packet: dict[str, Any],
) -> tuple[RepairIncidentEvidence | None, str | None]:
    try:
        return _repair_incident_evidence(packet), None
    except (EvidenceValidationError, TypeError, ValueError) as exc:
        return None, str(exc)


def _human_boundary_result(
    packet_path: Path,
    packet: dict[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any] | None:
    """Stop external-authority failures before any local/frontier model call."""

    failure_class = str(packet.get("failure_class") or "")
    if not is_human_required_failure(failure_class):
        return None
    result: dict[str, Any] = {
        "packet": str(packet_path),
        "failure_id": packet.get("failure_id"),
        "status": "dry_run" if dry_run else "human_required",
        "reason": "external_authority_boundary",
        "failure_class": failure_class,
    }
    if dry_run:
        result["projected_status"] = "human_required"
        return result

    frontier_result = {
        "decision": "needs_retry",
        "summary": "external authority must be restored by the user",
        "human_required": True,
        "notify_user": True,
        "frontier_failure": {
            "failure_class": failure_class,
            "rescue_status": "human_required",
            "summary": "external authority must be restored by the user",
            "human_required": True,
            "notify_user": True,
        },
        "rescue_status": "human_required",
    }
    notification = maybe_notify_human_required(packet, frontier_result)
    _update_packet(
        packet_path,
        packet,
        status="human_required",
        frontier_result=frontier_result,
        frontier_status="not_attempted",
        human_notification=notification,
        human_required_at=datetime.now().isoformat(timespec="seconds"),
        next_attempt_at=None,
    )
    result["frontier_result"] = frontier_result
    result["human_notification"] = notification
    return result


def _run_frontier(
    packet_path: Path,
    packet: dict[str, Any],
    local_decision: dict[str, Any] | None,
    *,
    evidence: RepairIncidentEvidence,
    execute_patch: bool,
) -> dict[str, Any]:
    from chronovisor.decision.frontier_review import run_frontier_review

    result = run_frontier_review(
        packet,
        local_decision,
        repo_root=_repo_root(),
        execute_patch=execute_patch,
        evidence=evidence,
    )
    payload = result.to_dict()
    _write_json(_frontier_decision_dir() / packet_path.name, payload)
    return payload


def _save_pending_frontier_review(
    packet_path: Path,
    packet: dict[str, Any],
    local_decision: dict[str, Any] | None,
    frontier_result: dict[str, Any],
    *,
    status: str,
) -> Path:
    payload = {
        "queued_at": datetime.now().isoformat(),
        "status": status,
        "packet_path": str(packet_path),
        "packet": packet,
        "local_decision": local_decision,
        "frontier_result": frontier_result,
        "access_repair": frontier_result.get("access_repair"),
        "rescue_attempt": frontier_result.get("rescue_attempt"),
    }
    path = _pending_frontier_review_dir() / packet_path.name
    _write_json(path, payload)
    return path


def _budget_deferred_result(
    packet_path: Path,
    packet: dict[str, Any],
    *,
    kind: str,
    reason: str,
    local_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a no-progress budget deferral for pre-attempt gates."""

    result: dict[str, Any] = {
        "packet": str(packet_path),
        "failure_id": packet.get("failure_id"),
        "status": "budget_deferred",
        "budget_kind": kind,
        "reason": reason,
    }
    if local_decision is not None:
        result["local_decision"] = local_decision
    return result


def _frontier_final_status(frontier_result: dict[str, Any]) -> str:
    if is_human_required_result(frontier_result):
        return "human_required"
    if (
        frontier_result.get("decision") == "approved"
        and frontier_result.get("verified") is True
    ):
        return "frontier_approved"
    rescue_status = frontier_result.get("rescue_status")
    if rescue_status in PENDING_REVIEW_STATUSES:
        return str(rescue_status)
    if frontier_result.get("decision") == "needs_retry":
        return "frontier_retry"
    if frontier_result.get("decision") == "quarantined":
        return "frontier_quarantined"
    return "frontier_rejected"


def _read_back_packet_retirement_kind(packet: dict[str, Any]) -> str | None:
    """Classify non-actionable read-back incidents in strict retirement order."""

    if _is_transient_read_back_packet(packet):
        return "transient"
    if _is_empty_query_read_back_packet(packet):
        return "empty_query"
    if _is_exhausted_query_hint_read_back_packet(packet):
        return "exhausted_query_hint"
    if _is_unverifiable_query_hint_read_back_packet(packet):
        return "unverifiable_query_hint"
    return None


def _frontier_attempt_outcome(
    frontier_result: dict[str, Any],
    *,
    attempt: int,
    max_attempts: int,
    backoff_base_seconds: int,
    now: datetime | None = None,
) -> tuple[str, str | None]:
    """Derive terminal status and retry time without mutating packet state."""

    final_status = _frontier_final_status(frontier_result)
    if frontier_result.get("execution_started") is True and final_status not in {
        "frontier_approved",
        "human_required",
    }:
        final_status = "frontier_quarantined"
    next_attempt_at = None
    if final_status in {"frontier_retry", *PENDING_REVIEW_STATUSES}:
        if attempt >= max(1, max_attempts):
            final_status = "frontier_quarantined"
        else:
            delay = max(0, backoff_base_seconds) * (2 ** max(0, attempt - 1))
            next_attempt_at = ((now or datetime.now()) + timedelta(seconds=delay)).isoformat(
                timespec="seconds"
            )
    return final_status, next_attempt_at


def _finalize_frontier_attempt(
    *,
    packet_path: Path,
    packet: dict[str, Any],
    frontier_result: dict[str, Any],
    attempt: int,
    max_attempts: int,
    backoff_base_seconds: int,
    requires_frontier_action: bool,
    decision: LocalRepairDecision | None,
    local_decision: dict[str, Any],
    dry_run: bool,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Persist the terminal effect and operator record of one frontier attempt."""

    final_status, next_attempt_at = _frontier_attempt_outcome(
        frontier_result,
        attempt=attempt,
        max_attempts=max_attempts,
        backoff_base_seconds=backoff_base_seconds,
    )

    approved_action: dict[str, Any] | None = None
    action_error: str | None = None
    if final_status == "frontier_approved" and requires_frontier_action:
        approved_decision = decision or _local_decision_from_payload(local_decision)
        if approved_decision is None:
            action_error = "frontier-approved local action artifact is invalid"
        else:
            try:
                with _local_decision_effect(approved_decision):
                    _raise_if_packet_cancelled(packet_path, packet)
                    approved_action = apply_local_decision(
                        packet,
                        approved_decision,
                        dry_run=False,
                    )
                    action_path = _save_action(
                        packet_path, approved_action, applied=True
                    )
                    result["action"] = approved_action
                    result["applied_action_path"] = str(action_path)
                    _append_registry(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "failure_id": packet.get("failure_id"),
                            "raw_file": packet.get("raw_file"),
                            "failure_class": packet.get("failure_class"),
                            "fingerprint": packet.get("fingerprint"),
                            "resolution": "frontier_approved_local_action",
                            "decision": local_decision,
                            "frontier": frontier_result,
                            "action": approved_action,
                        }
                    )
            except _PacketCancellationRequested:
                raise
            except Exception as exc:
                action_error = f"frontier-approved local action failed: {exc}"
                _save_action(
                    packet_path,
                    {
                        "action": approved_decision.action,
                        "error": action_error,
                        "decision": local_decision,
                        "frontier": frontier_result,
                    },
                    applied=False,
                )
        if action_error is not None:
            final_status = "frontier_quarantined"
            next_attempt_at = None
            result["action_error"] = action_error

    human_notification = None
    pending_review_path = None
    if final_status == "human_required" and not dry_run:
        human_notification = maybe_notify_human_required(packet, frontier_result)
    if final_status in PENDING_REVIEW_STATUSES:
        pending_review_path = _save_pending_frontier_review(
            packet_path,
            packet,
            local_decision,
            frontier_result,
            status=final_status,
        )
    _update_packet(
        packet_path,
        packet,
        status=final_status,
        frontier_result=frontier_result,
        human_notification=human_notification,
        pending_frontier_review_path=str(pending_review_path)
        if pending_review_path
        else None,
        next_attempt_at=next_attempt_at,
        approved_action=approved_action,
        action_error=action_error,
        human_required_at=(
            datetime.now().isoformat(timespec="seconds")
            if final_status == "human_required"
            else packet.get("human_required_at")
        ),
        quarantined_at=(
            datetime.now().isoformat(timespec="seconds")
            if final_status == "frontier_quarantined"
            else packet.get("quarantined_at")
        ),
    )
    _append_registry(
        {
            "timestamp": datetime.now().isoformat(),
            "failure_id": packet.get("failure_id"),
            "raw_file": packet.get("raw_file"),
            "failure_class": packet.get("failure_class"),
            "fingerprint": packet.get("fingerprint"),
            "resolution": "frontier",
            "decision": local_decision,
            "frontier": frontier_result,
            "human_notification": human_notification,
            "pending_frontier_review_path": str(pending_review_path)
            if pending_review_path
            else None,
        }
    )
    event_level = (
        "success"
        if final_status == "frontier_approved"
        else "error"
        if final_status in HUMAN_REQUIRED_STATUSES
        else "warn"
    )
    event_message = (
        f"self-heal | human required for {packet.get('raw_file')}"
        if final_status == "human_required"
        else f"self-heal | frontier {frontier_result.get('decision')} for {packet.get('raw_file')}"
    )
    runtime_status.safe_append_event(
        event_level,
        event_message,
        source="self-heal",
        packet=str(packet_path),
        frontier_status=final_status,
        human_required=final_status == "human_required",
    )
    result["status"] = final_status
    result["frontier_result"] = frontier_result
    result["human_notification"] = human_notification
    result["pending_frontier_review_path"] = (
        str(pending_review_path) if pending_review_path else None
    )
    return result


def _handle_packet_unlocked(
    packet_path: Path,
    *,
    use_qwen: bool = True,
    enable_frontier: bool = False,
    execute_frontier_patch: bool = True,
    dry_run: bool = False,
    max_attempts: int = 3,
    backoff_base_seconds: int = 6 * 60 * 60,
    frontier_budget: Any | None = None,
) -> dict[str, Any]:
    packet = _read_json(packet_path)
    cancellation = _read_packet_cancellation(packet_path, packet)
    if cancellation is not None and dry_run:
        return {
            "packet": str(packet_path),
            "failure_id": packet.get("failure_id"),
            "status": "dry_run",
            "projected_status": cancellation.get("status"),
            "reason": cancellation.get("reason"),
            "superseded_by_packet": cancellation.get("superseded_by_packet"),
            "would_cancel": True,
        }
    if cancellation is not None:
        _raise_if_packet_cancelled(packet_path, packet)
    human_boundary = _human_boundary_result(
        packet_path,
        packet,
        dry_run=dry_run,
    )
    if human_boundary is not None:
        return human_boundary
    repair_evidence, frontier_ineligible_reason = _frontier_eligibility(packet)
    if (
        repair_evidence is not None
        and repair_evidence.notes.get("producer")
        == "trusted_operational_failure_supervisor"
    ):
        try:
            from chronovisor.ingest.system_incident_supervisor import (
                validate_operational_incident_packet,
            )

            validate_operational_incident_packet(packet_path)
        except Exception as exc:
            validation_error = (
                f"trusted operational incident read-back failed: "
                f"{exc.__class__.__name__}: {exc}"
            )
            if dry_run:
                return {
                    "packet": str(packet_path),
                    "failure_id": packet.get("failure_id"),
                    "status": "dry_run",
                    "projected_status": "frontier_quarantined",
                    "reason": "operational_incident_evidence_invalid",
                    "frontier_eligibility_error": validation_error,
                }
            _update_packet(
                packet_path,
                packet,
                status="frontier_quarantined",
                next_attempt_at=None,
                frontier_eligibility_error=validation_error,
                quarantined_at=datetime.now().isoformat(timespec="seconds"),
            )
            return {
                "packet": str(packet_path),
                "failure_id": packet.get("failure_id"),
                "status": "frontier_quarantined",
                "reason": "operational_incident_evidence_invalid",
                "frontier_eligibility_error": validation_error,
            }
    read_back_retirement_kind = _read_back_packet_retirement_kind(packet)
    if read_back_retirement_kind is not None:
        retirement_handlers = {
            "transient": _retire_transient_read_back_packet,
            "empty_query": _retire_empty_query_read_back_packet,
            "exhausted_query_hint": _retire_exhausted_query_hint_read_back_packet,
            "unverifiable_query_hint": _retire_unverifiable_query_hint_read_back_packet,
        }
        return retirement_handlers[read_back_retirement_kind](
            packet_path,
            packet,
            dry_run=dry_run,
        )
    if packet.get("status") == "local_quarantined":
        exact_hold, stale_hold = _current_local_semantic_hold(
            packet,
            lock_authority=not dry_run,
        )
        if exact_hold is not None:
            active_hold = _packet_semantic_hold(packet)
            if not dry_run and active_hold != exact_hold:
                _update_packet(
                    packet_path,
                    packet,
                    semantic_hold=exact_hold,
                    semantic_hold_history=_semantic_hold_history_with(
                        packet,
                        exact_hold,
                    ),
                    invalidated_semantic_hold=None,
                    terminal_reason="semantic_no_quorum",
                    last_failure_class=LOCAL_SEMANTIC_NO_QUORUM,
                    next_attempt_at=None,
                )
            return {
                "packet": str(packet_path),
                "failure_id": packet.get("failure_id"),
                "status": "dry_run" if dry_run else "local_quarantined",
                "projected_status": "local_quarantined",
                "cached": True,
                "terminal_reason": "semantic_no_quorum",
                "semantic_deferred": True,
                "semantic_hold": exact_hold,
            }
        if stale_hold is not None:
            if dry_run:
                return {
                    "packet": str(packet_path),
                    "failure_id": packet.get("failure_id"),
                    "status": "dry_run",
                    "projected_status": "pending_local_repair",
                    "would_resume_semantic_hold": True,
                }
            _update_packet(
                packet_path,
                packet,
                status="pending_local_repair",
                semantic_hold=None,
                semantic_hold_history=_semantic_hold_history_with(
                    packet,
                    stale_hold,
                ),
                invalidated_semantic_hold=stale_hold,
                semantic_hold_invalidated_at=datetime.now().isoformat(
                    timespec="seconds"
                ),
                next_attempt_at=None,
            )
    restored_semantic_hold = _restore_invalidated_local_semantic_hold(
        packet_path,
        packet,
        dry_run=dry_run,
    )
    if restored_semantic_hold is not None:
        return restored_semantic_hold
    resume_kind = _resume_terminal_packet(
        packet_path,
        packet,
        dry_run=dry_run,
    )
    if resume_kind is not None and dry_run:
        return {
            "packet": str(packet_path),
            "failure_id": packet.get("failure_id"),
            "status": "dry_run",
            "would_resume_terminal": True,
            "would_reclassify_human_boundary": resume_kind == "legacy_nonhuman",
            "terminal_resume_kind": resume_kind,
            "projected_status": "frontier_retry",
        }
    if resume_kind == "legacy_nonhuman":
        # Compatibility field retained for existing operators/tests.
        packet["human_boundary_reclassified_at"] = datetime.now().isoformat(
            timespec="seconds"
        )
        if not dry_run:
            _write_json(packet_path, packet)
    if packet.get("status") == "human_required" and not is_human_required_result(
        packet.get("frontier_result")
    ):
        if dry_run:
            return {
                "packet": str(packet_path),
                "failure_id": packet.get("failure_id"),
                "status": "dry_run",
                "would_reclassify_human_boundary": True,
                "projected_status": "frontier_retry",
            }
        _update_packet(
            packet_path,
            packet,
            status="frontier_retry",
            human_boundary_reclassified_at=datetime.now().isoformat(timespec="seconds"),
            next_attempt_at=None,
        )
    frontier_only = packet.get(
        "status"
    ) in FRONTIER_ONLY_STATUSES and not _frontier_nonconvergence_should_reenter_local(
        packet
    )
    decision: LocalRepairDecision | None = None
    persisted_decision = packet.get("local_decision")
    local_decision = (
        dict(persisted_decision) if isinstance(persisted_decision, dict) else None
    )
    prior_frontier_attempts = int(
        packet.get("frontier_attempts")
        if packet.get("frontier_attempts") is not None
        else packet.get("self_heal_attempts") or 0
    )
    max_frontier_attempts = max(1, max_attempts)
    will_apply_local = False
    requires_frontier_action = False
    mutation_reserved = False

    if frontier_only:
        if dry_run:
            return {
                "packet": str(packet_path),
                "failure_id": packet.get("failure_id"),
                "status": "dry_run",
                "local_decision": local_decision,
                "frontier_only": True,
            }
    else:
        if not dry_run and frontier_budget is not None:
            allowed, reason = frontier_budget.consume("local")
            if not allowed:
                return _budget_deferred_result(
                    packet_path,
                    packet,
                    kind="local",
                    reason=reason,
                )
        if not dry_run:
            _raise_if_packet_cancelled(packet_path, packet)
        # A dry-run is a byte-for-byte read-only projection, not an inference
        # rehearsal.  Local structured sessions write active/audit/replay
        # artifacts and change model residency even when the packet itself is
        # untouched, so only deterministic repair logic may run here.  Exact
        # persisted semantic holds were already handled above without a model.
        decision = propose_repair(
            packet,
            use_qwen=False if dry_run else use_qwen,
        )
        if not dry_run:
            _raise_if_packet_cancelled(packet_path, packet)
        local_decision = decision.to_dict()
        if dry_run:
            deterministic_terminal = (
                decision.status == "resolved"
                and decision.action
                in {
                    "resolve_update_target",
                    "retry_raw",
                    "quarantine_raw",
                }
            )
            semantic_review_skipped = bool(use_qwen and not deterministic_terminal)
            return {
                "packet": str(packet_path),
                "failure_id": packet.get("failure_id"),
                "status": "dry_run",
                "local_decision": local_decision,
                **(
                    {
                        "projected_status": "local_quarantined",
                        "terminal_reason": "semantic_no_quorum",
                    }
                    if decision.source == "semantic_hold"
                    else {}
                ),
                **(
                    {
                        "projected_status": "local_review_required",
                        "model_review_skipped": True,
                    }
                    if semantic_review_skipped
                    else {}
                ),
            }

        semantic_hold = persisted_semantic_no_quorum_hold(
            local_decision,
            "local_repair",
            epoch=semantic_hold_epoch(packet),
            authority=decision.authority,
        )
        if decision.source == "semantic_hold" and semantic_hold is not None:
            with decision_authority_lock():
                current_authority, authority_error = current_semantic_authority(
                    "local_repair"
                )
                if authority_error is None and current_authority is None:
                    authority_error = "local repair decision authority is unavailable"
                if authority_error is None:
                    authority_error = semantic_no_quorum_hold_error(
                        semantic_hold,
                        "local_repair",
                        epoch=semantic_hold_epoch(packet),
                        authority=current_authority,
                    )
                if authority_error is None:
                    _update_packet(
                        packet_path,
                        packet,
                        status="local_quarantined",
                        local_decision=local_decision,
                        semantic_hold=semantic_hold,
                        semantic_hold_history=_semantic_hold_history_with(
                            packet,
                            semantic_hold,
                        ),
                        invalidated_semantic_hold=None,
                        terminal_reason="semantic_no_quorum",
                        last_failure_class=LOCAL_SEMANTIC_NO_QUORUM,
                        next_attempt_at=None,
                        quarantined_at=datetime.now().isoformat(timespec="seconds"),
                    )
                    return {
                        "packet": str(packet_path),
                        "failure_id": packet.get("failure_id"),
                        "status": "local_quarantined",
                        "terminal_reason": "semantic_no_quorum",
                        "semantic_deferred": True,
                        "local_decision": local_decision,
                    }
            decision = replace(
                decision,
                source="local_deferred",
                reason=f"local semantic hold authority changed: {authority_error}",
                semantic_hold=None,
            )
            local_decision = decision.to_dict()

        if decision.source == "local_deferred":
            delay = max(0, backoff_base_seconds)
            _update_packet(
                packet_path,
                packet,
                status="pending_local_repair",
                local_decision=local_decision,
                last_failure_class="decision_authority_changed",
                next_attempt_at=(datetime.now() + timedelta(seconds=delay)).isoformat(
                    timespec="seconds"
                ),
            )
            return {
                "packet": str(packet_path),
                "failure_id": packet.get("failure_id"),
                "status": "pending_local_repair",
                "reason": decision.reason,
                "local_decision": local_decision,
            }

        # Validated deterministic repairs and two-vote local consensus repairs
        # complete in the local data plane.  A legacy one-model generator is a
        # test/compatibility seam and is never mutation authority.
        will_apply_local = (
            not _is_operational_source_packet(packet)
            and decision.status == "resolved"
            and decision.action
            in {"resolve_update_target", "retry_raw", "quarantine_raw"}
            and decision.source in {"deterministic", "local_consensus"}
        )
        requires_frontier_action = decision.status == "escalate" or decision.action in {
            "escalate_to_frontier",
            "propose_prompt_fix",
            "propose_test_case",
        }
        if will_apply_local and frontier_budget is not None:
            allowed, reason = frontier_budget.consume("mutation")
            if not allowed:
                return _budget_deferred_result(
                    packet_path,
                    packet,
                    kind="mutation",
                    reason=reason,
                    local_decision=local_decision,
                )
            mutation_reserved = True

    if frontier_only:
        requires_frontier_action = bool(
            isinstance(local_decision, dict)
            and (
                local_decision.get("status") == "escalate"
                or local_decision.get("action")
                in {"escalate_to_frontier", "propose_prompt_fix", "propose_test_case"}
            )
        )
    routes_directly_to_frontier = frontier_only or not will_apply_local
    if (
        routes_directly_to_frontier
        and enable_frontier
        and repair_evidence is not None
        and prior_frontier_attempts < max_frontier_attempts
        and frontier_budget is not None
    ):
        needs_mutation = (
            (execute_frontier_patch or requires_frontier_action)
            and not dry_run
            and not mutation_reserved
        )
        if needs_mutation:
            mutation_allowed, mutation_reason = frontier_budget.can_consume("mutation")
            if not mutation_allowed:
                return _budget_deferred_result(
                    packet_path,
                    packet,
                    kind="mutation",
                    reason=mutation_reason,
                    local_decision=local_decision,
                )
            frontier_budget.consume("mutation")
            mutation_reserved = True

    lease_owner = uuid.uuid4().hex
    result: dict[str, Any] = {
        "packet": str(packet_path),
        "failure_id": packet.get("failure_id"),
        "local_decision": local_decision,
    }

    if not frontier_only:
        assert decision is not None
        local_attempt = int(packet.get("local_repair_attempts") or 0) + 1
        operational_attempt_evidence = _next_operational_attempt_evidence(
            packet,
            local_decision or {},
            local_attempt,
        )
        operational_updates = (
            {
                "operational_local_repair_evidence": operational_attempt_evidence,
            }
            if operational_attempt_evidence is not None
            else {}
        )
        try:
            with _local_decision_effect(decision):
                _update_packet(
                    packet_path,
                    packet,
                    status="local_repairing",
                    local_repair_attempts=local_attempt,
                    local_decision=local_decision,
                    last_attempt_at=datetime.now().isoformat(timespec="seconds"),
                    next_attempt_at=None,
                    **_lease_updates(lease_owner),
                    **operational_updates,
                )
                decision_path = _save_local_decision(packet_path, decision)
                result["local_decision_path"] = str(decision_path)
            if will_apply_local:
                with _local_decision_effect(decision):
                    _raise_if_packet_cancelled(packet_path, packet)
                    action = apply_local_decision(packet, decision, dry_run=False)
                    action_path = _save_action(packet_path, action, applied=True)
                    _update_packet(
                        packet_path,
                        packet,
                        status="local_repair_applied",
                        local_decision=local_decision,
                        applied_action_path=str(action_path),
                    )
                    _append_registry(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "failure_id": packet.get("failure_id"),
                            "raw_file": packet.get("raw_file"),
                            "failure_class": packet.get("failure_class"),
                            "fingerprint": packet.get("fingerprint"),
                            "resolution": "local",
                            "decision": local_decision,
                            "action": action,
                        }
                    )
                runtime_status.safe_append_event(
                    "success",
                    f"self-heal | local repair applied for {packet.get('raw_file')}",
                    source="self-heal",
                    packet=str(packet_path),
                    action=decision.action,
                )
                result["status"] = "local_repair_applied"
                result["action"] = action
                return result
        except _PacketCancellationRequested:
            raise
        except Exception as exc:
            action = {
                "action": decision.action,
                "error": str(exc),
                "decision": local_decision,
            }
            _save_action(packet_path, action, applied=False)
            _update_packet(
                packet_path,
                packet,
                status="local_repair_failed",
                local_decision=local_decision,
                local_error=str(exc),
            )
            result["local_error"] = str(exc)

    effect_decision = decision or _local_decision_from_payload(local_decision)

    if not enable_frontier or repair_evidence is None:
        local_attempts = int(packet.get("local_repair_attempts") or 0)
        terminal = local_attempts >= max(1, max_attempts)
        delay = max(0, backoff_base_seconds) * (2 ** max(0, local_attempts - 1))
        status = "local_quarantined" if terminal else "pending_local_repair"
        local_failure_reason = (
            "repair_plane_disabled"
            if not enable_frontier
            else "frontier_repair_not_eligible"
        )
        try:
            with _local_decision_effect(effect_decision):
                _update_packet(
                    packet_path,
                    packet,
                    status=status,
                    local_decision=local_decision,
                    frontier_queue_path=None,
                    next_attempt_at=(
                        None
                        if terminal
                        else (datetime.now() + timedelta(seconds=delay)).isoformat(
                            timespec="seconds"
                        )
                    ),
                    quarantined_at=(
                        datetime.now().isoformat(timespec="seconds")
                        if terminal
                        else packet.get("quarantined_at")
                    ),
                    local_failure_reason=local_failure_reason,
                    frontier_eligibility_error=(
                        frontier_ineligible_reason if repair_evidence is None else None
                    ),
                )
        except _PacketCancellationRequested:
            raise
        except RuntimeError as exc:
            _update_packet(
                packet_path,
                packet,
                status="local_repair_failed",
                local_decision=local_decision,
                local_error=str(exc),
            )
            result["status"] = "local_repair_failed"
            result["local_error"] = str(exc)
            return result
        if terminal:
            _raise_if_packet_cancelled(packet_path, packet)
            incident_result = _promote_operational_source_packet(packet_path, packet)
        else:
            incident_result = None
        result["status"] = status
        result["reason"] = local_failure_reason
        if incident_result is not None:
            result["system_incident"] = incident_result
        if repair_evidence is None:
            result["frontier_eligibility_error"] = frontier_ineligible_reason
        return result

    try:
        with _local_decision_effect(effect_decision):
            _raise_if_packet_cancelled(packet_path, packet)
            queue_path = _queue_frontier(packet_path, packet, local_decision)
    except _PacketCancellationRequested:
        raise
    except RuntimeError as exc:
        _update_packet(
            packet_path,
            packet,
            status="local_repair_failed",
            local_decision=local_decision,
            local_error=str(exc),
        )
        result["status"] = "local_repair_failed"
        result["local_error"] = str(exc)
        return result
    result["frontier_queue_path"] = str(queue_path)

    if prior_frontier_attempts >= max_frontier_attempts:
        try:
            with _local_decision_effect(effect_decision):
                _update_packet(
                    packet_path,
                    packet,
                    status="frontier_quarantined",
                    local_decision=local_decision,
                    frontier_queue_path=str(queue_path),
                    next_attempt_at=None,
                    frontier_error="frontier attempt limit reached before execution",
                    quarantined_at=datetime.now().isoformat(timespec="seconds"),
                )
        except _PacketCancellationRequested:
            raise
        except RuntimeError as exc:
            result["status"] = "local_repair_failed"
            result["local_error"] = str(exc)
            return result
        result["status"] = "frontier_quarantined"
        result["reason"] = "frontier_attempt_limit_reached"
        return result

    if frontier_budget is not None:
        needs_mutation = (
            (execute_frontier_patch or requires_frontier_action)
            and not dry_run
            and not mutation_reserved
        )
        mutation_allowed, mutation_reason = (
            frontier_budget.can_consume("mutation") if needs_mutation else (True, "ok")
        )
        if not mutation_allowed:
            return _budget_deferred_result(
                packet_path,
                packet,
                kind="mutation",
                reason=mutation_reason,
                local_decision=local_decision,
            )
        if needs_mutation:
            frontier_budget.consume("mutation")
            mutation_reserved = True

    attempt = prior_frontier_attempts + 1
    try:
        with _local_decision_effect(effect_decision):
            _raise_if_packet_cancelled(packet_path, packet)
            _update_packet(
                packet_path,
                packet,
                status="frontier_running",
                frontier_attempts=attempt,
                # Keep the legacy aggregate field in sync for old dashboards and
                # packets, but never increment it before a real frontier execution.
                self_heal_attempts=attempt,
                local_decision=local_decision,
                frontier_queue_path=str(queue_path),
                **_lease_updates(lease_owner),
            )
    except _PacketCancellationRequested:
        raise
    except RuntimeError as exc:
        _update_packet(
            packet_path,
            packet,
            status="local_repair_failed",
            local_decision=local_decision,
            local_error=str(exc),
        )
        result["status"] = "local_repair_failed"
        result["local_error"] = str(exc)
        return result
    try:
        _raise_if_packet_cancelled(packet_path, packet)
        frontier_result = _run_frontier(
            packet_path,
            packet,
            local_decision,
            evidence=repair_evidence,
            execute_patch=(
                execute_frontier_patch and not requires_frontier_action and not dry_run
            ),
        )
    except _PacketCancellationRequested:
        raise
    except Exception as exc:
        # Once control enters the guarded execution call we cannot safely
        # prove that a child was never spawned.  Fail terminally so one packet
        # can never create a second subscription session.
        final_status = "frontier_quarantined"
        next_attempt_at = None
        frontier_error = {
            "exception_type": exc.__class__.__name__,
            "message": str(exc),
        }
        _update_packet(
            packet_path,
            packet,
            status=final_status,
            local_decision=local_decision,
            frontier_queue_path=str(queue_path),
            frontier_error=frontier_error,
            next_attempt_at=next_attempt_at,
            quarantined_at=(
                datetime.now().isoformat(timespec="seconds")
                if final_status == "frontier_quarantined"
                else packet.get("quarantined_at")
            ),
        )
        _append_registry(
            {
                "timestamp": datetime.now().isoformat(),
                "failure_id": packet.get("failure_id"),
                "raw_file": packet.get("raw_file"),
                "failure_class": packet.get("failure_class"),
                "fingerprint": packet.get("fingerprint"),
                "resolution": "frontier_error",
                "decision": local_decision,
                "frontier_error": frontier_error,
                "status": final_status,
            }
        )
        runtime_status.safe_append_event(
            "warn",
            f"self-heal | frontier exception for {packet.get('raw_file')}",
            source="self-heal",
            packet=str(packet_path),
            frontier_status=final_status,
            frontier_error=frontier_error,
        )
        result["status"] = final_status
        result["frontier_error"] = frontier_error
        return result
    _raise_if_packet_cancelled(packet_path, packet)
    if frontier_result.get("execution_started") is False:
        # Guard denial, guard failure, and preflight failure happen before the
        # one permitted Codex process begins.  They are durable deferrals, not
        # repair attempts, and therefore must not consume the packet limit.
        attempt = prior_frontier_attempts
        failure = frontier_result.get("frontier_failure")
        failure = failure if isinstance(failure, dict) else {}
        rescue = frontier_result.get("rescue_attempt")
        rescue = rescue if isinstance(rescue, dict) else {}
        guard_reason = str(rescue.get("guard_reason") or "")
        if guard_reason == "incident_already_started":
            deferred_status = "frontier_quarantined"
            deferred_next_attempt = None
        else:
            deferred_status = "repair_deferred"
            retry_at = _parse_iso(rescue.get("retry_at"))
            deferred_next_attempt = (
                retry_at.isoformat(timespec="seconds")
                if retry_at is not None
                else (
                    datetime.now() + timedelta(seconds=max(60, backoff_base_seconds))
                ).isoformat(timespec="seconds")
            )
        _update_packet(
            packet_path,
            packet,
            status=deferred_status,
            frontier_attempts=prior_frontier_attempts,
            self_heal_attempts=prior_frontier_attempts,
            frontier_result=frontier_result,
            next_attempt_at=deferred_next_attempt,
        )
        result["status"] = deferred_status
        result["frontier_result"] = frontier_result
        result["next_attempt_at"] = deferred_next_attempt
        result["reason"] = guard_reason or failure.get("failure_class")
        return result
    return _finalize_frontier_attempt(
        packet_path=packet_path,
        packet=packet,
        frontier_result=frontier_result,
        attempt=attempt,
        max_attempts=max_attempts,
        backoff_base_seconds=backoff_base_seconds,
        requires_frontier_action=requires_frontier_action,
        decision=decision,
        local_decision=local_decision,
        dry_run=dry_run,
        result=result,
    )


def handle_packet(
    packet_path: Path,
    *,
    use_qwen: bool = True,
    enable_frontier: bool = False,
    execute_frontier_patch: bool = True,
    dry_run: bool = False,
    max_attempts: int = 3,
    backoff_base_seconds: int = 6 * 60 * 60,
    frontier_budget: Any | None = None,
) -> dict[str, Any]:
    """Handle one packet with process-safe single-flight execution.

    Dry runs intentionally avoid even creating the sidecar lock file, keeping
    the established byte-for-byte read-only contract. ``frontier_budget``
    retains its public name for compatibility but is the shared cycle budget
    for local, frontier, and mutation work.
    """

    kwargs = {
        "use_qwen": use_qwen,
        "enable_frontier": enable_frontier,
        "execute_frontier_patch": execute_frontier_patch,
        "dry_run": dry_run,
        "max_attempts": max_attempts,
        "backoff_base_seconds": backoff_base_seconds,
        "frontier_budget": frontier_budget,
    }
    if dry_run:
        return _handle_packet_unlocked(packet_path, **kwargs)
    with _packet_lock(packet_path) as acquired:
        if not acquired:
            return {
                "packet": str(packet_path),
                "status": "busy",
                "reason": "packet_already_running",
            }
        try:
            current = _read_json(packet_path)
            _raise_if_packet_cancelled(packet_path, current)
            current_status = current.get("status")
            exact_semantic_hold: dict[str, Any] | None = None
            stale_semantic_hold: dict[str, Any] | None = None
            if current_status == "local_quarantined":
                exact_semantic_hold, stale_semantic_hold = _current_local_semantic_hold(
                    current
                )
                active_semantic_hold = _packet_semantic_hold(current)
                if (
                    exact_semantic_hold is not None
                    and active_semantic_hold != exact_semantic_hold
                    and not dry_run
                ):
                    _update_packet(
                        packet_path,
                        current,
                        semantic_hold=exact_semantic_hold,
                        semantic_hold_history=_semantic_hold_history_with(
                            current,
                            exact_semantic_hold,
                        ),
                        invalidated_semantic_hold=None,
                        terminal_reason="semantic_no_quorum",
                        last_failure_class=LOCAL_SEMANTIC_NO_QUORUM,
                        next_attempt_at=None,
                    )
                if stale_semantic_hold is not None:
                    _update_packet(
                        packet_path,
                        current,
                        status="pending_local_repair",
                        semantic_hold=None,
                        semantic_hold_history=_semantic_hold_history_with(
                            current,
                            stale_semantic_hold,
                        ),
                        invalidated_semantic_hold=stale_semantic_hold,
                        semantic_hold_invalidated_at=datetime.now().isoformat(
                            timespec="seconds"
                        ),
                        next_attempt_at=None,
                    )
                    current_status = "pending_local_repair"
            legacy_non_external_human = (
                current_status == "human_required"
                and not is_human_required_result(current.get("frontier_result"))
            )
            resumable_terminal = _terminal_resume_kind(current) is not None
            if current_status in RUNNING_STATUSES and not _running_lease_expired(
                current
            ):
                return {
                    "packet": str(packet_path),
                    "failure_id": current.get("failure_id"),
                    "status": "busy",
                    "reason": "running_lease_active",
                }
            if (
                current_status
                and current_status not in SELF_HEAL_STATUSES
                and current_status not in RUNNING_STATUSES
                and not legacy_non_external_human
                and not resumable_terminal
            ):
                result = {
                    "packet": str(packet_path),
                    "failure_id": current.get("failure_id"),
                    "status": current_status,
                    "cached": True,
                }
                if exact_semantic_hold is not None:
                    result.update(
                        {
                            "terminal_reason": "semantic_no_quorum",
                            "semantic_deferred": True,
                            "semantic_hold": exact_semantic_hold,
                        }
                    )
                source_sync = _sync_system_incident_outcome(packet_path)
                if source_sync is not None:
                    result["operational_source_sync"] = source_sync
                return result
            # The packet is read inside the lock by the implementation.  This is
            # the CAS boundary that prevents a stale pre-lock snapshot from being
            # applied after another worker completes.
            result = _handle_packet_unlocked(packet_path, **kwargs)
        except _PacketCancellationRequested as exc:
            return exc.result
        source_sync = _sync_system_incident_outcome(packet_path)
        if source_sync is not None:
            result["operational_source_sync"] = source_sync
        return result


def run_pending(
    *,
    max_packets: int = 3,
    use_qwen: bool = True,
    enable_frontier: bool = False,
    execute_frontier_patch: bool = True,
    dry_run: bool = False,
    max_attempts: int = 3,
    backoff_base_seconds: int = 6 * 60 * 60,
    frontier_budget: Any | None = None,
) -> dict[str, Any]:
    packets = pending_packets(lock_authority=not dry_run)[:max_packets]
    results: list[dict[str, Any]] = []
    for packet in packets:
        try:
            result = handle_packet(
                packet,
                use_qwen=use_qwen,
                enable_frontier=enable_frontier,
                execute_frontier_patch=execute_frontier_patch,
                dry_run=dry_run,
                max_attempts=max_attempts,
                backoff_base_seconds=backoff_base_seconds,
                frontier_budget=frontier_budget,
            )
        except Exception as exc:
            # One corrupt or externally-failing packet must not abort the
            # bounded drain of unrelated packets.
            result = {
                "packet": str(packet),
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        results.append(result)
    return {
        "status": "ok",
        "packets_seen": len(packets),
        "results": results,
    }


def run_auto_apply_error_self_heal(
    *,
    threshold: int = 3,
    max_packets: int = 3,
    use_qwen: bool = True,
    enable_frontier: bool = False,
    execute_frontier_patch: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    from chronovisor.ingest.auto_apply_error_supervisor import (
        pending_auto_apply_error_packets,
        supervise_auto_apply_log,
    )

    supervision = supervise_auto_apply_log(
        threshold=threshold,
        start_background=False,
        dry_run=dry_run,
    )
    created = [
        Path(path)
        for path in supervision.get("packets_created", [])
        if isinstance(path, str)
    ]
    packets = created or pending_auto_apply_error_packets()
    packets = packets[:max_packets]
    results: list[dict[str, Any]] = []
    for packet in packets:
        try:
            result = handle_packet(
                packet,
                use_qwen=use_qwen,
                enable_frontier=enable_frontier,
                execute_frontier_patch=execute_frontier_patch,
                dry_run=dry_run,
            )
        except Exception as exc:
            result = {
                "packet": str(packet),
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        results.append(result)
    return {
        "status": "ok",
        "supervision": supervision,
        "packets_seen": len(packets),
        "results": results,
    }




def enqueue_system_code_repair(packet_path: Path) -> dict[str, Any]:
    """Durably enqueue one trusted repair-plane packet without spawning.

    Only :mod:`chronovisor.ingest.system_incident_supervisor` should call this
    helper.  The converge worker owns execution through the background-job
    ledger; watchdog and hook processes must never detach a worker directly.
    """

    from chronovisor.core.background_jobs import enqueue_job

    resolved = packet_path.expanduser().resolve(strict=False)
    packet = _read_json(resolved)
    cancellation = _read_packet_cancellation(resolved, packet)
    if cancellation is not None:
        return {
            "job_id": None,
            "status": cancellation.get("status"),
            "enqueued": False,
            "coalesced": False,
            "cancelled": True,
            "cancellation_reason": cancellation.get("reason"),
        }
    _repair_incident_evidence(packet)
    payload = packet.get("repair_evidence")
    if (
        not isinstance(payload, dict)
        or _trusted_repair_packet_job_id(packet, payload) is None
    ):
        raise EvidenceValidationError(
            ["only allowlisted trusted system incidents may be durably enqueued"]
        )
    notes = payload.get("notes")
    if (
        isinstance(notes, dict)
        and notes.get("producer") == "trusted_operational_failure_supervisor"
    ):
        from chronovisor.ingest.system_incident_supervisor import (
            validate_operational_incident_packet,
        )

        validate_operational_incident_packet(resolved)
    return enqueue_job(
        name="system-code-repair",
        module="chronovisor.ops.self_heal",
        args=["--packet", str(resolved), "--enable-frontier-repair"],
        env={},
        stdin_text="",
    )


def _promote_due_operational_sources(*, limit: int) -> list[dict[str, Any]]:
    promoted: list[dict[str, Any]] = []
    if limit <= 0 or not _packet_dir().exists():
        return promoted
    for path in sorted(_packet_dir().glob("*.json")):
        if len(promoted) >= limit:
            break
        try:
            packet = _read_json(path)
        except Exception:
            continue
        if packet.get(
            "status"
        ) != "local_quarantined" or not _is_operational_source_packet(packet):
            continue
        with _packet_lock(path) as acquired:
            if not acquired:
                continue
            current = _read_json(path)
            if current.get("status") != "local_quarantined":
                continue
            result = _promote_operational_source_packet(path, current)
            if result is not None and result.get("packet_path"):
                promoted.append({"packet": str(path), **result})
    return promoted


def _sync_completed_operational_incidents(*, limit: int) -> list[dict[str, Any]]:
    synced: list[dict[str, Any]] = []
    if limit <= 0 or not _packet_dir().exists():
        return synced
    for path in sorted(_packet_dir().glob("system-operational-*.json")):
        if len(synced) >= limit:
            break
        result = _sync_system_incident_outcome(path)
        if result is not None:
            synced.append({"packet": str(path), **result})
    return synced


def enqueue_due_system_repairs(*, limit: int = 2) -> dict[str, Any]:
    """Requeue guard-deferred incidents only when their durable time is due."""

    bounded_limit = max(0, limit)
    operational_promotions = _promote_due_operational_sources(limit=bounded_limit)
    operational_source_sync = _sync_completed_operational_incidents(limit=bounded_limit)
    queued: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    due_paths: list[Path] = []
    for path in pending_packets():
        try:
            if _read_json(path).get("status") == "repair_deferred":
                due_paths.append(path)
        except Exception:
            continue
        if len(due_paths) >= bounded_limit:
            break
    for path in due_paths:
        try:
            _read_json(path)
            job = enqueue_system_code_repair(path)
            queued.append(
                {
                    "packet": str(path),
                    "job_id": job.get("job_id"),
                    "enqueued": bool(job.get("enqueued")),
                    "coalesced": bool(job.get("coalesced")),
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "packet": str(path),
                    "error_type": exc.__class__.__name__,
                }
            )
    return {
        "status": "ok" if not errors else "attention",
        "queued": queued,
        "errors": errors,
        "operational_promotions": operational_promotions,
        "operational_source_sync": operational_source_sync,
    }


def drill_packet() -> dict[str, Any]:
    return {
        "failure_id": "drill-update-target-not-found",
        "created_at": datetime.now().isoformat(),
        "raw_file": "drill.md",
        "job_id": "drill",
        "failure_class": "apply.update_target_not_found",
        "fingerprint": (
            "apply.update_target_not_found:opus-4-7-evaluation-and-industry-geopolitics"
        ),
        "attempts": 3,
        "error": (
            "update target not found for page_id "
            "'opus-4-7-evaluation-and-industry-geopolitics'"
        ),
        "requested_page_id": "opus-4-7-evaluation-and-industry-geopolitics",
        "similar_existing_pages": ["ai/opus-4.7-evaluation-and-industry-geopolitics"],
        "status": "pending_local_repair",
    }


def run_drill(*, use_qwen: bool = True) -> dict[str, Any]:
    packet = drill_packet()
    decision = propose_repair(packet, use_qwen=use_qwen)
    return {"packet": packet, "decision": decision.to_dict()}


def _patch_chronovisor_paths(chronovisor_root: Path) -> dict[str, Any]:
    """Point path globals at a sandbox Chronovisor store for a drill."""

    pages = chronovisor_root / "pages"
    raw = chronovisor_root / "raw"
    system = chronovisor_root / "system"
    runtime = chronovisor_root / "runtime"
    for path in (pages, raw, system, runtime):
        path.mkdir(parents=True, exist_ok=True)

    from chronovisor.ingest import ingest, orchestrator

    snapshot = {
        "chronovisor": {
            "CHRONOVISOR_ROOT": chronovisor_store.CHRONOVISOR_ROOT,
            "PAGES_DIR": chronovisor_store.PAGES_DIR,
            "RAW_DIR": chronovisor_store.RAW_DIR,
            "SYSTEM_DIR": chronovisor_store.SYSTEM_DIR,
            "INDEX_FILE": chronovisor_store.INDEX_FILE,
            "LOG_FILE": chronovisor_store.LOG_FILE,
            "SCHEMA_FILE": chronovisor_store.SCHEMA_FILE,
            "ACTIVITY_FILE": chronovisor_store.ACTIVITY_FILE,
        },
        "ingest": {
            "PAGES_DIR": ingest.PAGES_DIR,
            "CHRONOVISOR_ROOT": ingest.CHRONOVISOR_ROOT,
            "ACTIVITY_FILE": ingest.ACTIVITY_FILE,
        },
        "orchestrator": {
            "RAW_DIR": orchestrator.RAW_DIR,
            "CHRONOVISOR_ROOT": orchestrator.CHRONOVISOR_ROOT,
            "ACTIVITY_FILE": orchestrator.ACTIVITY_FILE,
            "STATE_FILE": orchestrator.STATE_FILE,
        },
        "runtime_status": {
            "RUNTIME_DIR": runtime_status.RUNTIME_DIR,
            "STATUS_FILE": runtime_status.STATUS_FILE,
            "EVENTS_FILE": runtime_status.EVENTS_FILE,
            "METRICS_FILE": runtime_status.METRICS_FILE,
        },
    }

    chronovisor_store.CHRONOVISOR_ROOT = chronovisor_root
    chronovisor_store.PAGES_DIR = pages
    chronovisor_store.RAW_DIR = raw
    chronovisor_store.SYSTEM_DIR = system
    chronovisor_store.INDEX_FILE = pages / "index.md"
    chronovisor_store.LOG_FILE = pages / "log.md"
    chronovisor_store.SCHEMA_FILE = system / "schema.md"
    chronovisor_store.ACTIVITY_FILE = runtime / "activity.jsonl"

    ingest.PAGES_DIR = pages
    ingest.CHRONOVISOR_ROOT = chronovisor_root
    ingest.ACTIVITY_FILE = runtime / "activity.jsonl"
    orchestrator.RAW_DIR = raw
    orchestrator.CHRONOVISOR_ROOT = chronovisor_root
    orchestrator.ACTIVITY_FILE = runtime / "activity.jsonl"
    orchestrator.STATE_FILE = chronovisor_root / ".orchestrator_state.json"

    chronovisor_store.INDEX_FILE.write_bytes(reserved_documents.render_pages_index(()))
    chronovisor_store.LOG_FILE.write_bytes(reserved_documents.render_pages_log())
    chronovisor_store.SCHEMA_FILE.write_text(chronovisor_store.SCHEMA_CONTENT)
    chronovisor_store.ACTIVITY_FILE.touch(mode=0o600)
    live_layout.write_live_layout_proof(chronovisor_root, state="ready")

    runtime_status.RUNTIME_DIR = runtime
    runtime_status.STATUS_FILE = runtime / "status.json"
    runtime_status.EVENTS_FILE = runtime / "events.jsonl"
    runtime_status.METRICS_FILE = runtime / "metrics.jsonl"
    return snapshot


def _restore_chronovisor_paths(snapshot: dict[str, Any]) -> None:
    """Restore path globals after a sandbox drill."""

    from chronovisor.ingest import ingest, orchestrator

    for name, value in snapshot["chronovisor"].items():
        setattr(chronovisor_store, name, value)
    for name, value in snapshot["ingest"].items():
        setattr(ingest, name, value)
    for name, value in snapshot["orchestrator"].items():
        setattr(orchestrator, name, value)
    for name, value in snapshot["runtime_status"].items():
        setattr(runtime_status, name, value)


def run_sandbox_drill(*, use_qwen: bool = True) -> dict[str, Any]:
    """Exercise pending raw -> failure packet -> self-heal -> retry success."""

    sandbox_root = Path(
        tempfile.mkdtemp(prefix="chronovisor-self-heal-drill-")
    ).resolve()
    path_snapshot = _patch_chronovisor_paths(sandbox_root)

    page = (
        sandbox_root
        / "pages"
        / "ai"
        / "opus-4.7-evaluation-and-industry-geopolitics.md"
    )
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(
        "---\ntitle: Opus\nupdated: 2026-01-01\nstatus: stable\n"
        "type: knowledge\n---\nold\n",
        encoding="utf-8",
    )
    raw_path = sandbox_root / "raw" / "broken.md"
    raw_path.write_text("sandbox drill raw\n", encoding="utf-8")

    old_autorun = os.environ.get("CHRONOVISOR_SELF_HEAL_AUTORUN")
    os.environ["CHRONOVISOR_SELF_HEAL_AUTORUN"] = "0"

    from chronovisor.core.alias_store import load_aliases
    from chronovisor.core.jobs import JobStatus, job_store
    from chronovisor.ingest import ingest as ingest_mod
    from chronovisor.ingest import orchestrator

    original_run_ingest = ingest_mod.run_ingest
    original_run_frontier = globals()["_run_frontier"]
    original_authority_preflight = orchestrator.ingest_authority_preflight

    def fake_run_ingest(
        content,
        job_id,
        on_complete=None,
        on_finally=None,
        *,
        metadata=None,
        frontier_reviewer=None,
    ):
        if frontier_reviewer is not None:
            raise AssertionError("sandbox drill must not invoke frontier authority")
        aliases = load_aliases()
        if (
            aliases.get("opus-4-7-evaluation-and-industry-geopolitics")
            == "ai/opus-4.7-evaluation-and-industry-geopolitics"
        ):
            job_store.update(job_id, status=JobStatus.COMPLETED)
            if on_complete:
                on_complete()
            if on_finally:
                on_finally(failed=False, triage_failed=False)
            return
        job_store.update(
            job_id,
            status=JobStatus.FAILED,
            error=(
                "update target not found for page_id "
                "'opus-4-7-evaluation-and-industry-geopolitics'"
            ),
        )
        if on_finally:
            on_finally(failed=True, triage_failed=False)

    ingest_mod.run_ingest = fake_run_ingest
    orchestrator.ingest_authority_preflight = lambda **_kwargs: {
        "ok": True,
        "status": "ready",
        "blocked_by": None,
        "retryable": False,
        "error": None,
        "artifact_sha256": "sandbox-drill-authority",
    }
    globals()["_run_frontier"] = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("sandbox semantic repair must remain in the local data plane")
    )
    try:
        batches = [orchestrator.run_pending_ingest(force=True) for _ in range(3)]
        packet_paths = sorted((_packet_dir()).glob("*.json"))
        heal_result = None
        if packet_paths:
            heal_result = handle_packet(
                packet_paths[0],
                use_qwen=use_qwen,
                enable_frontier=True,
                dry_run=False,
            )
        pending_after = [p.name for p in orchestrator.get_pending_raw_files()]
        aliases = load_aliases()
    finally:
        ingest_mod.run_ingest = original_run_ingest
        orchestrator.ingest_authority_preflight = original_authority_preflight
        globals()["_run_frontier"] = original_run_frontier
        if old_autorun is None:
            os.environ.pop("CHRONOVISOR_SELF_HEAL_AUTORUN", None)
        else:
            os.environ["CHRONOVISOR_SELF_HEAL_AUTORUN"] = old_autorun
        _restore_chronovisor_paths(path_snapshot)

    return {
        "status": "ok",
        "sandbox_root": str(sandbox_root),
        "batches": batches,
        "packet_paths": [str(p) for p in packet_paths],
        "heal_result": heal_result,
        "pending_after": pending_after,
        "aliases": aliases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Chronovisor self-healing.")
    parser.add_argument("--packet", type=Path, help="Process one failure packet.")
    parser.add_argument(
        "--release-operational-repair",
        type=Path,
        metavar="PACKET",
        help=(
            "Mark one operational packet repaired after verifying its exact "
            "class, fingerprint, and pushed runtime commit."
        ),
    )
    parser.add_argument("--expected-status")
    parser.add_argument("--expected-failure-class")
    parser.add_argument("--expected-fingerprint")
    parser.add_argument(
        "--expected-raw-sha256",
        action="append",
        default=[],
        metavar="FILENAME=SHA256",
        help=(
            "Repeat once for every raw currently attached to the packet; the "
            "exact filename set and byte hashes form the release CAS."
        ),
    )
    parser.add_argument("--repair-commit")
    parser.add_argument("--repair-reason")
    parser.add_argument("--verification-command")
    parser.add_argument("--verification-result")
    parser.add_argument("--max-packets", type=int, default=3)
    parser.add_argument(
        "--auto-apply-errors",
        action="store_true",
        help="Promote repeated recall auto-apply errors into self-heal packets.",
    )
    parser.add_argument("--auto-apply-error-threshold", type=int, default=3)
    parser.add_argument("--no-qwen", action="store_true")
    parser.add_argument(
        "--enable-frontier-repair",
        action="store_true",
        help="Allow an eligible system-code incident to request the guarded repair plane.",
    )
    parser.add_argument("--no-frontier", action="store_true")
    parser.add_argument(
        "--review-only", action="store_true", help="Frontier may review but not patch."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--drill", action="store_true", help="Run a synthetic local repair drill."
    )
    parser.add_argument(
        "--sandbox-drill",
        action="store_true",
        help="Run a sandbox pending-raw self-heal drill without touching production store.",
    )
    return parser


_BACKGROUND_RETRY_STATUSES = frozenset(
    {
        "busy",
        "error",
        "budget_deferred",
        "pending_local_repair",
        "local_repair_failed",
        "pending_frontier",
        "frontier_running",
        "frontier_retry",
        "frontier_preflight_failed",
        "pending_frontier_review",
    }
)
_BACKGROUND_QUARANTINE_STATUSES = frozenset(
    {
        "local_quarantined",
        "frontier_quarantined",
        "frontier_rejected",
        "human_required",
    }
)


def _background_exit_code(result: dict[str, Any]) -> int:
    """Map durable repair state to the background-ledger exit protocol."""

    from chronovisor.core.background_jobs import (
        QUARANTINE_EXIT_CODE,
        RETRYABLE_EXIT_CODE,
    )

    rows = [result]
    nested = result.get("results")
    if isinstance(nested, list):
        rows.extend(row for row in nested if isinstance(row, dict))
    statuses = {str(row.get("status") or "") for row in rows}
    if statuses & _BACKGROUND_RETRY_STATUSES:
        return RETRYABLE_EXIT_CODE
    if statuses & _BACKGROUND_QUARANTINE_STATUSES:
        return QUARANTINE_EXIT_CODE
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-self-heal`` command-line entry point."""
    args = build_parser().parse_args(argv)
    if args.dry_run:
        return _main_locked(args)
    from chronovisor.core.okf_cutover import OKFStartupBlocked

    try:
        with chronovisor_store.okf_runtime_operation(
            chronovisor_store.CHRONOVISOR_ROOT
        ):
            return _main_locked(args)
    except OKFStartupBlocked:
        print(json.dumps({"status": "blocked", "category": "okf_startup_blocked"}))
        return 75


def _main_locked(args: argparse.Namespace) -> int:
    frontier_enabled = args.enable_frontier_repair and not args.no_frontier
    release_evidence = (
        args.expected_status,
        args.expected_failure_class,
        args.expected_fingerprint,
        *(args.expected_raw_sha256 or []),
        args.repair_commit,
        args.repair_reason,
        args.verification_command,
        args.verification_result,
    )
    if args.release_operational_repair is not None:
        if (
            args.packet is not None
            or args.sandbox_drill
            or args.drill
            or args.auto_apply_errors
        ):
            result = {
                "status": "refused",
                "accepted": False,
                "reason": "release_operational_repair_action_conflict",
            }
        else:
            result = release_operational_failure_after_local_repair(
                args.release_operational_repair,
                expected_status=str(args.expected_status or ""),
                expected_failure_class=str(args.expected_failure_class or ""),
                expected_fingerprint=str(args.expected_fingerprint or ""),
                expected_raw_sha256=args.expected_raw_sha256 or [],
                repair_commit=str(args.repair_commit or ""),
                reason=str(args.repair_reason or ""),
                verification_command=args.verification_command,
                verification_result=args.verification_result,
                dry_run=args.dry_run,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 0 if result.get("accepted") is True else 2
    if any(value is not None for value in release_evidence):
        result = {
            "status": "refused",
            "accepted": False,
            "reason": "release_operational_repair_evidence_without_action",
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2
    if args.sandbox_drill:
        print(
            json.dumps(
                run_sandbox_drill(use_qwen=not args.no_qwen),
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return 0
    if args.drill:
        print(
            json.dumps(
                run_drill(use_qwen=not args.no_qwen), ensure_ascii=False, indent=2
            )
        )
        return 0
    if args.auto_apply_errors:
        result = run_auto_apply_error_self_heal(
            threshold=args.auto_apply_error_threshold,
            max_packets=args.max_packets,
            use_qwen=not args.no_qwen,
            enable_frontier=frontier_enabled,
            execute_frontier_patch=not args.review_only,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return _background_exit_code(result)
    if args.packet:
        result = handle_packet(
            args.packet,
            use_qwen=not args.no_qwen,
            enable_frontier=frontier_enabled,
            execute_frontier_patch=not args.review_only,
            dry_run=args.dry_run,
        )
    else:
        result = run_pending(
            max_packets=args.max_packets,
            use_qwen=not args.no_qwen,
            enable_frontier=frontier_enabled,
            execute_frontier_patch=not args.review_only,
            dry_run=args.dry_run,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return _background_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
