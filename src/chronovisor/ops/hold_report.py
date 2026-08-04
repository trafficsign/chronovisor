"""Read-only cross-source reporting for durable semantic holds."""

from __future__ import annotations

import fcntl
import json
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import (
    DurableStateError,
    read_sealed_json,
)
from chronovisor.decision.decision_router import QUORUM_SAFETY_POLICY_VERSION
from chronovisor.search.semantic_hold import STRUCTURED_REVIEW_HOLD_CACHE_KIND

HOLD_REPORT_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _iso(value: object) -> str:
    if not isinstance(value, str) or not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _mtime(path: Path) -> str:
    try:
        timestamp = path.stat().st_mtime
    except OSError:
        return ""
    return datetime.fromtimestamp(timestamp, UTC).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")


def _artifact_prefix(value: object) -> str:
    text = value if isinstance(value, str) else ""
    return text[:8] if _SHA256_RE.fullmatch(text) else "unknown"


@contextmanager
def _semantic_entry_read_lock(root: Path, entry: Path) -> Iterator[None]:
    """Share the cache writer's per-key lock without creating new state."""

    lock_path = root / "locks" / f"{entry.stem}.lock"
    try:
        descriptor = os.open(lock_path, os.O_RDONLY)
    except OSError:
        # Entry publication is atomic.  A missing historical lock therefore
        # still permits one all-old or all-new read without creating a file.
        yield
        return
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@contextmanager
def _managed_state_read_lock(lock_path: Path) -> Iterator[None]:
    """Share an existing managed-state lock without creating filesystem state."""

    try:
        descriptor = os.open(lock_path, os.O_RDONLY)
    except FileNotFoundError:
        # Managed state is atomically published.  Without a historical lock,
        # one direct read therefore observes either the old or new seal.
        yield
        return
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _semantic_rows(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    cache_root = root / "runtime" / "semantic-holds" / "structured-review"
    entries = cache_root / "entries"
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted(entries.glob("*.json")):
        try:
            with _semantic_entry_read_lock(cache_root, path):
                snapshot = path.read_bytes()
            record = json.loads(snapshot.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"semantic:{path.name}:{exc.__class__.__name__}")
            continue
        if not isinstance(record, Mapping) or record.get("kind") != (
            STRUCTURED_REVIEW_HOLD_CACHE_KIND
        ):
            errors.append(f"semantic:{path.name}:invalid_record")
            continue
        hold = record.get("hold")
        if not isinstance(hold, Mapping):
            errors.append(f"semantic:{path.name}:missing_hold")
            continue
        authority = hold.get("authority")
        authority = authority if isinstance(authority, Mapping) else {}
        router = authority.get("router")
        router = router if isinstance(router, Mapping) else {}
        consensus = hold.get("local_consensus")
        consensus = consensus if isinstance(consensus, Mapping) else {}
        policy_version = authority.get("quorum_safety_policy_version")
        if not isinstance(policy_version, int):
            policy_version = consensus.get("quorum_safety_policy_version")
        rows.append(
            {
                "source": "structured_review_cache",
                "lane": str(hold.get("lane") or "unknown"),
                "quarantine_reason": str(
                    consensus.get("quarantine_reason") or "unknown"
                ),
                "artifact_sha256_prefix": _artifact_prefix(
                    router.get("artifact_sha256")
                ),
                "created_at": _mtime(path),
                # Cache entries are immutable.  A quorum-policy epoch mismatch
                # makes the entry non-reusable and therefore resolved for this
                # report; current-version entries remain active.
                "state": (
                    "active"
                    if policy_version == QUORUM_SAFETY_POLICY_VERSION
                    else "resolved"
                ),
            }
        )
    return rows, errors


def _managed_rows(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    state_path = root / "runtime" / "managed-holds" / "state.json"
    if not state_path.exists():
        return [], []
    lock_path = state_path.with_name(f"{state_path.name}.lock")
    try:
        with _managed_state_read_lock(lock_path):
            state = read_sealed_json(state_path, recover_backup=False)
    except (DurableStateError, OSError) as exc:
        return [], [f"managed:{exc.__class__.__name__}"]
    entries = state.get("entries")
    if not isinstance(entries, Mapping):
        return [], ["managed:invalid_entries"]
    rows: list[dict[str, Any]] = []
    for _identity, entry in sorted(entries.items(), key=lambda item: str(item[0])):
        if not isinstance(entry, Mapping):
            continue
        created_at = _iso(
            entry.get("created_at")
            or entry.get("scheduled_at")
            or entry.get("updated_at")
            or entry.get("finished_at")
        )
        rows.append(
            {
                "source": "managed_holds",
                "lane": str(entry.get("lane") or "unknown"),
                "quarantine_reason": str(
                    entry.get("quarantine_reason")
                    or "managed_semantic_no_quorum"
                ),
                "artifact_sha256_prefix": _artifact_prefix(
                    entry.get("authority_epoch")
                ),
                "created_at": created_at,
                "state": (
                    "resolved" if entry.get("state") == "resolved" else "active"
                ),
            }
        )
    return rows, []


def build_hold_report(root: Path) -> dict[str, Any]:
    """Aggregate semantic and managed holds without changing either store."""

    semantic_rows, semantic_errors = _semantic_rows(root)
    managed_rows, managed_errors = _managed_rows(root)
    rows = [*semantic_rows, *managed_rows]
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["lane"]),
            str(row["quarantine_reason"]),
            str(row["artifact_sha256_prefix"]),
        )
        group = grouped.setdefault(
            key,
            {
                "lane": key[0],
                "quarantine_reason": key[1],
                "artifact_sha256_prefix": key[2],
                "created_at_min": None,
                "created_at_max": None,
                "active": 0,
                "resolved": 0,
                "total": 0,
                "sources": {},
            },
        )
        state = str(row["state"])
        group[state] += 1
        group["total"] += 1
        source = str(row["source"])
        group["sources"][source] = int(group["sources"].get(source, 0)) + 1
        created_at = str(row.get("created_at") or "")
        if created_at:
            current_min = group["created_at_min"]
            current_max = group["created_at_max"]
            group["created_at_min"] = (
                created_at if current_min is None else min(current_min, created_at)
            )
            group["created_at_max"] = (
                created_at if current_max is None else max(current_max, created_at)
            )
    groups = [grouped[key] for key in sorted(grouped)]
    return {
        "schema_version": HOLD_REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "root": str(root),
        "totals": {
            "active": sum(int(group["active"]) for group in groups),
            "resolved": sum(int(group["resolved"]) for group in groups),
            "total": sum(int(group["total"]) for group in groups),
        },
        "source_totals": {
            "structured_review_cache": len(semantic_rows),
            "managed_holds": len(managed_rows),
        },
        "groups": groups,
        "errors": [*semantic_errors, *managed_errors],
    }


def render_hold_report(report: Mapping[str, Any]) -> str:
    """Render the stable human-readable hold-report table."""

    totals = report.get("totals")
    totals = totals if isinstance(totals, Mapping) else {}
    lines = [
        "hold-report\t"
        f"active={int(totals.get('active') or 0)}\t"
        f"resolved={int(totals.get('resolved') or 0)}\t"
        f"total={int(totals.get('total') or 0)}",
        (
            "lane\tquarantine_reason\tartifact\tcreated_min\tcreated_max\t"
            "active\tresolved\ttotal"
        ),
    ]
    groups = report.get("groups")
    for group in groups if isinstance(groups, list) else []:
        if not isinstance(group, Mapping):
            continue
        lines.append(
            "\t".join(
                (
                    str(group.get("lane") or "unknown"),
                    str(group.get("quarantine_reason") or "unknown"),
                    str(group.get("artifact_sha256_prefix") or "unknown"),
                    str(group.get("created_at_min") or "--"),
                    str(group.get("created_at_max") or "--"),
                    str(int(group.get("active") or 0)),
                    str(int(group.get("resolved") or 0)),
                    str(int(group.get("total") or 0)),
                )
            )
        )
    errors = report.get("errors")
    for error in errors if isinstance(errors, list) else []:
        lines.append(f"warning\t{error}")
    return "\n".join(lines)


__all__ = ["HOLD_REPORT_SCHEMA_VERSION", "build_hold_report", "render_hold_report"]
