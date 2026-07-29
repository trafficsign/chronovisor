"""Exactly-once lease registry for semantic no-quorum holds.

Operational formatting, truncation, resource, and transport failures are not
accepted by this store.  The existing failure supervisor remains the source of
truth for immutable raw/authority evidence; this registry makes its lifecycle
explicit and recoverable across authority epochs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import (
    DurableStateError,
    canonical_bytes,
    file_lock,
    read_sealed_json,
    write_sealed_json,
)
from chronovisor.core.timeutil import iso_seconds as _iso

MANAGED_HOLD_SCHEMA_VERSION = 1
STATES = frozenset({"active", "scheduled", "leased", "resolved", "reheld"})
DEFAULT_RETRY_BASE_SECONDS = 15 * 60
DEFAULT_MAX_BACKOFF_SECONDS = 24 * 60 * 60
DEFAULT_LANE_LEASE_INTERVAL_SECONDS = 30


class ManagedHoldError(RuntimeError):
    pass


def _utc(value: datetime | None = None) -> datetime:
    current = value or datetime.now(UTC)
    if current.tzinfo is None:
        return current.replace(tzinfo=UTC)
    return current.astimezone(UTC)




def _parse(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return _utc(parsed)


def hold_identity(
    *,
    hold_sha256: str,
    authority_epoch: str,
    raw_sha256: str,
    lane: str,
) -> str:
    return hashlib.sha256(
        canonical_bytes(
            {
                "hold_sha256": hold_sha256,
                "authority_epoch": authority_epoch,
                "raw_sha256": raw_sha256,
                "lane": lane,
            }
        )
    ).hexdigest()


class ManagedHoldStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.with_name(f"{path.name}.lock")

    def _empty(self) -> dict[str, Any]:
        return {
            "schema_version": MANAGED_HOLD_SCHEMA_VERSION,
            "entries": {},
            "authority_epochs": {},
            "lane_last_lease_at": {},
        }

    def _load(self) -> dict[str, Any]:
        try:
            state = read_sealed_json(self.path, recover_backup=True)
        except DurableStateError as exc:
            if self.path.exists() or self.path.with_name(f"{self.path.name}.bak").exists():
                raise ManagedHoldError("managed hold state is not recoverable") from exc
            return self._empty()
        if not isinstance(state.get("entries"), dict):
            raise ManagedHoldError("managed hold entries are malformed")
        return state

    def _write(self, state: Mapping[str, Any]) -> dict[str, Any]:
        return write_sealed_json(self.path, state, backup=True)

    def register(
        self,
        *,
        hold_sha256: str,
        authority_epoch: str,
        raw_sha256: str,
        lane: str,
        raw_files: Iterable[str] = (),
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = _utc(now)
        identity = hold_identity(
            hold_sha256=hold_sha256,
            authority_epoch=authority_epoch,
            raw_sha256=raw_sha256,
            lane=lane,
        )
        with file_lock(self.lock_path, exclusive=True):
            state = self._load()
            entries = state["entries"]
            entry = entries.get(identity)
            if not isinstance(entry, dict):
                entry = {
                    "identity": identity,
                    "hold_sha256": hold_sha256,
                    "authority_epoch": authority_epoch,
                    "raw_sha256": raw_sha256,
                    "lane": lane,
                    "raw_files": sorted(set(raw_files)),
                    "state": "active",
                    "attempts": 0,
                    "created_at": _iso(current),
                }
            elif entry.get("raw_sha256") != raw_sha256:
                raise ManagedHoldError("managed hold raw hash invariant changed")
            entry["updated_at"] = _iso(current)
            entries[identity] = entry
            self._write(state)
            return dict(entry)

    def register_many(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        now: datetime | None = None,
    ) -> int:
        """Register one inventory with a single lock and durable publication."""

        current = _utc(now)
        registered = 0
        with file_lock(self.lock_path, exclusive=True):
            state = self._load()
            entries = state["entries"]
            for row in rows:
                identity = hold_identity(
                    hold_sha256=str(row["hold_sha256"]),
                    authority_epoch=str(row["authority_epoch"]),
                    raw_sha256=str(row["raw_sha256"]),
                    lane=str(row["lane"]),
                )
                entry = entries.get(identity)
                if not isinstance(entry, dict):
                    entry = {
                        "identity": identity,
                        "hold_sha256": row["hold_sha256"],
                        "authority_epoch": row["authority_epoch"],
                        "raw_sha256": row["raw_sha256"],
                        "lane": row["lane"],
                        "raw_files": sorted(set(row.get("raw_files") or ())),
                        "state": "active",
                        "attempts": 0,
                        "created_at": _iso(current),
                    }
                    registered += 1
                elif entry.get("raw_sha256") != row["raw_sha256"]:
                    raise ManagedHoldError("managed hold raw hash invariant changed")
                entry["updated_at"] = _iso(current)
                entries[identity] = entry
            if registered:
                self._write(state)
        return registered

    def reconcile_authorities(
        self,
        current_epochs: Mapping[str, str],
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = _utc(now)
        scheduled: list[str] = []
        with file_lock(self.lock_path, exclusive=True):
            state = self._load()
            normalized_epochs = dict(sorted(current_epochs.items()))
            changed = state.get("authority_epochs") != normalized_epochs
            state["authority_epochs"] = normalized_epochs
            for identity, entry in state["entries"].items():
                if not isinstance(entry, dict):
                    continue
                lane = str(entry.get("lane") or "")
                observed = current_epochs.get(lane)
                next_attempt = _parse(entry.get("next_attempt_at"))
                if (
                    observed
                    and observed != entry.get("authority_epoch")
                    and entry.get("state") in {"active", "reheld"}
                    and (next_attempt is None or next_attempt <= current)
                ):
                    entry["state"] = "scheduled"
                    entry["scheduled_for_authority_epoch"] = observed
                    entry["scheduled_at"] = _iso(current)
                    entry["updated_at"] = _iso(current)
                    scheduled.append(identity)
            if changed or scheduled:
                self._write(state)
        return {"scheduled": scheduled, "count": len(scheduled)}

    def recover_expired(
        self,
        *,
        now: datetime | None = None,
    ) -> list[str]:
        current = _utc(now)
        recovered: list[str] = []
        with file_lock(self.lock_path, exclusive=True):
            state = self._load()
            for identity, entry in state["entries"].items():
                if not isinstance(entry, dict) or entry.get("state") != "leased":
                    continue
                expiry = _parse(entry.get("lease_expires_at"))
                if expiry is None or expiry <= current:
                    entry["state"] = "scheduled"
                    entry["lease_owner"] = None
                    entry["lease_token"] = None
                    entry["lease_expires_at"] = None
                    entry["updated_at"] = _iso(current)
                    recovered.append(identity)
            if recovered:
                self._write(state)
        return recovered

    def resolve_absent_scheduled(
        self,
        active_identities: set[str],
        *,
        now: datetime | None = None,
    ) -> list[str]:
        """Close scheduled holds whose canonical packet has been retired.

        The existing ingest transaction retires a semantic packet only after
        its raw unit succeeds or is superseded by a new authority-bound hold.
        This registry records the exactly-once lease transition under one
        lock after observing that durable evidence; it never guesses from a
        missing queue counter.
        """

        current = _utc(now)
        resolved: list[str] = []
        with file_lock(self.lock_path, exclusive=True):
            state = self._load()
            for identity, entry in state["entries"].items():
                if (
                    not isinstance(entry, dict)
                    or entry.get("state") != "scheduled"
                    or identity in active_identities
                ):
                    continue
                attempt = int(entry.get("attempts") or 0) + 1
                token = hashlib.sha256(
                    canonical_bytes(
                        {
                            "identity": identity,
                            "owner": "canonical-packet-retirement",
                            "attempt": attempt,
                            "retired_at": _iso(current),
                        }
                    )
                ).hexdigest()
                history = entry.get("transition_history")
                history = list(history) if isinstance(history, list) else []
                history.extend(
                    [
                        {
                            "from": "scheduled",
                            "to": "leased",
                            "at": _iso(current),
                            "lease_token_sha256": hashlib.sha256(
                                token.encode("utf-8")
                            ).hexdigest(),
                        },
                        {
                            "from": "leased",
                            "to": "resolved",
                            "at": _iso(current),
                            "evidence": "canonical_semantic_packet_retired",
                        },
                    ]
                )
                entry.update(
                    {
                        "state": "resolved",
                        "attempts": attempt,
                        "lease_owner": None,
                        "lease_token": None,
                        "lease_expires_at": None,
                        "finished_at": _iso(current),
                        "updated_at": _iso(current),
                        "transition_history": history[-20:],
                    }
                )
                resolved.append(identity)
            if resolved:
                self._write(state)
        return resolved

    def acquire(
        self,
        *,
        owner: str,
        lease_seconds: int = 1800,
        max_attempts: int = 5,
        minimum_lane_interval_seconds: int = DEFAULT_LANE_LEASE_INTERVAL_SECONDS,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        current = _utc(now)
        self.recover_expired(now=current)
        with file_lock(self.lock_path, exclusive=True):
            state = self._load()
            lane_last_lease_at = state.get("lane_last_lease_at")
            if not isinstance(lane_last_lease_at, dict):
                lane_last_lease_at = {}
                state["lane_last_lease_at"] = lane_last_lease_at
            candidates = []
            for entry in state["entries"].values():
                if (
                    not isinstance(entry, dict)
                    or entry.get("state") != "scheduled"
                    or int(entry.get("attempts") or 0) >= max(1, max_attempts)
                ):
                    continue
                due = _parse(entry.get("next_attempt_at"))
                if due is not None and due > current:
                    continue
                lane = str(entry.get("lane") or "")
                last_lease = _parse(lane_last_lease_at.get(lane))
                if (
                    last_lease is not None
                    and current
                    < last_lease
                    + timedelta(seconds=max(0, minimum_lane_interval_seconds))
                ):
                    continue
                candidates.append(entry)
            candidates.sort(
                key=lambda entry: (
                    str(entry.get("scheduled_at") or entry.get("created_at") or ""),
                    str(entry.get("identity") or ""),
                )
            )
            if not candidates:
                return None
            entry = candidates[0]
            attempt = int(entry.get("attempts") or 0) + 1
            token = hashlib.sha256(
                canonical_bytes(
                    {
                        "identity": entry["identity"],
                        "owner": owner,
                        "attempt": attempt,
                        "scheduled_for": entry.get("scheduled_for_authority_epoch"),
                    }
                )
            ).hexdigest()
            entry.update(
                {
                    "state": "leased",
                    "attempts": attempt,
                    "lease_owner": owner,
                    "lease_token": token,
                    "lease_expires_at": _iso(
                        current + timedelta(seconds=max(1, lease_seconds))
                    ),
                    "updated_at": _iso(current),
                }
            )
            lane_last_lease_at[str(entry.get("lane") or "")] = _iso(current)
            self._write(state)
            return dict(entry)

    def finish(
        self,
        *,
        identity: str,
        lease_token: str,
        outcome: str,
        observed_raw_sha256: str,
        retry_base_seconds: int = DEFAULT_RETRY_BASE_SECONDS,
        max_backoff_seconds: int = DEFAULT_MAX_BACKOFF_SECONDS,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        if outcome not in {"resolved", "reheld"}:
            raise ValueError("managed hold outcome must be resolved or reheld")
        current = _utc(now)
        with file_lock(self.lock_path, exclusive=True):
            state = self._load()
            entry = state["entries"].get(identity)
            if not isinstance(entry, dict) or entry.get("state") != "leased":
                raise ManagedHoldError("managed hold is not leased")
            if entry.get("lease_token") != lease_token:
                raise ManagedHoldError("managed hold lease token mismatch")
            if entry.get("raw_sha256") != observed_raw_sha256:
                raise ManagedHoldError("managed hold raw hash invariant changed")
            entry.update(
                {
                    "state": outcome,
                    "lease_owner": None,
                    "lease_token": None,
                    "lease_expires_at": None,
                    "finished_at": _iso(current),
                    "updated_at": _iso(current),
                }
            )
            if outcome == "reheld":
                attempts = max(1, int(entry.get("attempts") or 1))
                delay = min(
                    max(1, max_backoff_seconds),
                    max(1, retry_base_seconds) * (2 ** max(0, attempts - 1)),
                )
                entry["next_attempt_at"] = _iso(
                    current + timedelta(seconds=delay)
                )
            else:
                entry.pop("next_attempt_at", None)
            self._write(state)
            return dict(entry)

    def release_unattempted(
        self,
        *,
        identity: str,
        lease_token: str,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Return a lease that did not reach its raw to the scheduled queue."""

        current = _utc(now)
        with file_lock(self.lock_path, exclusive=True):
            state = self._load()
            entry = state["entries"].get(identity)
            if not isinstance(entry, dict) or entry.get("state") != "leased":
                raise ManagedHoldError("managed hold is not leased")
            if entry.get("lease_token") != lease_token:
                raise ManagedHoldError("managed hold lease token mismatch")
            entry.update(
                {
                    "state": "scheduled",
                    "attempts": max(0, int(entry.get("attempts") or 0) - 1),
                    "lease_owner": None,
                    "lease_token": None,
                    "lease_expires_at": None,
                    "updated_at": _iso(current),
                }
            )
            self._write(state)
            return dict(entry)

    def snapshot(self) -> dict[str, Any]:
        with file_lock(self.lock_path, exclusive=False):
            state = self._load()
        counts = {name: 0 for name in sorted(STATES)}
        for entry in state["entries"].values():
            if isinstance(entry, dict) and entry.get("state") in counts:
                counts[str(entry["state"])] += 1
        return {
            "schema_version": MANAGED_HOLD_SCHEMA_VERSION,
            "total": sum(counts.values()),
            "counts": counts,
            "authority_epochs": state.get("authority_epochs", {}),
        }


def ingest_semantic_hold_inventory(chronovisor_root: Path) -> list[dict[str, Any]]:
    packets = chronovisor_root / "runtime" / "failures" / "packets"
    rows: list[dict[str, Any]] = []
    for path in sorted(packets.glob("*.json")):
        try:
            packet = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if (
            not isinstance(packet, dict)
            or packet.get("failure_class") != "ingest.semantic_no_quorum"
            or packet.get("terminal_deferred") is not True
            or packet.get("status") != "local_quarantined"
        ):
            continue
        sources = packet.get("source_raws")
        if not isinstance(sources, list) or not sources:
            continue
        normalized = [
            {
                "filename": row.get("filename"),
                "bytes": row.get("bytes"),
                "sha256": row.get("sha256"),
            }
            for row in sources
            if isinstance(row, dict)
        ]
        if not normalized or any(not isinstance(row.get("sha256"), str) for row in normalized):
            continue
        raw_sha = hashlib.sha256(canonical_bytes(normalized)).hexdigest()
        authority = str(
            packet.get("authority_epoch")
            or packet.get("authority_artifact_sha256")
            or ""
        )
        hold_sha = hashlib.sha256(
            canonical_bytes(
                {
                    "failure_class": packet.get("failure_class"),
                    "fingerprint": packet.get("fingerprint"),
                    "authority": authority,
                    "sources": normalized,
                }
            )
        ).hexdigest()
        rows.append(
            {
                "hold_sha256": hold_sha,
                "authority_epoch": authority,
                "raw_sha256": raw_sha,
                "lane": "ingest_reconciliation",
                "raw_files": [str(row["filename"]) for row in normalized],
                "packet_path": str(path),
            }
        )
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity_fields = {
            key: row[key]
            for key in ("hold_sha256", "authority_epoch", "raw_sha256", "lane")
        }
        unique[hold_identity(**identity_fields)] = row
    return list(unique.values())


def sync_ingest_semantic_holds(
    *,
    chronovisor_root: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    inventory = ingest_semantic_hold_inventory(chronovisor_root)
    try:
        from chronovisor.decision.failure_supervisor import (
            _current_adopted_authority_epoch,
        )

        current_authority = _current_adopted_authority_epoch()
    except Exception:
        current_authority = None
    if dry_run:
        return {
            "status": "dry_run",
            "inventory": len(inventory),
            "current_authority_epoch": current_authority,
            "would_schedule": sum(
                bool(current_authority and row["authority_epoch"] != current_authority)
                for row in inventory
            ),
        }
    store = ManagedHoldStore(
        chronovisor_root / "runtime" / "managed-holds" / "state.json"
    )
    registered = store.register_many(inventory)
    scheduled = store.reconcile_authorities(
        {"ingest_reconciliation": current_authority}
        if isinstance(current_authority, str)
        else {}
    )
    active_identities = {
        hold_identity(
            **{
                key: row[key]
                for key in (
                    "hold_sha256",
                    "authority_epoch",
                    "raw_sha256",
                    "lane",
                )
            }
        )
        for row in inventory
    }
    resolved = store.resolve_absent_scheduled(active_identities)
    return {
        "status": "ok",
        "inventory": len(inventory),
        "registered": registered,
        "current_authority_epoch": current_authority,
        "scheduled": scheduled["count"],
        "resolved": len(resolved),
        "snapshot": store.snapshot(),
    }
