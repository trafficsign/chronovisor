"""Librarian inventory and synchronization for managed semantic holds."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import canonical_bytes
from chronovisor.core.managed_hold import (
    ManagedHoldError,
    ManagedHoldStore,
    hold_identity,
)

__all__ = [
    "ManagedHoldError",
    "ManagedHoldStore",
    "hold_identity",
    "ingest_semantic_hold_inventory",
    "sync_ingest_semantic_holds",
]


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


def _superseded_hold_evidence(
    inventory: Iterable[Mapping[str, Any]],
    current_failures: Mapping[str, Mapping[str, Any]],
) -> dict[str, set[str]]:
    superseded: dict[str, set[str]] = {}
    for row in inventory:
        packet_path = row.get("packet_path")
        row_raw_files = row.get("raw_files")
        if not isinstance(packet_path, str) or not isinstance(row_raw_files, list):
            continue
        packet_identity = Path(packet_path).expanduser().resolve(strict=False)
        current_packet_paths: set[str] = set()
        for raw_file in row_raw_files:
            current_entry = current_failures.get(str(raw_file))
            current_packet = (
                current_entry.get("packet_path")
                if isinstance(current_entry, Mapping)
                else None
            )
            if not isinstance(current_packet, str):
                break
            current_identity = Path(current_packet).expanduser().resolve(strict=False)
            if current_identity == packet_identity:
                break
            current_packet_paths.add(str(current_identity))
        else:
            identity_fields = {
                key: row[key]
                for key in ("hold_sha256", "authority_epoch", "raw_sha256", "lane")
            }
            superseded[hold_identity(**identity_fields)] = current_packet_paths
    return superseded


def sync_ingest_semantic_holds(
    *,
    chronovisor_root: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    inventory = ingest_semantic_hold_inventory(chronovisor_root)
    try:
        from chronovisor.raw.failure_supervisor import (
            current_adopted_authority_epoch,
        )

        current_authority = current_adopted_authority_epoch()
    except Exception:
        current_authority = None
    raw_files = {
        raw_file
        for row in inventory
        for raw_file in row.get("raw_files", [])
        if isinstance(raw_file, str)
    }
    try:
        from chronovisor.raw.failure_supervisor import raw_failure_snapshot

        current_failures = raw_failure_snapshot(raw_files)
    except Exception:
        current_failures = {}
    superseded = _superseded_hold_evidence(inventory, current_failures)
    if dry_run:
        return {
            "status": "dry_run",
            "inventory": len(inventory),
            "current_authority_epoch": current_authority,
            "would_schedule": sum(
                bool(current_authority and row["authority_epoch"] != current_authority)
                for row in inventory
            ),
            "would_supersede": len(superseded),
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
    superseded_resolved = store.resolve_superseded_scheduled(superseded)
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
    absent_resolved = store.resolve_absent_scheduled(active_identities)
    resolved = [*superseded_resolved, *absent_resolved]
    return {
        "status": "ok",
        "inventory": len(inventory),
        "registered": registered,
        "current_authority_epoch": current_authority,
        "scheduled": scheduled["count"],
        "resolved": len(resolved),
        "superseded": len(superseded_resolved),
        "snapshot": store.snapshot(),
    }
