"""Durable self-heal packet cancellation sidecars."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from chronovisor.core import store as chronovisor_store

PACKET_CANCELLATION_SCHEMA_VERSION = 1
PACKET_CANCELLATION_STATUS = "superseded_semantic_defer"
VERIFIED_LOCAL_REPAIR_CANCELLATION_STATUS = "superseded_verified_local_repair"
PACKET_CANCELLATION_STATUSES = frozenset(
    {PACKET_CANCELLATION_STATUS, VERIFIED_LOCAL_REPAIR_CANCELLATION_STATUS}
)
PACKET_SUCCESS_STATUSES = frozenset({"local_repair_applied", "frontier_approved"})


def packet_cancellation_dir() -> Path:
    return (
        chronovisor_store.CHRONOVISOR_ROOT
        / "runtime"
        / "failures"
        / "packet-cancellations"
    )


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception:
        with suppress(OSError):
            tmp.unlink()
        raise


def packet_cancellation_path(packet_path: Path) -> Path:
    return packet_cancellation_dir() / f"{packet_path.name}.json"


def read_packet_cancellation(
    packet_path: Path,
    packet: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    path = packet_cancellation_path(packet_path)
    try:
        cancellation = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    resolved_packet = str(packet_path.expanduser().resolve(strict=False))
    if (
        not isinstance(cancellation, dict)
        or cancellation.get("schema_version") != PACKET_CANCELLATION_SCHEMA_VERSION
        or cancellation.get("status") not in PACKET_CANCELLATION_STATUSES
        or cancellation.get("packet_path") != resolved_packet
        or not isinstance(cancellation.get("requested_at"), str)
        or not str(cancellation.get("requested_at") or "").strip()
    ):
        return None
    if packet is not None:
        for field in ("failure_id", "fingerprint"):
            expected = cancellation.get(field)
            observed = packet.get(field)
            if expected is not None and expected != observed:
                return None
    return cancellation


def request_packet_cancellation(
    packet_path: Path,
    *,
    reason: str,
    superseded_by_packet: Path,
    cancellation_status: str = PACKET_CANCELLATION_STATUS,
) -> dict[str, Any]:
    """Publish a lock-free cancellation observed by an in-flight worker."""

    resolved = packet_path.expanduser().resolve(strict=False)
    superseded_by = superseded_by_packet.expanduser().resolve(strict=False)
    if cancellation_status not in PACKET_CANCELLATION_STATUSES:
        raise ValueError("packet cancellation status is not allowlisted")
    try:
        loaded_packet = read_json(resolved)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        loaded_packet = {}
    packet = loaded_packet if isinstance(loaded_packet, dict) else {}
    if packet.get("status") in PACKET_SUCCESS_STATUSES:
        return {
            "accepted": False,
            "reason": "packet_already_completed",
            "packet_path": str(resolved),
            "status": packet.get("status"),
        }
    cancellation_path = packet_cancellation_path(resolved)
    prior = read_packet_cancellation(resolved, packet)
    requested_at = (
        str(prior.get("requested_at"))
        if isinstance(prior, dict)
        else datetime.now().isoformat()
    )
    cancellation = {
        "schema_version": PACKET_CANCELLATION_SCHEMA_VERSION,
        "status": cancellation_status,
        "packet_path": str(resolved),
        "packet_name": resolved.name,
        "failure_id": packet.get("failure_id"),
        "fingerprint": packet.get("fingerprint"),
        "requested_at": requested_at,
        "reason": str(reason).strip() or "superseded",
        "superseded_by_packet": str(superseded_by),
    }
    write_json(cancellation_path, cancellation)
    return {
        "accepted": True,
        "cancellation_path": str(cancellation_path),
        **cancellation,
    }
