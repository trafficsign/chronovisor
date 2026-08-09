"""Durable event ledger, content-addressed evidence, and rebuildable checkpoints."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import zstandard

from chronovisor.core.jsonl_write import append_jsonl_durable
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.core.timeutil import utc_now as _now
from chronovisor.search.research_types import EvidenceArtifact


def _iso(value: datetime | None = None) -> str:
    return (value or _now()).isoformat(timespec="seconds")


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temporary)


class ResearchStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (CHRONOVISOR_ROOT / "runtime" / "research")
        self.runs = self.root / "runs"
        self.cas = self.root / "evidence-cas"
        self.durable_cas = CHRONOVISOR_ROOT / "research" / "evidence-cas"
        self.durable_manifests = CHRONOVISOR_ROOT / "research" / "evidence-manifests"
        self.checkpoints = CHRONOVISOR_ROOT / "runtime" / "session-checkpoints"

    def run_dir(self, run_id: str) -> Path:
        return self.runs / run_id

    def append_event(self, run_id: str, event: Mapping[str, Any]) -> dict[str, Any]:
        path = self.run_dir(run_id) / "events.jsonl"
        sequence = 1
        try:
            with path.open("rb") as handle:
                sequence += sum(1 for _line in handle)
        except OSError:
            pass
        record = {"ts": _iso(), "sequence": sequence, **dict(event)}
        append_jsonl_durable(path, [record], sort_keys=True)
        return record

    def events(self, run_id: str) -> list[dict[str, Any]]:
        path = self.run_dir(run_id) / "events.jsonl"
        rows: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        except OSError:
            pass
        return rows

    def put_artifact(
        self,
        content: bytes | str,
        *,
        source_type: str,
        source_uri: str,
        title: str = "",
        mime_type: str = "text/plain",
        citation: str = "",
        trust: str = "untrusted",
        durable: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> EvidenceArtifact:
        raw = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        digest = hashlib.sha256(raw).hexdigest()
        artifact_id = f"sha256:{digest}"
        blob_root = self.durable_cas if durable else self.cas
        blob = blob_root / digest[:2] / f"{digest}.zst"
        if not blob.exists():
            compressor = zstandard.ZstdCompressor(level=9)
            _atomic_bytes(blob, compressor.compress(raw))
        preview = raw[:1200].decode("utf-8", errors="replace")
        artifact = EvidenceArtifact(
            artifact_id=artifact_id,
            source_type=source_type,
            source_uri=source_uri,
            retrieved_at=_iso(),
            sha256=digest,
            byte_length=len(raw),
            preview=preview,
            trust=trust,
            title=title,
            mime_type=mime_type,
            citation=citation,
            durable=durable,
            metadata=dict(metadata or {}),
        )
        manifest_root = self.durable_manifests if durable else self.cas / digest[:2]
        manifest = manifest_root / f"{digest}.json"
        if not manifest.exists():
            _atomic_bytes(
                manifest,
                (json.dumps(artifact.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
            )
        return artifact

    def read_artifact(self, artifact_id: str) -> bytes:
        digest = artifact_id.removeprefix("sha256:")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("invalid artifact ID")
        paths = (
            self.durable_cas / digest[:2] / f"{digest}.zst",
            self.cas / digest[:2] / f"{digest}.zst",
        )
        compressed = next((path.read_bytes() for path in paths if path.exists()), None)
        if compressed is None:
            raise FileNotFoundError(artifact_id)
        raw = zstandard.ZstdDecompressor().decompress(compressed)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("evidence artifact checksum mismatch")
        return raw

    def artifact_manifest(self, artifact_id: str) -> EvidenceArtifact | None:
        digest = artifact_id.removeprefix("sha256:")
        for path in (
            self.durable_manifests / f"{digest}.json",
            self.cas / digest[:2] / f"{digest}.json",
        ):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                return EvidenceArtifact(**payload)
            except (OSError, json.JSONDecodeError, TypeError):
                continue
        return None

    def write_summary(self, run_id: str, payload: Mapping[str, Any]) -> Path:
        path = self.run_dir(run_id) / "summary.json"
        _atomic_bytes(
            path,
            (json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        )
        return path

    def write_bundle(self, run_id: str, payload: Mapping[str, Any]) -> Path:
        path = CHRONOVISOR_ROOT / "research" / "bundles" / f"{run_id}.json"
        _atomic_bytes(
            path,
            (json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        )
        return path

    def checkpoint(
        self,
        session_id: str,
        payload: Mapping[str, Any],
        *,
        active: bool,
        durable_receipt: bool,
    ) -> Path:
        safe_id = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
        path = self.checkpoints / f"{safe_id}.json"
        record = {
            "schema_version": 1,
            "session_id_sha256": hashlib.sha256(session_id.encode("utf-8")).hexdigest(),
            "updated_at": _iso(),
            "active": bool(active),
            "durable_receipt": bool(durable_receipt),
            "payload": dict(payload),
        }
        _atomic_bytes(
            path,
            (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        )
        return path

    def mark_checkpoint_receipt(self, path: Path) -> bool:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        payload["durable_receipt"] = True
        payload["active"] = False
        payload["updated_at"] = _iso()
        _atomic_bytes(
            path,
            (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        )
        return True

    def gc_checkpoints(
        self,
        *,
        ttl_seconds: int,
        max_total_bytes: int,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = now or _now()
        cutoff = current - timedelta(seconds=max(0, ttl_seconds))
        records: list[tuple[Path, dict[str, Any], int, datetime]] = []
        try:
            paths = list(self.checkpoints.glob("*.json"))
        except OSError:
            paths = []
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                size = path.stat().st_size
                updated = datetime.fromisoformat(str(payload.get("updated_at") or ""))
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=UTC)
            except (OSError, json.JSONDecodeError, ValueError, AttributeError):
                continue
            records.append((path, payload, size, updated))

        removed: list[str] = []
        protected: list[str] = []

        def eligible(payload: Mapping[str, Any]) -> bool:
            return payload.get("active") is not True and payload.get("durable_receipt") is True

        for path, payload, _size, updated in records:
            if updated < cutoff and eligible(payload):
                path.unlink(missing_ok=True)
                removed.append(path.name)
            elif not eligible(payload):
                protected.append(path.name)
        survivors = [row for row in records if row[0].exists()]
        total = sum(row[2] for row in survivors)
        for path, payload, size, _updated in sorted(survivors, key=lambda row: row[3]):
            if total <= max(0, max_total_bytes):
                break
            if not eligible(payload):
                continue
            path.unlink(missing_ok=True)
            removed.append(path.name)
            total -= size
        return {
            "status": "ok",
            "removed": removed,
            "protected": sorted(protected),
            "remaining_bytes": total,
            "max_total_bytes": max(0, max_total_bytes),
            "converged": total <= max(0, max_total_bytes),
        }


def compact_event_context(
    events: Iterable[Mapping[str, Any]],
    *,
    max_chars: int = 12_000,
) -> dict[str, Any]:
    """Compact complete action/observation pairs or fall back to full history."""

    rows = [dict(event) for event in events]
    pending: dict[tuple[int, int], dict[str, Any]] = {}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        key = (int(row.get("epoch") or 0), int(row.get("iteration") or 0))
        kind = row.get("kind")
        if kind == "action":
            pending[key] = row
        elif kind == "observation" and key in pending:
            pairs.append((pending.pop(key), row))
    if pending:
        return {"status": "full_history", "events": rows, "reason": "unpaired_action"}
    selected: list[dict[str, Any]] = []
    used = 0
    for action, observation in reversed(pairs):
        encoded = json.dumps([action, observation], ensure_ascii=False, default=str)
        if selected and used + len(encoded) > max_chars:
            break
        selected[0:0] = [action, observation]
        used += len(encoded)
    return {
        "status": "compacted",
        "events": selected,
        "dropped_pairs": max(0, len(pairs) - len(selected) // 2),
        "chars": used,
    }


def reduce_events(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Rebuild the minimal run state solely from the append-only ledger."""

    actions: dict[tuple[int, int], dict[str, Any]] = {}
    observations: dict[tuple[int, int], dict[str, Any]] = {}
    stop: dict[str, Any] | None = None
    max_epoch = 0
    for source in events:
        row = dict(source)
        epoch = int(row.get("epoch") or 0)
        iteration = int(row.get("iteration") or 0)
        max_epoch = max(max_epoch, epoch)
        key = (epoch, iteration)
        if row.get("kind") == "action":
            actions[key] = row
        elif row.get("kind") == "observation":
            observations[key] = row
        elif row.get("kind") == "stop":
            stop = row
    orphan_actions = [
        {"epoch": key[0], "iteration": key[1], "action": row.get("action")}
        for key, row in sorted(actions.items())
        if key not in observations
    ]
    return {
        "epoch": max_epoch,
        "actions": len(actions),
        "observations": len(observations),
        "orphan_actions": orphan_actions,
        "terminal": stop is not None,
        "stop_reason": str((stop or {}).get("stop_reason") or ""),
    }
