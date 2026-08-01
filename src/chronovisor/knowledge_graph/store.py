"""Durable append-only relation ledger and sealed materialized snapshots."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import (
    DurableStateError,
    file_lock,
    read_sealed_json,
    write_sealed_json,
)
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.knowledge_graph.schema import (
    RELATION_ACTIONS,
    SCHEMA_VERSION,
    RelationRecord,
    canonical_json,
    sha256,
    validate_event,
)


class KnowledgeGraphStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or CHRONOVISOR_ROOT / "knowledge-graph"
        self.events_file = self.root / "relation-events.jsonl"
        self.snapshot_file = self.root / "relation-snapshot.json"
        self.entity_snapshot_file = self.root / "entity-snapshot.json"
        self.community_snapshot_file = self.root / "community-snapshot.json"
        self.builder_state_file = self.root / "builder-state.json"
        self.community_summary_state_file = self.root / "community-summary-state.json"
        self.lock_file = self.root / "store.lock"

    def _ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)

    def read_events(self, *, recover_tail: bool = False) -> list[dict[str, Any]]:
        try:
            lines = self.events_file.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        rows: list[dict[str, Any]] = []
        prior = ""
        valid_lines: list[str] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                validate_event(row)
                if row["previous_hash"] != prior:
                    raise ValueError("relation event chain mismatch")
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                if not recover_tail:
                    raise DurableStateError("corrupt relation event chain") from exc
                break
            rows.append(row)
            prior = row["event_hash"]
            valid_lines.append(canonical_json(row))
        if recover_tail and len(valid_lines) != len(
            [line for line in lines if line.strip()]
        ):
            self._atomic_replace("\n".join(valid_lines) + ("\n" if valid_lines else ""))
        return rows

    def _atomic_replace(self, content: str) -> None:
        self._ensure()
        fd, name = tempfile.mkstemp(prefix=".relation-events.", dir=self.root)
        path = Path(name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(path, self.events_file)
            if self.events_file.read_text(encoding="utf-8") != content:
                raise DurableStateError("relation ledger read-back mismatch")
        finally:
            path.unlink(missing_ok=True)

    def append(
        self,
        relation: RelationRecord,
        *,
        action: str,
        reason_code: str = "",
        created_at: str | None = None,
    ) -> dict[str, Any]:
        if action not in RELATION_ACTIONS:
            raise ValueError("invalid relation action")
        relation.validate()
        self._ensure()
        with file_lock(self.lock_file):
            rows = self.read_events(recover_tail=True)
            duplicate = next(
                (
                    row
                    for row in reversed(rows)
                    if row["action"] == action
                    and row["relation"]["relation_id"] == relation.relation_id
                    and sha256(row["relation"]) == sha256(relation.to_dict())
                ),
                None,
            )
            if duplicate is not None:
                return {**duplicate, "idempotent": True}
            unsigned = {
                "schema_version": SCHEMA_VERSION,
                "event_id": uuid.uuid4().hex,
                "previous_hash": rows[-1]["event_hash"] if rows else "",
                "action": action,
                "created_at": created_at
                or datetime.now(UTC).isoformat(timespec="seconds"),
                "relation": relation.to_dict(),
                "reason_code": reason_code[:160],
            }
            event = {**unsigned, "event_hash": sha256(unsigned)}
            validate_event(event)
            fd = os.open(
                self.events_file, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600
            )
            try:
                with os.fdopen(fd, "a", encoding="utf-8") as stream:
                    stream.write(canonical_json(event) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                pass
            observed = self.read_events()[-1]
            if observed["event_hash"] != event["event_hash"]:
                raise DurableStateError("relation append read-back mismatch")
            self.materialize_snapshot(rows=[*rows, event])
            return event

    def materialize_snapshot(
        self, *, rows: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        events = self.read_events() if rows is None else rows
        latest: dict[str, dict[str, Any]] = {}
        for event in events:
            latest[event["relation"]["relation_id"]] = event["relation"]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "event_count": len(events),
            "head_hash": events[-1]["event_hash"] if events else "",
            "relations": dict(sorted(latest.items())),
        }
        return write_sealed_json(self.snapshot_file, payload)

    def load_snapshot(self) -> dict[str, Any]:
        try:
            return read_sealed_json(self.snapshot_file, recover_backup=True)
        except DurableStateError:
            return self.materialize_snapshot()

    def relations(
        self,
        *,
        statuses: Iterable[str] | None = None,
    ) -> list[RelationRecord]:
        allowed = set(statuses) if statuses is not None else None
        snapshot = self.load_snapshot()
        values = snapshot.get("relations")
        if not isinstance(values, dict):
            return []
        records = [RelationRecord.from_dict(row) for row in values.values()]
        return sorted(
            (row for row in records if allowed is None or row.status in allowed),
            key=lambda row: row.relation_id,
        )

    def write_derived_snapshot(
        self, name: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        targets = {
            "entities": self.entity_snapshot_file,
            "communities": self.community_snapshot_file,
            "builder": self.builder_state_file,
            "community_summary": self.community_summary_state_file,
        }
        if name not in targets:
            raise ValueError("unknown derived snapshot")
        self._ensure()
        return write_sealed_json(targets[name], payload)
