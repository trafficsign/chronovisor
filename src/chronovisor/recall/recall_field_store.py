"""Sealed sparse snapshots and ordered per-session Recall Field events."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import exclusive_text_file_lock
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.recall.recall_field_schema import (
    FieldEvent,
    RecallFieldConfig,
    RecallFieldState,
)

FIELD_ROOT = CHRONOVISOR_ROOT / "recall" / "field"
SESSION_ROOT = FIELD_ROOT / "sessions"
EVENT_ROOT = FIELD_ROOT / "events"


class RecallFieldStore:
    def __init__(
        self,
        root: Path = FIELD_ROOT,
        *,
        config: RecallFieldConfig | None = None,
    ) -> None:
        self.root = root
        self.session_root = root / "sessions"
        self.event_root = root / "events"
        self.config = config or RecallFieldConfig()

    def _state_path(self, session_hash: str) -> Path:
        return self.session_root / f"{session_hash}.json"

    def _event_path(self, session_hash: str) -> Path:
        return self.event_root / f"{session_hash}.jsonl"

    def _lock_path(self, session_hash: str) -> Path:
        return self.session_root / f"{session_hash}.lock"

    @staticmethod
    def _sealed_payload(state: RecallFieldState) -> dict[str, Any]:
        payload = state.to_dict()
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            **payload,
            "snapshot_sha256": hashlib.sha256(encoded).hexdigest(),
        }

    @staticmethod
    def _verify_snapshot(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("field snapshot must be an object")
        seal = value.get("snapshot_sha256")
        payload = {key: item for key, item in value.items() if key != "snapshot_sha256"}
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if not isinstance(seal, str) or seal != hashlib.sha256(encoded).hexdigest():
            raise ValueError("field snapshot seal mismatch")
        return payload

    def _new_state(self, session_hash: str, now: float) -> RecallFieldState:
        return RecallFieldState(
            session_hash=session_hash,
            created_at_epoch=now,
            updated_at_epoch=now,
        )

    def _load_unlocked(self, session_hash: str, *, now: float) -> RecallFieldState:
        path = self._state_path(session_hash)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            payload = self._verify_snapshot(value)
            return RecallFieldState.from_dict(payload, session_hash=session_hash)
        except FileNotFoundError:
            return self._new_state(session_hash, now)
        except (OSError, json.JSONDecodeError, ValueError):
            if path.exists():
                corrupt = path.with_name(
                    f"{path.stem}.corrupt-{int(now * 1000)}{path.suffix}"
                )
                try:
                    path.replace(corrupt)
                except OSError:
                    pass
            return self._new_state(session_hash, now)

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        tmp = Path(name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def _save_unlocked(self, state: RecallFieldState) -> None:
        self._atomic_write(
            self._state_path(state.session_hash),
            json.dumps(
                self._sealed_payload(state),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    def _append_events_unlocked(
        self,
        session_hash: str,
        events: list[FieldEvent],
    ) -> None:
        if not events:
            return
        path = self._event_path(session_hash)
        try:
            rows = [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError:
            rows = []
        rows.extend(
            json.dumps(
                event.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for event in events
        )
        rows = rows[-self.config.event_retention :]
        self._atomic_write(path, "\n".join(rows) + "\n")

    def load(self, session_hash: str, *, now: float | None = None) -> RecallFieldState:
        observed = time.time() if now is None else now
        with exclusive_text_file_lock(self._lock_path(session_hash)):
            return self._load_unlocked(session_hash, now=observed)

    def transact(
        self,
        session_hash: str,
        mutator: Callable[
            [RecallFieldState],
            tuple[RecallFieldState, list[FieldEvent]],
        ],
        *,
        now: float | None = None,
    ) -> tuple[RecallFieldState, list[FieldEvent]]:
        observed = time.time() if now is None else now
        with exclusive_text_file_lock(self._lock_path(session_hash)):
            state = self._load_unlocked(session_hash, now=observed)
            updated, events = mutator(state)
            self._save_unlocked(updated)
            self._append_events_unlocked(session_hash, events)
            return updated, events

    def read_events(
        self,
        session_hash: str,
        *,
        after_seq: int = 0,
    ) -> list[dict[str, Any]]:
        path = self._event_path(session_hash)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        events: list[dict[str, Any]] = []
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(row, dict)
                and isinstance(row.get("seq"), int)
                and row["seq"] > after_seq
            ):
                events.append(row)
        return sorted(events, key=lambda row: int(row["seq"]))

    def latest_session_hash(
        self,
        *,
        host: str = "",
        max_age_seconds: float = 180.0,
        now: float | None = None,
    ) -> str:
        """Return the freshest valid Field session for an MCP client host."""

        observed = time.time() if now is None else now
        normalized_host = host.strip().casefold()
        try:
            paths = sorted(
                self.session_root.glob("*.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return ""
        for path in paths[:24]:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                payload = self._verify_snapshot(value)
                state = RecallFieldState.from_dict(payload, session_hash=path.stem)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            if normalized_host and state.host != normalized_host:
                continue
            age = max(0.0, observed - state.updated_at_epoch)
            if age <= max(0.0, max_age_seconds):
                return state.session_hash
        return ""

    def cleanup(self, *, now: float | None = None) -> int:
        observed = time.time() if now is None else now
        cutoff = observed - self.config.session_ttl_seconds
        removed = 0
        if not self.session_root.exists():
            return 0
        for path in self.session_root.glob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                updated = float(value.get("updated_at_epoch") or path.stat().st_mtime)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                updated = path.stat().st_mtime
            if updated >= cutoff:
                continue
            session_hash = path.stem
            with exclusive_text_file_lock(self._lock_path(session_hash)):
                path.unlink(missing_ok=True)
                self._event_path(session_hash).unlink(missing_ok=True)
            removed += 1
        return removed
