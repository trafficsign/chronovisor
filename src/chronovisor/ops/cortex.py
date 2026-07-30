"""Data and event transport for the Synaptic Cortex dashboard view."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.core.frontmatter import parse as parse_frontmatter
from chronovisor.core.link_fix import extract_targets

_GRAPH_CACHE_LOCK = threading.Lock()
_GRAPH_CACHE: dict[str, dict[str, Any]] = {}
_WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_INGEST_PAGE_RE = re.compile(
    r"(?:^|\s)ingest \| (?:created|updated) (?P<page>[^\r\n]+)$"
)
_ENTRYPOINT_PAGES = {
    "claude-code",
    "current-state",
    "lessons-learned",
    "user-profile",
}
_FIELD_SESSION_RE = re.compile(r"^[0-9a-f]{16}$")
_FIELD_EVENT_KEYS = {
    "seq",
    "timestamp_epoch",
    "session_hash",
    "topic_epoch",
    "kind",
    "page_id",
    "source_page_id",
    "target_page_id",
    "edge_type",
    "delta",
    "activation",
    "reason_code",
    "certificate_id",
    "components",
}
_FIELD_COMPONENT_KEYS = {"direct", "spread", "negative", "inhibition"}


@dataclass(frozen=True)
class _Page:
    path: Path
    page_id: str
    category: str
    title: str
    updated: str
    tags: tuple[str, ...]
    line_count: int
    byte_count: int
    targets: tuple[str, ...]


def _page_sources(root: Path) -> list[tuple[Path, str]]:
    sources: list[tuple[Path, str]] = []
    pages_dir = root / "pages"
    system_dir = root / "system"
    if pages_dir.exists():
        sources.extend((path, "pages") for path in pages_dir.rglob("*.md"))
    if system_dir.exists():
        sources.extend((path, "system") for path in system_dir.glob("*.md"))
    return sorted(sources, key=lambda item: str(item[0]))


def _source_fingerprint(
    root: Path,
    sources: list[tuple[Path, str]],
    *,
    commit: str,
) -> str:
    rows: list[tuple[str, int, int]] = []
    for path, _source_kind in sources:
        try:
            stat = path.stat()
        except OSError:
            continue
        rows.append(
            (
                str(path.relative_to(root)),
                stat.st_size,
                stat.st_mtime_ns,
            )
        )
    encoded = json.dumps(
        {"commit": commit, "sources": rows},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _string_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item).strip())
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()


def _read_page(path: Path, source_kind: str, root: Path) -> _Page | None:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    metadata, _body = parse_frontmatter(content)
    if source_kind == "system":
        category = "system"
    else:
        relative = path.relative_to(root / "pages")
        category = relative.parts[0] if len(relative.parts) > 1 else "root"
    page_id = path.stem
    title = str(metadata.get("title") or page_id)
    updated = str(metadata.get("updated") or "")
    return _Page(
        path=path,
        page_id=page_id,
        category=category,
        title=title,
        updated=updated,
        tags=_string_list(metadata.get("tags")),
        line_count=max(1, content.count("\n") + 1),
        byte_count=len(content.encode("utf-8")),
        targets=tuple(extract_targets(content, strip=True)),
    )


def _target_key(value: str) -> str:
    normalized = value.strip().removesuffix(".md")
    return normalized.rsplit("/", 1)[-1].casefold()


def _build_graph(
    root: Path,
    sources: list[tuple[Path, str]],
    *,
    commit: str,
    generated: str,
) -> dict[str, Any]:
    pages = [
        page
        for path, source_kind in sources
        if (page := _read_page(path, source_kind, root)) is not None
    ]
    pages.sort(key=lambda page: (page.category.casefold(), page.page_id.casefold()))

    index_by_key: dict[str, int] = {}
    ambiguous_keys: set[str] = set()
    for index, page in enumerate(pages):
        key = page.page_id.casefold()
        if key in index_by_key:
            ambiguous_keys.add(key)
        else:
            index_by_key[key] = index
    for key in ambiguous_keys:
        index_by_key.pop(key, None)

    edges: list[list[int]] = []
    edge_keys: set[tuple[int, int]] = set()
    unresolved = 0
    fan_in = [0] * len(pages)
    fan_out = [0] * len(pages)
    for source_index, page in enumerate(pages):
        for target in page.targets:
            target_index = index_by_key.get(_target_key(target))
            if target_index is None:
                unresolved += 1
                continue
            if target_index == source_index:
                continue
            edge_key = (source_index, target_index)
            if edge_key in edge_keys:
                continue
            edge_keys.add(edge_key)
            edges.append([source_index, target_index, 0])
            fan_out[source_index] += 1
            fan_in[target_index] += 1

    category_counts: dict[str, int] = {}
    for page in pages:
        category_counts[page.category] = category_counts.get(page.category, 0) + 1
    categories = [
        {"id": category, "count": count}
        for category, count in sorted(
            category_counts.items(),
            key=lambda item: (-item[1], item[0].casefold()),
        )
    ]
    nodes = [
        {
            "id": page.page_id,
            "pkg": page.category,
            "l": page.line_count,
            "b": page.byte_count,
            "fi": fan_in[index],
            "fo": fan_out[index],
            "ep": int(page.page_id in _ENTRYPOINT_PAGES),
            "title": page.title,
            "updated": page.updated,
            "tags": list(page.tags),
        }
        for index, page in enumerate(pages)
    ]
    short_commit = commit[:7] if commit else "local"
    return {
        "meta": {
            "generated": generated,
            "commit": short_commit,
            "totalLines": sum(page.line_count for page in pages),
            "static": len(edges),
            "deferred": unresolved,
            "spawn": 0,
            "entrypoints": sum(node["ep"] for node in nodes),
            "source": "local-wiki",
        },
        "nodes": nodes,
        "links": edges,
        "categories": categories,
    }


def build_cortex_graph(
    root: Path,
    *,
    commit: str = "",
    generated: str | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Build the browser-safe Wiki graph without exposing page bodies."""

    resolved_root = root.expanduser().resolve()
    sources = _page_sources(resolved_root)
    fingerprint = _source_fingerprint(resolved_root, sources, commit=commit)
    cache_key = str(resolved_root)
    if use_cache:
        with _GRAPH_CACHE_LOCK:
            cached = _GRAPH_CACHE.get(cache_key)
            if cached and cached.get("fingerprint") == fingerprint:
                return cached["graph"]

    graph = _build_graph(
        resolved_root,
        sources,
        commit=commit,
        generated=generated or datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    if use_cache:
        with _GRAPH_CACHE_LOCK:
            _GRAPH_CACHE[cache_key] = {
                "fingerprint": fingerprint,
                "graph": graph,
            }
    return graph


def _read_sealed_field_snapshot(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("field snapshot must be an object")
    seal = value.get("snapshot_sha256")
    payload = {
        key: item for key, item in value.items() if key != "snapshot_sha256"
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if not isinstance(seal, str) or seal != hashlib.sha256(encoded).hexdigest():
        raise ValueError("field snapshot seal mismatch")
    return payload


def _project_field_event(value: Any) -> dict[str, Any] | None:
    """Return the strict browser-safe subset of one durable Field event."""

    if not isinstance(value, dict):
        return None
    session = value.get("session_hash")
    seq = value.get("seq")
    kind = value.get("kind")
    if (
        not isinstance(session, str)
        or not _FIELD_SESSION_RE.fullmatch(session)
        or not isinstance(seq, int)
        or seq < 1
        or not isinstance(kind, str)
    ):
        return None
    projected = {
        key: value.get(key)
        for key in _FIELD_EVENT_KEYS
        if key in value
    }
    components = value.get("components")
    safe_components = components if isinstance(components, dict) else {}
    projected["components"] = {
        key: round(float(safe_components.get(key) or 0.0), 6)
        for key in sorted(_FIELD_COMPONENT_KEYS)
        if isinstance(safe_components.get(key, 0.0), int | float)
    }
    projected["source"] = "stateful-recall-field"
    return projected


def _read_field_events(
    field_root: Path,
    session_hash: str,
    *,
    limit: int = 256,
) -> list[dict[str, Any]]:
    if not _FIELD_SESSION_RE.fullmatch(session_hash):
        return []
    path = field_root / "events" / f"{session_hash}.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in lines[-max(1, limit) :]:
        try:
            projected = _project_field_event(json.loads(line))
        except json.JSONDecodeError:
            continue
        if projected is not None and projected["session_hash"] == session_hash:
            events.append(projected)
    return sorted(events, key=lambda row: int(row["seq"]))


def _field_recall_metrics(
    root: Path,
    session_hash: str,
    *,
    limit: int = 400,
) -> dict[str, Any]:
    """Aggregate Field latency and teacher agreement without exposing prompts."""

    path = root / "recall" / "recall-log.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    latencies: list[float] = []
    teacher_total = 0
    teacher_agreed = 0
    for line in lines[-max(1, limit) :]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        features = row.get("evidence_features")
        field = features.get("field_shadow") if isinstance(features, dict) else None
        if not isinstance(field, dict) or field.get("session_hash") != session_hash:
            continue
        latency = field.get("latency_ms")
        if isinstance(latency, int | float) and latency >= 0:
            latencies.append(float(latency))
        candidates = {
            str(page_id)
            for page_id in field.get("candidate_page_ids") or []
            if isinstance(page_id, str)
        }
        pages = {
            str(page_id)
            for page_id in row.get("pages") or []
            if isinstance(page_id, str)
        }
        if pages:
            teacher_total += len(pages)
            teacher_agreed += len(pages & candidates)
    latencies.sort()

    def percentile(fraction: float) -> float | None:
        if not latencies:
            return None
        index = min(len(latencies) - 1, round((len(latencies) - 1) * fraction))
        return round(latencies[index], 1)

    return {
        "samples": len(latencies),
        "latency_ms": {
            "p50": percentile(0.5),
            "p95": percentile(0.95),
            "max": round(max(latencies), 1) if latencies else None,
        },
        "teacher_agreement": (
            round(teacher_agreed / teacher_total, 4)
            if teacher_total
            else None
        ),
        "teacher_pages": teacher_total,
    }


def build_cortex_field_projection(
    root: Path,
    *,
    session_hash: str = "",
    now: float | None = None,
    event_limit: int = 256,
) -> dict[str, Any]:
    """Build a browser-safe projection of recent Stateful Recall Field state."""

    from chronovisor.recall.recall_field_schema import load_recall_field_config

    observed = time.time() if now is None else now
    field_root = root.expanduser().resolve() / "recall" / "field"
    session_root = field_root / "sessions"
    config = load_recall_field_config()
    sessions: list[tuple[dict[str, Any], dict[str, Any]]] = []
    corrupt_snapshots = 0
    try:
        paths = sorted(
            session_root.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        paths = []
    for path in paths[:24]:
        if not _FIELD_SESSION_RE.fullmatch(path.stem):
            continue
        try:
            payload = _read_sealed_field_snapshot(path)
        except (OSError, ValueError, json.JSONDecodeError):
            corrupt_snapshots += 1
            continue
        if payload.get("session_hash") != path.stem:
            corrupt_snapshots += 1
            continue
        mode = config.mode
        buffer_name = "shadow" if mode == "shadow" else "active"
        buffer = payload.get(buffer_name)
        if not isinstance(buffer, dict):
            buffer = {}
        updated = float(payload.get("updated_at_epoch") or 0.0)
        sessions.append(
            (
                {
                    "session_hash": path.stem,
                    "updated_at_epoch": updated,
                    "topic_epoch": int(payload.get("topic_epoch") or 0),
                    "turn": int(payload.get("turn") or 0),
                    "seq": int(payload.get("seq") or 0),
                    "mode": mode,
                    "nodes": len(buffer),
                },
                payload,
            )
        )
    requested = session_hash if _FIELD_SESSION_RE.fullmatch(session_hash) else ""
    selected = next(
        (row for row in sessions if row[0]["session_hash"] == requested),
        sessions[0] if sessions else None,
    )
    if selected is None:
        return {
            "status": "fault" if corrupt_snapshots else "offline",
            "source": "stateful-recall-field",
            "mode": config.mode,
            "session_hash": "",
            "sessions": [],
            "snapshot": None,
            "events": [],
            "summary": {
                "active": 0,
                "candidate": 0,
                "commit": 0,
                "reject": 0,
                "teacher_agreement": None,
                "latency_ms": {"p50": None, "p95": None, "max": None},
                "stale": True,
                "corrupt_snapshots": corrupt_snapshots,
            },
        }

    session, payload = selected
    mode = session["mode"]
    buffer_name = "shadow" if mode == "shadow" else "active"
    raw_buffer = payload.get(buffer_name)
    buffer = raw_buffer if isinstance(raw_buffer, dict) else {}
    nodes: list[dict[str, Any]] = []
    for page_id, value in buffer.items():
        if not isinstance(page_id, str) or not isinstance(value, dict):
            continue
        activation = value.get("activation")
        if not isinstance(activation, int | float):
            continue
        components = {
            key: round(float(value.get(key) or 0.0), 6)
            for key in _FIELD_COMPONENT_KEYS
        }
        nodes.append(
            {
                "page_id": page_id,
                "activation": round(float(activation), 6),
                "components": components,
                "last_seq": int(value.get("last_seq") or 0),
            }
        )
    nodes.sort(key=lambda row: (-row["activation"], row["page_id"]))
    nodes = nodes[: config.max_active_nodes]
    events = _read_field_events(
        field_root,
        session["session_hash"],
        limit=event_limit,
    )
    counts: dict[str, int] = {}
    for event in events:
        kind = str(event.get("kind") or "")
        counts[kind] = counts.get(kind, 0) + 1
    metrics = _field_recall_metrics(root, session["session_hash"])
    stale_after_seconds = max(
        60,
        min(600, config.wall_half_life_seconds * 2),
    )
    age_seconds = max(0.0, observed - session["updated_at_epoch"])
    stale = age_seconds > stale_after_seconds
    status = "fault" if corrupt_snapshots else ("stale" if stale else "online")
    return {
        "status": status,
        "source": "stateful-recall-field",
        "mode": mode,
        "session_hash": session["session_hash"],
        "sessions": [row[0] for row in sessions[:12]],
        "snapshot": {
            "session_hash": session["session_hash"],
            "topic_epoch": session["topic_epoch"],
            "turn": session["turn"],
            "seq": session["seq"],
            "updated_at_epoch": session["updated_at_epoch"],
            "full_search_fallback": payload.get("full_search_fallback") is not False,
            "nodes": nodes,
        },
        "events": events,
        "summary": {
            "active": sum(node["activation"] >= 0.05 for node in nodes),
            "candidate": min(len(nodes), config.working_set_size),
            "commit": counts.get("commit_queued", 0)
            + counts.get("commit_applied", 0),
            "reject": counts.get("reject", 0) + counts.get("inhibit", 0),
            "teacher_agreement": metrics["teacher_agreement"],
            "latency_ms": metrics["latency_ms"],
            "stale": stale,
            "age_seconds": round(age_seconds, 1),
            "corrupt_snapshots": corrupt_snapshots,
        },
    }


def websocket_accept(key: str) -> str:
    """Return the RFC 6455 accept token for a validated browser key."""

    try:
        raw = base64.b64decode(key, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid Sec-WebSocket-Key") from exc
    if len(raw) != 16:
        raise ValueError("invalid Sec-WebSocket-Key")
    digest = hashlib.sha1(f"{key}{_WEBSOCKET_GUID}".encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def websocket_text_frame(payload: dict[str, Any]) -> bytes:
    """Encode one unmasked server-to-browser JSON text frame."""

    body = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    size = len(body)
    if size < 126:
        header = bytes((0x81, size))
    elif size <= 0xFFFF:
        header = bytes((0x81, 126)) + size.to_bytes(2, "big")
    else:
        header = bytes((0x81, 127)) + size.to_bytes(8, "big")
    return header + body


class CortexEventCursor:
    """Tail durable Chronovisor telemetry and expose browser-safe firing events."""

    def __init__(
        self,
        root: Path,
        *,
        recall_log: Path | None = None,
        pull_log: Path | None = None,
        activity_log: Path | None = None,
        field_session: str = "",
    ) -> None:
        self.root = root.expanduser().resolve()
        self.recall_log = recall_log or self.root / "recall" / "recall-log.jsonl"
        self.pull_log = pull_log or self.root / "recall" / "pull-log.jsonl"
        self.activity_log = activity_log or self.root / "log.md"
        self.raw_dir = self.root / "raw"
        self.field_session = (
            field_session
            if _FIELD_SESSION_RE.fullmatch(field_session)
            else ""
        )
        self.field_event_log = (
            self.root
            / "recall"
            / "field"
            / "events"
            / f"{self.field_session}.jsonl"
            if self.field_session
            else None
        )
        self._offsets = {
            self.recall_log: self._file_size(self.recall_log),
            self.pull_log: self._file_size(self.pull_log),
            self.activity_log: self._file_size(self.activity_log),
        }
        if self.field_event_log is not None:
            self._offsets[self.field_event_log] = self._file_size(
                self.field_event_log
            )
        self._remainders: dict[Path, bytes] = {}
        self._raw_dir_mtime_ns = self._directory_mtime(self.raw_dir)

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    @staticmethod
    def _directory_mtime(path: Path) -> int:
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return 0

    def _tail_lines(self, path: Path) -> list[str]:
        size = self._file_size(path)
        offset = self._offsets.get(path, 0)
        if size < offset:
            offset = 0
            self._remainders.pop(path, None)
        if size == offset:
            return []
        try:
            with path.open("rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
        except OSError:
            return []
        self._offsets[path] = offset + len(chunk)
        data = self._remainders.pop(path, b"") + chunk
        if data and not data.endswith(b"\n"):
            data, remainder = data.rsplit(b"\n", 1) if b"\n" in data else (b"", data)
            self._remainders[path] = remainder
        return data.decode("utf-8", errors="replace").splitlines()

    @staticmethod
    def _event(kind: str, page_ids: list[str], label: str) -> dict[str, Any]:
        return {
            "kind": kind,
            "page_ids": list(dict.fromkeys(page_ids))[:24],
            "label": label,
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": "telemetry-fallback",
        }

    def _field_events(self) -> list[dict[str, Any]]:
        if self.field_event_log is None:
            return []
        events: list[dict[str, Any]] = []
        for line in self._tail_lines(self.field_event_log):
            try:
                event = _project_field_event(json.loads(line))
            except json.JSONDecodeError:
                continue
            if event is None or event["session_hash"] != self.field_session:
                continue
            events.append(event)
        return events

    def _automatic_recall_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in self._tail_lines(self.recall_log):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            page_ids = [
                str(page_id)
                for page_id in row.get("pages") or []
                if isinstance(page_id, str) and page_id
            ]
            if (
                row.get("stage") != "injected"
                or row.get("status") != "ok"
                or row.get("decision") != "read"
                or not page_ids
            ):
                continue
            events.append(
                self._event(
                    "auto_recall",
                    page_ids,
                    f"AUTO RECALL · {len(page_ids)} page"
                    f"{'' if len(page_ids) == 1 else 's'}",
                )
            )
        return events

    def _pull_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in self._tail_lines(self.pull_log):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_type = row.get("type")
            if event_type == "read" and row.get("page_id"):
                events.append(
                    self._event("read", [str(row["page_id"])], "MCP READ")
                )
            elif event_type == "search":
                page_ids = [
                    str(page_id)
                    for page_id in row.get("direct_pages") or []
                    if page_id
                ]
                if page_ids:
                    events.append(self._event("search", page_ids, "MCP SEARCH"))
            elif event_type == "used":
                page_ids = [
                    str(page_id)
                    for page_id in row.get("page_ids") or []
                    if page_id
                ]
                if page_ids:
                    events.append(
                        self._event("used", page_ids, "RECALL USED")
                    )
        return events

    def _save_events(self) -> list[dict[str, Any]]:
        current = self._directory_mtime(self.raw_dir)
        if current <= self._raw_dir_mtime_ns:
            return []
        self._raw_dir_mtime_ns = current
        return [self._event("save", [], "SAVE raw capture")]

    def _ingest_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in self._tail_lines(self.activity_log):
            match = _INGEST_PAGE_RE.search(line)
            if not match:
                continue
            page_id = Path(match.group("page").strip()).stem
            events.append(self._event("ingest", [page_id], "INGEST page apply"))
        return events

    def poll(self) -> list[dict[str, Any]]:
        if self.field_session:
            return self._field_events()
        return [
            *self._automatic_recall_events(),
            *self._pull_events(),
            *self._save_events(),
            *self._ingest_events(),
        ]
