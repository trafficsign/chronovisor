"""Data and event transport for the Synaptic Cortex dashboard view."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
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
        pull_log: Path | None = None,
        activity_log: Path | None = None,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.pull_log = pull_log or self.root / "runtime" / "recall" / "pull-log.jsonl"
        self.activity_log = activity_log or self.root / "log.md"
        self.raw_dir = self.root / "raw"
        self._offsets = {
            self.pull_log: self._file_size(self.pull_log),
            self.activity_log: self._file_size(self.activity_log),
        }
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
        }

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
                    self._event("recall", [str(row["page_id"])], "RECALL read")
                )
            elif event_type == "search":
                page_ids = [
                    str(page_id)
                    for page_id in [
                        *(row.get("direct_pages") or []),
                        *(row.get("expanded_pages") or []),
                    ]
                    if page_id
                ]
                events.append(self._event("recall", page_ids, "RECALL search"))
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
        return [
            *self._pull_events(),
            *self._save_events(),
            *self._ingest_events(),
        ]
