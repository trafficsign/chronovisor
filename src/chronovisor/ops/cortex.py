"""Page-link graph and capture events for Synaptic Cortex."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from chronovisor.core.canonical_document import (
    Namespace,
    parse_document,
    resolve_internal_markdown_links,
)
from chronovisor.core.raw_segment import RawSegmentCommit, RawSegmentCorrupt
from chronovisor.core.store import canonical_document_paths

_GRAPH_CACHE_LOCK = threading.Lock()
_GRAPH_CACHE: dict[str, dict[str, Any]] = {}
_ENTRYPOINT_PAGES = {"claude-code", "current-state", "lessons-learned", "user-profile"}


@dataclass(frozen=True)
class _Page:
    path: Path
    page_id: str
    canonical_key: str
    category: str
    title: str
    updated: str
    tags: tuple[str, ...]
    line_count: int
    byte_count: int
    targets: tuple[str, ...]


def _page_sources(root: Path) -> list[tuple[Path, str]]:
    return [
        (path, "system" if root / "system" in path.parents else "pages")
        for path in canonical_document_paths(
            root / "pages",
            system_dir=root / "system",
            require_stable=True,
        )
    ]


def _string_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item).strip())
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()


def _read_page(path: Path, source_kind: str, root: Path) -> _Page | None:
    try:
        raw = path.read_bytes()
        content = raw.decode("utf-8")
        document = parse_document(raw)
    except OSError, UnicodeDecodeError, ValueError:
        return None
    metadata = document.metadata
    namespace: Namespace = "system" if source_kind == "system" else "pages"
    namespace_root = root / namespace
    relative_path = path.relative_to(namespace_root).as_posix()
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
        canonical_key=(
            f"{namespace}/{PurePosixPath(relative_path).with_suffix('').as_posix()}"
        ),
        category=category,
        title=title,
        updated=updated,
        tags=_string_list(metadata.get("tags")),
        line_count=max(1, content.count("\n") + 1),
        byte_count=len(content.encode("utf-8")),
        targets=tuple(
            f"{link.namespace}/{PurePosixPath(link.path).with_suffix('').as_posix()}"
            for link in resolve_internal_markdown_links(
                document.body,
                source_namespace=namespace,
                source_path=relative_path,
            )
        ),
    )


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
        key = page.canonical_key.casefold()
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
            target_index = index_by_key.get(target.casefold())
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

    try:
        backlinks = json.loads((root / ".index/backlinks.json").read_bytes()).get(
            "edges", {}
        )
    except OSError, ValueError, AttributeError:
        backlinks = {}
    unique_ids: dict[str, int] = {}
    id_counts = Counter(page.page_id for page in pages)
    for index, page in enumerate(pages):
        if id_counts[page.page_id] == 1:
            unique_ids[page.page_id] = index
    if isinstance(backlinks, dict):
        for target, refs in backlinks.items():
            target_index = unique_ids.get(target)
            if target_index is None or not isinstance(refs, list):
                continue
            for source in refs:
                source_index = (
                    unique_ids.get(source) if isinstance(source, str) else None
                )
                if (
                    source_index is None
                    or source_index == target_index
                    or (source_index, target_index) in edge_keys
                ):
                    continue
                edge_keys.add((source_index, target_index))
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
    """Build frozen pages once per process, without writing or trusting old graph caches."""
    root = root.expanduser().resolve()
    with _GRAPH_CACHE_LOCK:
        if use_cache and str(root) in _GRAPH_CACHE:
            return _GRAPH_CACHE[str(root)]
        graph = _build_graph(
            root,
            _page_sources(root),
            commit=commit,
            generated=generated
            or datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        if use_cache:
            _GRAPH_CACHE[str(root)] = graph
        return graph


class CortexEventCursor:
    """Tail new raw captures without reading retired subsystem telemetry."""

    def __init__(
        self,
        root: Path,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.raw_dir = self.root / "raw"
        self._offsets: dict[Path, int] = {}
        self._remainders: dict[Path, bytes] = {}
        self._raw_commit_files: dict[Path, tuple[int, int]] = {}
        self._known_raw_commit_files: set[tuple[int, int]] = set()
        self._raw_commit_rebaselines: set[Path] = set()
        self._seen_raw_commit_ids: set[str] = set()
        for path in self._raw_commit_paths():
            identity = self._file_identity(path)
            if identity is None:
                continue
            self._offsets[path] = self._file_size(path)
            self._raw_commit_files[path] = identity
            self._known_raw_commit_files.add(identity)
        self._raw_snapshot = self._raw_file_snapshot()

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    @staticmethod
    def _file_identity(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        return stat.st_dev, stat.st_ino

    def _tail_lines(self, path: Path) -> list[str]:
        descriptor = -1
        try:
            flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(path, flags)
            snapshot = os.fstat(descriptor)
            if not stat.S_ISREG(snapshot.st_mode):
                return []
            size = snapshot.st_size
            offset = self._offsets.get(path, 0)
            if size < offset:
                offset = 0
                self._remainders.pop(path, None)
            if size == offset:
                return []
            os.lseek(descriptor, offset, os.SEEK_SET)
            chunk = os.read(descriptor, min(size - offset, 1024 * 1024))
        except OSError:
            return []
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        self._offsets[path] = offset + len(chunk)
        data = self._remainders.pop(path, b"") + chunk
        if data and not data.endswith(b"\n"):
            data, remainder = data.rsplit(b"\n", 1) if b"\n" in data else (b"", data)
            self._remainders[path] = remainder
        return data.decode("utf-8", errors="replace").splitlines()

    def _raw_file_snapshot(self) -> dict[str, tuple[int, int]]:
        snapshot: dict[str, tuple[int, int]] = {}
        try:
            paths = list(self.raw_dir.glob("*.md")) + list(
                self.raw_dir.glob("[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/*.md")
            )
        except OSError:
            return snapshot
        for path in paths:
            if path.is_symlink():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[path.relative_to(self.raw_dir).as_posix()] = (
                stat.st_size,
                stat.st_mtime_ns,
            )
        return snapshot

    def _raw_commit_paths(self) -> list[Path]:
        pattern = "[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/*.commits.jsonl"
        try:
            return sorted(self.raw_dir.glob(pattern))
        except OSError:
            return []

    def _raw_segment_commits(self) -> list[RawSegmentCommit]:
        commits: list[RawSegmentCommit] = []
        for path in self._raw_commit_paths():
            identity = self._file_identity(path)
            if identity is None:
                continue
            size = self._file_size(path)
            previous_identity = self._raw_commit_files.get(path)
            if previous_identity is None:
                self._offsets[path] = (
                    size if identity in self._known_raw_commit_files else 0
                )
                self._raw_commit_files[path] = identity
                self._known_raw_commit_files.add(identity)
            elif previous_identity != identity:
                self._offsets[path] = size
                self._remainders.pop(path, None)
                self._raw_commit_files[path] = identity
                self._known_raw_commit_files.add(identity)
                self._raw_commit_rebaselines.discard(path)
                continue
            elif size < self._offsets.get(path, 0):
                # A same-inode truncation may be followed by historical journal
                # restoration. Baseline both the shrink and its first regrowth.
                self._offsets[path] = size
                self._remainders.pop(path, None)
                self._raw_commit_rebaselines.add(path)
                continue
            elif path in self._raw_commit_rebaselines:
                if size > self._offsets.get(path, 0):
                    self._offsets[path] = size
                    self._remainders.pop(path, None)
                    self._raw_commit_rebaselines.discard(path)
                continue
            for line in self._tail_lines(path):
                try:
                    commit = RawSegmentCommit.from_dict(json.loads(line))
                except (
                    json.JSONDecodeError,
                    RawSegmentCorrupt,
                    TypeError,
                    ValueError,
                ):
                    continue
                if commit.raw_id in self._seen_raw_commit_ids:
                    continue
                self._seen_raw_commit_ids.add(commit.raw_id)
                commits.append(commit)
        return commits

    def _save_events(self) -> list[dict[str, Any]]:
        commits = self._raw_segment_commits()
        snapshot = self._raw_file_snapshot()
        changed = [
            name
            for name, identity in snapshot.items()
            if self._raw_snapshot.get(name) != identity
        ]
        self._raw_snapshot = snapshot
        if not commits and not changed:
            return []
        changed.sort(key=lambda name: (snapshot[name][1], name), reverse=True)
        byte_count = sum(commit.length for commit in commits) + sum(
            snapshot[name][0] for name in changed
        )
        identities = [f"raw:{commit.raw_id}:{commit.sha256}" for commit in commits] + [
            f"legacy:{name}:{snapshot[name][0]}:{snapshot[name][1]}" for name in changed
        ]
        capture_id = hashlib.sha256(
            "\n".join(sorted(identities)).encode("utf-8")
        ).hexdigest()[:12]
        details: dict[str, str | int] = {
            "phase": "capture",
            "capture_id": capture_id,
            "byte_count": byte_count,
            "raw_count": len(commits) + len(changed),
        }
        if commits:
            newest_commit = max(
                commits,
                key=lambda commit: (commit.captured_at, commit.raw_id),
            )
            details["raw_id"] = newest_commit.raw_id
        if changed:
            details["file_name"] = changed[0]
        if commits and changed:
            origin = "raw-journal+raw-snapshot"
        elif commits:
            origin = "raw-journal"
        else:
            origin = "raw-snapshot"
        return [
            self._event(
                "save",
                [],
                f"CAPTURED · {byte_count} B · ID {capture_id}",
                origin=origin,
                **details,
            )
        ]

    @staticmethod
    def _event(
        kind: str, page_ids: list[Any], label: str, *, origin: str, **details: str | int
    ) -> dict[str, Any]:
        return {
            "schema": "chronovisor.cortex.event.v2",
            "kind": kind,
            "page_ids": page_ids,
            "label": label[:160],
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": "telemetry-fallback",
            "origin": origin,
            **details,
        }

    def poll_payload(self) -> dict[str, Any]:
        return {"type": "events", "events": self._save_events()}

    def poll(self) -> list[dict[str, Any]]:
        return self.poll_payload()["events"]
