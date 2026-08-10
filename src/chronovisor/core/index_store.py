"""Persistent canonical-page metadata and link index.

Keeps derived state in ``~/.chronovisor/.index/`` and refreshes incrementally
based on ``(mtime_ns, size, path, is_system)`` per file.

Backlinks are rebuilt in full on every refresh from the canonical
`pages.outlinks` data, both to keep them consistent with the source of
truth and to preserve scan-order parity with the legacy implementation.

The store is a process-wide singleton accessed via :func:`get_store`.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from chronovisor.core.canonical_document import (
    CanonicalDocumentError,
    Namespace,
    parse_document,
    resolve_internal_markdown_links,
)
from chronovisor.core.link_fix import atomic_write
from chronovisor.core.store import (
    CHRONOVISOR_ROOT,
    PAGES_DIR,
    SYSTEM_DIR,
    okf_runtime_operation,
)

SCHEMA_VERSION = 11  # canonical YAML, paths, links, and lifecycle eligibility
INDEX_DIR = CHRONOVISOR_ROOT / ".index"
PAGES_INDEX_FILE = INDEX_DIR / "pages.json"
BACKLINKS_INDEX_FILE = INDEX_DIR / "backlinks.json"
VALID_LIFECYCLE_STATUSES = frozenset({"draft", "stable", "deprecated"})
PAGE_RESERVED_FILENAMES = frozenset({"index.md", "log.md", "schema.md"})
SYSTEM_RESERVED_FILENAMES = frozenset({"index.md", "log.md"})
VALID_PAGE_TYPES = {
    "knowledge",
    "reference",
    "episodic",
    "semantic",
    "procedural",
    "state",
    "lesson",
    "decision",
}
VALID_SENSITIVITY_TIERS = {"normal", "high"}


def _normalize_lifecycle_status(value: object) -> str:
    if not isinstance(value, str) or value not in VALID_LIFECYCLE_STATUSES:
        raise ValueError("invalid canonical lifecycle status")
    return value


def _normalize_page_type(value: object, *, path: Path | None = None) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in VALID_PAGE_TYPES:
            return normalized
    if path is not None:
        try:
            if path.parent == SYSTEM_DIR:
                if path.stem == "current-state":
                    return "state"
                if path.stem == "lessons-learned":
                    return "lesson"
            if path.parent.name == "car-spec":
                return "reference"
        except OSError:
            pass
    return "knowledge"


def _normalize_sensitivity(value: object, *, path: Path | None = None) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in VALID_SENSITIVITY_TIERS:
            return normalized
    if path is not None:
        try:
            if path.parent.name == "career":
                return "high"
        except OSError:
            pass
    return "normal"


def contained_file(path: Path, root: Path) -> Path | None:
    """Return a regular file only when its real path stays in its namespace."""

    if root.is_symlink():
        return None
    try:
        relative = path.relative_to(root)
        root_resolved = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (OSError, RuntimeError, ValueError):
        return None
    # The final layout is direct: reject leaf and descendant symlinks, including
    # links that currently happen to resolve back inside the namespace.
    if resolved != root_resolved / relative or not resolved.is_file():
        return None
    return resolved


def _read_bytes_stable(path: Path, root: Path, retries: int = 1) -> bytes | None:
    """Read a file and verify (mtime_ns, size) didn't change mid-read.

    Guards against picking up a half-written page (ingest doesn't use
    atomic_write today). On retry exhaustion, returns the last best-effort
    read; callers should treat None as "skip this file".
    """
    last: bytes | None = None
    for _ in range(retries + 1):
        resolved = contained_file(path, root)
        if resolved is None:
            return None
        try:
            st_before = resolved.stat()
            data = resolved.read_bytes()
            st_after = resolved.stat()
        except OSError:
            return None
        if (st_before.st_mtime_ns, st_before.st_size) == (
            st_after.st_mtime_ns,
            st_after.st_size,
        ) and contained_file(path, root) == resolved:
            return data
        last = data
    return last


@dataclass
class PageEntry:
    page_id: str
    path: str  # absolute path string for stable comparison
    is_system: bool
    mtime_ns: int
    size: int
    title: str
    updated: str
    relative_path: str = ""
    uid: str = ""
    classification_primary: str = ""
    classification_notation: str = ""
    classification_status: str = "unclassified"
    outlinks: list[str] = field(
        default_factory=list
    )  # raw, preserves duplicates + order
    raw_keywords: list[str] = field(default_factory=list)
    """Frontmatter ``raw_keywords`` lifted from disk. Internal-only — not
    surfaced via ``meta()`` / ``all_pages_meta()`` yet (that decision is
    deferred to a separate change so the public API stays stable through
    Phase 5). The field is kept here so future readers (search, lint,
    distribution reports) can use it without an extra disk read."""
    tags: list[str] = field(default_factory=list)
    """Frontmatter ``tags`` (Tag Taxonomy v0.1: ``d/`` ``t/`` ``s/`` axes).
    Surfaced via the dedicated ``IndexStore.tags(page_id)`` accessor and
    via ``all_tags()`` for the dedup candidate pool. Stored as the raw
    list from frontmatter; per-tag form validation is the responsibility
    of ``chronovisor_check`` lint, not the index."""
    description: str = ""
    recall_questions: list[str] = field(default_factory=list)
    status: str = "stable"
    superseded_by: str = ""
    page_type: str = "knowledge"
    entities: list[str] = field(default_factory=list)
    sensitivity: str = "normal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "path": self.path,
            "is_system": self.is_system,
            "mtime_ns": self.mtime_ns,
            "size": self.size,
            "title": self.title,
            "updated": self.updated,
            "relative_path": self.relative_path,
            "uid": self.uid,
            "classification_primary": self.classification_primary,
            "classification_notation": self.classification_notation,
            "classification_status": self.classification_status,
            "outlinks": list(self.outlinks),
            "raw_keywords": list(self.raw_keywords),
            "tags": list(self.tags),
            "description": self.description,
            "recall_questions": list(self.recall_questions),
            "status": self.status,
            "superseded_by": self.superseded_by,
            "page_type": self.page_type,
            "entities": list(self.entities),
            "sensitivity": self.sensitivity,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PageEntry:
        # Defensively coerce list-of-string fields: a manually edited
        # cache file or a future-format mismatch shouldn't crash the
        # singleton at startup.
        def _coerce_str_list(value: object) -> list[str]:
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                return list(value)
            return []

        return cls(
            page_id=d["page_id"],
            path=d["path"],
            is_system=bool(d.get("is_system", False)),
            mtime_ns=int(d["mtime_ns"]),
            size=int(d["size"]),
            title=d.get("title", d["page_id"]),
            updated=d.get("updated", "unknown"),
            relative_path=d.get("relative_path", ""),
            uid=d.get("uid", "") if isinstance(d.get("uid", ""), str) else "",
            classification_primary=(
                d.get("classification_primary", "")
                if isinstance(d.get("classification_primary", ""), str)
                else ""
            ),
            classification_notation=(
                d.get("classification_notation", "")
                if isinstance(d.get("classification_notation", ""), str)
                else ""
            ),
            classification_status=(
                d.get("classification_status", "unclassified")
                if isinstance(
                    d.get("classification_status", "unclassified"), str
                )
                else "unclassified"
            ),
            outlinks=list(d.get("outlinks", [])),
            raw_keywords=_coerce_str_list(d.get("raw_keywords")),
            tags=_coerce_str_list(d.get("tags")),
            description=d.get("description", "")
            if isinstance(d.get("description", ""), str)
            else "",
            recall_questions=_coerce_str_list(d.get("recall_questions")),
            status=_normalize_lifecycle_status(d.get("status")),
            superseded_by=d.get("superseded_by", "")
            if isinstance(d.get("superseded_by", ""), str)
            else "",
            page_type=_normalize_page_type(d.get("page_type")),
            entities=_coerce_str_list(d.get("entities")),
            sensitivity=_normalize_sensitivity(d.get("sensitivity")),
        )


class DuplicatePageIdError(RuntimeError):
    """Raised when two files claim the same page_id stem.

    Current code keys by stem everywhere (`page_id_from_path`), so silent
    aliasing would corrupt every lookup. We fail closed instead.
    """


class IndexStore:
    """Persistent page metadata + backlinks index.

    Thread-safe via an internal RLock. All public methods that read
    derived state perform an internal lock acquisition; callers don't
    need to hold a lock externally.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, PageEntry] = {}
        self._page_order: list[str] = []  # rglob scan order, preserved for parity
        self._backlinks: dict[str, list[str]] = {}
        self._tag_pages: dict[str, list[str]] = {}
        self._entity_pages: dict[str, list[str]] = {}
        self._loaded = False
        self._persistence_dirty = False
        self._last_refresh_monotonic = 0.0

    # -- persistence ------------------------------------------------------

    def _load_from_disk(self) -> None:
        if not PAGES_INDEX_FILE.exists() or not BACKLINKS_INDEX_FILE.exists():
            return
        try:
            pages_doc = json.loads(PAGES_INDEX_FILE.read_text())
            backlinks_doc = json.loads(BACKLINKS_INDEX_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if pages_doc.get("schema_version") != SCHEMA_VERSION:
            return
        if backlinks_doc.get("schema_version") != SCHEMA_VERSION:
            return
        # Cross-file integrity: matching generation IDs.
        # Mismatch => one file is stale; rebuild from scratch on next refresh.
        if pages_doc.get("generation") != backlinks_doc.get("generation"):
            return
        try:
            entries = {
                pid: PageEntry.from_dict(d)
                for pid, d in pages_doc.get("entries", {}).items()
            }
            order = list(pages_doc.get("page_order", []))
            backlinks = {
                pid: list(refs) for pid, refs in backlinks_doc.get("edges", {}).items()
            }
        except (KeyError, TypeError, ValueError):
            return
        self._entries = entries
        self._page_order = [pid for pid in order if pid in entries]
        self._backlinks = backlinks
        self._rebuild_associations()

    def _persist(self, generation: int) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        pages_doc = {
            "schema_version": SCHEMA_VERSION,
            "generation": generation,
            "page_order": list(self._page_order),
            "entries": {pid: e.to_dict() for pid, e in self._entries.items()},
        }
        backlinks_doc = {
            "schema_version": SCHEMA_VERSION,
            "generation": generation,
            "edges": {pid: list(refs) for pid, refs in self._backlinks.items()},
        }
        # atomic_write each file. Cross-file consistency is guarded by the
        # shared `generation` field — readers that see mismatched generations
        # discard both and rebuild.
        atomic_write(PAGES_INDEX_FILE, json.dumps(pages_doc, ensure_ascii=False))
        atomic_write(
            BACKLINKS_INDEX_FILE, json.dumps(backlinks_doc, ensure_ascii=False)
        )

    # -- refresh ----------------------------------------------------------

    @staticmethod
    def _scan_disk() -> list[tuple[str, Path, bool, int, int]]:
        """Walk pages/ and system/ in deterministic order.

        Returns list of (page_id, path, is_system, mtime_ns, size).
        Pages directory comes first to mirror legacy scan order in
        `_find_backlinks` (pages/ then system/).
        """
        out: list[tuple[str, Path, bool, int, int]] = []
        for root, is_system in ((PAGES_DIR, False), (SYSTEM_DIR, True)):
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*.md")):
                reserved = (
                    SYSTEM_RESERVED_FILENAMES
                    if is_system
                    else PAGE_RESERVED_FILENAMES
                )
                if path.name in reserved:
                    continue
                resolved = contained_file(path, root)
                if resolved is None:
                    continue
                try:
                    st = resolved.stat()
                except OSError:
                    continue
                out.append(
                    (resolved.stem, resolved, is_system, st.st_mtime_ns, st.st_size)
                )
        return out

    def refresh(self) -> None:
        """Sync the in-memory index with disk.

        Cheap when nothing changed (one stat per page, no parsing).
        Persists only when entries or backlinks actually changed.
        """
        with okf_runtime_operation(CHRONOVISOR_ROOT):
            self._refresh_locked()

    def _refresh_locked(self) -> None:
        with self._lock:
            if not self._loaded:
                self._load_from_disk()
                self._loaded = True

            current = self._scan_disk()
            seen_ids: dict[str, tuple[Path, bool, int, int]] = {}
            duplicates: list[str] = []
            for pid, path, is_system, mtime_ns, size in current:
                if pid in seen_ids:
                    duplicates.append(pid)
                    continue
                seen_ids[pid] = (path, is_system, mtime_ns, size)

            if duplicates:
                # Fail closed: stem collisions break every lookup keyed by
                # stem (the existing convention). Surface the issue rather
                # than silently dropping one of the files.
                raise DuplicatePageIdError(
                    f"Duplicate page_id stems detected: {sorted(set(duplicates))}"
                )

            # Diff entries.
            old_ids = set(self._entries.keys())
            new_ids = set(seen_ids.keys())
            removed = old_ids - new_ids

            changed = False
            for pid in removed:
                del self._entries[pid]
                changed = True

            for pid, (path, is_system, mtime_ns, size) in seen_ids.items():
                existing = self._entries.get(pid)
                path_str = str(path)
                if existing is None:
                    entry = self._build_entry(pid, path, is_system, mtime_ns, size)
                    if entry is not None:
                        self._entries[pid] = entry
                        changed = True
                    continue
                # Re-parse if any of (mtime_ns, size, path, is_system) differs.
                if (
                    existing.mtime_ns != mtime_ns
                    or existing.size != size
                    or existing.path != path_str
                    or existing.is_system != is_system
                ):
                    entry = self._build_entry(pid, path, is_system, mtime_ns, size)
                    if entry is not None:
                        self._entries[pid] = entry
                        changed = True
                    else:
                        del self._entries[pid]
                        changed = True

            new_order = [pid for pid, *_ in current if pid in self._entries]
            if self._page_order != new_order:
                self._page_order = new_order
                changed = True

            if changed:
                self._rebuild_backlinks()
                self._rebuild_associations()
                self._persistence_dirty = True
            if (
                self._persistence_dirty
                and os.environ.get("CHRONOVISOR_READ_ONLY") != "1"
            ):
                generation = self._next_generation()
                try:
                    self._persist(generation)
                except OSError:
                    # Persistence failure is non-fatal; in-memory index is
                    # still consistent and the next refresh retries the same
                    # snapshot even if disk content has not changed again.
                    pass
                else:
                    self._persistence_dirty = False
            self._last_refresh_monotonic = time.monotonic()

    def refresh_if_stale(self, max_age_seconds: float = 2.0) -> None:
        """Refresh at most once per short read transaction window.

        A recall request consults metadata through several independent
        retrieval channels. They must share one coherent snapshot instead of
        each walking every page on disk. Explicit ``refresh()`` calls retain
        their immediate semantics for mutation and administrative paths.
        """

        with self._lock:
            if (
                self._loaded
                and self._last_refresh_monotonic
                and time.monotonic() - self._last_refresh_monotonic
                < max(0.0, max_age_seconds)
            ):
                return
            self.refresh()

    @staticmethod
    def _build_entry(
        pid: str,
        path: Path,
        is_system: bool,
        mtime_ns: int,
        size: int,
    ) -> PageEntry | None:
        root = SYSTEM_DIR if is_system else PAGES_DIR
        resolved = contained_file(path, root)
        if resolved is None:
            return None
        data = _read_bytes_stable(resolved, root)
        if data is None:
            return None
        try:
            document = parse_document(data)
            status = _normalize_lifecycle_status(document.metadata.get("status"))
            namespace: Namespace = "system" if is_system else "pages"
            relative_path = resolved.relative_to(root.resolve(strict=True)).as_posix()
            resolved_links = resolve_internal_markdown_links(
                document.body,
                source_namespace=namespace,
                source_path=relative_path,
            )
        except (CanonicalDocumentError, OSError, RuntimeError, ValueError):
            return None
        fm = document.metadata
        title_value = fm.get("title")
        title = title_value if isinstance(title_value, str) else resolved.stem
        updated = str(fm.get("updated") or "unknown")
        outlinks = [
            PurePosixPath(link.path).stem
            for link in resolved_links
            if PurePosixPath(link.path).name
            not in (
                SYSTEM_RESERVED_FILENAMES
                if link.namespace == "system"
                else PAGE_RESERVED_FILENAMES
            )
        ]

        # raw_keywords / tags: trust the frontmatter only when it's an
        # actual ``list[str]``. Anything else (scalar string, missing,
        # broken cache from a manual edit) collapses to an empty list so
        # the rest of the system can rely on the type without
        # re-validating.
        def _coerce_str_list(value: object) -> list[str]:
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                return list(value)
            return []

        return PageEntry(
            page_id=pid,
            path=str(resolved),
            is_system=is_system,
            mtime_ns=mtime_ns,
            size=size,
            title=title,
            updated=updated,
            relative_path=relative_path,
            uid=fm.get("uid", "") if isinstance(fm.get("uid", ""), str) else "",
            classification_primary=(
                fm.get("classification_primary", "")
                if isinstance(fm.get("classification_primary", ""), str)
                else ""
            ),
            classification_notation=(
                fm.get("classification_notation", "")
                if isinstance(fm.get("classification_notation", ""), str)
                else ""
            ),
            classification_status=(
                fm.get("classification_status", "unclassified")
                if isinstance(
                    fm.get("classification_status", "unclassified"), str
                )
                else "unclassified"
            ),
            outlinks=outlinks,
            raw_keywords=_coerce_str_list(fm.get("raw_keywords")),
            tags=_coerce_str_list(fm.get("tags")),
            description=fm.get("description", "")
            if isinstance(fm.get("description", ""), str)
            else "",
            recall_questions=_coerce_str_list(fm.get("recall_questions")),
            status=status,
            superseded_by=fm.get("superseded_by", "")
            if isinstance(fm.get("superseded_by", ""), str)
            else "",
            page_type=_normalize_page_type(fm.get("type"), path=path),
            entities=_coerce_str_list(fm.get("entities")),
            sensitivity=_normalize_sensitivity(fm.get("sensitivity"), path=path),
        )

    def _rebuild_backlinks(self) -> None:
        """Rebuild `_backlinks` in scan order.

        Iterates `self._page_order` (rglob order, pages/ then system/) and
        appends each source page_id once per target. Sources within a
        target's list are deduplicated (a single page that links to the
        same target twice contributes one backlink edge), matching the
        existing `_find_backlinks` semantics.

        Edges are recorded for *all* targets present in any outlinks list,
        whether or not the target page exists today. This matches the
        current behaviour and lets a freshly-created page see its
        pre-existing inbound references immediately.
        """
        backlinks: dict[str, list[str]] = {}
        seen: dict[str, set[str]] = {}
        for source_pid in self._page_order:
            entry = self._entries.get(source_pid)
            if entry is None or entry.status != "stable":
                continue
            for target in entry.outlinks:
                if target == source_pid:
                    continue
                target_entry = self._entries.get(target)
                if target_entry is not None and target_entry.status != "stable":
                    continue
                src_set = seen.setdefault(target, set())
                if source_pid in src_set:
                    continue
                src_set.add(source_pid)
                backlinks.setdefault(target, []).append(source_pid)
        self._backlinks = backlinks

    def _rebuild_associations(self) -> None:
        tag_pages: dict[str, list[str]] = {}
        entity_pages: dict[str, list[str]] = {}
        for page_id in self._page_order:
            entry = self._entries.get(page_id)
            if entry is None or entry.status != "stable":
                continue
            for tag in dict.fromkeys(entry.tags):
                tag_pages.setdefault(tag, []).append(page_id)
            for entity in dict.fromkeys(entry.entities):
                entity_pages.setdefault(entity.casefold(), []).append(page_id)
        self._tag_pages = tag_pages
        self._entity_pages = entity_pages

    def _next_generation(self) -> int:
        # Use mtime-of-process-clock surrogate via a monotonically increasing
        # counter tied to the wall clock. We don't actually need wall time;
        # any value that increases with every persist is fine.
        import time

        return time.time_ns()

    # -- public read API --------------------------------------------------

    def meta(self, page_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._entries.get(page_id)
            if entry is None:
                return None
            return {
                "page_id": entry.page_id,
                "title": entry.title,
                "updated": entry.updated,
                "uid": entry.uid,
                "classification_primary": entry.classification_primary,
                "classification_notation": entry.classification_notation,
                "classification_status": entry.classification_status,
                "path": entry.path,
                "relative_path": entry.relative_path,
                "mtime_ns": entry.mtime_ns,
                "is_system": entry.is_system,
                "namespace": "system" if entry.is_system else "pages",
                "description": entry.description,
                # Compatibility projection while downstream consumers move to OKF
                # ``description``; canonical input never reads legacy ``summary``.
                "summary": entry.description,
                "recall_questions": list(entry.recall_questions),
                "status": entry.status,
                "superseded_by": entry.superseded_by,
                "page_type": entry.page_type,
                "entities": list(entry.entities),
                "sensitivity": entry.sensitivity,
            }

    def outlinks(self, page_id: str) -> list[str]:
        """Return exact-source links to stable or not-yet-indexed targets."""
        with self._lock:
            entry = self._entries.get(page_id)
            if entry is None:
                return []
            return [
                target
                for target in entry.outlinks
                if (target_entry := self._entries.get(target)) is None
                or target_entry.status == "stable"
            ]

    def raw_keywords(self, page_id: str) -> list[str]:
        """Return the page's frontmatter ``raw_keywords`` list.

        Empty list if the page has no field, the value isn't a clean
        ``list[str]``, or the page is unknown — callers don't need to
        distinguish "absent" from "empty" because both mean "no
        keyword-driven signal to use here". Surfaced as a dedicated
        method (rather than a key on ``meta()``) so the public meta
        contract stays stable; downstream features that need keywords
        (search expansion, distribution reports, link suggestions)
        can opt in by calling this directly.
        """
        with self._lock:
            entry = self._entries.get(page_id)
            return list(entry.raw_keywords) if entry else []

    def tags(self, page_id: str) -> list[str]:
        """Return the page's frontmatter ``tags`` list."""
        with self._lock:
            entry = self._entries.get(page_id)
            return list(entry.tags) if entry else []

    def all_tags(self, include_system: bool = False) -> list[str]:
        """Return every distinct tag across the corpus, sorted.

        Used by ingest's ``dedupe_with_existing`` step (existing-tag
        preference at >= 0.80 cosine similarity) and by
        ``chronovisor_search``'s tag filter to surface the available filter
        values.
        """
        with self._lock:
            seen: set[str] = set()
            for entry in self._entries.values():
                if entry.status != "stable" or (
                    not include_system and entry.is_system
                ):
                    continue
                seen.update(entry.tags)
            return sorted(seen)

    def backlinks(self, page_id: str) -> list[str]:
        """Return source page_ids that link to `page_id`, in scan order."""
        with self._lock:
            return list(self._backlinks.get(page_id, []))

    def pages_for_tag(self, tag: str) -> list[str]:
        with self._lock:
            return list(self._tag_pages.get(tag, []))

    def pages_for_entity(self, entity: str) -> list[str]:
        with self._lock:
            return list(self._entity_pages.get(entity.casefold(), []))

    def all_page_ids(self, include_system: bool = False) -> set[str]:
        with self._lock:
            return {
                pid
                for pid, entry in self._entries.items()
                if entry.status == "stable" and (include_system or not entry.is_system)
            }

    def all_pages_meta(self, include_system: bool = False) -> list[dict[str, Any]]:
        """Return meta dicts for every page, in mtime-descending order.

        Mirrors `chronovisor_index`'s sort: `path.stat().st_mtime` desc. We sort
        by `mtime_ns` desc — equivalent ordering modulo nanosecond ties,
        which Python's stable sort resolves identically to legacy
        behaviour for any practical case.
        """
        with self._lock:
            items = [
                entry
                for entry in self._entries.values()
                if entry.status == "stable"
                and (include_system or not entry.is_system)
            ]
            items.sort(key=lambda e: e.mtime_ns, reverse=True)
            return [
                {
                    "page_id": e.page_id,
                    "title": e.title,
                    "updated": e.updated,
                    "uid": e.uid,
                    "classification_primary": e.classification_primary,
                    "classification_notation": e.classification_notation,
                    "classification_status": e.classification_status,
                    "description": e.description,
                    "status": e.status,
                    "superseded_by": e.superseded_by,
                    "page_type": e.page_type,
                    "entities": list(e.entities),
                    "sensitivity": e.sensitivity,
                }
                for e in items
            ]

    def page_type(self, page_id: str) -> str:
        with self._lock:
            entry = self._entries.get(page_id)
            return entry.page_type if entry else "knowledge"

    def sensitivity(self, page_id: str) -> str:
        with self._lock:
            entry = self._entries.get(page_id)
            return entry.sensitivity if entry else "normal"

    def orphans(self, include_system: bool = False) -> list[str]:
        """Page IDs with no inbound backlinks (excluding self-links)."""
        with self._lock:
            out: list[str] = []
            for pid in self._page_order:
                entry = self._entries.get(pid)
                if entry is None or entry.status != "stable":
                    continue
                if entry.is_system and not include_system:
                    continue
                if not self._backlinks.get(pid):
                    out.append(pid)
            return out

    def page_count(self, include_system: bool = False) -> int:
        with self._lock:
            if include_system:
                return sum(
                    1 for entry in self._entries.values() if entry.status == "stable"
                )
            return sum(
                1
                for entry in self._entries.values()
                if entry.status == "stable" and not entry.is_system
            )

    def corpus_version(self) -> str:
        """Stable fingerprint of the current corpus state.

        Hash inputs are `(page_id, mtime_ns, is_system)` for every entry,
        sorted by page_id. Two refreshes that produced identical entries
        return identical fingerprints; any add / remove / mtime change /
        pages⇄system move flips the hash. Suitable as a memoization key
        for derived results that depend solely on the corpus (e.g. lint).

        Note: callers whose results also depend on external state (clock,
        config) must mix that into their own key on top of this one.
        """
        import hashlib

        with self._lock:
            items = sorted(
                (pid, e.mtime_ns, int(e.is_system)) for pid, e in self._entries.items()
            )
        h = hashlib.sha256()
        for pid, mt, sysflag in items:
            h.update(pid.encode("utf-8"))
            h.update(b"\x00")
            h.update(str(mt).encode("ascii"))
            h.update(b"\x00")
            h.update(b"1" if sysflag else b"0")
            h.update(b"\x01")
        return h.hexdigest()


_store_lock = threading.Lock()
_store: IndexStore | None = None


def get_store() -> IndexStore:
    """Return the process-wide IndexStore singleton."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = IndexStore()
    return _store
