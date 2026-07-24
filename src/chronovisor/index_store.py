"""Persistent page metadata + backlinks index.

Mirrors the semantics of `server._find_backlinks`, `server._page_metadata`,
`server._extract_wiki_links` and `lint._collect_all_page_ids` exactly, but
keeps everything in `~/.chronovisor/.index/` and refreshes incrementally based on
`(mtime_ns, size, path, is_system)` per file.

Backlinks are rebuilt in full on every refresh from the canonical
`pages.outlinks` data, both to keep them consistent with the source of
truth and to preserve scan-order parity with the legacy implementation.

The store is a process-wide singleton accessed via :func:`get_store`.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from chronovisor.frontmatter import parse as _frontmatter_parse
from chronovisor.link_fix import atomic_write, extract_targets
from chronovisor.store import CHRONOVISOR_ROOT, PAGES_DIR, SYSTEM_DIR

SCHEMA_VERSION = 9  # bumped for canonical alias targets
INDEX_DIR = CHRONOVISOR_ROOT / ".index"
PAGES_INDEX_FILE = INDEX_DIR / "pages.json"
BACKLINKS_INDEX_FILE = INDEX_DIR / "backlinks.json"
VALID_LIFECYCLE_STATUSES = {"active", "deprecated", "archived"}
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


def _canonical_aliases() -> dict[str, str]:
    """Return page-id aliases normalized to the stem-based index contract."""

    from chronovisor.alias_store import load_aliases

    canonical: dict[str, str] = {}
    for alias, target in load_aliases().items():
        alias_id = Path(str(alias).removesuffix(".md")).name
        target_id = Path(str(target).removesuffix(".md")).name
        if alias_id and target_id and alias_id != target_id:
            canonical[alias_id] = target_id
    return canonical


def _alias_sha256(aliases: dict[str, str]) -> str:
    encoded = json.dumps(
        aliases, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_page_id(page_id: str, aliases: dict[str, str]) -> str:
    return aliases.get(page_id, page_id)


def _normalize_lifecycle_status(value: object) -> str:
    if not isinstance(value, str):
        return "active"
    normalized = value.strip().lower()
    if normalized in VALID_LIFECYCLE_STATUSES:
        return normalized
    return "active"


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


def _parse_frontmatter(text: str) -> dict:
    """Thin wrapper around :func:`frontmatter.parse` returning only the
    metadata dict. Kept as a local function so existing call sites in
    this module don't need to change."""
    meta, _ = _frontmatter_parse(text)
    return meta


def _read_text_stable(path: Path, retries: int = 1) -> str | None:
    """Read a file and verify (mtime_ns, size) didn't change mid-read.

    Guards against picking up a half-written page (ingest doesn't use
    atomic_write today). On retry exhaustion, returns the last best-effort
    read; callers should treat None as "skip this file".
    """
    last: str | None = None
    for _ in range(retries + 1):
        try:
            st_before = path.stat()
            text = path.read_text()
            st_after = path.stat()
        except (OSError, UnicodeDecodeError):
            return None
        if (st_before.st_mtime_ns, st_before.st_size) == (
            st_after.st_mtime_ns,
            st_after.st_size,
        ):
            return text
        last = text
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
    summary: str = ""
    recall_questions: list[str] = field(default_factory=list)
    status: str = "active"
    superseded_by: str = ""
    page_type: str = "knowledge"
    entities: list[str] = field(default_factory=list)
    sensitivity: str = "normal"

    def to_dict(self) -> dict:
        return {
            "page_id": self.page_id,
            "path": self.path,
            "is_system": self.is_system,
            "mtime_ns": self.mtime_ns,
            "size": self.size,
            "title": self.title,
            "updated": self.updated,
            "outlinks": list(self.outlinks),
            "raw_keywords": list(self.raw_keywords),
            "tags": list(self.tags),
            "summary": self.summary,
            "recall_questions": list(self.recall_questions),
            "status": self.status,
            "superseded_by": self.superseded_by,
            "page_type": self.page_type,
            "entities": list(self.entities),
            "sensitivity": self.sensitivity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PageEntry":
        # Defensively coerce list-of-string fields: a manually edited
        # cache file or a future-format mismatch shouldn't crash the
        # singleton at startup.
        def _coerce_str_list(value):
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
            outlinks=list(d.get("outlinks", [])),
            raw_keywords=_coerce_str_list(d.get("raw_keywords")),
            tags=_coerce_str_list(d.get("tags")),
            summary=d.get("summary", "")
            if isinstance(d.get("summary", ""), str)
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
        self._alias_sha256 = ""
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
        if pages_doc.get("alias_sha256") != backlinks_doc.get("alias_sha256"):
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
        self._alias_sha256 = str(pages_doc.get("alias_sha256") or "")

    def _persist(self, generation: int) -> None:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        pages_doc = {
            "schema_version": SCHEMA_VERSION,
            "generation": generation,
            "alias_sha256": self._alias_sha256,
            "page_order": list(self._page_order),
            "entries": {pid: e.to_dict() for pid, e in self._entries.items()},
        }
        backlinks_doc = {
            "schema_version": SCHEMA_VERSION,
            "generation": generation,
            "alias_sha256": self._alias_sha256,
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
        for path in PAGES_DIR.rglob("*.md"):
            try:
                st = path.stat()
            except OSError:
                continue
            out.append((path.stem, path, False, st.st_mtime_ns, st.st_size))
        if SYSTEM_DIR.exists():
            for path in SYSTEM_DIR.rglob("*.md"):
                try:
                    st = path.stat()
                except OSError:
                    continue
                out.append((path.stem, path, True, st.st_mtime_ns, st.st_size))
        return out

    def refresh(self) -> None:
        """Sync the in-memory index with disk.

        Cheap when nothing changed (one stat per page, no parsing).
        Persists only when entries or backlinks actually changed.
        """
        with self._lock:
            if not self._loaded:
                self._load_from_disk()
                self._loaded = True

            aliases = _canonical_aliases()
            alias_sha256 = _alias_sha256(aliases)
            aliases_changed = alias_sha256 != self._alias_sha256

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

            new_order = [pid for pid, *_ in current if pid in seen_ids]

            # Diff entries.
            old_ids = set(self._entries.keys())
            new_ids = set(seen_ids.keys())
            removed = old_ids - new_ids
            added = new_ids - old_ids

            changed = False
            for pid in removed:
                del self._entries[pid]
                changed = True

            for pid, (path, is_system, mtime_ns, size) in seen_ids.items():
                existing = self._entries.get(pid)
                path_str = str(path)
                if existing is None:
                    entry = self._build_entry(
                        pid, path, is_system, mtime_ns, size, aliases=aliases
                    )
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
                    or aliases_changed
                ):
                    entry = self._build_entry(
                        pid, path, is_system, mtime_ns, size, aliases=aliases
                    )
                    if entry is not None:
                        self._entries[pid] = entry
                        changed = True

            if self._page_order != new_order:
                self._page_order = new_order
                changed = True

            if aliases_changed:
                self._alias_sha256 = alias_sha256
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
        *,
        aliases: dict[str, str] | None = None,
    ) -> PageEntry | None:
        text = _read_text_stable(path)
        if text is None:
            return None
        fm = _parse_frontmatter(text)
        title = fm.get("title", path.stem)
        updated = fm.get("updated", "unknown")
        canonical_aliases = aliases or {}
        outlinks = [
            _canonical_page_id(target, canonical_aliases)
            for target in extract_targets(text, strip=True)
        ]

        # raw_keywords / tags: trust the frontmatter only when it's an
        # actual ``list[str]``. Anything else (scalar string, missing,
        # broken cache from a manual edit) collapses to an empty list so
        # the rest of the system can rely on the type without
        # re-validating.
        def _coerce_str_list(value):
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                return list(value)
            return []

        return PageEntry(
            page_id=pid,
            path=str(path),
            is_system=is_system,
            mtime_ns=mtime_ns,
            size=size,
            title=title,
            updated=updated,
            outlinks=outlinks,
            raw_keywords=_coerce_str_list(fm.get("raw_keywords")),
            tags=_coerce_str_list(fm.get("tags")),
            summary=fm.get("summary", "")
            if isinstance(fm.get("summary", ""), str)
            else "",
            recall_questions=_coerce_str_list(fm.get("recall_questions")),
            status=_normalize_lifecycle_status(fm.get("status")),
            superseded_by=_canonical_page_id(
                fm.get("superseded_by", ""), canonical_aliases
            )
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
            if entry is None:
                continue
            for target in entry.outlinks:
                if target == source_pid:
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
            if entry is None:
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

    def meta(self, page_id: str) -> dict | None:
        with self._lock:
            entry = self._entries.get(page_id)
            if entry is None:
                return None
            return {
                "page_id": entry.page_id,
                "title": entry.title,
                "updated": entry.updated,
                "path": entry.path,
                "mtime_ns": entry.mtime_ns,
                "is_system": entry.is_system,
                "summary": entry.summary,
                "recall_questions": list(entry.recall_questions),
                "status": entry.status,
                "superseded_by": entry.superseded_by,
                "page_type": entry.page_type,
                "entities": list(entry.entities),
                "sensitivity": entry.sensitivity,
            }

    def outlinks(self, page_id: str) -> list[str]:
        """Return raw outlinks list (preserves duplicates + order)."""
        with self._lock:
            entry = self._entries.get(page_id)
            return list(entry.outlinks) if entry else []

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
                if not include_system and entry.is_system:
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

    def all_page_ids(self, include_system: bool = True) -> set[str]:
        with self._lock:
            if include_system:
                return set(self._entries.keys())
            return {pid for pid, e in self._entries.items() if not e.is_system}

    def all_pages_meta(self, include_system: bool = False) -> list[dict]:
        """Return meta dicts for every page, in mtime-descending order.

        Mirrors `chronovisor_index`'s sort: `path.stat().st_mtime` desc. We sort
        by `mtime_ns` desc — equivalent ordering modulo nanosecond ties,
        which Python's stable sort resolves identically to legacy
        behaviour for any practical case.
        """
        with self._lock:
            items = [
                e for e in self._entries.values() if include_system or not e.is_system
            ]
            items.sort(key=lambda e: e.mtime_ns, reverse=True)
            return [
                {
                    "page_id": e.page_id,
                    "title": e.title,
                    "updated": e.updated,
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
                if entry is None:
                    continue
                if entry.is_system and not include_system:
                    continue
                if not self._backlinks.get(pid):
                    out.append(pid)
            return out

    def page_count(self, include_system: bool = False) -> int:
        with self._lock:
            if include_system:
                return len(self._entries)
            return sum(1 for e in self._entries.values() if not e.is_system)

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
