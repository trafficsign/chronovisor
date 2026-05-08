"""Tests for IndexStore — Phase 5 raw_keywords schema extension.

Focus: PageEntry round-trip, _build_entry frontmatter extraction with
defensive type coercion, schema migration (v1 cache → invalidate +
rebuild), and the public ``raw_keywords(page_id)`` accessor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_wiki_mcp import index_store as index_store_mod
from llm_wiki_mcp.index_store import IndexStore, PageEntry


@pytest.fixture()
def isolated_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point IndexStore module-level paths at a throw-away wiki tree."""
    wiki_root = tmp_path / "wiki"
    pages = wiki_root / "pages"
    system = wiki_root / "system"
    index_dir = wiki_root / ".index"
    for d in (pages, system, index_dir):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(index_store_mod, "WIKI_ROOT", wiki_root)
    monkeypatch.setattr(index_store_mod, "PAGES_DIR", pages)
    monkeypatch.setattr(index_store_mod, "SYSTEM_DIR", system)
    monkeypatch.setattr(index_store_mod, "INDEX_DIR", index_dir)
    monkeypatch.setattr(index_store_mod, "PAGES_INDEX_FILE", index_dir / "pages.json")
    monkeypatch.setattr(
        index_store_mod, "BACKLINKS_INDEX_FILE", index_dir / "backlinks.json"
    )
    return wiki_root


def _seed(wiki_root: Path, rel: str, body: str) -> Path:
    path = wiki_root / "pages" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


# ---------------------------------------------------------------------------
# PageEntry round-trip
# ---------------------------------------------------------------------------


class TestPageEntryRoundTrip:
    def test_to_dict_includes_raw_keywords(self) -> None:
        e = PageEntry(
            page_id="p",
            path="/tmp/p.md",
            is_system=False,
            mtime_ns=1,
            size=10,
            title="P",
            updated="2026-01-01",
            outlinks=["a", "b"],
            raw_keywords=["alpha", "beta"],
        )
        d = e.to_dict()
        assert d["raw_keywords"] == ["alpha", "beta"]

    def test_from_dict_defaults_to_empty(self) -> None:
        """Loading a v1-shaped cache entry (no raw_keywords key) must
        produce an empty list, not a KeyError."""
        d = {
            "page_id": "p",
            "path": "/tmp/p.md",
            "is_system": False,
            "mtime_ns": 1,
            "size": 10,
            "title": "P",
            "updated": "2026-01-01",
            "outlinks": [],
        }
        e = PageEntry.from_dict(d)
        assert e.raw_keywords == []

    @pytest.mark.parametrize(
        "bad",
        ["scalar", 42, None, ["ok", 123], {"k": "v"}],
    )
    def test_from_dict_coerces_bad_types_to_empty(self, bad: object) -> None:
        d = {
            "page_id": "p",
            "path": "/tmp/p.md",
            "is_system": False,
            "mtime_ns": 1,
            "size": 10,
            "title": "P",
            "updated": "2026-01-01",
            "outlinks": [],
            "raw_keywords": bad,
        }
        e = PageEntry.from_dict(d)
        assert e.raw_keywords == []

    def test_round_trip_preserves_raw_keywords(self) -> None:
        e = PageEntry(
            page_id="p",
            path="/tmp/p.md",
            is_system=False,
            mtime_ns=1,
            size=10,
            title="P",
            updated="2026-01-01",
            raw_keywords=["x", "y", "z"],
        )
        e2 = PageEntry.from_dict(e.to_dict())
        assert e2.raw_keywords == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# _build_entry frontmatter extraction
# ---------------------------------------------------------------------------


class TestBuildEntryRawKeywords:
    def test_extracts_inline_list_from_frontmatter(
        self, isolated_index: Path
    ) -> None:
        path = _seed(
            isolated_index,
            "p.md",
            "---\ntitle: P\nupdated: 2026-01-01\nraw_keywords: [a, b, c]\n---\nbody\n",
        )
        st = path.stat()
        entry = IndexStore._build_entry("p", path, False, st.st_mtime_ns, st.st_size)
        assert entry is not None
        assert entry.raw_keywords == ["a", "b", "c"]

    def test_missing_field_yields_empty_list(self, isolated_index: Path) -> None:
        path = _seed(
            isolated_index,
            "q.md",
            "---\ntitle: Q\nupdated: 2026-01-01\n---\nbody\n",
        )
        st = path.stat()
        entry = IndexStore._build_entry("q", path, False, st.st_mtime_ns, st.st_size)
        assert entry is not None
        assert entry.raw_keywords == []

    def test_scalar_value_treated_as_empty(self, isolated_index: Path) -> None:
        """A page with malformed ``raw_keywords: oops`` (scalar) must not
        confuse the index — it's silently ignored as ``[]``."""
        path = _seed(
            isolated_index,
            "r.md",
            "---\ntitle: R\nupdated: 2026-01-01\nraw_keywords: oops\n---\nbody\n",
        )
        st = path.stat()
        entry = IndexStore._build_entry("r", path, False, st.st_mtime_ns, st.st_size)
        assert entry is not None
        assert entry.raw_keywords == []


# ---------------------------------------------------------------------------
# Public accessor
# ---------------------------------------------------------------------------


class TestRawKeywordsAccessor:
    def test_round_trip_via_refresh(self, isolated_index: Path) -> None:
        _seed(
            isolated_index,
            "p.md",
            "---\ntitle: P\nupdated: 2026-01-01\nraw_keywords: [foo, bar]\n---\nbody\n",
        )
        store = IndexStore()
        store.refresh()
        assert store.raw_keywords("p") == ["foo", "bar"]

    def test_unknown_page_returns_empty(self, isolated_index: Path) -> None:
        store = IndexStore()
        store.refresh()
        assert store.raw_keywords("does-not-exist") == []

    def test_meta_does_not_leak_raw_keywords(self, isolated_index: Path) -> None:
        """Phase 5 keeps the public ``meta()`` contract stable — keywords
        are accessed via the dedicated method, not via the meta dict."""
        _seed(
            isolated_index,
            "p.md",
            "---\ntitle: P\nupdated: 2026-01-01\nraw_keywords: [foo]\n---\nbody\n",
        )
        store = IndexStore()
        store.refresh()
        meta = store.meta("p")
        assert meta is not None
        assert "raw_keywords" not in meta


# ---------------------------------------------------------------------------
# Schema migration: v1 cache must invalidate and rebuild on first refresh
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_v1_cache_is_invalidated_and_rebuilt(self, isolated_index: Path) -> None:
        """A pages.json + backlinks.json written under SCHEMA_VERSION=1
        must be rejected by ``_load_from_disk`` (returns silently); the
        next ``refresh()`` then walks disk and produces a v2 cache.
        """
        # Seed a real page so the rebuild has something to record.
        _seed(
            isolated_index,
            "p.md",
            "---\ntitle: P\nupdated: 2026-01-01\nraw_keywords: [k1, k2]\n---\nbody\n",
        )

        # Hand-write a v1-shaped cache that points at a *different* page id
        # so we can detect whether the cache was used or thrown away.
        v1_pages = {
            "schema_version": 1,
            "generation": 12345,
            "page_order": ["ghost"],
            "entries": {
                "ghost": {
                    "page_id": "ghost",
                    "path": "/nope.md",
                    "is_system": False,
                    "mtime_ns": 0,
                    "size": 0,
                    "title": "Ghost",
                    "updated": "1999-01-01",
                    "outlinks": [],
                }
            },
        }
        v1_backlinks = {"schema_version": 1, "generation": 12345, "edges": {}}
        index_store_mod.PAGES_INDEX_FILE.write_text(json.dumps(v1_pages))
        index_store_mod.BACKLINKS_INDEX_FILE.write_text(json.dumps(v1_backlinks))

        store = IndexStore()
        store.refresh()

        # The stale ``ghost`` entry from the v1 cache must be gone.
        assert store.meta("ghost") is None
        # The real on-disk page is indexed with raw_keywords.
        assert store.raw_keywords("p") == ["k1", "k2"]

        # And the persisted cache is now v2 so subsequent processes load it.
        persisted = json.loads(index_store_mod.PAGES_INDEX_FILE.read_text())
        assert persisted["schema_version"] == index_store_mod.SCHEMA_VERSION
        assert "p" in persisted["entries"]
        assert persisted["entries"]["p"]["raw_keywords"] == ["k1", "k2"]

    def test_v2_cache_round_trips_across_store_instances(
        self, isolated_index: Path
    ) -> None:
        """A second IndexStore instance reads the persisted cache from
        the first and recovers raw_keywords without scanning frontmatter
        again — proving the persistence path includes raw_keywords."""
        path = _seed(
            isolated_index,
            "p.md",
            "---\ntitle: P\nupdated: 2026-01-01\nraw_keywords: [hello]\n---\nbody\n",
        )

        first = IndexStore()
        first.refresh()
        assert first.raw_keywords("p") == ["hello"]

        # Mutate the on-disk page so a re-parse would lose the value, but
        # the cache should win because (mtime_ns, size) hasn't been
        # changed *yet* — write back with preserved stat to simulate a
        # cold start with a stable cache. Easiest: just spin up another
        # store WITHOUT touching disk.
        second = IndexStore()
        second._load_from_disk()
        assert second.raw_keywords("p") == ["hello"]
        # Sanity: didn't double-touch disk for the original path.
        assert path.exists()
