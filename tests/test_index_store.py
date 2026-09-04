"""Tests for the persistent canonical-document index."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronovisor.core import index_store as index_store_mod
from chronovisor.core.index_store import IndexStore, PageEntry


@pytest.fixture()
def isolated_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point IndexStore module-level paths at a throw-away wiki tree."""
    chronovisor_root = tmp_path / "wiki"
    pages = chronovisor_root / "pages"
    system = chronovisor_root / "system"
    index_dir = chronovisor_root / ".index"
    for d in (pages, system, index_dir):
        d.mkdir(parents=True, exist_ok=True)
    for name in ("index.md", "log.md", "schema.md"):
        (chronovisor_root / name).write_text("legacy\n", encoding="utf-8")

    monkeypatch.setattr(index_store_mod, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(index_store_mod, "PAGES_DIR", pages)
    monkeypatch.setattr(index_store_mod, "SYSTEM_DIR", system)
    monkeypatch.setattr(index_store_mod, "INDEX_DIR", index_dir)
    monkeypatch.setattr(index_store_mod, "PAGES_INDEX_FILE", index_dir / "pages.json")
    monkeypatch.setattr(
        index_store_mod, "BACKLINKS_INDEX_FILE", index_dir / "backlinks.json"
    )
    return chronovisor_root


def _seed(chronovisor_root: Path, rel: str, body: str) -> Path:
    if body.startswith("---\n"):
        closing = body.find("\n---\n", 4)
        if closing >= 0 and "\nstatus:" not in body[:closing]:
            body = body.replace("---\n", "---\nstatus: stable\n", 1)
        closing = body.find("\n---\n", 4)
        if closing >= 0 and "\ntype:" not in body[:closing]:
            body = body.replace("---\n", "---\ntype: knowledge\n", 1)
    path = chronovisor_root / "pages" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def _seed_system(chronovisor_root: Path, rel: str, body: str) -> Path:
    path = chronovisor_root / "system" / rel
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
            "status": "stable",
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
            "status": "stable",
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

    def test_extracts_lifecycle_frontmatter(self, isolated_index: Path) -> None:
        path = _seed(
            isolated_index,
            "old.md",
            (
                "---\n"
                "title: Old\n"
                "updated: 2026-01-01\n"
                "status: deprecated\n"
                "superseded_by: new-page\n"
                "---\n"
                "body\n"
            ),
        )
        st = path.stat()
        entry = IndexStore._build_entry("old", path, False, st.st_mtime_ns, st.st_size)
        assert entry is not None
        assert entry.status == "deprecated"
        assert entry.superseded_by == "new-page"

    def test_invalid_lifecycle_status_is_not_indexed(self, isolated_index: Path) -> None:
        path = _seed(
            isolated_index,
            "bad.md",
            "---\ntitle: Bad\nupdated: 2026-01-01\nstatus: stale-ish\n---\nbody\n",
        )
        st = path.stat()
        entry = IndexStore._build_entry("bad", path, False, st.st_mtime_ns, st.st_size)
        assert entry is None

    def test_car_spec_preserves_reference_page_type(self, isolated_index: Path) -> None:
        path = _seed(
            isolated_index,
            "car-spec/123.md",
            "---\ntitle: 123\ntype: reference\nupdated: 2026-01-01\n---\nbody\n",
        )
        st = path.stat()
        entry = IndexStore._build_entry("123", path, False, st.st_mtime_ns, st.st_size)
        assert entry is not None
        assert entry.page_type == "reference"

    def test_explicit_page_type_is_indexed(self, isolated_index: Path) -> None:
        path = _seed(
            isolated_index,
            "refs/p.md",
            "---\ntitle: P\nupdated: 2026-01-01\ntype: procedural\n---\nbody\n",
        )
        st = path.stat()
        entry = IndexStore._build_entry("p", path, False, st.st_mtime_ns, st.st_size)
        assert entry is not None
        assert entry.page_type == "procedural"

    def test_extracts_entities_frontmatter(self, isolated_index: Path) -> None:
        path = _seed(
            isolated_index,
            "p.md",
            "---\ntitle: P\nupdated: 2026-01-01\nentities: [chronovisor, qwen]\n---\nbody\n",
        )
        st = path.stat()
        entry = IndexStore._build_entry("p", path, False, st.st_mtime_ns, st.st_size)
        assert entry is not None
        assert entry.entities == ["chronovisor", "qwen"]

    def test_extracts_sensitivity_frontmatter(self, isolated_index: Path) -> None:
        path = _seed(
            isolated_index,
            "p.md",
            "---\ntitle: P\nupdated: 2026-01-01\nsensitivity: high\n---\nbody\n",
        )
        st = path.stat()
        entry = IndexStore._build_entry("p", path, False, st.st_mtime_ns, st.st_size)
        assert entry is not None
        assert entry.sensitivity == "high"

    def test_career_folder_infers_high_sensitivity(self, isolated_index: Path) -> None:
        path = _seed(
            isolated_index,
            "career/interview.md",
            "---\ntitle: Interview\nupdated: 2026-01-01\n---\nbody\n",
        )
        st = path.stat()
        entry = IndexStore._build_entry("interview", path, False, st.st_mtime_ns, st.st_size)
        assert entry is not None
        assert entry.sensitivity == "high"


# ---------------------------------------------------------------------------
# Public accessor
# ---------------------------------------------------------------------------


class TestRawKeywordsAccessor:
    def test_legacy_wikilink_rejects_entire_page(
        self, isolated_index: Path
    ) -> None:
        _seed(
            isolated_index,
            "source.md",
            "---\ntitle: Source\n---\n[Current](current-page.md)\n[[legacy]]\n",
        )
        _seed(isolated_index, "current-page.md", "---\ntitle: Current\n---\nbody\n")
        store = IndexStore()

        store.refresh()

        assert store.meta("source") is None
        assert store.outlinks("source") == []
        assert store.backlinks("current-page") == []
        assert store.backlinks("legacy") == []

    def test_read_only_refresh_rebuilds_memory_without_persisting(
        self, isolated_index: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(isolated_index, "p.md", "---\ntitle: P\n---\nbody\n")
        monkeypatch.setenv("CHRONOVISOR_READ_ONLY", "1")
        store = IndexStore()

        store.refresh()

        assert store.meta("p") is not None
        assert not index_store_mod.PAGES_INDEX_FILE.exists()
        assert not index_store_mod.BACKLINKS_INDEX_FILE.exists()

        monkeypatch.delenv("CHRONOVISOR_READ_ONLY")
        store.refresh()

        assert index_store_mod.PAGES_INDEX_FILE.exists()
        assert index_store_mod.BACKLINKS_INDEX_FILE.exists()

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

    def test_meta_exposes_lifecycle_fields(self, isolated_index: Path) -> None:
        _seed(
            isolated_index,
            "old.md",
            (
                "---\n"
                "title: Old\n"
                "updated: 2026-01-01\n"
                "status: deprecated\n"
                "superseded_by: new-page\n"
                "---\n"
                "body\n"
            ),
        )
        store = IndexStore()
        store.refresh()
        meta = store.meta("old")
        assert meta is not None
        assert meta["status"] == "deprecated"
        assert meta["superseded_by"] == "new-page"

    def test_meta_exposes_page_type(self, isolated_index: Path) -> None:
        _seed(
            isolated_index,
            "refs/p.md",
            "---\ntitle: P\nupdated: 2026-01-01\ntype: decision\nentities: [qwen]\n---\nbody\n",
        )
        store = IndexStore()
        store.refresh()
        meta = store.meta("p")
        assert meta is not None
        assert meta["page_type"] == "decision"
        assert meta["entities"] == ["qwen"]
        assert meta["sensitivity"] == "normal"
        assert store.page_type("p") == "decision"

    def test_meta_exposes_sensitivity(self, isolated_index: Path) -> None:
        _seed(
            isolated_index,
            "career/interview.md",
            "---\ntitle: Interview\nupdated: 2026-01-01\n---\nbody\n",
        )
        store = IndexStore()
        store.refresh()
        meta = store.meta("interview")
        assert meta is not None
        assert meta["sensitivity"] == "high"
        assert store.sensitivity("interview") == "high"


class TestRefreshWindow:
    def test_cached_refresh_shares_snapshot_but_explicit_refresh_is_immediate(
        self, isolated_index: Path
    ) -> None:
        path = _seed(
            isolated_index,
            "p.md",
            "---\ntitle: Before\nupdated: 2026-01-01\n---\nbody\n",
        )
        store = IndexStore()
        store.refresh()

        path.write_text(
            "---\ntitle: After\nupdated: 2026-01-02\nstatus: stable\n"
            "type: knowledge\n---\nlonger body\n"
        )
        store.refresh_if_stale(max_age_seconds=60)
        assert store.meta("p")["title"] == "Before"

        store.refresh()
        assert store.meta("p")["title"] == "After"


class TestDerivedRefreshScope:
    def test_metadata_only_refresh_updates_entry_without_rebuilding_indexes(
        self, isolated_index: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(
            isolated_index,
            "target.md",
            "---\ntitle: Target\nupdated: 2026-01-01\ntags: [target]\n"
            "entities: [Chronovisor]\n---\nbody\n",
        )
        source = _seed(
            isolated_index,
            "source.md",
            "---\ntitle: Source\nupdated: 2026-01-01\ntags: [source]\n"
            "entities: [Chronovisor]\n---\n[Target](target.md)\n",
        )
        store = IndexStore()
        store.refresh()
        before_backlinks = store.backlinks("target")
        before_tags = store.pages_for_tag("source")
        before_entities = store.pages_for_entity("chronovisor")

        calls: list[str] = []
        for name in (
            "_rebuild_canonical_entries",
            "_rebuild_backlinks",
            "_rebuild_associations",
        ):
            original = getattr(store, name)

            def track(*args: object, _name=name, _original=original, **kwargs: object):
                calls.append(_name)
                return _original(*args, **kwargs)

            monkeypatch.setattr(store, name, track)

        source.write_text(
            "---\ntitle: Renamed in metadata\nupdated: 2026-01-01\n"
            "status: stable\ntype: knowledge\n"
            "tags: [source]\nentities: [Chronovisor]\n---\n[Target](target.md)\n",
            encoding="utf-8",
        )
        store.refresh()

        assert store.meta("source")["title"] == "Renamed in metadata"
        assert store.backlinks("target") == before_backlinks
        assert store.pages_for_tag("source") == before_tags
        assert store.pages_for_entity("chronovisor") == before_entities
        assert calls == []

    def test_targeted_metadata_update_updates_canonical_map_without_rebuild(
        self, isolated_index: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(
            isolated_index,
            "target.md",
            "---\ntitle: Target\nupdated: 2026-01-01\n---\nbody\n",
        )
        source = _seed(
            isolated_index,
            "source.md",
            "---\ntitle: Source\nupdated: 2026-01-01\n---\n[Target](target.md)\n",
        )
        store = IndexStore()
        store.refresh()
        calls: list[str] = []
        for name in (
            "_rebuild_canonical_entries",
            "_rebuild_backlinks",
            "_rebuild_associations",
        ):
            original = getattr(store, name)

            def track(*args: object, _name=name, _original=original, **kwargs: object):
                calls.append(_name)
                return _original(*args, **kwargs)

            monkeypatch.setattr(store, name, track)

        source.write_text(
            "---\ntitle: Targeted metadata\nupdated: 2026-01-01\n"
            "status: stable\ntype: knowledge\n---\n"
            "[Target](target.md)\n",
            encoding="utf-8",
        )
        store.apply_changes([source])

        assert store.meta("source")["title"] == "Targeted metadata"
        assert store.outlinks("source") == ["target"]
        assert store.backlinks("target") == ["source"]
        assert calls == []

    def test_refresh_preserves_derived_results_when_entry_links_change(
        self, isolated_index: Path
    ) -> None:
        for page_id in ("target", "other"):
            _seed(
                isolated_index,
                f"{page_id}.md",
                f"---\ntitle: {page_id}\nupdated: 2026-01-01\n---\nbody\n",
            )
        other = isolated_index / "pages" / "other.md"
        source = _seed(
            isolated_index,
            "source.md",
            "---\ntitle: Source\nupdated: 2026-01-01\ntags: [old]\n"
            "entities: [Source]\n---\n[Target](target.md)\n",
        )
        store = IndexStore()
        store.refresh()

        source.write_text(
            "---\ntitle: Source\nupdated: 2026-01-01\nstatus: stable\n"
            "type: knowledge\ntags: [new]\nentities: [Source]\n---\n"
            "[Target](target.md)\n",
            encoding="utf-8",
        )
        store.refresh()
        assert store.backlinks("target") == ["source"]
        assert store.pages_for_tag("new") == ["source"]

        source.write_text(
            "---\ntitle: Source\nupdated: 2026-01-01\nstatus: stable\n"
            "type: knowledge\ntags: [new]\nentities: [Source]\n---\n"
            "[Other](other.md)\n",
            encoding="utf-8",
        )
        store.refresh()
        assert store.backlinks("target") == []
        assert store.backlinks("other") == ["source"]

        other.write_text(
            "---\ntitle: other\nupdated: 2026-01-01\nstatus: draft\n"
            "type: knowledge\n---\nbody\n",
            encoding="utf-8",
        )
        store.refresh()
        assert store.backlinks("other") == []
        assert store.outlinks("source") == []

    def test_targeted_metadata_update_rollback_restores_canonical_entry(
        self, isolated_index: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(
            isolated_index,
            "target.md",
            "---\ntitle: Target\nupdated: 2026-01-01\n---\nbody\n",
        )
        source = _seed(
            isolated_index,
            "source.md",
            "---\ntitle: Source\nupdated: 2026-01-01\n---\n[Target](target.md)\n",
        )
        store = IndexStore()
        store.refresh()
        monkeypatch.setattr(
            store,
            "_persist",
            lambda _generation: (_ for _ in ()).throw(OSError("disk full")),
        )
        source.write_text(
            "---\ntitle: Changed\nupdated: 2026-01-01\n"
            "status: stable\ntype: knowledge\n---\n[Target](target.md)\n",
            encoding="utf-8",
        )

        with pytest.raises(OSError, match="disk full"):
            store.apply_changes([source])

        assert store.meta("source")["title"] == "Source"
        assert store.outlinks("source") == ["target"]
        assert store.backlinks("target") == ["source"]


class TestTargetedChanges:
    def test_created_page_is_visible_without_a_second_full_scan(
        self,
        isolated_index: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed(
            isolated_index,
            "existing.md",
            "---\ntitle: Existing\nupdated: 2026-01-01\n---\nbody\n",
        )
        store = IndexStore()
        store.refresh()
        created = _seed(
            isolated_index,
            "nested/created.md",
            "---\ntitle: Created\nupdated: 2026-08-30\n---\nnew body\n",
        )

        monkeypatch.setattr(
            store,
            "_scan_disk",
            lambda: pytest.fail("targeted update performed a full disk scan"),
        )

        store.apply_changes([created])

        assert store.meta("created") == {
            "page_id": "created",
            "title": "Created",
            "updated": "2026-08-30",
            "uid": "",
            "classification_primary": "",
            "classification_notation": "",
            "classification_status": "unclassified",
            "path": str(created.resolve()),
            "relative_path": "nested/created.md",
            "mtime_ns": created.stat().st_mtime_ns,
            "is_system": False,
            "namespace": "pages",
            "description": "",
            "summary": "",
            "recall_questions": [],
            "status": "stable",
            "superseded_by": "",
            "page_type": "knowledge",
            "entities": [],
            "sensitivity": "normal",
        }
        assert store.page_count() == 2

    def test_duplicate_receipt_path_is_parsed_once(
        self,
        isolated_index: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = IndexStore()
        store.refresh()
        created = _seed(
            isolated_index,
            "created.md",
            "---\ntitle: Created\nupdated: 2026-08-30\n---\nbody\n",
        )
        real_build = store._build_entry
        calls: list[Path] = []

        def build(*args, **kwargs):
            calls.append(args[1])
            return real_build(*args, **kwargs)

        monkeypatch.setattr(store, "_build_entry", build)

        store.apply_changes([created, created])

        assert calls == [created.resolve()]

    def test_other_process_loads_new_generation_without_a_full_scan(
        self,
        isolated_index: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed(
            isolated_index,
            "existing.md",
            "---\ntitle: Existing\nupdated: 2026-01-01\n---\nbody\n",
        )
        writer = IndexStore()
        reader = IndexStore()
        writer.refresh()
        reader.refresh()
        created = _seed(
            isolated_index,
            "created.md",
            "---\ntitle: Created\nupdated: 2026-08-30\n---\nbody\n",
        )
        writer.apply_changes([created])
        monkeypatch.setattr(
            reader,
            "_scan_disk",
            lambda: pytest.fail("cache generation reload performed a full scan"),
        )

        reader.refresh_if_stale(max_age_seconds=0)

        assert reader.meta("created")["title"] == "Created"

    def test_persist_failure_restores_previous_in_memory_snapshot(
        self,
        isolated_index: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _seed(
            isolated_index,
            "existing.md",
            "---\ntitle: Existing\nupdated: 2026-01-01\n---\nbody\n",
        )
        store = IndexStore()
        store.refresh()
        created = _seed(
            isolated_index,
            "created.md",
            "---\ntitle: Created\nupdated: 2026-08-30\n---\nbody\n",
        )
        monkeypatch.setattr(
            store,
            "_persist",
            lambda _generation: (_ for _ in ()).throw(OSError("disk full")),
        )

        with pytest.raises(OSError, match="disk full"):
            store.apply_changes([created])

        assert store.meta("created") is None
        assert store.page_count() == 1


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


class TestCanonicalIndex:
    def test_changed_entry_rejects_missing_type_and_legacy_wikilink(
        self, isolated_index: Path
    ) -> None:
        pages = isolated_index / "pages"
        missing_type = pages / "missing-type.md"
        legacy_link = pages / "legacy-link.md"
        missing_type.write_text(
            "---\ntitle: Missing\nstatus: stable\n---\nbody\n",
            encoding="utf-8",
        )
        legacy_link.write_text(
            "---\ntitle: Legacy\nstatus: stable\ntype: knowledge\n---\n"
            "[[target]]\n",
            encoding="utf-8",
        )

        store = IndexStore()
        store.refresh()

        assert store.meta("missing-type") is None
        assert store.meta("legacy-link") is None

    def test_full_yaml_and_namespace_aware_standard_links(
        self, isolated_index: Path
    ) -> None:
        _seed(
            isolated_index,
            "target.md",
            "---\ntitle: Target\nstatus: stable\n---\nbody\n",
        )
        _seed(
            isolated_index,
            "root.md",
            "---\ntitle: Root\nstatus: stable\n---\nbody\n",
        )
        _seed(
            isolated_index,
            "notes/source.md",
            """---
title: Source
updated: 2026-08-10
status: stable
description: Canonical description
classification_status: active
extension:
  nested:
    enabled: true
    weights: [1, 2, 3]
---
[Relative](../target.md) and [Root](/root.md).
""",
        )
        _seed_system(
            isolated_index,
            "current-state.md",
            """---
identity: current-state
status: stable
registry_state: internal
---
[Portable](../pages/target.md) and [Schema](schema.md)
""",
        )
        _seed_system(
            isolated_index,
            "schema.md",
            "---\nidentity: canonical-schema\nstatus: stable\n---\nSchema\n",
        )

        store = IndexStore()
        store.refresh()

        assert store.outlinks("source") == ["target", "root"]
        assert store.backlinks("target") == ["source", "current-state"]
        assert store.outlinks("current-state") == ["target", "schema"]
        assert store.backlinks("schema") == ["current-state"]
        source = store.meta("source")
        assert source is not None
        assert source["description"] == "Canonical description"
        assert source["classification_status"] == "active"
        assert source["updated"] == "2026-08-10"
        assert source["relative_path"] == "notes/source.md"
        system_schema = store.meta("schema")
        assert system_schema is not None
        assert system_schema["namespace"] == "system"
        assert system_schema["relative_path"] == "schema.md"

    def test_boundary_violations_and_reserved_pages_are_excluded(
        self, isolated_index: Path, tmp_path: Path
    ) -> None:
        canonical = "---\ntitle: Reserved\nstatus: stable\n---\nbody\n"
        for relative in ("index.md", "log.md", "schema.md", "nested/index.md"):
            _seed(isolated_index, relative, canonical)
        _seed(
            isolated_index,
            "pages-to-system.md",
            "---\ntitle: Denied\nstatus: stable\n---\n[Private](/system/private.md)\n",
        )
        _seed(
            isolated_index,
            "traversal.md",
            "---\ntitle: Escape\nstatus: stable\n---\n[Escape](../outside.md)\n",
        )
        outside = tmp_path / "outside.md"
        outside.write_text("---\ntitle: Outside\nstatus: stable\n---\nbody\n")
        (isolated_index / "pages" / "symlink.md").symlink_to(outside)

        store = IndexStore()
        store._refresh_locked()

        assert store.all_page_ids() == set()
        assert store.meta("index") is None
        assert store.meta("log") is None
        assert store.meta("schema") is None
        assert store.meta("pages-to-system") is None
        assert store.meta("traversal") is None
        assert store.meta("symlink") is None
        stat = outside.stat()
        assert (
            IndexStore._build_entry(
                "outside", outside, False, stat.st_mtime_ns, stat.st_size
            )
            is None
        )

    def test_symlinked_namespace_root_is_rejected(
        self, isolated_index: Path
    ) -> None:
        pages = isolated_index / "pages"
        real_pages = isolated_index / "real-pages"
        pages.rmdir()
        real_pages.mkdir()
        (real_pages / "page.md").write_text(
            "---\ntitle: Page\nstatus: stable\n---\nbody\n"
        )
        pages.symlink_to(real_pages, target_is_directory=True)

        store = IndexStore()
        store._refresh_locked()

        assert store.all_page_ids() == set()

    def test_default_collections_are_stable_only_but_exact_lookup_is_not(
        self, isolated_index: Path
    ) -> None:
        for page_id, status in (
            ("stable-target", "stable"),
            ("draft-target", "draft"),
            ("deprecated-target", "deprecated"),
        ):
            _seed(
                isolated_index,
                f"{page_id}.md",
                f"---\ntitle: {page_id}\nstatus: {status}\n---\nbody\n",
            )
        _seed(
            isolated_index,
            "source.md",
            """---
title: Source
status: stable
---
[Stable](stable-target.md)
[Draft](draft-target.md)
[Deprecated](deprecated-target.md)
""",
        )

        store = IndexStore()
        store.refresh()

        assert store.all_page_ids() == {"source", "stable-target"}
        assert {item["page_id"] for item in store.all_pages_meta()} == {
            "source",
            "stable-target",
        }
        assert store.page_count() == 2
        assert store.outlinks("source") == ["stable-target"]
        assert store.meta("draft-target")["status"] == "draft"
        assert store.meta("deprecated-target")["status"] == "deprecated"


def test_page_id_resolver_validates_only_exact_filename_candidates(
    isolated_index: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index in range(40):
        _seed(
            isolated_index,
            f"bulk/{index}/other-{index}.md",
            f"---\ntitle: Other {index}\nstatus: stable\n---\nbody\n",
        )
    target = _seed(
        isolated_index,
        "nested/target.md",
        "---\ntitle: Target\nstatus: stable\n---\nbody\n",
    )
    calls: list[Path] = []
    real = index_store_mod.canonical_document_path

    def tracking(path: Path, *args: object, **kwargs: object) -> Path | None:
        calls.append(path)
        return real(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(index_store_mod, "canonical_document_path", tracking)

    assert index_store_mod.canonical_document_path_for_id(
        "target",
        pages_dir=isolated_index / "pages",
        system_dir=isolated_index / "system",
    ) == target.resolve()
    assert calls == [target, isolated_index / "system" / "target.md"]

    calls.clear()
    assert index_store_mod.canonical_document_path_for_id(
        "../target",
        pages_dir=isolated_index / "pages",
        system_dir=isolated_index / "system",
    ) is None
    assert calls == []
