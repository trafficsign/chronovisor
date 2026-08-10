from __future__ import annotations

import json
import sqlite3

import pytest

from chronovisor.core import index_store as index_store_mod
from chronovisor.core import search
from chronovisor.core.knowledge_graph_retrieval import CommunityCandidate
from chronovisor.core.runtime_config import EmbeddingConfig, SearchEmbeddingConfig
from chronovisor.core.search import (
    ScoredPage,
    apply_filters,
    fuse_results,
    usage_prior_results,
)


def test_semantic_search_strict_mode_surfaces_backend_unavailability(
    monkeypatch,
) -> None:
    monkeypatch.setattr(search, "_embedding_count", lambda **_kwargs: 0)
    monkeypatch.setattr(search, "_search_embedding_profile", lambda: ("test", "", ""))

    assert search.semantic_search("test") == []
    with pytest.raises(RuntimeError, match="index has no embeddings"):
        search.semantic_search("test", strict=True)


def test_semantic_search_strict_mode_surfaces_embedding_failure(
    monkeypatch,
) -> None:
    monkeypatch.setattr(search, "_embedding_count", lambda **_kwargs: 1)
    monkeypatch.setattr(
        search,
        "_search_embedding_profile",
        lambda: ("test", "", ""),
    )
    monkeypatch.setattr(
        search,
        "_embed_search_texts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    with pytest.raises(RuntimeError, match="offline"):
        search.semantic_search("test")
    with pytest.raises(RuntimeError, match="offline"):
        search.semantic_search("test", strict=True)


def test_read_only_embedding_probe_does_not_create_missing_database(
    tmp_path, monkeypatch
) -> None:
    database = tmp_path / "missing" / "embeddings.sqlite"
    monkeypatch.setattr(search, "EMBEDDINGS_DB", database)
    monkeypatch.setattr(search, "JSON_EMBEDDINGS_FILE", tmp_path / "missing.json")
    monkeypatch.setattr(search, "_json_import_done", False)
    monkeypatch.setenv("CHRONOVISOR_READ_ONLY", "1")

    assert search._embedding_count() == 0
    assert not database.exists()


def test_json_embedding_import_uses_raw_connection_once(tmp_path, monkeypatch) -> None:
    database = tmp_path / ".index" / "embeddings.sqlite"
    legacy = tmp_path / ".embeddings.json"
    legacy.write_text(
        json.dumps({"legacy-page": {"vector": [3.0, 4.0], "mtime": 12.5}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(search, "EMBEDDINGS_DB", database)
    monkeypatch.setattr(search, "JSON_EMBEDDINGS_FILE", legacy)
    monkeypatch.setattr(search, "_json_import_done", False)
    monkeypatch.setattr(
        search,
        "load_embedding_config",
        lambda: EmbeddingConfig(model=search.EMBED_MODEL),
    )

    first = search._load_embedding("legacy-page")
    second = search._load_embedding("legacy-page")

    assert first == ([3.0, 4.0], 12.5, 5.0)
    assert second == first
    assert search._json_import_done is True


def test_bm25_read_only_refresh_persists_dirty_cache_on_next_normal_build(
    tmp_path, monkeypatch
) -> None:
    page_path = tmp_path / "page.md"
    page_path.write_text(
        "---\ntitle: Page\nupdated: 2026-01-01\n---\nsearchable body\n"
    )
    cache_path = tmp_path / ".index" / "bm25.json"
    monkeypatch.setattr(search, "_BM25_CACHE_FILE", cache_path)
    monkeypatch.setattr(search, "searchable_pages", lambda: [page_path])
    monkeypatch.setenv("CHRONOVISOR_READ_ONLY", "1")
    index = search.BM25Index()

    index.build()

    assert not cache_path.exists()
    monkeypatch.delenv("CHRONOVISOR_READ_ONLY")
    index.build()
    assert cache_path.exists()


def page(
    page_id: str,
    score: float,
    *,
    status: str = "active",
    folder: str = "",
    page_type: str = "knowledge",
) -> ScoredPage:
    return ScoredPage(
        page_id=page_id,
        title=page_id,
        folder=folder,
        updated="2026-06-11",
        score=score,
        status=status,
        page_type=page_type,
    )


def test_fusion_keeps_strong_bm25_match_ahead_of_semantic_only_neighbor() -> None:
    results = fuse_results(
        bm25_results=[page("exact", 100.0), page("neighbor", 10.0)],
        semantic_results=[page("neighbor", 0.9), page("other", 0.8)],
    )

    assert [result.page_id for result in results[:2]] == ["exact", "neighbor"]


def test_associative_graph_reaches_two_hops_with_path_trace(monkeypatch) -> None:
    class FakeStore:
        def refresh(self):
            return None

        def outlinks(self, page_id):
            return {
                "seed": ["middle", "draft-target"],
                "middle": ["target"],
            }.get(page_id, [])

        def backlinks(self, _page_id):
            return []

        def tags(self, _page_id):
            return []

        def pages_for_tag(self, _tag):
            return []

        def pages_for_entity(self, _entity):
            return []

        def meta(self, page_id):
            if page_id not in {"seed", "middle", "target", "draft-target"}:
                return None
            return {
                "page_id": page_id,
                "title": page_id,
                "updated": "2026-07-24",
                "path": f"/tmp/pages/topic/{page_id}.md",
                "status": "draft" if page_id == "draft-target" else "stable",
                "entities": [],
            }

    from chronovisor.core import index_store

    monkeypatch.setattr(index_store, "get_store", lambda: FakeStore())
    monkeypatch.setattr(
        "chronovisor.core.cofire.neighbors",
        lambda *_args, **_kwargs: [],
    )

    expanded = search.graph_expand_results(
        [page("seed", 1.0)],
        decay=0.3,
        limit=50,
    )
    paths = search.graph_expansion_trace()

    assert [result.page_id for result in expanded] == ["middle", "target"]
    assert "draft-target" not in paths
    assert paths["target"]["path"] == ["seed", "middle", "target"]
    assert paths["target"]["hops"] == 2


def test_global_query_uses_community_branch_without_relation_traversal(
    monkeypatch,
) -> None:
    class FakeStore:
        def refresh(self):
            return None

        def meta(self, page_id):
            if page_id not in {"seed", "global-target"}:
                return None
            return {
                "page_id": page_id,
                "title": page_id,
                "updated": "2026-08-01",
                "path": f"/tmp/pages/{page_id}.md",
                "status": "active",
                "entities": [],
            }

    from chronovisor.core import graph_edges, index_store
    from chronovisor.core import knowledge_graph_retrieval as retrieval

    monkeypatch.setattr(index_store, "get_store", lambda: FakeStore())
    monkeypatch.setattr(
        retrieval,
        "community_candidates",
        lambda *_args, **_kwargs: [
            CommunityCandidate(
                page_id="global-target",
                score=0.8,
                community_id="community-1",
                relation_ids=("rel-1",),
                source_digests=("a" * 64,),
                summary_sha256="b" * 64,
            )
        ],
    )
    relation_flags: list[bool] = []

    def no_relation_traversal(*_args, **kwargs):
        relation_flags.append(bool(kwargs.get("include_typed_relations")))
        return []

    monkeypatch.setattr(graph_edges, "typed_neighbors", no_relation_traversal)
    search._GRAPH_QUERY.value = "全体をまとめて"
    try:
        expanded = search.graph_expand_results([page("seed", 1.0)])
    finally:
        search._GRAPH_QUERY.value = ""
    trace = search.graph_expansion_trace()

    assert [row.page_id for row in expanded] == ["global-target"]
    assert trace["global-target"]["community_id"] == "community-1"
    assert relation_flags == [False]


def test_fusion_bm25_bonus_is_parameterized() -> None:
    without_bonus = fuse_results(
        bm25_results=[page("exact", 100.0)],
        semantic_results=[page("semantic", 0.9)],
        weights={
            "bm25": 0.1,
            "semantic": 1.0,
            "bm25_score_bonus": 0.0,
            "bm25_rank_bonus": 0.0,
            "bm25_rank_decay": 0.0,
        },
    )
    with_bonus = fuse_results(
        bm25_results=[page("exact", 100.0)],
        semantic_results=[page("semantic", 0.9)],
        weights={
            "bm25": 0.1,
            "semantic": 1.0,
            "bm25_score_bonus": 0.0,
            "bm25_rank_bonus": 0.2,
            "bm25_rank_decay": 0.0,
        },
    )

    assert without_bonus[0].page_id == "semantic"
    assert with_bonus[0].page_id == "exact"


def test_fusion_degrades_flat_semantic_channel() -> None:
    results = fuse_results(
        bm25_results=[page("bm25", 100.0)],
        semantic_results=[page("semantic-a", 0.500), page("semantic-b", 0.499)],
        weights={
            "bm25": 1.0,
            "semantic": 1.0,
            "bm25_score_bonus": 0.0,
            "bm25_rank_bonus": 0.0,
            "bm25_rank_decay": 0.0,
            "semantic_min_top_score": 0.45,
            "semantic_min_margin": 0.01,
            "semantic_low_confidence_weight": 0.0,
        },
    )

    assert results[0].page_id == "bm25"


def test_searchable_pages_includes_system_pages(tmp_path, monkeypatch) -> None:
    pages_dir = tmp_path / "pages"
    system_dir = tmp_path / "system"
    pages_dir.mkdir()
    system_dir.mkdir()
    page = pages_dir / "normal.md"
    system = system_dir / "claude-code.md"
    page.write_text("# Normal\n", encoding="utf-8")
    system.write_text("# System\n", encoding="utf-8")

    class FakeStore:
        def refresh(self) -> None:
            pass

        def all_page_ids(self, *, include_system: bool = False):
            assert include_system is True
            return {"normal", "claude-code"}

        def meta(self, page_id: str):
            return {"path": str(page if page_id == "normal" else system)}

    monkeypatch.setattr(index_store_mod, "get_store", lambda: FakeStore())

    assert search.searchable_pages() == [page, system]


def test_apply_filters_exposes_only_canonical_stable_pages() -> None:
    results = apply_filters(
        [
            page("stable", 1.0, status="stable"),
            page("draft", 2.0, status="draft"),
            page("old", 2.0, status="deprecated"),
            page("gone", 3.0, status="archived"),
        ]
    )

    assert [(result.page_id, result.status) for result in results] == [
        ("stable", "stable")
    ]


def test_apply_filters_excludes_reference_pages_by_default() -> None:
    results = apply_filters(
        [
            page("knowledge", 1.0),
            page("car", 2.0, folder="car-spec", page_type="reference"),
        ]
    )

    assert [result.page_id for result in results] == ["knowledge"]


def test_apply_filters_allows_reference_pages_when_folder_requested() -> None:
    results = apply_filters(
        [
            page("knowledge", 1.0),
            page("car", 2.0, folder="car-spec", page_type="reference"),
        ],
        folder="car-spec",
    )

    assert [result.page_id for result in results] == ["car"]


def test_fusion_preserves_lifecycle_status_for_filtering() -> None:
    results = fuse_results(
        bm25_results=[page("old", 100.0, status="deprecated")],
        semantic_results=[],
        weights={"bm25": 1.0},
    )

    assert results[0].status == "deprecated"
    assert apply_filters(results) == []


class _CanonicalPageStore:
    def __init__(self, *paths) -> None:
        self.paths = {path.stem: path for path in paths}

    def refresh(self) -> None:
        pass

    def all_page_ids(self, *, include_system: bool = False):
        assert include_system is True
        return set(self.paths)

    def meta(self, page_id: str):
        path = self.paths[page_id]
        return {
            "title": page_id.title(),
            "updated": "2026-06-11",
            "path": str(path),
            "mtime_ns": path.stat().st_mtime_ns,
            "is_system": path.parent.name == "system",
            "status": "stable",
            "recall_questions": [],
            "entities": [],
            "page_type": "knowledge",
            "sensitivity": "normal",
        }


def test_update_embeddings_rebuilds_when_model_profile_changes(
    tmp_path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    pages_dir = chronovisor_root / "pages"
    system_dir = chronovisor_root / "system"
    index_dir = chronovisor_root / ".index"
    pages_dir.mkdir(parents=True)
    system_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    page_path = pages_dir / "p.md"
    page_path.write_text(
        "---\ntitle: P\nupdated: 2026-06-11\nstatus: stable\n---\nbody\n",
        encoding="utf-8",
    )
    db_path = index_dir / "embeddings.sqlite"

    monkeypatch.setattr(search, "EMBEDDINGS_DB", db_path)
    monkeypatch.setattr(
        search, "JSON_EMBEDDINGS_FILE", chronovisor_root / ".embeddings.json"
    )
    monkeypatch.setattr(
        index_store_mod, "get_store", lambda: _CanonicalPageStore(page_path)
    )
    monkeypatch.setattr(search, "_json_import_done", True)
    monkeypatch.setattr(
        search,
        "load_search_embedding_config",
        lambda: SearchEmbeddingConfig(backend="legacy_ollama"),
    )

    profile = {"value": ("m1", "D1:", "Q1:")}
    calls: list[tuple[str | None, list[str]]] = []

    def fake_embed(texts, *, source, timeout_ms=None):
        del source, timeout_ms
        model = profile["value"][0]
        calls.append((model, list(texts)))
        return [[float(len(text)), 1.0] for text in texts], model

    monkeypatch.setattr(search, "_search_embedding_profile", lambda: profile["value"])
    monkeypatch.setattr(search, "_embed_search_texts", fake_embed)

    assert search.update_embeddings() == 1
    assert calls[-1][0] == "m1"
    assert calls[-1][1][0].startswith("D1:")
    assert search._embedding_count(model="m1", text_prefix="D1:") == 1

    profile["value"] = ("m2", "D2:", "Q2:")
    assert search._embedding_count(model="m2", text_prefix="D2:") == 0
    assert search.update_embeddings() == 1
    assert calls[-1][0] == "m2"
    assert calls[-1][1][0].startswith("D2:")
    assert search._embedding_count(model="m2", text_prefix="D2:") == 1

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT model, text_prefix, COUNT(*) "
            "FROM embeddings GROUP BY model, text_prefix"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("m2", "D2:", 1)]


def test_update_embeddings_preserves_old_rows_on_runtime_failure(
    tmp_path, monkeypatch
) -> None:
    page_path = tmp_path / "pages" / "p.md"
    page_path.parent.mkdir()
    page_path.write_text("---\ntitle: P\nstatus: stable\n---\nbody\n", encoding="utf-8")
    database = tmp_path / ".index" / "embeddings.sqlite"
    monkeypatch.setattr(search, "EMBEDDINGS_DB", database)
    monkeypatch.setattr(search, "JSON_EMBEDDINGS_FILE", tmp_path / "missing.json")
    monkeypatch.setattr(search, "_json_import_done", True)
    monkeypatch.setattr(
        index_store_mod, "get_store", lambda: _CanonicalPageStore(page_path)
    )
    monkeypatch.setattr(
        search,
        "load_search_embedding_config",
        lambda: SearchEmbeddingConfig(backend="legacy_ollama"),
    )
    monkeypatch.setattr(search, "_search_embedding_profile", lambda: ("new", "", ""))
    monkeypatch.setattr(
        search,
        "_embed_search_texts",
        lambda texts, **_kwargs: ([[1.0, 0.0] for _ in texts], "wrong"),
    )
    search._replace_search_embeddings(
        [("p", [0.0, 1.0], 1.0)],
        [],
        [],
        page_ids={"p"},
        model="old",
        text_prefix="",
    )

    with pytest.raises(RuntimeError, match="route identity"):
        search.update_embeddings()

    monkeypatch.setattr(
        search,
        "_embed_search_texts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )
    for strict in (False, True):
        with pytest.raises(OSError, match="offline"):
            search.update_embeddings(strict=strict)

    conn = sqlite3.connect(database)
    try:
        assert conn.execute("SELECT model FROM embeddings").fetchall() == [("old",)]
    finally:
        conn.close()


def test_update_embeddings_stores_markdown_chunks(tmp_path, monkeypatch) -> None:
    chronovisor_root = tmp_path / "wiki"
    pages_dir = chronovisor_root / "pages"
    system_dir = chronovisor_root / "system"
    index_dir = chronovisor_root / ".index"
    pages_dir.mkdir(parents=True)
    system_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    page_path = pages_dir / "chunky.md"
    page_path.write_text(
        "---\ntitle: Chunky\nupdated: 2026-06-11\nstatus: stable\n---\n"
        "# First Heading\n"
        "This is a paragraph about early context.\n\n"
        "## Late Heading\n"
        "This later paragraph mentions the hidden retrieval target.\n",
        encoding="utf-8",
    )
    db_path = index_dir / "embeddings.sqlite"

    monkeypatch.setattr(search, "EMBEDDINGS_DB", db_path)
    monkeypatch.setattr(
        search, "JSON_EMBEDDINGS_FILE", chronovisor_root / ".embeddings.json"
    )
    monkeypatch.setattr(
        index_store_mod, "get_store", lambda: _CanonicalPageStore(page_path)
    )
    monkeypatch.setattr(search, "_json_import_done", True)
    monkeypatch.setattr(
        search,
        "load_search_embedding_config",
        lambda: SearchEmbeddingConfig(backend="legacy_ollama"),
    )
    monkeypatch.setattr(search, "_search_embedding_profile", lambda: ("m", "", ""))
    sources = []

    def fake_embed(texts, *, source, timeout_ms=None):
        del timeout_ms
        sources.append(source)
        return [[float(i + 1), 1.0] for i, _ in enumerate(texts)], "m"

    monkeypatch.setattr(search, "_embed_search_texts", fake_embed)

    assert search.update_embeddings() == 1
    assert {source.data_class for source in sources} == {search.SourceDataClass.PAGE}
    assert {source.sensitivity for source in sources} == {
        search.SourceSensitivity.NORMAL
    }

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT page_id, chunk_idx, text FROM chunk_embeddings ORDER BY chunk_idx"
        ).fetchall()
    finally:
        conn.close()
    assert rows
    assert rows[0][0] == "chunky"
    assert any("Late Heading" in row[2] for row in rows)


def test_update_embeddings_prunes_rows_for_deleted_pages(tmp_path, monkeypatch) -> None:
    chronovisor_root = tmp_path / "wiki"
    pages_dir = chronovisor_root / "pages"
    system_dir = chronovisor_root / "system"
    index_dir = chronovisor_root / ".index"
    pages_dir.mkdir(parents=True)
    system_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    page_path = pages_dir / "current.md"
    page_path.write_text(
        "---\ntitle: Current\nstatus: stable\n---\nbody\n", encoding="utf-8"
    )
    db_path = index_dir / "embeddings.sqlite"

    monkeypatch.setattr(search, "EMBEDDINGS_DB", db_path)
    monkeypatch.setattr(
        search, "JSON_EMBEDDINGS_FILE", chronovisor_root / ".embeddings.json"
    )
    monkeypatch.setattr(
        index_store_mod, "get_store", lambda: _CanonicalPageStore(page_path)
    )
    monkeypatch.setattr(search, "_json_import_done", True)
    monkeypatch.setattr(
        search,
        "load_search_embedding_config",
        lambda: SearchEmbeddingConfig(backend="legacy_ollama"),
    )
    monkeypatch.setattr(search, "_search_embedding_profile", lambda: ("m", "", ""))
    monkeypatch.setattr(
        search,
        "_embed_search_texts",
        lambda texts, **_kwargs: ([[1.0, 2.0] for _ in texts], "m"),
    )

    assert search.update_embeddings() == 1
    conn = sqlite3.connect(db_path)
    try:
        vector, mtime, norm, dim, model, prefix = conn.execute(
            "SELECT vector, mtime, norm, dim, model, text_prefix FROM embeddings "
            "WHERE page_id = 'current'"
        ).fetchone()
        conn.execute(
            "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("retired-page", vector, mtime, norm, dim, model, prefix),
        )
        conn.execute(
            "INSERT INTO question_embeddings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "retired-page#q0",
                "retired-page",
                0,
                "old question",
                vector,
                mtime,
                norm,
                dim,
                model,
                prefix,
            ),
        )
        conn.execute(
            "INSERT INTO chunk_embeddings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "retired-page#c0",
                "retired-page",
                0,
                "old chunk",
                vector,
                mtime,
                norm,
                dim,
                model,
                prefix,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    assert search.update_embeddings() == 0
    conn = sqlite3.connect(db_path)
    try:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE page_id = 'retired-page'"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM question_embeddings WHERE page_id = 'retired-page'"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM chunk_embeddings WHERE page_id = 'retired-page'"
            ).fetchone()[0]
            == 0
        )
    finally:
        conn.close()


def test_semantic_search_applies_query_prefix(tmp_path, monkeypatch) -> None:
    captured: list[tuple[str | None, list[str]]] = []

    def fake_embed(texts, **_kwargs):
        captured.append(("ruri", list(texts)))
        return [[1.0, 0.0] for _ in texts], "ruri"

    class FakeStore:
        def refresh(self) -> None:
            pass

        def meta(self, page_id: str):
            return {
                "title": page_id,
                "updated": "2026-06-11",
                "path": str(tmp_path / f"{page_id}.md"),
                "status": "stable",
                "superseded_by": "",
            }

    monkeypatch.setattr(
        search,
        "_search_embedding_profile",
        lambda: ("ruri", "検索文書: ", "検索クエリ: "),
    )
    monkeypatch.setattr(search, "_embed_search_texts", fake_embed)
    monkeypatch.setattr(search, "_embedding_count", lambda **_kwargs: 1)
    monkeypatch.setattr(
        search,
        "_iter_all_embeddings",
        lambda **_kwargs: [("p", [1.0, 0.0], 0.0, 1.0)],
    )
    monkeypatch.setattr(search, "_iter_all_question_embeddings", lambda **_kwargs: [])
    monkeypatch.setattr(index_store_mod, "get_store", lambda: FakeStore())

    results = search.semantic_search("テスト", top_n=1)

    assert [result.page_id for result in results] == ["p"]
    assert captured[0] == ("ruri", ["検索クエリ: テスト"])


def test_semantic_search_aggregates_chunk_hits(tmp_path, monkeypatch) -> None:
    captured: list[tuple[str | None, list[str]]] = []

    def fake_embed(texts, **_kwargs):
        captured.append(("m", list(texts)))
        return [[1.0, 0.0] for _ in texts], "m"

    class FakeStore:
        def refresh(self) -> None:
            pass

        def meta(self, page_id: str):
            return {
                "title": page_id,
                "updated": "2026-06-11",
                "path": str(tmp_path / f"{page_id}.md"),
                "status": "stable",
                "superseded_by": "",
            }

    monkeypatch.setattr(search, "_search_embedding_profile", lambda: ("m", "", ""))
    monkeypatch.setattr(search, "_embed_search_texts", fake_embed)
    monkeypatch.setattr(search, "_embedding_count", lambda **_kwargs: 1)
    monkeypatch.setattr(search, "_iter_all_embeddings", lambda **_kwargs: [])
    monkeypatch.setattr(search, "_iter_all_question_embeddings", lambda **_kwargs: [])
    monkeypatch.setattr(
        search,
        "_iter_all_chunk_embeddings",
        lambda **_kwargs: [("p#c0", "p", 0, "late chunk text", [1.0, 0.0], 0.0, 1.0)],
    )
    monkeypatch.setattr(index_store_mod, "get_store", lambda: FakeStore())

    results = search.semantic_search("late target", top_n=1)

    assert [result.page_id for result in results] == ["p"]
    assert results[0].snippet == "late chunk text"


def test_semantic_search_skips_chunks_for_confident_page_hits(
    tmp_path, monkeypatch
) -> None:
    def fake_embed(texts, **_kwargs):
        return [[1.0, 0.0] for _ in texts], "m"

    class FakeStore:
        def refresh(self) -> None:
            pass

        def meta(self, page_id: str):
            return {
                "title": page_id,
                "updated": "2026-06-11",
                "path": str(tmp_path / f"{page_id}.md"),
                "status": "stable",
                "superseded_by": "",
            }

    def fail_if_scanned(**_kwargs):
        raise AssertionError(
            "chunk embeddings should not be scanned for confident page hits"
        )

    monkeypatch.setattr(search, "_search_embedding_profile", lambda: ("m", "", ""))
    monkeypatch.setattr(search, "_embed_search_texts", fake_embed)
    monkeypatch.setattr(search, "_embedding_count", lambda **_kwargs: 2)
    monkeypatch.setattr(
        search,
        "_iter_all_embeddings",
        lambda **_kwargs: [
            ("p", [1.0, 0.0], 0.0, 1.0),
            ("other", [0.0, 1.0], 0.0, 1.0),
        ],
    )
    monkeypatch.setattr(search, "_iter_all_question_embeddings", lambda **_kwargs: [])
    monkeypatch.setattr(search, "_iter_all_chunk_embeddings", fail_if_scanned)
    monkeypatch.setattr(index_store_mod, "get_store", lambda: FakeStore())

    results = search.semantic_search("clear target", top_n=1)

    assert [result.page_id for result in results] == ["p"]


def test_usage_prior_applies_recency_decay_and_cap(tmp_path, monkeypatch) -> None:
    chronovisor_root = tmp_path / "wiki"
    recall_dir = chronovisor_root / "recall"
    recall_dir.mkdir(parents=True)
    rows = [
        {"kind": "injection_used", "expected_pages": ["p"]},
        {"kind": "injection_used", "expected_pages": ["p"]},
        {"kind": "injection_used", "expected_pages": ["p"]},
    ]
    (recall_dir / "feedback.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    class FakeStore:
        def refresh(self) -> None:
            pass

        def meta(self, page_id: str):
            return {
                "title": page_id,
                "updated": "2026-06-11",
                "path": str(tmp_path / f"{page_id}.md"),
                "status": "active",
                "superseded_by": "",
            }

    monkeypatch.setattr(search, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(index_store_mod, "get_store", lambda: FakeStore())

    results = usage_prior_results({"p"}, decay=0.5, cap=1.2)

    assert len(results) == 1
    assert results[0].page_id == "p"
    assert results[0].score == 1.2


def test_active_fusion_policy_fails_closed_on_partial_or_invalid_weights(
    tmp_path,
) -> None:
    policy = tmp_path / "search-policy.json"
    policy.write_text(
        json.dumps(
            {"weights": {"semantic": 0.7, "graph": -1, "unknown": 99, "bm25": "bad"}}
        ),
        encoding="utf-8",
    )

    weights = search.load_active_fusion_weights(policy)

    assert weights == search.DEFAULT_FUSION_WEIGHTS


def test_active_fusion_policy_accepts_complete_versioned_artifact(tmp_path) -> None:
    policy = tmp_path / "search-policy.json"
    expected = {**search.DEFAULT_FUSION_WEIGHTS, "semantic": 0.7}
    policy.write_text(
        json.dumps(
            {
                "version": 1,
                "source": "search_eval.self_tune",
                "holdout": {"mrr": 0.8},
                "weights": expected,
            }
        ),
        encoding="utf-8",
    )

    assert search.load_active_fusion_weights(policy) == expected


def test_active_fusion_policy_rejects_all_zero_retrieval_channels(tmp_path) -> None:
    policy = tmp_path / "search-policy.json"
    weights = {
        **search.DEFAULT_FUSION_WEIGHTS,
        "anchor": 0.0,
        "bm25": 0.0,
        "semantic": 0.0,
        "graph": 0.0,
        "context": 0.0,
        "usage_prior": 0.0,
    }
    policy.write_text(
        json.dumps(
            {
                "version": 1,
                "source": "search_eval.self_tune",
                "holdout": {},
                "weights": weights,
            }
        ),
        encoding="utf-8",
    )

    assert search.load_active_fusion_weights(policy) == search.DEFAULT_FUSION_WEIGHTS
