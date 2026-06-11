from __future__ import annotations

import sqlite3

from llm_wiki_mcp import search
from llm_wiki_mcp import ollama
from llm_wiki_mcp import index_store as index_store_mod
from llm_wiki_mcp.runtime_config import EmbeddingConfig
from llm_wiki_mcp.search import ScoredPage, apply_filters, fuse_results


def page(page_id: str, score: float, *, status: str = "active") -> ScoredPage:
    return ScoredPage(
        page_id=page_id,
        title=page_id,
        folder="",
        updated="2026-06-11",
        score=score,
        status=status,
    )


def test_fusion_keeps_strong_bm25_match_ahead_of_semantic_only_neighbor() -> None:
    results = fuse_results(
        bm25_results=[page("exact", 100.0), page("neighbor", 10.0)],
        semantic_results=[page("neighbor", 0.9), page("other", 0.8)],
    )

    assert [result.page_id for result in results[:2]] == ["exact", "neighbor"]


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
    monkeypatch.setattr(search, "SYSTEM_DIR", system_dir)
    monkeypatch.setattr(search, "all_pages", lambda: [page])

    assert search.searchable_pages() == [page, system]


def test_apply_filters_excludes_deprecated_and_archived_pages() -> None:
    results = apply_filters(
        [
            page("active", 1.0),
            page("old", 2.0, status="deprecated"),
            page("gone", 3.0, status="archived"),
        ]
    )

    assert [result.page_id for result in results] == ["active"]


def test_fusion_preserves_lifecycle_status_for_filtering() -> None:
    results = fuse_results(
        bm25_results=[page("old", 100.0, status="deprecated")],
        semantic_results=[],
        weights={"bm25": 1.0},
    )

    assert results[0].status == "deprecated"
    assert apply_filters(results) == []


def test_update_embeddings_rebuilds_when_model_profile_changes(tmp_path, monkeypatch) -> None:
    wiki_root = tmp_path / "wiki"
    pages_dir = wiki_root / "pages"
    system_dir = wiki_root / "system"
    index_dir = wiki_root / ".index"
    pages_dir.mkdir(parents=True)
    system_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    page_path = pages_dir / "p.md"
    page_path.write_text(
        "---\ntitle: P\nupdated: 2026-06-11\n---\nbody\n",
        encoding="utf-8",
    )
    db_path = index_dir / "embeddings.sqlite"

    monkeypatch.setattr(search, "PAGES_DIR", pages_dir)
    monkeypatch.setattr(search, "SYSTEM_DIR", system_dir)
    monkeypatch.setattr(search, "EMBEDDINGS_DB", db_path)
    monkeypatch.setattr(search, "LEGACY_EMBEDDINGS_FILE", wiki_root / ".embeddings.json")
    monkeypatch.setattr(search, "all_pages", lambda: [page_path])
    monkeypatch.setattr(search, "_legacy_migration_done", True)
    monkeypatch.setattr(ollama, "is_available", lambda: True)

    profile = {"value": EmbeddingConfig(model="m1", document_prefix="D1:", query_prefix="Q1:")}
    calls: list[tuple[str | None, list[str]]] = []

    def fake_embed(texts, *, model=None):
        calls.append((model, list(texts)))
        return [[float(len(text)), 1.0] for text in texts]

    monkeypatch.setattr(search, "load_embedding_config", lambda: profile["value"])
    monkeypatch.setattr(ollama, "embed", fake_embed)

    assert search.update_embeddings() == 1
    assert calls[-1][0] == "m1"
    assert calls[-1][1][0].startswith("D1:")
    assert search._embedding_count() == 1

    profile["value"] = EmbeddingConfig(model="m2", document_prefix="D2:", query_prefix="Q2:")
    assert search._embedding_count() == 0
    assert search.update_embeddings() == 1
    assert calls[-1][0] == "m2"
    assert calls[-1][1][0].startswith("D2:")
    assert search._embedding_count() == 1

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT model, text_prefix, COUNT(*) FROM embeddings GROUP BY model, text_prefix"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("m2", "D2:", 1)]


def test_update_embeddings_stores_markdown_chunks(tmp_path, monkeypatch) -> None:
    wiki_root = tmp_path / "wiki"
    pages_dir = wiki_root / "pages"
    system_dir = wiki_root / "system"
    index_dir = wiki_root / ".index"
    pages_dir.mkdir(parents=True)
    system_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    page_path = pages_dir / "chunky.md"
    page_path.write_text(
        "---\ntitle: Chunky\nupdated: 2026-06-11\n---\n"
        "# First Heading\n"
        "This is a paragraph about early context.\n\n"
        "## Late Heading\n"
        "This later paragraph mentions the hidden retrieval target.\n",
        encoding="utf-8",
    )
    db_path = index_dir / "embeddings.sqlite"

    monkeypatch.setattr(search, "PAGES_DIR", pages_dir)
    monkeypatch.setattr(search, "SYSTEM_DIR", system_dir)
    monkeypatch.setattr(search, "EMBEDDINGS_DB", db_path)
    monkeypatch.setattr(search, "LEGACY_EMBEDDINGS_FILE", wiki_root / ".embeddings.json")
    monkeypatch.setattr(search, "all_pages", lambda: [page_path])
    monkeypatch.setattr(search, "_legacy_migration_done", True)
    monkeypatch.setattr(ollama, "is_available", lambda: True)
    monkeypatch.setattr(
        search,
        "load_embedding_config",
        lambda: EmbeddingConfig(model="m", document_prefix="", query_prefix=""),
    )
    monkeypatch.setattr(
        ollama,
        "embed",
        lambda texts, *, model=None: [[float(i + 1), 1.0] for i, _ in enumerate(texts)],
    )

    assert search.update_embeddings() == 1

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


def test_semantic_search_applies_query_prefix(tmp_path, monkeypatch) -> None:
    captured: list[tuple[str | None, list[str]]] = []

    def fake_embed(texts, *, model=None):
        captured.append((model, list(texts)))
        return [[1.0, 0.0] for _ in texts]

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

    monkeypatch.setattr(ollama, "is_available", lambda: True)
    monkeypatch.setattr(ollama, "embed", fake_embed)
    monkeypatch.setattr(
        search,
        "load_embedding_config",
        lambda: EmbeddingConfig(model="ruri", document_prefix="検索文書: ", query_prefix="検索クエリ: "),
    )
    monkeypatch.setattr(search, "_embedding_count", lambda: 1)
    monkeypatch.setattr(search, "_iter_all_embeddings", lambda: [("p", [1.0, 0.0], 0.0, 1.0)])
    monkeypatch.setattr(search, "_iter_all_question_embeddings", lambda: [])
    monkeypatch.setattr(index_store_mod, "get_store", lambda: FakeStore())

    results = search.semantic_search("テスト", top_n=1)

    assert [result.page_id for result in results] == ["p"]
    assert captured[0] == ("ruri", ["検索クエリ: テスト"])


def test_semantic_search_aggregates_chunk_hits(tmp_path, monkeypatch) -> None:
    captured: list[tuple[str | None, list[str]]] = []

    def fake_embed(texts, *, model=None):
        captured.append((model, list(texts)))
        return [[1.0, 0.0] for _ in texts]

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

    monkeypatch.setattr(ollama, "is_available", lambda: True)
    monkeypatch.setattr(ollama, "embed", fake_embed)
    monkeypatch.setattr(
        search,
        "load_embedding_config",
        lambda: EmbeddingConfig(model="m", document_prefix="", query_prefix=""),
    )
    monkeypatch.setattr(search, "_embedding_count", lambda: 1)
    monkeypatch.setattr(search, "_iter_all_embeddings", lambda: [])
    monkeypatch.setattr(search, "_iter_all_question_embeddings", lambda: [])
    monkeypatch.setattr(
        search,
        "_iter_all_chunk_embeddings",
        lambda: [("p#c0", "p", 0, "late chunk text", [1.0, 0.0], 0.0, 1.0)],
    )
    monkeypatch.setattr(index_store_mod, "get_store", lambda: FakeStore())

    results = search.semantic_search("late target", top_n=1)

    assert [result.page_id for result in results] == ["p"]
    assert results[0].snippet == "late chunk text"


def test_semantic_search_skips_chunks_for_confident_page_hits(tmp_path, monkeypatch) -> None:
    def fake_embed(texts, *, model=None):
        return [[1.0, 0.0] for _ in texts]

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

    def fail_if_scanned():
        raise AssertionError("chunk embeddings should not be scanned for confident page hits")

    monkeypatch.setattr(ollama, "is_available", lambda: True)
    monkeypatch.setattr(ollama, "embed", fake_embed)
    monkeypatch.setattr(
        search,
        "load_embedding_config",
        lambda: EmbeddingConfig(model="m", document_prefix="", query_prefix=""),
    )
    monkeypatch.setattr(search, "_embedding_count", lambda: 2)
    monkeypatch.setattr(
        search,
        "_iter_all_embeddings",
        lambda: [
            ("p", [1.0, 0.0], 0.0, 1.0),
            ("other", [0.0, 1.0], 0.0, 1.0),
        ],
    )
    monkeypatch.setattr(search, "_iter_all_question_embeddings", lambda: [])
    monkeypatch.setattr(search, "_iter_all_chunk_embeddings", fail_if_scanned)
    monkeypatch.setattr(index_store_mod, "get_store", lambda: FakeStore())

    results = search.semantic_search("clear target", top_n=1)

    assert [result.page_id for result in results] == ["p"]
