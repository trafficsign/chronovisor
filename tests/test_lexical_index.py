from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest

from chronovisor.core.index_store import IndexStore
from chronovisor.core.lexical_index import LexicalIndex


def test_inverted_bm25_and_anchor_channels_find_japanese_and_metadata(
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    pages.mkdir()
    semantic = pages / "semantic"
    semantic.mkdir()
    target = semantic / "nemotron-search.md"
    target.write_text(
        """---
title: Nemotron 検索設計
status: stable
type: knowledge
updated: 2026-07-24
tags: [d/ai-tools, t/retrieval]
entities: [NVIDIA, Nemotron]
raw_keywords: [agentic retrieval]
---
日本語の意味検索と関連ページ探索を高速化する。
""",
        encoding="utf-8",
    )
    other = semantic / "unrelated.md"
    other.write_text(
        """---
title: unrelated
status: stable
type: knowledge
updated: 2026-07-24
---
別の記録。
""",
        encoding="utf-8",
    )
    index = LexicalIndex(
        path=tmp_path / "lexical.sqlite",
        pages=lambda: [target, other],
        refresh_interval_seconds=0,
    )

    index.build()

    assert index.query("意味検索", top_n=2)[0].page_id == "nemotron-search"
    assert index.anchor_query("NVIDIA", top_n=2)[0].page_id == "nemotron-search"
    assert index.anchor_query("retrieval", top_n=2)[0].page_id == "nemotron-search"
    assert index.stats()["backend"] == "sqlite_inverted_bm25"


def test_inverted_bm25_refresh_removes_deleted_pages(tmp_path: Path) -> None:
    page = tmp_path / "obsolete.md"
    page.write_text(
        "---\ntitle: Obsolete\nstatus: stable\ntype: knowledge\nupdated: 2026-07-24\n---\nretired token\n",
        encoding="utf-8",
    )
    index = LexicalIndex(
        path=tmp_path / "lexical.sqlite",
        pages=lambda: [page] if page.exists() else [],
        refresh_interval_seconds=0,
    )
    index.build()
    assert index.query("retired")

    page.unlink()
    index.build(force=True)

    assert index.query("retired") == []


def test_query_existing_reads_only_valid_built_projection(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text(
        "---\ntitle: Page\nstatus: stable\ntype: knowledge\n---\nexistingtoken\n",
        encoding="utf-8",
    )
    path = tmp_path / "lexical.sqlite"
    index = LexicalIndex(path=path, pages=lambda: [page], refresh_interval_seconds=0)

    assert index.query_existing("existingtoken") == []
    assert not path.exists()
    index.build()
    before = path.stat().st_mtime_ns
    rows = index.query_existing("existingtoken")
    assert [row.page_id for row in rows] == ["page"]
    assert rows[0].content_sha256 == hashlib.sha256(page.read_bytes()).hexdigest()
    assert path.stat().st_mtime_ns == before


def test_force_rebuild_refreshes_actual_content_digest_with_preserved_stat(
    tmp_path: Path,
) -> None:
    page = tmp_path / "page.md"
    page.write_text(
        "---\ntitle: Page\nstatus: stable\ntype: knowledge\n---\ndigesttoken alpha\n",
        encoding="utf-8",
    )
    index = LexicalIndex(path=tmp_path / "lexical.sqlite", pages=lambda: [page])
    index.build()
    first = index.query_existing("digesttoken")[0].content_sha256
    stat = page.stat()
    page.write_text(
        "---\ntitle: Page\nstatus: stable\ntype: knowledge\n---\ndigesttoken bravo\n",
        encoding="utf-8",
    )
    assert page.stat().st_size == stat.st_size
    os.utime(page, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    index.build(force=True)
    second = index.query_existing("digesttoken")[0].content_sha256

    assert first != second
    assert second == hashlib.sha256(page.read_bytes()).hexdigest()


def test_anchor_query_existing_reads_only_valid_built_projection(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text(
        "---\ntitle: Existing Anchor\nstatus: stable\ntype: knowledge\n"
        "entities: [FastAnchor]\n---\nbody\n",
        encoding="utf-8",
    )
    path = tmp_path / "lexical.sqlite"
    index = LexicalIndex(path=path, pages=lambda: [page], refresh_interval_seconds=0)

    assert index.anchor_query_existing("FastAnchor") == []
    assert not path.exists()
    index.build()
    before = path.stat().st_mtime_ns

    assert [row.page_id for row in index.anchor_query_existing("FastAnchor")] == [
        "page"
    ]
    assert path.stat().st_mtime_ns == before


def test_query_existing_fails_empty_for_invalid_projection(tmp_path: Path) -> None:
    path = tmp_path / "lexical.sqlite"
    path.write_text("not sqlite", encoding="utf-8")
    index = LexicalIndex(path=path, pages=lambda: [])

    assert index.query_existing("query") == []
    assert path.read_text(encoding="utf-8") == "not sqlite"


def test_query_existing_fails_empty_for_schema_mismatch(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text(
        "---\ntitle: Page\nstatus: stable\ntype: knowledge\n---\nexistingtoken\n",
        encoding="utf-8",
    )
    path = tmp_path / "lexical.sqlite"
    index = LexicalIndex(path=path, pages=lambda: [page])
    index.build()
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 999")

    assert index.query_existing("existingtoken") == []


def test_writer_rebuilds_unversioned_legacy_projection(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text(
        "---\ntitle: Page\nstatus: stable\ntype: knowledge\n---\n"
        "legacyrebuildtoken\n",
        encoding="utf-8",
    )
    path = tmp_path / "lexical.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE pages (page_id TEXT PRIMARY KEY)")

    index = LexicalIndex(path=path, pages=lambda: [page], refresh_interval_seconds=0)
    index.build()

    with sqlite3.connect(path) as connection:
        assert int(connection.execute("PRAGMA user_version").fetchone()[0]) == 8
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(pages)")
        }
    assert "content_sha256" in columns
    assert [row.page_id for row in index.query("legacyrebuildtoken")] == ["page"]


def test_lexical_index_includes_only_canonical_stable_pages_and_system(
    tmp_path: Path,
) -> None:
    pages = tmp_path / "pages"
    system = tmp_path / "system"
    pages.mkdir()
    system.mkdir()
    documents = {
        pages / "stable.md": (
            "---\ntitle: Stable\nstatus: stable\ntype: knowledge\n---\n"
            "stabletoken\n"
        ),
        pages / "draft.md": (
            "---\ntitle: Draft\nstatus: draft\ntype: knowledge\n---\ndrafttoken\n"
        ),
        pages / "deprecated.md": (
            "---\ntitle: Deprecated\nstatus: deprecated\ntype: knowledge\n---\n"
            "deprecatedtoken\n"
        ),
        pages / "missing-type.md": (
            "---\ntitle: Missing\nstatus: stable\n---\nmissingtypetoken\n"
        ),
        pages / "legacy-link.md": (
            "---\ntitle: Legacy\nstatus: stable\ntype: knowledge\n---\n"
            "[[stable]] legacytoken\n"
        ),
        system / "current-state.md": (
            "---\ntitle: Current\nstatus: stable\n---\n"
            "[Stable](</pages/stable.md>) systemtoken\n"
        ),
    }
    for path, content in documents.items():
        path.write_text(content, encoding="utf-8")
    index = LexicalIndex(
        path=tmp_path / "lexical.sqlite",
        pages=lambda: list(documents),
        refresh_interval_seconds=0,
    )

    index.build()

    assert [row.page_id for row in index.query("stabletoken")] == ["stable"]
    assert [row.page_id for row in index.query("systemtoken")] == ["current-state"]
    for excluded in ("drafttoken", "deprecatedtoken", "missingtypetoken", "legacytoken"):
        assert index.query(excluded) == []


def test_cold_reader_loads_generation_matched_projection_without_scanning(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wiki"
    pages = root / "pages"
    index_dir = root / ".index"
    pages.mkdir(parents=True)
    index_dir.mkdir()
    page = pages / "page.md"
    page.write_text(
        "---\ntitle: Page\nstatus: stable\ntype: knowledge\n"
        "uid: page-uid\n---\nprojectiontoken\n",
        encoding="utf-8",
    )
    metadata = IndexStore(root)
    metadata._refresh_locked()
    lexical = LexicalIndex(
        path=index_dir / "lexical.sqlite",
        pages=lambda: [page],
        refresh_interval_seconds=0,
    )
    lexical.build()
    generation = metadata.snapshot_generation
    lexical.close()

    reader = LexicalIndex(
        path=index_dir / "lexical.sqlite",
        pages=lambda: (_ for _ in ()).throw(
            AssertionError("cold projection load scanned canonical pages")
        ),
        refresh_interval_seconds=0,
    )
    assert generation is not None
    assert reader.load_existing(expected_generation=generation) is True
    rows = reader.query_existing("projectiontoken")
    assert [row.page_id for row in rows] == ["page"]
    assert rows[0].uid == "page-uid"


def test_generation_mismatch_is_unavailable_without_rebuilding(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wiki"
    pages = root / "pages"
    index_dir = root / ".index"
    pages.mkdir(parents=True)
    index_dir.mkdir()
    page = pages / "page.md"
    page.write_text(
        "---\ntitle: Page\nstatus: stable\ntype: knowledge\n---\nprojectiontoken\n",
        encoding="utf-8",
    )
    metadata = IndexStore(root)
    metadata._refresh_locked()
    lexical = LexicalIndex(
        path=index_dir / "lexical.sqlite",
        pages=lambda: [page],
        refresh_interval_seconds=0,
    )
    lexical.build()
    lexical.close()

    pages_doc = json.loads((index_dir / "pages.json").read_text())
    pages_doc["generation"] += 1
    (index_dir / "pages.json").write_text(json.dumps(pages_doc))
    reader = LexicalIndex(
        path=index_dir / "lexical.sqlite",
        pages=lambda: (_ for _ in ()).throw(
            AssertionError("mismatched projection triggered a rebuild")
        ),
    )

    assert reader.load_existing() is False
    assert reader.snapshot_available is False
    assert reader.snapshot_error == "snapshot_missing_or_invalid"


def test_root_bound_reader_requires_writer_metadata_pair(tmp_path: Path) -> None:
    root = tmp_path / "wiki"
    pages = root / "pages"
    index_dir = root / ".index"
    pages.mkdir(parents=True)
    index_dir.mkdir()
    page = pages / "page.md"
    page.write_text(
        "---\ntitle: Page\nstatus: stable\ntype: knowledge\n---\n"
        "standalonewriter\n",
        encoding="utf-8",
    )

    # Normal maintenance can still build a new projection from canonical
    # pages while the owning metadata writer has not produced its pair yet.
    writer = LexicalIndex(
        path=index_dir / "lexical.sqlite",
        pages=lambda: [page],
        refresh_interval_seconds=0,
    )
    writer.build()
    writer.close()

    reader = LexicalIndex(
        path=index_dir / "lexical.sqlite",
        pages=lambda: (_ for _ in ()).throw(
            AssertionError("root-bound snapshot reader scanned canonical pages")
        ),
    )
    assert reader.load_existing() is False
    assert reader.snapshot_available is False
    assert reader.snapshot_error == "snapshot_missing_or_invalid"


@pytest.mark.parametrize("field", ["path", "relative_path", "mtime_ns", "size", "uid"])
def test_root_bound_reader_rejects_mutated_writer_metadata(
    tmp_path: Path,
    field: str,
) -> None:
    root = tmp_path / "wiki"
    pages = root / "pages"
    index_dir = root / ".index"
    pages.mkdir(parents=True)
    index_dir.mkdir()
    page = pages / "page.md"
    page.write_text(
        "---\ntitle: Page\nstatus: stable\ntype: knowledge\nuid: page-uid\n"
        "---\nmutationcheck\n",
        encoding="utf-8",
    )
    metadata = IndexStore(root)
    metadata._refresh_locked()
    lexical = LexicalIndex(
        path=index_dir / "lexical.sqlite",
        pages=lambda: [page],
        refresh_interval_seconds=0,
    )
    lexical.build()
    lexical.close()

    pages_doc = json.loads((index_dir / "pages.json").read_text())
    entry = pages_doc["entries"]["page"]
    if field == "path":
        entry[field] = str(root / "pages" / "other.md")
    elif field == "relative_path":
        entry[field] = "../escape.md"
    elif field in {"mtime_ns", "size"}:
        entry[field] += 1
    else:
        entry[field] = "mutated-page-uid"
    (index_dir / "pages.json").write_text(json.dumps(pages_doc), encoding="utf-8")

    reader = LexicalIndex(
        path=index_dir / "lexical.sqlite",
        pages=lambda: (_ for _ in ()).throw(
            AssertionError("mutated metadata triggered a canonical page scan")
        ),
    )
    assert reader.load_existing() is False
    assert reader.snapshot_available is False
    assert reader.snapshot_error == "snapshot_missing_or_invalid"


def test_root_bound_reader_rejects_projection_uid_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "wiki"
    pages = root / "pages"
    index_dir = root / ".index"
    pages.mkdir(parents=True)
    index_dir.mkdir()
    page = pages / "page.md"
    page.write_text(
        "---\ntitle: Page\nstatus: stable\ntype: knowledge\nuid: page-uid\n"
        "---\nuidprojection\n",
        encoding="utf-8",
    )
    metadata = IndexStore(root)
    metadata._refresh_locked()
    path = index_dir / "lexical.sqlite"
    lexical = LexicalIndex(path=path, pages=lambda: [page], refresh_interval_seconds=0)
    lexical.build()
    lexical.close()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE pages SET page_uid = ? WHERE page_id = ?",
            ("mutated-page-uid", "page"),
        )

    reader = LexicalIndex(
        path=path,
        pages=lambda: (_ for _ in ()).throw(
            AssertionError("UID mismatch triggered a canonical page scan")
        ),
    )
    assert reader.load_existing() is False
    assert reader.snapshot_available is False
    assert reader.snapshot_error == "snapshot_missing_or_invalid"


@pytest.mark.parametrize("column, value", [("status", "draft")])
def test_root_bound_reader_rejects_projection_identity_or_eligibility_mutation(
    tmp_path: Path,
    column: str,
    value: str,
) -> None:
    root = tmp_path / "wiki"
    pages = root / "pages"
    index_dir = root / ".index"
    pages.mkdir(parents=True)
    index_dir.mkdir()
    page = pages / "page.md"
    page.write_text(
        "---\ntitle: Page\nstatus: stable\ntype: knowledge\nuid: page-uid\n"
        "---\nprojectionmutation\n",
        encoding="utf-8",
    )
    metadata = IndexStore(root)
    metadata._refresh_locked()
    path = index_dir / "lexical.sqlite"
    lexical = LexicalIndex(path=path, pages=lambda: [page], refresh_interval_seconds=0)
    lexical.build()
    lexical.close()
    with sqlite3.connect(path) as connection:
        connection.execute(
            f"UPDATE pages SET {column} = ? WHERE page_id = ?",
            (value, "page"),
        )

    reader = LexicalIndex(path=path, pages=lambda: [])
    assert reader.load_existing() is False
    assert reader.snapshot_available is False
    assert reader.snapshot_error == "snapshot_missing_or_invalid"


@pytest.mark.parametrize(
    ("method", "query_text"),
    [("query_existing", "toctou token"), ("anchor_query_existing", "toctouanchor")],
)
def test_existing_query_rejects_metadata_change_during_sql_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    query_text: str,
) -> None:
    root = tmp_path / "wiki"
    pages = root / "pages"
    index_dir = root / ".index"
    pages.mkdir(parents=True)
    index_dir.mkdir()
    page = pages / "page.md"
    page.write_text(
        "---\ntitle: Page\nstatus: stable\ntype: knowledge\n"
        "uid: page-uid\nentities: [toctouanchor]\n---\n"
        "toctou token\n",
        encoding="utf-8",
    )
    metadata = IndexStore(root)
    metadata._refresh_locked()
    path = index_dir / "lexical.sqlite"
    lexical = LexicalIndex(path=path, pages=lambda: [page], refresh_interval_seconds=0)
    lexical.build()
    lexical.close()

    reader = LexicalIndex(path=path, pages=lambda: [])
    original_connect = sqlite3.connect
    ro_connections = 0
    mutated = False

    def mutate_metadata() -> None:
        nonlocal mutated
        pages_doc = json.loads((index_dir / "pages.json").read_text())
        pages_doc["generation"] += 1
        (index_dir / "pages.json").write_text(json.dumps(pages_doc), encoding="utf-8")
        mutated = True

    def connect_with_metadata_race(*args: object, **kwargs: object) -> sqlite3.Connection:
        nonlocal ro_connections
        if args and isinstance(args[0], str) and "mode=ro" in args[0]:
            ro_connections += 1
            # load_existing opens the first connection; mutate immediately
            # before the query's second connection to reproduce the TOCTOU.
            if ro_connections == 2:
                mutate_metadata()
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect_with_metadata_race)

    assert getattr(reader, method)(query_text) == []
    assert mutated is True


def test_projection_reader_tracks_writer_lifecycle_without_page_scan(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wiki"
    pages = root / "pages"
    index_dir = root / ".index"
    pages.mkdir(parents=True)
    index_dir.mkdir()
    old = pages / "old.md"
    old.write_text(
        "---\ntitle: Old\nstatus: stable\ntype: knowledge\nuid: old-uid\n"
        "---\noldprojectiontoken\n",
        encoding="utf-8",
    )
    metadata = IndexStore(root)
    metadata._refresh_locked()
    lexical = LexicalIndex(
        path=index_dir / "lexical.sqlite",
        pages=lambda: sorted(pages.glob("*.md")),
        refresh_interval_seconds=0,
    )
    lexical.build()
    old_generation = metadata.snapshot_generation
    assert old_generation is not None

    old.unlink()
    replacement = pages / "replacement.md"
    replacement.write_text(
        "---\ntitle: Replacement\nstatus: stable\ntype: knowledge\n"
        "uid: replacement-uid\n---\nreplacementprojectiontoken\n",
        encoding="utf-8",
    )
    metadata._refresh_locked()
    lexical.build(force=True)
    new_generation = metadata.snapshot_generation
    assert new_generation is not None and new_generation != old_generation
    lexical.close()

    reader = LexicalIndex(
        path=index_dir / "lexical.sqlite",
        pages=lambda: (_ for _ in ()).throw(
            AssertionError("lifecycle projection load scanned canonical pages")
        ),
    )
    assert reader.load_existing(expected_generation=new_generation) is True
    assert reader.query_existing(
        "replacementprojectiontoken", expected_generation=old_generation
    ) == []
    assert reader.query_existing("oldprojectiontoken") == []
    rows = reader.query_existing(
        "replacementprojectiontoken", expected_generation=new_generation
    )
    assert [row.page_id for row in rows] == ["replacement"]
    assert rows[0].uid == "replacement-uid"


def test_normal_writer_rebuilds_when_metadata_snapshot_is_invalid(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wiki"
    pages = root / "pages"
    index_dir = root / ".index"
    pages.mkdir(parents=True)
    index_dir.mkdir()
    page = pages / "page.md"
    page.write_text(
        "---\ntitle: Page\nstatus: stable\ntype: knowledge\n---\n"
        "writerrecoverytoken\n",
        encoding="utf-8",
    )
    # A maintenance writer is still allowed to recover from a malformed
    # metadata pair by scanning canonical pages.  Read-only load_existing()
    # remains fail-closed for the same pair.
    (index_dir / "pages.json").write_text("not json", encoding="utf-8")
    calls = 0

    def pages_with_observation() -> list[Path]:
        nonlocal calls
        calls += 1
        return [page]

    index = LexicalIndex(
        path=index_dir / "lexical.sqlite",
        pages=pages_with_observation,
        refresh_interval_seconds=0,
    )
    index.build()

    assert calls == 1
    assert [row.page_id for row in index.query("writerrecoverytoken")] == ["page"]
    assert index.snapshot_available is True
