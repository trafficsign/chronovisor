from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
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


def test_query_surfaces_sqlite_errors_instead_of_empty_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = LexicalIndex(path=tmp_path / "lexical.sqlite", pages=lambda: [])

    class BrokenConnection:
        def execute(self, *_args: object, **_kwargs: object) -> object:
            raise sqlite3.OperationalError("synthetic sqlite failure")

    monkeypatch.setattr(index, "_open", lambda: BrokenConnection())

    with pytest.raises(sqlite3.DatabaseError, match="^lexical query unavailable$"):
        index.query("token")


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


def test_anchor_query_existing_surfaces_invalid_projection(tmp_path: Path) -> None:
    path = tmp_path / "lexical.sqlite"
    path.write_text("not sqlite", encoding="utf-8")
    index = LexicalIndex(path=path, pages=lambda: [])

    with pytest.raises(sqlite3.DatabaseError, match="lexical index unavailable"):
        index.anchor_query_existing("anchor")


def test_query_existing_surfaces_invalid_projection(tmp_path: Path) -> None:
    path = tmp_path / "lexical.sqlite"
    path.write_text("not sqlite", encoding="utf-8")
    index = LexicalIndex(path=path, pages=lambda: [])

    with pytest.raises(sqlite3.DatabaseError, match="lexical index unavailable"):
        index.query_existing("query")
    assert path.read_text(encoding="utf-8") == "not sqlite"


def test_query_existing_surfaces_schema_mismatch(tmp_path: Path) -> None:
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

    with pytest.raises(sqlite3.DatabaseError, match="lexical index unavailable"):
        index.query_existing("existingtoken")


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


@pytest.mark.parametrize("writer_method", ["refresh", "apply_changes"])
@pytest.mark.parametrize("change", ["update", "remove", "system"])
def test_metadata_writer_keeps_existing_lexical_projection_readable(
    tmp_path: Path, writer_method: str, change: str,
) -> None:
    root = tmp_path / "wiki"
    page = root / "pages" / "page.md"
    page.parent.mkdir(parents=True)
    for name in ("index.md", "log.md", "schema.md"):
        (root / name).write_text("legacy\n")
    original = "---\ntitle: Page\nstatus: stable\ntype: knowledge\n---\noldtoken\n"
    page.write_text(original)
    metadata = IndexStore(root)
    metadata.refresh()
    path = root / ".index" / "lexical.sqlite"
    lexical = LexicalIndex(path=path, pages=lambda: [page])
    lexical.build()
    lexical.close()
    old_generation = metadata.snapshot_generation

    upserts, removed = [], []
    if change == "remove":
        page.unlink()
        removed = ["page"]
    elif change == "system":
        system = root / "system" / "state.md"
        system.parent.mkdir()
        system.write_text(original.replace("oldtoken", "newtoken"))
        upserts = [system]
    else:
        page.write_text(original.replace("oldtoken", "newtoken"))
        upserts = [page]
    if writer_method == "refresh":
        metadata.refresh()
    else:
        metadata.apply_changes(upserts, removed_page_ids=removed)

    assert metadata.snapshot_generation != old_generation
    reader = LexicalIndex(
        path=path, pages=lambda: pytest.fail("Recall reader scanned pages"),
    )
    assert reader.load_existing(expected_generation=metadata.snapshot_generation)
    expected = [] if change == "remove" else ["state" if change == "system" else "page"]
    assert [row.page_id for row in reader.query_existing("newtoken")] == expected
    if change != "system":
        assert reader.query_existing("oldtoken") == []


def test_lexical_writer_serializes_metadata_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "wiki"
    page = root / "pages" / "page.md"
    page.parent.mkdir(parents=True)
    for name in ("index.md", "log.md", "schema.md"):
        (root / name).write_text("legacy\n", encoding="utf-8")
    page.write_text(
        "---\ntitle: Page\nstatus: stable\ntype: knowledge\n---\noldtoken\n",
        encoding="utf-8",
    )
    metadata_a = IndexStore(root)
    metadata_a.refresh()
    path = root / ".index" / "lexical.sqlite"
    lexical = LexicalIndex(path=path, pages=lambda: [page], refresh_interval_seconds=0)
    lexical.build()
    lexical.close()
    metadata_b = IndexStore(root)
    metadata_b.refresh()
    page.write_text(
        "---\ntitle: Page\nstatus: stable\ntype: knowledge\n---\nnewtoken\n",
        encoding="utf-8",
    )

    original_generation = LexicalIndex._writer_generation
    first_generation_ready = threading.Event()
    release_first = threading.Event()
    second_generation_entered = threading.Event()
    call_lock = threading.Lock()
    calls = 0

    def gated_generation(self: LexicalIndex) -> tuple[bool, int | None]:
        nonlocal calls
        with call_lock:
            calls += 1
            call_number = calls
        result = original_generation(self)
        if call_number == 1:
            first_generation_ready.set()
            if not release_first.wait(timeout=5):
                raise AssertionError("first lexical writer was not released")
        elif call_number == 2:
            second_generation_entered.set()
        return result

    monkeypatch.setattr(LexicalIndex, "_writer_generation", gated_generation)
    errors: list[BaseException] = []

    def refresh(store: IndexStore) -> None:
        try:
            store.refresh()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    first = threading.Thread(target=refresh, args=(metadata_a,))
    second = threading.Thread(target=refresh, args=(metadata_b,))
    first.start()
    try:
        assert first_generation_ready.wait(timeout=5)
        second.start()
        overlapped = second_generation_entered.wait(timeout=1)
    finally:
        release_first.set()
        first.join(timeout=5)
        if second.ident is not None:
            second.join(timeout=5)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert overlapped is False

    reader_metadata = IndexStore(root)
    assert reader_metadata.load_existing() is True
    generation = reader_metadata.snapshot_generation
    assert generation is not None
    reader = LexicalIndex(
        path=path,
        pages=lambda: pytest.fail("consistent projection load scanned pages"),
    )
    try:
        assert reader.load_existing(expected_generation=generation) is True
        assert [row.page_id for row in reader.query_existing("newtoken")] == ["page"]
    finally:
        reader.close()


def test_lexical_writer_reads_generation_after_pages_callback(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wiki"
    page = root / "pages" / "page.md"
    page.parent.mkdir(parents=True)
    for name in ("index.md", "log.md", "schema.md"):
        (root / name).write_text("legacy\n", encoding="utf-8")
    page.write_text(
        "---\ntitle: Page\nstatus: stable\ntype: knowledge\n---\noldtoken\n",
        encoding="utf-8",
    )
    metadata = IndexStore(root)
    metadata.refresh()
    path = root / ".index" / "lexical.sqlite"
    lexical = LexicalIndex(path=path, pages=lambda: [page], refresh_interval_seconds=0)
    lexical.build()
    lexical.close()
    old_generation = metadata.snapshot_generation
    assert old_generation is not None

    page.write_text(
        "---\ntitle: Page\nstatus: stable\ntype: knowledge\n---\nnewtoken\n",
        encoding="utf-8",
    )

    def pages_after_refresh() -> list[Path]:
        metadata.refresh()
        return [page]

    writer = LexicalIndex(
        path=path,
        pages=pages_after_refresh,
        refresh_interval_seconds=0,
    )
    try:
        writer.build(force=True)
    finally:
        writer.close()

    reader_metadata = IndexStore(root)
    assert reader_metadata.load_existing() is True
    new_generation = reader_metadata.snapshot_generation
    assert new_generation is not None and new_generation != old_generation
    reader = LexicalIndex(path=path, pages=lambda: pytest.fail("reader scanned pages"))
    try:
        assert reader.load_existing(expected_generation=new_generation) is True
        assert [row.page_id for row in reader.query_existing("newtoken")] == ["page"]
    finally:
        reader.close()


def test_stale_apply_receipt_merges_newer_metadata_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "wiki"
    pages = root / "pages"
    pages.mkdir(parents=True)
    for name in ("index.md", "log.md", "schema.md"):
        (root / name).write_text("legacy\n", encoding="utf-8")
    for page_id, token in (("first", "firstold"), ("second", "secondold")):
        (pages / f"{page_id}.md").write_text(
            f"---\ntitle: {page_id.title()}\nstatus: stable\ntype: knowledge\n---\n"
            f"{token}\n",
            encoding="utf-8",
        )
    writer_a = IndexStore(root)
    writer_a.refresh()
    path = root / ".index" / "lexical.sqlite"
    lexical = LexicalIndex(
        path=path,
        pages=lambda: sorted(pages.glob("*.md")),
        refresh_interval_seconds=0,
    )
    lexical.build()
    lexical.close()
    writer_b = IndexStore(root)
    writer_b.refresh()

    first = pages / "first.md"
    first.write_text(
        "---\ntitle: First Updated\nstatus: stable\ntype: knowledge\n---\n"
        "firstnew\n",
        encoding="utf-8",
    )
    writer_a.apply_changes([first])

    second = pages / "second.md"
    second.write_text(
        "---\ntitle: Second Updated\nstatus: stable\ntype: knowledge\n---\n"
        "secondnew\n",
        encoding="utf-8",
    )
    writer_b.apply_changes([second])

    reader_metadata = IndexStore(root)
    assert reader_metadata.load_existing() is True
    assert reader_metadata.meta("first")["title"] == "First Updated"
    assert reader_metadata.meta("second")["title"] == "Second Updated"
    generation = reader_metadata.snapshot_generation
    assert generation is not None
    reader = LexicalIndex(path=path, pages=lambda: pytest.fail("receipt reader scanned pages"))
    try:
        assert reader.load_existing(expected_generation=generation) is True
        assert [row.page_id for row in reader.query_existing("firstnew")] == ["first"]
        assert [row.page_id for row in reader.query_existing("secondnew")] == ["second"]
    finally:
        reader.close()


def test_failed_refresh_then_stale_apply_reconciles_other_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "wiki"
    pages = root / "pages"
    pages.mkdir(parents=True)
    for name in ("index.md", "log.md", "schema.md"):
        (root / name).write_text("legacy\n", encoding="utf-8")
    for page_id, token in (("first", "firstold"), ("second", "secondold")):
        (pages / f"{page_id}.md").write_text(
            f"---\ntitle: {page_id.title()}\nstatus: stable\ntype: knowledge\n---\n"
            f"{token}\n",
            encoding="utf-8",
        )
    writer_a = IndexStore(root)
    writer_a.refresh()
    path = root / ".index" / "lexical.sqlite"
    lexical = LexicalIndex(
        path=path,
        pages=lambda: sorted(pages.glob("*.md")),
        refresh_interval_seconds=0,
    )
    lexical.build()
    lexical.close()
    writer_b = IndexStore(root)
    writer_b.refresh()

    first = pages / "first.md"
    first.write_text(
        "---\ntitle: First Updated\nstatus: stable\ntype: knowledge\n---\n"
        "firstnew\n",
        encoding="utf-8",
    )
    writer_a.apply_changes([first])

    second = pages / "second.md"
    second.write_text(
        "---\ntitle: Second Updated\nstatus: stable\ntype: knowledge\n---\n"
        "secondnew\n",
        encoding="utf-8",
    )
    real_persist = writer_b._persist

    def fail_persist(_generation: int) -> None:
        raise OSError("injected metadata publication failure")

    monkeypatch.setattr(writer_b, "_persist", fail_persist)
    writer_b.refresh()
    assert writer_b._persistence_dirty is True
    monkeypatch.setattr(writer_b, "_persist", real_persist)

    first.write_text(
        "---\ntitle: First Latest\nstatus: stable\ntype: knowledge\n---\n"
        "firstlatest\n",
        encoding="utf-8",
    )
    writer_a.apply_changes([first])
    writer_b.apply_changes([second])

    reader_metadata = IndexStore(root)
    assert reader_metadata.load_existing() is True
    assert reader_metadata.meta("first")["title"] == "First Latest"
    assert reader_metadata.meta("second")["title"] == "Second Updated"
    generation = reader_metadata.snapshot_generation
    assert generation is not None
    reader = LexicalIndex(path=path, pages=lambda: pytest.fail("recovery reader scanned pages"))
    try:
        assert reader.load_existing(expected_generation=generation) is True
        assert [row.page_id for row in reader.query_existing("firstlatest")] == ["first"]
        assert [row.page_id for row in reader.query_existing("secondnew")] == ["second"]
    finally:
        reader.close()


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


@pytest.fixture
def quick_check_projection(tmp_path, monkeypatch):
    monkeypatch.delenv("CHRONOVISOR_READ_ONLY", raising=False)
    page = tmp_path / "page.md"
    page.write_text("---\ntitle: Alpha\nstatus: stable\ntype: knowledge\n---\nexistingtoken\n")
    path = tmp_path / "lexical.sqlite"
    writer = LexicalIndex(path=path, pages=lambda: [page])
    writer.build()
    writer.close()
    path.with_suffix(".quick-check.json").unlink(missing_ok=True)
    checks = []
    original = LexicalIndex._run_quick_check

    def check(connection):
        checks.append(True)
        original(connection)

    monkeypatch.setattr(LexicalIndex, "_run_quick_check", staticmethod(check))
    return path, checks


def test_quick_check_success_reused_across_instances(quick_check_projection):
    path, checks = quick_check_projection
    first = LexicalIndex(path=path, pages=lambda: [])
    assert first.load_existing()
    expected = first.query_existing("existingtoken")
    first.close()
    assert len(checks) == 1
    second = LexicalIndex(path=path, pages=lambda: [])
    assert second.load_existing()
    assert second.query_existing("existingtoken") == expected
    second.close()
    assert len(checks) == 1
    assert path.with_suffix(".quick-check.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("corruption", ["truncated", "wrong_digest", "wrong_result"])
def test_quick_check_corrupt_receipt_falls_back(quick_check_projection, corruption):
    path, checks = quick_check_projection
    reader = LexicalIndex(path=path, pages=lambda: [])
    assert reader.load_existing()
    reader.close()
    sidecar = path.with_suffix(".quick-check.json")
    value = json.loads(sidecar.read_text())
    if corruption == "truncated":
        sidecar.write_text("{")
    else:
        if corruption == "wrong_digest":
            value["source"]["db_sha256"] = "0" * 64
        else:
            value["quick_check"] = "failed"
        sidecar.write_text(json.dumps(value))
    assert LexicalIndex(path=path, pages=lambda: []).load_existing()
    assert len(checks) == 2


def test_quick_check_content_hash_invalidates_preserved_stat(quick_check_projection):
    path, checks = quick_check_projection
    reader = LexicalIndex(path=path, pages=lambda: [])
    assert reader.load_existing()
    reader.close()
    before = path.stat()
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE pages SET title = 'Gamma'")
    connection.close()
    assert path.stat().st_size == before.st_size
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert LexicalIndex(path=path, pages=lambda: []).load_existing()
    assert len(checks) == 2


def test_quick_check_wal_changes_invalidate(quick_check_projection):
    path, checks = quick_check_projection
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("UPDATE pages SET title = 'Gamma'")
    writer.commit()
    assert Path(f"{path}-wal").stat().st_size > 0
    assert LexicalIndex(path=path, pages=lambda: []).load_existing()
    writer.execute("UPDATE pages SET title = 'Delta'")
    writer.commit()
    assert LexicalIndex(path=path, pages=lambda: []).load_existing()
    assert len(checks) == 2
    writer.close()


def test_quick_check_hit_keeps_schema_and_metadata_validation(quick_check_projection, monkeypatch):
    path, _checks = quick_check_projection
    reader = LexicalIndex(path=path, pages=lambda: [])
    assert reader.load_existing()
    reader.close()
    # Even an alleged cached success cannot override per-request validation.
    monkeypatch.setattr(LexicalIndex, "_quick_check_cached", lambda *_: True)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE pages SET content_sha256 = 'invalid'")
    connection.close()
    assert not LexicalIndex(path=path, pages=lambda: []).load_existing()
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 999")
    connection.close()
    assert not LexicalIndex(path=path, pages=lambda: []).load_existing()


def test_quick_check_concurrent_update_rechecks_snapshot(quick_check_projection, monkeypatch):
    path, checks = quick_check_projection
    reader = LexicalIndex(path=path, pages=lambda: [])
    assert reader.load_existing()
    reader.close()
    original = LexicalIndex._quick_check_cached

    def racing_cache(self, key):
        cached = original(self, key)
        with sqlite3.connect(path) as writer:
            writer.execute("UPDATE pages SET title = 'Gamma'")
        writer.close()
        return cached

    monkeypatch.setattr(LexicalIndex, "_quick_check_cached", racing_cache)
    assert LexicalIndex(path=path, pages=lambda: []).load_existing()
    assert len(checks) == 2


def test_quick_check_read_only_never_writes_receipt(quick_check_projection, monkeypatch):
    path, checks = quick_check_projection
    sidecar = path.with_suffix(".quick-check.json")
    monkeypatch.setenv("CHRONOVISOR_READ_ONLY", "1")
    assert LexicalIndex(path=path, pages=lambda: []).load_existing()
    assert not sidecar.exists()
    monkeypatch.delenv("CHRONOVISOR_READ_ONLY")
    assert LexicalIndex(path=path, pages=lambda: []).load_existing()
    before = sidecar.read_bytes(), sidecar.stat().st_mtime_ns
    monkeypatch.setenv("CHRONOVISOR_READ_ONLY", "1")
    assert LexicalIndex(path=path, pages=lambda: []).load_existing()
    assert (sidecar.read_bytes(), sidecar.stat().st_mtime_ns) == before
    assert len(checks) == 2


def test_matching_schema_open_does_not_touch_wal(quick_check_projection):
    path, _checks = quick_check_projection
    keeper = sqlite3.connect(path)
    keeper.execute("PRAGMA journal_mode=WAL")
    keeper.execute("UPDATE pages SET title = 'Gamma'")
    keeper.commit()
    wal = Path(f"{path}-wal")
    before = wal.read_bytes(), wal.stat().st_mtime_ns, path.read_bytes()
    index = LexicalIndex(path=path, pages=lambda: [])
    index._open()
    index.close()
    assert (wal.read_bytes(), wal.stat().st_mtime_ns, path.read_bytes()) == before
    keeper.close()


def test_writer_update_and_close_publish_quick_check(tmp_path, monkeypatch):
    monkeypatch.delenv("CHRONOVISOR_READ_ONLY", raising=False)
    page = tmp_path / "page.md"
    page.write_text("---\ntitle: Alpha\nstatus: stable\ntype: knowledge\n---\nfirsttoken\n")
    path = tmp_path / "lexical.sqlite"
    writer = LexicalIndex(path=path, pages=lambda: [page], refresh_interval_seconds=0)
    writer.build()
    check = LexicalIndex._run_quick_check

    def forbidden(_connection):
        pytest.fail("reader repeated writer integrity check")

    for token in ("firsttoken", "secondtoken"):
        if token == "secondtoken":
            monkeypatch.setattr(LexicalIndex, "_run_quick_check", staticmethod(check))
            page.write_text(page.read_text().replace("firsttoken", token))
            writer.build()
        monkeypatch.setattr(LexicalIndex, "_run_quick_check", staticmethod(forbidden))
        reader = LexicalIndex(path=path, pages=lambda: [])
        assert reader.load_existing()
        assert [p.page_id for p in reader.query_existing(token)] == ["page"]
        reader.close()
    monkeypatch.setattr(LexicalIndex, "_run_quick_check", staticmethod(check))
    writer.close()
    monkeypatch.setattr(LexicalIndex, "_run_quick_check", staticmethod(forbidden))
    reader = LexicalIndex(path=path, pages=lambda: [])
    assert reader.load_existing()
    reader.close()


def test_writer_receipt_not_published_for_concurrent_bytes(quick_check_projection, monkeypatch):
    path, _checks = quick_check_projection
    index = LexicalIndex(path=path, pages=lambda: [])
    connection = index._open()
    receipt = path.with_suffix(".quick-check.json")
    assert not receipt.exists()
    original = LexicalIndex._run_quick_check

    def racing_check(reader):
        original(reader)
        with sqlite3.connect(path) as writer:
            writer.execute("UPDATE pages SET title = 'Gamma'")
        writer.close()

    monkeypatch.setattr(LexicalIndex, "_run_quick_check", staticmethod(racing_check))
    index._refresh_quick_check(connection)
    assert not receipt.exists()
    index.close()
