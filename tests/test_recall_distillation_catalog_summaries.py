from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from chronovisor.core.raw_store import RawStore
from chronovisor.core.store import RuntimeContext, init_chronovisor
from chronovisor.recall import recall_distillation_catalog as catalog
from tests.test_recall_distillation_catalog import _capture, _message


def _bootstrap(root: Path) -> tuple[Path, list[dict[str, object]]]:
    init_chronovisor(RuntimeContext(root))
    raw_dir = _capture(
        root,
        "save-codex-summary.md",
        "a" * 24,
        [
            _message("user", "summary query", "2026-08-01T00:00:00Z"),
            _message("assistant", "summary answer", "2026-08-01T00:00:01Z"),
        ],
    )
    catalog.advance(raw_dir, root, 4096)
    return raw_dir, catalog.rallies(root)


def _refresh_catalog_checkpoint(
    root: Path, checkpoint: dict[str, object] | None = None
) -> None:
    checkpoint = checkpoint or catalog._read_catalog_checkpoint(root)
    assert checkpoint is not None
    lineage = catalog._catalog_lineage(checkpoint)
    assert lineage is not None
    catalog._write_catalog_checkpoint(
        root,
        str(checkpoint["catalog_watermark"]),
        int(checkpoint["event_rowid"]),
        catalog_lineage=lineage,
    )


def test_rally_summaries_and_digest_bound_resolution(tmp_path: Path) -> None:
    _, full_rows = _bootstrap(tmp_path)

    summaries = catalog.rally_summaries(tmp_path)

    assert len(summaries) == len(full_rows) == 1
    summary = summaries[0]
    assert "actual_answer_refs" not in summary
    assert summary["has_actual_answer"] is True
    assert summary["eligibility"] == full_rows[0]["eligibility"]
    assert catalog.resolve_rally_summaries(tmp_path, summaries) == full_rows

    wrong = {**summary, "source_rally_sha256": "0" * 64}
    with pytest.raises(catalog.CatalogError, match="changed during summary resolve"):
        catalog.resolve_rally_summaries(tmp_path, [wrong])


def test_identical_duplicate_rally_rows_are_idempotent(tmp_path: Path) -> None:
    _, full_rows = _bootstrap(tmp_path)
    before = catalog.rally_changes(tmp_path)
    with sqlite3.connect(catalog.catalog_path(tmp_path)) as connection:
        catalog._store_rallies(connection, [full_rows[0], full_rows[0]])
    assert catalog.rally_changes(tmp_path) == before
    assert catalog.rallies(tmp_path) == full_rows


def test_summary_covers_every_changed_rally(tmp_path: Path) -> None:
    init_chronovisor(RuntimeContext(tmp_path))
    raw_dir = _capture(
        tmp_path,
        "save-codex-summary-one.md",
        "a" * 24,
        [
            _message("user", "summary one", "2026-08-01T00:00:00Z"),
            _message("assistant", "answer one", "2026-08-01T00:00:01Z"),
        ],
    )
    _capture(
        tmp_path,
        "save-codex-summary-two.md",
        "b" * 24,
        [
            _message("user", "summary two", "2026-08-02T00:00:00Z"),
            _message("assistant", "answer two", "2026-08-02T00:00:01Z"),
        ],
    )
    catalog.advance(raw_dir, tmp_path, 4096)
    full_rows = catalog.rallies(tmp_path)
    summaries = catalog.rally_summaries(tmp_path)
    assert len(full_rows) == len(summaries) == 2
    assert {row["rally_id"] for row in full_rows} == {
        row["rally_id"] for row in summaries
    }


def test_summary_updates_only_changed_rally_and_noop_is_stable(tmp_path: Path) -> None:
    raw_dir, full_rows = _bootstrap(tmp_path)
    first = catalog.rally_summaries(tmp_path)
    first_changes = catalog.rally_changes(tmp_path)

    _capture(
        tmp_path,
        "save-codex-summary-tail.md",
        "a" * 24,
        [_message("assistant", "summary tail", "2026-08-01T00:00:02Z")],
        after_line=2,
    )
    advanced = catalog.advance(raw_dir, tmp_path, 4096)
    assert advanced.status == "advanced"
    second = catalog.rally_summaries(tmp_path)
    second_changes = catalog.rally_changes(tmp_path)
    assert second[0]["rally_id"] == first[0]["rally_id"]
    assert second[0]["source_rally_sha256"] != first[0]["source_rally_sha256"]
    assert second_changes["revision"] > first_changes["revision"]

    assert catalog.advance(raw_dir, tmp_path, 4096).status == "noop"
    assert catalog.rally_summaries(tmp_path) == second
    assert catalog.rally_changes(tmp_path) == second_changes


def test_existing_valid_catalog_migrates_summary_without_raw_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, full_rows = _bootstrap(tmp_path)
    checkpoint = catalog._read_catalog_checkpoint(tmp_path)
    assert checkpoint is not None
    path = catalog.catalog_path(tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE rally_summaries")
        connection.executemany(
            "DELETE FROM metadata WHERE key=?",
            [
                ("rally_summary_schema",),
                ("rally_summary_complete",),
                ("rally_summary_count",),
                ("rally_summary_revision",),
            ],
        )
    _refresh_catalog_checkpoint(tmp_path, checkpoint)

    def unexpected_read(self: RawStore, raw: object) -> bytes:
        raise AssertionError(f"summary migration read Raw: {raw}")

    monkeypatch.setattr(RawStore, "read_bytes", unexpected_read)
    assert catalog.advance(tmp_path / "raw", tmp_path, 4096).status == "noop"
    assert catalog.resolve_rally_summaries(tmp_path, catalog.rally_summaries(tmp_path)) == full_rows


def test_summary_partial_marker_and_catalog_mutation_fail_closed(tmp_path: Path) -> None:
    _bootstrap(tmp_path)
    checkpoint = catalog._read_catalog_checkpoint(tmp_path)
    assert checkpoint is not None
    path = catalog.catalog_path(tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE metadata SET value='0' WHERE key='rally_summary_complete'"
        )
    _refresh_catalog_checkpoint(tmp_path, checkpoint)
    with pytest.raises(catalog.CatalogError, match="requires migration"):
        catalog.rally_summaries(tmp_path)

    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE metadata SET value='1' WHERE key='rally_summary_complete'"
        )
    _refresh_catalog_checkpoint(tmp_path, checkpoint)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE rally_summaries SET source_rally_sha256=?",
            ("f" * 64,),
        )
    with pytest.raises(catalog.CatalogError, match="checkpoint requires repair"):
        catalog.rally_summaries(tmp_path)


def test_summary_reader_rejects_checkpoint_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bootstrap(tmp_path)
    original = catalog._read_catalog_checkpoint
    calls = 0

    def changing_checkpoint(root: Path) -> dict[str, object] | None:
        nonlocal calls
        calls += 1
        checkpoint = original(root)
        if calls >= 2 and checkpoint is not None:
            checkpoint = dict(checkpoint)
            checkpoint["event_rowid"] = int(checkpoint["event_rowid"]) + 1
        return checkpoint

    monkeypatch.setattr(catalog, "_read_catalog_checkpoint", changing_checkpoint)
    with pytest.raises(catalog.CatalogError, match="changed during rally summary read"):
        catalog.rally_summaries(tmp_path)
    assert calls >= 2


def test_summary_resolve_rejects_checkpoint_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bootstrap(tmp_path)
    summaries = catalog.rally_summaries(tmp_path)
    original = catalog._read_catalog_checkpoint
    calls = 0

    def changing_checkpoint(root: Path) -> dict[str, object] | None:
        nonlocal calls
        calls += 1
        checkpoint = original(root)
        if calls >= 3 and checkpoint is not None:
            checkpoint = dict(checkpoint)
            checkpoint["event_rowid"] = int(checkpoint["event_rowid"]) + 1
        return checkpoint

    monkeypatch.setattr(catalog, "_read_catalog_checkpoint", changing_checkpoint)
    with pytest.raises(catalog.CatalogError, match="changed during rally summary resolve"):
        catalog.resolve_rally_summaries(tmp_path, summaries)
    assert calls >= 3


def test_repair_resets_and_rebuilds_rally_summaries(tmp_path: Path) -> None:
    raw_dir, full_rows = _bootstrap(tmp_path)
    before = catalog._read_catalog_checkpoint(tmp_path)
    assert before is not None
    before_lineage = catalog._catalog_lineage(before)
    assert before_lineage is not None
    path = catalog.catalog_path(tmp_path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE rally_summaries SET source_rally_sha256=?", ("f" * 64,)
        )
        connection.execute(
            "UPDATE metadata SET value='old' WHERE key='rally_changes_schema'"
        )

    result = catalog.advance(raw_dir, tmp_path, 4096)
    assert result.status == "repaired"
    after = catalog._read_catalog_checkpoint(tmp_path)
    assert after is not None
    assert catalog._catalog_lineage(after) != before_lineage
    assert catalog.resolve_rally_summaries(tmp_path, catalog.rally_summaries(tmp_path)) == full_rows
