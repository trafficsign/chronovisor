from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from chronovisor.core.raw_segment import append_capture, capture_date
from chronovisor.core.raw_store import RawStore
from chronovisor.core.save_transaction import (
    attach_save_transaction_marker,
    make_save_transaction,
)
from chronovisor.ingest.raw_semantic_projection import (
    project_native_transcript,
    project_parent_raw,
)
from chronovisor.raw import record_raw as raw_record
from chronovisor.raw.raw_archive import (
    archive_status,
    export_raw,
    migrate_legacy,
    projection_status,
    restore_segment,
    seal_eligible,
    verify_archive,
)


@pytest.fixture(autouse=True)
def _legacy_root(tmp_path: Path) -> None:
    for name in ("index.md", "log.md", "schema.md"):
        (tmp_path / name).write_text("legacy\n", encoding="utf-8")


def _open_segment(raw_dir: Path, source: Path, payload: bytes):
    source.write_bytes(payload)
    return append_capture(
        raw_dir=raw_dir,
        raw_id="save-archive-test.md",
        idempotency_key="archive-test",
        host="codex",
        session_key="c" * 24,
        session_id="session",
        source_file=source,
        after_line=0,
        until_line=1,
        source_bytes=payload,
        record_count=1,
        now=datetime(2026, 7, 17, 23, 59, tzinfo=ZoneInfo("Asia/Tokyo")),
    )


def test_status_verify_seal_export_and_restore(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    payload = b'{"source":"archive"}\n'
    receipt = _open_segment(raw_dir, tmp_path / "session.jsonl", payload)
    (raw_dir / "semantic-a.manifest.json").write_text("{}\n")

    status = archive_status(raw_dir)
    assert status["segment_units"] == 1
    assert status["open_segments"] == 1
    assert status["unsealed_bytes"] == len(payload)
    assert status["projection_artifacts"] == 1
    assert status["physical_files"] >= 4
    assert verify_archive(raw_dir, full=True)["status"] == "ok"

    preview = seal_eligible(raw_dir, before="2026/07/18", dry_run=True)
    assert preview["eligible"] == 1
    assert receipt.data_path.exists()

    applied = seal_eligible(raw_dir, before="2026/07/18", dry_run=False)
    assert applied["status"] == "ok"
    assert not receipt.data_path.exists()
    ledger = tmp_path / "runtime" / "raw-relocation-ledger.jsonl"
    assert '"kind":"segment_seal"' in ledger.read_text()
    verified = verify_archive(raw_dir, full=True)
    assert verified["status"] == "ok"
    assert verified["sealed_segments"] == 1

    exported = tmp_path / "exported.jsonl"
    result = export_raw(raw_dir, "save-archive-test.md", exported)
    assert result["bytes"] == len(payload)
    assert exported.read_bytes() == payload

    manifest = next(
        path
        for path in raw_dir.rglob("*.manifest.json")
        if not path.name.startswith("semantic-")
    )
    restored = tmp_path / "restored-segment.jsonl"
    restore_segment(manifest, restored)
    assert restored.read_bytes() == payload


def test_today_segment_is_not_eligible_by_default_cutoff(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    payload = b'{"source":"today"}\n'
    source = tmp_path / "session.jsonl"
    source.write_bytes(payload)
    append_capture(
        raw_dir=raw_dir,
        raw_id="save-today.md",
        idempotency_key="today",
        host="codex",
        session_key="d" * 24,
        session_id=None,
        source_file=source,
        after_line=0,
        until_line=1,
        source_bytes=payload,
        record_count=1,
    )

    assert seal_eligible(raw_dir, dry_run=True)["eligible"] == 0


def test_v2_manual_raw_is_published_directly_under_capture_date(
    tmp_path: Path, monkeypatch
) -> None:
    raw_dir = tmp_path / "raw"
    monkeypatch.setenv("CHRONOVISOR_RAW_LAYOUT", "v2")
    monkeypatch.setattr(raw_record, "RAW_DIR", raw_dir)

    path = raw_record.publish_raw("manual bytes\n", prefix="api")

    assert path.read_bytes() == b"manual bytes\n"
    assert path.relative_to(raw_dir).parts[:3] == tuple(capture_date().split("/"))
    assert path.name.startswith("manual-")
    assert path.stat().st_mode & 0o777 == 0o600


def test_legacy_raw_staging_and_final_files_are_private(
    tmp_path: Path, monkeypatch
) -> None:
    raw_dir = tmp_path / "raw"
    monkeypatch.setenv("CHRONOVISOR_RAW_LAYOUT", "legacy")
    monkeypatch.setattr(raw_record, "RAW_DIR", raw_dir)

    staging = raw_record.allocate_raw_path()
    published = raw_record.publish_raw("manual bytes\n", prefix="api")

    assert staging.stat().st_mode & 0o777 == 0o600
    assert published.stat().st_mode & 0o777 == 0o600


def test_idempotent_raw_corrects_existing_target_mode(
    tmp_path: Path, monkeypatch
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    target = raw_dir / "save-existing.md"
    target.write_text("same bytes\n", encoding="utf-8")
    target.chmod(0o644)
    monkeypatch.setenv("CHRONOVISOR_RAW_LAYOUT", "legacy")
    monkeypatch.setattr(raw_record, "RAW_DIR", raw_dir)

    published, deduplicated = raw_record.publish_raw_idempotent(
        "same bytes\n",
        idempotency_key="existing",
    )

    assert published == target
    assert deduplicated is True
    assert target.stat().st_mode & 0o777 == 0o600


def test_completed_projection_json_archives_with_processed_bundle(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    session_file = tmp_path / "session.jsonl"
    transaction = make_save_transaction(
        host="codex",
        session_file=session_file,
        session_id="session",
        after_line=0,
        until_line=1,
    )
    parent = raw_dir / f"save-{transaction.idempotency_key}.md"
    content = "\n".join(
        [
            "# Codex Session Transcript Delta",
            "",
            "## Transcript Delta",
            "",
            "```json",
            json.dumps([{"line": 1, "role": "user", "text": "archive bundle"}]),
            "```",
            "",
        ]
    )
    parent.write_text(attach_save_transaction_marker(transaction, content))
    projection = project_parent_raw(
        parent,
        output_dir=raw_dir,
        max_child_bytes=32_000,
    )
    processed = [parent.name, *(path.name for path in projection.child_paths)]
    (tmp_path / ".orchestrator_state.json").write_text(
        json.dumps({"processed_raw_files": processed})
    )
    old = datetime(2026, 7, 16, tzinfo=ZoneInfo("Asia/Tokyo")).timestamp()
    for path in raw_dir.iterdir():
        os.utime(path, (old, old))

    shadow = migrate_legacy(raw_dir, before="2026/07/18", dry_run=False)
    assert shadow["members"] == 4
    migrate_legacy(
        raw_dir,
        before="2026/07/18",
        dry_run=False,
        remove_source=True,
    )

    assert not list(raw_dir.glob("semantic-*.json"))
    store = RawStore(raw_dir)
    assert {unit.raw_id for unit in store.iter_units()} == set(processed)


def test_projection_status_reports_missing_and_invalid_without_runtime_mutation(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    projections: list[tuple[Path, Path]] = []
    for index in range(2):
        session_file = tmp_path / f"session-{index}.jsonl"
        transaction = make_save_transaction(
            host="codex",
            session_file=session_file,
            session_id=f"session-{index}",
            after_line=0,
            until_line=1,
        )
        parent = raw_dir / f"save-{transaction.idempotency_key}.md"
        content = "\n".join(
            [
                "# Codex Session Transcript Delta",
                "",
                "## Transcript Delta",
                "",
                "```json",
                json.dumps(
                    [{"line": 1, "role": "user", "text": f"secret-{index}"}]
                ),
                "```",
                "",
            ]
        )
        parent.write_text(attach_save_transaction_marker(transaction, content))
        projection = project_parent_raw(
            parent,
            output_dir=raw_dir,
            max_child_bytes=32_000,
        )
        assert projection.manifest_path is not None
        assert projection.child_paths
        projections.append((projection.manifest_path, projection.child_paths[0]))

    (tmp_path / ".orchestrator_state.json").write_text(
        json.dumps(
            {
                "processed_raw_files": [
                    path.name
                    for manifest, child in projections
                    for path in (manifest, child)
                ]
                + [
                    path.name
                    for path in raw_dir.glob("save-*.md")
                ]
            }
        ),
        encoding="utf-8",
    )
    projections[0][1].unlink()
    projections[1][1].with_name(
        next(
            path.name
            for path in raw_dir.glob("semantic-*.receipt.json")
            if projections[1][1].stem.split("-child-")[0] in path.name
        )
    ).write_text("{}\n", encoding="utf-8")

    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in (tmp_path, *sorted(tmp_path.rglob("*")))
        if path.is_file()
    }
    report = projection_status(raw_dir, full=True)
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in (tmp_path, *sorted(tmp_path.rglob("*")))
        if path.is_file()
    }

    assert report["total"] == 2
    assert report["valid"] == 0
    assert report["missing"] == 1
    assert report["invalid"] == 1
    assert before == after
    serialized = json.dumps(report, ensure_ascii=False)
    assert "secret-0" not in serialized
    assert "secret-1" not in serialized
    assert all(
        not isinstance(value, str) or not value.startswith("/")
        for value in _walk_values(report)
    )


def _walk_values(value: object):
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child)
    else:
        yield value


def test_projection_status_resolves_v2_segment_parent_without_reference_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import nullcontext

    monkeypatch.setenv("CHRONOVISOR_RAW_LAYOUT", "v2")
    monkeypatch.setattr(
        "chronovisor.core.store.okf_runtime_operation",
        lambda *_args, **_kwargs: nullcontext(),
    )
    raw_dir = tmp_path / "raw"
    payload = (
        json.dumps(
            {
                "timestamp": "2026-07-17T00:00:00Z",
                "message": {"role": "user", "content": "v2 secret"},
            }
        )
        + "\n"
    ).encode()
    source_file = tmp_path / "session.jsonl"
    source_file.write_bytes(payload)
    receipt = append_capture(
        raw_dir=raw_dir,
        raw_id="save-hermes-v2-test.md",
        idempotency_key="hermes-v2-test",
        host="hermes",
        session_key="a" * 24,
        session_id="session",
        source_file=source_file,
        after_line=0,
        until_line=1,
        source_bytes=payload,
        record_count=1,
        now=datetime(2026, 7, 17, tzinfo=ZoneInfo("Asia/Tokyo")),
    )
    unit = next(iter(RawStore(raw_dir, mode="v2").iter_units()))
    artifact_dir = tmp_path / "runtime" / "raw-projections" / "artifacts"
    projection = project_native_transcript(
        Path(unit.raw_id),
        payload,
        receipt.commit,
        output_dir=artifact_dir,
        max_child_bytes=32_000,
    )
    (tmp_path / ".orchestrator_state.json").write_text(
        json.dumps(
            {
                "processed_raw_files": [
                    unit.raw_id,
                    *(path.name for path in projection.child_paths),
                ]
            }
        ),
        encoding="utf-8",
    )
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in (tmp_path, *sorted(tmp_path.rglob("*")))
        if path.is_file()
    }
    report = projection_status(raw_dir, full=True)
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in (tmp_path, *sorted(tmp_path.rglob("*")))
        if path.is_file()
    }

    assert report["total"] == 1
    assert report["valid"] == 1
    item = report["projections"][0]
    assert item["parent"]["raw_id"] == unit.raw_id
    assert item["parent"]["processed"] is True
    assert item["manifest"].startswith("runtime/raw-projections/artifacts/")
    assert before == after
    assert "v2 secret" not in json.dumps(report)
