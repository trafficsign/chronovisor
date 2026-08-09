from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from chronovisor.core.raw_segment import append_capture, capture_date
from chronovisor.core.raw_store import RawStore
from chronovisor.core.save_transaction import (
    attach_save_transaction_marker,
    make_save_transaction,
)
from chronovisor.ingest.raw_semantic_projection import project_parent_raw
from chronovisor.raw import record_raw as raw_record
from chronovisor.raw.raw_archive import (
    archive_status,
    export_raw,
    migrate_legacy,
    restore_segment,
    seal_eligible,
    verify_archive,
)


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
