from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from llm_wiki_mcp.raw_archive import (
    archive_status,
    export_raw,
    restore_segment,
    seal_eligible,
    verify_archive,
)
from llm_wiki_mcp.raw_segment import append_capture, capture_date
from llm_wiki_mcp import server


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
    monkeypatch.setenv("LLM_WIKI_RAW_LAYOUT", "v2")
    monkeypatch.setattr(server, "RAW_DIR", raw_dir)

    path = server._publish_raw("manual bytes\n", prefix="api")

    assert path.read_bytes() == b"manual bytes\n"
    assert path.relative_to(raw_dir).parts[:3] == tuple(capture_date().split("/"))
    assert path.name.startswith("manual-")
