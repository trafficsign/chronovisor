from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from chronovisor.core.raw_segment import (
    RawSegmentCorrupt,
    append_capture,
    copy_source_interval,
    find_commit,
    read_sealed_range,
    restored_segment_bytes,
    seal_segment,
    verify_manifest,
)

NOW = datetime(2026, 7, 18, 9, 30, tzinfo=ZoneInfo("Asia/Tokyo"))
SESSION_KEY = "a" * 24


def _source(path: Path) -> bytes:
    rows = [
        {"type": "session_meta", "payload": {"id": "session-1"}},
        {"type": "message", "role": "user", "content": "hello"},
        {"type": "message", "role": "assistant", "content": "world"},
    ]
    raw = b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    path.write_bytes(raw)
    return raw


def _append(raw_dir: Path, source: Path, raw: bytes, *, key: str = "tx-1"):
    return append_capture(
        raw_dir=raw_dir,
        raw_id=f"save-{key}.md",
        idempotency_key=key,
        host="codex",
        session_key=SESSION_KEY,
        session_id="session-1",
        source_file=source,
        after_line=0,
        until_line=3,
        source_bytes=raw,
        record_count=3,
        now=NOW,
        max_part_bytes=1024 * 1024,
    )


def test_append_uses_date_only_layout_and_is_idempotent(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    source = tmp_path / "session.jsonl"
    raw = _source(source)

    first = _append(raw_dir, source, raw)
    second = _append(raw_dir, source, raw)

    assert first.data_path.relative_to(raw_dir).parts[:3] == ("2026", "07", "18")
    assert first.data_path.name.endswith(".jsonl.open")
    assert first.commit_path.name.endswith(".commits.jsonl")
    assert first.commit.offset == 0
    assert first.commit.length == len(raw)
    assert not first.deduplicated
    assert second.deduplicated
    assert second.commit == first.commit
    assert first.data_path.read_bytes() == raw


def test_append_repairs_data_without_a_durable_commit(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    source = tmp_path / "session.jsonl"
    raw = _source(source)
    first = _append(raw_dir, source, raw)
    first.data_path.write_bytes(raw + b"uncommitted-tail\n")

    second_raw = b'{"next":true}\n'
    second = append_capture(
        raw_dir=raw_dir,
        raw_id="save-tx-2.md",
        idempotency_key="tx-2",
        host="codex",
        session_key=SESSION_KEY,
        session_id="session-1",
        source_file=source,
        after_line=3,
        until_line=4,
        source_bytes=second_raw,
        record_count=1,
        now=NOW,
    )

    assert second.commit.offset == len(raw)
    assert second.data_path.read_bytes() == raw + second_raw


def test_seal_verifies_full_restore_and_preserves_logical_receipts(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    source = tmp_path / "session.jsonl"
    raw = _source(source)
    receipt = _append(raw_dir, source, raw)

    manifest = seal_segment(receipt.data_path, remove_open=True)
    manifest_path = receipt.data_path.with_name(
        receipt.data_path.name.removesuffix(".jsonl.open") + ".manifest.json"
    )
    sealed_path = manifest_path.with_name(str(manifest["segment"]))

    assert not receipt.data_path.exists()
    assert not receipt.commit_path.exists()
    assert restored_segment_bytes(sealed_path) == raw
    assert read_sealed_range(sealed_path, 0, len(raw)) == raw
    assert verify_manifest(manifest_path, full=True) == manifest
    located = find_commit(raw_dir, "save-tx-1.md")
    assert located is not None
    assert located.sealed
    assert located.commit.sha256 == receipt.commit.sha256


def test_sealed_receipt_rejects_changed_idempotent_payload(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    source = tmp_path / "session.jsonl"
    raw = _source(source)
    receipt = _append(raw_dir, source, raw)
    seal_segment(receipt.data_path, remove_open=True)

    with pytest.raises(RawSegmentCorrupt, match="collision"):
        _append(raw_dir, source, raw.replace(b"hello", b"HELLO"))


def test_copy_source_interval_preserves_original_jsonl_bytes(tmp_path: Path) -> None:
    source = tmp_path / "session.jsonl"
    source.write_bytes('{ "spacing": true }\n{"unicode":"日本語"}\npartial'.encode())

    assert copy_source_interval(source, after_line=0, until_line=2) == (
        '{ "spacing": true }\n{"unicode":"日本語"}\n'.encode()
    )


def test_part_rollover_never_splits_a_committed_capture(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    source = tmp_path / "session.jsonl"
    first = b'{"first":true}\n'
    second = b'{"second":true}\n'
    source.write_bytes(first + second)
    one = append_capture(
        raw_dir=raw_dir,
        raw_id="save-roll-1.md",
        idempotency_key="roll-1",
        host="codex",
        session_key=SESSION_KEY,
        session_id=None,
        source_file=source,
        after_line=0,
        until_line=1,
        source_bytes=first,
        record_count=1,
        now=NOW,
        max_part_bytes=len(first) + 1,
    )
    two = append_capture(
        raw_dir=raw_dir,
        raw_id="save-roll-2.md",
        idempotency_key="roll-2",
        host="codex",
        session_key=SESSION_KEY,
        session_id=None,
        source_file=source,
        after_line=1,
        until_line=2,
        source_bytes=second,
        record_count=1,
        now=NOW,
        max_part_bytes=len(first) + 1,
    )

    assert one.commit.part == 1
    assert two.commit.part == 2
    assert one.data_path.read_bytes() == first
    assert two.data_path.read_bytes() == second


def test_parallel_appends_share_one_lock_and_keep_contiguous_ranges(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    source = tmp_path / "session.jsonl"
    rows = [b'{"row":1}\n', b'{"row":2}\n']
    source.write_bytes(b"".join(rows))

    def publish(index: int):
        return append_capture(
            raw_dir=raw_dir,
            raw_id=f"save-parallel-{index}.md",
            idempotency_key=f"parallel-{index}",
            host="codex",
            session_key=SESSION_KEY,
            session_id=None,
            source_file=source,
            after_line=index - 1,
            until_line=index,
            source_bytes=rows[index - 1],
            record_count=1,
            now=NOW,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(publish, (1, 2)))

    data_path = receipts[0].data_path
    assert receipts[1].data_path == data_path
    commits = sorted(receipts, key=lambda receipt: receipt.commit.offset)
    assert commits[0].commit.offset == 0
    assert commits[1].commit.offset == commits[0].commit.length
    assert data_path.read_bytes() in {rows[0] + rows[1], rows[1] + rows[0]}


def test_same_transaction_cannot_duplicate_across_capture_day_boundary(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    source = tmp_path / "session.jsonl"
    payload = b'{"same":"transaction"}\n'
    source.write_bytes(payload)
    moments = (
        datetime(2026, 7, 18, 23, 59, tzinfo=ZoneInfo("Asia/Tokyo")),
        datetime(2026, 7, 19, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo")),
    )

    def publish(moment: datetime):
        return append_capture(
            raw_dir=raw_dir,
            raw_id="save-cross-day.md",
            idempotency_key="cross-day",
            host="codex",
            session_key=SESSION_KEY,
            session_id=None,
            source_file=source,
            after_line=0,
            until_line=1,
            source_bytes=payload,
            record_count=1,
            now=moment,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(publish, moments))

    assert receipts[0].data_path == receipts[1].data_path
    assert sum(1 for _path in raw_dir.rglob("*.commits.jsonl")) == 1


def test_append_never_reopens_part_with_published_seal_manifest(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    source = tmp_path / "session.jsonl"
    first = b'{"first":true}\n'
    second = b'{"second":true}\n'
    source.write_bytes(first + second)
    one = append_capture(
        raw_dir=raw_dir,
        raw_id="save-seal-race-1.md",
        idempotency_key="seal-race-1",
        host="codex",
        session_key=SESSION_KEY,
        session_id=None,
        source_file=source,
        after_line=0,
        until_line=1,
        source_bytes=first,
        record_count=1,
        now=NOW,
    )
    seal_segment(one.data_path, remove_open=False)

    two = append_capture(
        raw_dir=raw_dir,
        raw_id="save-seal-race-2.md",
        idempotency_key="seal-race-2",
        host="codex",
        session_key=SESSION_KEY,
        session_id=None,
        source_file=source,
        after_line=1,
        until_line=2,
        source_bytes=second,
        record_count=1,
        now=NOW,
    )

    assert one.commit.part == 1
    assert two.commit.part == 2
    assert two.data_path.read_bytes() == second
