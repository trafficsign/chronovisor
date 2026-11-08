from __future__ import annotations

import pytest

from chronovisor.core.raw_segment import RawSegmentCorrupt
from chronovisor.core.save_transaction import find_published_save_transaction


def test_corrupt_journal_stops_recovery_without_leaking_or_replacing_bytes(tmp_path):
    raw = tmp_path / "raw"
    day = raw / "2026" / "11" / "08"
    day.mkdir(parents=True)
    journal = day / "private-transcript.commits.jsonl"
    journal.write_bytes(b'{"private transcript"\n')
    before = journal.read_bytes(), journal.stat().st_mtime_ns
    with pytest.raises(RawSegmentCorrupt, match="^raw journal validation failed; capture stopped$"):
        find_published_save_transaction(
            raw_dir=raw,
            host="codex",
            session_file=tmp_path / "session.jsonl",
            session_id="test",
            after_line=0,
        )
    assert (journal.read_bytes(), journal.stat().st_mtime_ns) == before
