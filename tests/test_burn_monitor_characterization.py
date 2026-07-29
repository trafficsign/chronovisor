from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from chronovisor.ops import burn_monitor


def test_evidence_writer_uses_exclusive_create_sorted_utf8_and_fsync(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "evidence.jsonl"
    fsynced: list[int] = []
    monkeypatch.setattr(burn_monitor.os, "fsync", fsynced.append)
    writer = burn_monitor.EvidenceWriter(path)
    writer.write({"z": 1, "message": "継続計測", "a": 2})
    writer.close()

    assert path.read_text(encoding="utf-8") == (
        '{"a": 2, "message": "継続計測", "z": 1}\n'
    )
    assert len(fsynced) == 1
    with pytest.raises(FileExistsError):
        burn_monitor.EvidenceWriter(path)


def test_file_delta_tracker_counts_append_and_detects_truncation(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"baseline\n")
    tracker = burn_monitor.FileDeltaTracker.start("events", path)
    with path.open("ab") as handle:
        handle.write(b"one\ntwo\n")

    appended = tracker.poll()

    assert appended["appended_bytes"] == len(b"one\ntwo\n")
    assert appended["appended_lines"] == 2
    assert appended["appended_sha256"] == hashlib.sha256(b"one\ntwo\n").hexdigest()

    path.write_bytes(b"replacement\n")
    replaced = tracker.poll()
    assert replaced["resets"] == 1
    assert replaced["appended_lines"] == 3


def test_keyed_jsonl_delta_rejects_duplicate_and_non_advancing_ids() -> None:
    duplicate = b'{"ts":"2"}\n{"ts":"2"}\n'
    _lines, _ids, valid, error = burn_monitor.keyed_jsonl_delta(
        duplicate, set(), key="ts"
    )
    assert valid is False
    assert error == "duplicate ts"

    payload = b'{"ts":"1"}\n{"ts":"3"}\n'
    _lines, _ids, valid, error = burn_monitor.keyed_jsonl_delta(
        payload, {"2"}, key="ts"
    )
    assert valid is False
    assert error == "new ts did not advance"


def test_runtime_identity_requires_exact_commit_and_no_drift() -> None:
    expected = "a" * 40
    projection = {
        "runtime": {
            "commit_id": expected,
            "expected_commit": expected,
            "drift": False,
        }
    }

    assert burn_monitor.runtime_is_expected(projection, expected) is True
    projection["runtime"]["drift"] = True
    assert burn_monitor.runtime_is_expected(projection, expected) is False
