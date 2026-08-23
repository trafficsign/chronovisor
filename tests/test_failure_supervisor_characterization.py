from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from chronovisor.core import store
from chronovisor.core.raw_segment import RawSegmentCorrupt, append_capture
from chronovisor.core.raw_store import RawStore
from chronovisor.ingest import failure_supervisor
from chronovisor.ingest.raw_semantic_projection import project_native_transcript


def test_corrupt_failure_state_fails_closed_to_empty_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", tmp_path)
    path = tmp_path / "runtime" / "failures" / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text("{broken", encoding="utf-8")

    assert failure_supervisor._load_state() == {"failures": {}}
    assert path.read_text(encoding="utf-8") == "{broken"


def test_save_failure_state_preserves_utf8_and_trailing_newline(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", tmp_path)
    payload = {"failures": {"raw-a.md": {"error": "意味的な失敗"}}}

    failure_supervisor._save_state(payload)

    path = tmp_path / "runtime" / "failures" / "state.json"
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "意味的な失敗" in text
    assert json.loads(text) == payload


def test_group_snapshot_is_sorted_and_does_not_create_lock_file(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", tmp_path)
    failures = tmp_path / "runtime" / "failures"
    failures.mkdir(parents=True)
    packet = tmp_path / "review" / "packet.json"
    failure_supervisor._save_state(
        {
            "failures": {
                "z.md": {"packet_path": str(packet), "attempts": 2},
                "a.md": {"packet_path": str(packet), "attempts": 1},
                "other.md": {"packet_path": str(tmp_path / "other.json")},
            }
        }
    )

    snapshot = failure_supervisor.operational_failure_group_snapshot(packet)

    assert [name for name, _entry in snapshot] == ["a.md", "z.md"]
    assert not (failures / "state.lock").exists()


def test_v2_projection_retry_hashes_authoritative_raw_and_artifact_sibling(
    tmp_path: Path, monkeypatch
) -> None:
    raw_dir = tmp_path / "raw"
    source = tmp_path / "session.jsonl"
    for name in ("index.md", "log.md", "schema.md"):
        (tmp_path / name).write_text("legacy\n", encoding="utf-8")
    payload = (
        b'{"type":"user","sessionId":"session",'
        b'"message":{"content":"retry from authoritative bytes"}}\n'
    )
    source.write_bytes(payload)
    receipt = append_capture(
        raw_dir=raw_dir,
        raw_id="save-retry.md",
        idempotency_key="retry",
        host="codex",
        session_key="a" * 24,
        session_id="session",
        source_file=source,
        after_line=0,
        until_line=1,
        source_bytes=payload,
        record_count=1,
        now=datetime(2026, 8, 23, tzinfo=ZoneInfo("Asia/Tokyo")),
    )
    assert receipt.commit.raw_id == "save-retry.md"
    monkeypatch.setenv("CHRONOVISOR_RAW_LAYOUT", "v2")
    store_instance = RawStore(raw_dir, mode="v2")
    unit = store_instance.resolve("save-retry.md")
    assert unit is not None and unit.commit is not None
    reference = store_instance.materialize_ingest(
        unit, tmp_path / "runtime" / "raw-projections" / "parents"
    )
    project_native_transcript(
        reference,
        store_instance.read_bytes(unit),
        unit.commit,
        output_dir=tmp_path / "runtime" / "raw-projections" / "artifacts",
        max_child_bytes=2_000,
    )
    monkeypatch.setattr(store, "RAW_DIR", raw_dir)

    assert failure_supervisor._projection_parent_can_retry(
        reference,
        {"failure_class": "ingest.runtime_semantic_projection_failure"},
    )


def test_v2_projection_retry_fails_closed_on_corrupt_reference(
    tmp_path: Path, monkeypatch
) -> None:
    reference = tmp_path / "runtime" / "raw-projections" / "parents" / "save.md"
    reference.parent.mkdir(parents=True)
    reference.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        RawStore,
        "resolve_reference",
        lambda _store, _path: (_ for _ in ()).throw(RawSegmentCorrupt("corrupt")),
    )

    assert not failure_supervisor._projection_parent_can_retry(
        reference,
        {"failure_class": "ingest.runtime_semantic_projection_failure"},
    )
