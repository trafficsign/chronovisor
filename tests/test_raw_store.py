from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from llm_wiki_mcp.raw_segment import append_capture, seal_segment
from llm_wiki_mcp.raw_store import RawStore, raw_layout_mode
from llm_wiki_mcp.raw_semantic_projection import project_native_transcript


def _append(raw_dir: Path, source: Path, payload: bytes):
    return append_capture(
        raw_dir=raw_dir,
        raw_id="save-shared.md",
        idempotency_key="shared",
        host="claude-code",
        session_key="b" * 24,
        session_id="session",
        source_file=source,
        after_line=0,
        until_line=1,
        source_bytes=payload,
        record_count=1,
        now=datetime(2026, 7, 18, tzinfo=ZoneInfo("Asia/Tokyo")),
    )


def test_dual_read_precedence_is_reversible(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    legacy = b"legacy markdown\n"
    segment = b'{"source":"native"}\n'
    (raw_dir / "save-shared.md").write_bytes(legacy)
    source = tmp_path / "session.jsonl"
    source.write_bytes(segment)
    _append(raw_dir, source, segment)

    assert RawStore(raw_dir, mode="legacy").read_bytes("save-shared.md") == legacy
    assert RawStore(raw_dir, mode="shadow").read_bytes("save-shared.md") == legacy
    assert RawStore(raw_dir, mode="v2").read_bytes("save-shared.md") == segment


def test_store_reads_open_and_sealed_ranges_by_logical_raw_id(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    source = tmp_path / "session.jsonl"
    payload = b'{"source":"native"}\n'
    source.write_bytes(payload)
    receipt = _append(raw_dir, source, payload)
    store = RawStore(raw_dir, mode="v2")

    unit = store.resolve("save-shared.md")
    assert unit is not None and unit.storage == "segment_open"
    assert store.read_bytes(unit) == payload
    reference = store.materialize_ingest(unit, tmp_path / "runtime" / "parents")
    reference_bytes = reference.read_bytes()

    seal_segment(receipt.data_path, remove_open=True)
    sealed_store = RawStore(raw_dir, mode="v2")
    sealed_unit = sealed_store.resolve("save-shared.md")
    assert sealed_unit is not None and sealed_unit.storage == "segment_sealed"
    assert sealed_store.read_bytes(sealed_unit) == payload
    assert (
        sealed_store.materialize_ingest(
            sealed_unit, tmp_path / "runtime" / "parents"
        ).read_bytes()
        == reference_bytes
    )
    assert sealed_store.resolve_reference(reference) == sealed_unit


def test_logical_reference_survives_raw_root_relocation(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    source = tmp_path / "session.jsonl"
    payload = b'{"source":"relocatable"}\n'
    source.write_bytes(payload)
    receipt = _append(raw_dir, source, payload)
    store = RawStore(raw_dir, mode="v2")
    unit = store.resolve_segment("save-shared.md")
    assert unit is not None
    reference = store.materialize_ingest(unit, tmp_path / "runtime" / "parents")
    seal_segment(receipt.data_path, remove_open=True)

    relocated = tmp_path / "external-volume" / "raw"
    relocated.parent.mkdir()
    raw_dir.rename(relocated)
    relocated_store = RawStore(relocated, mode="v2")

    assert relocated_store.resolve_reference(reference) is not None
    assert relocated_store.read_bytes("save-shared.md") == payload


def test_store_includes_flat_and_date_partitioned_manual_markdown(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    nested = raw_dir / "2026" / "07" / "18"
    nested.mkdir(parents=True)
    (raw_dir / "legacy.md").write_text("old")
    (nested / "manual-new.md").write_text("new")

    store = RawStore(raw_dir)

    assert {unit.raw_id for unit in store} == {"legacy.md", "manual-new.md"}
    assert store.read_text("manual-new.md") == "new"


def test_layout_mode_uses_wiki_config_with_environment_override(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("LLM_WIKI_RAW_LAYOUT")
    (tmp_path / "config.toml").write_text('[raw]\nlayout = "shadow"\n')

    assert raw_layout_mode(wiki_root=tmp_path) == "shadow"
    assert RawStore(tmp_path / "raw").mode == "shadow"

    monkeypatch.setenv("LLM_WIKI_RAW_LAYOUT", "v2")
    assert raw_layout_mode(wiki_root=tmp_path) == "v2"
    assert RawStore(tmp_path / "raw").mode == "v2"


def test_repeated_resolve_builds_one_immutable_store_index(
    tmp_path: Path, monkeypatch
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "first.md").write_text("first")
    (raw_dir / "second.md").write_text("second")
    store = RawStore(raw_dir)
    original = store._legacy_units
    scans = 0

    def counted_units():
        nonlocal scans
        scans += 1
        yield from original()

    monkeypatch.setattr(store, "_legacy_units", counted_units)

    assert store.resolve("first.md") is not None
    assert store.resolve("second.md") is not None
    assert store.resolve("missing.md") is None
    assert scans == 1


def test_materialized_reference_projects_native_transcript_without_copying_it(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    source = tmp_path / "session.jsonl"
    payload = (
        b'{"type":"user","sessionId":"session","message":{"content":"remember me"}}\n'
    )
    source.write_bytes(payload)
    receipt = _append(raw_dir, source, payload)
    store = RawStore(raw_dir, mode="v2")
    unit = store.resolve_segment("save-shared.md")
    assert unit is not None and unit.commit is not None

    reference = store.materialize_ingest(unit, tmp_path / "runtime" / "parents")
    projection = project_native_transcript(
        reference,
        store.read_bytes(unit),
        unit.commit,
        output_dir=tmp_path / "runtime" / "artifacts",
        max_child_bytes=32_000,
    )

    assert len(reference.read_bytes()) < len(payload) + 1024
    assert projection.kind == "children"
    child = json.loads(projection.child_paths[0].read_text())
    assert child["records"][0]["text"] == "remember me"
    assert receipt.commit.sha256 == unit.sha256


def test_v2_parent_and_semantic_child_use_separate_physical_stores(
    tmp_path: Path, monkeypatch
) -> None:
    from llm_wiki_mcp import failure_supervisor, orchestrator

    raw_dir = tmp_path / "raw"
    source = tmp_path / "session.jsonl"
    payload = (
        b'{"type":"user","sessionId":"session","message":{"content":"separate me"}}\n'
    )
    source.write_bytes(payload)
    _append(raw_dir, source, payload)
    monkeypatch.setenv("LLM_WIKI_RAW_LAYOUT", "v2")
    monkeypatch.setattr(orchestrator, "RAW_DIR", raw_dir)
    monkeypatch.setattr(orchestrator, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(
        failure_supervisor, "operational_deferred_raw_files", lambda _paths: {}
    )

    parent_reference = orchestrator.get_pending_raw_files()
    assert len(parent_reference) == 1
    units, quarantined, deferred = orchestrator._prepare_pending_raw_units(
        parent_reference
    )
    assert quarantined == [] and deferred == []
    native = units[0]
    assert native.native_raw_bytes == payload
    assert native.native_commit is not None
    artifact_dir = tmp_path / "runtime" / "raw-projections" / "artifacts"
    projection = project_native_transcript(
        parent_reference[0],
        native.native_raw_bytes,
        native.native_commit,
        output_dir=artifact_dir,
        max_child_bytes=32_000,
    )
    orchestrator.mark_raw_processed([parent_reference[0].name])

    pending = orchestrator.get_pending_raw_files()

    assert pending == list(projection.child_paths)
    assert projection.child_paths[0].parent == artifact_dir
    assert not list(raw_dir.glob("semantic-*.md"))


def test_raw_replay_projects_v2_transport_before_ingest(
    tmp_path: Path, monkeypatch
) -> None:
    from llm_wiki_mcp import raw_replay

    raw_dir = tmp_path / "raw"
    source = tmp_path / "session.jsonl"
    payload = (
        b'{"type":"user","sessionId":"secret-transport-id",'
        b'"message":{"content":"remember this"}}\n'
    )
    source.write_bytes(payload)
    _append(raw_dir, source, payload)
    store = RawStore(raw_dir, mode="v2")
    unit = store.resolve_segment("save-shared.md")
    assert unit is not None
    reference = store.materialize_ingest(unit, tmp_path / "runtime" / "parents")
    monkeypatch.setattr(raw_replay, "RAW_DIR", raw_dir)
    monkeypatch.setattr(raw_replay, "WIKI_ROOT", tmp_path)

    content, summary = raw_replay._replay_ingest_content(reference.name, reference)

    assert content is not None
    bundle = json.loads(content)
    assert bundle["schema"] == "llm-wiki.raw-replay-semantic-bundle.v1"
    assert bundle["children"][0]["records"][0]["text"] == "remember this"
    assert "secret-transport-id" not in content
    assert summary["kind"] == "children"
