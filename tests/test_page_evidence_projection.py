from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chronovisor.core.canonical_document import parse_document
from chronovisor.core.canonical_json import canonical_json_line_bytes_strict
from chronovisor.core.page_identity import new_page_uid
from chronovisor.research.page_evidence_projection import (
    PAGE_EVIDENCE_PROJECTION_SCHEMA,
    PageEvidenceProjectionError,
    build_page_evidence_artifact,
    checkpoint_page_evidence_projection,
    load_page_evidence_artifact,
    page_evidence_projection_key,
    project_page_evidence,
    reconstruct_page_body,
    store_page_evidence_projection,
)
from chronovisor.search.research_store import ResearchStore


def _source(body: str, *, uid: str, status: str = "stable") -> bytes:
    header = (
        "---\n"
        f"status: {status}\n"
        "title: Evidence page\n"
        f"uid: {uid}\n"
        "updated: 2026-09-08\n"
        "description: Deterministic page context\n"
        "entities: [Chronovisor, Recall]\n"
        "recall_questions:\n"
        "  - unique recall question marker\n"
        "page_type: knowledge\n"
        "---\n"
    )
    return (header + body).encode("utf-8")


def test_projection_covers_body_sections_and_long_tail_without_copying_text() -> None:
    uid = new_page_uid(timestamp_ms=1_725_000_000_001, random_bits=101)
    long_tail = " tail-only-marker " + ("本文 " * 400)
    source = _source(
        "Preamble は見出しの前です。\r\n\r\n"
        "# 親\r\n"
        "親の本文です。\r\n\r\n"
        "## 子\r\n"
        + long_tail
        + "\r\n",
        uid=uid,
    )

    projection = project_page_evidence(source, "evidence-page")
    body = parse_document(source).body

    assert projection.schema == PAGE_EVIDENCE_PROJECTION_SCHEMA
    assert projection.page_uid == uid
    assert projection.source_updated == "2026-09-08"
    assert projection.recorded_at is None
    assert projection.valid_from is None
    assert projection.valid_to is None
    assert projection.source_body_byte_range[1] - projection.source_body_byte_range[0] == len(body)
    assert projection.records
    assert reconstruct_page_body(source, projection) == body
    assert long_tail.encode("utf-8") not in projection.canonical_bytes()
    assert any(record.section_context == ("親",) for record in projection.records)
    assert any(record.section_context == ("親", "子") for record in projection.records)
    assert any("親" in record.search_keys or "子" in record.search_keys for record in projection.records)
    assert any("unique" in record.search_keys for record in projection.records)
    assert any(
        b"tail-only-marker" in source[record.byte_start : record.byte_end]
        for record in projection.records
    )
    previous_end = projection.source_body_byte_range[0]
    for record in projection.records:
        assert record.byte_start == previous_end
        assert record.byte_end > record.byte_start
        assert hashlib.sha256(source[record.byte_start : record.byte_end]).hexdigest() == record.span_sha256
        previous_end = record.byte_end
    assert previous_end == projection.source_body_byte_range[1]


def test_projection_loader_rejects_tamper_stale_source_and_unknown_validity() -> None:
    uid = new_page_uid(timestamp_ms=1_725_000_000_002, random_bits=102)
    source = _source("# A\n本文\n# B\n末尾\n", uid=uid)
    payload = build_page_evidence_artifact(source, "evidence-page")
    loaded = load_page_evidence_artifact(payload, source=source, page_id="evidence-page")
    assert loaded.canonical_bytes() == payload

    tampered = json.loads(payload)
    tampered["projection_id"] = "page-projection:" + "0" * 64
    with pytest.raises(PageEvidenceProjectionError, match="identity mismatch"):
        load_page_evidence_artifact(
            (json.dumps(tampered, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(),
            source=source,
        )

    stale_source = source.replace("末尾".encode(), "別の末尾".encode())
    with pytest.raises(PageEvidenceProjectionError, match="stale"):
        load_page_evidence_artifact(payload, source=stale_source)

    unknown_time = json.loads(payload)
    unknown_time["recorded_at"] = "2026-09-08T00:00:00+09:00"
    unsigned = dict(unknown_time)
    unsigned.pop("projection_id")
    from chronovisor.core.canonical_json import canonical_json_line_bytes_strict

    digest = hashlib.sha256(canonical_json_line_bytes_strict(unsigned)).hexdigest()
    unknown_time["projection_id"] = f"page-projection:{digest}"
    with pytest.raises(PageEvidenceProjectionError, match="values are invalid"):
        load_page_evidence_artifact(
            (json.dumps(unknown_time, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )

    noncanonical = payload.replace(b"\n", b" \n", 1)
    with pytest.raises(PageEvidenceProjectionError):
        load_page_evidence_artifact(noncanonical)


def test_loader_rejects_rehashed_context_metadata_against_source_transform() -> None:
    uid = new_page_uid(timestamp_ms=1_725_000_000_004, random_bits=104)
    source = _source("# A\n本文\n", uid=uid)
    tampered = json.loads(build_page_evidence_artifact(source, "evidence-page"))
    record = tampered["records"][0]
    record["search_keys"] = ["tampered-key"]
    unsigned_record = dict(record)
    unsigned_record.pop("record_id")
    record["record_id"] = "page-record:" + hashlib.sha256(
        canonical_json_line_bytes_strict(unsigned_record)
    ).hexdigest()
    unsigned_projection = dict(tampered)
    unsigned_projection.pop("projection_id")
    tampered["projection_id"] = "page-projection:" + hashlib.sha256(
        canonical_json_line_bytes_strict(unsigned_projection)
    ).hexdigest()
    payload = canonical_json_line_bytes_strict(tampered)
    with pytest.raises(PageEvidenceProjectionError, match="transform mismatch"):
        load_page_evidence_artifact(payload, source=source, page_id="evidence-page")


def test_projection_heading_stack_skips_levels_and_ignores_fenced_headings() -> None:
    uid = new_page_uid(timestamp_ms=1_725_000_000_005, random_bits=105)
    source = _source(
        "# Top\n"
        "top\n"
        "### Deep\n"
        "deep\n"
        "## Mid\n"
        "mid\n"
        "```markdown\n"
        "# fake\n"
        "## fake nested\n"
        "```\n"
        "## Real\n"
        "real\n",
        uid=uid,
    )

    projection = project_page_evidence(source, "evidence-page")
    assert [record.section_context for record in projection.records] == [
        ("Top",),
        ("Top", "Deep"),
        ("Top", "Mid"),
        ("Top", "Real"),
    ]
    assert reconstruct_page_body(source, projection) == parse_document(source).body


def test_projection_uses_only_string_entities_as_search_keys() -> None:
    source = _source("# A\nbody\n", uid=new_page_uid()).replace(
        b"entities: [Chronovisor, Recall]",
        b"entities: [Chronovisor, {ignored: nested}, 123, null]",
    )
    keys = project_page_evidence(source, "page").records[0].search_keys
    assert "chronovisor" in keys
    assert not {"ignored", "nested", "123", "none"}.intersection(keys)


def test_projection_rejects_unstable_or_invalid_uid_and_reuses_cas_checkpoint(
    tmp_path: Path,
) -> None:
    uid = new_page_uid(timestamp_ms=1_725_000_000_003, random_bits=103)
    with pytest.raises(PageEvidenceProjectionError, match="not stable"):
        project_page_evidence(_source("draft\n", uid=uid, status="draft"), "evidence-page")
    with pytest.raises(PageEvidenceProjectionError, match="UID"):
        project_page_evidence(_source("bad uid\n", uid="not-a-uuid"), "evidence-page")

    source = _source("# A\nbody\n", uid=uid)
    store = ResearchStore(tmp_path / "research")
    first = store_page_evidence_projection(store, source, "evidence-page")
    second = store_page_evidence_projection(store, source, "evidence-page")
    assert first.artifact_id == second.artifact_id
    assert store.read_artifact(first.artifact_id) == build_page_evidence_artifact(
        source, "evidence-page"
    )
    assert first.metadata["projection_key"] == page_evidence_projection_key(
        source, "evidence-page"
    )
    assert first.metadata["schema"] == PAGE_EVIDENCE_PROJECTION_SCHEMA

    checkpoint = checkpoint_page_evidence_projection(store, "backfill-1", first)
    checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert checkpoint_payload["payload"]["artifact_id"] == first.artifact_id
    assert checkpoint_payload["payload"]["projection_key"] == first.metadata["projection_key"]
    session_hash = hashlib.sha256(b"backfill-1").hexdigest()
    assert checkpoint == store.run_dir(session_hash) / "summary.json"
    assert checkpoint.is_relative_to(store.root)
    assert not checkpoint.is_relative_to(store.checkpoints)


def test_store_rechecks_existing_cas_before_reporting_success(tmp_path: Path) -> None:
    import zstandard

    source = _source("# A\nbody\n", uid=new_page_uid())
    store = ResearchStore(tmp_path / "research")
    artifact = store_page_evidence_projection(store, source, "page")
    digest = artifact.sha256
    blob = store.cas / digest[:2] / f"{digest}.zst"
    blob.write_bytes(zstandard.ZstdCompressor().compress(b"wrong existing payload"))
    with pytest.raises(ValueError, match="checksum"):
        store_page_evidence_projection(store, source, "page")
