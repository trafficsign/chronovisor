from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, replace

from chronovisor.core.page_evidence import project_page_evidence
from chronovisor.core.page_identity import new_page_uid
from chronovisor.core.search_types import SemanticEvidence, parse_semantic_evidence
from chronovisor.core.semantic_evidence import (
    SourcePassage,
    resolve_semantic_evidence,
)


def _canonical_source(body: str, *, uid: str, status: str = "stable") -> bytes:
    header = (
        "---\n"
        f"status: {status}\n"
        "title: Span page\n"
        f"uid: {uid}\n"
        "description: Never use this summary as a passage.\n"
        "---\n"
    )
    return (header + body).encode("utf-8")


def _evidence(source: bytes, uid: str, ordinal: int, *, page_id: str = "span") -> SemanticEvidence:
    return SemanticEvidence(
        page_id=page_id,
        doc_id=f"{uid}#c{ordinal}",
        kind="chunk",
        ordinal=ordinal,
        source_sha256=hashlib.sha256(source).hexdigest(),
        page_uid=uid,
        generation_id="generation-1",
        score=1.0 - ordinal / 10,
    )


def test_resolve_returns_exact_multibyte_crlf_body_spans() -> None:
    uid = new_page_uid(timestamp_ms=1_725_000_000_123, random_bits=12345)
    source = _canonical_source(
        "  first passage  \r\n"
        "\r\n"
        "# セクション\r\n"
        "同じ  文\r\n"
        "同じ 文\r\n"
        "\r\n"
        "終わりです。\r\n",
        uid=uid,
    )
    passages = resolve_semantic_evidence(
        source,
        "span",
        tuple(_evidence(source, uid, ordinal) for ordinal in range(3)),
    )

    assert all(isinstance(passage, SourcePassage) for passage in passages)
    assert len(passages) == 3
    assert [passage.evidence.ordinal for passage in passages] == [0, 1, 2]
    for passage in passages:
        assert source[passage.byte_start : passage.byte_end].decode("utf-8") == passage.text
        assert passage.byte_start < passage.byte_end
        assert "Page: Span page" not in passage.text
        assert "Never use this summary" not in passage.text
    assert passages[0].text == "first passage"
    assert passages[1].text == "同じ  文\r\n同じ 文"
    assert passages[2].text == "終わりです。"
    assert "# セクション" not in passages[1].text


def test_resolve_tracks_repeated_long_paragraphs_sequentially() -> None:
    uid = new_page_uid(timestamp_ms=1_725_000_000_124, random_bits=12346)
    paragraph = ("repeat " * 220).strip()
    source = _canonical_source(f"{paragraph}\n\n{paragraph}\n", uid=uid)
    passages = resolve_semantic_evidence(
        source,
        "span",
        tuple(_evidence(source, uid, ordinal) for ordinal in range(4)),
    )

    assert len(passages) == 4
    assert all(
        source[p.byte_start : p.byte_end].decode("utf-8") == p.text
        for p in passages
    )
    assert [p.byte_start for p in passages] == sorted(p.byte_start for p in passages)
    assert passages[0].byte_start < passages[2].byte_start
    assert all("repeat" in p.text for p in passages)


def test_resolve_fail_closed_for_stale_or_non_chunk_references() -> None:
    uid = new_page_uid(timestamp_ms=1_725_000_000_125, random_bits=12347)
    source = _canonical_source("actual body\n", uid=uid)
    valid = _evidence(source, uid, 0)
    stale_hash = replace(valid, source_sha256="0" * 64)
    stale_uid = replace(
        valid,
        page_uid=new_page_uid(timestamp_ms=1_725_000_000_126, random_bits=12348),
        doc_id=f"{uid}#c0",
    )
    page_match = replace(valid, kind="page", doc_id=uid, ordinal=-1)
    question_match = replace(valid, kind="question", doc_id=f"{uid}#q0", ordinal=0)
    bad_generation = replace(valid, generation_id=" ")
    bad_score = replace(valid, score=math.nan)
    bad_ordinal = replace(valid, ordinal=8, doc_id=f"{uid}#c8")

    passages = resolve_semantic_evidence(
        source,
        "span",
        (
            stale_hash,
            stale_uid,
            page_match,
            question_match,
            bad_generation,
            bad_score,
            bad_ordinal,
            valid,
        ),
    )

    assert len(passages) == 1
    assert passages[0].evidence == valid
    assert resolve_semantic_evidence(
        _canonical_source("actual body\n", uid=uid, status="draft"),
        "span",
        (valid,),
    ) == ()


def test_invalid_runtime_values_and_mixed_generations_do_not_escape_resolver() -> None:
    uid = new_page_uid()
    source = _canonical_source("first\n\nsecond\n", uid=uid)
    valid = _evidence(source, uid, 0)
    for invalid in (
        replace(valid, doc_id=[]),
        replace(valid, score=10**1000),
        replace(valid, score="0.9"),
    ):
        assert resolve_semantic_evidence(source, "span", (invalid,)) == ()
    other = replace(_evidence(source, uid, 1), generation_id="other-generation")
    assert resolve_semantic_evidence(source, "span", (valid, other)) == ()


def test_section_evidence_dispatches_and_rejects_chunk_mixing() -> None:
    uid = new_page_uid(timestamp_ms=1_725_000_000_127, random_bits=12349)
    source = _canonical_source(
        "# Current\n現在の接続先は port 7443。\n\n"
        "# History\nThe old endpoint was port 7442.\n",
        uid=uid,
    )
    projection = project_page_evidence(source, "span")
    section = tuple(
        SemanticEvidence(
            page_id="span",
            doc_id=record.record_id,
            kind="section-v1",
            ordinal=index,
            source_sha256=projection.content_sha256,
            page_uid=uid,
            generation_id="generation-section",
            score=0.9 - index / 10,
        )
        for index, record in enumerate(projection.records[:2])
    )
    passages = resolve_semantic_evidence(source, "span", section)
    assert [passage.evidence.doc_id for passage in passages] == [
        record.record_id for record in projection.records[:2]
    ]
    assert all(
        source[passage.byte_start : passage.byte_end].decode() == passage.text
        for passage in passages
    )
    page_match = replace(section[0], kind="page", doc_id=uid, ordinal=-1)
    assert resolve_semantic_evidence(source, "span", (page_match, section[0])) == passages[:1]
    assert resolve_semantic_evidence(
        source,
        "span",
        (section[0], _evidence(source, uid, 0)),
    ) == ()


def test_section_wire_identity_uses_record_id_and_unbounded_ordinal() -> None:
    uid = new_page_uid(timestamp_ms=1_725_000_000_128, random_bits=12350)
    source = _canonical_source("section\n", uid=uid)
    projection = project_page_evidence(source, "span")
    record = projection.records[0]
    item = SemanticEvidence(
        page_id="span",
        doc_id=record.record_id,
        kind="section-v1",
        ordinal=12,
        source_sha256=projection.content_sha256,
        page_uid=uid,
        generation_id="generation-section",
        score=0.8,
    )
    parsed = parse_semantic_evidence(
        [asdict(item)], page_id="span", generation_id="generation-section"
    )
    assert parsed == (item,)
    bad_record = replace(item, doc_id="not-a-page-record")
    try:
        parse_semantic_evidence([asdict(bad_record)], page_id="span")
    except ValueError as exc:
        assert str(exc) == "invalid semantic evidence"
    else:
        raise AssertionError("section doc IDs must be page-record digests")
