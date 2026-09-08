from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from chronovisor.core.recall_context import parse_recall_payload
from chronovisor.core.search_types import SemanticEvidence
from chronovisor.core.semantic_index import (
    SemanticIndexError,
    build_generation,
    load_generation,
)
from chronovisor.core.store import SYSTEM_DIR
from chronovisor.recall.recall_runtime import (
    ContextItem,
    RecallPolicy,
    RecallResult,
    format_recall_context,
)
from chronovisor.research.page_evidence_projection import project_page_evidence
from chronovisor.research.page_section_semantic_adapter import (
    SECTION_SEMANTIC_KIND,
    PageSectionSemanticError,
    build_page_section_documents,
    resolve_page_section_evidence,
)

UID = "019fea6e-8a33-7401-b37b-c7afcf6711e1"
ROUTE = {
    "role": "search.semantic.foreground",
    "provider": "isolated-test",
    "model": "fake-encoder",
    "location": "local",
}


def _source(body: str) -> bytes:
    return (
        f"---\nuid: {UID}\ntitle: Section adapter\nstatus: stable\n"
        "sensitivity: normal\nrecall_questions:\n  - Which section is current?\n---\n"
        f"{body}"
    ).encode()


def _encoder(documents, _batch_size):
    return np.asarray(
        [
            [1.0, 0.0]
            if "current railway" in document.text
            else [0.0, 1.0]
            for document in documents
        ],
        dtype=np.float32,
    )


def test_all_projection_records_build_section_documents_and_resolve_exact_bytes(
    tmp_path: Path,
) -> None:
    source = _source(
        "# Current\nThe current railway uses a blue token.\n\n"
        "# History\nThe old railway used a red token.\n"
    )
    path = tmp_path / "section.md"
    path.write_bytes(source)
    projection = project_page_evidence(source, "section")

    documents = build_page_section_documents(
        source,
        "section",
        source_path=path,
        source_mtime_ns=path.stat().st_mtime_ns,
        projection=projection,
    )

    assert len(documents) == len(projection.records)
    assert len(documents) >= 2
    assert all(document.kind == SECTION_SEMANTIC_KIND for document in documents)
    assert [document.ordinal for document in documents] == list(range(len(documents)))
    for document, record in zip(documents, projection.records, strict=True):
        original = source[record.byte_start : record.byte_end].decode()
        assert document.doc_id == record.record_id
        assert "[section_context]" in document.text
        assert "[search_keys]" in document.text
        assert f"[source]\n{original}" in document.text
        assert document.source_sha256 == hashlib.sha256(source).hexdigest()
        assert document.page_uid == UID
    evidence = tuple(
        SemanticEvidence(
            page_id="section",
            doc_id=document.doc_id,
            kind=SECTION_SEMANTIC_KIND,
            ordinal=document.ordinal,
            source_sha256=document.source_sha256,
            page_uid=document.page_uid,
            generation_id="isolated-generation",
            score=0.9 - document.ordinal / 100,
        )
        for document in documents
    )
    passages = resolve_page_section_evidence(
        source, "section", evidence, projection=projection
    )
    assert len(passages) == len(documents)
    for passage, record in zip(passages, projection.records, strict=True):
        assert source[passage.byte_start : passage.byte_end].decode() == passage.text
        assert passage.byte_start == record.byte_start
        assert passage.byte_end == record.byte_end


def test_source_classification_is_derived_from_canonical_metadata(tmp_path: Path) -> None:
    source = (
        f"---\nuid: {UID}\ntitle: System section\nstatus: stable\n"
        "is_system: true\nsensitivity: normal\n---\nSystem source.\n"
    ).encode()
    path = tmp_path / "system.md"
    path.write_bytes(source)

    document = build_page_section_documents(
        source,
        "system",
        source_path=path,
        source_mtime_ns=path.stat().st_mtime_ns,
    )[0]

    assert document.source_data_class == "system"
    assert document.source_sensitivity == "high"

    path_classified_source = _source("# Current\nPath classified source.\n")
    path_document = build_page_section_documents(
        path_classified_source,
        "section",
        source_path=SYSTEM_DIR / "path-classified.md",
        source_mtime_ns=1,
    )[0]
    assert path_document.source_data_class == "system"
    assert path_document.source_sensitivity == "high"


def test_isolated_generation_search_and_recall_publication_use_exact_source(
    tmp_path: Path,
) -> None:
    source = _source("# Current\nThe current railway uses a blue token.\n")
    path = tmp_path / "section.md"
    path.write_bytes(source)
    projection = project_page_evidence(source, "section")
    documents = build_page_section_documents(
        source,
        "section",
        source_path=path,
        source_mtime_ns=path.stat().st_mtime_ns,
        projection=projection,
    )
    manifest = build_generation(
        documents,
        encode_documents=_encoder,
        **ROUTE,
        revision="isolated-c",
        dimensions=2,
        query_prefix="query: ",
        document_prefix="passage: ",
        batch_size=2,
        root=tmp_path / "semantic",
        repo_commit="isolated-test",
        extractor_schema_version=3,
    )
    loaded = load_generation(manifest.generation_id, root=tmp_path / "semantic")
    rows = loaded.search_with_evidence([1.0, 0.0], top_n=1)
    assert len(rows) == 1
    page_id, score, evidence = rows[0]
    assert page_id == "section"
    assert score > 0
    assert evidence and evidence[0].kind == SECTION_SEMANTIC_KIND
    assert evidence[0].doc_id == documents[0].doc_id
    assert evidence[0].ordinal == documents[0].ordinal
    assert manifest.extractor_schema_version == 3

    passages = resolve_page_section_evidence(source, page_id, evidence, projection=projection)
    # The existing compiler accepts the shared passage type; production C
    # routing remains gated separately from this isolated index adapter.
    item = ContextItem(
        page_id=page_id, title="Section adapter", updated="", score=score,
        sensitivity=documents[0].source_sensitivity, uid=UID,
        source_passages=passages, evidence_kind="semantic_section",
    )
    assert item is not None
    assert item.evidence_kind == "semantic_section"
    assert item.source_passages
    result = RecallResult(
        status="ok",
        decision="read",
        confidence=0.9,
        queries=["current railway"],
        reasons=[],
        matched_terms={},
        context_items=[item],
    )
    rendered = format_recall_context(result, RecallPolicy(max_context_chars=3_000))
    payload = parse_recall_payload(rendered)
    assert payload is not None
    assert payload["items"]
    cited = payload["items"][0]
    reference = cited["source_ref"]
    assert reference["doc_id"] == evidence[0].doc_id
    assert reference["uid"] == UID
    assert reference["sha256"] == hashlib.sha256(source).hexdigest()
    assert (
        source[reference["byte_start"] : reference["byte_end"]].decode()
        == cited["evidence"]
    )
    assert "current railway" in cited["evidence"]


def test_section_identity_and_source_staleness_fail_closed(tmp_path: Path) -> None:
    source = _source("# Current\nThe current railway uses a blue token.\n")
    projection = project_page_evidence(source, "section")
    records = projection.records
    documents = build_page_section_documents(
        source,
        "section",
        source_path=tmp_path / "section.md",
        source_mtime_ns=1,
        projection=projection,
    )
    valid = SemanticEvidence(
        "section",
        documents[0].doc_id,
        SECTION_SEMANTIC_KIND,
        0,
        documents[0].source_sha256,
        UID,
        "isolated-generation",
        0.9,
    )
    assert resolve_page_section_evidence(source, "section", (valid,))
    stale = source.replace(b"blue", b"green")
    assert resolve_page_section_evidence(stale, "section", (valid,)) == ()
    assert resolve_page_section_evidence(
        source,
        "section",
        (
            SemanticEvidence(
                **{
                    **asdict(valid),
                    "kind": "chunk",
                }
            ),
        ),
    ) == ()
    assert resolve_page_section_evidence(
        source,
        "section",
        (
            SemanticEvidence(
                **{
                    **asdict(valid),
                    "doc_id": records[0].record_id,
                    "ordinal": 99,
                }
            ),
        ),
    ) == ()

    try:
        build_page_section_documents(
            stale,
            "section",
            source_path=tmp_path / "section.md",
            source_mtime_ns=1,
            projection=projection,
        )
    except PageSectionSemanticError:
        pass
    else:
        raise AssertionError("stale projection must be rejected")


def test_invalid_section_refs_and_generation_schema_fail_closed(tmp_path: Path) -> None:
    source = _source(
        "# Current\nThe current railway uses a blue token.\n\n"
        "# History\nThe old railway used a red token.\n"
    )
    path = tmp_path / "section.md"
    path.write_bytes(source)
    documents = build_page_section_documents(
        source,
        "section",
        source_path=path,
        source_mtime_ns=1,
    )
    base = SemanticEvidence(
        "section",
        documents[0].doc_id,
        SECTION_SEMANTIC_KIND,
        0,
        documents[0].source_sha256,
        UID,
        "generation-a",
        0.9,
    )
    for changes in (
        {"score": "0.9"},
        {"score": 10**1000},
        {"doc_id": []},
        {"kind": "unknown"},
    ):
        assert resolve_page_section_evidence(
            source,
            "section",
            (SemanticEvidence(**{**asdict(base), **changes}),),
        ) == ()
    second = SemanticEvidence(
        "section",
        documents[1].doc_id,
        SECTION_SEMANTIC_KIND,
        1,
        documents[1].source_sha256,
        UID,
        "generation-b",
        0.8,
    )
    assert resolve_page_section_evidence(source, "section", (base, second)) == ()
    assert resolve_page_section_evidence(source, "section", (base,) * 4) == ()

    with pytest.raises(SemanticIndexError):
        build_generation(
            documents,
            encode_documents=_encoder,
            **ROUTE,
            revision="invalid-bool",
            dimensions=2,
            query_prefix="query: ",
            document_prefix="passage: ",
            batch_size=2,
            root=tmp_path / "invalid-bool",
            extractor_schema_version=True,
        )
    with pytest.raises(SemanticIndexError):
        build_generation(
            documents,
            encode_documents=_encoder,
            **ROUTE,
            revision="invalid-zero",
            dimensions=2,
            query_prefix="query: ",
            document_prefix="passage: ",
            batch_size=2,
            root=tmp_path / "invalid-zero",
            extractor_schema_version=0,
        )
    default_manifest = build_generation(
        documents,
        encode_documents=_encoder,
        **ROUTE,
        revision="legacy-b",
        dimensions=2,
        query_prefix="query: ",
        document_prefix="passage: ",
        batch_size=2,
        root=tmp_path / "legacy-b",
    )
    assert default_manifest.extractor_schema_version == 2


def test_section_projection_keeps_more_than_eight_records(tmp_path: Path) -> None:
    source = _source("# " + "Current" * 400 + "\n" + ("current railway detail. " * 400) + "\n")
    path = tmp_path / "long.md"
    path.write_bytes(source)
    projection = project_page_evidence(source, "long")
    documents = build_page_section_documents(
        source,
        "long",
        source_path=path,
        source_mtime_ns=1,
        projection=projection,
    )
    assert len(projection.records) > 8
    assert len(documents) == len(projection.records)
    assert documents[-1].ordinal == len(projection.records) - 1
    assert all(len(document.text) <= 2000 for document in documents)
