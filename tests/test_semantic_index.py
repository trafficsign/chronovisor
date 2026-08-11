from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import chronovisor.core.semantic_index as semantic_index
from chronovisor.core.semantic_index import (
    SemanticDocument,
    SemanticIndexError,
    activate_generation,
    archive_legacy_search_index,
    build_generation,
    extract_page_documents,
    load_active_generation,
    prune_expired_legacy_archives,
    read_active,
    rollback_generation,
    semantic_index_status,
    validate_generation,
    write_page_delta,
)

ROUTE = {
    "role": "search.semantic.foreground",
    "provider": "test",
    "model": "test-model",
    "location": "local",
}


def test_extract_page_documents_uses_full_canonical_yaml_and_body(
    tmp_path: Path,
) -> None:
    source = b"""---
status: stable
type: knowledge
title: Nested YAML title
description: Canonical summary
recall_questions:
  - Where is the answer?
metadata:
  nested:
    enabled: true
---
# Body heading

Body text only.
"""
    path = tmp_path / "canonical-page.md"
    path.write_bytes(source)

    documents = extract_page_documents(path)

    page = documents[0]
    assert page.kind == "page"
    assert page.source_sha256 == hashlib.sha256(source).hexdigest()
    assert page.source_data_class == "page"
    assert page.source_sensitivity == "high"
    assert "Nested YAML title" in page.text
    assert "Canonical summary" in page.text
    assert "Q: Where is the answer?" in page.text
    assert "Body text only." in page.text
    assert "metadata:" not in page.text
    assert [document.text for document in documents if document.kind == "question"] == [
        "Where is the answer?"
    ]
    chunks = [document.text for document in documents if document.kind == "chunk"]
    assert chunks and "Summary: Canonical summary" in chunks[0]
    assert "Body text only." in "\n".join(chunks)


@pytest.mark.parametrize(
    ("metadata", "data_class", "sensitivity"),
    [
        ("sensitivity: normal", "page", "normal"),
        ("sensitivity: high", "page", "high"),
        ("is_system: true\nsensitivity: normal", "system", "high"),
    ],
)
def test_extract_page_documents_classifies_egress_conservatively(
    tmp_path: Path,
    metadata: str,
    data_class: str,
    sensitivity: str,
) -> None:
    path = tmp_path / "page.md"
    path.write_text(
        f"---\nstatus: stable\ntype: knowledge\n{metadata}\n---\nBody\n",
        encoding="utf-8",
    )

    document = extract_page_documents(path)[0]

    assert document.source_data_class == data_class
    assert document.source_sensitivity == sensitivity


@pytest.mark.parametrize("status", ["draft", "deprecated"])
def test_extract_page_documents_excludes_non_stable_pages(
    tmp_path: Path, status: str
) -> None:
    path = tmp_path / f"{status}.md"
    path.write_text(
        f"---\nstatus: {status}\ntype: knowledge\ntitle: Hidden\n---\nBody\n",
        encoding="utf-8",
    )

    assert extract_page_documents(path) == []


def _documents(version: str = "a") -> list[SemanticDocument]:
    return [
        SemanticDocument(
            doc_id="alpha",
            page_id="alpha",
            kind="page",
            ordinal=-1,
            text=f"alpha text {version}",
            source_path="/pages/alpha.md",
            source_sha256=version * 64,
            source_mtime_ns=1,
        ),
        SemanticDocument(
            doc_id="alpha#q0",
            page_id="alpha",
            kind="question",
            ordinal=0,
            text="where is alpha",
            source_path="/pages/alpha.md",
            source_sha256=version * 64,
            source_mtime_ns=1,
        ),
        SemanticDocument(
            doc_id="beta",
            page_id="beta",
            kind="page",
            ordinal=-1,
            text="beta text",
            source_path="/pages/beta.md",
            source_sha256="b" * 64,
            source_mtime_ns=2,
        ),
    ]


def _encoder(documents: list[SemanticDocument], _batch_size: int) -> np.ndarray:
    rows = []
    for document in documents:
        if "alpha" in document.text:
            rows.append([1.0, 0.0, 0.0])
        else:
            rows.append([0.0, 1.0, 0.0])
    return np.asarray(rows, dtype=np.float32)


def _build(root: Path, version: str = "a"):
    return build_generation(
        _documents(version),
        encode_documents=_encoder,
        **ROUTE,
        revision="test-revision",
        dimensions=3,
        query_prefix="query: ",
        document_prefix="passage: ",
        batch_size=2,
        root=root,
        repo_commit="deadbeef",
    )


def test_build_validate_activate_and_search_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("chronovisor.core.search.searchable_pages", lambda: [])
    manifest = _build(tmp_path)

    assert validate_generation(manifest.generation_id, root=tmp_path) == manifest
    pointer = activate_generation(manifest.generation_id, root=tmp_path)
    loaded = load_active_generation(root=tmp_path)

    assert pointer["generation_id"] == manifest.generation_id
    assert {key: pointer[key] for key in ROUTE} == ROUTE
    assert loaded.search([1.0, 0.0, 0.0], top_n=2)[0][0] == "alpha"
    assert loaded.score_pages([1.0, 0.0, 0.0], ["alpha"]) == [
        ("alpha", pytest.approx(1.0))
    ]
    generation = tmp_path / "generations" / manifest.generation_id
    assert tmp_path.stat().st_mode & 0o777 == 0o700
    assert generation.stat().st_mode & 0o777 == 0o700
    assert (generation / "vectors.npy").stat().st_mode & 0o777 == 0o600
    assert (generation / "metadata.sqlite").stat().st_mode & 0o777 == 0o600
    status = semantic_index_status(root=tmp_path)
    assert status["status"] == "ok"
    assert status["page_count"] == 2
    assert status["document_count"] == 3
    assert {key: status[key] for key in ROUTE} == ROUTE


@pytest.mark.parametrize("field", ["role", "provider", "model", "location"])
def test_active_generation_rejects_route_identity_drift(
    tmp_path: Path, field: str
) -> None:
    manifest = _build(tmp_path)
    activate_generation(manifest.generation_id, root=tmp_path)
    expected = dict(ROUTE)
    expected[field] = "changed"

    with pytest.raises(SemanticIndexError, match="route identity mismatch"):
        load_active_generation(root=tmp_path, expected_route=expected)


def test_active_pointer_rejects_route_identity_drift(tmp_path: Path) -> None:
    manifest = _build(tmp_path)
    activate_generation(manifest.generation_id, root=tmp_path)
    active_path = tmp_path / "active.json"
    active = json.loads(active_path.read_text(encoding="utf-8"))
    active["role"] = "changed"
    active_path.write_text(json.dumps(active), encoding="utf-8")

    with pytest.raises(SemanticIndexError, match="route pointer mismatch"):
        load_active_generation(root=tmp_path)


def test_schema_three_active_generation_fails_closed_until_rebuild(
    tmp_path: Path,
) -> None:
    manifest = _build(tmp_path)
    activate_generation(manifest.generation_id, root=tmp_path)
    manifest_path = (
        tmp_path / "generations" / manifest.generation_id / "manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 3
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    active_path = tmp_path / "active.json"
    active = json.loads(active_path.read_text(encoding="utf-8"))
    active["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    active_path.write_text(json.dumps(active), encoding="utf-8")

    assert semantic_index_status(root=tmp_path)["error"] == "generation_invalid"
    with pytest.raises(SemanticIndexError, match="unsupported generation schema"):
        load_active_generation(root=tmp_path)


def test_hnsw_generation_proposes_then_full_vector_rescores(tmp_path: Path) -> None:
    pytest.importorskip("usearch")
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(96, 32)).astype(np.float32)
    documents = [
        SemanticDocument(
            doc_id=f"page-{index}",
            page_id=f"page-{index}",
            kind="page",
            ordinal=-1,
            text=f"page {index}",
            source_path=f"/pages/page-{index}.md",
            source_sha256=f"{index:064x}",
            source_mtime_ns=index,
        )
        for index in range(len(vectors))
    ]

    manifest = build_generation(
        documents,
        encode_documents=lambda _documents, _batch_size: vectors,
        role="search.semantic.foreground",
        provider="test",
        model="test-model",
        location="local",
        revision="test-revision",
        dimensions=32,
        query_prefix="query: ",
        document_prefix="passage: ",
        batch_size=16,
        root=tmp_path,
        repo_commit="deadbeef",
    )
    activate_generation(manifest.generation_id, root=tmp_path)
    loaded = load_active_generation(root=tmp_path)

    assert manifest.ann_kind == "usearch_hnsw_f16"
    assert manifest.ann_dimensions == 32
    assert loaded.search(vectors[37], top_n=5)[0][0] == "page-37"
    assert loaded.score_pages(vectors[37], ["page-37"])[0] == (
        "page-37",
        pytest.approx(1.0),
    )


def test_active_pointer_uses_compare_and_swap_and_rolls_back(tmp_path: Path) -> None:
    first = _build(tmp_path, "a")
    activate_generation(first.generation_id, root=tmp_path)
    second = _build(tmp_path, "c")

    with pytest.raises(SemanticIndexError, match="CAS failed"):
        activate_generation(
            second.generation_id,
            expected_current="not-current",
            root=tmp_path,
        )

    activate_generation(
        second.generation_id,
        expected_current=first.generation_id,
        root=tmp_path,
    )
    rollback_generation(root=tmp_path)

    assert read_active(root=tmp_path)["generation_id"] == first.generation_id


def test_default_semantic_root_uses_declared_activation_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _build(tmp_path)
    declared_lock = tmp_path / "declared-activation.lock"
    monkeypatch.setattr(semantic_index, "SEMANTIC_ROOT", tmp_path)
    monkeypatch.setattr(semantic_index, "ACTIVATION_LOCK", declared_lock)

    activate_generation(manifest.generation_id, root=tmp_path)

    assert declared_lock.exists()
    assert not (tmp_path / "activation.lock").exists()


def test_delta_shadows_all_base_documents_for_updated_page(tmp_path: Path) -> None:
    manifest = _build(tmp_path)
    activate_generation(manifest.generation_id, root=tmp_path)
    updated = [
        SemanticDocument(
            doc_id="alpha",
            page_id="alpha",
            kind="page",
            ordinal=-1,
            text="alpha now means z",
            source_path="/pages/alpha.md",
            source_sha256="c" * 64,
            source_mtime_ns=3,
        )
    ]
    write_page_delta(
        manifest.generation_id,
        "alpha",
        updated,
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        dimensions=3,
        root=tmp_path,
    )

    loaded = load_active_generation(root=tmp_path)

    assert loaded.search([1.0, 0.0, 0.0], top_n=2)[0][0] == "beta"
    assert loaded.search([0.0, 0.0, 1.0], top_n=2)[0][0] == "alpha"


def test_corrupt_generation_is_rejected_before_activation(tmp_path: Path) -> None:
    manifest = _build(tmp_path)
    vectors = tmp_path / "generations" / manifest.generation_id / "vectors.npy"
    vectors.write_bytes(vectors.read_bytes() + b"corrupt")

    with pytest.raises(SemanticIndexError, match="checksum"):
        activate_generation(manifest.generation_id, root=tmp_path)


def test_incomplete_generation_is_rejected(tmp_path: Path) -> None:
    generation = tmp_path / "generations" / "broken"
    generation.mkdir(parents=True)
    (generation / "manifest.json").write_text(json.dumps({}), encoding="utf-8")

    with pytest.raises(SemanticIndexError, match="incomplete"):
        validate_generation("broken", root=tmp_path)


def test_legacy_index_is_archived_only_after_fresh_cutover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "embeddings.sqlite"
    source.write_bytes(b"legacy-search-index" * 256)
    archive_dir = tmp_path / "archive"
    monkeypatch.setattr(
        semantic_index,
        "semantic_index_status",
        lambda **_kwargs: {
            "status": "ok",
            "coverage": 1.0,
            "missing_page_count": 0,
            "stale_page_count": 0,
            "generation_id": "generation-a",
            "model": "nemotron",
            "revision": "revision-a",
        },
    )

    result = archive_legacy_search_index(
        source=source,
        archive_dir=archive_dir,
        retain_days=14,
        root=tmp_path / "semantic",
    )

    archive = Path(result["archive"])
    assert result["status"] == "archived"
    assert not source.exists()
    assert archive.exists()
    assert archive.stat().st_mode & 0o777 == 0o600
    assert archive.with_suffix(archive.suffix + ".manifest.json").exists()


def test_legacy_index_cleanup_refuses_incomplete_cutover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "embeddings.sqlite"
    source.write_bytes(b"legacy")
    monkeypatch.setattr(
        semantic_index,
        "semantic_index_status",
        lambda **_kwargs: {
            "status": "ok",
            "coverage": 0.9,
            "missing_page_count": 1,
            "stale_page_count": 0,
        },
    )

    with pytest.raises(SemanticIndexError, match="complete fresh generation"):
        archive_legacy_search_index(
            source=source,
            archive_dir=tmp_path / "archive",
            root=tmp_path / "semantic",
        )

    assert source.exists()


def test_expired_legacy_archive_is_pruned(tmp_path: Path) -> None:
    archive = tmp_path / "embeddings-bge-old.sqlite.zst"
    archive.write_bytes(b"archive")
    manifest = archive.with_suffix(archive.suffix + ".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "archive": str(archive),
                "expires_at": "2000-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    removed = prune_expired_legacy_archives(archive_dir=tmp_path)

    assert removed == [str(archive)]
    assert not archive.exists()
    assert not manifest.exists()


def test_legacy_archive_prune_rejects_manifest_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.sqlite.zst"
    outside.write_bytes(b"keep")
    manifest = tmp_path / "embeddings-bge-hostile.sqlite.zst.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "archive": str(outside),
                "expires_at": "2000-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    assert prune_expired_legacy_archives(archive_dir=tmp_path) == []
    assert outside.exists()
    outside.unlink()
