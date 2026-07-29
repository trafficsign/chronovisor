from __future__ import annotations

import json
from pathlib import Path

from chronovisor.classification.classification import default_udc_package
from chronovisor.classification.classification_library_sources import (
    czech_authority_contract,
    czech_bibliography_contract,
    ndlsh_contract,
    write_external_package,
)
from chronovisor.lab import classification_library_evidence
from chronovisor.lab.classification_fixture_set import inference_dto
from chronovisor.lab.classification_library_evidence import (
    LibraryEvidenceIndex,
    LibraryEvidenceProvider,
    build_dense_index,
    build_source_index,
    external_test_cases,
)


def _package(
    tmp_path: Path,
    *,
    name: str,
    rows: list[dict],
) -> Path:
    package_root = tmp_path / "sources" / name
    if name == "bib":
        contract = czech_bibliography_contract("https://example.invalid/bib")
    elif name == "authority":
        contract = czech_authority_contract("https://example.invalid/auth")
    else:
        contract = ndlsh_contract("https://example.invalid/ndlsh")
    write_external_package(
        package_root,
        contract=contract,
        source_release="fixture-v1",
        rows=rows,
        acquisition={"kind": "test-fixture"},
    )
    return package_root / "manifest.json"


def _external_row(
    record_id: str,
    *,
    role: str,
    source_field: str,
    notation: str,
    title: str,
    rights_ref: str,
) -> dict:
    row = {
        "source_record_id": record_id,
        "record_sha256": f"sha256:{record_id:0>64}",
        "rights_ref": rights_ref,
        "title": title,
        "subject_headings": [
            {
                "pref_label": title,
                "alt_labels": ["artificial intelligence"],
            }
        ],
        "source_assignments": [
            {
                "role": role,
                "source_field": source_field,
                "notation_or_uri": notation,
                "generation_method": "not_machine_generated",
                "intellectual_assignment": "confirmed",
            }
        ],
    }
    return row


def test_index_is_split_first_and_field_boundaries_are_enforced(
    tmp_path: Path,
) -> None:
    rows = [
        _external_row(
            f"bib-{index}",
            role="bibliographic_assignment",
            source_field="080",
            notation="004.8",
            title="AI memory systems",
            rights_ref="czech-national-bibliography",
        )
        for index in range(20)
    ]
    rows.extend(
        [
            _external_row(
                "bad-composite",
                role="bibliographic_assignment",
                source_field="080",
                notation="004.8:025",
                title="bad",
                rights_ref="czech-national-bibliography",
            ),
            _external_row(
                "bad-field",
                role="authority_representative_classification",
                source_field="089$c",
                notation="004.8",
                title="bad",
                rights_ref="czech-topical-authorities",
            ),
        ]
    )
    manifest_path = _package(tmp_path, name="bib", rows=rows)
    index_path = tmp_path / "index" / "evidence.sqlite3"

    manifest = build_source_index(
        package_manifest_paths=[manifest_path],
        output_path=index_path,
        root=tmp_path,
    )

    assert manifest["split_policy"].endswith("train-only-index")
    assert manifest["rejected_counts_by_reason"]["unresolvable_or_composite_udc"] == 1
    assert manifest["rejected_counts_by_reason"]["authority_non_089a"] == 1
    assert manifest["working_set_gate"] is True


def test_provider_never_evicts_a0_and_ndlsh_never_votes(
    tmp_path: Path,
) -> None:
    authority_rows = [
        _external_row(
            f"auth-{index}",
            role="authority_representative_classification",
            source_field="089$a",
            notation="004.8",
            title="AI memory systems",
            rights_ref="czech-topical-authorities",
        )
        for index in range(20)
    ]
    ndlsh_rows = [
        {
            "source_record_id": f"ndl-{index}",
            "record_sha256": f"sha256:{index:064x}",
            "rights_ref": "ndlsh-authority",
            "title": "人工知能",
            "subject_headings": [
                {
                    "pref_label": "人工知能",
                    "alt_labels": ["AI", "機械知能"],
                }
            ],
            "source_assignments": [],
            "diagnostic_relations": [
                {"predicate": "notations", "target": "NDC10:007.13"}
            ],
        }
        for index in range(20)
    ]
    manifests = [
        _package(tmp_path, name="authority", rows=authority_rows),
        _package(tmp_path, name="ndlsh", rows=ndlsh_rows),
    ]
    index_path = tmp_path / "index" / "evidence.sqlite3"
    build_source_index(
        package_manifest_paths=manifests,
        output_path=index_path,
        root=tmp_path,
    )
    provider = LibraryEvidenceProvider(
        package=default_udc_package(),
        evidence_index=LibraryEvidenceIndex(index_path.with_suffix(".manifest.json")),
    )
    page = inference_dto(
        {
            "uid": "page-1",
            "source_sha256": "sha256:page",
            "title": "人工知能 AI memory systems",
            "summary": "agent memory",
            "tags": ["d/ai"],
            "raw_keywords": [],
            "excerpt": "",
        }
    )

    result = provider.candidates(
        page,
        arms=["B2", "B3", "C1"],
        limit=20,
    )

    assert result["baseline_candidates_evicted"] is False
    assert result["union"][:12] == result["official_baseline"]
    assert any(row["vocabulary_role"] == "C1" for row in result["query_expansion"])
    assert all(row["direct_udc_vote"] is False for row in result["query_expansion"])
    assert all("NDC10" not in str(row.get("notation")) for row in result["union"])
    serialized = json.dumps(result)
    assert "gold_" not in serialized


def test_dense_index_is_resumable_and_uses_the_frozen_embedding_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rows = [
        _external_row(
            f"dense-{index}",
            role="bibliographic_assignment",
            source_field="080",
            notation="004.8",
            title="AI memory systems",
            rights_ref="czech-national-bibliography",
        )
        for index in range(20)
    ]
    source_manifest = _package(tmp_path, name="bib", rows=rows)
    index_path = tmp_path / "index" / "evidence.sqlite3"
    build_source_index(
        package_manifest_paths=[source_manifest],
        output_path=index_path,
        root=tmp_path,
        dense_limit=10,
    )

    def embed(texts, **_kwargs):
        return (
            "bge-m3",
            [
                [
                    float("AI" in text),
                    float("memory" in text),
                    float(len(text) % 17) / 17,
                ]
                for text in texts
            ],
        )

    monkeypatch.setattr(
        classification_library_evidence,
        "embed_texts_cancellable",
        embed,
    )
    manifest_path = index_path.with_suffix(".manifest.json")
    built = build_dense_index(manifest_path, batch_size=3)
    resumed = build_dense_index(manifest_path, batch_size=3)

    assert built["dense_index_built"] is True
    assert built["dense_model"] == "bge-m3"
    assert built["dense_model_license"]
    assert built["dense_training_corpus_license"]
    assert built["dense_count"] <= 10
    assert resumed["dense_vectors_sha256"] == built["dense_vectors_sha256"]
    index = LibraryEvidenceIndex(manifest_path)
    matches = index.query_support_dense(
        "AI memory",
        arms=["B1a", "B1b"],
        limit=5,
    )
    assert matches
    assert all("dense_score" in row for row in matches)


def test_external_test_cases_are_group_held_out_and_gold_isolated(
    tmp_path: Path,
) -> None:
    rows = [
        _external_row(
            f"heldout-{index}",
            role="bibliographic_assignment",
            source_field="080",
            notation="004.8",
            title=f"AI systems {index}",
            rights_ref="czech-national-bibliography",
        )
        for index in range(100)
    ]
    manifest_path = _package(tmp_path, name="bib", rows=rows)

    cases = external_test_cases(
        package_manifest_paths=[manifest_path],
        package=default_udc_package(),
        arms=["B1b"],
    )

    assert cases
    assert all(row["external_arm"] == "B1b" for row in cases)
    assert all(row["gold_allowed_primary_notations"] == ["004.8"] for row in cases)
    assert len({row["fixture_group_id"] for row in cases}) == len(cases)
    assert all("source_assignments" not in inference_dto(row) for row in cases)
