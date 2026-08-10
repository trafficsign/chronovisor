from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor.core.canonical_document import (
    CanonicalDocument,
    CanonicalDocumentError,
    extract_markdown_links,
    parse_document,
    serialize_document,
)
from chronovisor.core.okf_v02 import (
    scan_concept_paths,
    validate_concept,
    validate_pages_bundle,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "okf_v02"


def test_full_yaml_semantic_round_trip_preserves_body_bytes() -> None:
    source = (FIXTURE_ROOT / "pages" / "topics" / "nested.md").read_bytes()
    document = parse_document(source)

    assert document.metadata["resource"]["trust"]["signals"] == [
        "local",
        "reviewed",
    ]
    assert document.metadata["chronovisor_extension"]["policy"] == {
        "enabled": True,
        "notes": None,
    }

    reparsed = parse_document(serialize_document(document))
    assert reparsed.metadata == document.metadata
    assert reparsed.body == document.body


def test_concept_contract_distinguishes_errors_from_warnings() -> None:
    issues = validate_concept({"type": "Concept", "status": "stable"})
    assert {issue.field for issue in issues if issue.severity == "warning"} == {
        "title",
        "description",
        "resource",
        "tags",
    }
    assert not [issue for issue in issues if issue.severity == "error"]

    invalid = validate_concept({"type": ["Concept"], "status": ["stable"]})
    assert {issue.code for issue in invalid if issue.severity == "error"} == {
        "type_required",
        "status_invalid",
    }


def test_pages_bundle_conformance_excludes_reserved_and_system_documents() -> None:
    pages_root = FIXTURE_ROOT / "pages"
    assert (FIXTURE_ROOT / "system" / "invalid.md").is_file()
    assert [
        path.relative_to(pages_root).as_posix()
        for path in scan_concept_paths(pages_root)
    ] == ["topics/nested.md"]

    issues = validate_pages_bundle(pages_root)
    assert not [issue for issue in issues if issue.severity == "error"]


def test_root_index_is_versioned_but_is_not_a_concept(tmp_path: Path) -> None:
    pages_root = tmp_path / "pages"
    pages_root.mkdir()
    (pages_root / "index.md").write_bytes(b"---\nokf_version: 0.1\n---\n")

    issues = validate_pages_bundle(pages_root)
    assert [(issue.code, issue.path) for issue in issues] == [
        ("okf_version_invalid", "index.md")
    ]


def test_standard_markdown_links_are_extracted_without_resolution() -> None:
    document = parse_document(
        (FIXTURE_ROOT / "pages" / "topics" / "nested.md").read_bytes()
    )
    assert extract_markdown_links(document.body) == (
        "../index.md#example-bundle",
        "missing.md",
    )


def test_serializer_never_rewrites_supplied_body_bytes() -> None:
    body = b"# Body\r\n\r\nbytes stay as supplied\r\n"
    rendered = serialize_document(
        CanonicalDocument(metadata={"type": "Concept"}, body=body)
    )
    assert parse_document(rendered).body == body


def test_duplicate_yaml_keys_fail_closed() -> None:
    with pytest.raises(CanonicalDocumentError, match="duplicate key 'policy'"):
        parse_document(
            b"---\ntype: Concept\nextension:\n  policy: first\n  policy: second\n---\n"
        )
