from __future__ import annotations

from pathlib import Path

import pytest

from chronovisor.core.canonical_document import (
    CanonicalDocument,
    CanonicalDocumentError,
    ResolvedMarkdownLink,
    extract_markdown_links,
    format_internal_markdown_link,
    parse_document,
    patch_document_metadata,
    resolve_internal_markdown_link,
    resolve_internal_markdown_links,
    serialize_document,
    validate_canonical_document,
)
from chronovisor.core.okf_v02 import (
    OKF_SPEC_PATH,
    OKF_SPEC_REVISION,
    OKF_SPEC_SHA256,
    OKF_SPEC_URL,
    main,
    scan_concept_paths,
    validate_concept,
    validate_pages_bundle,
    validate_production_concept,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "okf_v02"


def test_validator_pins_the_reviewed_upstream_spec() -> None:
    assert OKF_SPEC_REVISION == "374e0bc4c644310ff56cdf9c0fe81eccdec862b0"
    assert OKF_SPEC_PATH == "okf/SPEC.md"
    assert OKF_SPEC_SHA256 == (
        "5a3311d270bebb16d558010e75064f5b75323f284992641732b1c8097511f948"
    )
    assert OKF_SPEC_REVISION in OKF_SPEC_URL


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
    }
    assert {issue.code for issue in invalid if issue.severity == "warning"} >= {
        "status_invalid"
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


def test_root_index_is_optional_and_unknown_version_is_soft(tmp_path: Path) -> None:
    pages_root = tmp_path / "pages"
    pages_root.mkdir()
    assert not [
        issue
        for issue in validate_pages_bundle(pages_root)
        if issue.severity == "error"
    ]

    (pages_root / "index.md").write_bytes(
        b"---\nokf_version: 0.1\nextension: kept\n---\n"
        b"# Bundle\n\n* [Concept](concept.md) - description\n"
    )

    issues = validate_pages_bundle(pages_root)
    assert not [issue for issue in issues if issue.severity == "error"]
    assert {(issue.code, issue.field) for issue in issues} == {
        ("okf_version_unknown", "okf_version"),
        ("index_frontmatter_key_unknown", "extension"),
    }


def test_portable_conformance_rejects_only_required_concept_failures(
    tmp_path: Path,
) -> None:
    pages_root = tmp_path / "pages"
    pages_root.mkdir()
    (pages_root / "unknown.md").write_bytes(
        b"---\ntype: Producer Defined\nextra: preserved\n---\n[Broken](missing.md)\n"
    )
    (pages_root / "missing-type.md").write_bytes(b"---\ntitle: Missing\n---\n")
    (pages_root / "invalid.md").write_bytes(b"not frontmatter\n")

    errors = {
        (issue.code, issue.path)
        for issue in validate_pages_bundle(pages_root)
        if issue.severity == "error"
    }
    assert errors == {
        ("type_required", "missing-type.md"),
        ("document_invalid", "invalid.md"),
    }


def test_reserved_documents_are_validated_at_every_level(tmp_path: Path) -> None:
    pages_root = tmp_path / "pages"
    nested = pages_root / "nested"
    nested.mkdir(parents=True)
    (pages_root / "index.md").write_text(
        "* [Before section](premature.md) - ignored\n# Missing grouped link\n"
    )
    (nested / "index.md").write_text(
        "---\ntitle: forbidden\n---\n# Nested\n\n* [Page](page.md)\n"
    )
    (pages_root / "log.md").write_text(
        "# Updates\n\n## 2026-02-30\n* Invalid date\n## 2026-08-10\n* Out of order\n"
    )
    (nested / "log.md").write_text("# Updates\n\n* Ungrouped prose\n")

    issues = validate_pages_bundle(pages_root)
    assert {issue.code for issue in issues if issue.severity == "error"} == {
        "index_frontmatter_forbidden",
        "index_link_missing",
        "log_date_invalid",
        "log_entry_ungrouped",
    }
    assert {issue.code for issue in issues if issue.severity == "warning"} == {
        "index_description_missing"
    }


def test_log_dates_must_be_newest_first_but_action_labels_are_optional(
    tmp_path: Path,
) -> None:
    pages_root = tmp_path / "pages"
    pages_root.mkdir()
    log = pages_root / "log.md"
    log.write_text(
        "# Updates\n\n## 2026-08-08\n* Plain prose is enough.\n"
        "## 2026-08-09\n* Another entry.\n"
    )
    assert {issue.code for issue in validate_pages_bundle(pages_root)} == {
        "log_dates_not_newest_first"
    }


def test_log_allows_parseable_frontmatter_and_unknown_keys(tmp_path: Path) -> None:
    pages_root = tmp_path / "pages"
    pages_root.mkdir()
    (pages_root / "log.md").write_text(
        "---\ntype: Log\ntitle: Bundle history\nproducer_extension: kept\n---\n"
        "# Bundle history\n\n## 2026-07-01\n\n- Verified the bundle.\n"
    )

    assert validate_pages_bundle(pages_root) == ()


def test_log_rejects_malformed_frontmatter(tmp_path: Path) -> None:
    pages_root = tmp_path / "pages"
    pages_root.mkdir()
    (pages_root / "log.md").write_text(
        "---\ntype: [\n---\n# Bundle history\n\n## 2026-07-01\n- Entry\n"
    )

    assert {
        (issue.severity, issue.code) for issue in validate_pages_bundle(pages_root)
    } == {("error", "document_invalid")}


def test_index_rejects_entries_without_a_relative_target(tmp_path: Path) -> None:
    pages_root = tmp_path / "pages"
    pages_root.mkdir()
    (pages_root / "index.md").write_text(
        "# Concepts\n\n"
        "* [Valid](valid.md) - description\n"
        "* [External](https://example.test/page.md) - description\n"
        "* [Root absolute](/page.md) - description\n"
        "* Missing link - description\n"
    )

    assert {issue.code for issue in validate_pages_bundle(pages_root)} == {
        "index_entry_invalid"
    }


def test_cli_output_is_stable_and_never_changes_bundle_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pages_root = tmp_path / "pages"
    pages_root.mkdir()
    (pages_root / "index.md").write_text("# Bundle\n\n* [Page](page.md)\n")
    (pages_root / "page.md").write_text("---\ntype: Example\n---\n")
    before = {
        path.relative_to(pages_root).as_posix(): path.read_bytes()
        for path in sorted(pages_root.rglob("*"))
        if path.is_file()
    }

    assert main([str(pages_root)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "warning\tindex_description_missing\tindex.md\tdescription",
        "warning\trecommended_field_missing\tpage.md\tdescription",
        "warning\trecommended_field_missing\tpage.md\tresource",
        "warning\trecommended_field_missing\tpage.md\ttags",
        "warning\trecommended_field_missing\tpage.md\ttitle",
    ]
    assert before == {
        path.relative_to(pages_root).as_posix(): path.read_bytes()
        for path in sorted(pages_root.rglob("*"))
        if path.is_file()
    }


def test_cli_returns_one_when_conformance_has_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pages_root = tmp_path / "pages"
    pages_root.mkdir()
    (pages_root / "invalid.md").write_text("missing frontmatter\n")

    assert main([str(pages_root)]) == 1
    assert capsys.readouterr().out.startswith("error\tdocument_invalid\tinvalid.md\t")


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


def test_metadata_patch_adds_updates_and_deletes_without_touching_body() -> None:
    source = (FIXTURE_ROOT / "pages" / "topics" / "nested.md").read_bytes()
    original = parse_document(source)

    patched = parse_document(
        patch_document_metadata(
            source,
            {
                "status": "deprecated",
                "new_extension": {"nested": [True, None]},
            },
            delete=("description",),
        )
    )

    assert patched.metadata["status"] == "deprecated"
    assert "description" not in patched.metadata
    assert patched.metadata["new_extension"] == {"nested": [True, None]}
    assert (
        patched.metadata["chronovisor_extension"]
        == original.metadata["chronovisor_extension"]
    )
    assert patched.body == original.body

    with pytest.raises(CanonicalDocumentError, match="update and delete"):
        patch_document_metadata(
            source,
            {"status": "stable"},
            delete=("status",),
        )


def test_production_concept_requires_explicit_canonical_status() -> None:
    assert not [
        issue
        for issue in validate_production_concept(
            {"type": "Concept", "status": "stable"}
        )
        if issue.severity == "error"
    ]
    assert {
        issue.code
        for issue in validate_production_concept({"type": "Concept"})
        if issue.severity == "error"
    } == {"status_required"}
    assert {
        issue.code
        for issue in validate_production_concept({"type": "", "status": "active"})
        if issue.severity == "error"
    } == {"type_required", "status_invalid"}
    assert {
        issue.code
        for issue in validate_production_concept({"type": "Concept", "status": None})
        if issue.severity == "error"
    } == {"status_invalid"}


def test_link_extraction_owns_frontmatter_and_code_protection() -> None:
    markdown = """---
reference: "[Frontmatter](ignored-frontmatter.md)"
---
[Prose](kept.md)
`[Inline](ignored-inline.md)`

~~~markdown
[Fence](ignored-fence.md)
~~~

![Image](ignored-image.png)
"""

    assert extract_markdown_links(markdown) == ("kept.md",)


def test_internal_link_resolution_is_namespace_aware_and_pure() -> None:
    body = """[Index](../index.md#Bundle)
[Root](/guide.md)
[External](https://example.test/page.md)
[Mail](mailto:owner@example.test)
[Same](source.md#here)
[Fragment](#here)
"""

    assert resolve_internal_markdown_links(
        body,
        source_namespace="pages",
        source_path="topics/source.md",
    ) == (
        ResolvedMarkdownLink("pages", "index.md", "Bundle"),
        ResolvedMarkdownLink("pages", "guide.md"),
    )
    assert resolve_internal_markdown_links(
        "[Page](../pages/topics/topic.md#Part) [Schema](/schema.md)",
        source_namespace="system",
        source_path="state.md",
    ) == (
        ResolvedMarkdownLink("pages", "topics/topic.md", "Part"),
        ResolvedMarkdownLink("system", "schema.md"),
    )


@pytest.mark.parametrize(
    "target",
    [
        "../escape.md",
        "%2e%2e/escape.md",
        "%252e%252e/escape.md",
        "/system/private.md",
    ],
)
def test_pages_link_resolution_fails_closed_at_boundaries(target: str) -> None:
    with pytest.raises(CanonicalDocumentError, match="escapes|crosses"):
        resolve_internal_markdown_link(
            target,
            source_namespace="pages",
            source_path="source.md",
        )


def test_duplicate_yaml_keys_fail_closed() -> None:
    with pytest.raises(CanonicalDocumentError, match="duplicate key 'policy'"):
        parse_document(
            b"---\ntype: Concept\nextension:\n  policy: first\n  policy: second\n---\n"
        )


def test_canonical_writer_validation_rejects_legacy_and_missing_targets() -> None:
    with pytest.raises(CanonicalDocumentError, match="wikilinks"):
        validate_canonical_document(
            b"---\ntitle: Source\nstatus: stable\ntype: knowledge\n---\nSee [[target]].\n",
            namespace="pages",
            path="notes/source.md",
            require_stable=True,
            allowed_targets={("pages", "notes/target.md")},
        )

    with pytest.raises(CanonicalDocumentError, match="missing Markdown link"):
        validate_canonical_document(
            b"---\ntitle: Source\nstatus: stable\ntype: knowledge\n---\n[Target](target.md)\n",
            namespace="pages",
            path="notes/source.md",
            require_stable=True,
            allowed_targets=set(),
        )


@pytest.mark.parametrize("page_type", [None, "", "   "])
def test_pages_canonical_validation_requires_non_empty_type(
    page_type: str | None,
) -> None:
    type_line = "" if page_type is None else f"type: {page_type!r}\n"
    data = f"---\ntitle: Source\nstatus: stable\n{type_line}---\nbody\n".encode()

    with pytest.raises(CanonicalDocumentError, match="non-empty type"):
        validate_canonical_document(
            data,
            namespace="pages",
            path="notes/source.md",
        )


def test_internal_link_formatter_is_relative_and_namespace_safe() -> None:
    assert (
        format_internal_markdown_link(
            "Target",
            source_namespace="pages",
            source_path="hubs/source.md",
            target_namespace="pages",
            target_path="notes/target.md",
        )
        == "[Target](<../notes/target.md>)"
    )
    assert (
        format_internal_markdown_link(
            "Target",
            source_namespace="system",
            source_path="current-state.md",
            target_namespace="pages",
            target_path="notes/target.md",
        )
        == "[Target](</pages/notes/target.md>)"
    )
    with pytest.raises(CanonicalDocumentError, match="cannot cross"):
        format_internal_markdown_link(
            "Private",
            source_namespace="pages",
            source_path="source.md",
            target_namespace="system",
            target_path="private.md",
        )


@pytest.mark.parametrize("legacy_link", ["[[target]]", "![[target]]", "[[]]"])
def test_canonical_document_rejects_legacy_links_and_embeds(
    legacy_link: str,
) -> None:
    data = f"---\ntitle: Source\nstatus: stable\ntype: knowledge\n---\n{legacy_link}\n".encode()

    with pytest.raises(CanonicalDocumentError, match="legacy wikilinks"):
        validate_canonical_document(
            data,
            namespace="pages",
            path="notes/source.md",
        )

    escaped = data.replace(b"[[", b"\\[[")
    validate_canonical_document(
        escaped,
        namespace="pages",
        path="notes/source.md",
    )


def test_internal_link_formatter_escapes_labels_and_rejects_newlines() -> None:
    rendered = format_internal_markdown_link(
        "A [label] \\",
        source_namespace="pages",
        source_path="notes/source.md",
        target_namespace="pages",
        target_path="notes/target.md",
    )
    assert rendered == r"[A \[label\] \\](<target.md>)"
    with pytest.raises(CanonicalDocumentError, match="missing Markdown link"):
        validate_canonical_document(
            (
                "---\ntitle: Source\nstatus: stable\ntype: knowledge\n---\n"
                f"{rendered}\n"
            ).encode(),
            namespace="pages",
            path="notes/source.md",
            allowed_targets=set(),
        )
    with pytest.raises(CanonicalDocumentError, match="newlines"):
        format_internal_markdown_link(
            "unsafe\nlabel",
            source_namespace="pages",
            source_path="notes/source.md",
            target_namespace="pages",
            target_path="notes/target.md",
        )
