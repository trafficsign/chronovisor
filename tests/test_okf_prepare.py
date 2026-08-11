from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from chronovisor.core.canonical_document import (
    extract_markdown_links,
    parse_document,
    resolve_internal_markdown_links,
)
from chronovisor.core.okf_prepare import (
    InvalidStatusError,
    RawSource,
    SourceDocument,
    prepare_okf_migration,
    require_resolved_links,
)

FIXTURE = Path(__file__).parent / "fixtures" / "okf_prepare.json"


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _source(row: dict[str, Any]) -> SourceDocument:
    return SourceDocument(
        relative_path=row["path"],
        data=row["data"].encode(),
        namespace=row.get("namespace", "pages"),
    )


def _prepare():
    fixture = _fixture()
    return fixture, prepare_okf_migration(
        (_source(row) for row in fixture["documents"]),
        catalog=fixture["catalog"],
        raw_files=(
            RawSource(row["path"], row["data"].encode()) for row in fixture["raw"]
        ),
    )


def test_prepare_maps_status_cohorts_and_separates_non_concepts() -> None:
    _, plan = _prepare()
    cohorts = {
        cohort.input_status: (cohort.output_status, cohort.count, cohort.uids)
        for cohort in plan.manifest.status_cohorts
    }
    assert cohorts == {
        "missing": ("stable", 1, ("uid-missing",)),
        "active": ("stable", 1, ("uid-active",)),
        "draft": ("draft", 1, ("uid-draft",)),
        "stable": ("stable", 1, ("uid-stable",)),
        "deprecated": ("deprecated", 1, ("uid-deprecated",)),
        "archived": ("deprecated", 1, ("uid-archived",)),
    }
    assert [item.relative_path for item in plan.reserved_documents] == [
        "index.md",
        "log.md",
        "schema.md",
    ]
    assert [item.relative_path for item in plan.system_documents] == [
        "current-state.md"
    ]
    assert len(plan.converted_documents) == 6


def test_prepare_converts_metadata_links_and_archive_event() -> None:
    _, plan = _prepare()
    _, repeated = _prepare()
    active = next(item for item in plan.converted_documents if item.uid == "uid-active")
    document = parse_document(active.data)

    assert document.metadata["type"] == "Concept"
    assert document.metadata["description"] == "Legacy summary"
    assert document.metadata["status"] == "stable"
    assert document.metadata["superseded_by"] == "destination"
    assert not {
        "summary",
        "chronovisor_status",
        "archive_reason",
        "archive_provenance",
    }.intersection(document.metadata)
    assert document.body == (
        b"Before [destination](../deep/nested/destination.md) after [[unknown]].\n"
        b"\n`[[destination]]`\n\n```text\n[[destination]]\n```\n"
    )
    assert [(item.uid, item.target) for item in plan.manifest.unresolved_links] == [
        ("uid-active", "unknown")
    ]
    assert len(plan.events) == 1
    assert repeated.events == plan.events
    assert plan.events[0].event_id.startswith("okf-archive-")
    assert json.loads(plan.events[0].payload_json)["archive_reason"] == "merged"

    stable = next(item for item in plan.converted_documents if item.uid == "uid-stable")
    assert parse_document(stable.data).metadata["extension"] == {
        "nested": ["one", "two"]
    }


@pytest.mark.parametrize(
    ("yaml_value", "expected"),
    [
        ("[merged, duplicate]", '["merged","duplicate"]'),
        ("{source: review, rank: 1}", '{"rank":1,"source":"review"}'),
        ("2026-08-11", "2026-08-11"),
    ],
)
def test_archive_metadata_values_are_deterministic_activity_text(
    yaml_value: str,
    expected: str,
) -> None:
    source = SourceDocument(
        "archive.md",
        (
            "---\n"
            "uid: uid-archive\n"
            "status: archived\n"
            f"archive_reason: {yaml_value}\n"
            "---\n"
            "Archived body.\n"
        ).encode(),
    )

    first = prepare_okf_migration((source,), catalog={})
    second = prepare_okf_migration((source,), catalog={})

    assert first.events == second.events
    assert json.loads(first.events[0].payload_json)["archive_reason"] == expected


def test_prepare_converts_wikilink_labels_and_anchors_outside_code() -> None:
    source = SourceDocument(
        "notes/source.md",
        b"---\nuid: uid-source\ntype: Concept\nstatus: stable\n---\n"
        b"[[destination|Readable label]] "
        b"[[destination#Section heading]] "
        b"[[destination#Section heading|Heading label]] "
        b"[[unknown#Missing|Unknown label]]\n\n"
        b"`[[destination|inline code]]`\n\n"
        b"```text\n[[destination#Section heading|fenced code]]\n```\n",
    )

    plan = prepare_okf_migration(
        [source], catalog={"destination": "deep/nested/destination.md"}
    )
    converted = parse_document(plan.converted_documents[0].data)

    assert converted.body == (
        b"[Readable label](../deep/nested/destination.md) "
        b"[destination#Section heading](<../deep/nested/destination.md#Section heading>) "
        b"[Heading label](<../deep/nested/destination.md#Section heading>) "
        b"[[unknown#Missing|Unknown label]]\n\n"
        b"`[[destination|inline code]]`\n\n"
        b"```text\n[[destination#Section heading|fenced code]]\n```\n"
    )
    assert plan.manifest.documents[0].resolved_link_count == 3
    assert extract_markdown_links(converted.body) == (
        "../deep/nested/destination.md",
        "../deep/nested/destination.md#Section heading",
        "../deep/nested/destination.md#Section heading",
    )
    assert [(item.uid, item.target) for item in plan.manifest.unresolved_links] == [
        ("uid-source", "unknown")
    ]


def test_prepare_delinks_known_plain_text_targets_and_keeps_unknown_unresolved() -> None:
    source = SourceDocument(
        "notes/source.md",
        b"---\nuid: uid-source\ntype: Concept\nstatus: stable\n---\n"
        b"[[claude-code]] [[lessons-learned|Lessons archive]] [[unknown]]\n",
    )

    first = prepare_okf_migration(
        [source],
        catalog={},
        plain_text_targets=("claude-code", "lessons-learned"),
    )
    repeated = prepare_okf_migration(
        [source],
        catalog={},
        plain_text_targets=("lessons-learned", "claude-code"),
    )
    converted = parse_document(first.converted_documents[0].data)

    assert converted.body == b"claude-code Lessons archive [[unknown]]\n"
    assert extract_markdown_links(converted.body) == ()
    assert first.manifest.documents[0].resolved_link_count == 2
    assert [(item.uid, item.target) for item in first.manifest.unresolved_links] == [
        ("uid-source", "unknown")
    ]
    assert repeated == first
    with pytest.raises(ValueError, match=r"uid-source='unknown'"):
        require_resolved_links(first)


def test_prepare_rewrites_canonical_links_to_stable_only() -> None:
    def page(
        path: str, status: str, *, superseded_by: str | None = None
    ) -> SourceDocument:
        replacement = f"superseded_by: {superseded_by}\n" if superseded_by else ""
        return SourceDocument(
            path,
            (
                "---\n"
                f"uid: uid-{Path(path).stem}\n"
                "type: Concept\n"
                f"status: {status}\n"
                f"{replacement}"
                "---\n"
            ).encode(),
        )

    source = SourceDocument(
        "notes/source.md",
        b"---\nuid: uid-source\ntype: Concept\nstatus: stable\n---\n"
        b"[Stable](../stable.md#Kept) "
        b"[Deprecated](../deprecated.md#Old) "
        b"[Draft](../draft.md#Old) "
        b"[Invalid replacement](../invalid.md) "
        b"[No replacement](../orphan.md) "
        b"[Missing](../missing.md) "
        b"[Local](/Users/person/private.md).\n",
    )
    documents = (
        source,
        page("stable.md", "stable"),
        page("replacement.md", "stable"),
        page("deprecated.md", "deprecated", superseded_by="replacement"),
        page("draft.md", "draft", superseded_by="replacement"),
        page("invalid.md", "deprecated", superseded_by="orphan"),
        page("orphan.md", "deprecated"),
    )
    catalog = {Path(item.relative_path).stem: item.relative_path for item in documents}

    plan = prepare_okf_migration(documents, catalog=catalog)
    converted_source = next(
        item for item in plan.converted_documents if item.uid == "uid-source"
    )
    converted = parse_document(converted_source.data)

    assert converted.body == (
        b"[Stable](../stable.md#Kept) "
        b"[Deprecated](<../replacement.md>) "
        b"[Draft](<../replacement.md>) "
        b"Invalid replacement No replacement Missing Local.\n"
    )
    source_manifest = next(
        item for item in plan.manifest.documents if item.uid == "uid-source"
    )
    assert source_manifest.resolved_link_count == 6
    links = resolve_internal_markdown_links(
        converted.body, source_namespace="pages", source_path="notes/source.md"
    )
    assert [(link.path, link.fragment) for link in links] == [
        ("stable.md", "Kept"),
        ("replacement.md", None),
        ("replacement.md", None),
    ]


def test_resolved_link_gate_fails_closed_with_manifest_report() -> None:
    _, plan = _prepare()

    with pytest.raises(ValueError, match=r"unresolved wikilinks: uid-active='unknown'"):
        require_resolved_links(plan)

    resolved = prepare_okf_migration(
        [
            SourceDocument(
                "notes/source.md",
                b"---\nuid: uid-source\ntype: Concept\nstatus: stable\n---\n"
                b"[[destination]]\n",
            )
        ],
        catalog={"destination": "destination.md"},
    )
    assert require_resolved_links(resolved) is None


def test_prepare_fails_invalid_status_with_uid_and_value() -> None:
    fixture = _fixture()
    with pytest.raises(InvalidStatusError) as caught:
        prepare_okf_migration([_source(fixture["invalid"])], catalog=fixture["catalog"])
    assert [(item.uid, item.value) for item in caught.value.invalid_statuses] == [
        ("uid-invalid", "obsolete")
    ]

    explicit_missing = SourceDocument(
        "notes/not-absent.md",
        b"---\nuid: uid-not-absent\ntype: Concept\nstatus: missing\n---\n",
    )
    with pytest.raises(InvalidStatusError) as missing_caught:
        prepare_okf_migration([explicit_missing], catalog={})
    assert missing_caught.value.invalid_statuses[0].value == "missing"


def test_prepare_is_idempotent_and_raw_manifest_does_not_transform_raw() -> None:
    fixture, first = _prepare()
    second = prepare_okf_migration(
        (
            SourceDocument(item.relative_path, item.data)
            for item in first.converted_documents
        ),
        catalog=fixture["catalog"],
    )
    assert [item.data for item in second.converted_documents] == [
        item.data for item in first.converted_documents
    ]

    raw = fixture["raw"][0]["data"].encode()
    assert [
        (item.relative_path, item.size, item.sha256)
        for item in first.manifest.raw_files
    ] == [("sessions/raw.jsonl", len(raw), hashlib.sha256(raw).hexdigest())]
    assert fixture["raw"][0]["data"].encode() == raw


@pytest.mark.parametrize(
    "relative_path",
    [
        "notes\\page.md",
        "notes/\0page.md",
        "notes/\x1fpage.md",
        "/absolute.md",
        "C:/absolute.md",
        "notes/../page.md",
    ],
)
def test_prepare_rejects_unsafe_relative_paths(relative_path: str) -> None:
    with pytest.raises(ValueError, match="path must be bundle-relative"):
        prepare_okf_migration(
            [
                SourceDocument(
                    relative_path,
                    b"---\nuid: uid-source\ntype: Concept\nstatus: stable\n---\n",
                )
            ],
            catalog={},
        )
