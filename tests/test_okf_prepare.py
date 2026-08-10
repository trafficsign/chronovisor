from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from chronovisor.core.canonical_document import parse_document
from chronovisor.migration.okf_prepare import (
    InvalidStatusError,
    RawSource,
    SourceDocument,
    prepare_okf_migration,
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
