from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import ClassVar

from chronovisor import classification_library_sources
from chronovisor.classification_library_sources import (
    MARC_NS,
    czech_bibliography_contract,
    download_file,
    fetch_oai_window,
    ndl_bibliography_contract,
    normalize_czech_authority_record,
    normalize_czech_bibliographic_record,
    stable_sample,
    validate_ndl_provider,
)
from chronovisor.durable_state import write_sealed_json


def _record(fields: str) -> ET.Element:
    return ET.fromstring(
        f'<record xmlns="{MARC_NS}"><controlfield tag="001">id-1</controlfield>'
        '<controlfield tag="008">000000s2024                        cze  </controlfield>'
        f"{fields}</record>"
    )


def test_czech_080_preserves_components_and_links_matching_883() -> None:
    record = _record(
        '<datafield tag="245" ind1="1" ind2="0"><subfield code="a">AI systems</subfield></datafield>'
        '<datafield tag="080" ind1=" " ind2=" ">'
        '<subfield code="a">004.8</subfield><subfield code="x">test</subfield>'
        '<subfield code="2">MRF</subfield><subfield code="8">1.1\\a</subfield></datafield>'
        '<datafield tag="883" ind1="2" ind2=" ">'
        '<subfield code="8">1.1\\a</subfield><subfield code="a">human</subfield>'
        '<subfield code="q">agency</subfield></datafield>'
    )

    row = normalize_czech_bibliographic_record(record)
    assignment = row["source_assignments"][0]

    assert assignment["source_field"] == "080"
    assert assignment["components"]["a"] == ["004.8"]
    assert assignment["components"]["x"] == ["test"]
    assert assignment["components"]["subfield_2"] == ["MRF"]
    assert assignment["provenance_883"]["parsed_field_link_key"] == "1.1"
    assert assignment["generation_method"] == "not_machine_generated"
    assert assignment["intellectual_assignment"] == "confirmed"


def test_czech_authority_uses_only_089_a() -> None:
    record = _record(
        '<datafield tag="150" ind1=" " ind2=" "><subfield code="a">AI</subfield></datafield>'
        '<datafield tag="089" ind1=" " ind2=" "><subfield code="a">004.8</subfield>'
        '<subfield code="c">do-not-use</subfield><subfield code="d">do-not-use</subfield></datafield>'
    )

    row = normalize_czech_authority_record(record)
    assignment = row["source_assignments"][0]

    assert assignment["source_field"] == "089$a"
    assert assignment["notation_or_uri"] == "004.8"
    assert assignment["components"] == {
        "a": ["004.8"],
        "x": [],
        "subfield_2": [],
        "subfield_8": [],
    }
    assert "do-not-use" not in str(row)


def test_stable_sample_is_stratified_and_order_independent() -> None:
    rows = [
        {
            "source_record_id": f"id-{index}",
            "language": "jpn" if index % 2 else "cze",
            "major_class": str(index % 3),
            "year_bucket": "2020s",
        }
        for index in range(30)
    ]

    left = stable_sample(rows, limit=12)
    right = stable_sample(reversed(rows), limit=12)

    assert [row["source_record_id"] for row in left] == [
        row["source_record_id"] for row in right
    ]
    assert len({(row["language"], row["major_class"]) for row in left}) == 6


def test_ndl_unknown_provider_is_rejected() -> None:
    contract = ndl_bibliography_contract(
        "https://ndl.example/oai", provider_allowlist=["ndl"]
    )

    allowed, reason = validate_ndl_provider(
        {
            "provider_id": "other",
            "record_creator": "Other",
            "provider_terms_url": "https://other.example/terms",
            "record_license_class": "unknown",
        },
        contract,
    )

    assert allowed is False
    assert reason == "provider_not_allowlisted"
    assert (
        czech_bibliography_contract("https://nkp.example").redistribution_policy[
            "bundle_in_repo_or_wheel"
        ]
        is False
    )
    contract_fields = czech_bibliography_contract("https://nkp.example")
    assert contract_fields.software_license == "not-applicable"
    assert contract_fields.model_license == "not-applicable"
    assert contract_fields.training_corpus_license == "not-applicable"


def test_oai_resumption_checkpoint_replays_without_network(
    tmp_path: Path,
    monkeypatch,
) -> None:
    responses = [
        (
            f'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
            f'<ListRecords><record xmlns="{MARC_NS}">'
            '<controlfield tag="001">one</controlfield></record>'
            "<resumptionToken>next</resumptionToken></ListRecords></OAI-PMH>"
        ).encode(),
        (
            f'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
            f'<ListRecords><record xmlns="{MARC_NS}">'
            '<controlfield tag="001">two</controlfield></record>'
            "<resumptionToken></resumptionToken></ListRecords></OAI-PMH>"
        ).encode(),
    ]

    class Response:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

    class Client:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, _url, params):
            value = responses.pop(0)
            self.calls += 1
            return Response(value)

    monkeypatch.setattr(classification_library_sources.httpx, "Client", Client)
    checkpoint = tmp_path / "oai"
    first, receipt = fetch_oai_window(
        base_url="https://example.invalid/oai",
        metadata_prefix="marc21",
        from_date="2026-01-01",
        until_date="2026-01-02",
        checkpoint_dir=checkpoint,
    )

    assert receipt["request_count"] == 2
    assert receipt["record_count"] == 2
    assert b"one" in first and b"two" in first

    class OfflineClient(Client):
        def get(self, _url, params):
            raise AssertionError("completed checkpoint must not refetch")

    monkeypatch.setattr(
        classification_library_sources.httpx,
        "Client",
        OfflineClient,
    )
    replay, replay_receipt = fetch_oai_window(
        base_url="https://example.invalid/oai",
        metadata_prefix="marc21",
        from_date="2026-01-01",
        until_date="2026-01-02",
        checkpoint_dir=checkpoint,
    )
    assert replay == first
    assert replay_receipt["resumed_pages"] == 2


def test_download_resumes_only_from_matching_content_range(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "archive.gz"
    partial = tmp_path / ".archive.gz.part"
    partial.write_bytes(b"abc")
    write_sealed_json(
        tmp_path / ".archive.gz.part.json",
        {
            "schema": "chronovisor.download-checkpoint.v1",
            "url": "https://example.invalid/archive.gz",
            "etag": '"v1"',
        },
    )

    class Stream:
        status_code = 206
        headers: ClassVar[dict[str, str]] = {
            "content-range": "bytes 3-5/6",
            "etag": '"v1"',
            "last-modified": "today",
            "content-type": "application/gzip",
        }

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        def iter_bytes(self):
            yield b"def"

    monkeypatch.setattr(
        classification_library_sources.httpx,
        "stream",
        lambda *args, **kwargs: Stream(),
    )
    receipt = download_file(
        "https://example.invalid/archive.gz",
        target,
        size_cap_bytes=100,
    )

    assert target.read_bytes() == b"abcdef"
    assert receipt["resumed_from_bytes"] == 3
    assert receipt["reused_complete_file"] is False
