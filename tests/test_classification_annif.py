from __future__ import annotations

import json
from pathlib import Path

from chronovisor.classification import UDCPackage
from chronovisor.classification_annif import (
    _mapped_notations,
    _notation_matches,
    export_corpus,
    export_vocabulary,
    finalize_czech_checkpoint,
    write_projects,
)
from chronovisor.classification_fixture_set import sha256_file
from chronovisor.classification_library_sources import MARC_NS
from chronovisor.durable_state import read_sealed_json, write_sealed_json


def _package() -> UDCPackage:
    concepts = {
        "https://example.test/004.8": {
            "uri": "https://example.test/004.8",
            "notation": "004.8",
            "label_en": "Artificial intelligence",
            "label_ja": "人工知能",
        },
        "https://example.test/331": {
            "uri": "https://example.test/331",
            "notation": "331",
            "label_en": "Labour",
            "label_ja": "労働",
        },
    }
    return UDCPackage(
        release="test",
        checksum="sha256:test",
        source_url="https://example.test",
        license="CC BY-SA 3.0",
        attribution="test",
        complete=False,
        concepts=concepts,
    )


def _record(record_id: str, notation: str) -> dict:
    return {
        "source_record_id": record_id,
        "record_sha256": f"sha-{record_id}",
        "title": f"title {record_id}",
        "language": "cze",
        "subject_headings": [
            {"pref_label": "subject", "alt_labels": ["alternate"]}
        ],
        "source_assignments": [
            {
                "role": "bibliographic_assignment",
                "source_field": "080",
                "notation_or_uri": notation,
            }
        ],
    }


def test_export_vocabulary_writes_multilingual_csv(tmp_path: Path) -> None:
    output = tmp_path / "udc.csv"
    result = export_vocabulary(_package(), output)

    assert result["concept_count"] == 2
    content = output.read_text(encoding="utf-8")
    assert "label_en,label_ja,notation" in content
    assert "Artificial intelligence,人工知能,004.8" in content


def test_export_corpus_preserves_split_and_maps_composite_ancestors(
    tmp_path: Path,
) -> None:
    records = tmp_path / "records.jsonl"
    rows = [
        _record("record-1", "004.8"),
        _record("record-2", "331"),
        _record("record-3", "004.8:331"),
    ]
    records.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    write_sealed_json(
        manifest,
        {
            "records_path": str(records),
            "package_sha256": sha256_file(records),
        },
        backup=False,
    )

    result = export_corpus(manifest, _package(), tmp_path / "annif")

    assert result["train_documents"] + result["test_documents"] == 3
    assert result["rejected_counts"]["broadened_to_udc_summary_ancestor"] == 1
    all_rows = []
    for name in ("train.jsonl", "test.jsonl"):
        path = tmp_path / "annif" / "corpus" / name
        all_rows.extend(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
    assert all(row["subjects"][0]["uri"].startswith("https://") for row in all_rows)


def test_projects_include_fasttext_and_ensemble(tmp_path: Path) -> None:
    result = write_projects(tmp_path / "projects.cfg")
    content = Path(result["path"]).read_text(encoding="utf-8")

    assert "backend=fasttext" in content
    assert "backend=ensemble" in content
    assert "sources=udc-tfidf-ja,udc-fasttext-ja" in content


def test_notation_match_accepts_parent_child_relationships() -> None:
    assert _notation_matches("796.33", ["796"])
    assert _notation_matches("796", ["796.33"])
    assert not _notation_matches("327", ["796", "796.33"])


def test_mapped_notations_uses_only_explicit_summary_ancestors() -> None:
    package = _package()

    assert _mapped_notations("004.8:331.52", package) == ["004.8", "331"]
    assert _mapped_notations("999.123(437)", package) == []


def test_finalize_czech_checkpoint_freezes_sufficient_prefix(
    tmp_path: Path,
) -> None:
    checkpoint_root = (
        tmp_path
        / "classification"
        / "library-evidence"
        / "sources"
        / "czech-national-bibliography"
        / "2026-07-25"
        / "oai-checkpoint"
    )
    page = checkpoint_root / "pages" / "000001.xml"
    page.parent.mkdir(parents=True)
    record = f"""\
<record xmlns="{MARC_NS}">
  <leader>00000nam a2200000 a 4500</leader>
  <controlfield tag="001">record-1</controlfield>
  <controlfield tag="008">00000000000000000000000000000000000cze</controlfield>
  <datafield tag="080" ind1=" " ind2=" ">
    <subfield code="a">004.8</subfield>
  </datafield>
  <datafield tag="245" ind1=" " ind2=" ">
    <subfield code="a">Artificial intelligence</subfield>
  </datafield>
</record>
"""
    page.write_text(record, encoding="utf-8")
    checkpoint = checkpoint_root / "checkpoint.json"
    write_sealed_json(
        checkpoint,
        {
            "pages": [
                {
                    "path": str(page),
                    "sha256": sha256_file(page),
                    "bytes": page.stat().st_size,
                    "record_count": 1,
                }
            ],
            "complete": False,
        },
        backup=False,
    )

    manifest_path = finalize_czech_checkpoint(tmp_path, minimum_records=1)
    manifest = read_sealed_json(manifest_path)

    assert manifest["record_count"] == 1
    assert manifest["acquisition"]["resumption_complete"] is False
    assert manifest["acquisition"]["sampling_stop"] == (
        "minimum-labelled-pilot-records"
    )
