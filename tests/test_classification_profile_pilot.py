from __future__ import annotations

from pathlib import Path

from chronovisor.lab import classification_profile_pilot
from chronovisor.classification.classification import UDCPackage
from chronovisor.lab.classification_profile_pilot import (
    build_profile_index,
    build_profile_rows,
    notation_matches,
    query_profile_index,
)
from chronovisor.core.runtime_config import EmbeddingConfig


def _package() -> UDCPackage:
    concepts = {
        "urn:udc:0": {
            "uri": "urn:udc:0",
            "notation": "0",
            "label_en": "Science and knowledge",
            "label_ja": "科学と知識",
        },
        "urn:udc:004": {
            "uri": "urn:udc:004",
            "broader_uri": "urn:udc:0",
            "notation": "004",
            "label_en": "Computer science",
            "label_ja": "コンピュータ科学",
        },
        "urn:udc:004.4": {
            "uri": "urn:udc:004.4",
            "broader_uri": "urn:udc:004",
            "notation": "004.4",
            "label_en": "Software",
            "label_ja": "ソフトウェア",
        },
        "urn:udc:796": {
            "uri": "urn:udc:796",
            "notation": "796",
            "label_en": "Sport",
            "label_ja": "スポーツ",
        },
    }
    return UDCPackage(
        release="fixture-v1",
        checksum="sha256:fixture",
        source_url="https://example.invalid/udc",
        license="CC BY-SA 3.0",
        attribution="fixture",
        complete=True,
        concepts=concepts,
    )


def _embed(texts: list[str]) -> list[list[float]]:
    output = []
    for text in texts:
        lowered = text.casefold()
        if "football" in lowered or "sport" in lowered or "サッカー" in text:
            output.append([0.0, 1.0])
        elif "software" in lowered or "ソフトウェア" in text:
            output.append([1.0, 0.0])
        else:
            output.append([0.5, 0.5])
    return output


def test_profiles_use_leaf_and_ancestors_but_not_siblings() -> None:
    profiles = build_profile_rows(_package())
    by_notation = {row["notation"]: row for row in profiles}

    software = by_notation["004.4"]
    assert [row["notation"] for row in software["lineage"]] == [
        "0",
        "004",
        "004.4",
    ]
    assert "Science and knowledge" in software["profile_text"]
    assert "Computer science" in software["profile_text"]
    assert "Software" in software["profile_text"]
    assert "Sport" not in software["profile_text"]
    assert software["profile_sources"] == [
        "official_udc_caption_en",
        "official_udc_caption_ja",
        "official_udc_ancestry",
    ]


def test_notation_match_never_accepts_a_broader_parent() -> None:
    assert notation_matches("005.95", ["005.95/.96"])
    assert notation_matches("005.96.1", ["005.95/.96"])
    assert notation_matches("357", ["355/359"])
    assert notation_matches("796.33", ["796"])
    assert not notation_matches("005", ["005.95/.96"])
    assert not notation_matches("35", ["355/359"])
    assert not notation_matches("79", ["796"])


def test_build_and_query_persistent_dense_profile_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        classification_profile_pilot,
        "load_embedding_config",
        lambda: EmbeddingConfig(model="bge-m3"),
    )

    manifest = build_profile_index(
        tmp_path,
        package=_package(),
        embed_many=_embed,
        batch_size=2,
    )
    candidates = query_profile_index(
        tmp_path,
        {"title": "Football tactics", "summary": "", "excerpt": ""},
        limit=2,
        embed_many=_embed,
    )

    assert manifest["profile_count"] == 4
    assert manifest["external_library_records_used"] == 0
    assert manifest["local_page_label_associations_used"] == 0
    assert manifest["llm_calls"] == 0
    assert candidates[0]["notation"] == "796"
    assert candidates[0]["rank"] == 1
