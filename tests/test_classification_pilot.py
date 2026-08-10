from __future__ import annotations

import json
from pathlib import Path

from chronovisor.lab import classification_pilot
from chronovisor.lab.classification_pilot import (
    AuthoritativeCandidateIndex,
    PilotRunner,
    prepare_diagnostic_set,
    score_prediction,
    summarize_cases,
)
from chronovisor.recall.classification import UDCPackage


def _package() -> UDCPackage:
    concepts = {
        "udc:0": {
            "uri": "udc:0",
            "notation": "0",
            "label_en": "Science and knowledge. Organization. Computer science",
            "label_ja": "科学及び知識",
        },
        "udc:004": {
            "uri": "udc:004",
            "notation": "004",
            "label_en": "Computer science and technology. Computing",
            "label_ja": "コンピュータ科学",
            "broader_uri": "udc:0",
        },
        "udc:004.42": {
            "uri": "udc:004.42",
            "notation": "004.42",
            "label_en": "Computer programming. Computer programs",
            "label_ja": "コンピュータプログラミング",
            "broader_uri": "udc:004",
        },
        "udc:5": {
            "uri": "udc:5",
            "notation": "5",
            "label_en": "Mathematics and natural sciences",
            "label_ja": "数学及び自然科学",
        },
        "udc:57": {
            "uri": "udc:57",
            "notation": "57",
            "label_en": "Biological sciences in general",
            "label_ja": "生物科学",
            "broader_uri": "udc:5",
        },
        "udc:575": {
            "uri": "udc:575",
            "notation": "575",
            "label_en": "General genetics. General cytogenetics",
            "label_ja": "遺伝学",
            "broader_uri": "udc:57",
        },
    }
    return UDCPackage(
        release="test",
        checksum="sha256:test",
        source_url="https://example.invalid",
        license="CC BY-SA 3.0",
        attribution="test",
        complete=False,
        concepts=concepts,
    )


def test_prepare_diagnostic_set_joins_frozen_sources(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    spec = tmp_path / "spec.json"
    output = tmp_path / "diagnostic.jsonl"
    fixture.write_text(
        json.dumps(
            {
                "page_id": "genome",
                "uid": "uid-1",
                "title": "Genome",
                "source_sha256": "sha256:page",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline.write_text(
        json.dumps(
            {
                "uid": "uid-1",
                "status": "proposed",
                "primary_notation": "165",
                "quorum": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "page_id": "genome",
                        "bucket": "lexical_failure",
                        "reference": {
                            "expected_disposition": "leaf",
                            "primary_notation": "575",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    receipt = prepare_diagnostic_set(
        fixture_path=fixture,
        baseline_results_path=baseline,
        spec_path=spec,
        output_path=output,
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["case_count"] == 1
    assert row["reference"]["primary_notation"] == "575"
    assert row["baseline"]["primary_notation"] == "165"
    assert output.with_suffix(".receipt.json").exists()


def test_authoritative_query_excludes_local_mined_associations() -> None:
    query = AuthoritativeCandidateIndex.page_query(
        {
            "title": "Genome",
            "summary": "Population origins",
            "excerpt": "Genetic analysis.",
            "tags": ["wrong/local-tag"],
            "raw_keywords": ["wrong-mined-association"],
        }
    )

    assert "Genome" in query
    assert "wrong/local-tag" not in query
    assert "wrong-mined-association" not in query


def test_semantic_candidates_union_official_path_and_legacy_candidates() -> None:
    vectors = {
        "query": [1.0, 0.0],
        "575": [1.0, 0.0],
        "004.42": [0.0, 1.0],
        "other": [0.1, 0.9],
    }

    def fake_embed(texts: list[str]) -> list[list[float]]:
        output = []
        for text in texts:
            if text == "遺伝子解析":
                output.append(vectors["query"])
            elif text.startswith("575 "):
                output.append(vectors["575"])
            elif text.startswith("004.42 "):
                output.append(vectors["004.42"])
            else:
                output.append(vectors["other"])
        return output

    index = AuthoritativeCandidateIndex(_package(), embed_many=fake_embed)
    candidates = index.candidates(
        {
            "title": "遺伝子解析",
            "summary": "",
            "excerpt": "",
            "candidates": [
                {
                    "notation": "004.42",
                    "retrieval_score": 8.0,
                }
            ],
        },
        semantic_limit=1,
        total_limit=8,
    )
    by_notation = {row["notation"]: row for row in candidates}

    assert candidates[0]["notation"] == "575"
    assert by_notation["575"]["retrieval_sources"] == [
        "official_label_semantic"
    ]
    assert "004.42" in by_notation
    assert "legacy_lexical" in by_notation["004.42"]["retrieval_sources"]
    assert index.notation_path("575") == ["575", "57", "5"]


def test_default_embedding_separates_page_query_and_official_documents(
    monkeypatch,
) -> None:
    calls = []

    def embed(texts, **kwargs):
        calls.append((list(texts), kwargs))
        return (
            {
                "role": "classification.embedding",
                "provider": "remote-test",
                "model": "embedding-model",
                "location": "remote",
                "model_digest": None,
            },
            [[1.0, 0.0] for _text in texts],
        )

    monkeypatch.setattr(classification_pilot, "embed_texts_cancellable", embed)
    index = AuthoritativeCandidateIndex(_package())

    index.candidates(
        {"title": "Genome", "candidates": []},
        semantic_limit=1,
        total_limit=1,
    )

    assert calls[0][1] == {
        "source_data_class": "page",
        "source_sensitivity": "high",
        "embedding_purpose": "query",
    }
    assert all(
        kwargs
        == {
            "source_data_class": "derived_snippet",
            "source_sensitivity": "normal",
            "embedding_purpose": "document",
        }
        for _texts, kwargs in calls[1:]
    )


def test_reference_scoring_separates_acceptable_ancestor_and_catastrophe() -> None:
    reference = {
        "expected_disposition": "leaf",
        "primary_notation": "575",
        "acceptable_notations": ["575"],
        "acceptable_ancestor_notations": ["57"],
    }

    exact = score_prediction(
        reference, {"status": "proposed", "primary_notation": "575"}
    )
    ancestor = score_prediction(
        reference, {"status": "provisional", "primary_notation": "57"}
    )
    catastrophe = score_prediction(
        reference, {"status": "proposed", "primary_notation": "165"}
    )

    assert exact["exact"] is True
    assert ancestor["accepted"] is True
    assert ancestor["exact"] is False
    assert catastrophe["catastrophic"] is True


def test_model_stage_retries_transient_empty_json(
    tmp_path: Path, monkeypatch
) -> None:
    responses = iter(["", '{"decision":{"ok":true}}'])
    monkeypatch.setattr(
        "chronovisor.lab.classification_pilot.ollama.chat",
        lambda *args, **kwargs: next(responses),
    )
    runner = PilotRunner(package=_package(), cache_dir=tmp_path)

    result = runner._cached_model_call(
        model="test",
        keep_alive="0",
        prompt={"page": "test"},
        schema={"type": "object"},
        stage="test-stage",
    )

    assert result["attempts"] == 2
    assert result["payload"] == {"decision": {"ok": True}}


def test_gpt_oss_model_stage_uses_low_reasoning(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []

    def fake_chat(*args, **kwargs):
        calls.append(kwargs)
        return '{"decision":{"ok":true}}'

    monkeypatch.setattr(
        "chronovisor.lab.classification_pilot.ollama.chat",
        fake_chat,
    )
    runner = PilotRunner(package=_package(), cache_dir=tmp_path)

    runner._cached_model_call(
        model="gpt-oss:20b",
        keep_alive="0",
        prompt={"page": "test"},
        schema={"type": "object"},
        stage="test-stage",
    )

    assert calls[0]["think"] == "low"


def test_diagnostic_leader_is_not_declared_winner_below_quality_floor() -> None:
    cases = [
        {
            "reference": {
                "expected_disposition": "leaf",
                "primary_notation": "575",
            },
            "candidate_retrieval": {"notations": ["575"]},
            "variants": {
                "safe_but_weak": {
                    "status": "held",
                    "primary_notation": "",
                }
            },
        }
        for _ in range(10)
    ]

    summary = summarize_cases(cases)

    assert summary["diagnostic_leader"] == "safe_but_weak"
    assert summary["pilot_winner"] is None
    assert summary["production_qualified"] is False
    assert summary["candidate_primary_recall"] == 10
