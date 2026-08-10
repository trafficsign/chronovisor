from __future__ import annotations

import json

from chronovisor.classification import classification_query_worker
from chronovisor.classification.classification_query_worker import (
    QUERY_PROMPT_SHA256,
    WORKER_SCHEMA,
)


def test_worker_uses_only_candidate_blind_page_fields(monkeypatch) -> None:
    captured = {}
    model = "ornith:test"
    digest = "sha256:model"
    monkeypatch.setattr(
        classification_query_worker.ollama,
        "runtime_generation_routes",
        lambda roles: (
            classification_query_worker.ollama.RuntimeGenerationRoute(
                role=roles[0],
                provider="local",
                model=model,
                location="local",
                structured_output=True,
            ),
        ),
    )
    monkeypatch.setattr(
        classification_query_worker.ollama,
        "model_digests",
        lambda models: {models[0]: digest},
    )

    def fake_chat(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return classification_query_worker.ollama.ChatResponse(
            content=json.dumps(
                {
                    "subject_headings_ja": ["雇用", "職業指導"],
                    "subject_headings_en": ["Employment", "Vocational guidance"],
                    "literal_terms_to_ignore": ["horse", "operator"],
                    "evidence_basis": "The page discusses labor displacement.",
                },
                ensure_ascii=False,
            )
        )

    monkeypatch.setattr(
        classification_query_worker.ollama,
        "runtime_structured_chat",
        fake_chat,
    )
    result = classification_query_worker.run(
        {
            "schema": WORKER_SCHEMA,
            "model": model,
            "model_digest": digest,
            "source_sensitivity": "normal",
            "keep_alive": "1m",
            "read_timeout_ms": 1_000,
            "page": {
                "uid": "page-1",
                "title": "Horse analogy",
                "summary": "A labor displacement argument",
                "excerpt": "The operator is displaced by automation.",
            },
        }
    )

    user_payload = json.loads(captured["messages"][1]["content"])
    assert set(user_payload["page"]) == {"uid", "title", "summary", "excerpt"}
    assert result["prompt_sha256"] == QUERY_PROMPT_SHA256
    assert result["model_calls"] == 1
    assert result["model_digest"] == digest
    assert result["route_identity"] == {
        "role": "classification.query",
        "provider": "local",
        "model": model,
        "location": "local",
    }
    assert result["query"]["subject_headings_en"] == [
        "Employment",
        "Vocational guidance",
    ]
    assert captured["kwargs"]["temperature"] == 0
    assert captured["kwargs"]["seed"] == 0
    assert captured["kwargs"]["think"] is False
    assert captured["kwargs"]["runtime_role"] == "classification.query"
    assert captured["kwargs"]["source_data_class"] == "page"
    assert captured["kwargs"]["source_sensitivity"] == "normal"
