from __future__ import annotations

import json
from types import ModuleType
from typing import Any

import pytest

from chronovisor.classification import (
    classification_anchor_set_worker,
    classification_decision_worker,
    classification_direct_decision_worker,
    classification_hierarchy_worker,
    classification_query_worker,
    classification_query_worker_v2,
)
from chronovisor.core import llm_config
from chronovisor.core.llm_runtime import (
    BackendCapabilities,
    GenerationInput,
    GenerationResult,
    GenerationRoute,
    LLMRuntime,
    RouteLocation,
)
from chronovisor.recall import classification_anchor_worker
from chronovisor.recall.classification import ClassificationError

_CASES = (
    (
        classification_anchor_set_worker,
        "classification.anchor_set",
        {
            "schema": classification_anchor_set_worker.WORKER_SCHEMA,
            "operation": "extract",
            "page": {"title": "T", "summary": "S", "evidence_excerpt": "E"},
        },
        {
            "central_subject": "Software",
            "secondary_subjects": [],
            "rationale": "The page is about software.",
        },
    ),
    (
        classification_decision_worker,
        "classification.decision",
        {
            "schema": classification_decision_worker.WORKER_SCHEMA,
            "page": {"uid": "p1", "title": "T", "summary": "S", "excerpt": "E"},
            "candidates": [
                {"notation": "004.4", "label_en": "Software", "label_ja": "ソフトウェア"}
            ],
        },
        {
            "assessments": [
                {
                    "notation": "004.4",
                    "support": "yes",
                    "evidence": "direct",
                    "reason": "Software is the principal subject.",
                }
            ],
            "principal_class": "0",
            "disposition": "assign",
            "selected_notation": "004.4",
            "specificity_safe": True,
            "rationale": "The candidate is supported.",
        },
    ),
    (
        classification_direct_decision_worker,
        "classification.direct_decision",
        {
            "schema": classification_direct_decision_worker.WORKER_SCHEMA,
            "page": {"uid": "p1", "title": "T", "summary": "S", "excerpt": "E"},
            "subject_headings": ["Software"],
            "candidates": [
                {"notation": "004.4", "label_en": "Software", "label_ja": "ソフトウェア"}
            ],
        },
        {
            "central_subject": "Software",
            "principal_class": "0",
            "disposition": "assign",
            "selected_notation": "004.4",
            "rationale": "The candidate contains the subject.",
        },
    ),
    (
        classification_hierarchy_worker,
        "classification.hierarchy",
        {
            "schema": classification_hierarchy_worker.WORKER_SCHEMA,
            "operation": "extract",
            "page": {"title": "T", "summary": "S", "evidence_excerpt": "E"},
        },
        {
            "central_subject": "Software",
            "secondary_subjects": [],
            "rationale": "The page is about software.",
        },
    ),
    (
        classification_query_worker,
        "classification.query",
        {
            "schema": classification_query_worker.WORKER_SCHEMA,
            "page": {"uid": "p1", "title": "T", "summary": "S", "excerpt": "E"},
        },
        {
            "subject_headings_ja": ["ソフトウェア", "計算機科学"],
            "subject_headings_en": ["Software", "Computer science"],
            "literal_terms_to_ignore": [],
            "evidence_basis": "The page discusses software.",
        },
    ),
    (
        classification_query_worker_v2,
        "classification.query_v2",
        {
            "schema": classification_query_worker_v2.WORKER_SCHEMA,
            "page": {"uid": "p1", "title": "T", "summary": "S", "excerpt": "E"},
        },
        {
            "broad_headings_ja": ["ソフトウェア", "計算機科学"],
            "broad_headings_en": ["Software", "Computer science"],
            "headings": [
                {"role": "principal_shelf", "ja": "ソフトウェア", "en": "Software"},
                {"role": "problem_or_activity", "ja": "設計", "en": "Design"},
                {"role": "context", "ja": "計算機", "en": "Computing"},
            ],
            "surface_terms_to_ignore": [],
            "evidence_basis": "The page discusses software design.",
        },
    ),
    (
        classification_anchor_worker,
        "classification.anchor.primary",
        {
            "schema": classification_anchor_worker.WORKER_SCHEMA,
            "runtime_role": "classification.anchor.primary",
            "operation": "extract",
            "page": {"title": "T", "summary": "S", "evidence_excerpt": "E"},
        },
        {
            "central_subject": "Software",
            "secondary_subjects": [],
            "rationale": "The page is about software.",
        },
    ),
)


@pytest.mark.parametrize(("worker", "role", "payload", "response"), _CASES)
def test_remote_worker_uses_runtime_without_local_ollama_controls(
    monkeypatch: pytest.MonkeyPatch,
    worker: ModuleType,
    role: str,
    payload: dict[str, Any],
    response: dict[str, Any],
) -> None:
    route = worker.ollama.RuntimeGenerationRoute(
        role=role,
        provider="remote",
        model="remote-model",
        location="remote",
        structured_output=True,
    )
    captured: dict[str, Any] = {}

    def resolve(roles: tuple[str, ...]) -> tuple[Any, ...]:
        assert roles == (role,)
        return (route,)

    def structured_chat(messages: object, **kwargs: Any) -> Any:
        captured["messages"] = messages
        captured.update(kwargs)
        return worker.ollama.ChatResponse(
            content=json.dumps(response, ensure_ascii=False)
        )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("remote worker touched local Ollama controls")

    monkeypatch.setattr(worker.ollama, "runtime_generation_routes", resolve)
    monkeypatch.setattr(worker.ollama, "runtime_structured_chat", structured_chat)
    for name in (
        "model_digests",
        "model_resource_lease",
        "resident_model_rows",
        "plan_model_residency",
        "unload_named_model",
    ):
        monkeypatch.setattr(worker.ollama, name, forbidden)

    result = worker.run({**payload, "source_sensitivity": "normal"})

    assert result["model"] == "remote-model"
    assert result["model_digest"] is None
    assert result["route_identity"] == {
        "role": role,
        "provider": "remote",
        "model": "remote-model",
        "location": "remote",
    }
    assert captured["runtime_role"] == role
    assert captured["source_data_class"] == "page"
    assert captured["source_sensitivity"] == "normal"


def test_fixed_worker_rejects_runtime_role_override() -> None:
    with pytest.raises(ClassificationError, match="runtime role is invalid"):
        classification_query_worker.run(
            {
                "schema": classification_query_worker.WORKER_SCHEMA,
                "runtime_role": "classification.anchor.primary",
                "page": {"uid": "p1", "title": "T", "summary": "S", "excerpt": "E"},
            }
        )


def test_worker_rejects_mismatched_resolved_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = classification_query_worker.ollama.RuntimeGenerationRoute(
        role="classification.decision",
        provider="remote",
        model="remote-model",
        location="remote",
        structured_output=True,
    )
    monkeypatch.setattr(
        classification_query_worker.ollama,
        "runtime_generation_routes",
        lambda _roles: (route,),
    )

    with pytest.raises(ClassificationError, match="route identity mismatch"):
        classification_query_worker.run(
            {
                "schema": classification_query_worker.WORKER_SCHEMA,
                "page": {"uid": "p1", "title": "T", "summary": "S", "excerpt": "E"},
            }
        )


def test_remote_worker_defaults_high_and_runtime_blocks_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RemoteBackend:
        provider = "remote"
        location = RouteLocation.REMOTE

        def __init__(self) -> None:
            self.calls = 0

        def generate(
            self,
            request: GenerationInput,
            *,
            model: str,
        ) -> GenerationResult:
            del request
            self.calls += 1
            return GenerationResult(
                content=json.dumps(
                    {
                        "subject_headings_ja": ["ソフトウェア", "計算機科学"],
                        "subject_headings_en": ["Software", "Computer science"],
                        "literal_terms_to_ignore": [],
                        "evidence_basis": "The page discusses software.",
                    },
                    ensure_ascii=False,
                ),
                provider=self.provider,
                model=model,
            )

    backend = RemoteBackend()
    runtime = LLMRuntime(
        generation={
            "classification.query": GenerationRoute(
                backend,
                "remote-model",
                BackendCapabilities(
                    generation=True,
                    embedding=False,
                    structured_output=True,
                ),
            )
        }
    )
    monkeypatch.setattr(llm_config, "load_default_llm_runtime", lambda: runtime)
    payload = {
        "schema": classification_query_worker.WORKER_SCHEMA,
        "page": {"uid": "p1", "title": "T", "summary": "S", "excerpt": "E"},
    }

    with pytest.raises(
        classification_query_worker.ollama.RuntimeBridgeError
    ) as denied:
        classification_query_worker.run(payload)
    assert denied.value.category == "egress_denied"
    assert backend.calls == 0

    allowed = classification_query_worker.run(
        {**payload, "source_sensitivity": "normal"}
    )
    assert allowed["model"] == "remote-model"
    assert backend.calls == 1
