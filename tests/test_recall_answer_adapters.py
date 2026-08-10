from __future__ import annotations

import json
from typing import Any

import pytest

from chronovisor.core import llm_config, ollama
from chronovisor.core.llm_runtime import (
    BackendCapabilities,
    GenerationResult,
    GenerationRoute,
    LLMRuntime,
    RouteLocation,
)
from chronovisor.recall import recall_answer_adapters


def _routes(*, location: str = "remote") -> tuple[ollama.RuntimeGenerationRoute, ...]:
    provider = "ollama" if location == "local" else "remote-test"
    return (
        ollama.RuntimeGenerationRoute(
            recall_answer_adapters.RUNNER_RUNTIME_ROLE,
            provider,
            "runner-model",
            location,
            True,
        ),
        ollama.RuntimeGenerationRoute(
            recall_answer_adapters.SCORER_RUNTIME_ROLE,
            provider,
            "scorer-model",
            location,
            True,
        ),
    )


def _forbid_ollama_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("provider-neutral answer adapter touched an Ollama control")

    for name in (
        "chat",
        "model_digests",
        "model_resource_lease",
        "plan_model_residency",
        "resident_model_rows",
        "unload_named_model",
        "unload_model",
    ):
        monkeypatch.setattr(ollama, name, forbidden)


def test_remote_runner_uses_fixed_runtime_and_raw_high_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = _routes()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        ollama,
        "runtime_generation_routes",
        lambda roles: routes
        if tuple(roles)
        == (
            recall_answer_adapters.RUNNER_RUNTIME_ROLE,
            recall_answer_adapters.SCORER_RUNTIME_ROLE,
        )
        else pytest.fail("unexpected runtime roles"),
    )

    def structured_chat(_messages: object, **kwargs: Any) -> ollama.ChatResponse:
        captured.update(kwargs)
        return ollama.ChatResponse(content=json.dumps({"answer": "grounded answer"}))

    monkeypatch.setattr(ollama, "runtime_structured_chat", structured_chat)
    _forbid_ollama_controls(monkeypatch)

    result = recall_answer_adapters.builtin_ollama_answer_runner(
        "question",
        "context",
        {"seed": 7, "base_state_sha256": "a" * 64},
    )

    assert result["answer"] == "grounded answer"
    assert result["identity"]["identity_schema"] == recall_answer_adapters.IDENTITY_SCHEMA
    assert result["identity"]["route_identity"] == {
        "role": recall_answer_adapters.RUNNER_RUNTIME_ROLE,
        "provider": "remote-test",
        "model": "runner-model",
        "location": "remote",
        "model_digest": None,
    }
    assert captured["runtime_role"] == recall_answer_adapters.RUNNER_RUNTIME_ROLE
    assert captured["source_data_class"] == "raw"
    assert captured["source_sensitivity"] == "high"


def test_remote_scorer_uses_its_fixed_runtime_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = _routes()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: routes)

    def structured_chat(_messages: object, **kwargs: Any) -> ollama.ChatResponse:
        captured.update(kwargs)
        return ollama.ChatResponse(
            content=json.dumps(
                {"correctness": 0.9, "grounding": 0.8, "citation": 0.7}
            )
        )

    monkeypatch.setattr(ollama, "runtime_structured_chat", structured_chat)
    _forbid_ollama_controls(monkeypatch)
    gold = {
        "evidence": {
            "source_packet": {
                "evidence_chunks": [{"page_id": "page-1", "excerpt": "fact"}]
            }
        },
        "evidence_sha256": "b" * 64,
        "rubric_sha256": "c" * 64,
    }

    result = recall_answer_adapters.builtin_ollama_answer_scorer(
        "question",
        "answer",
        gold,
        {
            "seed": 9,
            "base_state_sha256": "d" * 64,
            "evidence_manifest_sha256": "e" * 64,
        },
    )

    assert result["dimensions"] == {
        "correctness": 0.9,
        "grounding": 0.8,
        "citation": 0.7,
    }
    assert result["identity"]["route_identity"]["role"] == (
        recall_answer_adapters.SCORER_RUNTIME_ROLE
    )
    assert captured["runtime_role"] == recall_answer_adapters.SCORER_RUNTIME_ROLE
    assert captured["source_data_class"] == "raw"
    assert captured["source_sensitivity"] == "high"


def test_local_ollama_identity_reads_real_digests_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = _routes(location="local")
    calls: list[list[str]] = []
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: routes)

    def digests(models: list[str]) -> dict[str, str]:
        calls.append(models)
        return {"runner-model": "digest-runner", "scorer-model": "digest-scorer"}

    monkeypatch.setattr(ollama, "model_digests", digests)

    runner, scorer = recall_answer_adapters.builtin_answer_adapter_identities(
        rubric_sha256="a" * 64,
        evidence_manifest_sha256="b" * 64,
    )

    assert calls == [["runner-model", "scorer-model"]]
    assert runner["route_identity"]["model_digest"] == "digest-runner"
    assert scorer["route_identity"]["model_digest"] == "digest-scorer"


def test_missing_local_digest_and_unstructured_route_fail_before_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = _routes(location="local")
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: routes)
    monkeypatch.setattr(ollama, "model_digests", lambda _models: {})
    monkeypatch.setattr(
        ollama,
        "runtime_structured_chat",
        lambda *_args, **_kwargs: pytest.fail("backend must not run"),
    )

    with pytest.raises(RuntimeError):
        recall_answer_adapters.builtin_answer_adapter_identities(
            rubric_sha256="a" * 64,
            evidence_manifest_sha256="b" * 64,
        )

    unstructured = (
        routes[0],
        ollama.RuntimeGenerationRoute(
            routes[1].role,
            routes[1].provider,
            routes[1].model,
            routes[1].location,
            False,
        ),
    )
    monkeypatch.setattr(
        ollama, "runtime_generation_routes", lambda _roles: unstructured
    )
    monkeypatch.setattr(
        ollama,
        "model_digests",
        lambda _models: pytest.fail("capability failure queried digests"),
    )
    with pytest.raises(RuntimeError):
        recall_answer_adapters.builtin_answer_adapter_identities(
            rubric_sha256="a" * 64,
            evidence_manifest_sha256="b" * 64,
        )


def test_malformed_runtime_answer_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: _routes())
    monkeypatch.setattr(
        ollama,
        "runtime_structured_chat",
        lambda *_args, **_kwargs: ollama.ChatResponse(content="{"),
    )
    monkeypatch.setattr(
        ollama,
        "model_digests",
        lambda _models: pytest.fail("remote route queried digests"),
    )

    with pytest.raises(json.JSONDecodeError):
        recall_answer_adapters.builtin_ollama_answer_runner(
            "question",
            "context",
            {"seed": 1, "base_state_sha256": "a" * 64},
        )


def test_remote_raw_high_egress_denial_runs_no_backend_or_ollama_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_calls: list[object] = []

    class Backend:
        provider = "remote-test"
        location = RouteLocation.REMOTE

        def generate(self, request: object, *, model: str) -> GenerationResult:
            backend_calls.append((request, model))
            return GenerationResult(
                content='{"answer":"unexpected"}',
                provider=self.provider,
                model=model,
            )

    backend = Backend()
    runtime = LLMRuntime(
        generation={
            recall_answer_adapters.RUNNER_RUNTIME_ROLE: GenerationRoute(
                backend,
                "runner-model",
                BackendCapabilities(True, False, structured_output=True),
            ),
            recall_answer_adapters.SCORER_RUNTIME_ROLE: GenerationRoute(
                backend,
                "scorer-model",
                BackendCapabilities(True, False, structured_output=True),
            ),
        }
    )
    monkeypatch.setattr(llm_config, "load_default_llm_runtime", lambda: runtime)
    _forbid_ollama_controls(monkeypatch)

    with pytest.raises(ollama.RuntimeBridgeError) as denied:
        recall_answer_adapters.builtin_ollama_answer_runner(
            "question",
            "context",
            {"seed": 1, "base_state_sha256": "a" * 64},
        )

    assert denied.value.category == "egress_denied"
    assert backend_calls == []
