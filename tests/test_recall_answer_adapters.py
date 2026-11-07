from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from chronovisor.core import llm_config, ollama
from chronovisor.core.decision_mlx import DecisionRoute, decision_fingerprint
from chronovisor.core.llm_runtime import (
    BackendCapabilities,
    GenerationResult,
    GenerationRoute,
    LLMRuntime,
    RouteLocation,
)
from chronovisor.recall import recall_answer_adapters


def test_local_effort_binds_sampler_identity_and_actual_runtime_budgets(monkeypatch):
    settings = {"level": "xhigh", "num_ctx": 65536, "num_predict": 16384}
    monkeypatch.setattr(
        recall_answer_adapters, "local_reasoning_settings", lambda *_args: settings
    )
    monkeypatch.setattr(
        recall_answer_adapters,
        "load_decision_router_config",
        lambda: SimpleNamespace(
            num_ctx=131072,
            num_predict=4096,
            read_timeout_ms=660000,
            max_output_chars=8000,
        ),
    )
    captured = {}

    def structured_chat(_messages, **kwargs):
        captured.update(kwargs)
        return ollama.ChatResponse(content='{"answer":"synthetic"}')

    monkeypatch.setattr(ollama, "runtime_structured_chat", structured_chat)
    identity = recall_answer_adapters._sampler_identity(
        recall_answer_adapters.RUNNER_RUNTIME_ROLE
    )
    recall_answer_adapters._chat_json(
        runtime_role=recall_answer_adapters.RUNNER_RUNTIME_ROLE,
        messages=[],
        schema={},
        seed=7,
    )
    assert captured["think"] == "xhigh"
    assert captured["num_ctx"] == 131072 and captured["num_predict"] == 16384
    assert identity == recall_answer_adapters.canonical_sha256(
        {
            "temperature": 0,
            "seed": "pair_seed",
            "think": "xhigh",
            "runtime_budgets": {"num_ctx": 131072, "num_predict": 16384},
        }
    )
    settings["level"] = "low"
    assert (
        recall_answer_adapters._sampler_identity(
            recall_answer_adapters.RUNNER_RUNTIME_ROLE
        )
        != identity
    )
    monkeypatch.setattr(
        recall_answer_adapters, "local_reasoning_settings", lambda *_args: None
    )
    assert (
        recall_answer_adapters._sampler_identity(
            recall_answer_adapters.RUNNER_RUNTIME_ROLE
        )
        == recall_answer_adapters.SAMPLER_SHA256
    )


def _runner_route(*, location: str = "remote") -> ollama.RuntimeGenerationRoute:
    provider = "ollama" if location == "local" else "remote-test"
    return ollama.RuntimeGenerationRoute(
        recall_answer_adapters.RUNNER_RUNTIME_ROLE,
        provider,
        "runner-model",
        location,
        True,
    )


def _decision_route() -> DecisionRoute:
    return DecisionRoute(
        role=recall_answer_adapters.SCORER_RUNTIME_ROLE,
        provider="local-decision",
        model="startlux-model",
        location="local",
        revision="f" * 64,
    )


def _install_identity_routes(
    monkeypatch: pytest.MonkeyPatch, *, runner_location: str = "remote"
) -> tuple[ollama.RuntimeGenerationRoute, DecisionRoute]:
    runner = _runner_route(location=runner_location)
    scorer = _decision_route()

    def generation_routes(roles: object) -> tuple[ollama.RuntimeGenerationRoute, ...]:
        if tuple(roles) != (recall_answer_adapters.RUNNER_RUNTIME_ROLE,):
            pytest.fail("unexpected runtime roles")
        return (runner,)

    monkeypatch.setattr(ollama, "runtime_generation_routes", generation_routes)
    monkeypatch.setattr(recall_answer_adapters, "decision_route", lambda _role: scorer)
    return runner, scorer


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
    _runner, _scorer = _install_identity_routes(monkeypatch)
    captured: dict[str, Any] = {}

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
    assert (
        result["identity"]["identity_schema"] == recall_answer_adapters.IDENTITY_SCHEMA
    )
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


def test_startlux_scorer_uses_the_decision_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runner, scorer = _install_identity_routes(monkeypatch)
    captured: dict[str, Any] = {}

    def decision_judge(
        kind: str,
        fields: object,
        *,
        role: str,
        config: object,
        timeout_ms: int,
        priority: int | None = None,
    ) -> dict[str, Any]:
        captured.update(
            kind=kind,
            fields=fields,
            role=role,
            config=config,
            timeout_ms=timeout_ms,
            priority=priority,
        )
        return {
            "scores": {
                "correctness": 0.9,
                "grounding": 0.8,
                "citation": 0.7,
            }
        }

    monkeypatch.setattr(
        recall_answer_adapters.reranker_client, "decision_judge", decision_judge
    )
    monkeypatch.setattr(
        recall_answer_adapters,
        "load_reranker_config",
        lambda: SimpleNamespace(service=SimpleNamespace(timeout_ms=1000)),
    )
    gold = {
        "evidence": {
            "source_packet": {
                "evidence_chunks": [{"page_id": "page-1", "excerpt": "fact"}]
            }
        },
        "evidence_sha256": "b" * 64,
        "rubric_sha256": "c" * 64,
    }

    result = recall_answer_adapters.builtin_startlux_answer_scorer(
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
    assert captured["kind"] == recall_answer_adapters.JUDGE_ANSWER_AXES
    assert captured["role"] == recall_answer_adapters.SCORER_RUNTIME_ROLE
    assert captured["timeout_ms"] == recall_answer_adapters.SCORER_TIMEOUT_MS
    assert captured["priority"] == 1
    assert captured["fields"]["query"] == "question"
    assert captured["fields"]["answer"] == "answer"
    assert "[PAGE page-1]" in captured["fields"]["reference"]
    assert result["identity"]["scorer_id"] == "builtin-startlux-evidence-scorer-v1"
    assert result["identity"]["route_identity"]["role"] == (
        recall_answer_adapters.SCORER_RUNTIME_ROLE
    )
    assert result["identity"]["route_identity"]["model_digest"] == (
        decision_fingerprint(scorer)
    )
    assert result["identity"]["model_digest"] == decision_fingerprint(scorer)
    assert result["identity"]["policy_sha256"] == (
        recall_answer_adapters.SCORER_POLICY_SHA256
    )
    assert (
        result["reset_receipt"]["reset_protocol_sha256"]
        == (result["identity"]["policy_sha256"])
    )


def test_local_ollama_runner_identity_reads_only_runner_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner_route(location="local")
    scorer = _decision_route()
    calls: list[list[str]] = []

    def generation_routes(roles: object) -> tuple[ollama.RuntimeGenerationRoute, ...]:
        assert tuple(roles) == (recall_answer_adapters.RUNNER_RUNTIME_ROLE,)
        return (runner,)

    monkeypatch.setattr(ollama, "runtime_generation_routes", generation_routes)
    monkeypatch.setattr(recall_answer_adapters, "decision_route", lambda _role: scorer)

    def digests(models: list[str]) -> dict[str, str]:
        calls.append(models)
        return {"runner-model": "digest-runner"}

    monkeypatch.setattr(ollama, "model_digests", digests)

    runner_identity, scorer_identity = (
        recall_answer_adapters.builtin_answer_adapter_identities(
            rubric_sha256="a" * 64,
            evidence_manifest_sha256="b" * 64,
        )
    )

    assert calls == [["runner-model"]]
    assert runner_identity["route_identity"]["model_digest"] == "digest-runner"
    assert scorer_identity["route_identity"]["model_digest"] == (
        decision_fingerprint(scorer)
    )


def test_missing_local_digest_unstructured_runner_and_missing_decision_route_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner_route(location="local")
    scorer = _decision_route()
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: (runner,))
    monkeypatch.setattr(recall_answer_adapters, "decision_route", lambda _role: scorer)
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

    unstructured = ollama.RuntimeGenerationRoute(
        runner.role,
        runner.provider,
        runner.model,
        runner.location,
        False,
    )
    monkeypatch.setattr(
        ollama, "runtime_generation_routes", lambda _roles: (unstructured,)
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

    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: (runner,))
    monkeypatch.setattr(
        ollama, "model_digests", lambda _models: {"runner-model": "a" * 64}
    )
    monkeypatch.setattr(
        recall_answer_adapters,
        "decision_route",
        lambda _role: (_ for _ in ()).throw(RuntimeError("missing")),
    )
    with pytest.raises(RuntimeError):
        recall_answer_adapters.builtin_answer_adapter_identities(
            rubric_sha256="a" * 64,
            evidence_manifest_sha256="b" * 64,
        )


def test_malformed_runtime_answer_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_identity_routes(monkeypatch)
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
        }
    )
    monkeypatch.setattr(llm_config, "load_default_llm_runtime", lambda: runtime)
    monkeypatch.setattr(
        recall_answer_adapters, "decision_route", lambda _role: _decision_route()
    )
    _forbid_ollama_controls(monkeypatch)

    with pytest.raises(ollama.RuntimeBridgeError) as denied:
        recall_answer_adapters.builtin_ollama_answer_runner(
            "question",
            "context",
            {"seed": 1, "base_state_sha256": "a" * 64},
        )

    assert denied.value.category == "egress_denied"
    assert backend_calls == []
