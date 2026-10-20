from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.core import llm_config, ollama
from chronovisor.core.llm_runtime import GenerationRoute, LLMRuntime, RouteLocation
from chronovisor.decision.local_structured import (
    LocalStructuredSession,
    preflight_structured_request,
)
from chronovisor.recall import recall_distillation_worker as worker


def _route(role: str) -> ollama.RuntimeGenerationRoute:
    return ollama.RuntimeGenerationRoute(
        role=role,
        provider="ollama",
        model="ornith:test",
        location="local",
        structured_output=True,
    )


def _payload(**override: object) -> dict[str, object]:
    return {
        "schema": worker.WORKER_SCHEMA,
        "operation": "teacher",
        "role": "recall.distill.teacher.a",
        "request_id": "rally-123",
        "deadline_ms": 30_000,
        "input": {
            "evidence": "private raw text",
            "candidates": [{"candidate_id": "candidate-1"}],
        },
        **override,
    }


def test_worker_resolves_fixed_local_route_and_hides_input(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        worker.ollama, "runtime_generation_routes", lambda roles: (_route(roles[0]),)
    )
    monkeypatch.setattr(
        worker.ollama, "model_digests", lambda models: {models[0]: "sha256:local"}
    )

    class Session:
        def __init__(self, **kwargs) -> None:
            captured["kwargs"] = kwargs

        def run(self, prompt, schema, *, system):
            captured["prompt"] = prompt
            captured["schema"] = schema
            captured["system"] = system
            return SimpleNamespace(
                ok=True,
                think="low",
                num_predict=3_072,
                effective_num_ctx=32_768,
                value={
                    "labels": [
                        {
                            "candidate_id": "candidate-1",
                            "verdict": "relevant",
                            "confidence": 0.9,
                            "rationale": "match",
                            "minimal_atom_ids": ["atom-1"],
                            "missing_slots": [],
                            "changing_claim": "",
                        }
                    ],
                },
            )

    monkeypatch.setattr(worker, "LocalStructuredSession", Session)
    result = worker.run(_payload())

    assert result == {
        "schema": worker.WORKER_SCHEMA,
        "ok": True,
        "operation": "teacher",
        "role": "recall.distill.teacher.a",
        "request_id": "rally-123",
        "route_identity": {
            "role": "recall.distill.teacher.a",
            "provider": "ollama",
            "model": "ornith:test",
            "location": "local",
        },
        "model_digest": "sha256:local",
        "result": {
            "_teacher_reasoning": {
                "level": "low",
                "output_tokens": 3_072,
                "context_tokens": 32_768,
            },
            "labels": [
                {
                    "candidate_id": "candidate-1",
                    "verdict": "relevant",
                    "confidence": 0.9,
                    "rationale": "match",
                    "minimal_atom_ids": ["atom-1"],
                    "missing_slots": [],
                    "changing_claim": "",
                }
            ],
        },
        "failure_class": "",
    }
    assert captured["kwargs"] == {
        "model": "ornith:test",
        "role": "recall_distillation_worker",
        "runtime_role": "recall.distill.teacher.a",
        "runtime_location": "local",
        "source_data_class": "raw",
        "source_sensitivity": "high",
        "num_ctx": 32_768,
        "num_predict": 3_072,
        "keep_alive": "0",
        "read_timeout_ms": 30_000,
        "max_input_chars": worker.MAX_SESSION_INPUT_BYTES,
        "max_output_chars": worker.MAX_OUTPUT_CHARS,
        "max_feedback_chars": 1_024,
        "max_responses": 2,
        "resource_lease_timeout_ms": 0,
        "require_returned_model": True,
    }
    assert worker.MAX_OUTPUT_CHARS == 8_000
    assert "private raw text" not in str(result)


def test_worker_session_ceiling_includes_fixed_schema_overhead() -> None:
    candidate_ids = tuple(f"{index:02d}" + "x" * 158 for index in range(16))
    preflight = preflight_structured_request(
        "x" * worker.MAX_INPUT_CHARS,
        worker._schema("teacher", candidate_ids=candidate_ids),
        system=worker._system("teacher"),
        max_input_chars=worker.MAX_SESSION_INPUT_BYTES,
    )

    assert preflight.ok


@pytest.mark.parametrize("level,repair_turns", [("xhigh", 1), ("off", 0)])
def test_teacher_provenance_keeps_configured_effort_when_thinking_is_disabled(monkeypatch, level, repair_turns) -> None:
    monkeypatch.setattr(worker, "_resolve_local_route", lambda role: (_route(role), "a" * 64))

    class Session:
        local_reasoning = {"level": level, "num_predict": 16384, "num_ctx": 65536}

        def __init__(self, **_kwargs):
            pass

        def run(self, *_args, **_kwargs):
            return SimpleNamespace(
                ok=True, think=False, repair_turns=repair_turns, num_predict=16384, effective_num_ctx=65536,
                value={"labels": [{
                    "candidate_id": "candidate-1", "verdict": "relevant", "confidence": 0.9,
                    "rationale": "synthetic match", "minimal_atom_ids": ["atom-1"],
                    "missing_slots": [], "changing_claim": "",
                }]},
            )

    monkeypatch.setattr(worker, "LocalStructuredSession", Session)
    result = worker.run(_payload())
    assert result["ok"]
    assert result["result"]["_teacher_reasoning"] == {
        "level": level, "final_attempt_level": False, "repair_turns": repair_turns,
        "output_tokens": 16384, "context_tokens": 65536,
    }


def test_worker_rejects_overrides_and_returns_safe_failure() -> None:
    result = worker.run(_payload(model="remote:override"))

    assert result["ok"] is False
    assert result["failure_class"] == "input_invalid"
    assert result["result"] == {}
    assert "private raw text" not in str(result)


def test_worker_rejects_nonlocal_route_before_model_call(monkeypatch) -> None:
    monkeypatch.setattr(
        worker.ollama,
        "runtime_generation_routes",
        lambda roles: (
            ollama.RuntimeGenerationRoute(
                role=roles[0],
                provider="openai",
                model="remote",
                location="remote",
                structured_output=True,
            ),
        ),
    )
    result = worker.run(_payload())

    assert result["ok"] is False
    assert result["failure_class"] == "route_unavailable"
    assert result["route_identity"] == {}


def test_worker_resolves_revision_pinned_local_omlx_without_ollama(
    monkeypatch,
) -> None:
    role = "recall.distill.teacher.a"
    route = ollama.RuntimeGenerationRoute(
        role=role,
        provider="omlx",
        model="qwen",
        location="local",
        structured_output=True,
        protocol="openai-compatible",
        endpoint_sha256="e" * 64,
        revision="qwen-rev",
    )
    monkeypatch.setattr(
        worker.ollama, "runtime_generation_routes", lambda _roles: (route,)
    )
    monkeypatch.setattr(
        worker.ollama,
        "model_digests",
        lambda _models: (_ for _ in ()).throw(AssertionError("Ollama queried")),
    )

    resolved, fingerprint = worker._resolve_local_route(role)

    assert resolved == route
    assert len(fingerprint) == 64


def test_worker_busy_resource_fails_before_transport_and_preserves_claim_budget(
    tmp_path: Path, monkeypatch
) -> None:
    role = "recall.distill.utility_judge"
    route = ollama.RuntimeGenerationRoute(
        role=role,
        provider="mtplx",
        model="ornith:test",
        location="local",
        structured_output=True,
        protocol="openai-compatible",
        endpoint_sha256="e" * 64,
        revision="ornith-test-v1",
    )
    monkeypatch.setattr(
        worker.ollama, "runtime_generation_routes", lambda _roles: (route,)
    )

    transport_calls: list[object] = []
    lease_calls: list[tuple[str, bool, int | None]] = []

    class Backend:
        provider = "mtplx"
        location = RouteLocation.LOCAL

        def generate(self, request: object, *, model: str) -> object:
            transport_calls.append((request, model))
            raise AssertionError("busy resource must fail before transport")

    class BusyControl:
        @contextmanager
        def resource_lease(
            self, *, exclusive: bool, timeout_ms: int | None = None
        ):
            lease_calls.append((role, exclusive, timeout_ms))
            raise TimeoutError("resource busy")
            yield  # pragma: no cover

    runtime = LLMRuntime(
        generation={role: GenerationRoute(Backend(), route.model)},
        local_controls={role: BusyControl()},
    )
    monkeypatch.setattr(llm_config, "load_default_llm_runtime", lambda: runtime)
    monkeypatch.setattr(
        worker,
        "LocalStructuredSession",
        lambda **kwargs: LocalStructuredSession(
            audit_root=tmp_path / "audit", **kwargs
        ),
    )

    result = worker.run(
        _payload(
            operation="utility",
            role=role,
            input={
                "rally_id": "rally-1",
                "candidate_id": "candidate-1",
                "query": "q",
                "context": "c",
                "answer_a": "a",
                "answer_b": "b",
                "blind_order": "a_first",
            },
        )
    )

    assert result["ok"] is False
    assert result["failure_class"] == "capacity_unavailable"
    assert result["route_identity"] == {
        "role": role,
        "provider": "mtplx",
        "model": "ornith:test",
        "location": "local",
    }
    assert lease_calls == [(role, True, 0)]
    assert transport_calls == []
    assert result["result"] == {}


def test_worker_hides_structured_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        worker.ollama, "runtime_generation_routes", lambda roles: (_route(roles[0]),)
    )
    monkeypatch.setattr(
        worker.ollama, "model_digests", lambda models: {models[0]: "sha256:local"}
    )

    class Session:
        def __init__(self, **kwargs) -> None:
            pass

        def run(self, *args, **kwargs):
            return SimpleNamespace(
                ok=False, value=None, failure_class="transport_timeout"
            )

    monkeypatch.setattr(worker, "LocalStructuredSession", Session)
    result = worker.run(_payload())

    assert result["ok"] is False
    assert result["failure_class"] == "transport_timeout"
    assert result["result"] == {}


def test_evidence_fields_are_operation_owned() -> None:
    teacher = worker._schema("teacher")
    utility = worker._schema("utility")

    assert teacher["required"] == ["labels"]
    assert {"minimal_atom_ids", "missing_slots", "changing_claim"} <= set(
        teacher["properties"]["labels"]["items"]["required"]
    )
    assert {"basis_atom_ids", "blind_order", "blind_choice"} <= set(utility["required"])


def test_teacher_batch_requires_one_label_per_candidate() -> None:
    candidate_ids = ("candidate-a", "candidate-b")

    assert worker._valid_teacher_labels(
        {
            "labels": [
                {"candidate_id": "candidate-a"},
                {"candidate_id": "candidate-b"},
            ]
        },
        candidate_ids,
    )
    assert not worker._valid_teacher_labels(
        {
            "labels": [
                {"candidate_id": "candidate-a"},
                {"candidate_id": "candidate-a"},
            ]
        },
        candidate_ids,
    )
