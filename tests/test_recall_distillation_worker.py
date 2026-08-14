from __future__ import annotations

from types import SimpleNamespace

from chronovisor.core import ollama
from chronovisor.decision.local_structured import preflight_structured_request
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
        "num_predict": 2_048,
        "keep_alive": "0",
        "read_timeout_ms": 30_000,
        "max_input_chars": worker.MAX_SESSION_INPUT_BYTES,
        "max_output_chars": worker.MAX_OUTPUT_CHARS,
        "max_feedback_chars": 512,
        "max_responses": 2,
        "require_returned_model": True,
    }
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
