from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
from chronovisor.recall import (
    classification as classification_runtime,
)
from chronovisor.recall import (
    classification_anchor_worker,
    classification_engine,
    classification_model_worker,
)
from chronovisor.recall.classification import (
    CONSENSUS_RUNTIME_ROLES,
    ClassificationError,
)

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


def _consensus_routes(
    *,
    location: str = "remote",
    models: tuple[str, str, str] = ("model-a", "model-b", "model-c"),
) -> tuple[Any, ...]:
    return tuple(
        classification_model_worker.ollama.RuntimeGenerationRoute(
            role=role,
            provider="ollama" if location == "local" else "remote",
            model=model,
            location=location,
            structured_output=True,
        )
        for role, model in zip(CONSENSUS_RUNTIME_ROLES, models, strict=True)
    )


def _route_contract(routes: tuple[Any, ...]) -> list[dict[str, object]]:
    return [
        {
            "role": route.role,
            "provider": route.provider,
            "model": route.model,
            "location": route.location,
            "model_digest": None,
        }
        for route in routes
    ]


def _worker_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_ctx=16_384,
        read_timeout_ms=5_000,
        primary_keep_alive="1m",
        challenger_keep_alive="1m",
        tie_break_keep_alive="1m",
    )


def _worker_payload(root: Path) -> dict[str, object]:
    return {
        "schema": classification_engine.CONSENSUS_SCHEMA,
        "root": str(root),
        "source_sensitivity": "normal",
        "pages": [
            {
                "uid": "uid-1",
                "source_sha256": "sha256:page",
                "title": "AI",
                "candidates": [{"notation": "004.8"}],
            }
        ],
    }


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


def test_consensus_worker_remote_only_uses_fixed_routes_without_local_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = _consensus_routes()
    resolved: list[tuple[str, ...]] = []
    called_roles: list[str] = []

    def resolve(roles: tuple[str, ...]) -> tuple[Any, ...]:
        resolved.append(roles)
        return routes

    def structured_chat(_messages: object, **kwargs: Any) -> Any:
        called_roles.append(str(kwargs["runtime_role"]))
        assert kwargs["source_data_class"] == "page"
        assert kwargs["source_sensitivity"] == "normal"
        return classification_model_worker.ollama.ChatResponse(
            content=json.dumps(
                {
                    "decisions": [
                        {
                            "uid": "uid-1",
                            "primary_notation": "004.8",
                            "secondary_notations": [],
                            "confidence": 0.9,
                            "rationale": "supported",
                        }
                    ]
                }
            )
        )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("remote-only consensus touched local Ollama controls")

    monkeypatch.setattr(
        classification_model_worker,
        "load_decision_router_config",
        _worker_config,
    )
    monkeypatch.setattr(
        classification_model_worker.ollama,
        "runtime_generation_routes",
        resolve,
    )
    monkeypatch.setattr(
        classification_model_worker.ollama,
        "runtime_structured_chat",
        structured_chat,
    )
    for name in (
        "chat",
        "model_digests",
        "model_resource_lease",
        "plan_model_residency",
        "resident_model_rows",
        "unload_named_model",
    ):
        monkeypatch.setattr(classification_model_worker.ollama, name, forbidden)

    result = classification_model_worker.run(_worker_payload(tmp_path))

    assert resolved == [CONSENSUS_RUNTIME_ROLES]
    assert called_roles == list(CONSENSUS_RUNTIME_ROLES[:2])
    assert result["runtime_routes"] == _route_contract(routes)
    assert result["decisions"][0]["primary_model"] == "model-a"
    assert result["decisions"][0]["challenger_model"] == "model-b"


def test_consensus_routes_fetch_local_digests_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = _consensus_routes(location="local")
    digest_calls: list[list[str]] = []
    monkeypatch.setattr(
        classification_runtime.ollama,
        "runtime_generation_routes",
        lambda roles: routes if roles == CONSENSUS_RUNTIME_ROLES else (),
    )

    def digests(models: list[str]) -> dict[str, str]:
        digest_calls.append(models)
        return {model: f"sha256:{model}" for model in models}

    monkeypatch.setattr(classification_runtime.ollama, "model_digests", digests)

    contract = classification_runtime.resolve_consensus_runtime_routes()

    assert digest_calls == [["model-a", "model-b", "model-c"]]
    assert [route["model_digest"] for route in contract] == [
        "sha256:model-a",
        "sha256:model-b",
        "sha256:model-c",
    ]


@pytest.mark.parametrize(
    "invalid",
    (
        {"adjudication_mode": []},
        {"stage_cache_epoch": ""},
        {"source_sensitivity": []},
        {"model": "payload-override"},
    ),
)
def test_invalid_consensus_scalar_skips_route_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid: dict[str, object],
) -> None:
    monkeypatch.setattr(
        classification_model_worker,
        "resolve_consensus_runtime_routes",
        lambda *_args: pytest.fail("invalid payload resolved runtime routes"),
    )

    with pytest.raises(ClassificationError):
        classification_model_worker.run({**_worker_payload(tmp_path), **invalid})


def test_consensus_worker_rejects_route_drift_before_backend_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = _consensus_routes(location="local")
    stale = _route_contract(routes)
    for row in stale:
        row["model_digest"] = f"sha256:{row['model']}"
    stale[0]["model"] = "stale-model"
    calls = 0

    def backend(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        classification_model_worker,
        "load_decision_router_config",
        _worker_config,
    )
    monkeypatch.setattr(
        classification_runtime.ollama,
        "runtime_generation_routes",
        lambda _roles: routes,
    )
    monkeypatch.setattr(
        classification_model_worker.ollama,
        "runtime_structured_chat",
        backend,
    )
    monkeypatch.setattr(
        classification_model_worker.ollama,
        "model_digests",
        lambda *_args, **_kwargs: pytest.fail("route drift touched local control"),
    )

    with pytest.raises(ClassificationError, match="route contract changed"):
        classification_model_worker.run(
            {**_worker_payload(tmp_path), "runtime_routes": stale}
        )
    assert calls == 0


def test_invalid_consensus_contract_shape_skips_route_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        classification_runtime.ollama,
        "runtime_generation_routes",
        lambda *_args: pytest.fail("invalid contract resolved runtime routes"),
    )

    with pytest.raises(ClassificationError, match="contract is invalid"):
        classification_runtime.resolve_consensus_runtime_routes([{}, {}, {}])


def test_stage_cache_identity_includes_complete_runtime_route(tmp_path: Path) -> None:
    pages = [
        {
            "uid": "uid-1",
            "source_sha256": "sha256:page",
            "candidates": [{"notation": "004.8"}],
        }
    ]
    base = {
        "role": "classification.primary",
        "provider": "ollama",
        "model": "model-a",
        "location": "local",
        "model_digest": "sha256:a",
    }
    contracts = []
    for field, value in (
        ("role", "classification.challenger"),
        ("provider", "remote"),
        ("model", "model-b"),
        ("location", "remote"),
        ("model_digest", "sha256:b"),
    ):
        contracts.append([{**base, field: value}])
    keys = {
        classification_model_worker._stage_cache_path(
            tmp_path,
            pages,
            runtime_routes=routes,
            adjudication_mode="proposal-audit",
            stage_cache_epoch="default",
        )[0]
        for routes in ([base], *contracts)
    }

    assert len(keys) == 6


def test_engine_route_contract_invalidates_legacy_batch_identity() -> None:
    routes = _route_contract(_consensus_routes())
    current = classification_engine._classification_batch_input(
        [
            {
                "uid": "uid-1",
                "source_sha256": "sha256:page",
                "candidates": [{"notation": "004.8"}],
            }
        ],
        package_checksum="sha256:package",
        adjudication_mode="proposal-audit",
        stage_cache_epoch="default",
        runtime_routes=routes,
        source_sensitivity="high",
    )
    legacy = {key: value for key, value in current.items() if key != "runtime_routes"}

    def digest(value: dict[str, object]) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    assert digest(current) != digest(legacy)


def test_empty_consensus_batch_skips_runtime_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        classification_engine,
        "resolve_consensus_runtime_routes",
        lambda: pytest.fail("empty batch resolved runtime routes"),
    )

    assert classification_engine.run_consensus_batches([], root=tmp_path) == []


def test_duplicate_consensus_models_fail_before_local_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routes = _consensus_routes(models=("same", "same", "other"))
    monkeypatch.setattr(
        classification_runtime.ollama,
        "runtime_generation_routes",
        lambda _roles: routes,
    )
    monkeypatch.setattr(
        classification_runtime.ollama,
        "model_digests",
        lambda *_args, **_kwargs: pytest.fail("duplicate models reached local control"),
    )

    with pytest.raises(ClassificationError, match="models must be distinct"):
        classification_runtime.resolve_consensus_runtime_routes()
