from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

from chronovisor.core import llm_config, ollama
from chronovisor.core.llm_runtime import (
    BackendCapabilities,
    GenerationInput,
    GenerationResult,
    GenerationRoute,
    LLMRuntime,
    RouteLocation,
    SafeBackendError,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)
from chronovisor.core.runtime_config import DecisionRouterConfig
from chronovisor.decision.decision_artifact import DecisionArtifactError
from chronovisor.decision.decision_router import DecisionRouter
from chronovisor.decision.decision_schema_manifest import FRONTIER_DECISION_SCHEMA

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "summary"],
    "properties": {
        "decision": {"type": "string", "enum": ["apply", "defer"]},
        "summary": {"type": "string"},
    },
}
NORMAL_PAGE = SourceDataClassification(
    SourceDataClass.PAGE, SourceSensitivity.NORMAL
)
SYSTEM_HIGH = SourceDataClassification(
    SourceDataClass.SYSTEM, SourceSensitivity.HIGH
)
STRUCTURED = BackendCapabilities(
    generation=True,
    embedding=False,
    structured_output=True,
)


@dataclass
class FakeGeneration:
    provider: str
    location: RouteLocation
    replies: deque[str | Exception]
    calls: list[tuple[str, GenerationInput]] = field(default_factory=list)
    returned_model: str | None = "configured"

    def generate(self, request: GenerationInput, *, model: str) -> GenerationResult:
        self.calls.append((model, request))
        reply = self.replies.popleft()
        if isinstance(reply, Exception):
            raise reply
        return GenerationResult(
            content=reply,
            provider=self.provider,
            model=model,
            metadata={
                "returned_model": (
                    model if self.returned_model == "configured" else self.returned_model
                )
            }
            if self.returned_model is not None
            else {},
        )


def _payload(decision: str) -> str:
    return json.dumps({"decision": decision, "summary": decision})


def _frontier_payload(decision: str) -> str:
    return json.dumps(
        {
            "decision": decision,
            "summary": decision,
            "tests_run": [],
            "commit": None,
            "committed": False,
            "pushed": False,
            "risk": None,
            "notes": None,
        }
    )


def _config() -> DecisionRouterConfig:
    return DecisionRouterConfig(
        num_ctx=16_384,
        num_predict=256,
        read_timeout_ms=5_000,
        max_input_chars=20_000,
        max_output_chars=1_000,
        max_feedback_chars=2_000,
    )


def _runtime(
    routes: Mapping[str, tuple[str, str, RouteLocation, list[str | Exception]]],
) -> tuple[LLMRuntime, dict[str, FakeGeneration]]:
    backends: dict[str, FakeGeneration] = {}
    generation: dict[str, GenerationRoute] = {}
    for role, (provider, model, location, replies) in routes.items():
        backend = FakeGeneration(provider, location, deque(replies))
        backends[role] = backend
        generation[role] = GenerationRoute(
            backend,
            model,
            STRUCTURED,
            "test-protocol",
            "e" * 64,
            f"revision-{role}" if location is RouteLocation.REMOTE else None,
        )
    return LLMRuntime(generation=generation), backends


def _install_runtime(monkeypatch: Any, runtime: LLMRuntime) -> None:
    monkeypatch.setattr(llm_config, "load_default_llm_runtime", lambda: runtime)
    from chronovisor.decision import local_model_eval

    models = [
        route.model
        for route in runtime._generation.values()
        if route.backend.provider == "ollama"
        and route.backend.location is RouteLocation.LOCAL
    ]
    monkeypatch.setattr(
        local_model_eval,
        "fetch_local_model_metadata",
        lambda requested: {
            "engine": {"name": "ollama", "version": "test"},
            "models": {
                model: {
                    "name": model,
                    "digest": f"digest-{model}",
                    "details": {"quantization_level": "Q4_K_M"},
                }
                for model in requested
                if model in models
            },
        },
    )


def _forbid_ollama(monkeypatch: Any) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("remote-only routing touched Ollama control")

    for name in (
        "model_resource_lease",
        "model_resource_lease_mode",
        "plan_model_residency",
        "observe_model_runtime",
        "unload_named_model",
        "model_digests",
        "runtime_generation_location",
    ):
        monkeypatch.setattr(ollama, name, forbidden)


def test_remote_only_uses_exact_routes_without_ollama_and_audits_identity(
    monkeypatch: Any,
) -> None:
    runtime, backends = _runtime(
        {
            "classification.primary": (
                "remote-a",
                "model-a",
                RouteLocation.REMOTE,
                [_payload("apply")],
            ),
            "classification.challenger": (
                "remote-b",
                "model-b",
                RouteLocation.REMOTE,
                [_payload("apply")],
            ),
            "classification.tie_break": (
                "remote-c",
                "model-c",
                RouteLocation.REMOTE,
                [],
            ),
        }
    )
    _install_runtime(monkeypatch, runtime)
    _forbid_ollama(monkeypatch)

    result = DecisionRouter(config=_config()).decide(
        "prompt", SCHEMA, source=NORMAL_PAGE
    )

    assert result.ok is True
    assert [(vote.provider, vote.model) for vote in result.votes] == [
        ("remote-a", "model-a"),
        ("remote-b", "model-b"),
    ]
    assert [
        (vote["provider"], vote["model"])
        for vote in result.audit_record()["votes"]
    ] == [("remote-a", "model-a"), ("remote-b", "model-b")]
    assert [len(backends[role].calls) for role in backends] == [1, 1, 0]
    assert backends["classification.primary"].calls[0][1].source == NORMAL_PAGE
    assert backends["classification.challenger"].calls[0][1].source == NORMAL_PAGE
    route_identity = DecisionRouter(config=_config()).authority_router()
    assert [route["revision"] for route in route_identity["routes"]] == [
        "revision-classification.primary",
        "revision-classification.challenger",
        "revision-classification.tie_break",
    ]
    assert all(route["protocol"] == "test-protocol" for route in route_identity["routes"])
    assert all(route["endpoint_sha256"] == "e" * 64 for route in route_identity["routes"])
    assert all(route["ollama"] is None for route in route_identity["routes"])


def test_remote_missing_revision_holds_before_any_backend_call(
    monkeypatch: Any,
) -> None:
    runtime, backends = _runtime(
        {
            f"classification.{role}": (
                f"remote-{role}",
                f"model-{role}",
                RouteLocation.REMOTE,
                [_payload("apply")],
            )
            for role in ("primary", "challenger", "tie_break")
        }
    )
    runtime._generation["classification.primary"] = replace(
        runtime._generation["classification.primary"], revision=None
    )
    _install_runtime(monkeypatch, runtime)
    _forbid_ollama(monkeypatch)

    result = DecisionRouter(config=_config()).decide(
        "prompt", SCHEMA, source=NORMAL_PAGE
    )

    assert result.ok is False
    assert result.failure_class == "route_configuration_invalid"
    assert result.votes[0].invalid_reason == "remote_route_revision_required"
    assert all(not backend.calls for backend in backends.values())


def test_remote_returned_model_mismatch_is_excluded_when_other_routes_reach_quorum(
    monkeypatch: Any,
) -> None:
    runtime, backends = _runtime(
        {
            f"classification.{role}": (
                f"remote-{role}",
                f"model-{role}",
                RouteLocation.REMOTE,
                [_payload("apply")],
            )
            for role in ("primary", "challenger", "tie_break")
        }
    )
    canary = "CANARY.private/model-metadata-123"
    backends["classification.primary"].returned_model = canary
    _install_runtime(monkeypatch, runtime)
    _forbid_ollama(monkeypatch)

    result = DecisionRouter(config=_config()).decide(
        "prompt", SCHEMA, source=NORMAL_PAGE
    )

    assert result.ok is True
    assert result.votes[0].invalid_reason == "returned_model_mismatch"
    assert result.votes[0].result.returned_model is None
    assert result.votes[0].audit_record()["returned_model"] is None
    assert canary not in repr(result)
    assert [vote.valid for vote in result.votes] == [False, True, True]
    assert [len(backends[role].calls) for role in backends] == [1, 1, 1]


def test_remote_returned_model_failures_hold_without_reroute_when_no_quorum(
    monkeypatch: Any,
) -> None:
    runtime, backends = _runtime(
        {
            f"classification.{role}": (
                f"remote-{role}",
                f"model-{role}",
                RouteLocation.REMOTE,
                [_payload("apply")],
            )
            for role in ("primary", "challenger", "tie_break")
        }
    )
    backends["classification.primary"].returned_model = "different-primary"
    backends["classification.tie_break"].returned_model = "different-tie"
    _install_runtime(monkeypatch, runtime)
    _forbid_ollama(monkeypatch)

    result = DecisionRouter(config=_config()).decide(
        "prompt", SCHEMA, source=NORMAL_PAGE
    )

    assert result.ok is False
    assert [vote.valid for vote in result.votes] == [False, True, False]
    assert [len(backends[role].calls) for role in backends] == [1, 1, 1]


def test_unsafe_returned_model_is_absent_from_result_audit(
    monkeypatch: Any,
) -> None:
    canary = "CANARY returned model\nprivate"
    runtime, backends = _runtime(
        {
            f"classification.{role}": (
                f"remote-{role}",
                f"model-{role}",
                RouteLocation.REMOTE,
                [_payload("apply")],
            )
            for role in ("primary", "challenger", "tie_break")
        }
    )
    backends["classification.primary"].returned_model = canary
    _install_runtime(monkeypatch, runtime)
    _forbid_ollama(monkeypatch)

    result = DecisionRouter(config=_config()).decide(
        "prompt", SCHEMA, source=NORMAL_PAGE
    )
    audit = result.audit_record()

    assert result.ok is True
    assert result.votes[0].invalid_reason == "returned_model_missing"
    assert result.votes[0].result.returned_model is None
    assert canary not in repr(audit)


def test_remote_egress_denial_excludes_votes_without_backend_calls(
    monkeypatch: Any,
) -> None:
    runtime, backends = _runtime(
        {
            f"classification.{role}": (
                f"remote-{role}",
                f"model-{role}",
                RouteLocation.REMOTE,
                [_payload("apply")],
            )
            for role in ("primary", "challenger", "tie_break")
        }
    )
    _install_runtime(monkeypatch, runtime)
    _forbid_ollama(monkeypatch)

    result = DecisionRouter(config=_config()).decide(
        "prompt", SCHEMA, source=SYSTEM_HIGH
    )

    assert result.ok is False
    assert result.quarantine_reason == "primary_and_challenger_invalid"
    assert [vote.invalid_reason for vote in result.votes] == [
        "egress_denied",
        "egress_denied",
    ]
    assert all(not backend.calls for backend in backends.values())


def test_distinct_providers_may_use_same_model_name(monkeypatch: Any) -> None:
    runtime, backends = _runtime(
        {
            "classification.primary": (
                "remote-a",
                "shared",
                RouteLocation.REMOTE,
                [_payload("apply")],
            ),
            "classification.challenger": (
                "remote-b",
                "shared",
                RouteLocation.REMOTE,
                [_payload("apply")],
            ),
            "classification.tie_break": (
                "remote-c",
                "shared",
                RouteLocation.REMOTE,
                [],
            ),
        }
    )
    for index, role in enumerate(runtime._generation):
        runtime._generation[role] = replace(
            runtime._generation[role],
            endpoint_sha256=str(index + 1) * 64,
        )
    _install_runtime(monkeypatch, runtime)

    result = DecisionRouter(config=_config()).decide(
        "prompt", SCHEMA, source=NORMAL_PAGE
    )

    assert result.ok is True
    assert [len(backend.calls) for backend in backends.values()] == [1, 1, 0]


def test_provider_aliases_cannot_duplicate_endpoint_and_model(
    monkeypatch: Any,
) -> None:
    runtime, backends = _runtime(
        {
            f"classification.{role}": (
                f"alias-{role}",
                "shared",
                RouteLocation.REMOTE,
                [_payload("apply")],
            )
            for role in ("primary", "challenger", "tie_break")
        }
    )
    _install_runtime(monkeypatch, runtime)

    result = DecisionRouter(config=_config()).decide(
        "prompt", SCHEMA, source=NORMAL_PAGE
    )

    assert result.quarantine_reason == (
        "router_config_invalid:decision roles must resolve to distinct "
        "provider/model identities"
    )
    assert all(not backend.calls for backend in backends.values())


def test_duplicate_provider_model_identity_fails_before_backend_call(
    monkeypatch: Any,
) -> None:
    runtime, backends = _runtime(
        {
            "classification.primary": (
                "remote",
                "same",
                RouteLocation.REMOTE,
                [_payload("apply")],
            ),
            "classification.challenger": (
                "remote",
                "same",
                RouteLocation.REMOTE,
                [_payload("apply")],
            ),
            "classification.tie_break": (
                "remote",
                "other",
                RouteLocation.REMOTE,
                [_payload("apply")],
            ),
        }
    )
    _install_runtime(monkeypatch, runtime)

    result = DecisionRouter(config=_config()).decide(
        "prompt", SCHEMA, source=NORMAL_PAGE
    )

    assert result.quarantine_reason == (
        "router_config_invalid:decision roles must resolve to distinct "
        "provider/model identities"
    )
    assert all(not backend.calls for backend in backends.values())


def test_missing_structured_capability_fails_before_calls(monkeypatch: Any) -> None:
    runtime, backends = _runtime(
        {
            f"classification.{role}": (
                f"remote-{role}",
                f"model-{role}",
                RouteLocation.REMOTE,
                [_payload("apply")],
            )
            for role in ("primary", "challenger", "tie_break")
        }
    )
    runtime._generation["classification.challenger"] = GenerationRoute(
        backends["classification.challenger"], "model-challenger"
    )
    _install_runtime(monkeypatch, runtime)
    _forbid_ollama(monkeypatch)

    result = DecisionRouter(config=_config()).decide(
        "prompt", SCHEMA, source=NORMAL_PAGE
    )

    assert result.quarantine_reason == "router_config_invalid:capability_unavailable"
    assert all(not backend.calls for backend in backends.values())


def test_failed_role_is_not_retried_or_rerouted(monkeypatch: Any) -> None:
    runtime, backends = _runtime(
        {
            "classification.primary": (
                "remote-a",
                "model-a",
                RouteLocation.REMOTE,
                [SafeBackendError("http_5xx")],
            ),
            "classification.challenger": (
                "remote-b",
                "model-b",
                RouteLocation.REMOTE,
                [_payload("apply")],
            ),
            "classification.tie_break": (
                "remote-c",
                "model-c",
                RouteLocation.REMOTE,
                [_payload("apply")],
            ),
        }
    )
    _install_runtime(monkeypatch, runtime)
    _forbid_ollama(monkeypatch)

    result = DecisionRouter(config=_config()).decide(
        "prompt", SCHEMA, source=NORMAL_PAGE
    )

    assert result.ok is True
    assert result.votes[0].invalid_reason == "http_5xx"
    assert [(vote.provider, vote.model) for vote in result.votes[1:]] == [
        ("remote-b", "model-b"),
        ("remote-c", "model-c"),
    ]
    assert [len(backends[role].calls) for role in backends] == [1, 1, 1]


def test_remote_and_custom_local_routes_share_quorum_and_veto(monkeypatch: Any) -> None:
    config = _config()
    local_replies = {
        config.primary_model: deque([_frontier_payload("approved")]),
        config.challenger_model: deque([_frontier_payload("rejected")]),
        config.tie_break_model: deque([_frontier_payload("approved")]),
    }

    def local_transport(request: Any) -> str:
        return local_replies[request.model].popleft()

    local = DecisionRouter(config=config, transport=local_transport).decide(
        "prompt", FRONTIER_DECISION_SCHEMA, source=NORMAL_PAGE
    )
    runtime, _backends = _runtime(
        {
            "classification.primary": (
                "remote-a",
                "model-a",
                RouteLocation.REMOTE,
                [_frontier_payload("approved")],
            ),
            "classification.challenger": (
                "remote-b",
                "model-b",
                RouteLocation.REMOTE,
                [_frontier_payload("rejected")],
            ),
            "classification.tie_break": (
                "remote-c",
                "model-c",
                RouteLocation.REMOTE,
                [_frontier_payload("approved")],
            ),
        }
    )
    _install_runtime(monkeypatch, runtime)
    remote = DecisionRouter(config=config).decide(
        "prompt", FRONTIER_DECISION_SCHEMA, source=NORMAL_PAGE
    )

    assert (remote.status, remote.quarantine_reason) == (
        local.status,
        local.quarantine_reason,
    )
    assert remote.conservative_veto_fired is local.conservative_veto_fired is True


def test_hybrid_remote_pair_does_not_require_optional_local_tie_capacity(
    monkeypatch: Any,
) -> None:
    runtime, backends = _runtime(
        {
            "classification.primary": (
                "remote-a",
                "model-a",
                RouteLocation.REMOTE,
                [_payload("apply")],
            ),
            "classification.challenger": (
                "remote-b",
                "model-b",
                RouteLocation.REMOTE,
                [_payload("apply")],
            ),
            "classification.tie_break": (
                "ollama",
                "local-tie",
                RouteLocation.LOCAL,
                [_payload("defer")],
            ),
        }
    )
    _install_runtime(monkeypatch, runtime)
    planned: list[tuple[str, ...]] = []
    leases: list[str] = []

    def planner(models: tuple[str, ...], **kwargs: object) -> ollama.ModelResidencyPlan:
        planned.append(models)
        return ollama.ModelResidencyPlan(
            num_ctx=int(kwargs["num_ctx"]),
            max_resident_models=0,
            capacity_bytes=0,
            reserve_bytes=0,
            available_bytes=0,
            total_bytes=0,
            estimated_model_bytes=(("local-tie", 0),),
            role_contexts=(("local-tie", int(kwargs["num_ctx"])),),
            resident_models=(),
            calibrated_models=(),
            source="no_local_capacity",
        )

    @contextmanager
    def lease(**_kwargs: object) -> Iterator[None]:
        leases.append("lease")
        yield

    monkeypatch.setattr(ollama, "model_resource_lease", lease)
    router = DecisionRouter(config=_config(), residency_planner=planner)

    result = router.decide("prompt", SCHEMA, source=NORMAL_PAGE)

    assert result.ok is True
    assert planned == []
    assert leases == []
    assert [len(backends[role].calls) for role in backends] == [1, 1, 0]


def test_hybrid_local_tie_acquires_control_only_after_remote_disagreement(
    monkeypatch: Any,
) -> None:
    runtime, backends = _runtime(
        {
            "classification.primary": (
                "remote-a",
                "model-a",
                RouteLocation.REMOTE,
                [_payload("apply")],
            ),
            "classification.challenger": (
                "remote-b",
                "model-b",
                RouteLocation.REMOTE,
                [_payload("defer")],
            ),
            "classification.tie_break": (
                "ollama",
                "local-tie",
                RouteLocation.LOCAL,
                [_payload("apply")],
            ),
        }
    )
    _install_runtime(monkeypatch, runtime)
    events: list[str] = []

    def planner(models: tuple[str, ...], **kwargs: object) -> ollama.ModelResidencyPlan:
        events.append(f"plan:{','.join(models)}")
        num_ctx = int(kwargs["num_ctx"])
        return ollama.ModelResidencyPlan(
            num_ctx=num_ctx,
            max_resident_models=1,
            capacity_bytes=1_000,
            reserve_bytes=0,
            available_bytes=1_000,
            total_bytes=1_000,
            estimated_model_bytes=(("local-tie", 100),),
            role_contexts=(("local-tie", num_ctx),),
            resident_models=(),
            calibrated_models=("local-tie",),
            source="test",
        )

    @contextmanager
    def lease(**_kwargs: object) -> Iterator[None]:
        events.append("lease")
        yield

    monkeypatch.setattr(ollama, "model_resource_lease", lease)
    monkeypatch.setattr(ollama, "model_resource_lease_mode", lambda: "exclusive")
    router = DecisionRouter(
        config=_config(),
        residency_planner=planner,
        model_observer=lambda _model: (100, 16_384),
        model_unloader=lambda model: events.append(f"unload:{model}") or True,
    )

    result = router.decide("prompt", SCHEMA, source=NORMAL_PAGE)

    assert result.ok is True
    assert events == ["lease", "plan:local-tie", "unload:local-tie"]
    assert [len(backends[role].calls) for role in backends] == [1, 1, 1]


def test_local_routes_keep_lease_observation_and_unload(monkeypatch: Any) -> None:
    runtime, backends = _runtime(
        {
            "classification.primary": (
                "ollama",
                "local-a",
                RouteLocation.LOCAL,
                [_payload("apply")],
            ),
            "classification.challenger": (
                "ollama",
                "local-b",
                RouteLocation.LOCAL,
                [_payload("apply")],
            ),
            "classification.tie_break": (
                "ollama",
                "local-c",
                RouteLocation.LOCAL,
                [],
            ),
        }
    )
    _install_runtime(monkeypatch, runtime)
    events: list[str] = []

    @contextmanager
    def lease(**_kwargs: object) -> Iterator[None]:
        events.append("lease")
        yield

    def planner(models: tuple[str, ...], **kwargs: object) -> ollama.ModelResidencyPlan:
        events.append(f"plan:{','.join(models)}")
        num_ctx = int(kwargs["num_ctx"])
        return ollama.ModelResidencyPlan(
            num_ctx=num_ctx,
            max_resident_models=3,
            capacity_bytes=10_000,
            reserve_bytes=0,
            available_bytes=10_000,
            total_bytes=10_000,
            estimated_model_bytes=tuple((model, 100) for model in models),
            role_contexts=tuple((model, num_ctx) for model in models),
            resident_models=(),
            calibrated_models=models,
            source="test",
        )

    monkeypatch.setattr(ollama, "model_resource_lease", lease)
    monkeypatch.setattr(ollama, "model_resource_lease_mode", lambda: "exclusive")
    router = DecisionRouter(
        config=_config(),
        residency_planner=planner,
        model_observer=lambda model: (100, 16_384),
        model_unloader=lambda model: events.append(f"unload:{model}") or True,
    )

    result = router.decide("prompt", SCHEMA, source=SYSTEM_HIGH)

    assert result.ok is True
    assert events == [
        "lease",
        "plan:local-a,local-b,local-c",
        "unload:local-a",
        "unload:local-b",
    ]
    assert [len(backends[role].calls) for role in backends] == [1, 1, 0]
    routes = router.authority_router()["routes"]
    assert all(route["ollama"]["digest"].startswith("digest-") for route in routes)


def test_non_ollama_local_routes_use_no_ollama_controls(monkeypatch: Any) -> None:
    runtime, backends = _runtime(
        {
            f"classification.{role}": (
                f"local-{role}",
                f"model-{role}",
                RouteLocation.LOCAL,
                [_payload("apply")],
            )
            for role in ("primary", "challenger", "tie_break")
        }
    )
    _install_runtime(monkeypatch, runtime)
    _forbid_ollama(monkeypatch)

    router = DecisionRouter(config=_config())
    result = router.decide("prompt", SCHEMA, source=NORMAL_PAGE)

    assert result.ok is True
    assert [len(backends[role].calls) for role in backends] == [1, 1, 0]
    assert all(route["ollama"] is None for route in router.authority_router()["routes"])


def test_artifact_store_error_text_is_not_exposed(monkeypatch: Any) -> None:
    runtime, _backends = _runtime(
        {
            f"classification.{role}": (
                f"remote-{role}",
                f"model-{role}",
                RouteLocation.REMOTE,
                [_payload("apply")],
            )
            for role in ("primary", "challenger", "tie_break")
        }
    )
    _install_runtime(monkeypatch, runtime)
    canary = "CANARY_PRIVATE_ARTIFACT_EXCEPTION"
    router = DecisionRouter(config=_config(), artifact_replay=True)
    monkeypatch.setattr(
        router,
        "_artifact_identity",
        lambda **_kwargs: ("a" * 64, {}, 16_384),
    )

    class FailingStore:
        def load(self, _fingerprint: str) -> None:
            raise DecisionArtifactError(canary)

    router.decision_artifact_store = FailingStore()  # type: ignore[assignment]

    result = router.decide(
        "prompt",
        SCHEMA,
        source=NORMAL_PAGE,
    )

    assert result.ok is False
    assert result.quarantine_reason == "canonical_decision_artifact_invalid"
    assert canary not in repr(result)
