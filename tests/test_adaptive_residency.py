from __future__ import annotations

import json
from collections import defaultdict, deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

from chronovisor.core import ollama
from chronovisor.core.runtime_config import DecisionRouterConfig, IngestConfig
from chronovisor.decision.decision_router import (
    DecisionRouter,
    decision_context_buckets,
)
from chronovisor.decision.local_structured import (
    MAX_REPAIR_TURNS,
    ChatRequest,
    required_structured_context_tokens,
)

PRIMARY = "primary:test"
CHALLENGER = "challenger:test"
TIE_BREAK = "tie-break:test"
MODELS = (PRIMARY, CHALLENGER, TIE_BREAK)


def _ingest_route(
    model: str = PRIMARY,
    *,
    provider: str = "ollama",
    location: str = "local",
) -> tuple:
    return (
        ollama.RuntimeGenerationRoute(
            role=ollama.INGEST_GENERATION_RUNTIME_ROLE,
            provider=provider,
            model=model,
            location=location,
            structured_output=True,
        ),
    )

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "target", "confidence", "summary", "reason", "notes"],
    "properties": {
        "decision": {"type": "string", "enum": ["apply", "defer", "reject"]},
        "target": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
        "reason": {"type": "string"},
        "notes": {"type": ["string", "null"]},
    },
}


def _payload(decision: str) -> str:
    return json.dumps(
        {
            "decision": decision,
            "target": "page-a",
            "confidence": 0.9,
            "summary": "summary",
            "reason": "reason",
            "notes": None,
        }
    )


def _config(**overrides: Any) -> DecisionRouterConfig:
    values: dict[str, Any] = {
        "primary_model": PRIMARY,
        "challenger_model": CHALLENGER,
        "tie_break_model": TIE_BREAK,
        "primary_keep_alive": "20m",
        "challenger_keep_alive": "20m",
        "tie_break_keep_alive": "2m",
        "num_ctx": 131_072,
        "min_num_ctx": 16_384,
        "num_predict": 256,
        "read_timeout_ms": 5_000,
        "max_input_chars": 160_000,
        "max_output_chars": 1_000,
        "max_feedback_chars": 2_000,
        "quorum": 2,
        "adaptive_residency": True,
        "memory_reserve_gib": 16,
        "max_resident_models": 3,
    }
    values.update(overrides)
    return DecisionRouterConfig(**values)


def _plan(
    max_resident_models: int,
    *,
    num_ctx: int = 16_384,
    capacity_bytes: int = 64 * ollama.GIB,
    estimates: tuple[int, int, int] = (8, 12, 10),
    resident_models: tuple[str, ...] = (),
    calibrated_models: tuple[str, ...] = MODELS,
    initial_eviction_models: tuple[str, ...] = (),
    pressure_forced_single: bool = False,
    compressed_bytes: int = 0,
    swap_used_bytes: int = 0,
) -> ollama.ModelResidencyPlan:
    estimated_bytes = tuple(value * ollama.GIB for value in estimates)
    return ollama.ModelResidencyPlan(
        num_ctx=num_ctx,
        max_resident_models=max_resident_models,
        capacity_bytes=capacity_bytes,
        reserve_bytes=16 * ollama.GIB,
        available_bytes=80 * ollama.GIB,
        total_bytes=128 * ollama.GIB,
        estimated_model_bytes=tuple(zip(MODELS, estimated_bytes, strict=True)),
        role_contexts=tuple((model, num_ctx) for model in MODELS),
        resident_models=resident_models,
        calibrated_models=calibrated_models,
        source="test",
        initial_eviction_models=initial_eviction_models,
        pressure_forced_single=pressure_forced_single,
        compressed_bytes=compressed_bytes,
        swap_used_bytes=swap_used_bytes,
    )


class EventTransport:
    def __init__(
        self,
        responses: dict[str, list[str | Exception]],
        events: list[tuple[Any, ...]],
    ) -> None:
        self.responses: dict[str, deque[str | Exception]] = defaultdict(deque)
        for model, queued in responses.items():
            self.responses[model].extend(queued)
        self.events = events
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest) -> str:
        self.requests.append(request)
        self.events.append(("chat", request.model, len(request.messages)))
        response = self.responses[request.model].popleft()
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.parametrize(
    ("capacity_gib", "expected_max"),
    [(12, 1), (24, 1), (27, 2), (40, 3)],
)
def test_build_model_residency_plan_admits_only_calibrated_prefix_that_fits(
    capacity_gib: int,
    expected_max: int,
) -> None:
    reserve = 16 * ollama.GIB
    plan = ollama.build_model_residency_plan(
        MODELS,
        num_ctx=32_768,
        max_num_ctx=131_072,
        memory=ollama.MemorySnapshot(
            total_bytes=128 * ollama.GIB,
            available_bytes=reserve + capacity_gib * ollama.GIB,
            source="test",
        ),
        installed_sizes={model: 10 * ollama.GIB for model in MODELS},
        resident={},
        calibrated_sizes={(model, 32_768): 12 * ollama.GIB for model in MODELS},
        reserve_bytes=reserve,
        configured_max_resident=3,
    )

    assert plan.capacity_bytes == capacity_gib * ollama.GIB
    assert plan.estimated_model_bytes == tuple(
        (model, 12 * ollama.GIB) for model in MODELS
    )
    assert plan.max_resident_models == expected_max
    assert plan.forced_single is False


def test_build_model_residency_plan_falls_back_to_one_without_calibration() -> None:
    plan = ollama.build_model_residency_plan(
        MODELS,
        num_ctx=32_768,
        max_num_ctx=131_072,
        memory=ollama.MemorySnapshot(
            total_bytes=128 * ollama.GIB,
            available_bytes=120 * ollama.GIB,
            source="test",
        ),
        installed_sizes={model: 10 * ollama.GIB for model in MODELS},
        resident={},
        calibrated_sizes={},
        reserve_bytes=16 * ollama.GIB,
        configured_max_resident=3,
    )

    assert all(
        estimate <= plan.capacity_bytes
        for _model, estimate in plan.estimated_model_bytes
    )
    assert plan.calibrated_models == ()
    assert plan.max_resident_models == 1
    assert plan.forced_single is False


def test_exact_small_context_calibration_is_not_inflated_by_larger_bucket() -> None:
    reserve = 16 * ollama.GIB
    plan = ollama.build_model_residency_plan(
        MODELS,
        num_ctx=16_384,
        max_num_ctx=114_688,
        memory=ollama.MemorySnapshot(
            total_bytes=128 * ollama.GIB,
            available_bytes=reserve + 33 * ollama.GIB,
            source="test",
        ),
        installed_sizes={model: 10 * ollama.GIB for model in MODELS},
        resident={},
        calibrated_sizes={
            **{(model, 16_384): 10 * ollama.GIB for model in MODELS},
            **{(model, 114_688): 20 * ollama.GIB for model in MODELS},
        },
        reserve_bytes=reserve,
        configured_max_resident=3,
    )

    assert plan.estimated_model_bytes == tuple(
        (model, 10 * ollama.GIB) for model in MODELS
    )
    assert plan.max_resident_models == 3


def test_production_residency_plan_reuses_a_larger_context_bucket() -> None:
    plan = ollama.build_model_residency_plan(
        MODELS,
        num_ctx=16_384,
        max_num_ctx=131_072,
        memory=ollama.MemorySnapshot(
            total_bytes=128 * ollama.GIB,
            available_bytes=100 * ollama.GIB,
            source="test",
        ),
        installed_sizes={model: 10 * ollama.GIB for model in MODELS},
        resident={PRIMARY: (20 * ollama.GIB, 65_536)},
        calibrated_sizes={(model, 16_384): 12 * ollama.GIB for model in MODELS},
        reserve_bytes=16 * ollama.GIB,
        configured_max_resident=3,
    )

    assert dict(plan.role_contexts) == {
        PRIMARY: 65_536,
        CHALLENGER: 16_384,
        TIE_BREAK: 16_384,
    }
    assert plan.estimate(PRIMARY) == 20 * ollama.GIB
    assert plan.reuse_larger_context is True


def test_production_residency_plan_limits_larger_reuse_per_model() -> None:
    plan = ollama.build_model_residency_plan(
        MODELS,
        num_ctx=98_304,
        max_num_ctx=262_144,
        memory=ollama.MemorySnapshot(
            total_bytes=128 * ollama.GIB,
            available_bytes=80 * ollama.GIB,
            source="test",
        ),
        installed_sizes={model: 10 * ollama.GIB for model in MODELS},
        resident={
            PRIMARY: (20 * ollama.GIB, 131_072),
            CHALLENGER: (24 * ollama.GIB, 262_144),
            TIE_BREAK: (18 * ollama.GIB, 131_072),
        },
        calibrated_sizes={
            (CHALLENGER, 98_304): 14 * ollama.GIB,
        },
        reserve_bytes=16 * ollama.GIB,
        configured_max_resident=3,
        reuse_context_ceilings={
            PRIMARY: 262_144,
            CHALLENGER: 131_072,
            TIE_BREAK: 131_072,
        },
    )

    assert dict(plan.role_contexts) == {
        PRIMARY: 131_072,
        CHALLENGER: 98_304,
        TIE_BREAK: 131_072,
    }
    assert plan.initial_eviction_models == (CHALLENGER,)


def test_per_model_reuse_ceilings_fail_closed_and_clamp_to_global_max() -> None:
    plan = ollama.build_model_residency_plan(
        MODELS,
        num_ctx=98_304,
        max_num_ctx=131_072,
        memory=ollama.MemorySnapshot(
            total_bytes=128 * ollama.GIB,
            available_bytes=80 * ollama.GIB,
            source="test",
        ),
        installed_sizes={model: 10 * ollama.GIB for model in MODELS},
        resident={model: (20 * ollama.GIB, 131_072) for model in MODELS},
        calibrated_sizes={},
        reserve_bytes=16 * ollama.GIB,
        configured_max_resident=3,
        reuse_context_ceilings={
            PRIMARY: 999_999,
            CHALLENGER: "invalid",  # type: ignore[dict-item]
            # TIE_BREAK is deliberately omitted.
        },
    )

    assert dict(plan.role_contexts) == {
        PRIMARY: 131_072,
        CHALLENGER: 98_304,
        TIE_BREAK: 98_304,
    }
    assert plan.initial_eviction_models == (CHALLENGER, TIE_BREAK)


def test_exact_bucket_evaluation_can_disable_larger_context_reuse() -> None:
    plan = ollama.build_model_residency_plan(
        MODELS,
        num_ctx=16_384,
        max_num_ctx=131_072,
        memory=ollama.MemorySnapshot(
            total_bytes=128 * ollama.GIB,
            available_bytes=100 * ollama.GIB,
            source="test",
        ),
        installed_sizes={model: 10 * ollama.GIB for model in MODELS},
        resident={PRIMARY: (20 * ollama.GIB, 65_536)},
        calibrated_sizes={(model, 16_384): 12 * ollama.GIB for model in MODELS},
        reserve_bytes=16 * ollama.GIB,
        configured_max_resident=3,
        reuse_larger_context=False,
    )

    assert dict(plan.role_contexts) == {model: 16_384 for model in MODELS}
    assert plan.reuse_larger_context is False
    assert plan.initial_eviction_models == (PRIMARY,)


def test_exact_bucket_reuses_calibrated_backend_context_floor_without_flapping() -> (
    None
):
    plan = ollama.build_model_residency_plan(
        MODELS,
        num_ctx=16_384,
        max_num_ctx=114_688,
        memory=ollama.MemorySnapshot(
            total_bytes=128 * ollama.GIB,
            available_bytes=70 * ollama.GIB,
            source="test",
        ),
        installed_sizes={model: 10 * ollama.GIB for model in MODELS},
        resident={PRIMARY: (25_300_000_000, 32_768)},
        calibrated_sizes={
            (PRIMARY, 16_384): 25_140_000_000,
            (CHALLENGER, 16_384): 12 * ollama.GIB,
            (TIE_BREAK, 16_384): 15 * ollama.GIB,
        },
        reserve_bytes=16 * ollama.GIB,
        configured_max_resident=3,
        reuse_larger_context=False,
    )

    assert dict(plan.role_contexts)[PRIMARY] == 16_384
    assert plan.estimate(PRIMARY) == 25_300_000_000
    assert plan.initial_eviction_models == ()
    assert plan.context_floor_models == (PRIMARY,)


def test_context_floor_outside_footprint_tolerance_still_requires_eviction() -> None:
    plan = ollama.build_model_residency_plan(
        MODELS,
        num_ctx=16_384,
        max_num_ctx=114_688,
        memory=ollama.MemorySnapshot(
            total_bytes=128 * ollama.GIB,
            available_bytes=50 * ollama.GIB,
            source="test",
        ),
        installed_sizes={model: 10 * ollama.GIB for model in MODELS},
        resident={PRIMARY: (20 * ollama.GIB, 32_768)},
        calibrated_sizes={
            (PRIMARY, 16_384): 15 * ollama.GIB,
            (CHALLENGER, 16_384): 12 * ollama.GIB,
            (TIE_BREAK, 16_384): 15 * ollama.GIB,
        },
        reserve_bytes=16 * ollama.GIB,
        configured_max_resident=3,
        reuse_larger_context=False,
    )

    assert plan.context_floor_models == ()
    assert plan.initial_eviction_models == (PRIMARY,)
    assert plan.estimate(PRIMARY) == 15 * ollama.GIB


def test_incompatible_oversized_resident_is_required_to_evict_before_admission() -> (
    None
):
    plan = ollama.build_model_residency_plan(
        MODELS,
        num_ctx=16_384,
        max_num_ctx=114_688,
        memory=ollama.MemorySnapshot(
            total_bytes=128 * ollama.GIB,
            available_bytes=20 * ollama.GIB,
            source="test",
        ),
        installed_sizes={model: 10 * ollama.GIB for model in MODELS},
        resident={TIE_BREAK: (60 * ollama.GIB, 262_144)},
        calibrated_sizes={
            (PRIMARY, 16_384): 20 * ollama.GIB,
            (CHALLENGER, 16_384): 15 * ollama.GIB,
            (TIE_BREAK, 16_384): 15 * ollama.GIB,
        },
        reserve_bytes=16 * ollama.GIB,
        configured_max_resident=3,
        reuse_larger_context=False,
    )

    assert plan.capacity_bytes == 64 * ollama.GIB
    assert plan.max_resident_models == 3
    assert plan.estimate(TIE_BREAK) == 15 * ollama.GIB
    assert plan.initial_eviction_models == (TIE_BREAK,)


def test_router_wires_explicit_production_and_evaluation_residency_modes() -> None:
    observed_modes: list[bool] = []

    def planner(_models: tuple[str, ...], **kwargs: Any) -> ollama.ModelResidencyPlan:
        observed_modes.append(kwargs["reuse_larger_context"])
        return _plan(1)

    production = DecisionRouter(
        config=_config(),
        transport=EventTransport({}, []),
        resolve_adoption=False,
        record_replay=False,
        residency_planner=planner,
        live_resource_control=True,
        reuse_larger_context=True,
    )
    evaluation = DecisionRouter(
        config=_config(),
        transport=EventTransport({}, []),
        audit_role="model_eval",
        resolve_adoption=False,
        record_replay=False,
        residency_planner=planner,
        live_resource_control=True,
        reuse_larger_context=False,
    )

    production._residency_plan(16_384)
    evaluation._residency_plan(16_384)

    assert observed_modes == [True, False]


def test_router_production_reuses_an_ingest_sized_resident_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[int, dict[str, int]]] = []

    monkeypatch.setattr(
        "chronovisor.decision.decision_router.load_ingest_config",
        lambda: IngestConfig(max_num_ctx=262_144),
    )
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: _ingest_route())

    def planner(_models: tuple[str, ...], **kwargs: Any) -> ollama.ModelResidencyPlan:
        observed.append((kwargs["max_num_ctx"], dict(kwargs["reuse_context_ceilings"])))
        return _plan(1)

    production = DecisionRouter(
        config=_config(),
        transport=EventTransport({}, []),
        resolve_adoption=False,
        record_replay=False,
        residency_planner=planner,
        live_resource_control=True,
        reuse_larger_context=True,
    )
    evaluation = DecisionRouter(
        config=_config(),
        transport=EventTransport({}, []),
        audit_role="model_eval",
        resolve_adoption=False,
        record_replay=False,
        residency_planner=planner,
        live_resource_control=True,
        reuse_larger_context=False,
    )

    production._residency_plan(98_304)
    evaluation._residency_plan(98_304)

    assert observed == [
        (
            262_144,
            {PRIMARY: 262_144, CHALLENGER: 131_072, TIE_BREAK: 131_072},
        ),
        (
            131_072,
            {PRIMARY: 131_072, CHALLENGER: 131_072, TIE_BREAK: 131_072},
        ),
    ]


def test_router_does_not_share_ingest_ceiling_with_a_different_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, int] = {}
    monkeypatch.setattr(
        "chronovisor.decision.decision_router.load_ingest_config",
        lambda: IngestConfig(max_num_ctx=262_144),
    )
    monkeypatch.setattr(
        ollama,
        "runtime_generation_routes",
        lambda _roles: _ingest_route("other:test"),
    )

    def planner(_models: tuple[str, ...], **kwargs: Any) -> ollama.ModelResidencyPlan:
        observed.update(kwargs["reuse_context_ceilings"])
        return _plan(1)

    router = DecisionRouter(
        config=_config(),
        transport=EventTransport({}, []),
        resolve_adoption=False,
        record_replay=False,
        residency_planner=planner,
        live_resource_control=True,
    )

    router._residency_plan(98_304)

    assert observed == {model: 131_072 for model in MODELS}


def test_router_does_not_share_ingest_ceiling_with_non_ollama_local_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, int] = {}
    monkeypatch.setattr(
        "chronovisor.decision.decision_router.load_ingest_config",
        lambda: IngestConfig(max_num_ctx=262_144),
    )
    monkeypatch.setattr(
        ollama,
        "runtime_generation_routes",
        lambda _roles: _ingest_route(provider="local-test"),
    )

    def planner(_models: tuple[str, ...], **kwargs: Any) -> ollama.ModelResidencyPlan:
        observed.update(kwargs["reuse_context_ceilings"])
        return _plan(1)

    router = DecisionRouter(
        config=_config(),
        transport=EventTransport({}, []),
        resolve_adoption=False,
        record_replay=False,
        residency_planner=planner,
        live_resource_control=True,
    )

    router._residency_plan(98_304)

    assert observed == {model: 131_072 for model in MODELS}


def test_router_duplicate_model_keeps_the_largest_role_reuse_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, int] = {}
    monkeypatch.setattr(
        "chronovisor.decision.decision_router.load_ingest_config",
        lambda: IngestConfig(max_num_ctx=262_144),
    )
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: _ingest_route())

    def planner(_models: tuple[str, ...], **kwargs: Any) -> ollama.ModelResidencyPlan:
        observed.update(kwargs["reuse_context_ceilings"])
        return _plan(1)

    router = DecisionRouter(
        config=_config(challenger_model=PRIMARY),
        transport=EventTransport({}, []),
        resolve_adoption=False,
        record_replay=False,
        residency_planner=planner,
        live_resource_control=True,
    )

    router._residency_plan(98_304)

    assert observed == {PRIMARY: 262_144, TIE_BREAK: 131_072}


def test_router_preserves_legacy_exact_signature_residency_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[int, int, bool]] = []
    monkeypatch.setattr(
        "chronovisor.decision.decision_router.load_ingest_config",
        lambda: IngestConfig(max_num_ctx=262_144),
    )
    monkeypatch.setattr(ollama, "runtime_generation_routes", lambda _roles: _ingest_route())

    def planner(
        _models: tuple[str, ...],
        *,
        num_ctx: int,
        max_num_ctx: int,
        reserve_bytes: int,
        configured_max_resident: int,
        reuse_larger_context: bool,
    ) -> ollama.ModelResidencyPlan:
        del reserve_bytes, configured_max_resident
        observed.append((num_ctx, max_num_ctx, reuse_larger_context))
        return _plan(1)

    router = DecisionRouter(
        config=_config(),
        transport=EventTransport({}, []),
        resolve_adoption=False,
        record_replay=False,
        residency_planner=planner,
        live_resource_control=True,
    )

    plan = router._residency_plan(98_304)

    assert plan.source == "test"
    assert observed == [(98_304, 131_072, True)]


@pytest.mark.parametrize(
    ("capacity_gib", "expected_residents"),
    [(31, 2), (33, 3)],
)
def test_residency_upshift_requires_two_gib_or_ten_percent_headroom(
    capacity_gib: int,
    expected_residents: int,
) -> None:
    plan = ollama.build_model_residency_plan(
        MODELS,
        num_ctx=16_384,
        max_num_ctx=131_072,
        memory=ollama.MemorySnapshot(
            total_bytes=128 * ollama.GIB,
            available_bytes=(capacity_gib + 16) * ollama.GIB,
            source="test",
        ),
        installed_sizes={model: 8 * ollama.GIB for model in MODELS},
        resident={},
        calibrated_sizes={(model, 16_384): 10 * ollama.GIB for model in MODELS},
        reserve_bytes=16 * ollama.GIB,
        configured_max_resident=3,
    )

    assert plan.capacity_bytes == capacity_gib * ollama.GIB
    assert plan.max_resident_models == expected_residents
    audit = plan.audit_record()
    assert audit["upshift_min_headroom_bytes"] == 2 * ollama.GIB
    assert audit["upshift_headroom_ratio"] == 0.10


def test_required_context_counts_japanese_and_full_repair_budget() -> None:
    base_schema = {
        "type": "object",
        "required": ["decision"],
        "properties": {"decision": {"type": "boolean"}},
    }
    japanese_schema = {
        **base_schema,
        "properties": {
            "decision": {
                "type": "boolean",
                "description": "記憶を更新してよいか",
            }
        },
    }
    limits = {
        "num_predict": 256,
        "max_output_chars": 1_000,
        "max_feedback_chars": 2_000,
    }

    base = required_structured_context_tokens(
        "",
        base_schema,
        system=None,
        **limits,
    )
    japanese_prompt = "日本語の判断"
    with_prompt = required_structured_context_tokens(
        japanese_prompt,
        base_schema,
        system=None,
        **limits,
    )
    assert with_prompt - base == len(japanese_prompt.encode("utf-8"))

    system = "安全な根拠だけを使う"
    with_system = required_structured_context_tokens(
        japanese_prompt,
        base_schema,
        system=system,
        **limits,
    )
    assert with_system - with_prompt == len(f"{system}\n\n".encode())

    with_schema = required_structured_context_tokens(
        japanese_prompt,
        japanese_schema,
        system=system,
        **limits,
    )
    rendered_base = json.dumps(
        base_schema, ensure_ascii=False, sort_keys=True, indent=2
    )
    rendered_japanese = json.dumps(
        japanese_schema,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    assert with_schema - with_system == len(rendered_japanese.encode("utf-8")) - len(
        rendered_base.encode("utf-8")
    )

    expanded_budget = required_structured_context_tokens(
        japanese_prompt,
        japanese_schema,
        system=system,
        num_predict=limits["num_predict"] + 5,
        max_output_chars=limits["max_output_chars"] + 7,
        max_feedback_chars=limits["max_feedback_chars"] + 11,
    )
    assert expanded_budget - with_schema == 5 + MAX_REPAIR_TURNS * (7 + 11)


@pytest.mark.parametrize(
    ("required", "expected_bucket"),
    [
        (16_384, 16_384),
        (16_385, 32_768),
        (32_768, 32_768),
        (32_769, 65_536),
        (65_536, 65_536),
        (65_537, 98_304),
        (98_304, 98_304),
        (98_305, 131_072),
        (131_072, 131_072),
        (131_073, 131_072),
    ],
)
def test_context_bucket_boundaries_use_full_structured_request_budget(
    required: int,
    expected_bucket: int,
) -> None:
    config = _config()
    router = DecisionRouter(
        config=config,
        transport=EventTransport({}, []),
        resolve_adoption=False,
        record_replay=False,
    )
    system = "日本語の system 指示"
    base_prompt = "日本語の記憶を安全に判断する。"
    base_required = required_structured_context_tokens(
        base_prompt,
        SCHEMA,
        system=system,
        num_predict=config.num_predict,
        max_output_chars=config.max_output_chars,
        max_feedback_chars=config.max_feedback_chars,
    )
    assert base_required <= required
    prompt = base_prompt + "x" * (required - base_required)

    actual_required, selected = router._request_context(prompt, SCHEMA, system)

    assert decision_context_buckets(config) == (
        16_384,
        32_768,
        65_536,
        98_304,
        131_072,
    )
    assert actual_required == required
    assert selected == expected_bucket


def test_uncalibrated_single_resident_bootstraps_and_unloads_after_each_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    transport = EventTransport(
        {
            PRIMARY: ['{"decision":"apply"}', _payload("apply")],
            CHALLENGER: ['{"decision":"apply"}', _payload("apply")],
        },
        events,
    )
    monkeypatch.setattr(
        ollama,
        "model_resource_lease",
        lambda **_kwargs: nullcontext(),
    )

    def observe(model: str) -> tuple[int, int]:
        events.append(("observe", model))
        return (8 * ollama.GIB, 16_384)

    def unload(model: str) -> bool:
        events.append(("unload", model))
        return True

    result = DecisionRouter(
        config=_config(),
        transport=transport,
        audit_root=tmp_path / "audit",
        resolve_adoption=False,
        record_replay=False,
        residency_planner=lambda _models, **_kwargs: _plan(
            1,
            # Bootstrap is allowed only when both uncalibrated pair
            # members fit alone in current reclaimable capacity.
            capacity_bytes=16 * ollama.GIB,
            estimates=(8, 12, 10),
            calibrated_models=(),
        ),
        model_observer=observe,
        model_unloader=unload,
        live_resource_control=True,
    ).decide("prompt", SCHEMA)

    assert result.ok is True
    assert [vote.result.repair_turns for vote in result.votes] == [1, 1]
    assert events == [
        ("chat", PRIMARY, 2),
        ("chat", PRIMARY, 4),
        ("observe", PRIMARY),
        ("unload", PRIMARY),
        ("chat", CHALLENGER, 2),
        ("chat", CHALLENGER, 4),
        ("observe", CHALLENGER),
        ("unload", CHALLENGER),
    ]
    assert result.residency is not None
    assert result.residency["evictions"] == [
        {"model": PRIMARY, "verified": True},
        {"model": CHALLENGER, "verified": True},
    ]


@pytest.mark.parametrize(
    "plan",
    [
        _plan(0, capacity_bytes=6 * ollama.GIB),
        _plan(
            1,
            capacity_bytes=10 * ollama.GIB,
            estimates=(8, 12, 11),
        ),
    ],
)
def test_non_fitting_role_quarantines_before_any_inference(
    plan: ollama.ModelResidencyPlan,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    transport = EventTransport({}, events)
    monkeypatch.setattr(
        ollama,
        "model_resource_lease",
        lambda **_kwargs: nullcontext(),
    )

    result = DecisionRouter(
        config=_config(),
        transport=transport,
        audit_root=tmp_path / "audit",
        resolve_adoption=False,
        record_replay=False,
        residency_planner=lambda _models, **_kwargs: plan,
        model_observer=lambda model: pytest.fail(f"unexpected observe: {model}"),
        model_unloader=lambda model: pytest.fail(f"unexpected unload: {model}"),
        live_resource_control=True,
    ).decide("prompt", SCHEMA)

    assert result.status == "quarantined"
    assert result.failure_class == "local_resource_quarantined"
    assert result.quarantine_reason == "decision_runner_does_not_fit_reserved_memory"
    assert result.votes == ()
    assert transport.requests == []
    assert events == []


def test_non_fitting_tie_does_not_block_an_agreeing_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    transport = EventTransport(
        {
            PRIMARY: [_payload("apply")],
            CHALLENGER: [_payload("apply")],
        },
        events,
    )
    monkeypatch.setattr(
        ollama,
        "model_resource_lease",
        lambda **_kwargs: nullcontext(),
    )
    plans = iter(
        (
            _plan(2, capacity_bytes=20 * ollama.GIB, estimates=(8, 10, 22)),
            _plan(
                2,
                capacity_bytes=20 * ollama.GIB,
                estimates=(8, 10, 22),
                resident_models=(PRIMARY, CHALLENGER),
            ),
        )
    )

    result = DecisionRouter(
        config=_config(),
        transport=transport,
        audit_root=tmp_path / "audit",
        resolve_adoption=False,
        record_replay=False,
        residency_planner=lambda _models, **_kwargs: next(plans),
        model_observer=lambda model: (8 * ollama.GIB, 16_384),
        model_unloader=lambda model: events.append(("unload", model)) or True,
        live_resource_control=True,
    ).decide("prompt", SCHEMA)

    assert result.ok is True
    assert [vote.model for vote in result.votes] == [PRIMARY, CHALLENGER]
    assert events == [("chat", PRIMARY, 2), ("chat", CHALLENGER, 2)]
    assert result.residency is not None
    assert result.residency["evictions"] == []
    assert result.residency["retained_models"] == [PRIMARY, CHALLENGER]


def test_non_fitting_tie_quarantines_only_after_pair_disagrees(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    transport = EventTransport(
        {
            PRIMARY: [_payload("apply")],
            CHALLENGER: [_payload("defer")],
        },
        events,
    )
    monkeypatch.setattr(
        ollama,
        "model_resource_lease",
        lambda **_kwargs: nullcontext(),
    )

    result = DecisionRouter(
        config=_config(),
        transport=transport,
        audit_root=tmp_path / "audit",
        resolve_adoption=False,
        record_replay=False,
        residency_planner=lambda _models, **_kwargs: _plan(
            2,
            capacity_bytes=20 * ollama.GIB,
            estimates=(8, 10, 22),
        ),
        model_observer=lambda model: (8 * ollama.GIB, 16_384),
        model_unloader=lambda model: events.append(("unload", model)) or True,
        live_resource_control=True,
    ).decide("prompt", SCHEMA)

    assert result.status == "quarantined"
    assert result.failure_class == "local_resource_quarantined"
    assert result.quarantine_reason == "tie_break_runner_no_longer_fits_reserved_memory"
    assert [vote.model for vote in result.votes] == [PRIMARY, CHALLENGER]
    assert events == [
        ("chat", PRIMARY, 2),
        ("chat", CHALLENGER, 2),
        ("unload", PRIMARY),
        ("unload", CHALLENGER),
    ]


def test_memory_probe_failure_quarantines_before_any_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    transport = EventTransport({}, events)
    monkeypatch.setattr(
        ollama,
        "model_resource_lease",
        lambda **_kwargs: nullcontext(),
    )

    def fail_probe(
        _models: tuple[str, ...], **_kwargs: Any
    ) -> ollama.ModelResidencyPlan:
        raise RuntimeError("memory probe unavailable")

    result = DecisionRouter(
        config=_config(),
        transport=transport,
        audit_root=tmp_path / "audit",
        resolve_adoption=False,
        record_replay=False,
        residency_planner=fail_probe,
        live_resource_control=True,
    ).decide("prompt", SCHEMA)

    assert result.status == "quarantined"
    assert result.failure_class == "local_resource_quarantined"
    assert result.votes == ()
    assert transport.requests == []


def test_oversize_request_quarantines_before_probe_eviction_or_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    transport = EventTransport({}, events)
    planner_calls: list[object] = []
    unload_calls: list[object] = []
    monkeypatch.setattr(
        ollama,
        "model_resource_lease",
        lambda **_kwargs: nullcontext(),
    )

    def planner(*args: object, **kwargs: object) -> ollama.ModelResidencyPlan:
        planner_calls.append((args, kwargs))
        pytest.fail("oversize request must fail before residency planning")

    def unload(*args: object, **kwargs: object) -> bool:
        unload_calls.append((args, kwargs))
        pytest.fail("oversize request must not evict a runner")

    result = DecisionRouter(
        config=_config(max_input_chars=4_096),
        transport=transport,
        audit_root=tmp_path / "audit",
        resolve_adoption=False,
        record_replay=False,
        residency_planner=planner,
        model_unloader=unload,
        live_resource_control=True,
    ).decide("oversized-user-input" * 1_000, SCHEMA)

    assert result.status == "quarantined"
    assert result.failure_class == "input_too_large"
    assert result.votes == ()
    assert result.residency is not None
    assert result.residency["source"] == "request_preflight_failed_no_probe"
    assert planner_calls == []
    assert unload_calls == []
    assert transport.requests == []
    assert events == []


def test_single_resident_unload_failure_quarantines_before_next_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    transport = EventTransport({PRIMARY: [_payload("apply")]}, events)
    monkeypatch.setattr(
        ollama,
        "model_resource_lease",
        lambda **_kwargs: nullcontext(),
    )

    def unload(model: str) -> bool:
        events.append(("unload", model))
        return False

    result = DecisionRouter(
        config=_config(),
        transport=transport,
        audit_root=tmp_path / "audit",
        resolve_adoption=False,
        record_replay=False,
        residency_planner=lambda _models, **_kwargs: _plan(1),
        model_observer=lambda model: (8 * ollama.GIB, 16_384),
        model_unloader=unload,
        live_resource_control=True,
    ).decide("prompt", SCHEMA)

    assert result.status == "quarantined"
    assert result.failure_class == "local_resource_quarantined"
    assert result.quarantine_reason == "unable_to_verify_primary_runner_eviction"
    assert [vote.model for vote in result.votes] == [PRIMARY]
    assert events == [("chat", PRIMARY, 2), ("unload", PRIMARY)]
    assert result.residency is not None
    assert result.residency["evictions"] == [{"model": PRIMARY, "verified": False}]


def test_two_residents_keep_smaller_pair_model_for_calibrated_tie_break(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    transport = EventTransport(
        {
            PRIMARY: [_payload("apply")],
            CHALLENGER: [_payload("defer")],
            TIE_BREAK: [_payload("apply")],
        },
        events,
    )
    monkeypatch.setattr(
        ollama,
        "model_resource_lease",
        lambda **_kwargs: nullcontext(),
    )

    def unload(model: str) -> bool:
        events.append(("unload", model))
        return True
    plans = iter(
        (
            _plan(2, capacity_bytes=20 * ollama.GIB),
            _plan(
                2,
                capacity_bytes=20 * ollama.GIB,
                resident_models=(PRIMARY, TIE_BREAK),
            ),
        )
    )

    result = DecisionRouter(
        config=_config(),
        transport=transport,
        audit_root=tmp_path / "audit",
        resolve_adoption=False,
        record_replay=False,
        residency_planner=lambda _models, **_kwargs: next(plans),
        model_observer=lambda model: (8 * ollama.GIB, 16_384),
        model_unloader=unload,
        live_resource_control=True,
    ).decide("prompt", SCHEMA)

    assert result.ok is True
    assert [vote.model for vote in result.votes] == list(MODELS)
    assert events == [
        ("chat", PRIMARY, 2),
        ("chat", CHALLENGER, 2),
        ("unload", CHALLENGER),
        ("chat", TIE_BREAK, 2),
        ("unload", TIE_BREAK),
    ]
    assert result.residency is not None
    assert result.residency["evictions"] == [
        {"model": CHALLENGER, "verified": True},
        {"model": TIE_BREAK, "verified": True},
    ]
    assert result.residency["retained_models"] == [PRIMARY]


def test_tie_substitution_keeps_the_same_two_gib_upshift_headroom(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    transport = EventTransport(
        {
            PRIMARY: [_payload("apply")],
            CHALLENGER: [_payload("defer")],
            TIE_BREAK: [_payload("apply")],
        },
        events,
    )
    monkeypatch.setattr(ollama, "model_resource_lease", lambda **_kwargs: nullcontext())

    result = DecisionRouter(
        config=_config(),
        transport=transport,
        audit_root=tmp_path / "audit",
        resolve_adoption=False,
        record_replay=False,
        residency_planner=lambda _models, **_kwargs: _plan(
            2,
            capacity_bytes=19 * ollama.GIB,
        ),
        model_observer=lambda model: (8 * ollama.GIB, 16_384),
        model_unloader=lambda model: events.append(("unload", model)) or True,
        live_resource_control=True,
    ).decide("prompt", SCHEMA)

    assert result.ok is True
    assert events == [
        ("chat", PRIMARY, 2),
        ("chat", CHALLENGER, 2),
        ("unload", PRIMARY),
        ("unload", CHALLENGER),
        ("chat", TIE_BREAK, 2),
        ("unload", TIE_BREAK),
    ]


@pytest.mark.parametrize("max_resident_models", [2, 3])
def test_multi_resident_pair_agreement_retains_safe_warm_runners(
    max_resident_models: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    transport = EventTransport(
        {
            PRIMARY: [_payload("apply")],
            CHALLENGER: [_payload("apply")],
        },
        events,
    )
    monkeypatch.setattr(
        ollama,
        "model_resource_lease",
        lambda **_kwargs: nullcontext(),
    )

    def unload(model: str) -> bool:
        events.append(("unload", model))
        return True
    plans = iter(
        (
            _plan(max_resident_models),
            _plan(
                max_resident_models,
                resident_models=(PRIMARY, CHALLENGER),
            ),
        )
    )

    result = DecisionRouter(
        config=_config(),
        transport=transport,
        audit_root=tmp_path / "audit",
        resolve_adoption=False,
        record_replay=False,
        residency_planner=lambda _models, **_kwargs: next(plans),
        model_observer=lambda model: (8 * ollama.GIB, 16_384),
        model_unloader=unload,
        live_resource_control=True,
    ).decide("prompt", SCHEMA)

    assert result.ok is True
    assert [vote.model for vote in result.votes] == [PRIMARY, CHALLENGER]
    assert events == [("chat", PRIMARY, 2), ("chat", CHALLENGER, 2)]
    assert result.residency is not None
    assert result.residency["evictions"] == []
    assert result.residency["retained_models"] == [PRIMARY, CHALLENGER]


def test_post_decision_pressure_reprobe_evicts_extra_warm_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    transport = EventTransport(
        {PRIMARY: [_payload("apply")], CHALLENGER: [_payload("apply")]},
        events,
    )
    monkeypatch.setattr(ollama, "model_resource_lease", lambda **_kwargs: nullcontext())
    plans = iter(
        (
            _plan(3),
            _plan(
                1,
                resident_models=(PRIMARY, CHALLENGER),
                pressure_forced_single=True,
                compressed_bytes=40 * ollama.GIB,
                swap_used_bytes=8 * ollama.GIB,
            ),
        )
    )

    result = DecisionRouter(
        config=_config(),
        transport=transport,
        audit_root=tmp_path / "audit",
        resolve_adoption=False,
        record_replay=False,
        residency_planner=lambda _models, **_kwargs: next(plans),
        model_observer=lambda model: (8 * ollama.GIB, 16_384),
        model_unloader=lambda model: events.append(("unload", model)) or True,
        live_resource_control=True,
    ).decide("prompt", SCHEMA)

    assert result.ok is True
    assert events == [
        ("chat", PRIMARY, 2),
        ("chat", CHALLENGER, 2),
        ("unload", CHALLENGER),
    ]
    assert result.residency is not None
    assert result.residency["retained_models"] == [PRIMARY]
    assert result.residency["post_decision"]["pressure_forced_single"] is True
    assert result.residency["post_decision"]["compressed_bytes"] == 40 * ollama.GIB
    assert result.residency["post_decision"]["swap_used_bytes"] == 8 * ollama.GIB


def test_three_residents_run_tie_break_then_retain_safe_warm_runners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    transport = EventTransport(
        {
            PRIMARY: [_payload("apply")],
            CHALLENGER: [_payload("defer")],
            TIE_BREAK: [_payload("apply")],
        },
        events,
    )
    monkeypatch.setattr(
        ollama,
        "model_resource_lease",
        lambda **_kwargs: nullcontext(),
    )
    plans = iter((_plan(3), _plan(3, resident_models=MODELS)))

    result = DecisionRouter(
        config=_config(),
        transport=transport,
        audit_root=tmp_path / "audit",
        resolve_adoption=False,
        record_replay=False,
        residency_planner=lambda _models, **_kwargs: next(plans),
        model_observer=lambda model: (8 * ollama.GIB, 16_384),
        model_unloader=lambda model: events.append(("unload", model)) or True,
        live_resource_control=True,
    ).decide("prompt", SCHEMA)

    assert result.ok is True
    assert [vote.model for vote in result.votes] == list(MODELS)
    assert events == [
        ("chat", PRIMARY, 2),
        ("chat", CHALLENGER, 2),
        ("chat", TIE_BREAK, 2),
    ]
    assert result.residency is not None
    assert result.residency["evictions"] == []
    assert result.residency["retained_models"] == list(MODELS)


def test_three_resident_plan_evicts_incompatible_runner_before_first_vote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[Any, ...]] = []
    transport = EventTransport(
        {
            PRIMARY: [_payload("apply")],
            CHALLENGER: [_payload("apply")],
        },
        events,
    )
    monkeypatch.setattr(
        ollama,
        "model_resource_lease",
        lambda **_kwargs: nullcontext(),
    )

    def unload(model: str) -> bool:
        events.append(("unload", model))
        return True
    plans = iter(
        (
            _plan(
                3,
                resident_models=(TIE_BREAK,),
                initial_eviction_models=(TIE_BREAK,),
            ),
            _plan(3, resident_models=(PRIMARY, CHALLENGER)),
        )
    )

    result = DecisionRouter(
        config=_config(),
        transport=transport,
        audit_root=tmp_path / "audit",
        resolve_adoption=False,
        record_replay=False,
        residency_planner=lambda _models, **_kwargs: next(plans),
        model_observer=lambda model: (8 * ollama.GIB, 16_384),
        model_unloader=unload,
        live_resource_control=True,
    ).decide("prompt", SCHEMA)

    assert result.ok is True
    assert events == [
        ("unload", TIE_BREAK),
        ("chat", PRIMARY, 2),
        ("chat", CHALLENGER, 2),
    ]
    assert result.residency is not None
    assert result.residency["evictions"] == [
        {"model": TIE_BREAK, "verified": True},
    ]
    assert result.residency["retained_models"] == [PRIMARY, CHALLENGER]
