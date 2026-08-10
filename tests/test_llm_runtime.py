from __future__ import annotations

from contextlib import nullcontext
from typing import cast

import pytest

from chronovisor.core import ollama
from chronovisor.core.llm_runtime import (
    MAX_CONTEXT_TOKENS,
    MAX_OUTPUT_CHARS,
    MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    BackendContractError,
    BackendExecutionError,
    CapabilityUnavailableError,
    EgressDeniedError,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingRoute,
    GenerationInput,
    GenerationRequest,
    GenerationResult,
    GenerationRoute,
    LLMRuntime,
    MessageGenerationRequest,
    RequestValidationError,
    RerankItem,
    RerankRequest,
    RerankResult,
    RerankRoute,
    RouteConfigurationError,
    RouteLocation,
    RuntimeFailureTelemetry,
    SourceClassificationError,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)
from chronovisor.core.llm_security import MAX_REQUEST_TIMEOUT_MS
from chronovisor.core.ollama_adapter import OllamaAdapter, compose_ollama_runtime

NORMAL_PAGE = SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.NORMAL)
NORMAL_SNIPPET = SourceDataClassification(
    SourceDataClass.DERIVED_SNIPPET, SourceSensitivity.NORMAL
)
RAW_NORMAL = SourceDataClassification(SourceDataClass.RAW, SourceSensitivity.NORMAL)
RAW_HIGH = SourceDataClassification(SourceDataClass.RAW, SourceSensitivity.HIGH)
SYSTEM_NORMAL = SourceDataClassification(
    SourceDataClass.SYSTEM, SourceSensitivity.NORMAL
)
PAGE_HIGH = SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.HIGH)


class FakeBackend:
    provider = "fake"
    location = RouteLocation.LOCAL

    def generate(self, request: GenerationInput, *, model: str) -> GenerationResult:
        content = (
            request.messages[-1]["content"]
            if isinstance(request, MessageGenerationRequest)
            else request.prompt
        )
        return GenerationResult(content, self.provider, model)

    def embed(self, request: EmbeddingRequest, *, model: str) -> EmbeddingResult:
        return EmbeddingResult(
            tuple((float(index), 1.0) for index, _ in enumerate(request.texts)),
            self.provider,
            model,
        )

    def rerank(self, request: RerankRequest, *, model: str) -> RerankResult:
        return RerankResult(
            tuple(
                RerankItem(index, float(len(request.candidates) - index))
                for index in range(len(request.candidates))
            ),
            self.provider,
            model,
        )


def test_runtime_routes_each_capability_without_fallback() -> None:
    backend = FakeBackend()
    runtime = LLMRuntime(
        generation={"review": GenerationRoute(backend, "writer")},
        embedding={"search": EmbeddingRoute(backend, "embedder")},
        rerank={"search": RerankRoute(backend, "reranker")},
    )

    assert (
        runtime.generate("review", GenerationRequest("hello", RAW_HIGH)).model
        == "writer"
    )
    assert (
        runtime.embed("search", EmbeddingRequest(("a",), NORMAL_PAGE)).model
        == "embedder"
    )
    assert (
        runtime.rerank("search", RerankRequest("q", ("a", "b"), NORMAL_PAGE)).model
        == "reranker"
    )
    with pytest.raises(CapabilityUnavailableError):
        runtime.generate("search", GenerationRequest("no fallback", NORMAL_PAGE))


@pytest.mark.parametrize(
    "result, reason",
    [
        (EmbeddingResult(((1.0,),), "fake", "embedder"), "vector count mismatch"),
        (
            EmbeddingResult(((1.0,), (1.0, 2.0)), "fake", "embedder"),
            "invalid vector dimensions",
        ),
    ],
)
def test_runtime_rejects_invalid_embedding_contract(
    result: EmbeddingResult, reason: str
) -> None:
    class InvalidEmbedding(FakeBackend):
        def embed(self, request: EmbeddingRequest, *, model: str) -> EmbeddingResult:
            return result

    events: list[RuntimeFailureTelemetry] = []
    runtime = LLMRuntime(
        embedding={"search": EmbeddingRoute(InvalidEmbedding(), "embedder")},
        telemetry=events.append,
    )

    with pytest.raises(BackendContractError, match=reason):
        runtime.embed("search", EmbeddingRequest(("a", "b"), NORMAL_PAGE))

    assert events == [
        RuntimeFailureTelemetry(
            "backend_contract_error", "search", "embedding", "fake", "local"
        )
    ]


def test_runtime_rejects_invalid_rerank_contract() -> None:
    class InvalidRerank(FakeBackend):
        def rerank(self, request: RerankRequest, *, model: str) -> RerankResult:
            return RerankResult(
                (RerankItem(0, 1.0), RerankItem(0, 0.5)),
                self.provider,
                model,
            )

    runtime = LLMRuntime(rerank={"search": RerankRoute(InvalidRerank(), "reranker")})

    with pytest.raises(BackendContractError, match="invalid ranking indices"):
        runtime.rerank("search", RerankRequest("q", ("a", "b"), NORMAL_PAGE))


def test_generation_contract_failure_emits_safe_request_id() -> None:
    class InvalidGeneration(FakeBackend):
        def generate(self, request: GenerationInput, *, model: str) -> GenerationResult:
            return GenerationResult(
                "content",
                "wrong-provider",
                model,
                metadata={"request_id": "req_contract_1"},
            )

    events: list[RuntimeFailureTelemetry] = []
    runtime = LLMRuntime(
        generation={"review": GenerationRoute(InvalidGeneration(), "writer")},
        telemetry=events.append,
    )

    with pytest.raises(BackendContractError, match="route identity mismatch"):
        runtime.generate("review", GenerationRequest("prompt", NORMAL_PAGE))

    assert events == [
        RuntimeFailureTelemetry(
            "backend_contract_error",
            "review",
            "generation",
            "fake",
            "local",
            request_id="req_contract_1",
        )
    ]


def test_runtime_normalizes_backend_errors_without_fallback() -> None:
    canary = "CANARY_PROVIDER_SECRET"
    events: list[RuntimeFailureTelemetry] = []

    class BrokenGeneration(FakeBackend):
        def generate(self, request: GenerationInput, *, model: str) -> GenerationResult:
            raise OSError(canary)

    runtime = LLMRuntime(
        generation={"review": GenerationRoute(BrokenGeneration(), "writer")},
        telemetry=events.append,
    )
    request = GenerationRequest(canary, NORMAL_PAGE, system=canary)

    with pytest.raises(BackendExecutionError, match="fake generation backend") as exc:
        runtime.generate("review", request)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None
    assert events == [
        RuntimeFailureTelemetry(
            category="backend_error",
            role="review",
            capability="generation",
            provider="fake",
            location="local",
        )
    ]
    assert canary not in repr(request)
    assert canary not in str(exc.value)
    assert canary not in repr(exc.value)
    assert canary not in repr(events)


class CountingRemoteBackend(FakeBackend):
    location = RouteLocation.REMOTE

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, request: GenerationInput, *, model: str) -> GenerationResult:
        self.calls.append("generation")
        return super().generate(request, model=model)

    def embed(self, request: EmbeddingRequest, *, model: str) -> EmbeddingResult:
        self.calls.append("embedding")
        return super().embed(request, model=model)

    def rerank(self, request: RerankRequest, *, model: str) -> RerankResult:
        self.calls.append("rerank")
        return super().rerank(request, model=model)


def test_runtime_requires_classification_before_backend_call() -> None:
    backend = CountingRemoteBackend()
    runtime = LLMRuntime(generation={"review": GenerationRoute(backend, "writer")})
    unclassified = cast(SourceDataClassification, None)

    with pytest.raises(SourceClassificationError):
        runtime.generate("review", GenerationRequest("secret", unclassified))

    assert backend.calls == []


@pytest.mark.parametrize("source", [NORMAL_PAGE, NORMAL_SNIPPET])
def test_remote_default_allows_only_normal_cloud_eligible_data(
    source: SourceDataClassification,
) -> None:
    backend = CountingRemoteBackend()
    runtime = LLMRuntime(generation={"review": GenerationRoute(backend, "writer")})

    runtime.generate("review", GenerationRequest("content", source))

    assert backend.calls == ["generation"]


@pytest.mark.parametrize("source", [RAW_NORMAL, SYSTEM_NORMAL, PAGE_HIGH, RAW_HIGH])
def test_remote_default_denies_restricted_data_before_backend_call(
    source: SourceDataClassification,
) -> None:
    backend = CountingRemoteBackend()
    events: list[RuntimeFailureTelemetry] = []
    runtime = LLMRuntime(
        generation={"review": GenerationRoute(backend, "writer")},
        telemetry=events.append,
    )

    with pytest.raises(EgressDeniedError):
        runtime.generate("review", GenerationRequest("content", source))

    assert backend.calls == []
    assert events[0].category == "egress_denied"
    assert events[0].location == "remote"


def test_remote_egress_opt_in_is_scoped_to_exact_role_and_data_class() -> None:
    backend = CountingRemoteBackend()
    route = GenerationRoute(backend, "writer")
    runtime = LLMRuntime(
        generation={"review": route, "other": route},
        remote_egress_opt_ins={("review", SourceDataClass.RAW)},
    )

    runtime.generate("review", GenerationRequest("raw", RAW_HIGH))
    with pytest.raises(EgressDeniedError):
        runtime.generate("other", GenerationRequest("raw", RAW_HIGH))
    with pytest.raises(EgressDeniedError):
        runtime.generate("review", GenerationRequest("system", SYSTEM_NORMAL))

    assert backend.calls == ["generation"]


def test_remote_denial_covers_all_capabilities_without_fallback() -> None:
    backend = CountingRemoteBackend()
    runtime = LLMRuntime(
        generation={"review": GenerationRoute(backend, "writer")},
        embedding={"search": EmbeddingRoute(backend, "embedder")},
        rerank={"search": RerankRoute(backend, "reranker")},
    )

    with pytest.raises(EgressDeniedError):
        runtime.generate("review", GenerationRequest("raw", RAW_NORMAL))
    with pytest.raises(EgressDeniedError):
        runtime.embed("search", EmbeddingRequest(("raw",), RAW_NORMAL))
    with pytest.raises(EgressDeniedError):
        runtime.rerank("search", RerankRequest("q", ("raw",), RAW_NORMAL))

    assert backend.calls == []


def test_invalid_route_location_fails_closed() -> None:
    class InvalidRouteBackend(CountingRemoteBackend):
        location = cast(RouteLocation, "cloud")

    backend = InvalidRouteBackend()
    runtime = LLMRuntime(generation={"review": GenerationRoute(backend, "writer")})

    with pytest.raises(RouteConfigurationError):
        runtime.generate("review", GenerationRequest("content", NORMAL_PAGE))

    assert backend.calls == []


@pytest.mark.parametrize(
    "generation_input,field_name",
    [
        (
            GenerationRequest(
                "prompt", NORMAL_PAGE, max_output_tokens=MAX_OUTPUT_TOKENS + 1
            ),
            "max_output_tokens",
        ),
        (GenerationRequest("prompt", NORMAL_PAGE, num_ctx=0), "num_ctx"),
        (
            GenerationRequest(
                "prompt", NORMAL_PAGE, timeout_ms=MAX_REQUEST_TIMEOUT_MS + 1
            ),
            "timeout_ms",
        ),
        (
            MessageGenerationRequest(
                messages=({"role": "user", "content": "prompt"},),
                format={},
                source=NORMAL_PAGE,
                num_ctx=MAX_CONTEXT_TOKENS,
                max_output_tokens=1,
                keep_alive="0",
                timeout_ms=1,
                max_output_chars=MAX_OUTPUT_CHARS + 1,
            ),
            "max_output_chars",
        ),
    ],
)
def test_runtime_rejects_generation_budget_before_backend_call(
    generation_input: GenerationInput,
    field_name: str,
) -> None:
    backend = CountingRemoteBackend()
    events: list[RuntimeFailureTelemetry] = []
    runtime = LLMRuntime(
        generation={"review": GenerationRoute(backend, "writer")},
        telemetry=events.append,
    )

    with pytest.raises(RequestValidationError) as exc:
        runtime.generate("review", generation_input)

    assert exc.value.field_name == field_name
    assert backend.calls == []
    assert events == [
        RuntimeFailureTelemetry(
            "request_invalid", "review", "generation", "fake", "remote"
        )
    ]


@pytest.mark.parametrize("capability", ["embedding", "rerank"])
def test_runtime_rejects_timeout_budget_for_every_non_generation_route(
    capability: str,
) -> None:
    backend = CountingRemoteBackend()
    runtime = LLMRuntime(
        embedding={"search": EmbeddingRoute(backend, "embedder")},
        rerank={"search": RerankRoute(backend, "reranker")},
    )

    with pytest.raises(RequestValidationError) as exc:
        if capability == "embedding":
            runtime.embed(
                "search", EmbeddingRequest(("text",), NORMAL_PAGE, timeout_ms=0)
            )
        else:
            runtime.rerank(
                "search",
                RerankRequest("query", ("text",), NORMAL_PAGE, timeout_ms=0),
            )

    assert exc.value.field_name == "timeout_ms"
    assert backend.calls == []


@pytest.mark.parametrize("max_retries", [-1, MAX_RETRIES + 1, True])
def test_runtime_retry_policy_is_explicit_and_bounded(max_retries: object) -> None:
    with pytest.raises(ValueError, match="max_retries"):
        LLMRuntime(max_retries=cast(int, max_retries))


def test_runtime_failure_telemetry_rejects_non_allowlisted_fields() -> None:
    with pytest.raises(ValueError, match="unsafe runtime failure telemetry"):
        RuntimeFailureTelemetry(
            "not-safe",
            "review",
            "generation",
            "fake",
            "remote",
            request_id="request id with spaces",
        )


def test_request_and_result_repr_hide_content() -> None:
    canary = "CANARY_PRIVATE_CONTENT"
    values = [
        GenerationRequest(canary, NORMAL_PAGE, system=canary, format={canary: canary}),
        MessageGenerationRequest(
            messages=({"role": "user", "content": canary},),
            format={canary: canary},
            source=NORMAL_PAGE,
            num_ctx=8,
            max_output_tokens=1,
            keep_alive="0",
            timeout_ms=1,
            max_output_chars=1,
        ),
        EmbeddingRequest((canary,), NORMAL_PAGE),
        RerankRequest(canary, (canary,), NORMAL_PAGE),
        GenerationResult(canary, "fake", "model", metadata={canary: canary}),
    ]

    assert all(canary not in repr(value) for value in values)


def test_ollama_adapter_preserves_generate_options_and_normalizes(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_generate(prompt: str, system: str | None, **kwargs: object):
        captured.update({"prompt": prompt, "system": system, **kwargs})
        return ollama.GenerateResponse(
            content="done",
            done=True,
            done_reason="stop",
            prompt_eval_count=7,
            eval_count=3,
            streamed=True,
        )

    monkeypatch.setattr(ollama, "generate", fake_generate)
    runtime = compose_ollama_runtime(generation_roles={"ingest": "ornith:test"})

    result = runtime.generate(
        "ingest",
        GenerationRequest(
            "prompt",
            NORMAL_PAGE,
            system="system",
            format={"type": "object"},
            num_ctx=4096,
            max_output_tokens=128,
            keep_alive="5m",
            timeout_ms=30_000,
            temperature=0.1,
            seed=4,
        ),
    )

    assert result.content == "done"
    assert result.completed is True
    assert result.finish_reason == "stop"
    assert result.usage.input_tokens == 7
    assert result.usage.output_tokens == 3
    assert result.metadata == {"streamed": True}
    assert captured == {
        "prompt": "prompt",
        "system": "system",
        "format": {"type": "object"},
        "progress_callback": None,
        "model": "ornith:test",
        "num_ctx": 4096,
        "num_predict": 128,
        "keep_alive": "5m",
        "read_timeout_ms": 30_000,
        "temperature": 0.1,
        "seed": 4,
        "return_metadata": True,
    }


def test_ollama_adapter_preserves_chat_and_embedding_paths(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_chat(messages: list[dict[str, str]], **kwargs: object):
        seen["messages"] = messages
        seen["chat"] = kwargs
        return ollama.ChatResponse("{}", 11, 2, True, "stop")

    def fake_embed(
        texts: list[str], *, model: str, read_timeout_ms: int | None
    ) -> list[list[float]]:
        seen["embed"] = (texts, model, read_timeout_ms)
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(ollama, "chat", fake_chat)
    monkeypatch.setattr(ollama, "embed", fake_embed)
    runtime = compose_ollama_runtime(
        generation_roles={"decision": "gpt-oss:test"},
        embedding_roles={"search": "bge:test"},
    )

    chat = runtime.generate(
        "decision",
        MessageGenerationRequest(
            messages=({"role": "user", "content": "vote"},),
            format={"type": "object"},
            source=NORMAL_PAGE,
            num_ctx=8192,
            max_output_tokens=64,
            keep_alive="10m",
            timeout_ms=60_000,
            max_output_chars=4000,
            think="low",
        ),
    )
    embedded = runtime.embed("search", EmbeddingRequest(("a", "b"), NORMAL_PAGE, 9000))

    assert chat.content == "{}"
    assert chat.usage.input_tokens == 11
    assert seen["messages"] == [{"role": "user", "content": "vote"}]
    assert seen["chat"] == {
        "model": "gpt-oss:test",
        "format": {"type": "object"},
        "num_ctx": 8192,
        "num_predict": 64,
        "keep_alive": "10m",
        "read_timeout_ms": 60_000,
        "max_output_chars": 4000,
        "temperature": 0,
        "seed": 0,
        "think": "low",
        "return_metadata": True,
    }
    assert embedded.vectors == ((1.0, 0.0), (1.0, 0.0))
    assert seen["embed"] == (["a", "b"], "bge:test", 9000)


def test_ollama_adapter_exposes_local_control_only_to_runtime(monkeypatch) -> None:
    adapter = OllamaAdapter()
    monkeypatch.setattr(ollama, "model_resource_lease", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(ollama, "resident_model_rows", lambda: {"m": (10, 4096)})
    monkeypatch.setattr(ollama, "unload_named_model", lambda *_args, **_kwargs: True)
    runtime = LLMRuntime(local_controls={"local": adapter})

    control = runtime._local_control_for("local")

    assert control is adapter
    assert control.resident_models() == {"m": (10, 4096)}
    assert control.unload("m") is True
    with control.resource_lease(exclusive=False):
        pass
    assert not hasattr(runtime, "unload")
