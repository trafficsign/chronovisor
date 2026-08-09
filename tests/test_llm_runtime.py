from __future__ import annotations

from contextlib import nullcontext

import pytest

from chronovisor.core import ollama
from chronovisor.core.llm_runtime import (
    BackendContractError,
    BackendExecutionError,
    CapabilityUnavailableError,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingRoute,
    GenerationInput,
    GenerationRequest,
    GenerationResult,
    GenerationRoute,
    LLMRuntime,
    MessageGenerationRequest,
    RerankItem,
    RerankRequest,
    RerankResult,
    RerankRoute,
)
from chronovisor.core.ollama_adapter import OllamaAdapter, compose_ollama_runtime


class FakeBackend:
    provider = "fake"

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

    assert runtime.generate("review", GenerationRequest("hello")).model == "writer"
    assert runtime.embed("search", EmbeddingRequest(("a",))).model == "embedder"
    assert runtime.rerank("search", RerankRequest("q", ("a", "b"))).model == "reranker"
    with pytest.raises(CapabilityUnavailableError):
        runtime.generate("search", GenerationRequest("no fallback"))


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

    runtime = LLMRuntime(
        embedding={"search": EmbeddingRoute(InvalidEmbedding(), "embedder")}
    )

    with pytest.raises(BackendContractError, match=reason):
        runtime.embed("search", EmbeddingRequest(("a", "b")))


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
        runtime.rerank("search", RerankRequest("q", ("a", "b")))


def test_runtime_normalizes_backend_errors_without_fallback() -> None:
    class BrokenGeneration(FakeBackend):
        def generate(self, request: GenerationInput, *, model: str) -> GenerationResult:
            raise OSError("provider detail")

    runtime = LLMRuntime(
        generation={"review": GenerationRoute(BrokenGeneration(), "writer")}
    )

    with pytest.raises(BackendExecutionError, match="fake generation backend") as exc:
        runtime.generate("review", GenerationRequest("hello"))
    assert isinstance(exc.value.__cause__, OSError)


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
            num_ctx=8192,
            max_output_tokens=64,
            keep_alive="10m",
            timeout_ms=60_000,
            max_output_chars=4000,
            think="low",
        ),
    )
    embedded = runtime.embed("search", EmbeddingRequest(("a", "b"), 9000))

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
