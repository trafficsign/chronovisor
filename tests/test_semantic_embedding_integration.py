import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from chronovisor.core import semantic_client
from chronovisor.core.llm_config import build_llm_runtime, parse_llm_config
from chronovisor.core.llm_runtime import (
    EmbeddingPurpose,
    EmbeddingRequest,
    SafeBackendError,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)
from chronovisor.core.runtime_config import SearchEmbeddingConfig
from chronovisor.search.semantic_service import SemanticServiceState, _ModelLock


def test_priority_lock_gives_waiting_query_next_slot():
    lock = _ModelLock()
    assert lock.acquire(timeout=1)
    order = []

    def run(label, background):
        assert lock.acquire(timeout=2, background=background)
        order.append(label)
        lock.release()

    background = threading.Thread(target=run, args=("background", True))
    foreground = threading.Thread(target=run, args=("query", False))
    background.start()
    foreground.start()
    deadline = time.monotonic() + 1
    while lock._waiting != 1 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert lock._waiting == 1
    lock.release()
    foreground.join(2)
    background.join(2)
    assert order == ["query", "background"]
    assert lock.acquire(timeout=1)
    assert not lock.acquire(timeout=0.001)
    lock.release()
    assert lock.acquire(timeout=1, background=True)
    lock.release()


def test_service_embedding_validates_identity_and_preserves_source(monkeypatch):
    state = object.__new__(SemanticServiceState)
    state.config = SearchEmbeddingConfig(dimensions=2)
    state._foreground_route = SimpleNamespace(model="nemotron")
    captured = []

    def encode(texts, purpose, **kwargs):
        captured.append((texts, purpose, kwargs))
        return np.array([[1.0, 0.0]])

    monkeypatch.setattr(state, "_embed_foreground", encode)
    payload = {
        "method": "embed",
        "model": "nemotron",
        "texts": ["日本語", "English"],
        "purpose": "query",
        "source_data_class": "system",
        "source_sensitivity": "high",
    }
    assert state.handle(payload)["vectors"] == [[1.0, 0.0], [1.0, 0.0]]
    assert len(captured) == 2
    assert all(item[2]["background"] for item in captured)
    assert captured[0][1] == EmbeddingPurpose.QUERY
    assert captured[0][2]["source"].sensitivity == SourceSensitivity.HIGH
    for bad in (
        {"model": "bge"},
        {"texts": [1]},
        {"texts": []},
        {"purpose": "unknown"},
        {"source_sensitivity": "unknown"},
        {"deadline_at": time.monotonic() - 1},
    ):
        with pytest.raises((ValueError, TimeoutError)):
            state.handle({**payload, **bad})


def test_embedding_ipc_uses_shared_deadline_and_no_search_breaker(monkeypatch):
    backend = semantic_client.SemanticEmbeddingBackend(
        SearchEmbeddingConfig(dimensions=2)
    )
    calls = []

    def request(payload, config, **kwargs):
        calls.append((payload, kwargs))
        return {"model": "nemotron", "vectors": [[1.0, 0.0]]}

    monkeypatch.setattr(semantic_client, "request", request)
    source = SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.HIGH)
    req = EmbeddingRequest(
        ("日本語", "English"), source, 3000, EmbeddingPurpose.DOCUMENT
    )
    result = backend.embed(req, model="nemotron")
    assert len(result.vectors) == 2
    assert result.provider == "semantic-service"
    assert calls[0][1]["deadline_at"] == calls[1][1]["deadline_at"]
    assert calls[0][1]["circuit_breaker"] is False
    assert calls[0][0]["source_sensitivity"] == "high"
    monkeypatch.setattr(
        semantic_client,
        "request",
        lambda *a, **kw: {"model": "wrong", "vectors": [[1.0, 0.0]]},
    )
    with pytest.raises(SafeBackendError):
        backend.embed(req, model="nemotron")


def test_mlx_native_lanes_share_backend_and_knowledge_uses_ipc():
    config = parse_llm_config(
        {
            "llm": {
                "providers": {
                    "fg": {"kind": "nemotron", "device": "mlx"},
                    "bg": {"kind": "nemotron", "device": "mlx"},
                    "shared": {"kind": "semantic-service"},
                },
                "roles": {
                    role: {
                        "capability": "embedding",
                        "provider": provider,
                        "model": "nvidia/Nemotron-3-Embed-1B-BF16",
                    }
                    for role, provider in (
                        ("search.semantic.foreground", "fg"),
                        ("search.semantic.incremental", "bg"),
                        ("knowledge.embedding", "shared"),
                        ("classification.embedding", "shared"),
                    )
                },
            }
        }
    )
    runtime = build_llm_runtime(
        config,
        search_embedding_config=replace(
            SearchEmbeddingConfig(), query_device="mlx", incremental_device="mlx"
        ),
    )
    assert (
        runtime._embedding["search.semantic.foreground"].backend
        is runtime._embedding["search.semantic.incremental"].backend
    )
    assert isinstance(
        runtime._embedding["knowledge.embedding"].backend,
        semantic_client.SemanticEmbeddingBackend,
    )
