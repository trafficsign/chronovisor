import threading
import time
from collections import OrderedDict, deque
from types import SimpleNamespace

import numpy as np
import pytest

from chronovisor.core.llm_runtime import (
    EmbeddingPurpose,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingRoute,
    LLMRuntime,
    RouteLocation,
)
from chronovisor.core.runtime_config import SearchEmbeddingConfig
from chronovisor.search.semantic_model import SemanticModelError
from chronovisor.search.semantic_service import (
    DOCUMENT_SOURCE,
    FOREGROUND_ROLE,
    INCREMENTAL_ROLE,
    QueryBatcher,
    SemanticServiceState,
    ServiceBusy,
    _drifted_page_ids,
)


class FakeEmbeddingBackend:
    provider = "nemotron"
    location = RouteLocation.LOCAL

    def __init__(self) -> None:
        self.requests: list[EmbeddingRequest] = []

    def embed(self, request: EmbeddingRequest, *, model: str) -> EmbeddingResult:
        self.requests.append(request)
        return EmbeddingResult(
            tuple((1.0, 0.0) for _ in request.texts), self.provider, model
        )


def test_drifted_page_ids_are_deduplicated_across_states() -> None:
    assert _drifted_page_ids(
        {
            "missing_page_ids": ["new", "shared"],
            "stale_page_ids": ["changed", "shared"],
            "deleted_page_ids": ["gone"],
        }
    ) == ["changed", "gone", "new", "shared"]


def test_semantic_routes_reject_remote_before_any_embedding_call() -> None:
    class RemoteBackend(FakeEmbeddingBackend):
        provider = "cloud"
        location = RouteLocation.REMOTE

    backend = RemoteBackend()
    runtime = LLMRuntime(
        embedding={
            FOREGROUND_ROLE: EmbeddingRoute(backend, "model"),
            INCREMENTAL_ROLE: EmbeddingRoute(backend, "model"),
        }
    )
    state = object.__new__(SemanticServiceState)
    state.config = SearchEmbeddingConfig(model="model")
    state._runtime = runtime

    with pytest.raises(SemanticModelError):
        state._validate_runtime_routes()

    assert backend.requests == []


def test_semantic_routes_reject_model_change_until_index_rebuild() -> None:
    backend = FakeEmbeddingBackend()
    runtime = LLMRuntime(
        embedding={
            FOREGROUND_ROLE: EmbeddingRoute(backend, "changed-model"),
            INCREMENTAL_ROLE: EmbeddingRoute(backend, "changed-model"),
        }
    )
    state = object.__new__(SemanticServiceState)
    state.config = SearchEmbeddingConfig(model="indexed-model")
    state._runtime = runtime

    with pytest.raises(SemanticModelError):
        state._validate_runtime_routes()

    assert backend.requests == []


def test_semantic_runtime_vectors_pass_role_purpose_and_page_source() -> None:
    backend = FakeEmbeddingBackend()
    state = object.__new__(SemanticServiceState)
    state.config = SearchEmbeddingConfig(model="model", dimensions=2)
    state._runtime = LLMRuntime(
        embedding={FOREGROUND_ROLE: EmbeddingRoute(backend, "model")}
    )

    vectors = state._runtime_vectors(
        FOREGROUND_ROLE,
        ["document"],
        EmbeddingPurpose.DOCUMENT,
    )

    assert vectors.shape == (1, 2)
    assert backend.requests[0].purpose is EmbeddingPurpose.DOCUMENT
    assert backend.requests[0].source == DOCUMENT_SOURCE


def test_incremental_model_is_released_when_lazy_self_test_fails() -> None:
    released: list[str] = []

    def fail(_role: str) -> dict[str, object]:
        raise SemanticModelError("failed")

    state = object.__new__(SemanticServiceState)
    state._cpu_ready = False
    state._runtime = SimpleNamespace(
        release_embedding=lambda role: released.append(role)
    )
    state._self_test_role = fail

    with pytest.raises(SemanticModelError):
        state._ensure_cpu()

    assert released == [INCREMENTAL_ROLE]
    assert state._cpu_ready is False


def test_query_batcher_combines_concurrent_requests() -> None:
    calls: list[list[str]] = []

    def encode(texts: list[str], _batch_size: int) -> np.ndarray:
        calls.append(texts)
        return np.asarray([[float(len(text)), 1.0] for text in texts])

    batcher = QueryBatcher(
        encode=encode,
        search=lambda vector, _top_n: [("page", float(vector[0]))],
        window_ms=5,
        max_batch=8,
        available=lambda: True,
    )
    try:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(batcher.submit, f"q{index}", 1, 1.0) for index in range(4)
            ]
            assert all(future.result()[0][0] == "page" for future in futures)
        assert len(calls) == 1
        assert len(calls[0]) == 4
    finally:
        batcher.close()


def test_query_batcher_rejects_when_generation_is_unavailable() -> None:
    batcher = QueryBatcher(
        encode=lambda texts, _batch: np.zeros((len(texts), 2)),
        search=lambda _vector, _top_n: [],
        window_ms=0,
        max_batch=1,
        available=lambda: False,
    )
    try:
        with pytest.raises(ServiceBusy):
            batcher.submit("query", 1, 1.0)
    finally:
        batcher.close()


def test_search_reuses_cached_query_vector_without_batcher() -> None:
    state = object.__new__(SemanticServiceState)
    vector = np.asarray([1.0, 0.0], dtype=np.float32)
    state.config = SimpleNamespace(interactive_timeout_ms=500)
    state._query_cache_lock = threading.Lock()
    state._query_vector_cache = OrderedDict(
        {"repeated query": (time.monotonic(), vector)}
    )
    state._generation_lock = threading.RLock()
    state._generation = SimpleNamespace(
        manifest=SimpleNamespace(generation_id="test-generation")
    )
    state._metrics_lock = threading.Lock()
    state._query_latencies_ms = deque(maxlen=10)
    state._reload_if_pointer_changed = lambda: None
    state._publish_status = lambda: pytest.fail(
        "status publication must stay off the foreground path"
    )
    state._search_vector = lambda cached, top_n: [("page", float(cached[0]) + top_n)]
    state._batcher = SimpleNamespace(
        submit=lambda *_args, **_kwargs: pytest.fail("batcher was called")
    )

    response = state.search("repeated query", 3)

    assert response["cache_hit"] is True
    assert response["results"] == [{"page_id": "page", "score": 4.0}]


def test_query_path_warmup_exercises_three_queries_and_ann_search() -> None:
    state = object.__new__(SemanticServiceState)
    encoded: list[list[str]] = []
    searched: list[np.ndarray] = []
    state._encode_queries = lambda queries, _batch: (
        encoded.append(list(queries))
        or np.asarray([[1.0, float(index)] for index in range(len(queries))])
    )
    state._search_vector = lambda vector, _top_n: (
        searched.append(vector) or [("page", 1.0)]
    )

    result = state._warm_query_path()

    assert len(encoded[0]) == 3
    assert len(searched) == 3
    assert result["hits"] == 3
