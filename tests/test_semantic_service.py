import io
import json
import threading
import time
from collections import OrderedDict, deque
from types import SimpleNamespace

import numpy as np
import pytest

from chronovisor.core import ollama
from chronovisor.core.llm_runtime import (
    EgressDeniedError,
    EmbeddingPurpose,
    EmbeddingRequest,
    EmbeddingResult,
    EmbeddingRoute,
    LLMRuntime,
    LLMRuntimeError,
    RouteLocation,
)
from chronovisor.core.runtime_config import SearchEmbeddingConfig
from chronovisor.core.semantic_index import SemanticDocument, SemanticIndexError
from chronovisor.search import semantic_service
from chronovisor.search.semantic_model import SemanticModelError
from chronovisor.search.semantic_service import (
    DOCUMENT_SOURCE,
    FOREGROUND_ROLE,
    INCREMENTAL_ROLE,
    QUERY_SOURCE,
    QueryBatcher,
    SemanticServiceState,
    ServiceBusy,
    _drifted_page_ids,
    _Handler,
    _safe_service_error,
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


class CanaryRuntimeError(LLMRuntimeError):
    category = "credential=secret"


def test_drifted_page_ids_are_deduplicated_across_states() -> None:
    assert _drifted_page_ids(
        {
            "missing_page_ids": ["new", "shared"],
            "stale_page_ids": ["changed", "shared"],
            "deleted_page_ids": ["gone"],
        }
    ) == ["changed", "gone", "new", "shared"]


def test_service_errors_expose_only_safe_categories() -> None:
    assert _safe_service_error(EgressDeniedError("role", "embedding")) == (
        "egress_denied"
    )
    assert _safe_service_error(
        SemanticModelError("secret prompt / private/path")
    ) == "model_unavailable"
    assert _safe_service_error(ValueError("credential=secret")) == "semantic_failure"

    assert _safe_service_error(CanaryRuntimeError("private/path")) == (
        "semantic_failure"
    )


def test_unix_handler_does_not_serialize_exception_details() -> None:
    errors: list[bool] = []
    handler = object.__new__(_Handler)
    handler.rfile = io.BytesIO(b'{"method":"search","query":"secret"}\n')
    handler.wfile = io.BytesIO()
    handler.server = SimpleNamespace(
        state=SimpleNamespace(
            handle=lambda _payload: (_ for _ in ()).throw(
                CanaryRuntimeError("prompt=/private/path credential=secret")
            ),
            note_error=lambda: errors.append(True),
        )
    )

    handler.handle()

    payload = handler.wfile.getvalue().decode()
    assert errors == [True]
    assert '"error":"semantic_failure"' in payload
    assert "private/path" not in payload
    assert "credential" not in payload


def _state(runtime: LLMRuntime) -> SemanticServiceState:
    state = object.__new__(SemanticServiceState)
    state.config = SearchEmbeddingConfig(dimensions=2)
    state._runtime = runtime
    state._model_lock = threading.Lock()
    state._validate_runtime_routes()
    return state


def _forbid_local_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("remote semantic route touched a local control")

    monkeypatch.setattr(semantic_service, "accelerator_lease", forbidden)
    monkeypatch.setattr(semantic_service, "model_activity", forbidden)
    for name in (
        "embed",
        "is_available",
        "model_digests",
        "model_resource_lease",
        "model_resource_lease_mode",
        "plan_model_residency",
        "resident_model_rows",
        "unload_model",
        "unload_named_model",
    ):
        monkeypatch.setattr(ollama, name, forbidden)


def test_remote_normal_document_uses_fixed_role_without_local_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    state = _state(runtime)
    _forbid_local_controls(monkeypatch)

    vectors = state._embed_foreground_documents(["document"])

    assert vectors.shape == (1, 2)
    assert backend.requests == [
        EmbeddingRequest(
            ("document",),
            DOCUMENT_SOURCE,
            state.config.query_timeout_ms,
            EmbeddingPurpose.DOCUMENT,
        )
    ]


@pytest.mark.parametrize("difference", ["provider", "model", "location"])
def test_semantic_routes_require_exact_compatible_identity(difference: str) -> None:
    foreground = FakeEmbeddingBackend()
    incremental = FakeEmbeddingBackend()
    if difference == "provider":
        incremental.provider = "other"  # type: ignore[misc]
    elif difference == "location":
        incremental.location = RouteLocation.REMOTE  # type: ignore[misc]
    runtime = LLMRuntime(
        embedding={
            FOREGROUND_ROLE: EmbeddingRoute(foreground, "model"),
            INCREMENTAL_ROLE: EmbeddingRoute(
                incremental, "other-model" if difference == "model" else "model"
            ),
        }
    )
    state = object.__new__(SemanticServiceState)
    state.config = SearchEmbeddingConfig()
    state._runtime = runtime

    with pytest.raises(SemanticModelError):
        state._validate_runtime_routes()

    assert foreground.requests == []
    assert incremental.requests == []


def test_semantic_runtime_vectors_pass_role_purpose_and_page_source() -> None:
    backend = FakeEmbeddingBackend()
    state = object.__new__(SemanticServiceState)
    state.config = SearchEmbeddingConfig(dimensions=2)
    state._runtime = LLMRuntime(
        embedding={FOREGROUND_ROLE: EmbeddingRoute(backend, "model")}
    )

    vectors = state._runtime_vectors(
        FOREGROUND_ROLE,
        ["document"],
        EmbeddingPurpose.DOCUMENT,
        source=DOCUMENT_SOURCE,
    )

    assert vectors.shape == (1, 2)
    assert backend.requests[0].purpose is EmbeddingPurpose.DOCUMENT
    assert backend.requests[0].source == DOCUMENT_SOURCE


def _remote_state(backend: FakeEmbeddingBackend) -> SemanticServiceState:
    backend.location = RouteLocation.REMOTE  # type: ignore[misc]
    return _state(
        LLMRuntime(
            embedding={
                FOREGROUND_ROLE: EmbeddingRoute(backend, "model"),
                INCREMENTAL_ROLE: EmbeddingRoute(backend, "model"),
            }
        )
    )


def test_remote_raw_query_denial_has_no_backend_or_local_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeEmbeddingBackend()
    state = _remote_state(backend)
    _forbid_local_controls(monkeypatch)

    with pytest.raises(EgressDeniedError):
        state._embed_foreground(
            ["query"], EmbeddingPurpose.QUERY, source=QUERY_SOURCE
        )

    assert backend.requests == []


@pytest.mark.parametrize(
    ("data_class", "sensitivity"),
    [("page", "high"), ("system", "high")],
)
def test_remote_sensitive_document_denial_has_no_backend_or_local_control(
    monkeypatch: pytest.MonkeyPatch,
    data_class: str,
    sensitivity: str,
) -> None:
    backend = FakeEmbeddingBackend()
    state = _remote_state(backend)
    _forbid_local_controls(monkeypatch)
    document = SemanticDocument(
        doc_id="doc",
        page_id="page",
        kind="page",
        ordinal=-1,
        text="sensitive",
        source_path="/wiki/page.md",
        source_sha256="a" * 64,
        source_mtime_ns=1,
        source_data_class=data_class,
        source_sensitivity=sensitivity,
    )

    with pytest.raises(EgressDeniedError):
        state._embed_incremental_documents(
            [document.text], source=state._document_source([document])
        )

    assert backend.requests == []


def test_incremental_model_is_released_when_lazy_self_test_fails() -> None:
    released: list[str] = []

    def fail(_texts: list[str]) -> np.ndarray:
        raise SemanticModelError("failed")

    state = object.__new__(SemanticServiceState)
    state._cpu_ready = False
    state._parity_text = "parity"
    state._runtime = SimpleNamespace(
        release_embedding=lambda role: released.append(role)
    )
    state._incremental_route = SimpleNamespace(
        provider="nemotron", location=RouteLocation.LOCAL
    )
    state._embed_incremental_documents = fail

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
    state._query_cache_seal = "route-seal"
    state._query_vector_cache = OrderedDict(
        {"route-seal\0repeated query": (time.monotonic(), vector)}
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


@pytest.mark.parametrize("field", ["role", "provider", "model", "location"])
def test_query_cache_route_identity_drift_is_a_miss(field: str) -> None:
    route = {
        "role": FOREGROUND_ROLE,
        "provider": "test",
        "model": "model",
        "location": "local",
    }
    old_seal = json.dumps(route, sort_keys=True)
    route[field] = "changed"
    state = object.__new__(SemanticServiceState)
    state._query_cache_lock = threading.Lock()
    state._query_cache_seal = json.dumps(route, sort_keys=True)
    state._query_vector_cache = OrderedDict(
        {
            f"{old_seal}\0query": (
                time.monotonic(),
                np.asarray([1.0, 0.0], dtype=np.float32),
            )
        }
    )

    assert state._query_vector_from_cache("query") is None


def test_reload_rejects_old_generation_without_exposing_details(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeEmbeddingBackend()
    state = _state(
        LLMRuntime(
            embedding={
                FOREGROUND_ROLE: EmbeddingRoute(backend, "model"),
                INCREMENTAL_ROLE: EmbeddingRoute(backend, "model"),
            }
        )
    )
    state.root = tmp_path
    state._generation = SimpleNamespace(name="old-generation")
    state._generation_lock = threading.RLock()
    state._maintenance = threading.Event()
    state.health = lambda: {"ready": state._query_available()}  # type: ignore[method-assign]
    monkeypatch.setattr(
        semantic_service,
        "load_active_generation",
        lambda **_kwargs: (_ for _ in ()).throw(
            SemanticIndexError("schema3 /private/path credential=secret")
        ),
    )

    result = state.reload()

    assert result == {"ready": False}
    assert state._generation is None
    assert state._last_error == "generation_invalid"
    with pytest.raises(ServiceBusy):
        state._search_vector(np.asarray([1.0, 0.0], dtype=np.float32), 1)


def test_worker_persists_only_safe_job_failure_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StopAfterOne:
        calls = 0

        def wait(self, _seconds: float) -> bool:
            self.calls += 1
            return self.calls > 1

    state = object.__new__(SemanticServiceState)
    state._stopped = StopAfterOne()
    state.config = SimpleNamespace(incremental_enabled=False)
    state._last_job_prune = time.monotonic()
    state._maintenance = threading.Event()
    state._unload_idle_cpu = lambda: None
    state._reload_if_pointer_changed = lambda: None
    state._publish_status = lambda: None
    state._pause_background_work = lambda: False
    state._rebuild = lambda: (_ for _ in ()).throw(
        CanaryRuntimeError("prompt=/private/path credential=secret")
    )
    job = SimpleNamespace(kind="rebuild", job_id="job")
    failures: list[tuple[str, str]] = []
    monkeypatch.setattr(semantic_service, "claim_next", lambda **_kwargs: job)
    monkeypatch.setattr(
        semantic_service,
        "fail",
        lambda job_id, error: failures.append((job_id, error)),
    )

    state._worker_loop()

    assert failures == [("job", "semantic_failure")]
    assert state._last_error == "semantic_failure"


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
