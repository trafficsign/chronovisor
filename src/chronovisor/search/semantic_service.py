"""Dedicated Nemotron semantic retrieval service.

The service owns exactly one foreground MPS model, micro-batches concurrent
queries, and keeps indexing off the synchronous request path.  Every query
uses one immutable base generation plus its generation-scoped delta.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import json
import math
import os
import queue
import signal
import socketserver
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from chronovisor.core import runtime_status
from chronovisor.core.link_fix import atomic_write
from chronovisor.core.llm_config import load_default_llm_runtime
from chronovisor.core.llm_runtime import (
    SAFE_FAILURE_CATEGORIES,
    EmbeddingPurpose,
    EmbeddingRequest,
    LLMRuntime,
    LLMRuntimeError,
    ResolvedEmbeddingRoute,
    RouteLocation,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)
from chronovisor.core.ollama import model_activity
from chronovisor.core.runtime_config import (
    SearchEmbeddingConfig,
    load_search_embedding_config,
)
from chronovisor.core.semantic_index import (
    SEMANTIC_ROOT,
    LoadedGeneration,
    SemanticDocument,
    SemanticIndexError,
    activate_generation,
    archive_legacy_search_index,
    build_generation,
    extract_all_documents,
    extract_page_documents,
    load_active_generation,
    prune_expired_legacy_archives,
    prune_generations,
    read_active,
    rollback_generation,
    semantic_index_status,
    upgrade_generation_with_ann,
    write_page_delta,
)
from chronovisor.core.semantic_jobs import (
    claim_next,
    complete,
    enqueue_pages,
    enqueue_rebuild,
    fail,
    job_status,
    prune_completed_jobs,
)
from chronovisor.core.store import (
    CHRONOVISOR_ROOT,
    SYSTEM_DIR,
    find_page,
    okf_runtime_operation,
    okf_startup_status,
)
from chronovisor.search.accelerator_lease import accelerator_lease
from chronovisor.search.semantic_model import (
    SemanticModelError,
    semantic_runtime_versions,
)

SERVICE_STATUS_FILE = CHRONOVISOR_ROOT / "runtime" / "semantic-service-status.json"
QUERY_CACHE_TTL_SECONDS = 600.0
FOREGROUND_ROLE = "search.semantic.foreground"
INCREMENTAL_ROLE = "search.semantic.incremental"
QUERY_SOURCE = SourceDataClassification(SourceDataClass.RAW, SourceSensitivity.NORMAL)
DOCUMENT_SOURCE = SourceDataClassification(
    SourceDataClass.PAGE, SourceSensitivity.NORMAL
)
_CURRENT_QUERY_DEADLINE: ContextVar[float | None] = ContextVar(
    "semantic_query_deadline", default=None
)


class ServiceBusy(RuntimeError):
    pass


def _deadline_for(
    timeout_ms: int | float | None,
    default_ms: int | float,
    *,
    deadline_at: float | None = None,
) -> float:
    """Resolve one absolute deadline without reviving expired budgets."""

    now = time.monotonic()
    if deadline_at is not None:
        try:
            deadline = float(deadline_at)
        except (TypeError, ValueError) as exc:
            raise TimeoutError("semantic query deadline exhausted") from exc
        if not math.isfinite(deadline) or deadline <= now:
            raise TimeoutError("semantic query deadline exhausted")
        if timeout_ms is not None:
            try:
                budget = float(timeout_ms)
            except (TypeError, ValueError) as exc:
                raise TimeoutError("semantic query deadline exhausted") from exc
            if not math.isfinite(budget) or budget <= 0:
                raise TimeoutError("semantic query deadline exhausted")
            deadline = min(deadline, now + budget / 1_000)
        return deadline
    raw = default_ms if timeout_ms is None else timeout_ms
    try:
        budget = float(raw)
    except (TypeError, ValueError) as exc:
        raise TimeoutError("semantic query deadline exhausted") from exc
    if not math.isfinite(budget) or budget <= 0:
        raise TimeoutError("semantic query deadline exhausted")
    return now + budget / 1_000


def _ensure_deadline(deadline_at: float) -> None:
    if deadline_at <= time.monotonic():
        raise TimeoutError("semantic query deadline exhausted")


def _remaining_seconds(deadline_at: float) -> float:
    remaining = deadline_at - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("semantic query deadline exhausted")
    return remaining


@contextlib.contextmanager
def _query_deadline_scope(deadline_at: float) -> Any:
    token = _CURRENT_QUERY_DEADLINE.set(deadline_at)
    try:
        yield
    finally:
        _CURRENT_QUERY_DEADLINE.reset(token)


@dataclass
class _QueryItem:
    query: str
    top_n: int
    future: Future[list[tuple[str, float]]]
    deadline_at: float
    enqueued_at: float
    expired_recorded: bool = False


def _safe_service_error(exc: BaseException) -> str:
    if isinstance(exc, LLMRuntimeError):
        category = getattr(exc, "category", "")
        return category if category in SAFE_FAILURE_CATEGORIES else "semantic_failure"
    if isinstance(exc, SemanticIndexError):
        return "generation_invalid"
    if isinstance(exc, SemanticModelError):
        return "model_unavailable"
    return "semantic_failure"


def _drifted_page_ids(status: dict[str, Any]) -> list[str]:
    page_ids: set[str] = set()
    for key in ("missing_page_ids", "stale_page_ids", "deleted_page_ids"):
        values = status.get(key)
        if not isinstance(values, list):
            continue
        page_ids.update(str(value) for value in values if value)
    return sorted(page_ids)


class QueryBatcher:
    def __init__(
        self,
        *,
        encode: Callable[[list[str], int], np.ndarray],
        search: Callable[[np.ndarray, int], list[tuple[str, float]]],
        window_ms: int,
        max_batch: int,
        available: Callable[[], bool],
    ) -> None:
        self._encode = encode
        self._search = search
        self._window_seconds = max(0, window_ms) / 1_000
        self._max_batch = max(1, max_batch)
        self._available = available
        self._queue: queue.Queue[_QueryItem] = queue.Queue(maxsize=64)
        self._metrics_lock = threading.Lock()
        self._queue_wait_ms: deque[float] = deque(maxlen=2_000)
        self._submitted = 0
        self._completed = 0
        self._expired = 0
        self._cancelled = 0
        self._queue_full = 0
        self._active_batch_size = 0
        self._close_lock = threading.Lock()
        self._closed = threading.Event()
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="semantic-query-batcher", daemon=True
        )
        self._thread.start()

    def submit(
        self,
        query: str,
        top_n: int,
        timeout: float | None = None,
        *,
        deadline_at: float | None = None,
    ) -> list[tuple[str, float]]:
        with self._close_lock:
            if self._closed.is_set():
                raise ServiceBusy("semantic foreground queue is closed")
            started = time.monotonic()
            if deadline_at is None:
                try:
                    budget_seconds = 5.0 if timeout is None else float(timeout)
                except (TypeError, ValueError) as exc:
                    raise TimeoutError(
                        "semantic query deadline exhausted"
                    ) from exc
                if not math.isfinite(budget_seconds) or budget_seconds <= 0:
                    raise TimeoutError("semantic query deadline exhausted")
                deadline = started + budget_seconds
            else:
                deadline = _deadline_for(None, 5_000, deadline_at=deadline_at)
                if timeout is not None:
                    try:
                        budget_seconds = float(timeout)
                    except (TypeError, ValueError) as exc:
                        raise TimeoutError(
                            "semantic query deadline exhausted"
                    ) from exc
                    if not math.isfinite(budget_seconds) or budget_seconds <= 0:
                        raise TimeoutError("semantic query deadline exhausted")
                    deadline = min(deadline, started + budget_seconds)
            future: Future[list[tuple[str, float]]] = Future()
            item = _QueryItem(query, top_n, future, deadline, started)
            with self._metrics_lock:
                self._submitted += 1
            try:
                self._queue.put_nowait(item)
            except queue.Full as exc:
                with self._metrics_lock:
                    self._queue_full += 1
                raise ServiceBusy("semantic foreground queue is full") from exc
        try:
            return future.result(timeout=_remaining_seconds(deadline))
        except TimeoutError:
            if future.done():
                raise
            cancelled = future.cancel()
            self._record_expired(item, cancelled=cancelled)
            raise TimeoutError("semantic query deadline exhausted") from None

    def close(self) -> None:
        with self._close_lock:
            if not self._closed.is_set():
                self._closed.set()
                self._stopped.set()
                while True:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    self._set_expired(item)
                    self._queue.task_done()
        self._thread.join(timeout=2)

    def metrics(self) -> dict[str, Any]:
        with self._metrics_lock:
            waits = list(self._queue_wait_ms)
            submitted = self._submitted
            completed = self._completed
            expired = self._expired
            cancelled = self._cancelled
            queue_full = self._queue_full
            active_batch_size = self._active_batch_size
        p50 = float(np.percentile(waits, 50)) if waits else 0.0
        p95 = float(np.percentile(waits, 95)) if waits else 0.0
        return {
            "queue_depth": self._queue.qsize(),
            "active_batch_size": active_batch_size,
            "queue_wait_samples": len(waits),
            "queue_wait_p50_ms": round(p50, 3),
            "queue_wait_p95_ms": round(p95, 3),
            "submitted": submitted,
            "completed": completed,
            "expired": expired,
            "cancelled": cancelled,
            "queue_full": queue_full,
        }

    def _record_expired(self, item: _QueryItem, *, cancelled: bool) -> None:
        with self._metrics_lock:
            if item.expired_recorded:
                return
            item.expired_recorded = True
            self._expired += 1
            if cancelled:
                self._cancelled += 1

    def _record_queue_wait(self, item: _QueryItem) -> None:
        wait_ms = max(0.0, (time.monotonic() - item.enqueued_at) * 1_000)
        with self._metrics_lock:
            self._queue_wait_ms.append(wait_ms)

    @staticmethod
    def _expired_or_cancelled(item: _QueryItem) -> bool:
        if item.future.cancelled():
            return True
        return item.deadline_at <= time.monotonic()

    def _set_expired(self, item: _QueryItem) -> None:
        self._record_expired(item, cancelled=item.future.cancelled())
        if not item.future.done():
            with contextlib.suppress(Exception):
                item.future.set_exception(
                    TimeoutError("semantic query deadline exhausted")
                )

    def _set_failure(self, item: _QueryItem, exc: BaseException) -> None:
        if item.future.done():
            if item.future.cancelled():
                self._record_expired(item, cancelled=True)
            return
        with contextlib.suppress(Exception):
            item.future.set_exception(exc)

    def _set_result(
        self, item: _QueryItem, result: list[tuple[str, float]]
    ) -> None:
        if self._expired_or_cancelled(item):
            self._set_expired(item)
            return
        if item.future.done():
            if item.future.cancelled():
                self._record_expired(item, cancelled=True)
            return
        try:
            item.future.set_result(result)
        except Exception:
            if item.future.cancelled():
                self._record_expired(item, cancelled=True)
            return
        with self._metrics_lock:
            self._completed += 1

    def _run(self) -> None:
        while not self._stopped.is_set():
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.monotonic() + self._window_seconds
            while len(batch) < self._max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break
            runnable: list[_QueryItem] = []
            try:
                if self._stopped.is_set():
                    for item in batch:
                        self._set_expired(item)
                    continue
                for item in batch:
                    self._record_queue_wait(item)
                for item in batch:
                    if self._expired_or_cancelled(item):
                        self._set_expired(item)
                        continue
                    if not item.future.set_running_or_notify_cancel():
                        self._set_expired(item)
                        continue
                    if self._expired_or_cancelled(item):
                        self._set_expired(item)
                        continue
                    runnable.append(item)
                if not runnable:
                    continue
                if not self._available():
                    error = ServiceBusy("semantic generation unavailable or rebuilding")
                    for item in runnable:
                        if self._expired_or_cancelled(item):
                            self._set_expired(item)
                        else:
                            self._set_failure(item, error)
                    continue
                live_runnable: list[_QueryItem] = []
                for item in runnable:
                    if self._stopped.is_set() or self._expired_or_cancelled(item):
                        self._set_expired(item)
                        continue
                    live_runnable.append(item)
                runnable = live_runnable
                if not runnable:
                    continue
                if self._stopped.is_set():
                    for item in runnable:
                        self._set_expired(item)
                    continue
                with self._metrics_lock:
                    self._active_batch_size = len(runnable)
                encode_deadline = max(item.deadline_at for item in runnable)
                with _query_deadline_scope(encode_deadline):
                    vectors = self._encode(
                        [item.query for item in runnable], self._max_batch
                    )
                for vector, item in zip(vectors, runnable, strict=False):
                    if self._expired_or_cancelled(item):
                        self._set_expired(item)
                        continue
                    try:
                        with _query_deadline_scope(item.deadline_at):
                            result = self._search(vector, item.top_n)
                        self._set_result(item, result)
                    except TimeoutError as exc:
                        self._set_expired(item)
                        self._set_failure(item, exc)
                    except BaseException as exc:
                        self._set_failure(item, exc)
            except TimeoutError as exc:
                for item in runnable:
                    self._set_expired(item)
                    self._set_failure(item, exc)
            except BaseException as exc:
                for item in runnable:
                    self._set_failure(item, exc)
            finally:
                with self._metrics_lock:
                    self._active_batch_size = 0
                for _item in batch:
                    self._queue.task_done()


def _ingest_is_active() -> bool:
    llm = runtime_status.read_status().get("llm")
    if not isinstance(llm, dict) or llm.get("active") is not True:
        return False
    lock_path = CHRONOVISOR_ROOT / "runtime" / "ingest-orchestrator.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = lock_path.open("a+")
    except OSError:
        return False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


def _set_background_qos() -> None:
    """Best-effort macOS per-thread QoS for incremental CPU inference."""

    try:
        pthread = ctypes.CDLL(None)
        function = pthread.pthread_set_qos_class_self_np
        function.argtypes = [ctypes.c_uint, ctypes.c_int]
        function.restype = ctypes.c_int
        function(0x09, 0)  # QOS_CLASS_BACKGROUND
    except Exception:
        pass


class SemanticServiceState:
    def __init__(
        self,
        config: SearchEmbeddingConfig,
        *,
        runtime: LLMRuntime | None = None,
        root: Path = SEMANTIC_ROOT,
    ) -> None:
        self.config = config
        self.root = root
        self._runtime = runtime or load_default_llm_runtime()
        self._validate_runtime_routes()
        self._model_lock = threading.Lock()
        self._generation_lock = threading.RLock()
        self._maintenance = threading.Event()
        self._stopped = threading.Event()
        self._self_test = self._self_test_foreground()
        self._parity_text = "Chronovisorの意味検索インデックス整合性テスト"
        self._parity_reference = self._embed_foreground_documents([self._parity_text])[
            0
        ]
        self._generation: LoadedGeneration | None = None
        self._last_error = ""
        self._cpu_ready = False
        self._cpu_last_used = 0.0
        self._last_job_prune = 0.0
        self._last_drift_scan = 0.0
        self._metrics_lock = threading.Lock()
        self._query_cache_lock = threading.Lock()
        self._query_vector_cache: OrderedDict[str, tuple[float, np.ndarray]] = (
            OrderedDict()
        )
        self._query_latencies_ms: deque[float] = deque(maxlen=2_000)
        self._query_errors = 0
        self._query_attempts = 0
        self._query_successes = 0
        self._query_timeouts = 0
        self._query_failures = 0
        self._late_reload_count = 0
        self._late_reload_ms = 0.0
        self._active_signature: tuple[int, int] | None = None
        self._status_lock = threading.Lock()
        self._last_status_publish = 0.0
        self._query_path_self_test: dict[str, object] = {}
        self.reload()
        self._query_path_self_test = self._warm_query_path()
        self._batcher = QueryBatcher(
            encode=self._encode_queries,
            search=self._search_vector,
            window_ms=config.foreground_batch_window_ms,
            max_batch=config.foreground_max_batch,
            available=self._query_available,
        )
        self._worker = threading.Thread(
            target=self._worker_loop, name="semantic-index-worker", daemon=True
        )
        self._worker.start()
        self._publish_status(force=True)

    def close(self) -> None:
        self._stopped.set()
        self._batcher.close()
        self._worker.join(timeout=2)
        if self._cpu_ready and self._uses_local_controls(self._incremental_route):
            self._runtime.release_embedding(INCREMENTAL_ROLE)
        if self._uses_local_controls(self._foreground_route):
            self._runtime.release_embedding(FOREGROUND_ROLE)

    @staticmethod
    def _route_identity(route: ResolvedEmbeddingRoute) -> dict[str, str]:
        return {
            "role": route.role,
            "provider": route.provider,
            "model": route.model,
            "location": route.location.value,
        }

    @staticmethod
    def _uses_local_controls(route: ResolvedEmbeddingRoute) -> bool:
        return route.provider == "nemotron" and route.location is RouteLocation.LOCAL

    def _validate_runtime_routes(self) -> None:
        foreground = self._runtime.resolve_embedding(FOREGROUND_ROLE)
        incremental = self._runtime.resolve_embedding(INCREMENTAL_ROLE)
        if (
            foreground.provider != incremental.provider
            or foreground.model != incremental.model
            or foreground.location is not incremental.location
        ):
            raise SemanticModelError(
                "semantic roles require the same provider, model, and location"
            )
        self._foreground_route = foreground
        self._incremental_route = incremental
        seal = json.dumps(self._route_identity(foreground), sort_keys=True)
        if getattr(self, "_query_cache_seal", seal) != seal and hasattr(
            self, "_query_vector_cache"
        ):
            with self._query_cache_lock:
                self._query_vector_cache.clear()
        self._query_cache_seal = seal

    def _runtime_vectors(
        self,
        role: str,
        texts: list[str],
        purpose: EmbeddingPurpose,
        *,
        source: SourceDataClassification,
        timeout_ms: int | None = None,
    ) -> np.ndarray:
        result = self._runtime.embed(
            role,
            EmbeddingRequest(
                tuple(texts),
                source,
                timeout_ms,
                purpose,
            ),
        )
        matrix = np.asarray(result.vectors, dtype=np.float32)
        if matrix.shape != (len(texts), self.config.dimensions):
            raise SemanticModelError(
                "semantic runtime dimensions do not match the configured index"
            )
        return np.ascontiguousarray(matrix, dtype=np.float32)

    def _embed_foreground(
        self,
        texts: list[str],
        purpose: EmbeddingPurpose,
        *,
        source: SourceDataClassification,
        timeout_ms: int | float | None = None,
        deadline_at: float | None = None,
    ) -> np.ndarray:
        context_deadline = _CURRENT_QUERY_DEADLINE.get()
        if deadline_at is None:
            deadline_at = context_deadline
        if deadline_at is None:
            deadline_at = _deadline_for(timeout_ms, self.config.query_timeout_ms)
        elif timeout_ms is not None:
            local_deadline = _deadline_for(timeout_ms, self.config.query_timeout_ms)
            deadline_at = min(deadline_at, local_deadline)
        _ensure_deadline(deadline_at)
        lock_timeout = _remaining_seconds(deadline_at)
        acquired = self._model_lock.acquire(timeout=lock_timeout)
        if not acquired:
            raise TimeoutError("semantic query deadline exhausted")
        try:
            remaining_ms = max(1, math.ceil(_remaining_seconds(deadline_at) * 1_000))
            resources = (
                accelerator_lease(timeout_ms=remaining_ms)
                if self._uses_local_controls(self._foreground_route)
                else contextlib.nullcontext()
            )
            activity = (
                model_activity(
                    model=self._foreground_route.model,
                    operation="search",
                    pipeline="recall",
                )
                if self._uses_local_controls(self._foreground_route)
                else contextlib.nullcontext()
            )
            with resources:
                _ensure_deadline(deadline_at)
                remaining_ms = max(
                    1, math.ceil(_remaining_seconds(deadline_at) * 1_000)
                )
                with activity:
                    _ensure_deadline(deadline_at)
                    remaining_ms = max(
                        1, math.ceil(_remaining_seconds(deadline_at) * 1_000)
                    )
                    result = self._runtime_vectors(
                        FOREGROUND_ROLE,
                        texts,
                        purpose,
                        source=source,
                        timeout_ms=remaining_ms,
                    )
            _ensure_deadline(deadline_at)
            return result
        finally:
            self._model_lock.release()

    def _embed_foreground_documents(
        self,
        texts: list[str],
        *,
        source: SourceDataClassification = DOCUMENT_SOURCE,
    ) -> np.ndarray:
        return self._embed_foreground(
            texts,
            EmbeddingPurpose.DOCUMENT,
            source=source,
            timeout_ms=self.config.interactive_timeout_ms,
        )

    def _embed_incremental_documents(
        self,
        texts: list[str],
        *,
        source: SourceDataClassification = DOCUMENT_SOURCE,
    ) -> np.ndarray:
        activity = (
            model_activity(
                model=self._incremental_route.model,
                operation="generate",
                pipeline="improve",
            )
            if self._uses_local_controls(self._incremental_route)
            else contextlib.nullcontext()
        )
        with activity:
            return self._runtime_vectors(
                INCREMENTAL_ROLE,
                texts,
                EmbeddingPurpose.DOCUMENT,
                source=source,
            )

    @staticmethod
    def _document_source(
        documents: Sequence[SemanticDocument],
    ) -> SourceDataClassification:
        if any(document.source_data_class == "system" for document in documents):
            return SourceDataClassification(
                SourceDataClass.SYSTEM, SourceSensitivity.HIGH
            )
        sensitivity = (
            SourceSensitivity.NORMAL
            if documents
            and all(
                document.source_sensitivity == "normal" for document in documents
            )
            else SourceSensitivity.HIGH
        )
        return SourceDataClassification(SourceDataClass.PAGE, sensitivity)

    def _self_test_foreground(self) -> dict[str, object]:
        query = self._embed_foreground(
            ["Chronovisorの検索インデックス"],
            EmbeddingPurpose.QUERY,
            source=QUERY_SOURCE,
            timeout_ms=self.config.interactive_timeout_ms,
        )[0]
        documents = self._embed_foreground(
            [
                "ChronovisorはローカルAI向けの記憶検索システムです。",
                "夕食のレシピと材料についてのメモです。",
            ],
            EmbeddingPurpose.DOCUMENT,
            source=DOCUMENT_SOURCE,
            timeout_ms=self.config.interactive_timeout_ms,
        )
        scores = documents @ query
        if not float(scores[0]) > float(scores[1]):
            raise SemanticModelError("known-vector ranking self-test failed")
        return {
            "device": self.config.query_device,
            "dimensions": int(query.shape[0]),
            "positive_score": float(scores[0]),
            "negative_score": float(scores[1]),
        }

    def _query_available(self) -> bool:
        return self._generation is not None and not self._maintenance.is_set()

    def _warm_query_path(self) -> dict[str, object]:
        """Exercise model and ANN search before the service advertises ready."""

        started = time.monotonic()
        queries = [
            "Chronovisorの記憶検索",
            "関連ページを思い出す",
            "semantic recall warmup",
        ]
        vectors = self._encode_queries(
            queries,
            len(queries),
            deadline_at=_deadline_for(None, self.config.interactive_timeout_ms),
        )
        hits = sum(bool(self._search_vector(vector, 1)) for vector in vectors)
        if hits != len(queries):
            raise ServiceBusy("semantic query-path warmup returned no result")
        return {
            "queries": len(queries),
            "hits": hits,
            "latency_ms": round((time.monotonic() - started) * 1_000, 3),
        }

    def reload(
        self,
        *,
        verify_checksums: bool = True,
        deadline_at: float | None = None,
    ) -> dict[str, Any]:
        if deadline_at is None:
            deadline_at = _CURRENT_QUERY_DEADLINE.get()
        if deadline_at is not None:
            _ensure_deadline(deadline_at)
        try:
            self._validate_runtime_routes()
            generation = load_active_generation(
                root=self.root,
                verify_checksums=verify_checksums,
                expected_route=self._route_identity(self._foreground_route),
            )
            manifest = generation.manifest
            if (
                manifest.revision != self.config.revision
                or manifest.dimensions != self.config.dimensions
            ):
                raise SemanticIndexError(
                    "active generation does not match the configured model profile"
                )
            if deadline_at is None:
                with self._generation_lock:
                    self._generation = generation
            elif self._generation_lock.acquire(
                timeout=_remaining_seconds(deadline_at)
            ):
                try:
                    self._generation = generation
                finally:
                    self._generation_lock.release()
            else:
                raise TimeoutError("semantic query deadline exhausted")
            self._last_error = ""
        except (SemanticIndexError, SemanticModelError, LLMRuntimeError) as exc:
            if deadline_at is None:
                with self._generation_lock:
                    self._generation = None
            elif self._generation_lock.acquire(
                timeout=_remaining_seconds(deadline_at)
            ):
                try:
                    self._generation = None
                finally:
                    self._generation_lock.release()
            else:
                raise TimeoutError("semantic query deadline exhausted") from exc
            self._last_error = _safe_service_error(exc)
        if deadline_at is not None:
            _ensure_deadline(deadline_at)
        active_path = self.root / "active.json"
        try:
            stat = active_path.stat()
            self._active_signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            self._active_signature = None
        result = self.health()
        if deadline_at is not None:
            _ensure_deadline(deadline_at)
        return result

    def _reload_if_pointer_changed(self, *, deadline_at: float | None = None) -> None:
        if deadline_at is None:
            deadline_at = _CURRENT_QUERY_DEADLINE.get()
        if deadline_at is not None:
            _ensure_deadline(deadline_at)
        active_path = self.root / "active.json"
        try:
            stat = active_path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            signature = None
        if signature != self._active_signature:
            if deadline_at is not None:
                _ensure_deadline(deadline_at)
            try:
                if deadline_at is not None:
                    with _query_deadline_scope(deadline_at):
                        self.reload()
                else:
                    self.reload()
            finally:
                if deadline_at is not None:
                    late_ms = max(0.0, (time.monotonic() - deadline_at) * 1_000)
                    if late_ms > 0:
                        self._record_late_reload(late_ms)
            if deadline_at is not None:
                _ensure_deadline(deadline_at)

    def health(self) -> dict[str, Any]:
        generation = self._generation
        with self._metrics_lock:
            latencies = list(self._query_latencies_ms)
            errors = self._query_errors
            attempts = getattr(self, "_query_attempts", 0)
            successes = getattr(self, "_query_successes", 0)
            timeouts = getattr(self, "_query_timeouts", 0)
            failures = getattr(self, "_query_failures", 0)
            late_reload_count = getattr(self, "_late_reload_count", 0)
            late_reload_ms = getattr(self, "_late_reload_ms", 0.0)
        batcher = getattr(self, "_batcher", None)
        batcher_metrics = (
            batcher.metrics()
            if batcher is not None
            else {
                "queue_depth": 0,
                "active_batch_size": 0,
                "queue_wait_samples": 0,
                "queue_wait_p50_ms": 0.0,
                "queue_wait_p95_ms": 0.0,
                "submitted": 0,
                "completed": 0,
                "expired": 0,
                "cancelled": 0,
                "queue_full": 0,
            }
        )
        p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        return {
            "status": "ok",
            "pid": os.getpid(),
            "ready": self._query_available(),
            "maintenance": self._maintenance.is_set(),
            "generation_id": (
                generation.manifest.generation_id if generation is not None else ""
            ),
            "model": self._foreground_route.model,
            "routes": {
                FOREGROUND_ROLE: self._route_identity(self._foreground_route),
                INCREMENTAL_ROLE: self._route_identity(self._incremental_route),
            },
            "revision": self.config.revision,
            "device": self.config.query_device,
            "self_test": self._self_test,
            "query_path_self_test": self._query_path_self_test,
            "runtime_versions": semantic_runtime_versions(),
            "index": semantic_index_status(
                root=self.root,
                expected_route=self._route_identity(self._foreground_route),
            ),
            "jobs": job_status(),
            "last_error": self._last_error,
            "batcher": batcher_metrics,
            "queries": {
                "samples": len(latencies),
                "errors": errors,
                "attempts": attempts,
                "successes": successes,
                "timeouts": timeouts,
                "failures": failures,
                "late_reload_count": late_reload_count,
                "late_reload_ms": round(late_reload_ms, 3),
                "p50_ms": round(p50, 3),
                "p95_ms": round(p95, 3),
                "max_ms": round(max(latencies), 3) if latencies else 0.0,
            },
        }

    def _publish_status(self, *, force: bool = False) -> None:
        now = time.monotonic()
        with self._status_lock:
            if not force and now - self._last_status_publish < 5.0:
                return
            payload = {
                **self.health(),
                "observed_at_epoch": time.time(),
                "pid": os.getpid(),
            }
            try:
                atomic_write(
                    SERVICE_STATUS_FILE,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
                )
                self._last_status_publish = now
            except OSError:
                pass

    def note_error(self) -> None:
        with self._metrics_lock:
            self._query_errors += 1

    def _bump_query_metric(self, name: str) -> None:
        with self._metrics_lock:
            setattr(self, name, getattr(self, name, 0) + 1)

    def _record_late_reload(self, late_ms: float) -> None:
        with self._metrics_lock:
            self._late_reload_count = getattr(self, "_late_reload_count", 0) + 1
            self._late_reload_ms = getattr(self, "_late_reload_ms", 0.0) + late_ms

    def _encode_queries(
        self,
        queries: list[str],
        _batch_size: int,
        *,
        deadline_at: float | None = None,
    ) -> np.ndarray:
        if deadline_at is None:
            deadline_at = _CURRENT_QUERY_DEADLINE.get()
        if deadline_at is not None:
            _ensure_deadline(deadline_at)
        vectors = self._embed_foreground(
            queries,
            EmbeddingPurpose.QUERY,
            source=QUERY_SOURCE,
            deadline_at=deadline_at,
        )
        if deadline_at is not None:
            _ensure_deadline(deadline_at)
        now = time.monotonic()
        if deadline_at is None:
            self._query_cache_lock.acquire()
        elif not self._query_cache_lock.acquire(
            timeout=_remaining_seconds(deadline_at)
        ):
            raise TimeoutError("semantic query deadline exhausted")
        try:
            for query, vector in zip(queries, vectors, strict=False):
                key = self._query_cache_key(query)
                self._query_vector_cache[key] = (
                    now,
                    np.ascontiguousarray(vector, dtype=np.float32),
                )
                self._query_vector_cache.move_to_end(key)
            while len(self._query_vector_cache) > 64:
                self._query_vector_cache.popitem(last=False)
        finally:
            self._query_cache_lock.release()
        return vectors

    def _query_vector_from_cache(
        self, query: str, *, deadline_at: float | None = None
    ) -> np.ndarray | None:
        if deadline_at is None:
            deadline_at = _CURRENT_QUERY_DEADLINE.get()
        now = time.monotonic()
        key = self._query_cache_key(query)
        if deadline_at is None:
            self._query_cache_lock.acquire()
        elif not self._query_cache_lock.acquire(
            timeout=_remaining_seconds(deadline_at)
        ):
            raise TimeoutError("semantic query deadline exhausted")
        try:
            cached = self._query_vector_cache.get(key)
            if cached is not None and now - cached[0] <= QUERY_CACHE_TTL_SECONDS:
                self._query_vector_cache.move_to_end(key)
                return cached[1]
            if cached is not None:
                self._query_vector_cache.pop(key, None)
        finally:
            self._query_cache_lock.release()
        return None

    def _query_cache_key(self, query: str) -> str:
        return f"{self._query_cache_seal}\0{query}"

    def _cached_query_vector(
        self, query: str, *, deadline_at: float | None = None
    ) -> tuple[np.ndarray, bool]:
        inherited_deadline = deadline_at is None
        if inherited_deadline:
            deadline_at = _CURRENT_QUERY_DEADLINE.get()
        cached = self._query_vector_from_cache(query)
        if cached is not None:
            if deadline_at is not None:
                _ensure_deadline(deadline_at)
            return cached, True
        if inherited_deadline:
            return self._encode_queries([query], 1)[0], False
        return self._encode_queries([query], 1, deadline_at=deadline_at)[0], False

    def _search_vector(
        self,
        vector: np.ndarray,
        top_n: int,
        *,
        deadline_at: float | None = None,
    ) -> list[tuple[str, float]]:
        if deadline_at is None:
            deadline_at = _CURRENT_QUERY_DEADLINE.get()
        if deadline_at is not None:
            _ensure_deadline(deadline_at)
        if deadline_at is None:
            self._generation_lock.acquire()
        else:
            if not self._generation_lock.acquire(
                timeout=_remaining_seconds(deadline_at)
            ):
                raise TimeoutError("semantic query deadline exhausted")
        try:
            generation = self._generation
        finally:
            self._generation_lock.release()
        if generation is None:
            raise ServiceBusy("no active semantic generation")
        result = generation.search(cast(Sequence[float], vector), top_n=top_n)
        if deadline_at is not None:
            _ensure_deadline(deadline_at)
        return result

    def search(
        self,
        query: str,
        top_n: int,
        *,
        timeout_ms: int | float | None = None,
        deadline_at: float | None = None,
    ) -> dict[str, Any]:
        self._bump_query_metric("_query_attempts")
        started = time.monotonic()
        try:
            deadline = _deadline_for(
                timeout_ms,
                self.config.interactive_timeout_ms,
                deadline_at=deadline_at,
            )
            with _query_deadline_scope(deadline):
                self._reload_if_pointer_changed()
            _ensure_deadline(deadline)
            bounded_top_n = max(1, min(100, top_n))
            with _query_deadline_scope(deadline):
                cached = self._query_vector_from_cache(query)
            cache_hit = cached is not None
            if cached is None:
                results = self._batcher.submit(
                    query,
                    bounded_top_n,
                    _remaining_seconds(deadline),
                    deadline_at=deadline,
                )
            else:
                with _query_deadline_scope(deadline):
                    results = self._search_vector(cached, bounded_top_n)
            _ensure_deadline(deadline)
            generation = self._generation
            latency_ms = (time.monotonic() - started) * 1_000
            with self._metrics_lock:
                self._query_latencies_ms.append(latency_ms)
            self._bump_query_metric("_query_successes")
            return {
                "status": "ok",
                "generation_id": (
                    generation.manifest.generation_id if generation is not None else ""
                ),
                "cache_hit": cache_hit,
                "latency_ms": round(latency_ms, 3),
                "results": [
                    {"page_id": page_id, "score": score} for page_id, score in results
                ],
            }
        except TimeoutError:
            self._bump_query_metric("_query_timeouts")
            raise
        except BaseException:
            self._bump_query_metric("_query_failures")
            raise

    def verify(
        self,
        query: str,
        page_ids: list[str],
        *,
        timeout_ms: int | float | None = None,
        deadline_at: float | None = None,
    ) -> dict[str, Any]:
        """Exactly verify graph candidates without a second model inference."""

        self._bump_query_metric("_query_attempts")
        try:
            deadline = _deadline_for(
                timeout_ms,
                self.config.interactive_timeout_ms,
                deadline_at=deadline_at,
            )
            with _query_deadline_scope(deadline):
                self._reload_if_pointer_changed()
            _ensure_deadline(deadline)
            unique = list(dict.fromkeys(page_id for page_id in page_ids if page_id))[:100]
            if not unique:
                self._bump_query_metric("_query_successes")
                return {"status": "ok", "cache_hit": False, "results": []}
            with _query_deadline_scope(deadline):
                vector, cache_hit = self._cached_query_vector(query)
                if not self._generation_lock.acquire(
                    timeout=_remaining_seconds(deadline)
                ):
                    raise TimeoutError("semantic query deadline exhausted")
                try:
                    generation = self._generation
                finally:
                    self._generation_lock.release()
                if generation is None:
                    raise ServiceBusy("no active semantic generation")
                _ensure_deadline(deadline)
                rows = generation.score_pages(cast(Sequence[float], vector), unique)
            _ensure_deadline(deadline)
            self._bump_query_metric("_query_successes")
            return {
                "status": "ok",
                "generation_id": generation.manifest.generation_id,
                "cache_hit": cache_hit,
                "results": [
                    {"page_id": page_id, "score": score} for page_id, score in rows
                ],
            }
        except TimeoutError:
            self._bump_query_metric("_query_timeouts")
            raise
        except BaseException:
            self._bump_query_metric("_query_failures")
            raise

    def enqueue_rebuild(self) -> dict[str, Any]:
        return {"status": "ok", "job_id": enqueue_rebuild()}

    def index_pages(self, page_ids: list[str], *, wait: bool) -> dict[str, Any]:
        unique = sorted({page_id for page_id in page_ids if page_id})
        if not wait:
            return {"status": "ok", "job_ids": enqueue_pages(unique)}
        if self._generation is None:
            raise ServiceBusy("full semantic rebuild is required first")
        updated = 0
        for page_id in unique:
            self._index_page(page_id, expected_hash="", foreground=True)
            updated += 1
        return {"status": "ok", "pages_updated": updated}

    def rollback(self) -> dict[str, Any]:
        pointer = rollback_generation(root=self.root)
        self.reload()
        return {"status": "ok", "active": pointer}

    def _pause_background_work(self) -> bool:
        if self.config.incremental_pause_during_research:
            from chronovisor.core import research_scheduler

            if (
                research_scheduler.sync_pending()
                or research_scheduler.ACTIVE_FILE.exists()
            ):
                return True
        return bool(
            self.config.incremental_pause_during_ingest_generation
            and _ingest_is_active()
        )

    def _ensure_cpu(self) -> None:
        if not self._cpu_ready:
            try:
                cpu_vector = self._embed_incremental_documents([self._parity_text])[0]
                parity = float(cpu_vector @ self._parity_reference)
                if parity < 0.999:
                    raise RuntimeError(
                        f"CPU/MPS semantic vector parity failed: cosine={parity:.6f}"
                    )
            except Exception:
                if self._uses_local_controls(self._incremental_route):
                    self._runtime.release_embedding(INCREMENTAL_ROLE)
                self._cpu_ready = False
                raise
            self._cpu_ready = True
        self._cpu_last_used = time.monotonic()

    def _unload_idle_cpu(self) -> None:
        if (
            self._cpu_ready
            and time.monotonic() - self._cpu_last_used
            >= self.config.incremental_idle_unload_seconds
        ):
            if self._uses_local_controls(self._incremental_route):
                self._runtime.release_embedding(INCREMENTAL_ROLE)
            self._cpu_ready = False

    def _index_page(
        self, page_id: str, *, expected_hash: str, foreground: bool = False
    ) -> None:
        generation = self._generation
        if generation is None:
            raise ServiceBusy("no active semantic generation")
        path = find_page(page_id)
        if path is None:
            system_path = SYSTEM_DIR / f"{page_id}.md"
            path = system_path if system_path.is_file() else None
        documents = extract_page_documents(path) if path is not None else []
        current_hash = documents[0].source_sha256 if documents else ""
        if expected_hash and current_hash != expected_hash:
            enqueue_pages([page_id], source_hashes={page_id: current_hash})
            return
        if documents:
            texts = [document.text for document in documents]
            source = self._document_source(documents)
            if foreground:
                try:
                    vectors = self._embed_foreground_documents(texts, source=source)
                except Exception:
                    self._ensure_cpu()
                    vectors = self._embed_incremental_documents(texts, source=source)
            else:
                self._ensure_cpu()
                vectors = self._embed_incremental_documents(texts, source=source)
            refreshed = extract_page_documents(path) if path is not None else []
            refreshed_hash = refreshed[0].source_sha256 if refreshed else ""
            if refreshed_hash != current_hash:
                enqueue_pages([page_id], source_hashes={page_id: refreshed_hash})
                return
        else:
            vectors = None
        write_page_delta(
            generation.manifest.generation_id,
            page_id,
            documents,
            vectors,
            dimensions=self.config.dimensions,
            root=self.root,
        )
        self.reload(verify_checksums=False)

    def _rebuild(self) -> None:
        self._maintenance.set()
        try:
            self._validate_runtime_routes()
            documents = extract_all_documents()
            built_hashes = {
                document.page_id: document.source_sha256 for document in documents
            }
            with self._model_lock:
                activity = (
                    model_activity(
                        model=self._foreground_route.model,
                        operation="generate",
                        pipeline="improve",
                    )
                    if self._uses_local_controls(self._foreground_route)
                    else contextlib.nullcontext()
                )
                with activity:
                    manifest = build_generation(
                        documents,
                        encode_documents=lambda rows, _batch_size: (
                            self._runtime_vectors(
                                FOREGROUND_ROLE,
                                [document.text for document in rows],
                                EmbeddingPurpose.DOCUMENT,
                                source=self._document_source(rows),
                            )
                        ),
                        **self._route_identity(self._foreground_route),
                        revision=self.config.revision,
                        dimensions=self.config.dimensions,
                        query_prefix=self.config.query_prefix,
                        document_prefix=self.config.document_prefix,
                        batch_size=self.config.maintenance_max_batch,
                        root=self.root,
                    )
            current = str(read_active(root=self.root).get("generation_id") or "")
            activate_generation(
                manifest.generation_id,
                expected_current=current,
                root=self.root,
            )
            prune_generations(root=self.root)
            self.reload()
            # Close the mutation window between the corpus snapshot and active
            # pointer publication. New, changed, and deleted pages become
            # generation-scoped delta jobs.
            current_documents = extract_all_documents()
            current_hashes = {
                document.page_id: document.source_sha256
                for document in current_documents
            }
            changed = {
                page_id
                for page_id in set(built_hashes) | set(current_hashes)
                if built_hashes.get(page_id) != current_hashes.get(page_id)
            }
            if changed:
                enqueue_pages(
                    changed,
                    source_hashes={
                        page_id: current_hashes.get(page_id, "") for page_id in changed
                    },
                )
        finally:
            self._maintenance.clear()

    def _worker_loop(self) -> None:
        _set_background_qos()
        priority_page_ids: tuple[str, ...] = ()
        while not self._stopped.wait(1.0):
            self._unload_idle_cpu()
            self._reload_if_pointer_changed()
            self._publish_status()
            if (
                self.config.incremental_enabled
                and time.monotonic() - self._last_drift_scan >= 60
            ):
                status = semantic_index_status(
                    root=self.root,
                    expected_route=self._route_identity(self._foreground_route),
                )
                drifted = _drifted_page_ids(status)
                if drifted:
                    enqueue_pages(drifted)
                priority_page_ids = tuple(drifted)
                self._last_drift_scan = time.monotonic()
            if time.monotonic() - self._last_job_prune >= 3_600:
                prune_completed_jobs()
                prune_expired_legacy_archives()
                self._last_job_prune = time.monotonic()
            if self._pause_background_work() or self._maintenance.is_set():
                continue
            kinds = (
                ("page", "rebuild") if self.config.incremental_enabled else ("rebuild",)
            )
            job = None
            if priority_page_ids:
                job = claim_next(kinds=kinds, page_ids=priority_page_ids)
                if job is None:
                    priority_page_ids = ()
            if job is None:
                job = claim_next(kinds=kinds)
            if job is None:
                continue
            try:
                if job.kind == "rebuild":
                    self._rebuild()
                else:
                    self._index_page(job.page_id, expected_hash=job.source_sha256)
                complete(job.job_id)
                self._last_error = ""
            except Exception as exc:
                self._last_error = _safe_service_error(exc)
                fail(job.job_id, self._last_error)

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        method = payload.get("method")
        if method == "health":
            self._reload_if_pointer_changed()
            return self.health()
        if method == "search":
            query = str(payload.get("query") or "").strip()
            if not query:
                raise ValueError("query is required")
            raw_timeout = payload.get("timeout_ms")
            timeout_ms = None if raw_timeout is None else int(raw_timeout)
            raw_deadline = payload.get("deadline_at")
            deadline_at = None if raw_deadline is None else float(raw_deadline)
            return self.search(
                query,
                int(payload.get("top_n") or 20),
                timeout_ms=timeout_ms,
                deadline_at=deadline_at,
            )
        if method == "verify":
            query = str(payload.get("query") or "").strip()
            raw_ids = payload.get("page_ids")
            if not query:
                raise ValueError("query is required")
            if not isinstance(raw_ids, list):
                raise ValueError("page_ids must be a list")
            raw_timeout = payload.get("timeout_ms")
            timeout_ms = None if raw_timeout is None else int(raw_timeout)
            raw_deadline = payload.get("deadline_at")
            deadline_at = None if raw_deadline is None else float(raw_deadline)
            return self.verify(
                query,
                [str(item) for item in raw_ids],
                timeout_ms=timeout_ms,
                deadline_at=deadline_at,
            )
        if method == "reload":
            return self.reload()
        if method == "rebuild":
            return self.enqueue_rebuild()
        if method == "rollback":
            return self.rollback()
        if method == "index_pages":
            raw_ids = payload.get("page_ids")
            if not isinstance(raw_ids, list):
                raise ValueError("page_ids must be a list")
            return self.index_pages(
                [str(item) for item in raw_ids],
                wait=bool(payload.get("wait")),
            )
        raise ValueError(f"unknown semantic service method: {method!r}")


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        response: dict[str, Any]
        try:
            raw = self.rfile.readline(1024 * 1024)
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("request must be an object")
            response = self.server.state.handle(payload)  # type: ignore[attr-defined]
        except Exception as exc:
            self.server.state.note_error()  # type: ignore[attr-defined]
            response = {
                "status": "error",
                "error": _safe_service_error(exc),
            }
        with contextlib.suppress(OSError):
            self.wfile.write(
                json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode()
                + b"\n"
            )


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(config: SearchEmbeddingConfig | None = None) -> None:
    with okf_runtime_operation(CHRONOVISOR_ROOT):
        _serve_locked(config)


def _serve_locked(config: SearchEmbeddingConfig | None) -> None:
    if not okf_startup_status(CHRONOVISOR_ROOT).allowed:
        raise SystemExit(75)
    config = config or load_search_embedding_config()
    if not config.enabled:
        raise SystemExit("semantic service is disabled in config")
    socket_path = Path(config.socket).expanduser()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    state = SemanticServiceState(config)
    server = _Server(str(socket_path), _Handler)
    server.state = state  # type: ignore[attr-defined]
    os.chmod(socket_path, 0o600)

    def stop(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        state.close()
        socket_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-semantic-service`` command-line entry point."""
    parser = argparse.ArgumentParser(prog="chronovisor-semantic-service")
    parser.add_argument(
        "command",
        choices=(
            "serve",
            "status",
            "rebuild",
            "rollback",
            "upgrade-ann",
            "archive-legacy",
        ),
        nargs="?",
        default="serve",
    )
    args = parser.parse_args(argv)
    if args.command == "status":
        return _main_locked(args)
    from chronovisor.core.okf_cutover import OKFStartupBlocked
    try:
        with okf_runtime_operation(CHRONOVISOR_ROOT):
            return _main_locked(args)
    except OKFStartupBlocked:
        print(json.dumps({"status": "blocked", "category": "okf_startup_blocked"}))
        return 75


def _main_locked(args: argparse.Namespace) -> int:
    if args.command != "status" and not okf_startup_status(CHRONOVISOR_ROOT).allowed:
        print(
            json.dumps({"status": "blocked", "category": "okf_startup_blocked"})
        )
        return 75
    config = load_search_embedding_config()
    if args.command == "serve":
        serve(config)
        return 0
    from chronovisor.core import semantic_client

    if args.command == "archive-legacy":
        result = archive_legacy_search_index()
    elif args.command == "upgrade-ann":
        active = read_active()
        current = str(active.get("generation_id") or "")
        if not current:
            raise SystemExit("no active semantic generation")
        manifest = upgrade_generation_with_ann(current)
        result = activate_generation(
            manifest.generation_id,
            expected_current=current,
        )
    elif args.command == "status":
        result = semantic_client.health(config)
    elif args.command == "rebuild":
        result = semantic_client.request_rebuild(config)
    else:
        result = semantic_client.request_rollback(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
