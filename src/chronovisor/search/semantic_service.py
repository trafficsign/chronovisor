"""Dedicated Nemotron semantic retrieval service.

The service owns exactly one foreground MPS model, micro-batches concurrent
queries, and keeps indexing off the synchronous request path.  Every query
uses one immutable base generation plus its generation-scoped delta.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
import queue
import signal
import socketserver
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable

import numpy as np

from chronovisor.core.link_fix import atomic_write
from chronovisor.core.runtime_config import (
    SearchEmbeddingConfig,
    load_search_embedding_config,
)
from chronovisor.search.semantic_index import (
    SEMANTIC_ROOT,
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
from chronovisor.search.semantic_jobs import (
    claim_next,
    complete,
    enqueue_pages,
    enqueue_rebuild,
    fail,
    job_status,
    prune_completed_jobs,
)
from chronovisor.search.semantic_model import (
    NemotronEncoder,
    semantic_runtime_versions,
)
from chronovisor.core.store import CHRONOVISOR_ROOT, SYSTEM_DIR, find_page

SERVICE_STATUS_FILE = CHRONOVISOR_ROOT / "runtime" / "semantic-service-status.json"
QUERY_CACHE_TTL_SECONDS = 600.0


class ServiceBusy(RuntimeError):
    pass


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
        self._queue: queue.Queue[tuple[str, int, Future]] = queue.Queue(maxsize=64)
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="semantic-query-batcher", daemon=True
        )
        self._thread.start()

    def submit(self, query: str, top_n: int, timeout: float) -> list[tuple[str, float]]:
        future: Future = Future()
        try:
            self._queue.put_nowait((query, top_n, future))
        except queue.Full as exc:
            raise ServiceBusy("semantic foreground queue is full") from exc
        return future.result(timeout=timeout)

    def close(self) -> None:
        self._stopped.set()
        self._thread.join(timeout=2)

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
            if not self._available():
                error = ServiceBusy("semantic generation unavailable or rebuilding")
                for _query, _top_n, future in batch:
                    future.set_exception(error)
                continue
            try:
                vectors = self._encode([item[0] for item in batch], self._max_batch)
                for vector, (_query, top_n, future) in zip(vectors, batch):
                    future.set_result(self._search(vector, top_n))
            except BaseException as exc:
                for _query, _top_n, future in batch:
                    if not future.done():
                        future.set_exception(exc)


def _ingest_is_active() -> bool:
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
        encoder_factory: Callable[..., NemotronEncoder] = NemotronEncoder,
        root: Path = SEMANTIC_ROOT,
    ) -> None:
        self.config = config
        self.root = root
        self._encoder_factory = encoder_factory
        self._model_lock = threading.Lock()
        self._generation_lock = threading.RLock()
        self._maintenance = threading.Event()
        self._stopped = threading.Event()
        self._foreground = encoder_factory(config, device=config.query_device)
        self._self_test = self._foreground.self_test()
        self._parity_text = "Chronovisorの意味検索インデックス整合性テスト"
        self._parity_reference = self._foreground.encode_documents(
            [self._parity_text], 1
        )[0]
        self._generation = None
        self._last_error = ""
        self._cpu_encoder: NemotronEncoder | None = None
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
        if self._cpu_encoder is not None:
            self._cpu_encoder.close()
        self._foreground.close()

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
        vectors = self._encode_queries(queries, len(queries))
        hits = sum(bool(self._search_vector(vector, 1)) for vector in vectors)
        if hits != len(queries):
            raise ServiceBusy("semantic query-path warmup returned no result")
        return {
            "queries": len(queries),
            "hits": hits,
            "latency_ms": round((time.monotonic() - started) * 1_000, 3),
        }

    def reload(self, *, verify_checksums: bool = True) -> dict[str, Any]:
        try:
            generation = load_active_generation(
                root=self.root, verify_checksums=verify_checksums
            )
            manifest = generation.manifest
            if (
                manifest.model != self.config.model
                or manifest.revision != self.config.revision
                or manifest.dimensions != self.config.dimensions
            ):
                raise SemanticIndexError(
                    "active generation does not match the configured model profile"
                )
            with self._generation_lock:
                self._generation = generation
            self._last_error = ""
        except SemanticIndexError as exc:
            with self._generation_lock:
                self._generation = None
            self._last_error = str(exc)
        active_path = self.root / "active.json"
        try:
            stat = active_path.stat()
            self._active_signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            self._active_signature = None
        return self.health()

    def _reload_if_pointer_changed(self) -> None:
        active_path = self.root / "active.json"
        try:
            stat = active_path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            signature = None
        if signature != self._active_signature:
            self.reload()

    def health(self) -> dict[str, Any]:
        generation = self._generation
        with self._metrics_lock:
            latencies = list(self._query_latencies_ms)
            errors = self._query_errors
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
            "model": self.config.model,
            "revision": self.config.revision,
            "device": self.config.query_device,
            "self_test": self._self_test,
            "query_path_self_test": self._query_path_self_test,
            "runtime_versions": semantic_runtime_versions(),
            "index": semantic_index_status(root=self.root),
            "jobs": job_status(),
            "last_error": self._last_error,
            "queries": {
                "samples": len(latencies),
                "errors": errors,
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

    def _encode_queries(self, queries: list[str], batch_size: int) -> np.ndarray:
        with self._model_lock:
            vectors = self._foreground.encode_queries(queries, batch_size)
        now = time.monotonic()
        with self._query_cache_lock:
            for query, vector in zip(queries, vectors):
                self._query_vector_cache[query] = (
                    now,
                    np.ascontiguousarray(vector, dtype=np.float32),
                )
                self._query_vector_cache.move_to_end(query)
            while len(self._query_vector_cache) > 64:
                self._query_vector_cache.popitem(last=False)
        return vectors

    def _query_vector_from_cache(self, query: str) -> np.ndarray | None:
        now = time.monotonic()
        with self._query_cache_lock:
            cached = self._query_vector_cache.get(query)
            if (
                cached is not None
                and now - cached[0] <= QUERY_CACHE_TTL_SECONDS
            ):
                self._query_vector_cache.move_to_end(query)
                return cached[1]
            if cached is not None:
                self._query_vector_cache.pop(query, None)
        return None

    def _cached_query_vector(self, query: str) -> tuple[np.ndarray, bool]:
        cached = self._query_vector_from_cache(query)
        if cached is not None:
            return cached, True
        return self._encode_queries([query], 1)[0], False

    def _search_vector(self, vector: np.ndarray, top_n: int) -> list[tuple[str, float]]:
        with self._generation_lock:
            generation = self._generation
            if generation is None:
                raise ServiceBusy("no active semantic generation")
            return generation.search(vector, top_n=top_n)

    def search(
        self, query: str, top_n: int, *, timeout_ms: int | None = None
    ) -> dict[str, Any]:
        self._reload_if_pointer_changed()
        started = time.monotonic()
        bounded_top_n = max(1, min(100, top_n))
        cached = self._query_vector_from_cache(query)
        cache_hit = cached is not None
        if cached is None:
            results = self._batcher.submit(
                query,
                bounded_top_n,
                max(
                    0.025,
                    float(timeout_ms or self.config.interactive_timeout_ms) / 1_000,
                ),
            )
        else:
            results = self._search_vector(cached, bounded_top_n)
        generation = self._generation
        latency_ms = (time.monotonic() - started) * 1_000
        with self._metrics_lock:
            self._query_latencies_ms.append(latency_ms)
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

    def verify(self, query: str, page_ids: list[str]) -> dict[str, Any]:
        """Exactly verify graph candidates without a second model inference."""

        self._reload_if_pointer_changed()
        unique = list(dict.fromkeys(page_id for page_id in page_ids if page_id))[:100]
        if not unique:
            return {"status": "ok", "cache_hit": False, "results": []}
        vector, cache_hit = self._cached_query_vector(query)
        with self._generation_lock:
            generation = self._generation
            if generation is None:
                raise ServiceBusy("no active semantic generation")
            rows = generation.score_pages(vector, unique)
        return {
            "status": "ok",
            "generation_id": generation.manifest.generation_id,
            "cache_hit": cache_hit,
            "results": [
                {"page_id": page_id, "score": score} for page_id, score in rows
            ],
        }

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
            self._index_page(page_id, expected_hash="")
            updated += 1
        return {"status": "ok", "pages_updated": updated}

    def rollback(self) -> dict[str, Any]:
        pointer = rollback_generation(root=self.root)
        self.reload()
        return {"status": "ok", "active": pointer}

    def _pause_background_work(self) -> bool:
        if self.config.incremental_pause_during_research:
            from chronovisor import research_scheduler

            if (
                research_scheduler.sync_pending()
                or research_scheduler.ACTIVE_FILE.exists()
            ):
                return True
        if (
            self.config.incremental_pause_during_ingest_generation
            and _ingest_is_active()
        ):
            return True
        return False

    def _cpu(self) -> NemotronEncoder:
        if self._cpu_encoder is None:
            candidate = self._encoder_factory(
                self.config, device=self.config.incremental_device
            )
            candidate.self_test()
            cpu_vector = candidate.encode_documents([self._parity_text], 1)[0]
            parity = float(cpu_vector @ self._parity_reference)
            if parity < 0.999:
                candidate.close()
                raise RuntimeError(
                    f"CPU/MPS semantic vector parity failed: cosine={parity:.6f}"
                )
            self._cpu_encoder = candidate
        self._cpu_last_used = time.monotonic()
        return self._cpu_encoder

    def _unload_idle_cpu(self) -> None:
        if (
            self._cpu_encoder is not None
            and time.monotonic() - self._cpu_last_used
            >= self.config.incremental_idle_unload_seconds
        ):
            self._cpu_encoder.close()
            self._cpu_encoder = None

    def _index_page(self, page_id: str, *, expected_hash: str) -> None:
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
            encoder = self._cpu()
            vectors = encoder.encode_documents(
                [document.text for document in documents], 1
            )
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
            documents = extract_all_documents()
            built_hashes = {
                document.page_id: document.source_sha256 for document in documents
            }
            with self._model_lock:
                manifest = build_generation(
                    documents,
                    encode_documents=self._foreground.encode_documents,
                    model=self.config.model,
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
        while not self._stopped.wait(1.0):
            self._unload_idle_cpu()
            self._reload_if_pointer_changed()
            self._publish_status()
            if (
                self.config.incremental_enabled
                and time.monotonic() - self._last_drift_scan >= 60
            ):
                status = semantic_index_status(root=self.root)
                drifted = _drifted_page_ids(status)
                if drifted:
                    enqueue_pages(drifted)
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
                self._last_error = f"{type(exc).__name__}: {exc}"
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
            return self.search(
                query,
                int(payload.get("top_n") or 20),
                timeout_ms=int(payload.get("timeout_ms") or 0) or None,
            )
        if method == "verify":
            query = str(payload.get("query") or "").strip()
            raw_ids = payload.get("page_ids")
            if not query:
                raise ValueError("query is required")
            if not isinstance(raw_ids, list):
                raise ValueError("page_ids must be a list")
            return self.verify(query, [str(item) for item in raw_ids])
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
                "error": f"{type(exc).__name__}: {exc}",
            }
        try:
            self.wfile.write(
                json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode()
                + b"\n"
            )
        except OSError:
            pass


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(config: SearchEmbeddingConfig | None = None) -> None:
    config = config or load_search_embedding_config()
    if not config.enabled or config.backend != "nemotron_service":
        raise SystemExit("Nemotron semantic service is disabled in config")
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


def main() -> None:
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
    args = parser.parse_args()
    config = load_search_embedding_config()
    if args.command == "serve":
        serve(config)
        return
    from chronovisor import semantic_client

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


if __name__ == "__main__":
    main()
