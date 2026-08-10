"""Resident single-worker service for provider-neutral reranking."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import socketserver
import statistics
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any

from chronovisor.core import index_store, llm_config
from chronovisor.core.link_fix import atomic_write
from chronovisor.core.llm_runtime import (
    SAFE_FAILURE_CATEGORIES,
    LLMRuntime,
    RerankRequest,
    RouteLocation,
    SourceDataClassification,
)
from chronovisor.core.ollama import model_activity
from chronovisor.core.reranker import (
    QUERY_SOURCE,
    RERANK_RUNTIME_ROLE,
    resolve_rerank_candidate,
    safe_reranker_error,
    warm_reranker,
)
from chronovisor.core.runtime_config import (
    RerankerConfig,
    load_reranker_config,
    runtime_identity,
)
from chronovisor.core.store import CHRONOVISOR_ROOT, okf_startup_status
from chronovisor.search.accelerator_lease import accelerator_lease

SERVICE_STATUS_FILE = CHRONOVISOR_ROOT / "runtime" / "reranker-service-status.json"
PASSAGE_CACHE_SIZE = 512


class ServiceBusy(RuntimeError):
    pass


_SERVICE_FAILURE_CATEGORIES = SAFE_FAILURE_CATEGORIES | {
    "reranker_unavailable",
    "service_busy",
    "service_unavailable",
}


def _safe_category(value: object) -> str:
    return (
        value
        if isinstance(value, str) and value in _SERVICE_FAILURE_CATEGORIES
        else "reranker_unavailable"
    )


def _safe_service_error(exc: Exception) -> str:
    return (
        "service_busy"
        if isinstance(exc, ServiceBusy)
        else _safe_category(safe_reranker_error(exc))
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * quantile))
    return float(ordered[index])


class RerankerServiceState:
    def __init__(
        self,
        config: RerankerConfig,
        *,
        llm_runtime: LLMRuntime | None = None,
    ) -> None:
        self.config = config
        self._llm_runtime = llm_runtime or llm_config.load_default_llm_runtime()
        self._route = self._llm_runtime.resolve_rerank(RERANK_RUNTIME_ROLE)
        self._store = index_store.get_store()
        self._work_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(config.service.queue_size)
        self._passages: OrderedDict[
            str,
            tuple[
                tuple[str, str, int, int, str],
                str,
                SourceDataClassification,
            ],
        ] = OrderedDict()
        self._latencies: deque[float] = deque(maxlen=2_000)
        self._requests = 0
        self._errors = 0
        self._last_error = ""
        self._runtime_identity = runtime_identity()
        self._warm = self._warm_route()
        self._publish_status()

    @property
    def ready(self) -> bool:
        return self._warm.get("status") == "ready"

    def _local_resources(self) -> contextlib.ExitStack:
        stack = contextlib.ExitStack()
        if (
            self._route.provider == "local-reranker"
            and self._route.location is RouteLocation.LOCAL
        ):
            stack.enter_context(
                accelerator_lease(timeout_ms=self.config.service.timeout_ms)
            )
            stack.enter_context(
                model_activity(
                    model=self._route.model,
                    operation="rerank",
                    pipeline="recall",
                )
            )
        return stack

    def _warm_route(self) -> dict[str, Any]:
        with self._local_resources():
            return warm_reranker(self.config, self._llm_runtime)

    def _candidate_passage(
        self, page_id: str, *, store: index_store.IndexStore | None
    ) -> tuple[str, SourceDataClassification, str]:
        passage, source, identity = resolve_rerank_candidate(page_id, store=store)
        cached = self._passages.get(page_id)
        if cached is not None and cached[0] == identity:
            self._passages.move_to_end(page_id)
            return cached[1], cached[2], cached[0][-1]
        self._passages[page_id] = (identity, passage, source)
        self._passages.move_to_end(page_id)
        while len(self._passages) > PASSAGE_CACHE_SIZE:
            self._passages.popitem(last=False)
        return passage, source, identity[-1]

    def _status_payload(self) -> dict[str, Any]:
        latencies = list(self._latencies)
        return {
            "status": "ok" if self.ready else "error",
            "ready": self.ready,
            "pid": os.getpid(),
            "observed_at_epoch": time.time(),
            "route": {
                "role": self._route.role,
                "provider": self._route.provider,
                "model": self._route.model,
                "location": self._route.location.value,
            },
            "socket": str(Path(self.config.service.socket).expanduser()),
            "warmup": self._warm,
            "requests": {
                "samples": len(latencies),
                "total": self._requests,
                "errors": self._errors,
                "p50_ms": round(statistics.median(latencies), 3)
                if latencies
                else 0.0,
                "p95_ms": round(_percentile(latencies, 0.95), 3),
                "max_ms": round(max(latencies), 3) if latencies else 0.0,
            },
            "passage_cache": {
                "entries": len(self._passages),
                "capacity": PASSAGE_CACHE_SIZE,
            },
            "last_error": self._last_error,
            "runtime": self._runtime_identity,
        }

    def _publish_status(self) -> None:
        SERVICE_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(
            SERVICE_STATUS_FILE,
            json.dumps(
                self._status_payload(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    def note_error(self, exc: Exception) -> None:
        self._errors += 1
        self._last_error = _safe_service_error(exc)
        self._publish_status()

    def _rerank(self, query: str, page_ids: list[str]) -> dict[str, Any]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not page_ids:
            raise ValueError("page_ids must not be empty")
        if len(page_ids) > max(1, self.config.top_n):
            raise ValueError("page_ids exceeds configured reranker top_n")
        if len(set(page_ids)) != len(page_ids):
            raise ValueError("page_ids must be unique")
        if not self._slots.acquire(blocking=False):
            raise ServiceBusy("reranker queue is full")
        started = time.perf_counter()
        try:
            with self._work_lock:
                try:
                    self._store.refresh()
                    candidate_store: index_store.IndexStore | None = self._store
                except Exception:
                    candidate_store = None
                passages_and_hashes = [
                    self._candidate_passage(page_id, store=candidate_store)
                    for page_id in page_ids
                ]
                with self._local_resources():
                    result = self._llm_runtime.rerank(
                        RERANK_RUNTIME_ROLE,
                        RerankRequest(
                            query,
                            tuple(
                                passage
                                for passage, _source, _digest in passages_and_hashes
                            ),
                            QUERY_SOURCE,
                            timeout_ms=self.config.service.timeout_ms,
                            candidate_sources=tuple(
                                source
                                for _passage, source, _digest in passages_and_hashes
                            ),
                        ),
                    )
        finally:
            self._slots.release()
        raw_scores = {
            item.index: float(item.score)
            for item in result.items
        }
        elapsed_ms = (time.perf_counter() - started) * 1_000
        self._latencies.append(elapsed_ms)
        self._requests += 1
        self._last_error = ""
        self._publish_status()
        return {
            "status": "ok",
            "method": "rerank",
            "scores": [
                {
                    "page_id": page_id,
                    "raw_score": float(score),
                    "content_sha256": digest,
                }
                for page_id, score, (_passage, _source, digest) in zip(
                    page_ids,
                    (raw_scores[index] for index in range(len(page_ids))),
                    passages_and_hashes,
                    strict=True,
                )
            ],
            "latency_ms": int(round(elapsed_ms)),
            "route": {
                "role": self._route.role,
                "provider": self._route.provider,
                "model": self._route.model,
                "location": self._route.location.value,
            },
        }

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        method = str(payload.get("method") or "")
        if method == "health":
            return {"status": "ok", **self._status_payload()}
        if method == "warm":
            self._warm = self._warm_route()
            self._publish_status()
            if self._warm.get("status") == "ready":
                return {"status": "ok", "warmup": self._warm}
            return {
                "status": "unavailable",
                "error": _safe_category(self._warm.get("reason")),
                "warmup": self._warm,
            }
        if method == "rerank":
            query = payload.get("query")
            page_ids = payload.get("page_ids")
            if not isinstance(query, str) or not isinstance(page_ids, list):
                raise ValueError("rerank requires query and page_ids")
            normalized_ids = [
                page_id for page_id in page_ids if isinstance(page_id, str) and page_id
            ]
            if len(normalized_ids) != len(page_ids):
                raise ValueError("page_ids must contain non-empty strings")
            return self._rerank(query, normalized_ids)
        raise ValueError(f"unknown reranker method: {method}")


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            raw = self.rfile.readline(1024 * 1024)
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("request must be an object")
            response = self.server.state.handle(payload)  # type: ignore[attr-defined]
        except Exception as exc:
            self.server.state.note_error(exc)  # type: ignore[attr-defined]
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


def serve(config: RerankerConfig | None = None) -> None:
    if not okf_startup_status(CHRONOVISOR_ROOT).allowed:
        raise SystemExit(75)
    config = config or load_reranker_config()
    if not config.enabled:
        raise SystemExit("reranker is disabled in config")
    socket_path = Path(config.service.socket).expanduser()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    state = RerankerServiceState(config)
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
        socket_path.unlink(missing_ok=True)


def _read_status() -> dict[str, Any]:
    try:
        payload = json.loads(SERVICE_STATUS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "unavailable", "error": "service_unavailable"}
    return payload if isinstance(payload, dict) else {"status": "unavailable"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chronovisor reranker service")
    parser.add_argument(
        "command", nargs="?", choices=("serve", "status", "health", "warm"), default="serve"
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the resident reranker service command-line entry point."""

    args = build_parser().parse_args(argv)
    if args.command not in {"status", "health"} and not okf_startup_status(
        CHRONOVISOR_ROOT
    ).allowed:
        print(
            json.dumps({"status": "blocked", "category": "okf_startup_blocked"})
        )
        return 75
    if args.command == "serve":
        serve()
        return 0
    if args.command == "status":
        payload = _read_status()
    else:
        config = load_reranker_config()
        from chronovisor.core import reranker_client

        try:
            payload = (
                reranker_client.health(config)
                if args.command == "health"
                else reranker_client.warm(config)
            )
        except Exception as exc:
            payload = {
                "status": "unavailable",
                "error": _safe_service_error(exc),
            }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"reranker_service\t{payload.get('status', 'unknown')}")
    return 0 if payload.get("status") in {"ok", "ready"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
