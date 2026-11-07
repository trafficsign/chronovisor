"""Resident single-worker service for provider-neutral reranking."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import socketserver
import statistics
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from chronovisor.core import index_store, llm_config
from chronovisor.core.decision_mlx import RECALL_FILTER_PURPOSE, DecisionMLXBackend

# isort: split
from chronovisor.core.decision_mlx import (
    ANSWER_AXES,
    ANSWER_AXIS_QUESTIONS,
    JUDGE_ANSWER_AXES,
    JUDGE_CERTIFICATE,
    JUDGE_QUESTIONS,
    JUDGE_ROLE_BY_KIND,
    JUDGE_ROLES,
    JUDGE_UTILITY,
    DecisionRoute,
    answer_axis_scores,
    certificate_verdict,
    decision_fingerprint,
    judge_state,
    utility_choice,
)
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
from chronovisor.core.search_types import (
    SemanticEvidence,
    parse_semantic_evidence,
)
from chronovisor.core.store import (
    CHRONOVISOR_ROOT,
    okf_runtime_operation,
    okf_startup_status,
)
from chronovisor.search.accelerator_lease import accelerator_lease

SERVICE_STATUS_FILE = CHRONOVISOR_ROOT / "runtime" / "reranker-service-status.json"
PASSAGE_CACHE_SIZE = 512
# A recall turn publishes at most a few items; this only bounds one filter request's work.
RECALL_FILTER_MAX_TEXTS = 16
# Lower number wins the next candidate turn.
PRIORITY_RECALL = 0
PRIORITY_SEARCH = 1


def _candidate_evidence_from_payload(
    payload: dict[str, Any], page_ids: list[str]
) -> dict[str, tuple[SemanticEvidence, ...]]:
    """Validate optional per-page semantic identities from a service request."""

    if "evidence" in payload:
        raise ValueError("candidate evidence is invalid")
    if "candidate_evidence" not in payload:
        return {}
    raw = payload["candidate_evidence"]
    if not isinstance(raw, list) or len(raw) > len(page_ids):
        raise ValueError("candidate evidence is invalid")
    allowed_page_ids = set(page_ids)
    parsed: dict[str, tuple[SemanticEvidence, ...]] = {}
    for row in raw:
        if not isinstance(row, dict):
            raise ValueError("candidate evidence is invalid")
        page_id = row.get("page_id")
        values = row.get("evidence")
        if (
            not isinstance(page_id, str)
            or not page_id
            or page_id not in allowed_page_ids
            or page_id in parsed
            or not isinstance(values, list)
            or not values
        ):
            raise ValueError("candidate evidence is invalid")
        try:
            parsed[page_id] = parse_semantic_evidence(values, page_id=page_id)
        except ValueError as exc:
            raise ValueError("candidate evidence is invalid") from exc
    return parsed


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


class _PriorityTurn:
    """Give a waiting recall request the next candidate turn ahead of search."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._busy = False
        self._waiting = {PRIORITY_RECALL: 0, PRIORITY_SEARCH: 0}

    @contextlib.contextmanager
    def slot(self, priority: int) -> Iterator[None]:
        with self._condition:
            self._waiting[priority] += 1
            self._condition.notify_all()
            try:
                while self._busy or any(
                    count for level, count in self._waiting.items() if level < priority
                ):
                    self._condition.wait()
            finally:
                self._waiting[priority] -= 1
            self._busy = True
        try:
            yield
        finally:
            with self._condition:
                self._busy = False
                self._condition.notify_all()


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
        self._decision_fingerprint = self._judge_fingerprint()
        self._store = index_store.get_store()
        self._work_lock = threading.Lock()
        self._turn = _PriorityTurn()
        self._slots = threading.BoundedSemaphore(config.service.queue_size)
        self._passages: OrderedDict[
            str,
            tuple[
                tuple[tuple[str, str, int, int, str], str, tuple[str, ...]],
                str,
                SourceDataClassification,
                str,
            ],
        ] = OrderedDict()
        self._latencies: deque[float] = deque(maxlen=2_000)
        self._requests = 0
        self._errors = 0
        self._last_error = ""
        self._status_file = SERVICE_STATUS_FILE
        self._runtime_identity = runtime_identity()
        self._warm = self._warm_route()
        self._publish_status()

    @property
    def ready(self) -> bool:
        return self._warm.get("status") == "ready"

    def _route_payload(self) -> dict[str, str]:
        return {
            "role": self._route.role,
            "provider": self._route.provider,
            "model": self._route.model,
            "location": self._route.location.value,
        }

    def _record_request(self, started: float) -> float:
        elapsed_ms = (time.perf_counter() - started) * 1_000
        with self._work_lock:
            self._latencies.append(elapsed_ms)
            self._requests += 1
            self._last_error = ""
        self._publish_status()
        return elapsed_ms

    def _local_resources(self) -> contextlib.ExitStack:
        stack = contextlib.ExitStack()
        # Any on-device model shares the accelerator, whatever its provider name.
        if self._route.location is RouteLocation.LOCAL:
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

    def _judge_fingerprint(self) -> str:
        if (
            self._route.provider != DecisionMLXBackend.provider
            or not self._route.revision
        ):
            return ""
        return decision_fingerprint(
            DecisionRoute(
                role=RERANK_RUNTIME_ROLE,
                provider=self._route.provider,
                model=self._route.model,
                location=self._route.location.value,
                revision=self._route.revision,
            )
        )

    def _judge_route(self, role: str) -> dict[str, str]:
        if role not in JUDGE_ROLES:
            raise ValueError("unknown decision judge role")
        route = self._llm_runtime.resolve_rerank(role)
        if (
            route.provider != self._route.provider
            or route.model != self._route.model
            or route.location is not self._route.location
            or route.revision != self._route.revision
            or not self._decision_fingerprint
        ):
            raise ValueError("decision judge route mismatch")
        return {
            "role": role,
            "provider": route.provider,
            "model": route.model,
            "location": route.location.value,
        }

    def _judge_forward(
        self, role: str, question: str, state: str, priority: int
    ) -> float:
        with self._turn.slot(priority), self._local_resources():
            scores = self._llm_runtime.decision_readout(role, question, (state,))
        if len(scores) != 1:
            raise ValueError("decision judge returned the wrong number of scores")
        return float(scores[0])

    def _judge(self, payload: dict[str, Any]) -> dict[str, Any]:
        kind = payload.get("kind")
        if not isinstance(kind, str) or kind not in JUDGE_ROLE_BY_KIND:
            raise ValueError("judge requires a known kind")
        role = payload.get("role")
        if not isinstance(role, str) or role != JUDGE_ROLE_BY_KIND[kind]:
            raise ValueError("judge role does not match kind")
        expected = {
            JUDGE_ANSWER_AXES: {
                "method",
                "kind",
                "role",
                "query",
                "answer",
                "reference",
            },
            JUDGE_CERTIFICATE: {
                "method",
                "kind",
                "role",
                "query",
                "rubric",
                "evidence",
            },
            JUDGE_UTILITY: {
                "method",
                "kind",
                "role",
                "query",
                "context",
                "answer_a",
                "answer_b",
            },
        }[kind]
        allowed = expected | ({"rubric_id"} if kind == JUDGE_CERTIFICATE else set())
        if set(payload) - allowed or not expected <= set(payload):
            raise ValueError("judge request has invalid fields")
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("judge requires a non-empty query")
        route = self._judge_route(role)
        priority = PRIORITY_RECALL if kind == JUDGE_CERTIFICATE else PRIORITY_SEARCH
        if not self._slots.acquire(blocking=False):
            raise ServiceBusy("reranker queue is full")
        started = time.perf_counter()
        body: dict[str, Any]
        try:
            if kind == JUDGE_ANSWER_AXES:
                answer = payload.get("answer")
                reference = payload.get("reference")
                if (
                    not isinstance(answer, str)
                    or not answer
                    or not isinstance(reference, str)
                    or not reference
                ):
                    raise ValueError("answer_axes requires answer and reference")
                state = judge_state(
                    kind, query=query, answer=answer, reference=reference
                )
                axis_logits = tuple(
                    self._judge_forward(
                        role, ANSWER_AXIS_QUESTIONS[axis], state, priority
                    )
                    for axis in ANSWER_AXES
                )
                body = {
                    "scores": answer_axis_scores(axis_logits),
                    "logits": dict(zip(ANSWER_AXES, axis_logits, strict=True)),
                }
            elif kind == JUDGE_CERTIFICATE:
                rubric = payload.get("rubric")
                rubric_id = payload.get("rubric_id", "")
                evidence = payload.get("evidence")
                if (
                    not isinstance(rubric, str)
                    or not rubric
                    or not isinstance(rubric_id, str)
                    or not isinstance(evidence, list)
                    or not 1 <= len(evidence) <= 2
                ):
                    raise ValueError(
                        "certificate requires rubric and 1..2 evidence spans"
                    )
                verdicts: list[dict[str, Any]] = []
                certificate_logits: list[float] = []
                seen: set[str] = set()
                for row in evidence:
                    if not isinstance(row, dict) or set(row) != {"page_id", "span"}:
                        raise ValueError("certificate evidence is invalid")
                    page_id = row.get("page_id")
                    span = row.get("span")
                    if (
                        not isinstance(page_id, str)
                        or not page_id
                        or page_id in seen
                        or not isinstance(span, str)
                        or not span
                    ):
                        raise ValueError("certificate evidence is invalid")
                    seen.add(page_id)
                    state = judge_state(
                        kind,
                        query=query,
                        rubric=rubric,
                        rubric_id=rubric_id,
                        page_id=page_id,
                        evidence=span,
                    )
                    logit = self._judge_forward(
                        role, JUDGE_QUESTIONS[JUDGE_CERTIFICATE], state, priority
                    )
                    certificate_logits.append(logit)
                    verdicts.append(certificate_verdict(page_id, logit))
                body = {
                    "verdicts": verdicts,
                    "logits": certificate_logits,
                }
            else:
                context = payload.get("context")
                answer_a = payload.get("answer_a")
                answer_b = payload.get("answer_b")
                if (
                    not isinstance(context, str)
                    or not isinstance(answer_a, str)
                    or not answer_a
                    or not isinstance(answer_b, str)
                    or not answer_b
                ):
                    raise ValueError("utility requires context, answer_a and answer_b")
                state = judge_state(
                    kind,
                    query=query,
                    context=context,
                    answer_a=answer_a,
                    answer_b=answer_b,
                )
                logit = self._judge_forward(
                    role, JUDGE_QUESTIONS[JUDGE_UTILITY], state, priority
                )
                body = {**utility_choice(logit), "logit": logit}
        finally:
            self._slots.release()
        elapsed_ms = self._record_request(started)
        return {
            "status": "ok",
            "method": "judge",
            "kind": kind,
            "role": role,
            "route": route,
            "decision_fingerprint": self._decision_fingerprint,
            "latency_ms": int(round(elapsed_ms)),
            **body,
        }

    def _score(self, request: RerankRequest, priority: int) -> list[float]:
        """Raw score per candidate, in request order, in priority turn order.

        A decision model is compute-bound (4B: ~0.2 s a candidate, batching does
        not help), so it takes the accelerator lease one candidate at a time:
        the semantic service then waits at most one forward, not ~2 s.  The
        priority turn is taken before the lease so a waiting recall request can
        take the next candidate turn ahead of a multi-candidate search rerank.
        """

        count = len(request.candidates)
        groups = (
            [[index] for index in range(count)]
            if self._route.provider == DecisionMLXBackend.provider
            else [list(range(count))]
        )
        scores = [0.0] * count
        for group in groups:
            part = replace(
                request,
                candidates=tuple(request.candidates[index] for index in group),
                candidate_sources=None
                if request.candidate_sources is None
                else tuple(request.candidate_sources[index] for index in group),
            )
            with self._turn.slot(priority), self._local_resources():
                result = self._llm_runtime.rerank(RERANK_RUNTIME_ROLE, part)
            for item in result.items:
                scores[group[item.index]] = float(item.score)
        return scores

    def _warm_route(self) -> dict[str, Any]:
        with self._local_resources():
            return warm_reranker(self.config, self._llm_runtime)

    def _candidate_passage(
        self,
        page_id: str,
        *,
        store: index_store.IndexStore | None,
        evidence: tuple[SemanticEvidence, ...] = (),
    ) -> tuple[str, SourceDataClassification, str]:
        passage, source, identity = resolve_rerank_candidate(
            page_id,
            store=store,
            evidence=evidence,
        )
        passage_hash = hashlib.sha256(passage.encode("utf-8")).hexdigest()
        evidence_doc_ids = tuple(
            item.doc_id for item in evidence if isinstance(item, SemanticEvidence)
        )
        cache_key = (identity, passage_hash, evidence_doc_ids)
        cached = self._passages.get(page_id)
        if cached is not None and cached[0] == cache_key:
            self._passages.move_to_end(page_id)
            return cached[1], cached[2], cached[3]
        self._passages[page_id] = (cache_key, passage, source, identity[-1])
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
            "device": self.config.device,
            "route": self._route_payload(),
            "socket": str(Path(self.config.service.socket).expanduser()),
            "warmup": self._warm,
            "requests": {
                "samples": len(latencies),
                "total": self._requests,
                "errors": self._errors,
                "p50_ms": round(statistics.median(latencies), 3) if latencies else 0.0,
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
        with self._work_lock:
            self._status_file.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(
                self._status_file,
                json.dumps(
                    self._status_payload(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )

    def note_error(self, exc: Exception) -> None:
        with self._work_lock:
            self._errors += 1
            self._last_error = _safe_service_error(exc)
        self._publish_status()

    def _rerank(
        self,
        query: str,
        page_ids: list[str],
        *,
        evidence_by_page: dict[str, tuple[SemanticEvidence, ...]] | None = None,
    ) -> dict[str, Any]:
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
                    # Snapshot reload, not a corpus walk: a full refresh() costs
                    # ~0.7 s per request and alone consumes most of the budget.
                    self._store.refresh_if_stale()
                    candidate_store: index_store.IndexStore | None = self._store
                except Exception:
                    candidate_store = None
                passages_and_hashes = [
                    self._candidate_passage(
                        page_id,
                        store=candidate_store,
                        evidence=(evidence_by_page or {}).get(page_id, ()),
                    )
                    for page_id in page_ids
                ]
            raw_scores = self._score(
                RerankRequest(
                    query,
                    tuple(passage for passage, _source, _digest in passages_and_hashes),
                    QUERY_SOURCE,
                    timeout_ms=self.config.service.timeout_ms,
                    candidate_sources=tuple(
                        source for _passage, source, _digest in passages_and_hashes
                    ),
                ),
                PRIORITY_SEARCH,
            )
        finally:
            self._slots.release()
        elapsed_ms = self._record_request(started)
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
                    raw_scores,
                    passages_and_hashes,
                    strict=True,
                )
            ],
            "latency_ms": int(round(elapsed_ms)),
            "route": self._route_payload(),
        }

    def _recall_filter(
        self, query: str, context: str, texts: list[str]
    ) -> dict[str, Any]:
        """Score exact publishable recall texts against the user turn (logit P(yes) each, in order)."""

        # The filter question and its threshold belong to a decision model; a
        # relevance reranker would answer a different question silently.
        if self._route.provider != DecisionMLXBackend.provider:
            raise ValueError("recall_filter requires a decision model route")
        if not query.strip():
            raise ValueError("query must not be empty")
        if not texts or len(texts) > RECALL_FILTER_MAX_TEXTS:
            raise ValueError("recall_filter requires 1..RECALL_FILTER_MAX_TEXTS texts")
        if not self._slots.acquire(blocking=False):
            raise ServiceBusy("reranker queue is full")
        started = time.perf_counter()
        try:
            scores = self._score(
                RerankRequest(
                    query,
                    tuple(texts),
                    QUERY_SOURCE,
                    timeout_ms=self.config.service.timeout_ms,
                    purpose=RECALL_FILTER_PURPOSE,
                    context=context,
                ),
                PRIORITY_RECALL,
            )
        finally:
            self._slots.release()
        elapsed_ms = self._record_request(started)
        return {
            "status": "ok",
            "method": "recall_filter",
            "scores": scores,
            "latency_ms": int(round(elapsed_ms)),
            "route": self._route_payload(),
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
            evidence_by_page = _candidate_evidence_from_payload(payload, normalized_ids)
            return self._rerank(
                query,
                normalized_ids,
                evidence_by_page=evidence_by_page,
            )
        if method == "judge":
            return self._judge(payload)
        if method == "recall_filter":
            query, context, texts = (
                payload.get("query"),
                payload.get("context", ""),
                payload.get("texts"),
            )
            if (
                not isinstance(query, str)
                or not isinstance(context, str)
                or not isinstance(texts, list)
                or not all(isinstance(text, str) and text for text in texts)
            ):
                raise ValueError(
                    "recall_filter requires query, context and non-empty texts"
                )
            return self._recall_filter(query, context, texts)
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
    _last_heartbeat = 0.0

    def service_actions(self) -> None:
        # The native server loop also runs while the resident model is idle.
        now = time.monotonic()
        if now - self._last_heartbeat >= 5:
            with contextlib.suppress(OSError):
                self.state._publish_status()  # type: ignore[attr-defined]
            self._last_heartbeat = now


def serve(config: RerankerConfig | None = None) -> None:
    with okf_runtime_operation(CHRONOVISOR_ROOT):
        _serve_locked(config)


def _serve_locked(config: RerankerConfig | None) -> None:
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
    except OSError, json.JSONDecodeError:
        return {"status": "unavailable", "error": "service_unavailable"}
    return payload if isinstance(payload, dict) else {"status": "unavailable"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chronovisor reranker service")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("serve", "status", "health", "warm"),
        default="serve",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the resident reranker service command-line entry point."""

    args = build_parser().parse_args(argv)
    if args.command in {"status", "health"}:
        return _main_locked(args)
    from chronovisor.core.okf_cutover import OKFStartupBlocked

    try:
        with okf_runtime_operation(CHRONOVISOR_ROOT):
            return _main_locked(args)
    except OKFStartupBlocked:
        print(json.dumps({"status": "blocked", "category": "okf_startup_blocked"}))
        return 75


def _main_locked(args: argparse.Namespace) -> int:
    if (
        args.command not in {"status", "health"}
        and not okf_startup_status(CHRONOVISOR_ROOT).allowed
    ):
        print(json.dumps({"status": "blocked", "category": "okf_startup_blocked"}))
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
