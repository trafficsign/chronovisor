"""Fail-open client for the resident BGE reranker service."""

from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

from chronovisor.core.runtime_config import RerankerConfig
from chronovisor.core.search_types import ScoredPage
from chronovisor.search.reranker import (
    RerankOutcome,
    apply_reranker_scores,
)


class RerankerServiceUnavailable(RuntimeError):
    pass


_BREAKER_LOCK = threading.Lock()
_BREAKER_FAILURES = 0
_BREAKER_OPEN_UNTIL = 0.0


def _breaker_before_request() -> None:
    with _BREAKER_LOCK:
        if time.monotonic() < _BREAKER_OPEN_UNTIL:
            raise RerankerServiceUnavailable("reranker circuit breaker is open")


def _breaker_success() -> None:
    global _BREAKER_FAILURES, _BREAKER_OPEN_UNTIL
    with _BREAKER_LOCK:
        _BREAKER_FAILURES = 0
        _BREAKER_OPEN_UNTIL = 0.0


def _breaker_failure() -> None:
    global _BREAKER_FAILURES, _BREAKER_OPEN_UNTIL
    with _BREAKER_LOCK:
        _BREAKER_FAILURES += 1
        if _BREAKER_FAILURES >= 3:
            _BREAKER_OPEN_UNTIL = time.monotonic() + 30.0


def _socket_path(config: RerankerConfig) -> Path:
    return Path(config.service.socket).expanduser()


def selected_for_rollout(query: str, config: RerankerConfig) -> bool:
    service = config.service
    if not config.enabled or not service.enabled:
        return False
    if service.mode in {"shadow", "on"}:
        return True
    if service.mode != "canary" or service.canary_percent <= 0:
        return False
    bucket = (
        int.from_bytes(
            hashlib.blake2s(query.encode("utf-8"), digest_size=4).digest(), "big"
        )
        % 100
    )
    return bucket < service.canary_percent


def request(
    payload: dict[str, Any],
    config: RerankerConfig,
    *,
    timeout_ms: int | None = None,
) -> dict[str, Any]:
    _breaker_before_request()
    timeout = max(
        0.025,
        float(timeout_ms or config.service.timeout_ms) / 1_000,
    )
    path = _socket_path(config)
    if not path.exists():
        raise RerankerServiceUnavailable(f"reranker socket is missing: {path}")
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    try:
        client.connect(str(path))
        client.sendall(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            + b"\n"
        )
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    except (OSError, TimeoutError) as exc:
        _breaker_failure()
        raise RerankerServiceUnavailable(str(exc)) from exc
    finally:
        client.close()
    try:
        response = json.loads(b"".join(chunks).split(b"\n", 1)[0])
    except (json.JSONDecodeError, IndexError) as exc:
        _breaker_failure()
        raise RerankerServiceUnavailable("invalid reranker service response") from exc
    if not isinstance(response, dict):
        _breaker_failure()
        raise RerankerServiceUnavailable("invalid reranker service payload")
    if response.get("status") != "ok":
        _breaker_failure()
        raise RerankerServiceUnavailable(str(response.get("error") or response))
    _breaker_success()
    return response


def rerank(
    query: str,
    candidates: list[ScoredPage],
    *,
    config: RerankerConfig,
    timeout_ms: int | None = None,
) -> RerankOutcome:
    if not candidates:
        return RerankOutcome(
            candidates, {"status": "skipped", "reason": "no_candidates"}
        )
    rerank_n = min(max(1, config.top_n), len(candidates))
    head = candidates[:rerank_n]
    response = request(
        {
            "method": "rerank",
            "query": query,
            "page_ids": [page.page_id for page in head],
        },
        config,
        timeout_ms=timeout_ms,
    )
    score_rows = response.get("scores")
    if not isinstance(score_rows, list):
        raise RerankerServiceUnavailable("reranker service omitted scores")
    raw_by_page: dict[str, float] = {}
    for row in score_rows:
        if not isinstance(row, dict) or not isinstance(row.get("page_id"), str):
            raise RerankerServiceUnavailable("invalid reranker score row")
        try:
            raw_by_page[row["page_id"]] = float(row["raw_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankerServiceUnavailable("invalid reranker raw score") from exc
    try:
        raw_scores = [raw_by_page[page.page_id] for page in head]
    except KeyError as exc:
        raise RerankerServiceUnavailable(
            f"reranker service omitted page: {exc.args[0]}"
        ) from exc
    return apply_reranker_scores(
        candidates,
        raw_scores,
        config=config,
        metadata={
            "execution": "service",
            "latency_ms": int(response.get("latency_ms") or 0),
            "revision": str(response.get("revision") or ""),
        },
    )


def health(config: RerankerConfig) -> dict[str, Any]:
    return request({"method": "health"}, config)


def warm(config: RerankerConfig) -> dict[str, Any]:
    return request({"method": "warm"}, config)
