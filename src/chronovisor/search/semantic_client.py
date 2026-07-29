"""Fail-open client for the local Nemotron semantic service."""

from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

from chronovisor.core.runtime_config import SearchEmbeddingConfig
from chronovisor.search.search_types import ScoredPage


class SemanticServiceUnavailable(RuntimeError):
    pass


_BREAKER_LOCK = threading.Lock()
_BREAKER_FAILURES = 0
_BREAKER_OPEN_UNTIL = 0.0


def _breaker_before_request() -> None:
    with _BREAKER_LOCK:
        if time.monotonic() < _BREAKER_OPEN_UNTIL:
            raise SemanticServiceUnavailable("semantic circuit breaker is open")


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


def _socket_path(config: SearchEmbeddingConfig) -> Path:
    return Path(config.socket).expanduser()


def request(
    payload: dict[str, Any],
    config: SearchEmbeddingConfig,
    *,
    timeout_ms: int | None = None,
) -> dict[str, Any]:
    _breaker_before_request()
    timeout = max(0.025, float(timeout_ms or config.query_timeout_ms) / 1_000)
    path = _socket_path(config)
    if not path.exists():
        raise SemanticServiceUnavailable(f"semantic socket is missing: {path}")
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
        raise SemanticServiceUnavailable(str(exc)) from exc
    finally:
        client.close()
    try:
        response = json.loads(b"".join(chunks).split(b"\n", 1)[0])
    except (json.JSONDecodeError, IndexError) as exc:
        _breaker_failure()
        raise SemanticServiceUnavailable("invalid semantic service response") from exc
    if not isinstance(response, dict):
        _breaker_failure()
        raise SemanticServiceUnavailable("invalid semantic service payload")
    if response.get("status") != "ok":
        _breaker_failure()
        raise SemanticServiceUnavailable(str(response.get("error") or response))
    _breaker_success()
    return response


def selected_for_rollout(query: str, config: SearchEmbeddingConfig) -> bool:
    if not config.enabled or config.backend != "nemotron_service":
        return False
    if config.rollout_mode == "on":
        return True
    if config.rollout_mode != "canary" or config.canary_percent <= 0:
        return False
    bucket = (
        int.from_bytes(
            hashlib.blake2s(query.encode("utf-8"), digest_size=4).digest(), "big"
        )
        % 100
    )
    return bucket < config.canary_percent


def search(
    query: str,
    top_n: int,
    *,
    include_reference: bool,
    config: SearchEmbeddingConfig,
    timeout_ms: int | None = None,
) -> list[ScoredPage]:
    effective_timeout = (
        config.interactive_timeout_ms
        if timeout_ms is None
        else min(timeout_ms, config.query_timeout_ms)
    )
    response = request(
        {
            "method": "search",
            "query": query,
            "top_n": top_n,
            "include_reference": include_reference,
            "timeout_ms": effective_timeout,
        },
        config,
        timeout_ms=effective_timeout,
    )
    rows = response.get("results")
    if not isinstance(rows, list):
        return []

    from chronovisor.search.index_store import get_store
    from chronovisor.search.search import (
        _REFERENCE_PAGE_TYPE,
        _folder_from_meta,
        _meta_page_type,
        _meta_sensitivity,
        _normalize_lifecycle_status,
    )

    store = get_store()
    store.refresh_if_stale()
    results: list[ScoredPage] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        page_id = str(row.get("page_id") or "")
        meta = store.meta(page_id)
        if meta is None:
            continue
        folder = _folder_from_meta(meta)
        page_type = _meta_page_type(meta, folder=folder)
        if not include_reference and page_type == _REFERENCE_PAGE_TYPE:
            continue
        results.append(
            ScoredPage(
                page_id=page_id,
                title=str(meta.get("title") or page_id),
                folder=folder,
                updated=str(meta.get("updated") or ""),
                score=float(row.get("score") or 0.0),
                status=_normalize_lifecycle_status(meta.get("status")),
                superseded_by=(
                    str(meta.get("superseded_by") or "")
                    if isinstance(meta.get("superseded_by"), str)
                    else ""
                ),
                page_type=page_type,
                sensitivity=_meta_sensitivity(meta, folder=folder),
            )
        )
    return results


def verify(
    query: str,
    page_ids: list[str],
    *,
    config: SearchEmbeddingConfig,
    timeout_ms: int | None = None,
) -> list[ScoredPage]:
    """Return full-dimensional scores for a bounded candidate page set."""

    if not page_ids:
        return []
    effective_timeout = min(
        int(timeout_ms or config.interactive_timeout_ms),
        config.query_timeout_ms,
    )
    response = request(
        {
            "method": "verify",
            "query": query,
            "page_ids": page_ids[:100],
        },
        config,
        timeout_ms=effective_timeout,
    )
    rows = response.get("results")
    if not isinstance(rows, list):
        return []

    from chronovisor.search.index_store import get_store
    from chronovisor.search.search import (
        _folder_from_meta,
        _meta_page_type,
        _meta_sensitivity,
        _normalize_lifecycle_status,
    )

    store = get_store()
    store.refresh_if_stale()
    results: list[ScoredPage] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        page_id = str(row.get("page_id") or "")
        meta = store.meta(page_id)
        if meta is None:
            continue
        folder = _folder_from_meta(meta)
        results.append(
            ScoredPage(
                page_id=page_id,
                title=str(meta.get("title") or page_id),
                folder=folder,
                updated=str(meta.get("updated") or ""),
                score=float(row.get("score") or 0.0),
                status=_normalize_lifecycle_status(meta.get("status")),
                superseded_by=(
                    str(meta.get("superseded_by") or "")
                    if isinstance(meta.get("superseded_by"), str)
                    else ""
                ),
                page_type=_meta_page_type(meta, folder=folder),
                sensitivity=_meta_sensitivity(meta, folder=folder),
            )
        )
    return results


def health(config: SearchEmbeddingConfig) -> dict[str, Any]:
    return request({"method": "health"}, config, timeout_ms=1_000)


def reload_active(config: SearchEmbeddingConfig) -> dict[str, Any]:
    return request({"method": "reload"}, config, timeout_ms=5_000)


def request_rebuild(config: SearchEmbeddingConfig) -> dict[str, Any]:
    return request({"method": "rebuild"}, config, timeout_ms=5_000)


def request_rollback(config: SearchEmbeddingConfig) -> dict[str, Any]:
    return request({"method": "rollback"}, config, timeout_ms=5_000)


def index_pages(
    page_ids: list[str],
    config: SearchEmbeddingConfig,
    *,
    wait: bool,
) -> dict[str, Any]:
    return request(
        {"method": "index_pages", "page_ids": page_ids, "wait": wait},
        config,
        timeout_ms=660_000 if wait else 5_000,
    )
