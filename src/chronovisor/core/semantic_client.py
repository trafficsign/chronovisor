"""Core client for the configured semantic retrieval service."""

from __future__ import annotations

import hashlib
import json
import math
import socket
import threading
import time
from pathlib import Path
from typing import Any

from chronovisor.core.runtime_config import SearchEmbeddingConfig
from chronovisor.core.search_types import ScoredPage


class SemanticServiceUnavailable(RuntimeError):
    pass


_BREAKER_LOCK = threading.Lock()
_BREAKER_FAILURES = 0
_BREAKER_OPEN_UNTIL = 0.0


def _deadline_for(
    timeout_ms: int | float | None,
    default_ms: int | float,
    *,
    cap_ms: int | float | None = None,
) -> tuple[float, float]:
    """Return one validated budget and its absolute monotonic deadline."""

    raw = default_ms if timeout_ms is None else timeout_ms
    try:
        budget_ms = float(raw)
    except (TypeError, ValueError) as exc:
        raise SemanticServiceUnavailable(
            "semantic request deadline exhausted"
        ) from exc
    if not math.isfinite(budget_ms) or budget_ms <= 0:
        raise SemanticServiceUnavailable("semantic request deadline exhausted")
    if cap_ms is not None:
        try:
            cap = float(cap_ms)
        except (TypeError, ValueError) as exc:
            raise SemanticServiceUnavailable(
                "semantic request deadline exhausted"
            ) from exc
        if not math.isfinite(cap) or cap <= 0:
            raise SemanticServiceUnavailable("semantic request deadline exhausted")
        budget_ms = min(budget_ms, cap)
    return budget_ms, time.monotonic() + budget_ms / 1_000


def _deadline_from_payload(deadline_at: float | None) -> float:
    if deadline_at is None:
        raise ValueError("deadline is required")
    try:
        deadline = float(deadline_at)
    except (TypeError, ValueError) as exc:
        raise SemanticServiceUnavailable(
            "semantic request deadline exhausted"
        ) from exc
    if not math.isfinite(deadline) or deadline <= time.monotonic():
        raise SemanticServiceUnavailable("semantic request deadline exhausted")
    return deadline


def _remaining_seconds(deadline_at: float) -> float:
    remaining = deadline_at - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("semantic request deadline exhausted")
    return remaining


def _load_metadata_snapshot(store: object) -> None:
    """Load only the persisted writer snapshot on the latency-sensitive path."""

    load_existing = getattr(store, "load_existing", None)
    if callable(load_existing):
        try:
            available = bool(load_existing())
        except Exception as exc:
            raise SemanticServiceUnavailable(
                "semantic metadata snapshot unavailable"
            ) from exc
        if not available:
            raise SemanticServiceUnavailable("semantic metadata snapshot unavailable")
        return
    # Small injected stores in tests and integrations predate load_existing.
    refresh_if_stale = getattr(store, "refresh_if_stale", None)
    if callable(refresh_if_stale):
        try:
            refresh_if_stale()
        except Exception as exc:
            raise SemanticServiceUnavailable(
                "semantic metadata snapshot unavailable"
            ) from exc
        return
    refresh = getattr(store, "refresh", None)
    if callable(refresh):
        try:
            refresh()
        except Exception as exc:
            raise SemanticServiceUnavailable(
                "semantic metadata snapshot unavailable"
            ) from exc
        return
    raise SemanticServiceUnavailable("semantic metadata snapshot unavailable")


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
    timeout_ms: int | float | None = None,
    deadline_at: float | None = None,
) -> dict[str, Any]:
    _, derived_deadline = _deadline_for(timeout_ms, config.query_timeout_ms)
    payload_deadline = payload.get("deadline_at")
    if deadline_at is not None:
        deadline = _deadline_from_payload(deadline_at)
    elif payload_deadline is not None:
        deadline = _deadline_from_payload(payload_deadline)
    else:
        deadline = derived_deadline
    if timeout_ms is not None:
        deadline = min(deadline, derived_deadline)
    _breaker_before_request()
    path = _socket_path(config)
    if not path.exists():
        raise SemanticServiceUnavailable(f"semantic socket is missing: {path}")
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.settimeout(_remaining_seconds(deadline))
        client.connect(str(path))
        client.settimeout(_remaining_seconds(deadline))
        wire_payload = payload
        if payload.get("method") in {"search", "verify"}:
            wire_payload = {**payload, "deadline_at": deadline}
        client.sendall(
            json.dumps(wire_payload, ensure_ascii=False, separators=(",", ":")).encode()
            + b"\n"
        )
        chunks: list[bytes] = []
        while True:
            client.settimeout(_remaining_seconds(deadline))
            chunk = client.recv(64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
    except (OSError, TimeoutError) as exc:
        _breaker_failure()
        raise SemanticServiceUnavailable(
            str(exc) or "semantic request deadline exhausted"
        ) from exc
    finally:
        client.close()
    try:
        _remaining_seconds(deadline)
    except TimeoutError as exc:
        _breaker_failure()
        raise SemanticServiceUnavailable(str(exc)) from exc
    try:
        response = json.loads(b"".join(chunks).split(b"\n", 1)[0])
    except (UnicodeDecodeError, json.JSONDecodeError, IndexError) as exc:
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
    if not config.enabled:
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
    timeout_ms: int | float | None = None,
) -> list[ScoredPage]:
    effective_timeout, deadline = _deadline_for(
        timeout_ms,
        config.interactive_timeout_ms,
        cap_ms=(config.query_timeout_ms if timeout_ms is not None else None),
    )
    response = request(
        {
            "method": "search",
            "query": query,
            "top_n": top_n,
            "include_reference": include_reference,
            "timeout_ms": int(effective_timeout)
            if effective_timeout.is_integer()
            else effective_timeout,
            "deadline_at": deadline,
        },
        config,
        timeout_ms=effective_timeout,
        deadline_at=deadline,
    )
    try:
        _remaining_seconds(deadline)
    except TimeoutError as exc:
        raise SemanticServiceUnavailable(str(exc)) from exc
    rows = response.get("results")
    if not isinstance(rows, list) or not rows:
        return []

    from chronovisor.core.index_store import get_store
    from chronovisor.core.search import (
        _REFERENCE_PAGE_TYPE,
        _folder_from_meta,
        _meta_page_type,
        _meta_sensitivity,
        _normalize_lifecycle_status,
    )

    try:
        _remaining_seconds(deadline)
    except TimeoutError as exc:
        raise SemanticServiceUnavailable(str(exc)) from exc
    store = get_store()
    try:
        _remaining_seconds(deadline)
    except TimeoutError as exc:
        raise SemanticServiceUnavailable(str(exc)) from exc
    _load_metadata_snapshot(store)
    try:
        _remaining_seconds(deadline)
    except TimeoutError as exc:
        raise SemanticServiceUnavailable(str(exc)) from exc
    results: list[ScoredPage] = []
    for row in rows:
        try:
            _remaining_seconds(deadline)
        except TimeoutError as exc:
            raise SemanticServiceUnavailable(str(exc)) from exc
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
    try:
        _remaining_seconds(deadline)
    except TimeoutError as exc:
        raise SemanticServiceUnavailable(str(exc)) from exc
    return results


def verify(
    query: str,
    page_ids: list[str],
    *,
    config: SearchEmbeddingConfig,
    timeout_ms: int | float | None = None,
) -> list[ScoredPage]:
    """Return full-dimensional scores for a bounded candidate page set."""

    effective_timeout, deadline = _deadline_for(
        timeout_ms,
        config.interactive_timeout_ms,
        cap_ms=config.query_timeout_ms,
    )
    if not page_ids:
        return []
    response = request(
        {
            "method": "verify",
            "query": query,
            "page_ids": page_ids[:100],
            "timeout_ms": int(effective_timeout)
            if effective_timeout.is_integer()
            else effective_timeout,
            "deadline_at": deadline,
        },
        config,
        timeout_ms=effective_timeout,
        deadline_at=deadline,
    )
    try:
        _remaining_seconds(deadline)
    except TimeoutError as exc:
        raise SemanticServiceUnavailable(str(exc)) from exc
    rows = response.get("results")
    if not isinstance(rows, list) or not rows:
        return []

    from chronovisor.core.index_store import get_store
    from chronovisor.core.search import (
        _folder_from_meta,
        _meta_page_type,
        _meta_sensitivity,
        _normalize_lifecycle_status,
    )

    try:
        _remaining_seconds(deadline)
    except TimeoutError as exc:
        raise SemanticServiceUnavailable(str(exc)) from exc
    store = get_store()
    try:
        _remaining_seconds(deadline)
    except TimeoutError as exc:
        raise SemanticServiceUnavailable(str(exc)) from exc
    _load_metadata_snapshot(store)
    try:
        _remaining_seconds(deadline)
    except TimeoutError as exc:
        raise SemanticServiceUnavailable(str(exc)) from exc
    results: list[ScoredPage] = []
    for row in rows:
        try:
            _remaining_seconds(deadline)
        except TimeoutError as exc:
            raise SemanticServiceUnavailable(str(exc)) from exc
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
    try:
        _remaining_seconds(deadline)
    except TimeoutError as exc:
        raise SemanticServiceUnavailable(str(exc)) from exc
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
