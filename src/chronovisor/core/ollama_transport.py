"""Raw HTTP transport for the local Ollama daemon."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import httpx

if TYPE_CHECKING:
    from chronovisor.core.runtime_config import IngestConfig

log = logging.getLogger("chronovisor.core.ollama")

OLLAMA_URL = "http://localhost:11434"
HEALTH_CACHE_TTL = 900
_health_cache: dict[str, Any] = {"status": None, "checked_at": 0.0}
_CLIENT_LOCK = threading.Lock()
_CLIENT: httpx.Client | None = None


class OutputTooLargeError(RuntimeError):
    """Raised when a structured chat response crosses its fixed char cap."""


@dataclass(frozen=True)
class ChatResponse:
    """Structured chat content plus Ollama's context accounting."""

    content: str
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    # Defaults preserve compatibility with in-process transports that created
    # ``ChatResponse`` before completion metadata was exposed.  The real HTTP
    # adapter always supplies these fields explicitly and treats an omitted
    # Ollama ``done`` flag as incomplete.
    done: bool = True
    done_reason: str | None = None


@dataclass(frozen=True)
class GenerateResponse:
    """Generate content plus Ollama's explicit completion accounting."""

    content: str
    done: bool
    done_reason: str | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    streamed: bool = False


def client(*, base_url: str) -> httpx.Client:
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = httpx.Client(base_url=base_url)
    return _CLIENT


def _raise_for_status_with_detail(response: httpx.Response) -> None:
    """Preserve Ollama's bounded error body in runtime diagnostics."""

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        with suppress(Exception):
            response.read()
        detail = ""
        try:
            body = response.json()
        except Exception:
            body = None
        if isinstance(body, Mapping) and isinstance(body.get("error"), str):
            detail = str(body["error"]).strip()
        if not detail:
            try:
                detail = response.text.strip()
            except Exception:
                detail = ""
        detail = re.sub(r"\s+", " ", detail)[:1_000]
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"Ollama HTTP {response.status_code}{suffix}") from exc


def is_available(
    *,
    client: Callable[[], httpx.Client],
    cache_ttl: float = HEALTH_CACHE_TTL,
) -> bool:
    """Check if Ollama is running (cached on failure)."""
    now = time.time()

    # If last check failed, use cache for TTL
    if (
        _health_cache["status"] is False
        and now - _health_cache["checked_at"] < cache_ttl
    ):
        return False

    try:
        resp = client().get("/api/tags", timeout=3)
        available = resp.status_code == 200
        _health_cache["status"] = available
        _health_cache["checked_at"] = now
        return available
    except Exception:
        _health_cache["status"] = False
        _health_cache["checked_at"] = now
        return False


def model_digests(
    models: Sequence[str],
    *,
    client: Callable[[], httpx.Client],
) -> dict[str, str]:
    """Return the currently installed digest for each exact Ollama tag.

    This metadata-only request never loads a model.  Missing tags are returned
    as an empty digest so adoption callers can fail closed without guessing.
    """

    resp = client().get("/api/tags", timeout=3)
    resp.raise_for_status()
    body = resp.json()
    rows = body.get("models") if isinstance(body, dict) else None
    rows = rows if isinstance(rows, list) else []
    result: dict[str, str] = {}
    for requested in models:
        match = next(
            (
                row
                for row in rows
                if isinstance(row, dict)
                and requested
                in {str(row.get("name") or ""), str(row.get("model") or "")}
            ),
            None,
        )
        digest = match.get("digest") if isinstance(match, dict) else None
        result[requested] = digest if isinstance(digest, str) else ""
    return result


def _post_json(
    endpoint: str,
    *,
    payload: Mapping[str, Any],
    timeout: httpx.Timeout,
    client: Callable[[], httpx.Client],
) -> Any:
    """POST one non-streaming Ollama request with the shared error contract."""

    response = client().post(endpoint, json=dict(payload), timeout=timeout)
    _raise_for_status_with_detail(response)
    return response.json()


def _ollama_resource_rows(
    *, client: Callable[[], httpx.Client]
) -> tuple[dict[str, int], dict[str, tuple[int, int]]]:
    tags_response = client().get("/api/tags", timeout=3)
    tags_response.raise_for_status()
    ps_response = client().get("/api/ps", timeout=3)
    ps_response.raise_for_status()
    tags_body = tags_response.json()
    ps_body = ps_response.json()
    installed: dict[str, int] = {}
    for row in tags_body.get("models", []) if isinstance(tags_body, Mapping) else []:
        if not isinstance(row, Mapping):
            continue
        size = row.get("size")
        if isinstance(size, int) and size > 0:
            for name in {str(row.get("name") or ""), str(row.get("model") or "")}:
                if name:
                    installed[name] = size
    resident: dict[str, tuple[int, int]] = {}
    for row in ps_body.get("models", []) if isinstance(ps_body, Mapping) else []:
        if not isinstance(row, Mapping):
            continue
        size_vram = row.get("size_vram")
        total_size = row.get("size")
        size = max(
            size_vram if isinstance(size_vram, int) and size_vram > 0 else 0,
            total_size if isinstance(total_size, int) and total_size > 0 else 0,
        )
        context = row.get("context_length")
        if not isinstance(size, int) or size <= 0:
            continue
        context_value = context if isinstance(context, int) and context > 0 else 0
        for name in {str(row.get("name") or ""), str(row.get("model") or "")}:
            if name:
                resident[name] = (size, context_value)
    return installed, resident


def unload_named_model(
    model: str,
    *,
    verify_timeout: float = 30.0,
    client: Callable[[], httpx.Client],
    resource_rows: Callable[[], tuple[dict[str, int], dict[str, tuple[int, int]]]],
) -> bool:
    """Unload one known runner and verify that it disappeared from /api/ps."""

    try:
        response = client().post(
            "/api/generate",
            json={"model": model, "keep_alive": 0, "prompt": ""},
            timeout=10,
        )
        if response.status_code != 200:
            return False
        deadline = time.monotonic() + max(0.0, verify_timeout)
        while True:
            try:
                _installed, resident = resource_rows()
                if model not in resident:
                    return True
            except Exception:
                return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
    except Exception:
        return False


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]
) -> None:
    if callback is None:
        return
    with suppress(Exception):
        callback(payload)


def _num_ctx_for_prompt(prompt: str, system: str | None, config: IngestConfig) -> int:
    # Keep ordinary saves on a smaller MLX context, but grow for unusually long
    # raw transcripts so the old 262K ceiling remains available when needed.
    prompt_chars = len(prompt) + (len(system) if system else 0)
    estimated_prompt_tokens = max(1, (prompt_chars + 1) // 2)
    needed = estimated_prompt_tokens + config.num_predict + 1024
    return min(max(config.num_ctx, needed), config.max_num_ctx)


def _generate_unlocked(
    prompt: str,
    system: str | None = None,
    *,
    client: Callable[[], httpx.Client],
    load_ingest_config: Callable[[], IngestConfig],
    format: dict[str, Any] | str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    model: str | None = None,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    keep_alive: str | None = None,
    read_timeout_ms: int | None = None,
    temperature: int | float | None = None,
    seed: int | None = None,
    return_metadata: bool = False,
) -> str | GenerateResponse:
    """Call Ollama generate API.

    Uses keep_alive="5m" to keep model loaded for 5 minutes after use.
    This avoids cold-start on consecutive calls (e.g. Ingest then Lint)
    while still freeing memory after a reasonable idle period.

    When ``progress_callback`` is provided, the call uses Ollama's streaming
    response and periodically emits lightweight progress dictionaries while
    still returning the final response string for existing callers.
    """
    config = load_ingest_config()
    selected_model = (
        model.strip() if isinstance(model, str) and model.strip() else config.model
    )
    selected_num_ctx = (
        num_ctx
        if isinstance(num_ctx, int) and not isinstance(num_ctx, bool) and num_ctx > 0
        else _num_ctx_for_prompt(prompt, system, config)
    )
    selected_num_predict = (
        num_predict
        if isinstance(num_predict, int)
        and not isinstance(num_predict, bool)
        and num_predict > 0
        else config.num_predict
    )
    selected_keep_alive = (
        keep_alive
        if isinstance(keep_alive, str) and keep_alive.strip()
        else config.keep_alive
    )
    selected_read_timeout_ms = (
        read_timeout_ms
        if isinstance(read_timeout_ms, int)
        and not isinstance(read_timeout_ms, bool)
        and read_timeout_ms > 0
        else config.read_timeout_ms
    )
    selected_temperature = (
        temperature
        if isinstance(temperature, (int, float)) and not isinstance(temperature, bool)
        else config.temperature
    )
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
    ):
        raise ValueError("generate seed must be a non-negative integer")
    prompt_chars = len(prompt) + (len(system) if system else 0)
    log.info(
        "generate num_ctx=%d prompt_chars=%d model=%s",
        selected_num_ctx,
        prompt_chars,
        selected_model,
    )
    payload: dict[str, Any] = {
        "model": selected_model,
        "prompt": prompt,
        "stream": progress_callback is not None,
        "think": False,
        # Never let Ollama silently discard the oldest input to satisfy a
        # smaller runner. Ingest performs its own fail-closed context sizing.
        "shift": False,
        "truncate": False,
        "keep_alive": selected_keep_alive,
        "options": {
            "temperature": selected_temperature,
            "num_predict": selected_num_predict,
            "num_ctx": selected_num_ctx,
        },
    }
    if seed is not None:
        payload["options"]["seed"] = seed
    if system:
        payload["system"] = system
    if format is not None:
        payload["format"] = format

    # Timeout: 60s for model load + 600s for generation
    timeout = httpx.Timeout(
        connect=10.0,
        read=selected_read_timeout_ms / 1000,
        write=10.0,
        pool=10.0,
    )
    if progress_callback is not None:
        chunks = 0
        chars = 0
        started = time.monotonic()
        last_emit = 0.0
        pieces: list[str] = []
        final_payload: dict[str, Any] | None = None

        with client().stream(
            "POST",
            "/api/generate",
            json=payload,
            timeout=timeout,
        ) as resp:
            _raise_for_status_with_detail(resp)
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                piece = data.get("response") or ""
                if piece:
                    pieces.append(piece)
                    chunks += 1
                    chars += len(piece)

                done = data.get("done") is True
                now = time.monotonic()
                elapsed = max(0.001, now - started)
                if done or now - last_emit >= 0.75:
                    update = {
                        "event": "done" if done else "chunk",
                        "active": not done,
                        "generated_chars": chars,
                        "chunks": chunks,
                        "elapsed_seconds": round(elapsed, 2),
                        "chars_per_second": round(chars / elapsed, 1),
                    }
                    for key in (
                        "total_duration",
                        "load_duration",
                        "prompt_eval_count",
                        "prompt_eval_duration",
                        "eval_count",
                        "eval_duration",
                    ):
                        if key in data:
                            update[key] = data[key]
                    _emit_progress(progress_callback, update)
                    last_emit = now

                if done:
                    final_payload = data
                    break

        if final_payload is None:
            _emit_progress(
                progress_callback,
                {
                    "event": "error",
                    "active": False,
                    "generated_chars": chars,
                    "chunks": chunks,
                    "elapsed_seconds": round(max(0.001, time.monotonic() - started), 2),
                    "error": "stream ended before done",
                },
            )
            if return_metadata:
                return GenerateResponse(
                    content="".join(pieces),
                    done=False,
                    done_reason=None,
                    streamed=True,
                )
            raise RuntimeError("Ollama stream ended before done")
        content = "".join(pieces)
        if not return_metadata:
            return content
        return GenerateResponse(
            content=content,
            done=final_payload.get("done") is True,
            done_reason=(
                str(final_payload["done_reason"])
                if isinstance(final_payload.get("done_reason"), str)
                else None
            ),
            prompt_eval_count=(
                int(final_payload["prompt_eval_count"])
                if isinstance(final_payload.get("prompt_eval_count"), int)
                and not isinstance(final_payload.get("prompt_eval_count"), bool)
                else None
            ),
            eval_count=(
                int(final_payload["eval_count"])
                if isinstance(final_payload.get("eval_count"), int)
                and not isinstance(final_payload.get("eval_count"), bool)
                else None
            ),
            streamed=True,
        )

    body = _post_json(
        "/api/generate",
        payload=payload,
        timeout=timeout,
        client=client,
    )
    if not isinstance(body, dict) or not isinstance(body.get("response"), str):
        raise RuntimeError("Ollama generate response is missing response content")
    content = str(body["response"])
    if not return_metadata:
        return content
    return GenerateResponse(
        content=content,
        done=body.get("done") is True,
        done_reason=(
            str(body["done_reason"])
            if isinstance(body.get("done_reason"), str)
            else None
        ),
        prompt_eval_count=(
            int(body["prompt_eval_count"])
            if isinstance(body.get("prompt_eval_count"), int)
            and not isinstance(body.get("prompt_eval_count"), bool)
            else None
        ),
        eval_count=(
            int(body["eval_count"])
            if isinstance(body.get("eval_count"), int)
            and not isinstance(body.get("eval_count"), bool)
            else None
        ),
    )


def _chat_unlocked(
    messages: list[dict[str, str]],
    *,
    client: Callable[[], httpx.Client],
    model: str,
    format: dict[str, Any],
    num_ctx: int,
    num_predict: int,
    keep_alive: str,
    read_timeout_ms: int,
    max_output_chars: int,
    temperature: int | float = 0,
    seed: int = 0,
    think: bool | str = False,
    return_metadata: bool = False,
) -> str | ChatResponse:
    """Call Ollama's chat API for one fixed-cap structured-output turn.

    Unlike :func:`generate`, this adapter never derives context size from the
    prompt.  Decision models therefore keep a stable runner allocation across
    initial and repair turns.  Only ``message.content`` is returned; any
    separate thinking field is intentionally ignored.
    """

    if not isinstance(model, str) or not model.strip():
        raise ValueError("model is required")
    if num_ctx < 1 or num_predict < 1 or max_output_chars < 1:
        raise ValueError("chat limits must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("chat seed must be a non-negative integer")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("chat temperature must be numeric")
    if not isinstance(think, bool) and think not in {"low", "medium", "high"}:
        raise ValueError("chat think must be a boolean or low, medium, high")
    payload = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "stream": False,
        "think": think,
        "shift": False,
        "truncate": False,
        "format": format,
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }
    timeout = httpx.Timeout(
        connect=10.0,
        read=read_timeout_ms / 1000,
        write=10.0,
        pool=10.0,
    )
    body = _post_json("/api/chat", payload=payload, timeout=timeout, client=client)
    message = body.get("message") if isinstance(body, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise RuntimeError("Ollama chat response is missing message.content")
    # Metadata callers are the bounded structured-session layer.  Return the
    # response to that layer even when it crossed the cap so it can record a
    # redacted attempt and ask the same model for a compact repair.  Plain
    # callers have no repair protocol and must continue to fail closed here.
    if len(content) > max_output_chars and not return_metadata:
        raise OutputTooLargeError(
            f"Ollama chat response exceeded max_output_chars={max_output_chars}"
        )
    if not return_metadata:
        return content
    prompt_eval_count = (
        body.get("prompt_eval_count") if isinstance(body, dict) else None
    )
    eval_count = body.get("eval_count") if isinstance(body, dict) else None
    return ChatResponse(
        content=content,
        prompt_eval_count=(
            prompt_eval_count
            if isinstance(prompt_eval_count, int)
            and not isinstance(prompt_eval_count, bool)
            else None
        ),
        eval_count=(
            eval_count
            if isinstance(eval_count, int) and not isinstance(eval_count, bool)
            else None
        ),
        done=body.get("done") is True,
        done_reason=(
            str(body["done_reason"])
            if isinstance(body.get("done_reason"), str)
            else None
        ),
    )


def embed(
    texts: list[str],
    *,
    model: str,
    read_timeout_ms: int | None,
    client: Callable[[], httpx.Client],
) -> list[list[float]]:
    """Get embedding vectors via Ollama /api/embed."""

    timeout_seconds = (
        max(0.2, read_timeout_ms / 1000.0)
        if isinstance(read_timeout_ms, int)
        else 120.0
    )
    response = client().post(
        "/api/embed",
        json={"model": model, "input": texts},
        timeout=httpx.Timeout(
            connect=min(10.0, timeout_seconds),
            read=timeout_seconds,
            write=min(10.0, timeout_seconds),
            pool=min(10.0, timeout_seconds),
        ),
    )
    response.raise_for_status()
    body = response.json()
    embeddings = body.get("embeddings") if isinstance(body, Mapping) else None
    if not isinstance(embeddings, list):
        raise RuntimeError("Ollama embed response is missing embeddings")
    return cast(list[list[float]], embeddings)
