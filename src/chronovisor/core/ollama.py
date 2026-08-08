"""Ollama API client for Ingest/Lint operations."""

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import dataclass
from typing import Any, cast

import httpx

from chronovisor.core import ollama_calibration as _ollama_calibration
from chronovisor.core import ollama_lease as _ollama_lease
from chronovisor.core import ollama_telemetry as _ollama_telemetry
from chronovisor.core.runtime_config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INGEST_MODEL,
    IngestConfig,
    load_embedding_config,
    load_ingest_config,
)
from chronovisor.core.store import CHRONOVISOR_ROOT

log = logging.getLogger(__name__)

GIB = _ollama_calibration.GIB
RESIDENCY_UPSHIFT_MIN_HEADROOM_BYTES = (
    _ollama_calibration.RESIDENCY_UPSHIFT_MIN_HEADROOM_BYTES
)
RESIDENCY_UPSHIFT_HEADROOM_RATIO = (
    _ollama_calibration.RESIDENCY_UPSHIFT_HEADROOM_RATIO
)
RESIDENCY_CONTEXT_FLOOR_TOLERANCE_BYTES = (
    _ollama_calibration.RESIDENCY_CONTEXT_FLOOR_TOLERANCE_BYTES
)
RESIDENCY_CONTEXT_FLOOR_TOLERANCE_RATIO = (
    _ollama_calibration.RESIDENCY_CONTEXT_FLOOR_TOLERANCE_RATIO
)
RESIDENCY_COMPRESSED_SINGLE_MIN_BYTES = (
    _ollama_calibration.RESIDENCY_COMPRESSED_SINGLE_MIN_BYTES
)
RESIDENCY_COMPRESSED_SINGLE_RATIO = (
    _ollama_calibration.RESIDENCY_COMPRESSED_SINGLE_RATIO
)
RESIDENCY_SWAP_SINGLE_MIN_BYTES = _ollama_calibration.RESIDENCY_SWAP_SINGLE_MIN_BYTES
RESIDENCY_SWAP_COMPRESSED_FLOOR_BYTES = (
    _ollama_calibration.RESIDENCY_SWAP_COMPRESSED_FLOOR_BYTES
)
RESIDENCY_SWAP_COMPRESSED_FLOOR_RATIO = (
    _ollama_calibration.RESIDENCY_SWAP_COMPRESSED_FLOOR_RATIO
)
MemorySnapshot = _ollama_calibration.MemorySnapshot
MacOSPressureSnapshot = _ollama_calibration.MacOSPressureSnapshot
ModelResidencyPlan = _ollama_calibration.ModelResidencyPlan
build_model_residency_plan = _ollama_calibration.build_model_residency_plan
memory_pressure_requires_single_resident = (
    _ollama_calibration.memory_pressure_requires_single_resident
)

OLLAMA_URL = "http://localhost:11434"
MODEL = DEFAULT_INGEST_MODEL

# Health check cache
_health_cache: dict[str, Any] = {"status": None, "checked_at": 0.0}
HEALTH_CACHE_TTL = 900  # 15 minutes on failure

# Shared httpx.Client — one per process, reused across is_available /
# generate / embed / unload. Connection pooling avoids paying TCP setup
# and DNS lookup cost on every call. Per-call timeouts are still passed
# explicitly so the long-running /api/generate doesn't inherit the short
# health-check default.
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




def _client() -> httpx.Client:
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = httpx.Client(base_url=OLLAMA_URL)
    return _CLIENT


def client() -> httpx.Client:
    return _client()


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


def model_resource_lease(
    *,
    exclusive: bool,
    timeout_ms: int | None = None,
) -> AbstractContextManager[None]:
    """Return a resource lease using the facade's current runtime root."""

    return _ollama_lease.model_resource_lease(
        exclusive=exclusive,
        timeout_ms=timeout_ms,
        root=CHRONOVISOR_ROOT,
    )


def model_resource_lease_mode() -> str | None:
    """Return the current thread's resource-lease mode through the facade."""

    return _ollama_lease.model_resource_lease_mode()


def is_available() -> bool:
    """Check if Ollama is running (cached on failure)."""
    now = time.time()

    # If last check failed, use cache for TTL
    if (
        _health_cache["status"] is False
        and now - _health_cache["checked_at"] < HEALTH_CACHE_TTL
    ):
        return False

    try:
        resp = _client().get("/api/tags", timeout=3)
        available = resp.status_code == 200
        _health_cache["status"] = available
        _health_cache["checked_at"] = now
        return available
    except Exception:
        _health_cache["status"] = False
        _health_cache["checked_at"] = now
        return False


def model_digests(models: Sequence[str]) -> dict[str, str]:
    """Return the currently installed digest for each exact Ollama tag.

    This metadata-only request never loads a model.  Missing tags are returned
    as an empty digest so adoption callers can fail closed without guessing.
    """

    resp = _client().get("/api/tags", timeout=3)
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


def _ollama_daemon_process_identity() -> str:
    return _ollama_calibration._ollama_daemon_process_identity()


def _ollama_engine_identity() -> str:
    return _ollama_calibration._ollama_engine_identity(
        client=_client,
        daemon_identity=_ollama_daemon_process_identity,
    )


def _post_json(
    endpoint: str,
    *,
    payload: Mapping[str, Any],
    timeout: httpx.Timeout,
) -> Any:
    """POST one non-streaming Ollama request with the shared error contract."""

    response = _client().post(endpoint, json=dict(payload), timeout=timeout)
    _raise_for_status_with_detail(response)
    return response.json()




def memory_snapshot() -> MemorySnapshot:
    return _ollama_calibration.memory_snapshot()


def macos_pressure_snapshot() -> MacOSPressureSnapshot:
    return _ollama_calibration.macos_pressure_snapshot()


def _ollama_resource_rows() -> tuple[dict[str, int], dict[str, tuple[int, int]]]:
    tags_response = _client().get("/api/tags", timeout=3)
    tags_response.raise_for_status()
    ps_response = _client().get("/api/ps", timeout=3)
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


def resident_model_rows() -> dict[str, tuple[int, int]]:
    """Return a read-only snapshot of resident model size and context rows."""

    _installed, resident = _ollama_resource_rows()
    return dict(resident)




def plan_model_residency(
    models: Sequence[str],
    *,
    num_ctx: int,
    max_num_ctx: int,
    reserve_bytes: int,
    configured_max_resident: int,
    reuse_larger_context: bool = True,
    reuse_context_ceilings: Mapping[str, int] | None = None,
) -> ModelResidencyPlan:
    return _ollama_calibration.plan_model_residency(
        models,
        num_ctx=num_ctx,
        max_num_ctx=max_num_ctx,
        reserve_bytes=reserve_bytes,
        configured_max_resident=configured_max_resident,
        reuse_larger_context=reuse_larger_context,
        reuse_context_ceilings=reuse_context_ceilings,
        root=CHRONOVISOR_ROOT,
        resource_rows=_ollama_resource_rows,
        digests_for=model_digests,
        engine_identity=_ollama_engine_identity,
        memory_snapshot_for=memory_snapshot,
        macos_pressure_snapshot_for=macos_pressure_snapshot,
    )


def observe_model_runtime(model: str) -> tuple[int, int] | None:
    return _ollama_calibration.observe_model_runtime(
        model,
        root=CHRONOVISOR_ROOT,
        resource_rows=_ollama_resource_rows,
        digests_for=model_digests,
        engine_identity=_ollama_engine_identity,
    )


def unload_named_model(model: str, *, verify_timeout: float = 30.0) -> bool:
    """Unload one known runner and verify that it disappeared from /api/ps."""

    with model_resource_lease(exclusive=True):
        try:
            response = _client().post(
                "/api/generate",
                json={"model": model, "keep_alive": 0, "prompt": ""},
                timeout=10,
            )
            if response.status_code != 200:
                return False
            deadline = time.monotonic() + max(0.0, verify_timeout)
            while True:
                try:
                    _installed, resident = _ollama_resource_rows()
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


@contextmanager
def model_activity(
    *,
    model: str,
    operation: str,
    pipeline: str | None = None,
) -> Iterator[None]:
    with _ollama_telemetry.model_activity(
        model=model,
        operation=operation,
        pipeline=pipeline,
        root=CHRONOVISOR_ROOT,
        facade_module=__name__,
    ):
        yield


def ingest_model() -> str:
    return load_ingest_config().model


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

        with _client().stream(
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

    body = _post_json("/api/generate", payload=payload, timeout=timeout)
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


def generate(
    prompt: str,
    system: str | None = None,
    *,
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
    with model_resource_lease(exclusive=False):
        selected_model = (
            model.strip()
            if isinstance(model, str) and model.strip()
            else ingest_model()
        )
        with model_activity(model=selected_model, operation="generate"):
            return _generate_unlocked(
                prompt,
                system,
                format=format,
                progress_callback=progress_callback,
                model=model,
                num_ctx=num_ctx,
                num_predict=num_predict,
                keep_alive=keep_alive,
                read_timeout_ms=read_timeout_ms,
                temperature=temperature,
                seed=seed,
                return_metadata=return_metadata,
            )


def _chat_unlocked(
    messages: list[dict[str, str]],
    *,
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
    body = _post_json("/api/chat", payload=payload, timeout=timeout)
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


def chat(
    messages: list[dict[str, str]],
    *,
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
    with model_resource_lease(exclusive=False):
        with model_activity(model=model, operation="chat"):
            return _chat_unlocked(
                messages,
                model=model,
                format=format,
                num_ctx=num_ctx,
                num_predict=num_predict,
                keep_alive=keep_alive,
                read_timeout_ms=read_timeout_ms,
                max_output_chars=max_output_chars,
                temperature=temperature,
                seed=seed,
                think=think,
                return_metadata=return_metadata,
            )


EMBED_MODEL = DEFAULT_EMBEDDING_MODEL


def embedding_model() -> str:
    return load_embedding_config().model


def embed(
    texts: list[str],
    *,
    model: str | None = None,
    read_timeout_ms: int | None = None,
) -> list[list[float]]:
    """Get embedding vectors via Ollama /api/embed."""
    timeout_seconds = (
        max(0.2, read_timeout_ms / 1000.0)
        if isinstance(read_timeout_ms, int)
        else 120.0
    )
    selected_model = model or embedding_model()
    with model_resource_lease(exclusive=False):
        with model_activity(
            model=selected_model,
            operation="search",
            pipeline="recall",
        ):
            resp = _client().post(
                "/api/embed",
                json={"model": selected_model, "input": texts},
                timeout=httpx.Timeout(
                    connect=min(10.0, timeout_seconds),
                    read=timeout_seconds,
                    write=min(10.0, timeout_seconds),
                    pool=min(10.0, timeout_seconds),
                ),
            )
            resp.raise_for_status()
            body = resp.json()
            embeddings = (
                body.get("embeddings") if isinstance(body, Mapping) else None
            )
            if not isinstance(embeddings, list):
                raise RuntimeError("Ollama embed response is missing embeddings")
            return cast(list[list[float]], embeddings)


def unload_model() -> None:
    """Explicitly unload model to free memory."""
    unload_named_model(ingest_model())


TRIAGE_SYSTEM_PROMPT = """\
You are a knowledge wiki triage engine. Analyze raw session data and decide \
what wiki pages to create or update. Do NOT generate page content — only output a structured plan.

Rules:
- 1 entity = 1 page
- Output valid JSON array only (no markdown fences, no explanation)
- For every new page, emit exactly `folder/kebab-case.md`; a bare filename is forbidden
- Prefer the best semantically matching folder from the provided existing-folder list
- Only when no existing folder fits, create one specific new top-level folder in
  English kebab-case and place the page there
- Do not use `misc/` merely to avoid choosing or creating a meaningful folder;
  use it only for genuinely miscellaneous knowledge
- For updates: reference the existing page ID in a field named "filename"
- Every update object MUST use "filename". Never emit a "page_id" field
- If the target page is not listed in the catalog, use create, not update
- Skip ephemeral conversation, greetings, and filler
- Include brief summary of what knowledge each page should contain
- Include keywords for finding related existing pages
- Use only these five object keys: type, filename, title, keywords, summary
- Every operation, including updates, MUST include non-empty title, keywords,
  and summary fields
- Emit exactly one operation per case/Unicode-insensitive target page ID. If
  several facts belong on one page, preserve all of them in one combined
  summary and keyword set; never emit multiple operations for that target

Output format (JSON array only):
[
  {
    "type": "create",
    "filename": "folder/kebab-case.md",
    "title": "Page Title",
    "keywords": ["keyword1", "keyword2"],
    "summary": "Brief description of what this page should cover"
  },
  {
    "type": "update",
    "filename": "existing-page.md",
    "title": "Existing Page Title",
    "keywords": ["keyword1", "keyword2"],
    "summary": "What new information to add"
  }
]

WRONG output (do NOT do these):
- Bare keyword list: ["keyword1", "keyword2"]   ← This is a list of strings, not operations
- Single object: {"type": "create", ...}        ← Must be wrapped in an array
- Code fences around the JSON                   ← Output raw JSON only
- Root-level create: {"type": "create", "filename": "topic.md", ...}
  ← Every create must use exactly one top-level folder: `folder/topic.md`

Each top-level element of the array MUST be an object with a "type" field.
"""

GENERATE_SYSTEM_PROMPT = """\
You are a knowledge wiki structuring engine. Generate content for a SINGLE NEW wiki page.

Rules:
- Frontmatter MUST include: title, updated, AND tags
- Use the exact current date supplied in the user prompt for `updated`
- Never invent or infer dates that are absent from the raw evidence
- Cross-references: use [[wiki-link]] notation (page ID only, no folder path)
- Write content in Japanese
- Focus on facts, decisions, and technical knowledge
- Use the provided context for cross-references but do not duplicate existing content

# Tag Taxonomy v0.1 (REQUIRED)

Every page must carry a ``tags:`` frontmatter list with prefixed entries
from a controlled taxonomy. Three axes:

  d/  Domain  (1-3 required) — subject area, kebab-case
       seeds: d/ai-industry, d/hardware, d/geopolitics, d/health, d/finance,
              d/personal-strategy, d/tools-config, d/japan, d/theory, d/paranormal
  t/  Type   (exactly 1 required) — content type, kebab-case
       seeds: t/analysis, t/chat-log, t/howto, t/reference, t/decision,
              t/scenario, t/news-summary
  s/  Scope  (exactly 1 required) — temporal/spatial scope
       seeds: s/2026, s/evergreen, s/historical

# Tag generation rules v1.0

1. Prefix REQUIRED (d/, t/, or s/). Never emit a tag without a prefix.
2. ASCII kebab-case body only (lowercase letters, digits, hyphens). No
   underscores, no spaces, no uppercase, no non-ASCII.
3. Maximum 2 words per tag (split by hyphen). Three+ words → keywords, not tags.
4. Singular form (analysis, not analyses).
5. NO proper nouns (product names, person names, project names) — those
   are keywords. Tags are categorical, not specific.
6. Numbers/years allowed only on the s/ axis (e.g. s/2026). The d/ and t/
   axes must start with a letter.
7. Prefer existing seed tags above when they fit. New tags should be
   genuinely novel categories, not synonyms of existing ones.

Output exactly one page block:
=== NEW PAGE: {filename} ===
---
title: Page Title
updated: YYYY-MM-DD
tags: [d/example-domain, t/analysis, s/evergreen]
---

Page content here with [[wiki-links]] to related topics.

=== END PAGE ===

The final non-whitespace line MUST be exactly `=== END PAGE ===`. Keep the
page concise enough to emit that closing line before stopping.
"""

UPDATE_SYSTEM_PROMPT = """\
You are a knowledge wiki structuring engine. Append content to an EXISTING wiki page.

Rules:
- DO NOT output frontmatter (no `---`, no title:, no updated: lines). The existing page already has frontmatter; your output is appended to its body.
- Never invent or infer dates that are absent from the raw evidence. Do not add a dated heading unless that date appears explicitly in the raw evidence.
- DO NOT repeat content that already exists on the page (it is provided in context).
- Output ONLY the new section(s) to add — Japanese prose, headings, lists, code, etc.
- Cross-references: use [[wiki-link]] notation (page ID only, no folder path)
- Focus on facts, decisions, and technical knowledge

Output exactly one block:
=== UPDATE PAGE: {filename} ===
New section(s) here. Markdown body only — NO frontmatter delimiters.

=== END PAGE ===

The final non-whitespace line MUST be exactly `=== END PAGE ===`. Keep the
update concise enough to emit that closing line before stopping.
"""
