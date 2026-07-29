"""Patch-safe transport adapters for ingest model calls and progress."""

from __future__ import annotations

import time
from collections.abc import Callable
from inspect import Parameter, signature
from typing import Any, Protocol

from chronovisor.core.ollama import ChatResponse, GenerateResponse
from chronovisor.decision.local_structured import ChatRequest, ChatTransport


class RuntimeStatusProtocol(Protocol):
    def now_iso(self) -> str: ...

    def safe_write_status(self, **fields: Any) -> None: ...


def supports_keyword(function: Callable[..., Any], name: str) -> bool:
    try:
        parameters = signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == Parameter.VAR_KEYWORD or parameter.name == name
        for parameter in parameters
    )


def generate_with_progress(
    generate_fn: Callable[..., Any],
    prompt: str,
    *,
    system: str | None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    format: dict[str, Any] | str | None = None,
    model: str | None = None,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    keep_alive: str | None = None,
    read_timeout_ms: int | None = None,
    temperature: int | float | None = None,
    seed: int | None = None,
    return_metadata: bool = False,
) -> str | GenerateResponse:
    kwargs: dict[str, Any] = {}
    if progress_callback is not None and supports_keyword(
        generate_fn, "progress_callback"
    ):
        kwargs["progress_callback"] = progress_callback
    if format is not None and supports_keyword(generate_fn, "format"):
        kwargs["format"] = format
    optional = {
        "model": model,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "keep_alive": keep_alive,
        "read_timeout_ms": read_timeout_ms,
        "temperature": temperature,
        "seed": seed,
        "return_metadata": return_metadata,
    }
    for name, value in optional.items():
        if (
            value is not None
            and (name != "return_metadata" or value is True)
            and supports_keyword(generate_fn, name)
        ):
            kwargs[name] = value
    try:
        return generate_fn(prompt, system=system, **kwargs)
    except Exception as error:
        if progress_callback is not None:
            progress_callback(
                {"event": "error", "active": False, "error": str(error)}
            )
        raise


def structured_generate_transport(
    generate_fn: Callable[..., Any],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ChatTransport:
    """Adapt the legacy generate seam to structured chat-style history."""

    def transport(request: ChatRequest) -> str | ChatResponse | GenerateResponse:
        system = request.messages[0]["content"] if request.messages else ""
        transcript = "\n\n".join(
            f"<{message['role'].upper()}>\n{message['content']}"
            for message in request.messages[1:]
        )
        kwargs: dict[str, Any] = {"system": system}
        optional = {
            "progress_callback": progress_callback,
            "model": request.model,
            "num_ctx": request.num_ctx,
            "num_predict": request.num_predict,
            "keep_alive": request.keep_alive,
            "read_timeout_ms": request.read_timeout_ms,
            "temperature": request.temperature,
            "seed": request.seed,
            "return_metadata": True,
        }
        for name, value in optional.items():
            if value is not None and supports_keyword(generate_fn, name):
                kwargs[name] = value
        if supports_keyword(generate_fn, "format"):
            kwargs["format"] = request.schema
        return generate_fn(transcript, **kwargs)

    return transport


def structured_chat_transport(chat_fn: Callable[..., Any]) -> ChatTransport:
    """Preserve native roles for production structured repair turns."""

    def transport(request: ChatRequest) -> str | ChatResponse:
        return chat_fn(
            [dict(message) for message in request.messages],
            model=request.model,
            format=request.schema,
            num_ctx=request.num_ctx,
            num_predict=request.num_predict,
            keep_alive=request.keep_alive,
            read_timeout_ms=request.read_timeout_ms,
            max_output_chars=request.max_output_chars,
            temperature=request.temperature,
            seed=request.seed,
            return_metadata=True,
        )

    return transport


def llm_progress_callback(
    runtime_status: RuntimeStatusProtocol,
    *,
    phase: str,
    target: str,
    job_id: str | None,
    source_raw: str | None,
    op_progress: dict[str, int] | None = None,
) -> Callable[[dict[str, Any]], None]:
    started = time.time()
    started_at = runtime_status.now_iso()

    def emit(update: dict[str, Any]) -> None:
        elapsed = update.get("elapsed_seconds")
        if not isinstance(elapsed, (int, float)):
            elapsed = round(max(0.001, time.time() - started), 2)
        status_payload: dict[str, Any] = {
            "active": bool(
                update.get("active", update.get("event") not in {"done", "error"})
            ),
            "event": update.get("event", "chunk"),
            "phase": phase,
            "target": target,
            "job_id": job_id,
            "raw": source_raw,
            "started_at": started_at,
            "updated_at": runtime_status.now_iso(),
            "generated_chars": update.get("generated_chars", update.get("chars", 0)),
            "chunks": update.get("chunks", 0),
            "elapsed_seconds": elapsed,
        }
        if op_progress is not None:
            status_payload["op_progress"] = dict(op_progress)
        for key in (
            "chars_per_second",
            "prompt_eval_count",
            "eval_count",
            "total_duration",
            "eval_duration",
            "error",
        ):
            if key in update:
                status_payload[key] = update[key]
        runtime_status.safe_write_status(llm=status_payload)

    emit(
        {
            "event": "start",
            "active": True,
            "generated_chars": 0,
            "chunks": 0,
            "elapsed_seconds": 0,
        }
    )
    return emit
