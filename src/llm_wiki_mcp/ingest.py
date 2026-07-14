"""Ingest engine - structures raw data into wiki pages (two-stage pipeline)."""

import ast
import hashlib
import json
import re
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Callable

from llm_wiki_mcp.wiki import (
    PAGES_DIR,
    INDEX_FILE,
    LOG_FILE,
    all_pages,
    find_page,
    page_id_from_path,
)
from llm_wiki_mcp.jobs import job_store, JobStatus
from llm_wiki_mcp.ollama import (
    generate,
    is_available,
    TRIAGE_SYSTEM_PROMPT,
    GENERATE_SYSTEM_PROMPT,
    UPDATE_SYSTEM_PROMPT,
)
from llm_wiki_mcp.local_structured import (
    ChatRequest,
    ChatTransport,
    LocalStructuredSession,
    ValidationIssue,
    required_structured_context_tokens,
    validate_json,
)
from llm_wiki_mcp.runtime_config import load_ingest_config
from llm_wiki_mcp import decision_authority, ollama as ollama_runtime, runtime_status
from llm_wiki_mcp.decision_lane_prompts import INGEST_PROPOSAL_SCHEMA_VERSION
from llm_wiki_mcp.entities import patch_entities_frontmatter
from llm_wiki_mcp.triage_plan import (
    collapse_exact_duplicate_operations,
    distinct_target_collisions,
)


# ---------------------------------------------------------------------------
# Stage 1: Triage — analyze raw content and produce a structured plan
# ---------------------------------------------------------------------------


def _extract_json_array(output: str) -> list[dict] | None:
    """Best-effort extraction of a JSON array from an LLM response.

    Uses ``json.JSONDecoder.raw_decode`` to try every ``[`` position as the
    start of a valid JSON value, taking the longest array that parses. This
    is robust against:

    * preamble fluff (``---\\n[...]``, ``Here is the plan: [...]``).
    * postamble prose containing brackets (``[...]\\nNote: [done]``).
    * markdown code fences.
    * literal ``]`` inside summary fields (the parser doesn't care).

    Returns ``None`` on parse failure so the caller can distinguish failure
    from a legitimate empty plan (``[]``).
    """
    if not output:
        return None

    text = output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    decoder = json.JSONDecoder()
    n = len(text)
    first_idx_in_text = text.find("[")
    if first_idx_in_text == -1:
        return None

    candidates: list[tuple[int, int, list]] = []  # (idx, consumed, value)
    pos = 0
    while pos < n:
        idx = text.find("[", pos)
        if idx == -1:
            break
        try:
            # Pass the full text + an offset so we don't re-allocate a slice
            # for every candidate position (was O(N²) on bracket-heavy input).
            value, end_offset = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            pos = idx + 1
            continue
        if isinstance(value, list):
            candidates.append((idx, end_offset - idx, value))
        pos = max(end_offset, idx + 1)

    if not candidates:
        # Some local models occasionally emit a Python literal despite an
        # explicit JSON contract (single quotes / True / None).  Accept only
        # a literal list whose entire value is JSON-shaped. ``literal_eval``
        # cannot execute code, and the recursive type check keeps tuples,
        # sets, bytes, and other Python-only values out of the ingest schema.
        last_idx = text.rfind("]")
        if last_idx > first_idx_in_text:
            try:
                literal = ast.literal_eval(text[first_idx_in_text : last_idx + 1])
            except (SyntaxError, ValueError, MemoryError, RecursionError):
                literal = None

            def is_json_value(value: object) -> bool:
                if value is None or isinstance(value, (str, int, float, bool)):
                    return True
                if isinstance(value, list):
                    return all(is_json_value(item) for item in value)
                if isinstance(value, dict):
                    return all(
                        isinstance(key, str) and is_json_value(item)
                        for key, item in value.items()
                    )
                return False

            if isinstance(literal, list) and is_json_value(literal):
                return literal
        return None

    # If the outermost `[` parsed cleanly, trust the LLM's intent and return
    # the longest array we found (preserves the historical "preamble [done]
    # then real plan" behavior).
    if candidates[0][0] == first_idx_in_text:
        candidates.sort(key=lambda c: c[1], reverse=True)
        return candidates[0][2]

    # The outer array did not parse — usually because the model truncated
    # mid-stream. raw_decode then picks up inner ``"keywords": []`` or
    # ``"keywords": ["x", "y"]`` lists as the "best" valid array, which
    # silently routes truncated triage as either "nothing wiki-worthy"
    # (empty plan → raws marked processed) or "schema invalid" (counted
    # toward dead-letter quarantine). Both are wrong: the LLM had more to
    # say. Only accept inner matches that fit the contract — non-empty
    # arrays of dicts; otherwise return ``None`` so the caller treats this
    # as a parse failure and the raws stay pending for retry.
    dict_arrays = [
        (consumed, value)
        for _, consumed, value in candidates
        if value and all(isinstance(e, dict) for e in value)
    ]
    if dict_arrays:
        dict_arrays.sort(reverse=True)
        return dict_arrays[0][1]
    return None


def _supports_keyword(fn: Callable[..., Any], name: str) -> bool:
    try:
        params = signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.kind == Parameter.VAR_KEYWORD or p.name == name for p in params)


def _generate_with_progress(
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
) -> str | ollama_runtime.GenerateResponse:
    kwargs: dict[str, Any] = {}
    if progress_callback is not None and _supports_keyword(
        generate, "progress_callback"
    ):
        kwargs["progress_callback"] = progress_callback
    if format is not None and _supports_keyword(generate, "format"):
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
            and _supports_keyword(generate, name)
        ):
            kwargs[name] = value
    try:
        return generate(prompt, system=system, **kwargs)
    except Exception as e:
        if progress_callback is not None:
            progress_callback({"event": "error", "active": False, "error": str(e)})
        raise


_DEFAULT_GENERATE_WITH_PROGRESS = _generate_with_progress


def _structured_generate_transport(
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ChatTransport:
    """Adapt the legacy generate fixture/progress seam to chat-style history."""

    def transport(
        request: ChatRequest,
    ) -> str | ollama_runtime.ChatResponse | ollama_runtime.GenerateResponse:
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
            if value is not None and _supports_keyword(_generate_with_progress, name):
                kwargs[name] = value
        # Existing tests and callers can still replace the historical helper
        # with a narrower fixture. Production receives Ollama's JSON schema.
        if _supports_keyword(_generate_with_progress, "format"):
            kwargs["format"] = request.schema
        return _generate_with_progress(transcript, **kwargs)

    return transport


def _structured_chat_transport() -> ChatTransport:
    """Preserve native chat roles for production structured repair turns.

    The historical ingest seam above flattens messages into one generate
    transcript so narrow test fixtures can keep replacing
    ``_generate_with_progress``.  Production must not use that compatibility
    path: Ollama's chat endpoint retains the assistant response and the exact
    validator feedback as separate roles, which is the contract
    ``LocalStructuredSession`` repairs against.
    """

    def transport(
        request: ChatRequest,
    ) -> str | ollama_runtime.ChatResponse:
        return ollama_runtime.chat(
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


def _triage_with_progress(
    content: str,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    *,
    frontier_feedback: str | None = None,
) -> list[dict] | None:
    kwargs: dict[str, Any] = {}
    if progress_callback is not None and _supports_keyword(
        _triage, "progress_callback"
    ):
        kwargs["progress_callback"] = progress_callback
    if frontier_feedback and _supports_keyword(_triage, "frontier_feedback"):
        kwargs["frontier_feedback"] = frontier_feedback
    if _supports_keyword(_triage, "raise_on_failure"):
        kwargs["raise_on_failure"] = True
    return _triage(content, **kwargs)


def _generate_one_with_progress(
    op: dict,
    raw_content: str,
    *,
    raw_keywords: list[str] | None,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    frontier_feedback: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict | None:
    kwargs: dict[str, Any] = {"raw_keywords": raw_keywords}
    if progress_callback is not None and _supports_keyword(
        _generate_one, "progress_callback"
    ):
        kwargs["progress_callback"] = progress_callback
    if frontier_feedback and _supports_keyword(_generate_one, "frontier_feedback"):
        kwargs["frontier_feedback"] = frontier_feedback
    if diagnostics is not None and _supports_keyword(_generate_one, "diagnostics"):
        kwargs["diagnostics"] = diagnostics
    return _generate_one(op, raw_content, **kwargs)


def _llm_progress_callback(
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
        generated_chars = update.get("generated_chars", update.get("chars", 0))
        chunks = update.get("chunks", 0)
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
            "generated_chars": generated_chars,
            "chunks": chunks,
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


_TRIAGE_CATALOG_TOP_N = 100
_TRIAGE_MAX_OPERATIONS = 8
_TRIAGE_MAX_OUTPUT_BYTES = 8_000
_TRIAGE_MAX_FEEDBACK_BYTES = 4_000
_TRIAGE_NUM_PREDICT = 4_096

# Canonical post-response contract.  This is deliberately separate from the
# grammar schema below: llama.cpp expands nested numeric repetition bounds and
# rejects their product before inference on Ollama 0.31.1.
_TRIAGE_PLAN_VALIDATION_SCHEMA: dict[str, Any] = {
    "type": "array",
    "maxItems": _TRIAGE_MAX_OPERATIONS,
    "items": {
        "type": "object",
        "additionalProperties": False,
        # Direct callers may still validate legacy update operations that do
        # not carry display metadata. The production wire schema below is
        # stricter and always requires all five fields.
        "required": ["type", "filename"],
        "properties": {
            "type": {"type": "string", "enum": ["create", "update"]},
            "filename": {"type": "string", "minLength": 1, "maxLength": 200},
            "title": {"type": "string", "minLength": 1, "maxLength": 300},
            "keywords": {
                "type": "array",
                "minItems": 1,
                "maxItems": 32,
                "items": {"type": "string", "minLength": 1, "maxLength": 200},
            },
            "summary": {"type": "string", "minLength": 1, "maxLength": 2_000},
        },
    },
}

TRIAGE_PLAN_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        # Only these five fields cross the semantic-plan boundary.  Unknown
        # model keys are not diagnostics: accepting them can hide a malformed
        # known key (for example ``"keywords: ["``) while the real field is
        # absent.  The structured session returns this exact schema violation
        # to the model on the next bounded repair turn.
        "additionalProperties": False,
        # A uniform five-field wire contract avoids conditional grammar while
        # still making every response usable by the generation stage. Numeric
        # bounds remain host-side because nested repetition bounds are rejected
        # by llama.cpp before inference on the deployed Ollama version.
        "required": ["type", "filename", "title", "keywords", "summary"],
        "properties": {
            "type": {"type": "string", "enum": ["create", "update"]},
            # Do not encode numeric repetition bounds in the grammar sent to
            # Ollama.  llama.cpp expands nested maxItems/maxLength products
            # and rejects this otherwise-small schema before inference.  The
            # same limits are enforced by _triage_plan_validation_issues on
            # every response and returned as targeted repair feedback.
            "filename": {"type": "string"},
            "title": {"type": "string"},
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
            },
            "summary": {"type": "string"},
        },
    },
}


_INGEST_CONTEXT_BUCKETS = (32_768, 65_536, 131_072, 262_144)


class IngestContextCapacityError(RuntimeError):
    """Raised before inference when no configured context bucket can fit."""


class IngestTriageFailure(RuntimeError):
    """Preserve one structured-session failure across the ingest boundary."""

    def __init__(self, failure_class: str, reason: str) -> None:
        self.failure_class = failure_class or "unknown"
        self.reason = reason or "structured triage failed"
        super().__init__(
            f"triage structured failure [{self.failure_class}]: {self.reason}"
        )


def ingest_context_buckets(*, num_ctx: int, max_num_ctx: int) -> tuple[int, ...]:
    """Return monotonic ingest buckets within the configured model envelope."""

    candidates = (num_ctx, *_INGEST_CONTEXT_BUCKETS, max_num_ctx)
    buckets = tuple(
        sorted({value for value in candidates if num_ctx <= value <= max_num_ctx})
    )
    return buckets or (max_num_ctx,)


def _select_ingest_context(
    required: int,
    *,
    num_ctx: int,
    max_num_ctx: int,
) -> int:
    for bucket in ingest_context_buckets(
        num_ctx=num_ctx,
        max_num_ctx=max_num_ctx,
    ):
        if bucket >= required:
            return bucket
    raise IngestContextCapacityError(
        f"required context {required} exceeds configured max_num_ctx {max_num_ctx}"
    )


def _admit_ingest_context(config: Any, requested_num_ctx: int) -> int:
    """Use the shared measured-residency planner for one ingest runner."""

    def evict_unrelated_residents() -> int:
        try:
            resident_models = ollama_runtime.resident_model_rows()
        except Exception as exc:
            raise IngestTriageFailure(
                "capacity_unavailable",
                f"resident model probe failed: {type(exc).__name__}: {exc}",
            ) from exc
        evicted = 0
        for model in sorted(resident_models):
            if model == config.model:
                continue
            if not ollama_runtime.unload_named_model(model):
                raise IngestTriageFailure(
                    "capacity_unavailable",
                    f"unable to verify ingest runner eviction: {model}",
                )
            evicted += 1
        return evicted

    def residency_plan():
        try:
            return ollama_runtime.plan_model_residency(
                [config.model],
                num_ctx=requested_num_ctx,
                max_num_ctx=config.max_num_ctx,
                reserve_bytes=config.memory_reserve_gib * ollama_runtime.GIB,
                configured_max_resident=1,
                reuse_larger_context=True,
            )
        except Exception as exc:
            raise IngestTriageFailure(
                "capacity_unavailable",
                f"residency planning failed: {type(exc).__name__}: {exc}",
            ) from exc

    # The 256K bucket is intentionally a one-runner mode. Its KV allocation is
    # the only ingest envelope large enough for legacy 120KB captures, so
    # keeping unrelated decision/recall runners resident would recreate the
    # memory-pressure failure this admission gate exists to prevent.
    if requested_num_ctx >= 262_144:
        evict_unrelated_residents()
    plan = residency_plan()
    # Smaller buckets normally coexist with recall/decision runners. If live
    # headroom cannot fit even one ingest runner, reclaim only unrelated Ollama
    # residents and re-plan from fresh measured memory before deferring. This
    # avoids a permanent low-context stall without evicting healthy models on
    # every request.
    if plan.max_resident_models < 1 and requested_num_ctx < 262_144:
        if evict_unrelated_residents() > 0:
            plan = residency_plan()
    if plan.max_resident_models < 1:
        raise IngestTriageFailure(
            "capacity_unavailable",
            "measured memory admission cannot fit one ingest runner",
        )
    for model in plan.initial_eviction_models:
        if not ollama_runtime.unload_named_model(model):
            raise IngestTriageFailure(
                "capacity_unavailable",
                f"unable to verify incompatible ingest runner eviction: {model}",
            )
    return max(requested_num_ctx, plan.context_for(config.model))


def _emit_triage_failure(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    failure: IngestTriageFailure,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        {
            "event": "error",
            "active": False,
            "failure_class": failure.failure_class,
            "error": failure.reason,
        }
    )


def _triage(
    content: str,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    frontier_feedback: str | None = None,
    transport: ChatTransport | None = None,
    raise_on_failure: bool = False,
) -> list[dict] | None:
    """Stage 1: Analyze raw content and return a plan, or None on parse failure.

    Distinguishing ``None`` (parser/model failure) from ``[]`` (model said
    "nothing wiki-worthy") matters for the caller: failures should leave
    raw files un-marked so the next tick retries them, while a legitimate
    empty plan should mark the raws processed to avoid forever-retry.
    """
    existing_folders = sorted(
        {p.parent.name for p in all_pages() if p.parent != PAGES_DIR}
    )
    catalog_lines = [
        f"Existing folders: {', '.join(f'{f}/' for f in existing_folders)}",
        "",
    ]

    catalog_lines.append("Existing wiki pages (page_id — title):")
    try:
        from llm_wiki_mcp.search import search as wiki_search

        query_text = content[:2000]
        results, _ = wiki_search(query_text, top_n=_TRIAGE_CATALOG_TOP_N, semantic=True)
        for r in results:
            catalog_lines.append(f"  [[{r.page_id}]] — {r.title}")
        _safe_log(
            f"ingest | triage catalog filtered to {len(results)} pages (of {len(list(all_pages()))} total)"
        )
    except Exception:
        for path in all_pages():
            content_text = path.read_text()
            fm_match = re.search(r"title:\s*(.+)", content_text)
            title = fm_match.group(1).strip() if fm_match else path.stem
            catalog_lines.append(f"  [[{page_id_from_path(path)}]] — {title}")

    catalog = "\n".join(catalog_lines)

    feedback_block = ""
    if frontier_feedback:
        feedback_block = f"""

---
Previous local consensus review (authoritative correction instructions):
---
{frontier_feedback}
---
Regenerate the plan from the raw evidence. Remove unsupported claims, keep
only durable facts explicitly grounded in the raw, and use the smallest
complete create/update set that resolves the review.
"""

    prompt = f"""{catalog}

---
Raw session data to triage:
---
{content}
---
{feedback_block}

Analyze the raw data above. Output a JSON array of page operations (create/update)."""

    config = load_ingest_config()
    triage_num_predict = min(config.num_predict, _TRIAGE_NUM_PREDICT)
    required_num_ctx = required_structured_context_tokens(
        prompt,
        TRIAGE_PLAN_SCHEMA,
        system=TRIAGE_SYSTEM_PROMPT,
        num_predict=triage_num_predict,
        max_output_chars=_TRIAGE_MAX_OUTPUT_BYTES,
        max_feedback_chars=_TRIAGE_MAX_FEEDBACK_BYTES,
    )
    try:
        selected_num_ctx = _select_ingest_context(
            required_num_ctx,
            num_ctx=config.num_ctx,
            max_num_ctx=config.max_num_ctx,
        )
        live_transport = (
            transport is None
            and _generate_with_progress is _DEFAULT_GENERATE_WITH_PROGRESS
        )
        lease = (
            ollama_runtime.model_resource_lease(exclusive=True)
            if live_transport
            else nullcontext()
        )
        with lease:
            if live_transport:
                selected_num_ctx = _admit_ingest_context(config, selected_num_ctx)
            session_transport = transport
            if session_transport is None:
                session_transport = (
                    _structured_chat_transport()
                    if live_transport
                    else _structured_generate_transport(progress_callback)
                )
            result = LocalStructuredSession(
                model=config.model,
                transport=session_transport,
                role="ingest_triage",
                num_ctx=selected_num_ctx,
                num_predict=triage_num_predict,
                keep_alive=config.keep_alive,
                read_timeout_ms=config.read_timeout_ms,
                # Context preflight below remains the authoritative bound. This
                # independent byte cap prevents an input larger than the exact
                # admitted runner from reaching Ollama.
                max_input_chars=selected_num_ctx,
                max_output_chars=_TRIAGE_MAX_OUTPUT_BYTES,
                max_feedback_chars=_TRIAGE_MAX_FEEDBACK_BYTES,
            ).run(
                prompt,
                TRIAGE_PLAN_SCHEMA,
                system=TRIAGE_SYSTEM_PROMPT,
                value_validator=lambda value: _triage_plan_validation_issues(
                    value,
                    resolve_effective_targets=True,
                ),
            )
    except IngestContextCapacityError as exc:
        failure = IngestTriageFailure("context_window_exceeded", str(exc))
        _emit_triage_failure(progress_callback, failure)
        if raise_on_failure:
            raise failure from exc
        return None
    except IngestTriageFailure as failure:
        _emit_triage_failure(progress_callback, failure)
        if raise_on_failure:
            raise
        return None
    if not result.ok:
        failure = IngestTriageFailure(
            result.failure_class or "unknown",
            result.failure_reason or "structured triage failed",
        )
        _safe_log(
            "ingest | triage structured session failed "
            f"({failure.failure_class}: {failure.reason[:160]})"
        )
        _emit_triage_failure(progress_callback, failure)
        if raise_on_failure:
            raise failure
        return None
    raw_plan = result.value
    if not isinstance(raw_plan, list):
        _safe_log("ingest | triage structured session returned a non-array")
        failure = IngestTriageFailure(
            "value_validation_error", "triage returned non-array"
        )
        _emit_triage_failure(progress_callback, failure)
        if raise_on_failure:
            raise failure
        return None
    validated = _validate_triage_plan(raw_plan, coerce_missing_updates=True)
    if validated is None:
        _safe_log(f"ingest | triage schema invalid (preview: {str(raw_plan)[:120]!r})")
        failure = IngestTriageFailure(
            "value_validation_error",
            "triage post-validation diverged from the structured-session validator",
        )
        _emit_triage_failure(progress_callback, failure)
        if raise_on_failure:
            raise failure
        return None
    if progress_callback is not None:
        progress_callback({"event": "done", "active": False})
    return validated


# Filename hardening: kebab-case ASCII, optional single folder segment,
# .md suffix, capped length. Anything else is treated as a triage failure
# so it accrues toward dead-letter instead of crashing later in apply.
_FILENAME_PATTERN = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?/)?"  # optional folder/
    r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"  # stem
    r"(?:\.md)?$"  # optional .md
)
_MAX_FILENAME_LEN = 200
_TRIAGE_PLAN_FIELDS = frozenset({"type", "filename", "title", "keywords", "summary"})


def _triage_plan_validation_issues(
    value: Any,
    *,
    resolve_effective_targets: bool = False,
) -> list[ValidationIssue]:
    """Return exact semantic-plan violations for same-session repair.

    JSON Schema owns the basic container and scalar types.  This host
    validator adds operation-aware and whitespace-aware rules that the
    intentionally small grammar schema cannot express. Keeping the same checks
    here and in
    :func:`_validate_triage_plan` prevents a schema-valid but unusable plan from
    escaping the repair session and becoming an opaque outer retry.
    """

    issues = validate_json(value, _TRIAGE_PLAN_VALIDATION_SCHEMA)
    if not isinstance(value, list):
        return issues
    root_limit = next(
        (
            issue
            for issue in issues
            if issue.pointer == "" and issue.keyword == "maxItems"
        ),
        None,
    )
    if root_limit is not None:
        return [root_limit]
    try:
        plan_bytes = len(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    except (TypeError, ValueError):
        plan_bytes = 0
    if plan_bytes > _TRIAGE_MAX_OUTPUT_BYTES:
        return [
            ValidationIssue(
                pointer="",
                keyword="maxUtf8Bytes",
                expected=_TRIAGE_MAX_OUTPUT_BYTES,
                received={"type": "array", "utf8_bytes": plan_bytes},
                message="triage plan exceeds the fixed UTF-8 output budget",
            )
        ]
    for index, entry in enumerate(value):
        pointer = f"/{index}"
        if not isinstance(entry, dict):
            continue

        op_type = entry.get("type")
        filename = entry.get("filename")
        if isinstance(filename, str):
            fn = filename.strip()
            filename_error: tuple[str, Any, str] | None = None
            if filename and not fn:
                filename_error = (
                    "minLength",
                    1,
                    "filename must contain a non-whitespace character",
                )
            elif any(ord(char) < 0x20 or char == "\x7f" for char in fn):
                filename_error = (
                    "pattern",
                    "no ASCII control characters",
                    "filename contains a control character",
                )
            elif ".." in fn.split("/"):
                filename_error = (
                    "pattern",
                    "no parent traversal segment",
                    "filename contains a parent-traversal segment",
                )
            elif op_type == "create" and not _FILENAME_PATTERN.fullmatch(fn):
                filename_error = (
                    "pattern",
                    _FILENAME_PATTERN.pattern,
                    "create filename must be ASCII kebab-case with at most one folder",
                )
            if filename_error is not None:
                keyword, expected, message = filename_error
                issues.append(
                    ValidationIssue(
                        pointer=f"{pointer}/filename",
                        keyword=keyword,
                        expected=expected,
                        received={"type": "string", "value": filename},
                        message=message,
                    )
                )
        for field in ("title", "summary"):
            field_value = entry.get(field)
            if isinstance(field_value, str) and field_value and not field_value.strip():
                issues.append(
                    ValidationIssue(
                        pointer=f"{pointer}/{field}",
                        keyword="minLength",
                        expected=1,
                        received={"type": "string", "value": field_value},
                        message=f"{field} must contain a non-whitespace character",
                    )
                )
        keywords = entry.get("keywords")
        if isinstance(keywords, list):
            for keyword_index, keyword in enumerate(keywords):
                if not isinstance(keyword, str) or not keyword or keyword.strip():
                    continue
                issues.append(
                    ValidationIssue(
                        pointer=f"{pointer}/keywords/{keyword_index}",
                        keyword="minLength",
                        expected=1,
                        received={"type": "string", "value": keyword},
                        message="keyword must contain a non-whitespace character",
                    )
                )
        if op_type == "create":
            required_fields = (
                ("title", "a non-empty page title string"),
                ("summary", "a non-empty grounded summary string"),
                ("keywords", "a non-empty list of non-empty keyword strings"),
            )
            for field, description in required_fields:
                if field in entry:
                    continue
                issues.append(
                    ValidationIssue(
                        pointer=f"{pointer}/{field}",
                        keyword="required",
                        expected=description,
                        received={"type": "missing"},
                        message=f"create operation requires {description}",
                    )
                )
    if issues:
        return issues

    operations = [entry for entry in value if isinstance(entry, dict)]
    _collapsed, collisions = distinct_target_collisions(
        operations,
        effective_filename=(
            _effective_triage_target_filename if resolve_effective_targets else None
        ),
    )
    for collision in collisions:
        issues.append(
            ValidationIssue(
                pointer="",
                keyword="uniqueTarget",
                expected={
                    "rule": (
                        "exactly one operation per case/Unicode-insensitive "
                        "target page_id"
                    ),
                    "repair": (
                        "return one operation that preserves every distinct fact, "
                        "summary, and keyword for this target"
                    ),
                },
                received=collision,
                message=(
                    "multiple distinct operations in your complete previous JSON "
                    "response resolve to one target page; use the listed indices and "
                    "merge every distinct fact, summary, and keyword into one "
                    "operation in this response. Do not drop an operation or split "
                    "the same page_id across folders."
                ),
            )
        )
    return issues


def _validate_triage_plan(
    plan: list,
    *,
    coerce_missing_updates: bool = False,
) -> list[dict] | None:
    """Reject any plan that doesn't match the documented operation schema.

    Validation is **op-type aware**:

    * ``create``: filename must be ASCII kebab-case (``[a-z0-9-]``,
      ≤200 chars, optional single folder segment, optional ``.md``).
      We're choosing the canonical id for a brand-new page, so strict
      hygiene is appropriate.
    * ``update``: filename may point at a legacy page predating the kebab
      rule (e.g. ``Foo.md``, ``snake_case.md``, non-ASCII titles), so the
      strict create regex would block valid updates forever. We only reject
      control characters and length blowups here.
    * When ``coerce_missing_updates`` is enabled for live triage, a model
      "update" for a missing but create-safe page id is retyped to
      ``create`` before generation. That avoids a later apply-stage
      quarantine while still leaving legacy or ambiguous update targets
      fail-closed.

    Anything else (string entries, nonsense types, missing fields, control
    chars) returns ``None`` so the caller treats it as an operational triage
    contract failure while leaving the immutable raw in place.

    Empty plan ([]) is valid and means "nothing wiki-worthy".
    """
    if not isinstance(plan, list):
        return None
    if _triage_plan_validation_issues(plan):
        return None
    cleaned: list[dict] = []
    for entry in plan:
        if not isinstance(entry, dict):
            return None
        if set(entry) - _TRIAGE_PLAN_FIELDS:
            return None
        op_type = entry.get("type")
        if op_type not in ("create", "update"):
            return None
        filename = entry.get("filename")
        if not isinstance(filename, str):
            return None
        fn = filename.strip()
        if not fn or len(fn) > _MAX_FILENAME_LEN:
            return None
        # Reject any control char (NUL, newline, tab, etc.) for any op type.
        if any(ord(c) < 0x20 or c == "\x7f" for c in fn):
            return None
        # Reject path-traversal markers up front for both op types — the
        # apply layer would catch these but we want them to count as a
        # triage failure (LLM produced garbage), not an apply failure.
        if ".." in fn.split("/"):
            return None
        if op_type == "create":
            if not _FILENAME_PATTERN.fullmatch(fn):
                return None
        # For update we don't enforce kebab — apply will look the page up
        # via find_page() (case-insensitive on macOS APFS) and reject if
        # the target doesn't exist. That way legacy corpus stays updatable.
        cleaned.append(entry)
    # Repeated byte-equivalent operations carry no additional semantic intent
    # and are safe to normalize deterministically. Any distinct operation for
    # the same page_id was rejected above and repaired inside the same local
    # structured session; it must never be silently dropped here.
    cleaned = collapse_exact_duplicate_operations(cleaned)
    if coerce_missing_updates:
        return _normalize_triage_plan(cleaned)
    return cleaned


def _title_from_page_id(page_id: str) -> str:
    words = re.sub(r"[^a-zA-Z0-9]+", " ", page_id).strip().split()
    title = " ".join(word if word.isdigit() else word.capitalize() for word in words)
    return title or page_id


def _keyword_fallback_from_page_id(page_id: str) -> list[str]:
    return [word for word in re.split(r"[^a-zA-Z0-9]+", page_id.strip()) if word]


def _filename_allowed_for_create(filename: str) -> bool:
    fn = filename.strip()
    if not fn.endswith(".md"):
        fn += ".md"
    return bool(_FILENAME_PATTERN.fullmatch(fn))


def _normalize_triage_plan(plan: list[dict]) -> list[dict]:
    """Repair safe triage op-type drift before generate/apply.

    The triage prompt says updates must reference an existing page, but local
    models sometimes choose ``update`` for a brand-new durable topic. Waiting
    until apply turns that into a repeated raw quarantine. If the target is
    definitely absent and the requested filename is valid for a new page, treat
    it as a create; ambiguous or unsafe names still fail closed later.
    """
    normalized: list[dict] = []
    for op in plan:
        if op.get("type") != "update":
            normalized.append(op)
            continue

        filename = op.get("filename")
        if not isinstance(filename, str) or not _filename_allowed_for_create(filename):
            normalized.append(op)
            continue

        try:
            full_path = _safe_resolve_page_path(filename)
            page_id = full_path.stem
            existing_path = (
                full_path if full_path.exists() else _find_page_resilient(page_id)
            )
        except IngestApplyError:
            normalized.append(op)
            continue

        if existing_path is not None and existing_path.exists():
            normalized.append(op)
            continue

        create_op = dict(op)
        create_op["type"] = "create"
        create_op.setdefault("title", _title_from_page_id(page_id))
        summary = create_op.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            # This is a topic label, not a generated factual claim.  It keeps
            # update-to-create coercion inside the same strict plan contract
            # without inventing a semantic summary that was absent upstream.
            create_op["summary"] = str(create_op["title"])
        keywords = create_op.get("keywords")
        if not (
            isinstance(keywords, list)
            and all(isinstance(keyword, str) for keyword in keywords)
            and keywords
        ):
            create_op["keywords"] = _keyword_fallback_from_page_id(page_id)
        normalized.append(create_op)
        _safe_log(
            f"ingest | triage update target {page_id!r} missing; converted to create"
        )

    return normalized


def _normalize_match_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    import unicodedata

    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[\W_]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def _op_title_or_slug(op: dict) -> str:
    title = op.get("title")
    if isinstance(title, str) and title.strip():
        return title
    filename = op.get("filename")
    if isinstance(filename, str) and filename.strip():
        try:
            return _title_from_page_id(_safe_resolve_page_path(filename).stem)
        except IngestApplyError:
            return Path(filename).stem
    return ""


def _relative_page_filename(path: Path) -> str:
    try:
        return str(path.relative_to(PAGES_DIR))
    except ValueError:
        return path.name


def _existing_candidate_metas() -> list[dict]:
    try:
        from llm_wiki_mcp.index_store import get_store

        store = get_store()
        store.refresh()
        metas: list[dict] = []
        for item in store.all_pages_meta(include_system=False):
            if item.get("page_type") == "reference":
                continue
            if item.get("status") not in (None, "active"):
                continue
            meta = store.meta(str(item.get("page_id", "")))
            if meta is not None:
                metas.append(meta)
        return metas
    except Exception:
        return []


def _candidate_score_for_create(op: dict, meta: dict) -> tuple[float, str]:
    requested_title = _normalize_match_text(_op_title_or_slug(op))
    existing_title = _normalize_match_text(meta.get("title"))
    requested_filename = op.get("filename")
    requested_slug = ""
    if isinstance(requested_filename, str):
        try:
            requested_slug = _normalize_for_loose_page_id(
                _safe_resolve_page_path(requested_filename).stem
            )
        except IngestApplyError:
            requested_slug = _normalize_for_loose_page_id(Path(requested_filename).stem)
    existing_slug = _normalize_for_loose_page_id(str(meta.get("page_id", "")))

    if requested_slug and requested_slug == existing_slug:
        return 1.0, "same-page-id"
    if requested_title and existing_title and requested_title == existing_title:
        return 1.0, "same-title"

    title_score = (
        SequenceMatcher(None, requested_title, existing_title).ratio()
        if requested_title and existing_title
        else 0.0
    )
    slug_score = (
        SequenceMatcher(None, requested_slug, existing_slug).ratio()
        if requested_slug and existing_slug
        else 0.0
    )
    if title_score >= 0.92:
        return title_score, "near-title"
    if slug_score >= 0.95:
        return slug_score, "near-page-id"
    return max(title_score, slug_score), "below-threshold"


def _search_candidate_metas(op: dict) -> list[dict]:
    query_parts = [
        _op_title_or_slug(op),
        op.get("summary") if isinstance(op.get("summary"), str) else "",
    ]
    keywords = op.get("keywords")
    if isinstance(keywords, list):
        query_parts.extend(str(k) for k in keywords if isinstance(k, str))
    query = " ".join(part for part in query_parts if part).strip()
    if not query:
        return []

    try:
        from llm_wiki_mcp.index_store import get_store
        from llm_wiki_mcp.search import search

        results, _mode = search(query, top_n=5, semantic=True)
        store = get_store()
        metas: list[dict] = []
        for result in results:
            meta = store.meta(result.page_id)
            if meta is not None and meta.get("page_type") != "reference":
                metas.append(meta)
        return metas
    except Exception:
        return []


def _find_existing_create_target(op: dict) -> tuple[Path, str, float] | None:
    filename = op.get("filename")
    if not isinstance(filename, str):
        return None
    try:
        requested_path = _safe_resolve_page_path(filename)
        existing = _find_page_resilient(requested_path.stem)
        if existing is not None and existing.exists():
            return existing, "same-page-id", 1.0
    except IngestApplyError:
        return None

    best_path: Path | None = None
    best_reason = ""
    best_score = 0.0
    for meta in _existing_candidate_metas():
        score, reason = _candidate_score_for_create(op, meta)
        if score > best_score:
            try:
                best_path = Path(str(meta["path"]))
            except (KeyError, TypeError):
                best_path = None
            best_reason = reason
            best_score = score

    if best_path is not None and best_score >= 0.92:
        return best_path, best_reason, best_score

    for meta in _search_candidate_metas(op):
        score, reason = _candidate_score_for_create(op, meta)
        if score >= 0.88:
            try:
                return Path(str(meta["path"])), f"search-{reason}", score
            except (KeyError, TypeError):
                continue
    return None


def _effective_triage_target_filename(op: dict) -> str | None:
    """Resolve the target filename exactly as search-before-create will.

    This runs inside the triage structured-session validator. It lets the
    model repair two differently named create proposals that both resolve to
    one existing page before either proposal spends a generation call.
    """

    filename = op.get("filename")
    if not isinstance(filename, str):
        return None
    if op.get("type") != "create":
        return filename
    match = _find_existing_create_target(op)
    if match is None:
        return filename
    existing_path, _reason, _score = match
    return _relative_page_filename(existing_path)


def _dedupe_create_ops_with_existing(plan: list[dict], raw_content: str) -> list[dict]:
    """Convert high-confidence duplicate create ops into updates before generation."""
    del raw_content  # Reserved for a future content-similarity gate.
    rewritten: list[dict] = []
    for op in plan:
        if op.get("type") != "create":
            rewritten.append(op)
            continue

        match = _find_existing_create_target(op)
        if match is None:
            rewritten.append(op)
            continue

        existing_path, reason, score = match
        update_op = dict(op)
        update_op["type"] = "update"
        update_op["filename"] = _relative_page_filename(existing_path)
        update_op["existing_page_id"] = existing_path.stem
        update_op["dedupe_reason"] = reason
        rewritten.append(update_op)
        _safe_log(
            "ingest | search-before-create: create "
            f"{op.get('filename', '?')!r} -> update "
            f"{_relative_page_filename(existing_path)!r} "
            f"({reason}, score={score:.2f})"
        )
    return rewritten


# ---------------------------------------------------------------------------
# Stage 2: Generate — produce each page with focused context
# ---------------------------------------------------------------------------


def _complete_related_context(
    heading: str,
    paths: list[Path],
    *,
    max_bytes: int | None,
    excluded: set[Path] | None = None,
) -> list[str]:
    """Select only complete related-page blocks within a deterministic budget.

    Related pages are optional retrieval context, but a partial page is
    misleading evidence.  Skip a page when its complete UTF-8 bytes do not fit
    instead of prefix-truncating it.  The current update target is handled
    separately and is never subject to this budget.
    """

    selected: list[str] = []
    used_bytes = 0
    excluded_paths = {path.resolve(strict=False) for path in (excluded or set())}
    for path in paths:
        if path.resolve(strict=False) in excluded_paths:
            continue
        block = "\n".join(
            [
                f"--- [[{page_id_from_path(path)}]] ---",
                path.read_text(),
            ]
        )
        block_bytes = len(block.encode("utf-8"))
        if max_bytes is not None and used_bytes + block_bytes > max_bytes:
            continue
        selected.append(block)
        used_bytes += block_bytes
    return [heading, *selected] if selected else []


def _build_focused_context(
    op: dict,
    raw_content: str,
    *,
    max_bytes: int | None = None,
) -> str:
    """Build context focused on a single page operation."""
    lines: list[str] = []
    current_path: Path | None = None

    # For updates, include the current page content
    if op.get("type") == "update":
        filename = op.get("filename", "")
        page_id = filename.replace(".md", "").split("/")[-1]
        existing_path = find_page(page_id)
        if existing_path:
            current_path = existing_path
            lines.append(f"--- Current content of [[{page_id}]] ---")
            lines.append(existing_path.read_text())
            lines.append("--- End current content ---\n")

    # Search for related pages using keywords from the triage plan
    keywords = op.get("keywords", [])
    if keywords:
        related = _search_related_pages(keywords, top_n=5)
        if related:
            lines.extend(
                _complete_related_context(
                    "Related existing pages for cross-referencing:",
                    related,
                    max_bytes=max_bytes,
                    excluded={current_path} if current_path is not None else None,
                )
            )
    elif op.get("type") == "create":
        # For creates without keywords, use title words as fallback
        title = op.get("title", "")
        if title:
            title_keywords = [w for w in title.split() if len(w) >= 2]
            related = _search_related_pages(title_keywords, top_n=3)
            if related:
                lines.extend(
                    _complete_related_context(
                        "Related existing pages:",
                        related,
                        max_bytes=max_bytes,
                    )
                )

    return "\n".join(lines)


_MAX_PAGE_GENERATION_REPAIR_TURNS = 2
_MAX_PAGE_GENERATION_RESPONSES = 1 + _MAX_PAGE_GENERATION_REPAIR_TURNS
_MAX_PAGE_REPAIR_FEEDBACK_BYTES = 2_400
_PAGE_GENERATION_CONTEXT_SAFETY_TOKENS = 256


def _required_generate_context_tokens(
    prompt: str,
    system: str | None,
    *,
    num_predict: int,
) -> int:
    """Return a fail-closed bound for the complete page-repair session.

    Initial input uses UTF-8 bytes as a tokenizer-independent upper bound.
    Every prior model completion is bounded by ``num_predict`` model tokens;
    reserve two such assistant turns plus bounded validator feedback and the
    final completion.  This prevents Ollama from shifting/truncating the
    original evidence when page validation needs a second or third turn.
    """

    prompt_bytes = len(prompt.encode("utf-8"))
    system_bytes = len(system.encode("utf-8")) if system else 0
    repair_history = _MAX_PAGE_GENERATION_REPAIR_TURNS * (
        num_predict + _MAX_PAGE_REPAIR_FEEDBACK_BYTES + 64
    )
    return (
        prompt_bytes
        + system_bytes
        + repair_history
        + num_predict
        + _PAGE_GENERATION_CONTEXT_SAFETY_TOKENS
    )


def _generation_completion_failure(
    response: ollama_runtime.GenerateResponse,
) -> tuple[str, str] | None:
    """Validate Ollama's explicit terminal metadata before page parsing."""

    reason = (response.done_reason or "").strip().casefold().replace("-", "_")
    known_limit = {
        "length",
        "max_length",
        "max_token",
        "max_tokens",
        "num_predict",
        "token_limit",
    }
    if reason in known_limit or (
        reason and ("token" in reason or "length" in reason) and reason != "stop"
    ):
        return (
            "output_truncated",
            f"Ollama stopped at an output limit (done_reason={reason!r})",
        )
    if response.done is not True:
        return (
            "stream_incomplete" if response.streamed else "completion_incomplete",
            "Ollama response did not contain an explicit completed turn "
            f"(done={response.done!r}, done_reason={response.done_reason!r})",
        )
    return None


def _search_related_pages(
    keywords: list[str], min_score: float = 0.5, top_n: int = 8
) -> list[Path]:
    """Search for related pages using keywords. Uses BM25 if available, falls back to simple matching."""
    try:
        from llm_wiki_mcp.search import get_bm25

        bm25 = get_bm25()
        bm25.build()
        results = bm25.query(" ".join(keywords), top_n=top_n)
        return [find_page(r.page_id) for r in results if find_page(r.page_id)]
    except Exception:
        pass

    # Fallback: simple keyword matching
    query_terms = [k.lower() for k in keywords]
    scored = []
    for path in all_pages():
        content = path.read_text()
        content_lower = content.lower()
        fm_match = re.search(r"title:\s*(.+)", content)
        title = fm_match.group(1) if fm_match else path.stem
        title_lower = title.lower()

        score = 0.0
        for term in query_terms:
            if term in title_lower:
                score += 0.5
            if term in path.stem.lower().replace("-", " "):
                score += 0.3
            count = content_lower.count(term)
            if count > 0:
                score += min(0.1 * count, 0.4)

        if score >= min_score:
            scored.append((score, path))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [path for _, path in scored[:top_n]]


_FRONTMATTER_BLOCK_RE = re.compile(r"^---\n.*?\n---(?:\n|$)", re.MULTILINE | re.DOTALL)


def _has_frontmatter(text: str) -> bool:
    """True if ``text`` starts with a ``---\\n...\\n---\\n`` block containing ``title:``."""
    if not text.startswith("---\n"):
        return False
    m = _FRONTMATTER_BLOCK_RE.match(text)
    if not m:
        return False
    return bool(re.search(r"^title:\s*\S", m.group(0), re.MULTILINE))


def _strip_all_frontmatter(text: str) -> str:
    """Remove every ``---\\n...\\n---\\n`` block from ``text``.

    Used as a defensive scrub on update bodies: even if the model ignored
    UPDATE_SYSTEM_PROMPT and wrote frontmatter, we drop it before append.
    """
    return _FRONTMATTER_BLOCK_RE.sub("", text)


@dataclass(frozen=True)
class _PageBodyValidation:
    body: str | None
    failure_class: str | None = None
    reason: str | None = None


def _validate_generated_page_output(
    output: str,
    op_type: str = "create",
) -> _PageBodyValidation:
    """Validate and extract one complete page block with an exact reason.

    Op-type rules:
    - ``create``: body MUST contain a frontmatter block with ``title:``. We accept
      strict (`=== NEW PAGE: ... ===`) or lenient (`=== anything ===`) wrappers,
      but always require an explicit ``=== END PAGE ===`` close before content
      can be persisted.
    - ``update``: body MUST NOT contain a frontmatter block. Stray FM is
      stripped here as a belt-and-braces against UPDATE_SYSTEM_PROMPT
      drift. The explicit close is required and an empty body after stripping
      is rejected.
    """
    if not isinstance(output, str) or not output.strip():
        return _PageBodyValidation(
            None,
            "empty_output",
            "the response was empty; return one complete page block",
        )

    op_type = (op_type or "create").lower()
    text = output.replace("\r\n", "\n").replace("\r", "\n").strip()
    if text.startswith("```"):
        if re.search(r"\n```\s*$", text) is None:
            return _PageBodyValidation(
                None,
                "incomplete_markdown_fence",
                "the outer markdown code fence was not closed",
            )
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    end_markers = re.findall(
        r"^===\s*END\s+PAGE\s*===\s*$",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    if (
        len(end_markers) != 1
        or re.search(
            r"\n===\s*END\s+PAGE\s*===\s*\Z",
            text,
            re.IGNORECASE,
        )
        is None
    ):
        return _PageBodyValidation(
            None,
            "missing_end_marker",
            "the response must contain exactly one `=== END PAGE ===` marker "
            "as its final non-whitespace line",
        )

    body: str | None = None

    # 1. Strict: === NEW/UPDATE PAGE: filename === ... === END PAGE ===
    m = re.fullmatch(
        r"===\s*(?:NEW|UPDATE)\s+PAGE:\s*\S+\s*===\n(.*?)\n===\s*END\s+PAGE\s*===",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        body = m.group(1).strip() or None

    # 2. Lenient: === <anything> === ... === END PAGE === (NEW PAGE: keyword dropped)
    if body is None:
        m = re.fullmatch(
            r"===[^\n]*===\n(.*?)\n===\s*END\s+PAGE\s*===",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            body = m.group(1).strip() or None

    if body is None:
        expected = "NEW" if op_type == "create" else "UPDATE"
        return _PageBodyValidation(
            None,
            "missing_page_wrapper",
            f"the response must start with `=== {expected} PAGE: <filename> ===`",
        )

    # Op-type sanity: enforce frontmatter contract.
    if op_type == "create":
        if not _has_frontmatter(body):
            return _PageBodyValidation(
                None,
                "missing_create_frontmatter",
                "a CREATE body requires a closed frontmatter block with a non-empty title",
            )
        return _PageBodyValidation(body)

    # update: drop any stray FM blocks, reject if nothing meaningful is left.
    cleaned = _strip_all_frontmatter(body).strip()
    if not cleaned:
        return _PageBodyValidation(
            None,
            "empty_update_body",
            "an UPDATE must contain new markdown after removing forbidden frontmatter",
        )
    # A partial frontmatter (opening `---` with no closing) cannot be safely
    # stripped; refuse rather than appending it raw.
    has_open_fence = bool(re.match(r"^---\s*$", cleaned, re.MULTILINE))
    has_closed_block = bool(_FRONTMATTER_BLOCK_RE.search(cleaned))
    if has_open_fence and not has_closed_block:
        return _PageBodyValidation(
            None,
            "partial_update_frontmatter",
            "the UPDATE contains an unclosed frontmatter delimiter",
        )
    return _PageBodyValidation(cleaned)


def _extract_page_body(output: str, op_type: str = "create") -> str | None:
    """Compatibility wrapper returning only a validated page body."""

    return _validate_generated_page_output(output, op_type=op_type).body


def _repair_transport_attested_page_boundary(
    output: str,
    response: ollama_runtime.GenerateResponse,
    *,
    op_type: str,
) -> tuple[str, _PageBodyValidation] | None:
    """Restore only a missing terminal marker on an explicit completed turn.

    The marker is a serialization boundary, not source evidence.  Ollama's
    ``done=True`` plus ``done_reason=stop`` is a stronger transport-level
    completion signal than asking the model to echo the same fixed suffix on
    another probabilistic turn.  We still fail closed unless adding that one
    line makes the *entire* strict page validator pass; existing/misplaced end
    markers, partial frontmatter, incomplete fences, and malformed wrappers
    therefore remain model-repair failures.
    """

    reason = (response.done_reason or "").strip().casefold().replace("-", "_")
    if response.done is not True or reason != "stop":
        return None
    initial = _validate_generated_page_output(output, op_type=op_type)
    if initial.failure_class != "missing_end_marker":
        return None
    if re.search(
        r"^===\s*END\s+PAGE\s*===\s*$",
        normalized := output.replace("\r\n", "\n").replace("\r", "\n"),
        re.MULTILINE | re.IGNORECASE,
    ):
        return None
    marker_like = re.compile(r"^\s*=+\s*END\b", re.IGNORECASE)
    if any(marker_like.match(line) for line in normalized.split("\n")):
        return None
    repaired = output.rstrip() + "\n=== END PAGE ==="
    validation = _validate_generated_page_output(repaired, op_type=op_type)
    if validation.body is None:
        return None
    return repaired, validation


def _page_generation_transcript(messages: list[dict[str, str]]) -> str:
    """Serialize client-side page history for Ollama's generate endpoint."""

    return "\n\n".join(
        f"<{message['role'].upper()}>\n{message['content']}" for message in messages
    )


def _page_generation_repair_prompt(
    validation: _PageBodyValidation,
    *,
    op_type: str,
    filename: str,
) -> str:
    """Build bounded, targeted feedback for the next logical turn."""

    expected_wrapper = "NEW" if op_type == "create" else "UPDATE"
    failure_class = validation.failure_class or "invalid_page_block"
    reason = validation.reason or "the page block failed deterministic validation"
    prompt = f"""Your previous response was rejected by the deterministic page validator.

Validator errors:
- code: {failure_class}
  reason: {reason}

Return a complete replacement response for `{filename}`. Do not describe the
fix and do not return a patch. Start with exactly
`=== {expected_wrapper} PAGE: {filename} ===` and finish with the exact final
line `=== END PAGE ===`. Preserve only facts grounded in the original source.
"""
    if len(prompt.encode("utf-8")) > _MAX_PAGE_REPAIR_FEEDBACK_BYTES:
        raise RuntimeError(
            "ingest generation feedback_too_large: page validator feedback "
            "exceeded the fixed repair cap"
        )
    return prompt


def _generate_one(
    op: dict,
    raw_content: str,
    *,
    raw_keywords: list[str] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    frontier_feedback: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict | None:
    """Stage 2: generate one page; return an operation dict ready for apply.

    ``raw_keywords`` is a side-channel list lifted from the source raw's
    frontmatter (not from triage). It rides on the returned operation dict
    so the apply layer can patch it onto the page frontmatter without an
    extra LLM round-trip. ``None`` means "no metadata propagation" — the
    field is omitted from the output, distinguishing it from an explicit
    empty list which would survive as ``[]``.
    """
    config = load_ingest_config()
    context_kwargs: dict[str, Any] = {}
    if _supports_keyword(_build_focused_context, "max_bytes"):
        context_kwargs["max_bytes"] = config.max_related_context_bytes
    context = _build_focused_context(op, raw_content, **context_kwargs)

    op_type = op.get("type", "create").lower()
    if op_type not in ("create", "update"):
        op_type = "create"
    filename = op.get("filename", "unknown.md")
    summary = op.get("summary", "")
    title = op.get("title", "")
    current_date = date.today().isoformat()

    feedback_block = ""
    if frontier_feedback:
        feedback_block = f"""

---
Previous local consensus review (authoritative correction instructions):
---
{frontier_feedback}
---
Rewrite this operation as the smallest grounded change. Do not infer missing
details, future plans, causal explanations, preferences, or outcomes that are
not explicit in the raw evidence.
"""

    prompt = f"""{context}

---
Raw session data (source material):
---
{raw_content}
---

Task: {op_type.upper()} page "{filename}"
Title: {title}
Summary: {summary}
{feedback_block}

Current date: {current_date}
For CREATE, use this exact date for the `updated` frontmatter field.
Do not add or infer any other date unless it is explicit in the raw evidence.
For UPDATE, do not create a dated heading unless that date is explicit in the raw evidence.

Generate the page content based on the raw data and context above."""

    system_prompt = (
        UPDATE_SYSTEM_PROMPT if op_type == "update" else GENERATE_SYSTEM_PROMPT
    )
    required_num_ctx = _required_generate_context_tokens(
        prompt,
        system_prompt,
        num_predict=config.num_predict,
    )
    attempts_made = 0
    try:
        selected_num_ctx = _select_ingest_context(
            required_num_ctx,
            num_ctx=config.num_ctx,
            max_num_ctx=config.max_num_ctx,
        )
    except IngestContextCapacityError as exc:
        if diagnostics is not None:
            diagnostics.update(
                {
                    "failure_class": "context_window_exceeded",
                    "reason": str(exc),
                    "attempts": 0,
                }
            )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "error",
                    "active": False,
                    "failure_class": "context_window_exceeded",
                    "error": str(exc),
                }
            )
        raise RuntimeError(f"ingest generation context_window_exceeded: {exc}") from exc

    try:
        live_transport = _generate_with_progress is _DEFAULT_GENERATE_WITH_PROGRESS
        lease = (
            ollama_runtime.model_resource_lease(exclusive=True)
            if live_transport
            else nullcontext()
        )
        with lease:
            if live_transport:
                try:
                    selected_num_ctx = _admit_ingest_context(
                        config,
                        selected_num_ctx,
                    )
                except IngestTriageFailure as exc:
                    raise RuntimeError(
                        "ingest generation capacity_unavailable: " + exc.reason
                    ) from exc
            generate_kwargs: dict[str, Any] = {
                "system": system_prompt,
                "progress_callback": progress_callback,
            }
            optional = {
                "model": config.model,
                "num_ctx": selected_num_ctx,
                "num_predict": config.num_predict,
                "keep_alive": config.keep_alive,
                "read_timeout_ms": config.read_timeout_ms,
                "temperature": config.temperature,
            }
            for name, value in optional.items():
                if _supports_keyword(_generate_with_progress, name):
                    generate_kwargs[name] = value
            if _supports_keyword(_generate_with_progress, "return_metadata"):
                generate_kwargs["return_metadata"] = True
            messages = [{"role": "user", "content": prompt}]
            seen_output_hashes: set[str] = set()
            for attempt_index in range(_MAX_PAGE_GENERATION_RESPONSES):
                attempts_made = attempt_index + 1
                output = _generate_with_progress(
                    _page_generation_transcript(messages),
                    **generate_kwargs,
                )
                response_metadata: ollama_runtime.GenerateResponse | None = None
                if isinstance(output, ollama_runtime.GenerateResponse):
                    response_metadata = output
                    completion_failure = _generation_completion_failure(output)
                    if completion_failure is not None:
                        failure_class, failure_reason = completion_failure
                        if diagnostics is not None:
                            diagnostics.update(
                                {
                                    "failure_class": failure_class,
                                    "reason": failure_reason,
                                    "attempts": attempt_index + 1,
                                }
                            )
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "event": "error",
                                    "active": False,
                                    "failure_class": failure_class,
                                    "error": failure_reason,
                                }
                            )
                        # A partial completion is not valid conversational
                        # history.  Never ask the model to repair text whose
                        # terminal boundary is unknown.
                        raise RuntimeError(
                            f"ingest generation {failure_class}: {failure_reason}"
                        )
                    if (
                        output.prompt_eval_count is not None
                        and output.prompt_eval_count
                        >= selected_num_ctx - _PAGE_GENERATION_CONTEXT_SAFETY_TOKENS
                    ) or (
                        output.prompt_eval_count is not None
                        and output.eval_count is not None
                        and output.prompt_eval_count + output.eval_count
                        > selected_num_ctx
                    ):
                        failure_reason = (
                            "Ollama context accounting reached or crossed the "
                            "admitted page-generation context"
                        )
                        if diagnostics is not None:
                            diagnostics.update(
                                {
                                    "failure_class": "context_truncation_suspected",
                                    "reason": failure_reason,
                                    "attempts": attempt_index + 1,
                                }
                            )
                        raise RuntimeError(
                            "ingest generation context_truncation_suspected: "
                            + failure_reason
                        )
                    output = output.content
                if not isinstance(output, str):
                    if diagnostics is not None:
                        diagnostics.update(
                            {
                                "failure_class": "completion_incomplete",
                                "reason": "transport returned non-string content",
                                "attempts": attempt_index + 1,
                            }
                        )
                    raise RuntimeError(
                        "ingest generation completion_incomplete: "
                        "transport returned non-string content"
                    )

                validation = _validate_generated_page_output(output, op_type=op_type)
                if response_metadata is not None:
                    boundary_repair = _repair_transport_attested_page_boundary(
                        output,
                        response_metadata,
                        op_type=op_type,
                    )
                    if boundary_repair is not None:
                        output, validation = boundary_repair
                        if diagnostics is not None:
                            diagnostics["transport_boundary_repaired"] = True
                        _safe_log(
                            "ingest | restored transport-attested end marker for "
                            f"{filename}"
                        )
                output_sha256 = hashlib.sha256(output.encode("utf-8")).hexdigest()
                if validation.body is not None:
                    if diagnostics is not None:
                        diagnostics.update(
                            {
                                "failure_class": None,
                                "reason": None,
                                "attempts": attempt_index + 1,
                                "repair_turns": attempt_index,
                                "output_sha256": output_sha256,
                            }
                        )
                    result: dict = {
                        "type": op_type,
                        "filename": filename,
                        "content": validation.body,
                    }
                    if raw_keywords is not None:
                        result["raw_keywords"] = list(raw_keywords)
                    return result

                _safe_log(
                    "ingest | generate validation failed for "
                    f"{filename} ({op_type}, {validation.failure_class}, "
                    f"attempt {attempt_index + 1}/{_MAX_PAGE_GENERATION_RESPONSES})"
                )
                if output_sha256 in seen_output_hashes:
                    if diagnostics is not None:
                        diagnostics.update(
                            {
                                "failure_class": "repeated_output",
                                "reason": (
                                    "model repeated the same invalid page output "
                                    f"({validation.failure_class})"
                                ),
                                "attempts": attempt_index + 1,
                                "repair_turns": attempt_index,
                                "output_sha256": output_sha256,
                            }
                        )
                    _safe_log(
                        f"ingest | generate repair stopped for {filename}: "
                        "same invalid output hash repeated"
                    )
                    return None
                seen_output_hashes.add(output_sha256)
                if attempt_index == _MAX_PAGE_GENERATION_RESPONSES - 1:
                    if diagnostics is not None:
                        diagnostics.update(
                            {
                                "failure_class": "repair_exhausted",
                                "reason": validation.reason,
                                "attempts": attempt_index + 1,
                                "repair_turns": attempt_index,
                                "output_sha256": output_sha256,
                            }
                        )
                    return None

                repair_prompt = _page_generation_repair_prompt(
                    validation,
                    op_type=op_type,
                    filename=filename,
                )
                messages.append({"role": "assistant", "content": output})
                messages.append({"role": "user", "content": repair_prompt})
                _safe_log(
                    f"ingest | targeted generate repair {attempt_index + 1}/"
                    f"{_MAX_PAGE_GENERATION_REPAIR_TURNS} for {filename}: "
                    f"{validation.failure_class}"
                )
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "repair",
                            "active": True,
                            "repair_turn": attempt_index + 1,
                            "failure_class": validation.failure_class,
                        }
                    )
    except RuntimeError as e:
        if str(e).startswith(
            (
                "ingest generation capacity_unavailable:",
                "ingest generation context_window_exceeded:",
                "ingest generation context_truncation_suspected:",
                "ingest generation completion_incomplete:",
                "ingest generation stream_incomplete:",
                "ingest generation output_truncated:",
                "ingest generation feedback_too_large:",
            )
        ):
            _safe_log(f"ingest | generate preflight failed for {filename}: {e}")
            raise
        _safe_log(f"ingest | generate failed for {filename}: {e}")
        if diagnostics is not None:
            diagnostics.update(
                {
                    "failure_class": "transport_error",
                    "reason": f"{type(e).__name__}: {str(e)[:500]}",
                    "attempts": attempts_made,
                }
            )
        return None
    except Exception as e:
        _safe_log(f"ingest | generate failed for {filename}: {e}")
        if diagnostics is not None:
            diagnostics.update(
                {
                    "failure_class": "transport_error",
                    "reason": f"{type(e).__name__}: {str(e)[:500]}",
                    "attempts": attempts_made,
                }
            )
        return None
    return None


# ---------------------------------------------------------------------------
# Apply (link reconciliation, write phase, rollback)
# ---------------------------------------------------------------------------


def _fenced_code_spans(text: str) -> list[tuple[int, int]]:
    """Return spans of fenced code blocks, including unclosed ones at EOF.

    `_FENCED_CODE_RE` from ``link_fix`` requires a closing ```````,
    so a model that emits an opener without a closer (truncation, formatting
    error) leaves the rest of the body unprotected. Walk the text once,
    toggling on every ```````: if we end inside a fence, treat
    the trailing region as a fence too.
    """
    from llm_wiki_mcp.link_fix import fenced_code_spans

    return fenced_code_spans(text)


def _reconcile_links(content: str, allowed_ids: set[str]) -> tuple[str, dict]:
    """Repair or unwrap [[wiki-links]] in prose, leaving code & frontmatter alone.

    For each ``[[target|alias#anchor]]`` outside of frontmatter / fenced code /
    inline code:

      * target resolves → leave intact (anchor + alias preserved).
      * target has ``folder/`` or ``.md`` clutter that strips to a known id
        → rewrite to the canonical form.
      * target unresolvable → unwrap to plain text (alias if given, else target),
        so the body keeps the entity name without polluting the link graph.

    Code / frontmatter regions are detected via ``link_fix`` (the existing
    canonical implementation used by lint/server) so we never break
    ``x = data[[1]]`` or fenced examples. Unclosed fences (truncated LLM
    output) are also covered — we treat everything after the dangling
    opener as code.

    Returns ``(rewritten_content, stats)``.
    """
    from llm_wiki_mcp.link_fix import (
        WIKI_LINK_RE,
        position_in_spans,
        protected_spans,
    )

    stats = {"resolved": 0, "rewritten": 0, "unwrapped": 0}

    # Pre-compute byte spans we must NOT touch.
    skip_ranges = protected_spans(content)

    def replace(m: re.Match) -> str:
        if position_in_spans(m.start(), skip_ranges):
            return m.group(0)  # inside code/frontmatter — never rewrite

        inside = m.group(1)
        target_part, _, alias_raw = inside.partition("|")
        alias = alias_raw if "|" in inside else None
        if "#" in target_part:
            target, anchor_body = target_part.split("#", 1)
            anchor = "#" + anchor_body
        else:
            target, anchor = target_part, ""
        target = target.strip()

        if target in allowed_ids:
            stats["resolved"] += 1
            return m.group(0)

        # Try canonicalizing: strip a single leading folder and a trailing .md.
        candidate = target
        if "/" in candidate:
            candidate = candidate.rsplit("/", 1)[-1]
        if candidate.endswith(".md"):
            candidate = candidate[:-3]
        candidate = candidate.strip()
        if candidate and candidate != target and candidate in allowed_ids:
            stats["rewritten"] += 1
            tail = anchor + (f"|{alias}" if alias is not None else "")
            return f"[[{candidate}{tail}]]"

        # Unresolvable → unwrap to plain text. Keep alias as display, else target.
        stats["unwrapped"] += 1
        return alias if alias is not None else target

    new_content = WIKI_LINK_RE.sub(replace, content)
    return new_content, stats


class IngestApplyError(Exception):
    """Raised when an operation cannot be safely applied (fail-closed)."""


@dataclass(frozen=True)
class PreparedIngestOperation:
    """Exact page preimage and proposed postimage awaiting final review."""

    op_type: str  # "create" | "update"
    path: Path
    page_id: str
    new_body: str
    previous_text: str | None
    new_tags: tuple[str, ...] = ()
    source_operation_index: int = -1
    source_operation_type: str = ""
    source_filename: str = ""

    @property
    def previous_sha256(self) -> str | None:
        if self.previous_text is None:
            return None
        return hashlib.sha256(self.previous_text.encode("utf-8")).hexdigest()

    @property
    def new_sha256(self) -> str:
        return hashlib.sha256(self.new_body.encode("utf-8")).hexdigest()

    def review_payload(self) -> dict[str, Any]:
        """Return the full exact bytes, plus hashes, for frontier review."""

        return {
            "op_type": self.op_type,
            "path": self.path.relative_to(PAGES_DIR.resolve()).as_posix(),
            "page_id": self.page_id,
            "source_operation_index": self.source_operation_index,
            "source_operation_type": self.source_operation_type,
            "source_filename": self.source_filename,
            "preimage_exists": self.previous_text is not None,
            "previous_text": self.previous_text,
            "previous_sha256": self.previous_sha256,
            "proposed_text": self.new_body,
            "proposed_sha256": self.new_sha256,
            "new_tags": list(self.new_tags),
        }


def _safe_resolve_page_path(filename: str) -> Path:
    """Resolve ``filename`` to a path strictly under ``PAGES_DIR``.

    The triage stage is LLM-controlled and can be steered by adversarial
    raw content, so we treat its filenames as untrusted input. Reject:

      * absolute paths (``/etc/passwd``)
      * parent traversal (``../../etc/passwd``)
      * symlink-escape after resolution
      * empty / whitespace-only / dotfile-only filenames

    Returns the resolved path; raises :class:`IngestApplyError` otherwise.
    """
    if not filename or not filename.strip():
        raise IngestApplyError("empty filename")

    # Normalize the .md suffix so callers don't need to do it themselves.
    fn = filename.strip()
    if not fn.endswith(".md"):
        fn = fn + ".md"

    candidate = Path(fn)
    if candidate.is_absolute():
        raise IngestApplyError(f"absolute filename refused: {filename!r}")
    if any(part in ("..", "") for part in candidate.parts):
        raise IngestApplyError(f"parent-traversal filename refused: {filename!r}")
    # Disallow filenames that resolve outside PAGES_DIR (e.g. via symlink).
    pages_root = PAGES_DIR.resolve()
    full = (PAGES_DIR / candidate).resolve()
    try:
        full.relative_to(pages_root)
    except ValueError as e:
        raise IngestApplyError(f"filename escapes PAGES_DIR: {filename!r}") from e

    if full.name in (".md", ""):
        raise IngestApplyError(f"degenerate filename: {filename!r}")

    return full


def _normalize_for_collision(name: str) -> str:
    """Canonical key for case- and Unicode-insensitive collision detection.

    macOS's default APFS is case-insensitive AND can ship the same logical
    name in two byte representations (NFC vs NFD): ``café.md`` (NFC,
    one ``é``) and ``café.md`` (NFD, ``e`` + combining acute) resolve to
    the same file but compare as different strings. NFC-normalize first,
    then casefold.
    """
    import unicodedata

    return unicodedata.normalize("NFC", name).casefold()


def _normalize_for_loose_page_id(name: str) -> str:
    """Canonical key for legacy slug drift.

    This is deliberately used only after exact/casefold lookup fails.  It
    catches model-normalized variants such as ``opus-4.7`` → ``opus-4-7``
    without making fuzzy semantic guesses.
    """
    import unicodedata

    normalized = unicodedata.normalize("NFC", name).casefold()
    return re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")


def _find_page_casefold(page_id: str) -> Path | None:
    """find_page with macOS-case-insensitive + NFC-normalized semantics."""
    direct = find_page(page_id)
    if direct is not None:
        return direct
    target = _normalize_for_collision(page_id)
    for p in PAGES_DIR.rglob("*.md"):
        if _normalize_for_collision(p.stem) == target:
            return p
    return None


def _find_page_resilient(page_id: str, *, emit_logs: bool = True) -> Path | None:
    """Find a page by exact id first, then by safe single-candidate loose id."""
    existing = _find_page_casefold(page_id)
    if existing is not None:
        return existing

    try:
        from llm_wiki_mcp.alias_store import resolve_alias_path

        alias_target = resolve_alias_path(page_id)
    except Exception:
        alias_target = None
    if alias_target is not None:
        if emit_logs:
            _safe_log(
                f"ingest | resolved page_id {page_id!r} by alias "
                f"→ {alias_target.relative_to(PAGES_DIR)}"
            )
        return alias_target

    target = _normalize_for_loose_page_id(page_id)
    if not target:
        return None
    matches = [
        p
        for p in PAGES_DIR.rglob("*.md")
        if _normalize_for_loose_page_id(p.stem) == target
    ]
    if not matches:
        return None
    if len(matches) > 1:
        choices = ", ".join(sorted(str(p.relative_to(PAGES_DIR)) for p in matches[:5]))
        raise IngestApplyError(
            f"ambiguous loose page_id match for {page_id!r}: {choices}"
        )
    resolved = matches[0]
    if emit_logs:
        _safe_log(
            f"ingest | resolved page_id {page_id!r} by loose match "
            f"→ {resolved.relative_to(PAGES_DIR)}"
        )
    return resolved


def _process_tags_in_body(
    body: str,
    existing_tags: list[str],
    parse,
    patch,
    *,
    record_changes: bool = True,
) -> str:
    """For ``create`` bodies: validate, dedupe, record tags from frontmatter.

    Soft-fail throughout — a malformed tag drops itself rather than
    aborting the page. Strict enforcement is wiki_check's job.

    Steps for each tag in the LLM output:
      1. ``validate_tag`` — drop on form-rule failure
      2. ``dedupe_with_existing`` — if cosine similarity to an existing
         same-axis tag is ``>= 0.80``, replace the new tag with the
         existing one (prevents proliferation of near-synonyms)
      3. ``record_new_tag`` — append truly-new tags to the changelog
    """
    from llm_wiki_mcp.tags import (
        dedupe_with_existing,
        record_new_tag,
        validate_tag,
    )

    meta, _ = parse(body)
    tags_raw = meta.get("tags")
    if not isinstance(tags_raw, list) or not tags_raw:
        return body

    existing_set = set(existing_tags)
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw_tag in tags_raw:
        if not isinstance(raw_tag, str):
            continue
        ok, _reason = validate_tag(raw_tag)
        if not ok:
            continue
        # Dedup against the corpus first; the LLM may have invented a
        # synonym for something we already have.
        canonical = dedupe_with_existing(raw_tag, existing_tags, threshold=0.80)
        if canonical in seen:
            continue
        seen.add(canonical)
        cleaned.append(canonical)
        # Audit only tags that survived as truly new (not redirected by
        # dedupe and not present in the corpus).
        if record_changes and canonical == raw_tag and canonical not in existing_set:
            record_new_tag(canonical, reason="ingest auto-gen")

    if cleaned == tags_raw:
        return body
    return patch(body, {"tags": cleaned})


_RECALL_FM_FORBIDDEN = frozenset(",[]:#{}\n\r")


def _safe_recall_field(value: str, *, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    text = "".join(
        " " if ch in _RECALL_FM_FORBIDDEN or ord(ch) < 0x20 else ch for ch in text
    )
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    candidate = text[:limit].rstrip()
    boundary = max(
        candidate.rfind(mark) for mark in ("。", "！", "？", ".", "!", "?", " ")
    )
    if boundary >= max(20, limit // 2):
        return candidate[: boundary + 1].strip()
    return candidate.rstrip("、,:;・-").strip()


def _fallback_recall_metadata(title: str, body: str, page_id: str) -> dict[str, Any]:
    first_line = ""
    for line in body.splitlines():
        stripped = line.strip(" #-\t")
        if stripped:
            first_line = stripped
            break
    summary = _safe_recall_field(first_line or title or page_id, limit=180)
    base = _safe_recall_field(title or page_id, limit=80)
    topic = _safe_recall_field(page_id.replace("-", " "), limit=80)
    questions = [
        f"{base} について何を話した?",
        f"{topic} の続きは?",
        f"{base} の決定事項は?",
    ]
    return {
        "summary": summary,
        "recall_questions": list(dict.fromkeys(q for q in questions if q)),
    }


RECALL_METADATA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "recall_questions"],
    "properties": {
        "summary": {"type": "string", "minLength": 1, "maxLength": 500},
        "recall_questions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": 300},
        },
    },
}


def _generate_recall_metadata(
    title: str,
    body: str,
    page_id: str,
    *,
    transport: ChatTransport | None = None,
) -> dict[str, Any]:
    fallback = _fallback_recall_metadata(title, body, page_id)
    try:
        if not is_available():
            return fallback
        prompt = {
            "task": "Create retrievability metadata for this wiki page.",
            "rules": [
                "summary must be one short line.",
                "recall_questions must be 3-5 questions a user may ask later.",
                "Return JSON only.",
            ],
            "page_id": page_id,
            "title": title,
            "body": body[:2500],
        }
        prompt_text = json.dumps(prompt, ensure_ascii=False)
        config = load_ingest_config()
        metadata_num_predict = min(config.num_predict, 1_024)
        required_num_ctx = required_structured_context_tokens(
            prompt_text,
            RECALL_METADATA_SCHEMA,
            system=None,
            num_predict=metadata_num_predict,
            max_output_chars=1_600,
            max_feedback_chars=1_600,
        )
        selected_num_ctx = _select_ingest_context(
            required_num_ctx,
            num_ctx=config.num_ctx,
            max_num_ctx=config.max_num_ctx,
        )
        live_transport = (
            transport is None
            and _generate_with_progress is _DEFAULT_GENERATE_WITH_PROGRESS
        )
        lease = (
            ollama_runtime.model_resource_lease(exclusive=True)
            if live_transport
            else nullcontext()
        )
        with lease:
            if live_transport:
                selected_num_ctx = _admit_ingest_context(config, selected_num_ctx)
            result = LocalStructuredSession(
                model=config.model,
                transport=transport or _structured_generate_transport(),
                role="ingest_recall_metadata",
                num_ctx=selected_num_ctx,
                num_predict=metadata_num_predict,
                keep_alive=config.keep_alive,
                read_timeout_ms=config.read_timeout_ms,
                max_input_chars=16_000,
                max_output_chars=1_600,
                max_feedback_chars=1_600,
            ).run(prompt_text, RECALL_METADATA_SCHEMA)
        if not result.ok or not isinstance(result.value, dict):
            return fallback
        parsed = result.value
        summary = parsed.get("summary")
        questions = parsed.get("recall_questions")
        if isinstance(summary, str) and isinstance(questions, list):
            cleaned_questions = [
                _safe_recall_field(q, limit=120)
                for q in questions
                if isinstance(q, str) and q.strip()
            ]
            cleaned_questions = list(dict.fromkeys(q for q in cleaned_questions if q))[
                :5
            ]
            cleaned_summary = _safe_recall_field(summary, limit=180)
            if cleaned_summary and cleaned_questions:
                return {
                    "summary": cleaned_summary,
                    "recall_questions": cleaned_questions,
                }
    except Exception:
        pass
    return fallback


def _ensure_recall_metadata_frontmatter(
    text: str,
    page_id: str,
    parse,
    patch,
    *,
    allow_local_model: bool = True,
    force_deterministic_rebuild: bool = False,
) -> str:
    meta, body = parse(text)
    title = meta.get("title", page_id)
    title_text = title if isinstance(title, str) else page_id
    if force_deterministic_rebuild:
        return patch(text, _fallback_recall_metadata(title_text, body, page_id))

    def generate_metadata() -> dict[str, Any]:
        if allow_local_model:
            return _generate_recall_metadata(title_text, body, page_id)
        return _fallback_recall_metadata(title_text, body, page_id)

    updates: dict[str, Any] = {}
    generated: dict[str, Any] | None = None
    if not isinstance(meta.get("summary"), str) or not str(meta.get("summary")).strip():
        generated = generated or generate_metadata()
        updates["summary"] = generated["summary"]
    questions = meta.get("recall_questions")
    if not isinstance(questions, list) or not questions:
        generated = generated or generate_metadata()
        updates["recall_questions"] = generated["recall_questions"]
        updates.setdefault("summary", generated["summary"])
    if not updates:
        return text
    return patch(text, updates)


def _ensure_page_metadata_frontmatter(
    text: str,
    page_id: str,
    parse,
    patch,
    *,
    allow_local_model: bool = True,
    force_deterministic_rebuild: bool = False,
) -> str:
    from llm_wiki_mcp.frontmatter import normalize_nested

    text, _normalization = normalize_nested(text)
    text = _ensure_recall_metadata_frontmatter(
        text,
        page_id,
        parse,
        patch,
        allow_local_model=allow_local_model,
        force_deterministic_rebuild=force_deterministic_rebuild,
    )
    return patch_entities_frontmatter(text)


def _prepare_operations(
    operations: list[dict],
    *,
    read_only: bool = False,
) -> tuple[list[PreparedIngestOperation], dict[str, int]]:
    """Resolve local proposals into exact page preimages and postimages.

    This stage is read-only with respect to Wiki pages.  Ollama triage and
    generation are proposals only; the returned byte-exact plan is what the
    frontier model reviews before :func:`_apply_prepared_operations` may run.

    Fail-closed: any unrecoverable problem raises :class:`IngestApplyError`.
    The caller marks the job FAILED without invoking ``on_complete``.

    Phase 4 propagation: any op that carries a non-empty ``raw_keywords``
    list (from the source raw's frontmatter, riding on metadata since
    Phase 3) gets that list patched onto the page frontmatter inside the
    prepare phase — never inside the write phase, so a partial-write
    rollback restores either the pre-batch text or nothing at all, never
    a half-patched frontmatter.

    Plan-4 tag processing: ``create`` op bodies whose generated frontmatter
    already includes a ``tags:`` list (per ``GENERATE_SYSTEM_PROMPT``) get
    each tag form-validated, dedup'd against the existing corpus's tag
    pool (cosine similarity >= 0.80 → reuse), and audited via
    ``tag-changelog.md``. ``update`` ops never touch ``tags`` because
    ``UPDATE_SYSTEM_PROMPT`` forbids the LLM from emitting frontmatter.
    """
    from llm_wiki_mcp.frontmatter import (
        parse as _frontmatter_parse,
        patch as _frontmatter_patch,
    )

    # Build the universe of valid link targets: every existing page plus every
    # page about to be created in this batch (so siblings can cross-reference).
    # Fail closed — stale or missing index would silently unwrap every link.
    try:
        if read_only:
            # ``IndexStore.refresh`` may persist derived cache files.  A dry
            # run must leave even runtime/index artifacts untouched, so scan
            # and parse the small corpus directly instead.
            from llm_wiki_mcp import wiki as _wiki

            page_paths = list(PAGES_DIR.rglob("*.md"))
            system_paths = list(_wiki.SYSTEM_DIR.rglob("*.md"))
            allowed_ids = {path.stem for path in [*page_paths, *system_paths]}
            reserved_system_ids = {
                _normalize_for_collision(path.stem) for path in system_paths
            }
            tag_values: set[str] = set()
            for path in page_paths:
                meta, _body = _frontmatter_parse(path.read_text(encoding="utf-8"))
                tags = meta.get("tags")
                if isinstance(tags, list):
                    tag_values.update(tag for tag in tags if isinstance(tag, str))
            existing_tags_snapshot = sorted(tag_values)
        else:
            from llm_wiki_mcp.index_store import get_store

            store = get_store()
            store.refresh()
            allowed_ids = store.all_page_ids(include_system=True)
            reserved_system_ids = {
                _normalize_for_collision(page_id)
                for page_id in (
                    store.all_page_ids(include_system=True)
                    - store.all_page_ids(include_system=False)
                )
            }
            # Snapshot the tag pool once for the whole batch so dedupe doesn't
            # re-walk the index on every op. Same-batch siblings can't see
            # each other's newly-coined tags here, but that's fine: dedup is
            # only meaningful against the *committed* corpus, and within-batch
            # divergence will be reconciled the next time wiki_check runs.
            existing_tags_snapshot = store.all_tags(include_system=False)
    except Exception as e:
        raise IngestApplyError(f"index_store unavailable: {e}") from e

    # ---- Prepare phase -----------------------------------------------------
    # Resolve every filename, validate every op, build the final write plan.
    # Nothing here touches disk except for read-only stat/read calls.

    planned: list[PreparedIngestOperation] = []
    seen_norm_ids: set[str] = set()
    seen_paths: set[Path] = set()

    for op in operations:
        op_type = op.get("type")
        if op_type not in ("create", "update"):
            raise IngestApplyError(f"unknown op type: {op_type!r}")

        full_path = _safe_resolve_page_path(op["filename"])
        page_id = full_path.stem

        # Detect intra-batch dups using the same case/Unicode-insensitive key
        # we use against the existing corpus, so two ops whose ids differ
        # only in case or NFC/NFD form are caught before any write.
        norm_key = _normalize_for_collision(page_id)
        if norm_key in reserved_system_ids:
            raise IngestApplyError(
                f"reserved system page_id cannot be mutated by ingest: {page_id!r}"
            )
        if norm_key in seen_norm_ids:
            raise IngestApplyError(
                f"duplicate page_id within batch (case/Unicode-insensitive): "
                f"{page_id!r}"
            )
        if full_path in seen_paths:
            raise IngestApplyError(f"duplicate target path within batch: {full_path}")
        seen_norm_ids.add(norm_key)
        seen_paths.add(full_path)

        allowed_ids.add(page_id)

    totals = {"resolved": 0, "rewritten": 0, "unwrapped": 0}

    for source_operation_index, op in enumerate(operations):
        source_operation_type = op["type"]
        source_filename = op["filename"]
        op_type = source_operation_type
        full_path = _safe_resolve_page_path(op["filename"])
        page_id = full_path.stem

        body, stats = _reconcile_links(op["content"], allowed_ids)
        for k in totals:
            totals[k] += stats[k]

        # Phase 4: lift the raw_keywords side channel off the op. Empty
        # lists are treated as "no propagation" — writing ``raw_keywords:
        # []`` to a page would create a zero-information diff against the
        # existing frontmatter. The propagate flag distinguishes "list[str]
        # with content" from anything else.
        op_raw_keywords = op.get("raw_keywords")
        propagate_raw_keywords = (
            isinstance(op_raw_keywords, list)
            and all(isinstance(v, str) for v in op_raw_keywords)
            and len(op_raw_keywords) > 0
        )

        if op_type == "create":
            existing = _find_page_resilient(page_id, emit_logs=not read_only)
            if existing is not None:
                if not read_only:
                    _safe_log(
                        f"ingest | create op for existing page_id {page_id!r} "
                        f"converted to update (existing: {existing}, target: {full_path})"
                    )
                op_type = "update"
                full_path = existing
                page_id = existing.stem
                body = _strip_all_frontmatter(body).strip()
                if not body:
                    raise IngestApplyError(
                        f"create collision for page_id {page_id!r} produced no update body"
                    )

        if op_type == "create":
            # Tag processing happens BEFORE raw_keywords patch so the
            # final frontmatter goes through one consistent serialization
            # path. Soft-fail: a missing or malformed ``tags`` list just
            # passes the body through unchanged — wiki_check's autonomous
            # lint/repair lane will surface and resolve absent tags.
            body = _process_tags_in_body(
                body,
                existing_tags_snapshot,
                _frontmatter_parse,
                _frontmatter_patch,
                record_changes=False,
            )
            if propagate_raw_keywords:
                # generate output already carries a frontmatter block
                # (enforced by ``_extract_page_body`` for create), so
                # ``patch`` will splice raw_keywords into it without
                # synthesizing a new block.
                body = _frontmatter_patch(body, {"raw_keywords": op_raw_keywords})
            # The model is not a clock. Even when the prompt supplies today's
            # date, enforce it deterministically so a plausible-looking guess
            # can never become page metadata.
            body = _frontmatter_patch(
                body,
                {"updated": date.today().isoformat()},
            )
            created_meta, _created_body = _frontmatter_parse(body)
            created_tags = created_meta.get("tags")
            new_tags = tuple(
                tag
                for tag in (created_tags if isinstance(created_tags, list) else [])
                if isinstance(tag, str) and tag not in set(existing_tags_snapshot)
            )
            planned.append(
                PreparedIngestOperation(
                    op_type="create",
                    path=full_path,
                    page_id=page_id,
                    new_body=body.rstrip() + "\n",
                    previous_text=None,
                    new_tags=new_tags,
                    source_operation_index=source_operation_index,
                    source_operation_type=source_operation_type,
                    source_filename=source_filename,
                )
            )

        else:  # update
            existing_path = (
                full_path
                if full_path.exists()
                else _find_page_resilient(page_id, emit_logs=not read_only)
            )
            if existing_path is None or not existing_path.exists():
                raise IngestApplyError(
                    f"update target not found for page_id {page_id!r}"
                )
            page_id = existing_path.stem
            previous = existing_path.read_text()
            # Preserve the on-disk text for rollback BEFORE we mutate
            # ``previous`` with a frontmatter patch — the rollback path
            # restores the file as it was before this batch ran, not as
            # it was after the patch.
            previous_text_for_rollback = previous

            # raw_keywords union with the existing page's value, preserving
            # insertion order so the diff stays deterministic. If the
            # existing field is missing or malformed (legacy data, manual
            # edit), treat it as empty rather than raising — the apply
            # phase shouldn't reject otherwise-valid updates because of
            # frontmatter rot somewhere upstream.
            if propagate_raw_keywords:
                existing_meta, _existing_body = _frontmatter_parse(previous)
                existing_kw_raw = existing_meta.get("raw_keywords")
                if isinstance(existing_kw_raw, list) and all(
                    isinstance(v, str) for v in existing_kw_raw
                ):
                    existing_kw = existing_kw_raw
                else:
                    existing_kw = []
                union_kw = list(dict.fromkeys(existing_kw + op_raw_keywords))
                previous = _frontmatter_patch(previous, {"raw_keywords": union_kw})

            today = date.today().isoformat()
            stamped = re.sub(
                r"updated:\s*.+",
                f"updated: {today}",
                previous,
                count=1,
            )
            new_body = stamped.rstrip() + "\n\n" + body + "\n"
            planned.append(
                PreparedIngestOperation(
                    op_type="update",
                    path=existing_path,
                    page_id=page_id,
                    new_body=new_body,
                    previous_text=previous_text_for_rollback,
                    source_operation_index=source_operation_index,
                    source_operation_type=source_operation_type,
                    source_filename=source_filename,
                )
            )

    # Apply every currently active correction tombstone to the exact proposal
    # *before* frontier review, including creates under a brand-new slug.
    # The lock-time pass below then acts only as a staleness detector.
    constrained_plans: list[PreparedIngestOperation] = []
    from llm_wiki_mcp.page_mutation import (
        PageMutationError,
        enforce_correction_constraints,
    )

    for entry in planned:
        try:
            constrained_body, enforced = enforce_correction_constraints(
                entry.page_id,
                entry.previous_text or "",
                entry.new_body,
            )
            # Recall metadata is derived only after active correction
            # tombstones have canonicalized the page.  If a stale claim was
            # rewritten, replace summary/questions deterministically from the
            # corrected body so an LLM paraphrase cannot resurrect it.  Dry
            # runs also stay byte-read-only by avoiding model audit artifacts.
            constrained_body = _ensure_page_metadata_frontmatter(
                constrained_body,
                entry.page_id,
                _frontmatter_parse,
                _frontmatter_patch,
                allow_local_model=not read_only and not enforced,
                force_deterministic_rebuild=bool(enforced),
            )
            constrained_body, metadata_enforced = enforce_correction_constraints(
                entry.page_id,
                entry.previous_text or "",
                constrained_body,
            )
        except PageMutationError as exc:
            raise IngestApplyError(
                f"content correction constraint failed for {entry.page_id}: {exc}"
            ) from exc
        all_enforced = [*enforced, *metadata_enforced]
        if all_enforced and not read_only:
            _safe_log(
                f"ingest | enforced {len(all_enforced)} global content correction(s) "
                f"for {entry.page_id}"
            )
        constrained_plans.append(
            PreparedIngestOperation(
                op_type=entry.op_type,
                path=entry.path,
                page_id=entry.page_id,
                new_body=constrained_body,
                previous_text=entry.previous_text,
                new_tags=entry.new_tags,
                source_operation_index=entry.source_operation_index,
                source_operation_type=entry.source_operation_type,
                source_filename=entry.source_filename,
            )
        )

    return constrained_plans, totals


def _apply_prepared_operations(
    planned: list[PreparedIngestOperation],
    *,
    link_totals: dict[str, int] | None = None,
    recovery_only: bool = False,
) -> tuple[list[str], list[str]]:
    """Apply an already frontier-approved exact plan with lock-time CAS.

    Current bytes must be either the reviewed preimage or the reviewed
    postimage.  Accepting the latter makes a durable approved proposal
    recoverable after a power loss between a page replace and job completion.
    Any third state is a race and fails closed for autonomous retry.
    """

    from llm_wiki_mcp.link_fix import atomic_write

    written: list[PreparedIngestOperation] = []
    created: list[str] = []
    updated: list[str] = []
    from llm_wiki_mcp.page_mutation import (
        PageMutationError,
        enforce_correction_constraints,
        wiki_mutation_lock,
    )

    # The same lock is used by the autonomous correction lane. This prevents
    # Stop-hook ingest and correction from both passing their read checks and
    # then replacing the same page with different snapshots.
    with wiki_mutation_lock():
        try:
            for entry in planned:
                # Re-evaluate global correction tombstones while holding the
                # same mutation lock as the correction lane. Preparation may
                # predate a correction on another page, and a stale replay may
                # choose an entirely new slug, so path-local CAS alone is not
                # sufficient here.
                try:
                    constrained_body, enforced = enforce_correction_constraints(
                        entry.page_id,
                        entry.previous_text or "",
                        entry.new_body,
                    )
                except PageMutationError as exc:
                    raise IngestApplyError(
                        f"content correction constraint failed for {entry.page_id}: {exc}"
                    ) from exc
                # The frontier approved ``entry.new_body`` exactly.  A newly
                # activated correction constraint is valid evidence that the
                # proposal became stale, but it cannot silently rewrite the
                # approved postimage.  Retry preparation + review instead.
                if constrained_body != entry.new_body:
                    raise IngestApplyError(
                        f"content correction constraints changed before ingest apply: "
                        f"{entry.page_id}"
                    )
                current = entry.path.read_text() if entry.path.exists() else None
                if current == entry.new_body:
                    # Power-loss recovery: this exact reviewed postimage was
                    # already installed, so finish the batch idempotently.
                    (created if entry.op_type == "create" else updated).append(
                        entry.page_id
                    )
                    continue
                if recovery_only:
                    raise IngestApplyError(
                        "reviewed postimage no longer present during recovery: "
                        f"{entry.page_id}"
                    )
                if entry.op_type == "create":
                    entry.path.parent.mkdir(parents=True, exist_ok=True)
                    if current is not None:
                        raise IngestApplyError(
                            f"page appeared before ingest create: {entry.page_id}"
                        )
                    atomic_write(entry.path, entry.new_body)
                    # Append BEFORE logging so a log failure could never drop
                    # an entry from the rollback set. _safe_log additionally
                    # ensures a logging exception (which atomic_write success
                    # already proves is irrelevant to data) never triggers
                    # rollback of a write that succeeded.
                    written.append(entry)
                    created.append(entry.page_id)
                    _safe_log(f"ingest | created {entry.page_id}")
                else:
                    # The prepare phase captured this exact preimage. Refuse
                    # to overwrite a correction or other cooperating writer
                    # that committed while the model was preparing the batch.
                    if current != (entry.previous_text or ""):
                        raise IngestApplyError(
                            f"page changed before ingest apply: {entry.page_id}"
                        )
                    atomic_write(entry.path, entry.new_body)
                    written.append(entry)
                    updated.append(entry.page_id)
                    _safe_log(f"ingest | updated {entry.page_id}")
        except Exception as write_err:
            # Best-effort rollback. Each revert is gated by a CAS check: only
            # restore if the file still contains exactly what we wrote. If
            # another writer has modified it since, leave their change intact.
            rollback_errors: list[str] = []
            for entry in reversed(written):
                try:
                    if entry.op_type == "create":
                        if (
                            entry.path.exists()
                            and entry.path.read_text() == entry.new_body
                        ):
                            entry.path.unlink()
                        elif entry.path.exists():
                            rollback_errors.append(
                                f"{entry.page_id}: skipped (modified by another writer)"
                            )
                    else:
                        if (
                            entry.path.exists()
                            and entry.path.read_text() == entry.new_body
                        ):
                            atomic_write(entry.path, entry.previous_text or "")
                        elif entry.path.exists():
                            rollback_errors.append(
                                f"{entry.page_id}: skipped (modified by another writer)"
                            )
                except Exception as rb_err:
                    rollback_errors.append(f"{entry.page_id}: {rb_err}")
            if rollback_errors:
                partial_summary = "; ".join(rollback_errors)
                _safe_log(
                    "ingest | rollback partial (other writers or IO failures): "
                    + partial_summary
                )
                raise IngestApplyError(
                    f"apply write failed: {write_err}; partial rollback: "
                    f"{partial_summary}"
                ) from write_err
            _safe_log(
                f"ingest | rolled back {len(written)} writes after error: {write_err}"
            )
            raise IngestApplyError(f"apply write failed: {write_err}") from write_err

    # Tag changelog entries are derived audit data.  They are emitted only
    # after the exact semantic page batch has frontier approval and commits.
    if created:
        from llm_wiki_mcp.tags import record_new_tag

        created_ids = set(created)
        for entry in planned:
            if entry.op_type != "create" or entry.page_id not in created_ids:
                continue
            for tag in entry.new_tags:
                record_new_tag(tag, reason="ingest auto-gen")

    totals = link_totals or {"resolved": 0, "rewritten": 0, "unwrapped": 0}
    if any(totals.values()):
        _safe_log(
            f"ingest | link reconcile: resolved={totals['resolved']} "
            f"rewritten={totals['rewritten']} unwrapped={totals['unwrapped']}"
        )

    return created, updated


def _apply_operations(operations: list[dict]) -> tuple[list[str], list[str]]:
    """Legacy/internal primitive: prepare and apply an already-approved plan.

    Production ingest must use :func:`_review_and_apply_ingest_operations`.
    Keeping this small wrapper preserves focused apply tests without creating a
    second semantic write path in the running ingest pipeline.
    """

    planned, totals = _prepare_operations(operations)
    return _apply_prepared_operations(planned, link_totals=totals)


INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION = INGEST_PROPOSAL_SCHEMA_VERSION
_INGEST_FRONTIER_LEGACY_ARTIFACT_SCHEMA_VERSION = 1
INGEST_FRONTIER_REVIEW_ARTIFACT_SCHEMA_VERSION = 2
INGEST_FRONTIER_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision",
        "summary",
        "failed_operations_disposition",
        "tests_run",
        "risk",
        "notes",
    ],
    "properties": {
        "decision": {
            "type": "string",
            "enum": [
                "apply_available",
                "confirmed_noop",
                "retry",
                "quarantined",
            ],
        },
        "summary": {"type": "string"},
        "failed_operations_disposition": {
            "type": "string",
            "enum": ["none", "confirmed_unnecessary", "retry_required"],
        },
        "tests_run": {"type": "array", "items": {"type": "string"}},
        "risk": {"type": ["string", "null"]},
        "notes": {"type": ["string", "null"]},
        "repair_option_id": {
            "type": "string",
            "pattern": "^rp_[0-9a-f]{32}$",
        },
        "invalid_tags": {
            "type": "array",
            "items": {"type": "string", "pattern": "^[dts]/[a-z0-9][a-z0-9-]*$"},
        },
        "replacement_operations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["filename", "content"],
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        },
    },
}


def _canonical_json_sha256(value: Any) -> str:
    from llm_wiki_mcp.decision_lane_prompts import canonical_json_sha256

    return canonical_json_sha256(value)


def _ingest_source_key(raw_content: str, raw_keywords: list[str] | None) -> str:
    return _canonical_json_sha256(
        {
            "raw_sha256": hashlib.sha256(raw_content.encode("utf-8")).hexdigest(),
            "raw_keywords": list(raw_keywords or []),
        }
    )


def _ingest_artifact_paths(source_key: str) -> tuple[Path, Path]:
    root = PAGES_DIR.parent / "runtime" / "ingest-frontier"
    return (
        root / f"{source_key}.proposal.json",
        root / f"{source_key}.review.json",
    )


def _write_ingest_artifact(path: Path, payload: dict[str, Any]) -> None:
    from llm_wiki_mcp.link_fix import atomic_write

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
    )


def _prepared_from_review_payload(
    rows: object,
    *,
    require_source_provenance: bool = True,
) -> list[PreparedIngestOperation] | None:
    if not isinstance(rows, list):
        return None
    prepared: list[PreparedIngestOperation] = []
    for row in rows:
        if not isinstance(row, dict):
            return None
        op_type = row.get("op_type")
        relative_path = row.get("path")
        page_id = row.get("page_id")
        provenance_fields = (
            "source_operation_index",
            "source_operation_type",
            "source_filename",
        )
        provenance_present = tuple(field in row for field in provenance_fields)
        if require_source_provenance and not all(provenance_present):
            return None
        if any(provenance_present) and not all(provenance_present):
            return None
        source_operation_index = row.get("source_operation_index", -1)
        source_operation_type = row.get("source_operation_type", "")
        source_filename = row.get("source_filename", "")
        previous_text = row.get("previous_text")
        proposed_text = row.get("proposed_text")
        new_tags_raw = row.get("new_tags", [])
        if (
            op_type not in {"create", "update"}
            or not isinstance(relative_path, str)
            or not isinstance(page_id, str)
            or not page_id
            or (
                all(provenance_present)
                and (
                    not isinstance(source_operation_index, int)
                    or isinstance(source_operation_index, bool)
                    or source_operation_index < 0
                    or source_operation_type not in {"create", "update"}
                    or not isinstance(source_filename, str)
                    or not source_filename
                )
            )
            or not isinstance(proposed_text, str)
            or (previous_text is not None and not isinstance(previous_text, str))
            or not isinstance(new_tags_raw, list)
            or not all(isinstance(tag, str) for tag in new_tags_raw)
        ):
            return None
        try:
            path = _safe_resolve_page_path(relative_path)
        except IngestApplyError:
            return None
        if path.stem != page_id:
            return None
        item = PreparedIngestOperation(
            op_type=op_type,
            path=path,
            page_id=page_id,
            new_body=proposed_text,
            previous_text=previous_text,
            new_tags=tuple(new_tags_raw),
            source_operation_index=source_operation_index,
            source_operation_type=source_operation_type,
            source_filename=source_filename,
        )
        if (
            row.get("preimage_exists") is not (previous_text is not None)
            or row.get("previous_sha256") != item.previous_sha256
            or row.get("proposed_sha256") != item.new_sha256
        ):
            return None
        prepared.append(item)
    return prepared


def rollback_ingest_proposal_artifact(
    artifact_path: Path,
    *,
    reason: str,
) -> dict[str, Any]:
    """CAS-rollback exact page postimages recorded by an ingest proposal.

    This is an incident-recovery primitive, not a second normal write path.
    Every target must still contain either the proposal's exact postimage or
    its exact preimage before any byte is changed.  A third state aborts the
    whole rollback, so later edits can never be overwritten.  Successful
    actions are recorded beside the immutable proposal artifact.
    """

    reason_text = reason.strip()
    if not reason_text:
        return {
            "status": "rejected",
            "reason": "rollback reason is required",
            "artifact": str(artifact_path),
        }
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid_artifact",
            "reason": f"{exc.__class__.__name__}: {exc}",
            "artifact": str(artifact_path),
        }
    proposal = artifact.get("proposal") if isinstance(artifact, dict) else None
    if not isinstance(proposal, dict):
        return {
            "status": "invalid_artifact",
            "reason": "proposal payload missing",
            "artifact": str(artifact_path),
        }
    proposal_sha256 = _canonical_json_sha256(proposal)
    artifact_version = artifact.get("schema_version")
    source_key = proposal.get("source_key")
    if (
        not isinstance(artifact_version, int)
        or isinstance(artifact_version, bool)
        or artifact_version
        not in {
            _INGEST_FRONTIER_LEGACY_ARTIFACT_SCHEMA_VERSION,
            INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION,
        }
        or artifact.get("kind") != "ingest_frontier_proposal_artifact"
        or not isinstance(source_key, str)
        or not source_key
        or artifact.get("source_key") != source_key
        or artifact.get("proposal_sha256") != proposal_sha256
        or proposal.get("schema_version") != artifact_version
    ):
        return {
            "status": "invalid_artifact",
            "reason": "artifact identity or digest mismatch",
            "artifact": str(artifact_path),
        }
    planned = _prepared_from_review_payload(
        proposal.get("prepared_operations"),
        require_source_provenance=(
            artifact_version >= INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION
        ),
    )
    if not planned:
        return {
            "status": "invalid_artifact",
            "reason": "no valid prepared operations",
            "artifact": str(artifact_path),
        }
    identities = [str(item.path.resolve(strict=False)) for item in planned]
    if len(identities) != len(set(identities)):
        return {
            "status": "invalid_artifact",
            "reason": "duplicate target path",
            "artifact": str(artifact_path),
        }
    page_ids = [item.page_id for item in planned]
    if len(page_ids) != len(set(page_ids)):
        return {
            "status": "invalid_artifact",
            "reason": "duplicate page_id",
            "artifact": str(artifact_path),
        }
    if any(
        (item.op_type == "create") is not (item.previous_text is None)
        for item in planned
    ):
        return {
            "status": "invalid_artifact",
            "reason": "operation type does not match recorded preimage",
            "artifact": str(artifact_path),
        }

    from llm_wiki_mcp.link_fix import atomic_write
    from llm_wiki_mcp.page_mutation import wiki_mutation_lock

    def read_optional(item: PreparedIngestOperation) -> str | None:
        return item.path.read_text(encoding="utf-8") if item.path.exists() else None

    rolled_back: list[PreparedIngestOperation] = []
    already_rolled_back: list[str] = []
    try:
        with wiki_mutation_lock():
            states: dict[str, str] = {}
            for item in planned:
                try:
                    current = read_optional(item)
                except (OSError, UnicodeDecodeError) as exc:
                    return {
                        "status": "conflict",
                        "reason": f"cannot read {item.page_id}: {exc}",
                        "artifact": str(artifact_path),
                        "pages": [],
                    }
                if current == item.new_body:
                    states[item.page_id] = "postimage"
                elif current == item.previous_text:
                    states[item.page_id] = "preimage"
                    already_rolled_back.append(item.page_id)
                else:
                    digest = (
                        hashlib.sha256(current.encode("utf-8")).hexdigest()
                        if current is not None
                        else None
                    )
                    return {
                        "status": "conflict",
                        "reason": "target no longer matches proposal preimage or postimage",
                        "artifact": str(artifact_path),
                        "page_id": item.page_id,
                        "current_sha256": digest,
                        "pages": [],
                    }
            try:
                for item in reversed(planned):
                    if states[item.page_id] == "preimage":
                        continue
                    if item.previous_text is None:
                        item.path.unlink()
                    else:
                        atomic_write(item.path, item.previous_text)
                    if read_optional(item) != item.previous_text:
                        raise IngestApplyError(
                            f"rollback verification failed: {item.page_id}"
                        )
                    rolled_back.append(item)
            except Exception as exc:
                restored: dict[str, bool] = {}
                for item in reversed(rolled_back):
                    try:
                        if read_optional(item) != item.previous_text:
                            restored[item.page_id] = False
                            continue
                        item.path.parent.mkdir(parents=True, exist_ok=True)
                        atomic_write(item.path, item.new_body)
                        restored[item.page_id] = read_optional(item) == item.new_body
                    except Exception:
                        restored[item.page_id] = False
                return {
                    "status": "rollback_failed",
                    "reason": f"{exc.__class__.__name__}: {exc}",
                    "artifact": str(artifact_path),
                    "restored_postimages": restored,
                    "pages": [],
                }
    except OSError as exc:
        return {
            "status": "rollback_failed",
            "reason": f"wiki mutation lock failed: {exc}",
            "artifact": str(artifact_path),
            "pages": [],
        }

    rolled_back_page_ids = [item.page_id for item in reversed(rolled_back)]
    status = "rolled_back" if rolled_back_page_ids else "already_rolled_back"
    audit_path = artifact_path.with_name(f"{source_key}.rollback.json")
    audit = {
        "schema_version": INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION,
        "kind": "ingest_frontier_rollback_artifact",
        "source_key": source_key,
        "proposal_sha256": proposal_sha256,
        "rolled_back_at": _now(),
        "reason": reason_text,
        "status": status,
        "pages": rolled_back_page_ids,
        "already_rolled_back": already_rolled_back,
    }
    audit_written = True
    audit_error: str | None = None
    if audit_path.exists():
        audit_written = False
        audit_error = "rollback audit already exists"
    else:
        try:
            _write_ingest_artifact(audit_path, audit)
        except OSError as exc:
            audit_written = False
            audit_error = f"{exc.__class__.__name__}: {exc}"
    return {
        "status": status,
        "artifact": str(artifact_path),
        "proposal_sha256": proposal_sha256,
        "pages": rolled_back_page_ids,
        "already_rolled_back": already_rolled_back,
        "audit_path": str(audit_path),
        "audit_written": audit_written,
        "audit_error": audit_error,
    }


def _prepared_plan_is_recoverable(planned: list[PreparedIngestOperation]) -> bool:
    """True when every page is still at the reviewed pre- or postimage."""

    for item in planned:
        try:
            current = (
                item.path.read_text(encoding="utf-8") if item.path.exists() else None
            )
        except (OSError, UnicodeDecodeError):
            return False
        if current not in {item.previous_text, item.new_body}:
            return False
    return True


def _build_ingest_frontier_proposal(
    *,
    raw_content: str,
    raw_keywords: list[str] | None,
    source_raw: str | None,
    operations: list[dict],
    planned: list[PreparedIngestOperation],
    link_totals: dict[str, int],
    triage_plan: list[dict] | None = None,
    failed_operation_specs: list[dict] | None = None,
    local_disposition: str = "operations_available",
) -> dict[str, Any]:
    raw_sha256 = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
    source_key = _ingest_source_key(raw_content, raw_keywords)
    return {
        "schema_version": INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION,
        "kind": "ingest_semantic_mutation_proposal",
        "source_key": source_key,
        "source_raw": source_raw,
        "raw_content": raw_content,
        "raw_sha256": raw_sha256,
        "raw_keywords": list(raw_keywords or []),
        "local_disposition": local_disposition,
        "triage_plan": list(triage_plan or []),
        "failed_operation_specs": list(failed_operation_specs or []),
        "local_generated_operations": operations,
        "prepared_operations": [item.review_payload() for item in planned],
        "link_reconciliation": dict(link_totals),
    }


def _load_ingest_proposal(
    path: Path,
    *,
    source_key: str,
    raw_content: str,
) -> tuple[dict[str, Any], list[PreparedIngestOperation]] | None:
    from llm_wiki_mcp.decision_lane_prompts import (
        validate_ingest_proposal_envelope,
    )

    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(artifact, dict):
        return None
    proposal = artifact.get("proposal")
    if not isinstance(proposal, dict):
        return None
    proposal_sha256 = _canonical_json_sha256(proposal)
    if (
        artifact.get("schema_version") != INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION
        or artifact.get("kind") != "ingest_frontier_proposal_artifact"
        or artifact.get("source_key") != source_key
        or artifact.get("proposal_sha256") != proposal_sha256
        or not validate_ingest_proposal_envelope(proposal)
        or proposal.get("source_key") != source_key
        or proposal.get("raw_content") != raw_content
        or proposal.get("raw_sha256")
        != hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
    ):
        return None
    prepared = _prepared_from_review_payload(proposal.get("prepared_operations"))
    if prepared is None or not _prepared_plan_is_recoverable(prepared):
        return None
    return proposal, prepared


def _load_ingest_review_artifact(
    path: Path,
    *,
    source_key: str,
    proposal_sha256: str,
) -> dict[str, Any] | None:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(artifact, dict):
        return None
    review = artifact.get("review")
    authority = artifact.get("authority")
    if (
        artifact.get("schema_version") != INGEST_FRONTIER_REVIEW_ARTIFACT_SCHEMA_VERSION
        or artifact.get("kind") != "ingest_frontier_review_artifact"
        or artifact.get("source_key") != source_key
        or artifact.get("proposal_sha256") != proposal_sha256
        or not isinstance(authority, dict)
        or _ingest_review_authority_shape_error(authority) is not None
        or not isinstance(review, dict)
        or _ingest_review_authority_error(review, authority) is not None
        or review.get("decision")
        not in {"apply_available", "confirmed_noop", "approved", "rejected"}
    ):
        return None
    return artifact


def _load_ingest_review(
    path: Path,
    *,
    source_key: str,
    proposal_sha256: str,
) -> dict[str, Any] | None:
    """Load a structurally valid durable verdict.

    Authority freshness is intentionally checked by the caller immediately
    before a verdict is reused.  Keeping structural and live-policy checks
    separate also permits an exact-postimage recovery path that never performs
    a new semantic mutation.
    """

    artifact = _load_ingest_review_artifact(
        path,
        source_key=source_key,
        proposal_sha256=proposal_sha256,
    )
    if artifact is None:
        return None
    review = artifact.get("review")
    return review if isinstance(review, dict) else None


def _sealed_ingest_review_artifact(
    *,
    source_key: str,
    proposal_sha256: str,
    review: dict[str, Any],
    authority: dict[str, Any],
) -> dict[str, Any]:
    """Build the common authority-sealed terminal ingest artifact."""

    return decision_authority.seal_semantic_artifact(
        {
            "schema_version": INGEST_FRONTIER_REVIEW_ARTIFACT_SCHEMA_VERSION,
            "kind": "ingest_frontier_review_artifact",
            "source_key": source_key,
            "proposal_sha256": proposal_sha256,
            "review": review,
        },
        authority=authority,
        lane="ingest_reconciliation",
    )


def _write_and_readback_ingest_review_artifact(
    path: Path,
    *,
    source_key: str,
    proposal_sha256: str,
    review: dict[str, Any],
    authority: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Atomically persist and verify a terminal verdict before any effect.

    The caller holds ``decision_authority_lock`` so adoption cannot change
    between the final proof, durable seal, verified readback, and page/no-op
    effect.
    """

    try:
        sealed = _sealed_ingest_review_artifact(
            source_key=source_key,
            proposal_sha256=proposal_sha256,
            review=review,
            authority=authority,
        )
        _write_ingest_artifact(path, sealed)
    except (OSError, ValueError) as exc:
        return None, f"frontier review artifact write failed: {exc}"
    readback = _load_ingest_review_artifact(
        path,
        source_key=source_key,
        proposal_sha256=proposal_sha256,
    )
    if readback != sealed:
        return None, "frontier review artifact readback verification failed"
    return readback, None


def _current_ingest_review_authority(
    *, reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve the exact enabled local authority allowed to affect ingest."""

    reviewer_module = getattr(_run_ingest_frontier_review, "__module__", None)
    return decision_authority.current_semantic_authority(
        "ingest_reconciliation",
        injected_reviewer=(reviewer is not None or reviewer_module != __name__),
    )


def _ingest_review_authority_error(
    review: dict[str, Any],
    authority: dict[str, Any],
) -> str | None:
    """Cross-check the verdict and trusted local quorum with its authority."""

    # Explicit dependency injection is the only boundary that is allowed to
    # bypass the production router proof.  Production ingest must use the same
    # central verifier as every other durable semantic lane so a copied policy
    # audit without its exact two-model quorum can never authorize a write.
    if authority.get("source") == "injected_reviewer_boundary":
        return None
    return decision_authority.semantic_verdict_authority_error(
        review,
        authority,
        lane="ingest_reconciliation",
    )


def _ingest_review_authority_shape_error(authority: dict[str, Any]) -> str | None:
    """Reject authority envelopes that cannot identify a review epoch."""

    return decision_authority.semantic_authority_shape_error(
        authority,
        lane="ingest_reconciliation",
    )


def _prepared_plan_is_fully_applied(
    planned: list[PreparedIngestOperation],
) -> bool:
    """Prove that recovery can finish without installing any semantic bytes."""

    if not planned:
        return False
    for item in planned:
        try:
            current = (
                item.path.read_text(encoding="utf-8") if item.path.exists() else None
            )
        except (OSError, UnicodeDecodeError):
            return False
        if current != item.new_body:
            return False
    return True


def _load_strict_ingest_proposal_for_recovery(
    path: Path,
    *,
    source_key: str,
    raw_content: str,
    raw_keywords: list[str] | None,
) -> tuple[dict[str, Any], list[PreparedIngestOperation]]:
    """Load one versioned terminal proposal without consulting model output.

    The ordinary retry path may replace an incomplete proposal after another
    bounded local attempt.  Pre-triage completion recovery is more privileged:
    it can retire a raw without asking a model again, so every artifact field
    that binds the raw and exact page postimages must already be intact.
    """

    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IngestApplyError(
            "pre-triage terminal proposal artifact is unreadable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    expected_top_level = {
        "schema_version",
        "kind",
        "source_key",
        "proposal_sha256",
        "proposal",
    }
    if not isinstance(artifact, dict) or set(artifact) != expected_top_level:
        raise IngestApplyError(
            "pre-triage terminal proposal artifact schema is invalid"
        )
    proposal = artifact.get("proposal")
    proposal_sha256 = (
        _canonical_json_sha256(proposal) if isinstance(proposal, dict) else None
    )
    artifact_version = artifact.get("schema_version")
    expected_raw_sha256 = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
    if (
        not isinstance(artifact_version, int)
        or isinstance(artifact_version, bool)
        or artifact_version
        not in {
            _INGEST_FRONTIER_LEGACY_ARTIFACT_SCHEMA_VERSION,
            INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION,
        }
        or artifact.get("kind") != "ingest_frontier_proposal_artifact"
        or artifact.get("source_key") != source_key
        or artifact.get("proposal_sha256") != proposal_sha256
        or not isinstance(proposal, dict)
        or proposal.get("schema_version") != artifact_version
        or proposal.get("kind") != "ingest_semantic_mutation_proposal"
        or proposal.get("source_key") != source_key
        or proposal.get("raw_content") != raw_content
        or proposal.get("raw_sha256") != expected_raw_sha256
        or proposal.get("raw_keywords") != list(raw_keywords or [])
    ):
        raise IngestApplyError(
            "pre-triage terminal proposal artifact binding is invalid"
        )
    required_proposal_fields = {
        "schema_version",
        "kind",
        "source_key",
        "source_raw",
        "raw_content",
        "raw_sha256",
        "raw_keywords",
        "local_disposition",
        "triage_plan",
        "failed_operation_specs",
        "local_generated_operations",
        "prepared_operations",
        "link_reconciliation",
    }
    if not required_proposal_fields <= set(proposal) or set(proposal) - (
        required_proposal_fields | {"audit_decision"}
    ):
        raise IngestApplyError("pre-triage terminal proposal payload schema is invalid")
    if (
        proposal.get("source_raw") is not None
        and not isinstance(proposal.get("source_raw"), str)
    ) or not isinstance(proposal.get("local_disposition"), str):
        raise IngestApplyError("pre-triage terminal proposal metadata is invalid")
    for field in (
        "triage_plan",
        "failed_operation_specs",
        "local_generated_operations",
        "prepared_operations",
    ):
        if not isinstance(proposal.get(field), list):
            raise IngestApplyError(f"pre-triage terminal proposal {field} is invalid")
    if not isinstance(proposal.get("link_reconciliation"), dict) or (
        "audit_decision" in proposal
        and not isinstance(proposal.get("audit_decision"), dict)
    ):
        raise IngestApplyError("pre-triage terminal proposal audit metadata is invalid")
    planned = _prepared_from_review_payload(
        proposal.get("prepared_operations"),
        require_source_provenance=(
            artifact_version >= INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION
        ),
    )
    if planned is None:
        raise IngestApplyError(
            "pre-triage terminal proposal page postimages are invalid"
        )
    target_paths = [str(item.path.resolve(strict=False)) for item in planned]
    page_ids = [item.page_id for item in planned]
    if len(target_paths) != len(set(target_paths)) or len(page_ids) != len(
        set(page_ids)
    ):
        raise IngestApplyError(
            "pre-triage terminal proposal has duplicate page targets"
        )
    return proposal, planned


def _load_pretriage_terminal_recovery(
    raw_content: str,
    raw_keywords: list[str] | None,
    *,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Return a model-free terminal recovery only when its proof is complete.

    A proposal without a current terminal review remains ordinary retry work.
    Legacy review artifacts are likewise left to the normal local-consensus
    path.  A malformed *current* artifact fails closed instead of being
    overwritten and silently treated as new work.
    """

    source_key = _ingest_source_key(raw_content, raw_keywords)
    proposal_path, review_path = _ingest_artifact_paths(source_key)
    if not proposal_path.exists():
        if review_path.exists():
            raise IngestApplyError(
                "pre-triage terminal review exists without its proposal artifact"
            )
        return None

    try:
        proposal_candidate = json.loads(proposal_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IngestApplyError(
            "pre-triage terminal proposal artifact is unreadable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(proposal_candidate, dict):
        raise IngestApplyError(
            "pre-triage terminal proposal artifact schema is invalid"
        )
    proposal_version = proposal_candidate.get("schema_version")
    if not review_path.exists():
        if proposal_version == _INGEST_FRONTIER_LEGACY_ARTIFACT_SCHEMA_VERSION:
            # A v1 proposal without a durable authority seal never authorized
            # an effect.  Its rows predate source-operation provenance, so the
            # ordinary path may safely replace it with a complete v2 proposal.
            return None
        _load_strict_ingest_proposal_for_recovery(
            proposal_path,
            source_key=source_key,
            raw_content=raw_content,
            raw_keywords=raw_keywords,
        )
        return None
    try:
        review_candidate = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IngestApplyError(
            "pre-triage terminal review artifact is unreadable: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(review_candidate, dict):
        raise IngestApplyError("pre-triage terminal review artifact is not an object")
    review_version = review_candidate.get("schema_version")
    if (
        isinstance(review_version, int)
        and not isinstance(review_version, bool)
        and review_version < INGEST_FRONTIER_REVIEW_ARTIFACT_SCHEMA_VERSION
    ):
        # Historical frontier-shaped verdicts have no local authority seal.
        # They are neither trusted nor treated as corruption; the normal local
        # path will replace them after a fresh adopted-consensus decision.
        if proposal_version == _INGEST_FRONTIER_LEGACY_ARTIFACT_SCHEMA_VERSION:
            return None
        _load_strict_ingest_proposal_for_recovery(
            proposal_path,
            source_key=source_key,
            raw_content=raw_content,
            raw_keywords=raw_keywords,
        )
        return None
    proposal, planned = _load_strict_ingest_proposal_for_recovery(
        proposal_path,
        source_key=source_key,
        raw_content=raw_content,
        raw_keywords=raw_keywords,
    )
    expected_review_fields = {
        "schema_version",
        "kind",
        "source_key",
        "proposal_sha256",
        "review",
        "authority",
    }
    if set(review_candidate) != expected_review_fields:
        raise IngestApplyError("pre-triage terminal review artifact schema is invalid")
    proposal_sha256 = _canonical_json_sha256(proposal)
    review_artifact = _load_ingest_review_artifact(
        review_path,
        source_key=source_key,
        proposal_sha256=proposal_sha256,
    )
    if review_artifact is None or review_artifact != review_candidate:
        raise IngestApplyError("pre-triage terminal review artifact binding is invalid")
    review = review_artifact.get("review")
    authority = review_artifact.get("authority")
    if not isinstance(review, dict) or not isinstance(authority, dict):
        raise IngestApplyError("pre-triage terminal review proof is missing")
    normalized_review = _normalize_ingest_frontier_review(review, proposal=proposal)
    if normalized_review != review:
        raise IngestApplyError(
            "pre-triage terminal review is not canonical for its proposal"
        )
    decision = review.get("decision")
    if decision not in {"apply_available", "confirmed_noop"}:
        return None
    if decision == "apply_available":
        if not planned or not _prepared_plan_is_fully_applied(planned):
            return None
    else:
        current_authority, current_authority_error = _current_ingest_review_authority(
            reviewer=reviewer
        )
        if current_authority_error is not None or (
            decision_authority.compare_semantic_authority(
                authority,
                current_authority,
                lane="ingest_reconciliation",
            )
            is not None
        ):
            return None

    created = [item.page_id for item in planned if item.op_type == "create"]
    updated = [item.page_id for item in planned if item.op_type == "update"]
    audit = proposal.get("audit_decision")
    failed_specs = proposal.get("failed_operation_specs")
    return {
        "status": decision,
        "source_key": source_key,
        "proposal_sha256": proposal_sha256,
        "review": review,
        "authority": authority,
        "created": created,
        "updated": updated,
        "audit": dict(audit) if isinstance(audit, dict) else {},
        "failed_operation_specs": (
            list(failed_specs) if isinstance(failed_specs, list) else []
        ),
        "recovered_artifact": True,
        "reused_review": True,
        "recovery_basis": (
            "exact_postimages_already_applied"
            if decision == "apply_available"
            else "durable_confirmed_noop"
        ),
    }


def _normalize_ingest_frontier_review(
    value: object,
    *,
    proposal: dict[str, Any],
) -> dict[str, Any]:
    """Normalize the final disposition and fail closed on silent data loss."""

    if not isinstance(value, dict):
        return {
            "decision": "retry",
            "summary": "local consensus reviewer returned a non-object payload",
            "failed_operations_disposition": "retry_required",
        }
    from llm_wiki_mcp.decision_schema_manifest import (
        canonical_ingest_repair_arrays,
    )

    value = canonical_ingest_repair_arrays(value)
    raw_decision = value.get("decision")
    decision = {
        # Compatibility with the older generic frontier schema.  Legacy
        # approval is safe only for a complete proposal; partial generation
        # requires the new explicit failed-operation disposition below.
        "approved": "apply_available",
        "rejected": "retry",
        "needs_retry": "retry",
    }.get(str(raw_decision), raw_decision)
    summary = value.get("summary")
    if decision not in {
        "apply_available",
        "confirmed_noop",
        "retry",
        "quarantined",
    }:
        return {
            **value,
            "decision": "retry",
            "summary": "local consensus reviewer returned an invalid decision",
            "failed_operations_disposition": "retry_required",
        }
    if not isinstance(summary, str) or not summary.strip():
        return {
            **value,
            "decision": "retry",
            "summary": "local consensus reviewer omitted its decision summary",
            "failed_operations_disposition": "retry_required",
        }
    repair_requested = any(
        isinstance(value.get(field), list) and bool(value.get(field))
        for field in ("invalid_tags", "replacement_operations")
    )
    if decision in {"apply_available", "confirmed_noop"} and repair_requested:
        # Repair arrays describe a new postimage which has not yet passed the
        # semantic gate.  They are therefore non-terminal instructions even if
        # a model accidentally combines them with an approval/no-op verdict.
        decision = "retry"
        summary = (
            "repair instructions require a fresh review before terminal "
            f"disposition: {summary.strip()}"
        )
    if decision == "apply_available" and value.get("frontier_failure"):
        return {
            **value,
            "decision": "retry",
            "summary": "local consensus verdict carried a failure payload",
            "failed_operations_disposition": "retry_required",
        }

    prepared = proposal.get("prepared_operations")
    has_available_operations = isinstance(prepared, list) and bool(prepared)
    failed_specs = proposal.get("failed_operation_specs")
    has_failed_operations = isinstance(failed_specs, list) and bool(failed_specs)
    disposition = value.get("failed_operations_disposition")
    if repair_requested:
        disposition = "retry_required"
    elif not has_failed_operations and disposition is None:
        disposition = "none"

    if disposition not in {"none", "confirmed_unnecessary", "retry_required"}:
        return {
            **value,
            "decision": "retry",
            "summary": (
                "local consensus must explicitly disposition locally failed operations"
                if has_failed_operations
                else "local consensus returned an invalid failed-operation disposition"
            ),
            "failed_operations_disposition": "retry_required",
        }
    if has_failed_operations and decision in {"apply_available", "confirmed_noop"}:
        if disposition != "confirmed_unnecessary":
            return {
                **value,
                "decision": "retry",
                "summary": (
                    "partial local generation remains replayable until local consensus "
                    "explicitly confirms failed operations are unnecessary"
                ),
                "failed_operations_disposition": "retry_required",
            }
    if not has_failed_operations and not repair_requested:
        # The disposition field only matters when local generation left
        # replayable failed ops behind. Frontier models may still emit a
        # non-`none` enum because the schema requires the field; treat that
        # as redundant noise instead of bouncing an otherwise-complete plan.
        disposition = "none"
    if decision == "apply_available" and not has_available_operations:
        return {
            **value,
            "decision": "retry",
            "summary": (
                "local consensus requested apply_available with no prepared operation"
            ),
            "failed_operations_disposition": (
                "retry_required" if has_failed_operations else "none"
            ),
        }
    return {
        **value,
        "decision": decision,
        "summary": summary.strip(),
        "failed_operations_disposition": disposition,
    }


def _run_ingest_frontier_review(
    proposal: dict[str, Any],
    *,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if reviewer is not None:
        return _normalize_ingest_frontier_review(
            reviewer(proposal),
            proposal=proposal,
        )

    from llm_wiki_mcp.decision_lane_prompts import (
        build_ingest_reconciliation_prompt,
    )
    from llm_wiki_mcp.frontier_review import run_structured_review

    prompt = build_ingest_reconciliation_prompt(proposal)
    result = run_structured_review(
        prompt,
        INGEST_FRONTIER_DECISION_SCHEMA,
        repo_root=Path(__file__).resolve().parents[2],
        execute_patch=False,
        decision_lane="ingest_reconciliation",
    )
    return _normalize_ingest_frontier_review(result, proposal=proposal)


def _review_and_apply_ingest_operations(
    operations: list[dict],
    *,
    raw_content: str,
    raw_keywords: list[str] | None = None,
    source_raw: str | None = None,
    triage_plan: list[dict] | None = None,
    failed_operation_specs: list[dict] | None = None,
    local_disposition: str = "operations_available",
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    force_frontier_review: bool = False,
    frontier_budget: "_FrontierCallBudget | None" = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Authorize by risk policy, durably bind the verdict, and CAS apply."""

    source_key = _ingest_source_key(raw_content, raw_keywords)
    proposal_path, review_path = _ingest_artifact_paths(source_key)
    audit_state_path = proposal_path.parent / "audit-state.json"
    recovered = _load_ingest_proposal(
        proposal_path,
        source_key=source_key,
        raw_content=raw_content,
    )
    # Only a terminal verdict can pin a previous local proposal.  A durable
    # proposal without such a verdict represents retryable local/frontier
    # work; rebuild it from this attempt so a transient generation failure
    # cannot suppress a later complete plan forever.
    if recovered is not None:
        recovered_proposal, _recovered_planned = recovered
        recovered_sha256 = _canonical_json_sha256(recovered_proposal)
        if (
            _load_ingest_review(
                review_path,
                source_key=source_key,
                proposal_sha256=recovered_sha256,
            )
            is None
        ):
            recovered = None
    if recovered is None:
        planned, totals = _prepare_operations(operations, read_only=dry_run)
        proposal = _build_ingest_frontier_proposal(
            raw_content=raw_content,
            raw_keywords=raw_keywords,
            source_raw=source_raw,
            operations=operations,
            planned=planned,
            link_totals=totals,
            triage_plan=triage_plan,
            failed_operation_specs=failed_operation_specs,
            local_disposition=local_disposition,
        )
        from llm_wiki_mcp.ingest_audit import decide_ingest_audit

        audit_decision = decide_ingest_audit(
            source_key=source_key,
            raw_content=raw_content,
            operations=operations,
            failed_operation_specs=list(failed_operation_specs or []),
            local_disposition=local_disposition,
            state_path=audit_state_path,
            force=force_frontier_review,
            explicit_reviewer=reviewer is not None,
        ).to_dict()
        proposal["audit_decision"] = audit_decision
        recovered_artifact = False
    else:
        proposal, planned = recovered
        audit_raw = proposal.get("audit_decision")
        audit_decision = (
            dict(audit_raw)
            if isinstance(audit_raw, dict)
            else {
                "required": True,
                "mode": "legacy-frontier",
                "reasons": ["legacy reviewed artifact"],
            }
        )
        totals_raw = proposal.get("link_reconciliation")
        totals = (
            {
                key: int(totals_raw.get(key, 0))
                for key in ("resolved", "rewritten", "unwrapped")
            }
            if isinstance(totals_raw, dict)
            else {"resolved": 0, "rewritten": 0, "unwrapped": 0}
        )
        recovered_artifact = True

    proposal_sha256 = _canonical_json_sha256(proposal)
    if dry_run:
        return {
            "status": "dry_run",
            "source_key": source_key,
            "proposal_sha256": proposal_sha256,
            "proposal": proposal,
            "audit": audit_decision,
            "created": [],
            "updated": [],
            "artifact_written": False,
        }

    if not recovered_artifact:
        try:
            _write_ingest_artifact(
                proposal_path,
                {
                    "schema_version": INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION,
                    "kind": "ingest_frontier_proposal_artifact",
                    "source_key": source_key,
                    "proposal_sha256": proposal_sha256,
                    "proposal": proposal,
                },
            )
        except OSError as exc:
            return {
                "status": "needs_retry",
                "source_key": source_key,
                "proposal_sha256": proposal_sha256,
                "summary": f"review proposal artifact write failed: {exc}",
                "created": [],
                "updated": [],
                "audit": audit_decision,
            }

    review_artifact = _load_ingest_review_artifact(
        review_path,
        source_key=source_key,
        proposal_sha256=proposal_sha256,
    )
    artifact_review = (
        review_artifact.get("review") if isinstance(review_artifact, dict) else None
    )
    artifact_review = artifact_review if isinstance(artifact_review, dict) else None
    artifact_authority = (
        review_artifact.get("authority") if isinstance(review_artifact, dict) else None
    )
    artifact_authority = (
        artifact_authority if isinstance(artifact_authority, dict) else None
    )

    # A process may die after the reviewed postimages are durable but before
    # the surrounding raw/job state is committed.  Exact postimage equality
    # proves this recovery performs no new semantic page write, so it remains
    # safe even if the adopted authority has since changed.  Partial batches,
    # confirmed-noop decisions, and legacy/malformed review artifacts never
    # receive this exception.
    if (
        artifact_review is not None
        and artifact_review.get("decision") == "apply_available"
        and artifact_authority is not None
        and _ingest_review_authority_shape_error(artifact_authority) is None
        and _ingest_review_authority_error(artifact_review, artifact_authority) is None
        and _prepared_plan_is_fully_applied(planned)
    ):
        created, updated = _apply_prepared_operations(
            planned,
            link_totals=totals,
            recovery_only=True,
        )
        return {
            "status": "apply_available",
            "source_key": source_key,
            "proposal_sha256": proposal_sha256,
            "review": artifact_review,
            "recovered_artifact": recovered_artifact,
            "reused_review": True,
            "recovery_basis": "exact_postimages_already_applied",
            "created": created,
            "updated": updated,
            "audit": audit_decision,
        }

    review_authority, authority_error = _current_ingest_review_authority(
        reviewer=reviewer
    )
    authority_shape_error = (
        _ingest_review_authority_shape_error(review_authority)
        if review_authority is not None
        else None
    )
    if (
        authority_error is not None
        or review_authority is None
        or authority_shape_error is not None
    ):
        return {
            "status": "needs_retry",
            "source_key": source_key,
            "proposal_sha256": proposal_sha256,
            "summary": authority_error
            or authority_shape_error
            or "ingest review authority is missing",
            "recovered_artifact": recovered_artifact,
            "reused_review": False,
            "created": [],
            "updated": [],
            "audit": audit_decision,
        }

    stale_review_reason: str | None = None
    if artifact_review is not None:
        if artifact_authority != review_authority:
            stale_review_reason = "ingest review authority changed before effect"
        else:
            stale_review_reason = _ingest_review_authority_error(
                artifact_review, review_authority
            )
    review = artifact_review if stale_review_reason is None else None
    reused_review = review is not None
    frontier_used = False
    if review is None:
        # Triage and generation are semantic model output.  Deterministic
        # schema/path/link validation proves that a proposal is well-formed;
        # it cannot prove that its claims are grounded in the raw.  Therefore
        # even an audit sampler's "low-risk" result must never authorize a
        # write or discard by itself.  Every semantic ingest disposition goes
        # through the lane-scoped local consensus gate, which fails closed
        # while shadowed or before an adoption artifact exists.
        if frontier_budget is not None and not frontier_budget.consume():
            runtime_status.safe_append_metric(
                "ingest_authorization",
                source_key=source_key,
                mode=str(audit_decision.get("mode") or "unknown"),
                frontier_used=False,
                required=True,
                sample_rate=audit_decision.get("sample_rate"),
                caught_issue_rate=audit_decision.get("caught_issue_rate"),
                decision="frontier_budget_exhausted",
            )
            return {
                "status": "frontier_budget_exhausted",
                "source_key": source_key,
                "proposal_sha256": proposal_sha256,
                "summary": (
                    "structured review budget exhausted "
                    f"({frontier_budget.used}/{frontier_budget.limit})"
                ),
                "review": {
                    "decision": "retry",
                    "summary": (
                        "structured review budget exhausted; keep the raw "
                        "pending for local consensus"
                    ),
                },
                "recovered_artifact": recovered_artifact,
                "reused_review": False,
                "created": [],
                "updated": [],
                "audit": audit_decision,
            }
        frontier_used = True
        runtime_status.safe_write_status(stage="local-consensus-review")
        try:
            review = _run_ingest_frontier_review(proposal, reviewer=reviewer)
        except Exception as exc:
            review = {
                "decision": "needs_retry",
                "summary": f"local consensus reviewer failed: {exc.__class__.__name__}: {exc}",
            }

    review = _normalize_ingest_frontier_review(review, proposal=proposal)
    decision = str(review.get("decision") or "retry")
    if decision in {"apply_available", "confirmed_noop"} and not reused_review:
        # A verdict is not durable until its own embedded audit and the live
        # adoption/policy epoch agree.  Re-resolving after the model call
        # catches authority replacement while consensus was in flight.
        current_authority, current_authority_error = _current_ingest_review_authority(
            reviewer=reviewer
        )
        if current_authority_error is not None or current_authority != review_authority:
            return {
                "status": "needs_retry",
                "source_key": source_key,
                "proposal_sha256": proposal_sha256,
                "review": review,
                "summary": current_authority_error
                or "ingest review authority changed during review",
                "recovered_artifact": recovered_artifact,
                "reused_review": False,
                "created": [],
                "updated": [],
                "audit": audit_decision,
            }
        if policy_error := _ingest_review_authority_error(review, review_authority):
            return {
                "status": "needs_retry",
                "source_key": source_key,
                "proposal_sha256": proposal_sha256,
                "review": review,
                "summary": policy_error,
                "recovered_artifact": recovered_artifact,
                "reused_review": False,
                "created": [],
                "updated": [],
                "audit": audit_decision,
            }
    runtime_status.safe_append_metric(
        "ingest_authorization",
        source_key=source_key,
        mode=str(audit_decision.get("mode") or "unknown"),
        frontier_used=frontier_used,
        required=audit_decision.get("required") is True,
        sample_rate=audit_decision.get("sample_rate"),
        caught_issue_rate=audit_decision.get("caught_issue_rate"),
        decision=decision,
    )
    _safe_log(
        f"ingest | authorization: {audit_decision.get('mode', 'unknown')} -> {decision}"
    )
    if frontier_used:
        try:
            from llm_wiki_mcp.ingest_audit import record_frontier_audit_outcome

            record_frontier_audit_outcome(
                state_path=audit_state_path,
                source_key=source_key,
                approved=decision in {"apply_available", "confirmed_noop"},
                mode=str(audit_decision.get("mode") or "mandatory"),
                reasons=[
                    str(reason)
                    for reason in audit_decision.get("reasons", [])
                    if isinstance(reason, str)
                ],
            )
        except Exception:
            pass
    if decision not in {"apply_available", "confirmed_noop"}:
        return {
            "status": "needs_retry" if decision == "retry" else decision,
            "source_key": source_key,
            "proposal_sha256": proposal_sha256,
            "review": review,
            "recovered_artifact": recovered_artifact,
            "reused_review": reused_review,
            "created": [],
            "updated": [],
            "audit": audit_decision,
        }

    # Adoption artifact writers hold this same lease.  Keep authority stable
    # across the final semantic effect: either the exact page CAS batch or the
    # confirmed-noop disposition that permits the caller to retire the raw.
    from llm_wiki_mcp.page_mutation import decision_authority_lock

    with decision_authority_lock():
        current_authority, current_authority_error = _current_ingest_review_authority(
            reviewer=reviewer
        )
        authority_compare_error = (
            decision_authority.compare_semantic_authority(
                review_authority,
                current_authority,
                lane="ingest_reconciliation",
            )
            if current_authority_error is None
            else current_authority_error
        )
        if authority_compare_error is not None:
            return {
                "status": "needs_retry",
                "source_key": source_key,
                "proposal_sha256": proposal_sha256,
                "review": review,
                "summary": authority_compare_error,
                "recovered_artifact": recovered_artifact,
                "reused_review": reused_review,
                "created": [],
                "updated": [],
                "audit": audit_decision,
            }
        if proof_error := _ingest_review_authority_error(review, review_authority):
            return {
                "status": "needs_retry",
                "source_key": source_key,
                "proposal_sha256": proposal_sha256,
                "review": review,
                "summary": proof_error,
                "recovered_artifact": recovered_artifact,
                "reused_review": reused_review,
                "created": [],
                "updated": [],
                "audit": audit_decision,
            }
        if reused_review:
            # Re-read inside the authority lease to close the gap between the
            # earlier reuse check and the final effect.
            durable_artifact = _load_ingest_review_artifact(
                review_path,
                source_key=source_key,
                proposal_sha256=proposal_sha256,
            )
            if (
                durable_artifact is None
                or durable_artifact.get("authority") != review_authority
                or durable_artifact.get("review") != review
            ):
                return {
                    "status": "needs_retry",
                    "source_key": source_key,
                    "proposal_sha256": proposal_sha256,
                    "review": review,
                    "summary": "frontier review artifact changed before effect",
                    "recovered_artifact": recovered_artifact,
                    "reused_review": True,
                    "created": [],
                    "updated": [],
                    "audit": audit_decision,
                }
        else:
            _readback, artifact_error = _write_and_readback_ingest_review_artifact(
                review_path,
                source_key=source_key,
                proposal_sha256=proposal_sha256,
                review=review,
                authority=review_authority,
            )
            if artifact_error is not None:
                return {
                    "status": "needs_retry",
                    "source_key": source_key,
                    "proposal_sha256": proposal_sha256,
                    "review": review,
                    "summary": artifact_error,
                    "recovered_artifact": recovered_artifact,
                    "reused_review": False,
                    "created": [],
                    "updated": [],
                    "audit": audit_decision,
                }
        if decision == "confirmed_noop":
            created, updated = [], []
        else:
            created, updated = _apply_prepared_operations(planned, link_totals=totals)
    return {
        "status": decision,
        "source_key": source_key,
        "proposal_sha256": proposal_sha256,
        "review": review,
        "authority": review_authority,
        "recovered_artifact": recovered_artifact,
        "reused_review": reused_review,
        "created": created,
        "updated": updated,
        "audit": audit_decision,
        **(
            {"stale_review_replaced": stale_review_reason}
            if stale_review_reason is not None
            else {}
        ),
    }


def _rebuild_index() -> None:
    """Rebuild index.md from all pages."""
    pages = sorted(all_pages())
    lines = [
        "---",
        "title: Index",
        f"updated: {date.today().isoformat()}",
        "---",
        "",
        "# Wiki Index",
        "",
    ]
    for p in pages:
        content = p.read_text()
        title_match = re.search(r"title:\s*(.+)", content)
        title = title_match.group(1) if title_match else p.stem
        lines.append(f"- [[{p.stem}]] — {title}")

    INDEX_FILE.write_text("\n".join(lines) + "\n")


def _append_log(message: str) -> None:
    """Append to log.md. Failures are intentionally swallowed.

    A dropped log line is recoverable; an exception escaping into the
    ingest pipeline is not. Letting an IO error here propagate would,
    for example, override a freshly-set ``COMPLETED`` job status with
    ``FAILED`` from the outer except block and skip ``on_complete`` —
    leaving disk pages persisted but raws marked pending, so the next
    tick collides on every page we just created.
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        with open(LOG_FILE, "a") as f:
            f.write(f"\n- [{timestamp}] {message}")
    except Exception:
        pass


def _safe_log(
    message: str,
    *,
    level: str | None = None,
    outcome_kind: str | None = None,
) -> None:
    """Defense-in-depth wrapper used by atomicity-critical call sites.

    ``_append_log`` is already internally crash-safe, but a test (or a
    monkeypatch in some future caller) can replace it with something
    that raises. The rollback path and the post-apply success path
    cannot afford to propagate such exceptions: doing so would either
    break atomicity (rollback aborts mid-loop) or override a
    successfully-set ``COMPLETED`` status with ``FAILED`` and skip
    ``on_complete``. So we wrap, swallow, and move on.
    """
    try:
        _append_log(message)
    except Exception:
        pass
    runtime_status.safe_append_event(
        level or runtime_status.classify_log_message(message),
        message,
        source="ingest",
        outcome_kind=outcome_kind,
    )


def _read_back_failure_log() -> Path:
    return PAGES_DIR.parent / "runtime" / "ingest-read-back-failures.jsonl"


def _read_back_run_log() -> Path:
    return PAGES_DIR.parent / "runtime" / "ingest-read-back-runs.jsonl"


def _read_back_query(meta: dict, page_id: str) -> str:
    questions = meta.get("recall_questions")
    if isinstance(questions, list):
        for question in questions:
            if isinstance(question, str) and question.strip():
                return question.strip()
    summary = meta.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    title = meta.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return page_id


def _verify_changed_pages_read_back(page_ids: list[str], *, top_n: int = 10) -> dict:
    if not page_ids:
        return {"checked": 0, "passed": 0, "failed": []}
    try:
        from llm_wiki_mcp.index_store import get_store
        from llm_wiki_mcp.search import search

        store = get_store()
        store.refresh()
    except Exception as e:
        _safe_log(f"ingest | read-back unavailable: {e}")
        return {"checked": 0, "passed": 0, "failed": [{"error": str(e)}]}

    checked = 0
    passed = 0
    failed: list[dict] = []
    for page_id in page_ids:
        meta = store.meta(page_id)
        if meta is None:
            failed.append({"page_id": page_id, "reason": "missing-meta"})
            continue
        query = _read_back_query(meta, page_id)
        if not query:
            failed.append({"page_id": page_id, "reason": "empty-query"})
            continue
        checked += 1
        try:
            results, mode = search(query, top_n=top_n, semantic=True)
        except Exception as e:
            failed.append(
                {"page_id": page_id, "reason": "search-error", "error": str(e)}
            )
            continue
        rank = next(
            (
                idx + 1
                for idx, result in enumerate(results)
                if result.page_id == page_id
            ),
            None,
        )
        if rank is None:
            failed.append(
                {
                    "page_id": page_id,
                    "reason": "not-in-top-results",
                    "query": query[:180],
                    "mode": mode,
                    "top": [result.page_id for result in results[:5]],
                }
            )
        else:
            passed += 1

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "schema_version": 2,
        "cohort": "all_ingest_runs",
        "checked": checked,
        "passed": passed,
        "failed": failed,
    }
    try:
        run_path = _read_back_run_log()
        run_path.parent.mkdir(parents=True, exist_ok=True)
        with run_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass
    if failed:
        try:
            log_path = _read_back_failure_log()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
        _safe_log(
            f"ingest | read-back: {len(failed)} failed of {checked} checked",
            level="warn",
            outcome_kind="read_back_warning",
        )
    elif checked:
        _safe_log(f"ingest | read-back: {checked} checked ok")

    return {"checked": checked, "passed": passed, "failed": failed}


def _refresh_ingest_derived_artifacts(
    changed_pages: list[str],
    *,
    source_raw: str | None,
) -> dict[str, Any]:
    """Refresh rebuildable indexes and return the normal read-back result."""

    try:
        _rebuild_index()
    except Exception as exc:
        _safe_log(f"ingest | index.md rebuild failed (non-fatal): {exc}")

    try:
        from llm_wiki_mcp.index_store import get_store

        get_store().refresh()
    except Exception as exc:
        _safe_log(f"ingest | index_store refresh failed: {exc}")

    if changed_pages:
        try:
            from llm_wiki_mcp.search import update_embeddings

            update_embeddings(page_ids=changed_pages)
        except Exception:
            pass
        try:
            from llm_wiki_mcp.claims import append_page_claims

            append_page_claims(
                changed_pages,
                source_raw=source_raw or "",
                op="ingest",
            )
        except Exception as exc:
            _safe_log(f"ingest | claim ledger failed (non-fatal): {exc}")
        try:
            from llm_wiki_mcp.state_register import refresh_state_register

            refresh_state_register(changed_pages, source_raw=source_raw or "")
        except Exception as exc:
            _safe_log(f"ingest | state register refresh failed (non-fatal): {exc}")
    return _verify_changed_pages_read_back(changed_pages)


def _complete_pretriage_terminal_recovery(
    recovery: dict[str, Any],
    *,
    raw_content: str,
    raw_keywords: list[str] | None,
    source_raw: str | None,
    job_id: str,
    on_complete: Callable[[], Any] | None,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> dict[str, Any]:
    """Finish a proven prior effect and publish its raw ACK under locks."""

    changed_pages = list(recovery.get("created") or []) + list(
        recovery.get("updated") or []
    )

    from llm_wiki_mcp.page_mutation import (
        decision_authority_lock,
        wiki_mutation_lock,
    )

    # Match the normal effect lock order.  The authority lease prevents a
    # terminal review artifact from being replaced while the page lease keeps
    # exact postimages stable through job completion and ACK publication.
    with decision_authority_lock():
        page_lease = (
            wiki_mutation_lock()
            if recovery.get("status") == "apply_available"
            else nullcontext()
        )
        with page_lease:
            verified = _load_pretriage_terminal_recovery(
                raw_content,
                raw_keywords,
                reviewer=reviewer,
            )
            if verified is None or verified != recovery:
                raise IngestApplyError(
                    "pre-triage terminal recovery proof changed before raw retirement"
                )
            # Derived refresh is intentionally inside the same proof/effect
            # locks. A stale or concurrently replaced terminal artifact must
            # fail before even rebuildable claims/index side effects occur.
            read_back_result = _refresh_ingest_derived_artifacts(
                changed_pages,
                source_raw=source_raw,
            )
            final_verified = _load_pretriage_terminal_recovery(
                raw_content,
                raw_keywords,
                reviewer=reviewer,
            )
            if final_verified is None or final_verified != recovery:
                raise IngestApplyError(
                    "pre-triage terminal recovery proof changed during derived refresh"
                )
            job_result: dict[str, Any] = {
                "frontier": {
                    "status": recovery.get("status"),
                    "proposal_sha256": recovery.get("proposal_sha256"),
                    "source_key": recovery.get("source_key"),
                    "review": recovery.get("review"),
                    "recovered_artifact": True,
                    "reused_review": True,
                },
                "audit": recovery.get("audit"),
                "pretriage_recovery": {
                    "basis": recovery.get("recovery_basis"),
                    "model_calls": 0,
                },
            }
            failed_specs = list(recovery.get("failed_operation_specs") or [])
            if failed_specs:
                job_result.update({"partial": True, "failed_ops": failed_specs})
            if read_back_result.get("failed"):
                job_result["read_back"] = read_back_result
            job_store.update(
                job_id,
                status=JobStatus.COMPLETED,
                processor="durable-ingest-recovery",
                stage="pre-triage-recovery",
                completed_at=_now(),
                pages_created=list(recovery.get("created") or []),
                pages_updated=list(recovery.get("updated") or []),
                result=job_result,
            )
            if on_complete:
                # For orchestrated raws this publishes the content-bound ACK
                # before the processed-state transition.  Any callback error
                # escapes and the outer job boundary records FAILED.
                on_complete()
    return job_result


# ---------------------------------------------------------------------------
# Main entry point — two-stage pipeline
# ---------------------------------------------------------------------------

_MAX_FRONTIER_CONVERGENCE_ATTEMPTS = 3
_MAX_FRONTIER_CALLS_PER_RAW = 2


@dataclass
class _FrontierCallBudget:
    """Bound frontier use while local generation converges for one raw."""

    limit: int = _MAX_FRONTIER_CALLS_PER_RAW
    used: int = 0

    def consume(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


def _frontier_feedback_text(result: dict[str, Any]) -> str:
    """Return compact authoritative feedback for the next local proposal."""

    review = result.get("review")
    if isinstance(review, dict):
        parts = [
            str(review.get(key)).strip()
            for key in ("summary", "risk", "notes")
            if isinstance(review.get(key), str) and str(review.get(key)).strip()
        ]
        if parts:
            return "\n".join(dict.fromkeys(parts))
    summary = result.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return "The previous proposal was not safe or complete enough to apply."


def _structured_frontier_failure_class(result: dict[str, Any]) -> str | None:
    """Return a stable control-plane reason from a structured review failure."""

    review = result.get("review")
    for candidate in (result, review):
        if not isinstance(candidate, dict):
            continue
        frontier_failure = candidate.get("frontier_failure")
        if isinstance(frontier_failure, dict) and frontier_failure:
            failure_class = str(frontier_failure.get("failure_class") or "").strip()
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", failure_class):
                return failure_class.casefold()
            return "structured_review_failure"
    return None


def _frontier_retry_is_actionable(result: dict[str, Any]) -> bool:
    """False when regenerating local content cannot repair the frontier lane."""

    if str(result.get("status") or "") == "quarantined":
        return False
    if _structured_frontier_failure_class(result) is not None:
        # A structured reviewer failure is control-plane evidence, not
        # semantic feedback about the generated page.  Regenerating the same
        # page cannot repair schema, quorum, transport, authority, or policy
        # failures and only burns the second review call before the raw is
        # mislabeled as non-convergent.
        return False
    feedback = _frontier_feedback_text(result).casefold()
    infrastructure_markers = (
        "adoption artifact",
        "adoption_artifact",
        "artifact write failed",
        "authority changed",
        "authority is missing",
        "frontier reviewer failed",
        "local consensus reviewer failed",
        "transport",
        "timeout",
        "timed out",
        "budget exhausted",
        "budget deferred",
        "authentication",
        "billing",
        "quota",
        "secret-store",
    )
    return not any(marker in feedback for marker in infrastructure_markers)


def _apply_frontier_replacement_operations(
    operations: list[dict],
    result: dict[str, Any],
) -> tuple[list[dict], list[str]]:
    """Materialize exact, filename-scoped repair postimages for re-review.

    Prose fields are deliberately ignored.  A tag mutation is accepted only
    when one structured ``invalid_tags`` value and one named full replacement
    postimage encode the same deletion.  The same tag on every other operation
    therefore remains untouched even when filenames share a taxonomy tag.
    """

    review = result.get("review")
    replacements_raw = (
        review.get("replacement_operations") if isinstance(review, dict) else None
    )
    if not isinstance(replacements_raw, list) or not replacements_raw:
        return operations, []
    invalid_tags_raw = review.get("invalid_tags") if isinstance(review, dict) else []
    if invalid_tags_raw is None:
        invalid_tags_raw = []
    if (
        not isinstance(invalid_tags_raw, list)
        or len(invalid_tags_raw) > 1
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[dts]/[a-z0-9][a-z0-9-]*", value) is None
            for value in invalid_tags_raw
        )
    ):
        return operations, []
    invalid_tags = set(invalid_tags_raw)

    existing = {
        operation.get("filename"): operation
        for operation in operations
        if isinstance(operation.get("filename"), str)
    }
    from llm_wiki_mcp.frontmatter import parse

    replacements: dict[str, str] = {}
    tag_changed_files: list[str] = []
    for item in replacements_raw:
        if not isinstance(item, dict):
            return operations, []
        filename = item.get("filename")
        content = item.get("content")
        if (
            not isinstance(filename, str)
            or filename not in existing
            or filename in replacements
            or not isinstance(content, str)
            or not content.strip()
        ):
            return operations, []
        op_type = str(existing[filename].get("type") or "")
        if op_type == "create":
            if not _has_frontmatter(content):
                return operations, []
            normalized_content = content
            original_content = existing[filename].get("content")
            if isinstance(original_content, str) and _has_frontmatter(original_content):
                original_meta, _ = parse(original_content)
                replacement_meta, _ = parse(normalized_content)
                original_tags = original_meta.get("tags")
                replacement_tags = replacement_meta.get("tags")
                if replacement_tags != original_tags:
                    if (
                        not invalid_tags
                        or not isinstance(original_tags, list)
                        or not isinstance(replacement_tags, list)
                        or replacement_tags
                        != [tag for tag in original_tags if tag not in invalid_tags]
                        or invalid_tags
                        != {tag for tag in original_tags if tag not in replacement_tags}
                    ):
                        return operations, []
                    tag_changed_files.append(filename)
        elif op_type == "update":
            # Repair arrays are hash-bound host postimages. Normalizing them
            # after quorum would apply bytes different from the authorized
            # action, so update repairs must already be body-only and are
            # preserved byte-for-byte.
            if _strip_all_frontmatter(content) != content or not content.strip():
                return operations, []
            normalized_content = content
        else:
            return operations, []
        replacements[filename] = normalized_content

    if invalid_tags and len(tag_changed_files) != 1:
        return operations, []

    repaired: list[dict] = []
    for operation in operations:
        filename = operation.get("filename")
        updated = dict(operation)
        if filename in replacements:
            updated["content"] = replacements[str(filename)]
        repaired.append(updated)
    return repaired, sorted(replacements)


def _generate_local_operations(
    plan: list[dict],
    *,
    content: str,
    raw_keywords: list[str] | None,
    source_raw: str | None,
    job_id: str,
    frontier_feedback: str | None,
) -> tuple[list[dict], list[dict]]:
    """Generate every operation through one bounded logical model session."""

    generation_stage = "local-regenerate" if frontier_feedback else "generate"
    generation_phase = "local-regenerate-generate" if frontier_feedback else "generate"
    job_store.update(job_id, stage="generate", total_ops=len(plan), completed_ops=0)
    runtime_status.safe_write_status(
        state="running",
        stage=generation_stage,
        current_job_id=job_id,
        current_raw=source_raw,
        current_op=None,
        op_progress={"index": 0, "total": len(plan)},
    )
    operations: list[dict] = []
    failed_specs: list[dict] = []
    for i, op in enumerate(plan):
        fname = op.get("filename", "?")
        runtime_status.safe_write_status(
            state="running",
            stage=generation_stage,
            current_job_id=job_id,
            current_raw=source_raw,
            current_op=fname,
            op_progress={"index": i + 1, "total": len(plan)},
        )
        _safe_log(f"ingest | generating {i + 1}/{len(plan)}: {fname}")
        generation_diagnostics: dict[str, Any] = {}
        generated = _generate_one_with_progress(
            op,
            content,
            raw_keywords=raw_keywords,
            progress_callback=_llm_progress_callback(
                phase=generation_phase,
                target=fname,
                job_id=job_id,
                source_raw=source_raw,
                op_progress={"index": i + 1, "total": len(plan)},
            ),
            frontier_feedback=frontier_feedback,
            diagnostics=generation_diagnostics,
        )
        if generated:
            operations.append(generated)
        else:
            failed_specs.append(
                {
                    "filename": fname,
                    "type": op.get("type", "?"),
                    "title": op.get("title", ""),
                    "summary": op.get("summary", ""),
                    "error": str(
                        generation_diagnostics.get("reason")
                        or "generation validation failed"
                    ),
                    "failure_class": str(
                        generation_diagnostics.get("failure_class") or "unknown"
                    ),
                    "attempts": int(generation_diagnostics.get("attempts") or 1),
                }
            )
        job_store.update(job_id, completed_ops=i + 1)
    return operations, failed_specs


def _all_generation_failure_error(
    plan: list[dict],
    operations: list[dict],
    failed_specs: list[dict],
) -> str | None:
    """Describe an all-generation runtime failure before semantic review.

    A reviewer cannot safely approve an empty postimage merely because the
    page generator failed.  Surface the bounded local failure directly so the
    failure supervisor can defer the valid raw behind one operational repair
    packet instead of quarantining it as a semantic disagreement.
    """

    if not plan or operations or not failed_specs:
        return None
    classes = {
        str(spec.get("failure_class") or "validation_failed").casefold()
        for spec in failed_specs
    }
    operational_priority = (
        "transport_error",
        "repeated_output",
        "repair_exhausted",
        "validation_failed",
    )
    failure_class = next(
        (candidate for candidate in operational_priority if candidate in classes),
        None,
    )
    if failure_class is None:
        return None
    representative = next(
        (
            spec
            for spec in failed_specs
            if str(spec.get("failure_class") or "validation_failed").casefold()
            == failure_class
        ),
        failed_specs[0],
    )
    filename = str(representative.get("filename") or "unknown")
    reason = str(representative.get("error") or "bounded page generation failed")[:500]
    return (
        f"ingest generation {failure_class}: all {len(plan)} planned page "
        f"operations failed locally; first={filename}: {reason}"
    )


def run_ingest(
    content: str,
    job_id: str,
    on_complete: "callable | None" = None,
    on_finally: "callable | None" = None,
    *,
    metadata: dict | None = None,
    frontier_reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> None:
    """Run two-stage ingest in background thread.

    ``on_complete`` fires whenever any pages were applied — full success or
    partial-with-some-ops-failed. The orchestrator uses it to mark raws
    processed; partial apply still counts because the next tick mustn't
    re-triage the same content (that's how the prior contract caused
    duplicate page creation). ``on_finally`` fires after every terminal
    state — success, partial, parse failure, or apply failure — and is
    called with two flags:

      * ``failed`` — True for any non-success terminal state.
      * ``triage_failed`` — True only when triage (stage 1) could not produce
        a parseable plan. Callers may use this for diagnostics, but retry and
        quarantine decisions are owned by the per-raw failure supervisor,
        keyed by raw filename and stable failure fingerprint. Failures from
        unrelated raws must never be aggregated into a batch mutation.

    ``metadata`` is a keyword-only optional dict carrying side-channel data
    that should be attached to the resulting operations. Currently supports
    ``raw_keywords: list[str]`` — keywords lifted from the raw frontmatter
    that need to land on the page frontmatter without re-running an LLM.
    Unknown keys are ignored so future extensions don't break callers.
    """
    job_store.update(job_id, status=JobStatus.RUNNING)
    failed = True  # flipped to False on full-success path
    triage_failed = False

    # Extract the raw_keywords side channel from metadata once, up front.
    # Every operation generated from this raw shares the same propagated
    # value: a single raw can produce N operations (e.g. 1 create + 1
    # update from the same session), and the source-of-truth keywords
    # belong to all of them. Anything that isn't a list[str] is treated
    # as "no metadata" so we don't fabricate values.
    raw_keywords_for_ops: list[str] | None = None
    source_raw: str | None = None
    if metadata is not None:
        candidate = metadata.get("raw_keywords")
        if isinstance(candidate, list) and all(isinstance(v, str) for v in candidate):
            raw_keywords_for_ops = list(candidate)
        source_candidate = metadata.get("source_raw")
        if isinstance(source_candidate, str):
            source_raw = source_candidate

    try:
        terminal_recovery = _load_pretriage_terminal_recovery(
            content,
            raw_keywords_for_ops,
            reviewer=frontier_reviewer,
        )
        if terminal_recovery is not None:
            runtime_status.safe_write_status(
                state="running",
                stage="pre-triage-recovery",
                current_job_id=job_id,
                current_raw=source_raw,
                current_op=None,
                llm=None,
            )
            recovered_result = _complete_pretriage_terminal_recovery(
                terminal_recovery,
                raw_content=content,
                raw_keywords=raw_keywords_for_ops,
                source_raw=source_raw,
                job_id=job_id,
                on_complete=on_complete,
                reviewer=frontier_reviewer,
            )
            runtime_status.safe_write_status(
                state="running",
                stage="complete",
                current_job_id=job_id,
                current_raw=source_raw,
                current_op=None,
                llm=None,
                last_success={
                    "job_id": job_id,
                    "raw": source_raw,
                    "created": list(terminal_recovery.get("created") or []),
                    "updated": list(terminal_recovery.get("updated") or []),
                    "frontier_status": terminal_recovery.get("status"),
                    "audit": terminal_recovery.get("audit"),
                    "recovery_basis": terminal_recovery.get("recovery_basis"),
                    "model_calls": 0,
                    "read_back": recovered_result.get("read_back"),
                },
            )
            failed = False
            return

        processor = "ollama" if is_available() else "unavailable"
        job_store.update(job_id, processor=processor)
        if processor == "unavailable":
            raise RuntimeError("ollama unavailable; no fallback processor configured")

        # Production semantic ingest cannot succeed without an adopted local
        # authority.  Resolve that proof before spending any triage or page-
        # generation tokens.  Tests and explicit dependency injection retain
        # their isolated reviewer boundary.
        if getattr(_review_and_apply_ingest_operations, "__module__", None) == __name__:
            review_authority, review_authority_error = _current_ingest_review_authority(
                reviewer=frontier_reviewer
            )
            review_authority_shape_error = (
                _ingest_review_authority_shape_error(review_authority)
                if review_authority is not None
                else None
            )
            review_authority_problem = (
                review_authority_error
                or review_authority_shape_error
                or (None if review_authority is not None else "authority is missing")
            )
            if review_authority_problem is not None:
                raise IngestApplyError(
                    "local consensus authority unavailable: " + review_authority_problem
                )

        # Triage chooses the mutation target once.  A local-consensus rejection
        # critiques the generated postimage, so later attempts regenerate only
        # that postimage with the exact feedback. Re-running triage can silently
        # switch pages and can misattribute a control-plane failure to the raw.
        frontier_feedback: str | None = None
        frontier_result: dict[str, Any] | None = None
        frontier_budget = _FrontierCallBudget()
        plan: list[dict] = []
        all_operations: list[dict] = []
        failed_op_specs: list[dict] = []
        for convergence_attempt in range(1, _MAX_FRONTIER_CONVERGENCE_ATTEMPTS + 1):
            job_store.update(
                job_id,
                stage="triage" if convergence_attempt == 1 else "generate",
            )
            runtime_status.safe_write_status(
                state="running",
                stage="triage" if convergence_attempt == 1 else "local-regenerate",
                current_job_id=job_id,
                current_raw=source_raw,
                current_op=None,
            )
            _safe_log(
                "ingest | stage 1: triage started"
                if convergence_attempt == 1
                else (
                    "ingest | local consensus convergence "
                    f"{convergence_attempt}/{_MAX_FRONTIER_CONVERGENCE_ATTEMPTS} started"
                )
            )
            if convergence_attempt == 1:
                try:
                    raw_plan = _triage_with_progress(
                        content,
                        _llm_progress_callback(
                            phase="triage",
                            target="operation plan",
                            job_id=job_id,
                            source_raw=source_raw,
                        ),
                    )
                except IngestTriageFailure as triage_error:
                    triage_failed = True
                    error = str(triage_error)
                    job_store.update(
                        job_id,
                        status=JobStatus.FAILED,
                        completed_at=_now(),
                        error=error,
                    )
                    runtime_status.safe_write_status(
                        state="error",
                        stage="triage",
                        current_job_id=job_id,
                        current_raw=source_raw,
                        current_op=None,
                        last_error=error,
                        llm=None,
                    )
                    _safe_log(f"ingest | triage: {error}")
                    return
                if raw_plan is None:
                    triage_failed = True
                    error = (
                        "triage structured failure [unknown]: triage returned no plan "
                        "after its bounded same-session repair turns"
                    )
                    job_store.update(
                        job_id,
                        status=JobStatus.FAILED,
                        completed_at=_now(),
                        error=error,
                    )
                    runtime_status.safe_write_status(
                        state="error",
                        stage="triage",
                        current_job_id=job_id,
                        current_raw=source_raw,
                        current_op=None,
                        last_error=error,
                        llm=None,
                    )
                    _safe_log(f"ingest | triage: {error}")
                    return

                plan = _normalize_triage_plan(raw_plan)
                plan = _dedupe_create_ops_with_existing(plan, content)
                _safe_log(f"ingest | triage: {len(plan)} operations planned")
            if plan:
                all_operations, failed_op_specs = _generate_local_operations(
                    plan,
                    content=content,
                    raw_keywords=raw_keywords_for_ops,
                    source_raw=source_raw,
                    job_id=job_id,
                    frontier_feedback=frontier_feedback,
                )
            else:
                all_operations, failed_op_specs = [], []

            generation_error = _all_generation_failure_error(
                plan,
                all_operations,
                failed_op_specs,
            )
            if generation_error is not None:
                raise IngestApplyError(generation_error)

            failed_ops = [spec["filename"] for spec in failed_op_specs]
            runtime_status.safe_write_status(
                state="running",
                stage="authorization",
                current_job_id=job_id,
                current_raw=source_raw,
                current_op=None,
                op_progress={"index": len(plan), "total": len(plan)},
                llm=None,
            )
            frontier_result = _review_and_apply_ingest_operations(
                all_operations,
                raw_content=content,
                raw_keywords=raw_keywords_for_ops,
                source_raw=source_raw,
                triage_plan=plan,
                failed_operation_specs=failed_op_specs,
                local_disposition=(
                    "triage_no_operations"
                    if not plan
                    else "all_generation_failed"
                    if failed_ops and not all_operations
                    else "partial_generation_failed"
                    if failed_ops
                    else "operations_available"
                ),
                reviewer=frontier_reviewer,
                force_frontier_review=frontier_feedback is not None,
                frontier_budget=frontier_budget,
            )
            frontier_status = str(frontier_result.get("status") or "needs_retry")
            if frontier_status in {"apply_available", "confirmed_noop"}:
                break
            if frontier_status == "frontier_budget_exhausted":
                raise IngestApplyError(
                    "local consensus ingest review did not converge after "
                    f"{frontier_budget.used} local review calls: "
                    + _frontier_feedback_text(frontier_result)
                )
            repaired_operations, replaced_files = (
                _apply_frontier_replacement_operations(
                    all_operations,
                    frontier_result,
                )
            )
            if replaced_files:
                _safe_log(
                    "ingest | exact local quorum repair replaced: "
                    + ", ".join(replaced_files)
                )
                all_operations = repaired_operations
                frontier_result = _review_and_apply_ingest_operations(
                    all_operations,
                    raw_content=content,
                    raw_keywords=raw_keywords_for_ops,
                    source_raw=source_raw,
                    triage_plan=plan,
                    failed_operation_specs=failed_op_specs,
                    local_disposition=(
                        "triage_no_operations"
                        if not plan
                        else "all_generation_failed"
                        if failed_ops and not all_operations
                        else "partial_generation_failed"
                        if failed_ops
                        else "operations_available"
                    ),
                    reviewer=frontier_reviewer,
                    force_frontier_review=True,
                    frontier_budget=frontier_budget,
                )
                frontier_status = str(frontier_result.get("status") or "needs_retry")
                if frontier_status in {"apply_available", "confirmed_noop"}:
                    break
                if frontier_status == "frontier_budget_exhausted":
                    raise IngestApplyError(
                        "local consensus ingest review did not converge after "
                        f"{frontier_budget.used} local review calls: "
                        + _frontier_feedback_text(frontier_result)
                    )
            frontier_feedback = _frontier_feedback_text(frontier_result)
            if not _frontier_retry_is_actionable(frontier_result):
                structured_failure = _structured_frontier_failure_class(frontier_result)
                typed_feedback = (
                    f"{structured_failure}: {frontier_feedback}"
                    if structured_failure is not None
                    else frontier_feedback
                )
                raise IngestApplyError(
                    "local consensus authority unavailable: " + typed_feedback
                )
            if frontier_budget.used >= frontier_budget.limit:
                raise IngestApplyError(
                    "local consensus ingest review did not converge after "
                    f"{frontier_budget.used} local review calls: structured review "
                    f"budget exhausted ({frontier_budget.used}/{frontier_budget.limit}); "
                    + frontier_feedback
                )
            _safe_log(
                "ingest | local consensus requested regeneration: "
                + frontier_feedback.replace("\n", " ")[:300]
            )
        else:
            raise IngestApplyError(
                "local consensus ingest review did not converge after "
                f"{_MAX_FRONTIER_CONVERGENCE_ATTEMPTS} attempts: "
                + (frontier_feedback or "unknown local consensus rejection")
            )

        assert frontier_result is not None
        frontier_status = str(frontier_result.get("status") or "needs_retry")
        created = list(frontier_result.get("created") or [])
        updated = list(frontier_result.get("updated") or [])

        # Side effects (rebuild_index, IndexStore refresh, embeddings) are
        # derived artifacts. Pages are already on disk; failures here must
        # NOT undo the apply or block on_complete — that would leave raws
        # pending forever and re-create the same pages on retry. Use
        # _safe_log so a logging error in a side-effect handler doesn't
        # promote a derived-artifact failure into a hard ingest failure.
        changed_pages = created + updated
        read_back_result = _refresh_ingest_derived_artifacts(
            changed_pages,
            source_raw=source_raw,
        )

        # Build the user-facing local-consensus result.  The legacy
        # ``frontier`` alias is retained for old job readers, but both views
        # deliberately exclude raw/page bodies; exact bytes stay in artifacts.
        consensus_result = {
            "status": frontier_status,
            "proposal_sha256": frontier_result.get("proposal_sha256"),
            "source_key": frontier_result.get("source_key"),
            "review": frontier_result.get("review"),
            "recovered_artifact": bool(frontier_result.get("recovered_artifact")),
            "reused_review": bool(frontier_result.get("reused_review")),
        }
        job_result: dict | None = {
            "local_consensus": consensus_result,
            "frontier": dict(consensus_result),
            "audit": frontier_result.get("audit"),
        }
        if failed_op_specs:
            job_result.update(
                {
                    "partial": True,
                    "failed_ops": failed_op_specs,
                }
            )
        if read_back_result["failed"]:
            job_result = job_result or {}
            job_result["read_back"] = read_back_result

        noop_callback_completed = False
        if frontier_status == "confirmed_noop":
            # Unlike apply_available there is no page CAS receipt that makes
            # later raw retirement an exact-postimage recovery.  Keep the
            # semantic authority epoch stable through both the terminal job
            # transition and the callback that retires the source raw.
            from llm_wiki_mcp.page_mutation import decision_authority_lock

            confirmed_authority = frontier_result.get("authority")
            confirmed_review = frontier_result.get("review")
            with decision_authority_lock():
                current_authority, current_authority_error = (
                    _current_ingest_review_authority(reviewer=frontier_reviewer)
                )
                authority_compare_error = (
                    decision_authority.compare_semantic_authority(
                        confirmed_authority,
                        current_authority,
                        lane="ingest_reconciliation",
                    )
                    if current_authority_error is None
                    else current_authority_error
                )
                if authority_compare_error is not None:
                    raise IngestApplyError(
                        "ingest confirmed-noop authority changed before raw retirement: "
                        f"{authority_compare_error}"
                    )
                if not isinstance(confirmed_review, dict) or not isinstance(
                    confirmed_authority, dict
                ):
                    raise IngestApplyError(
                        "ingest confirmed-noop proof missing before raw retirement"
                    )
                if proof_error := _ingest_review_authority_error(
                    confirmed_review,
                    confirmed_authority,
                ):
                    raise IngestApplyError(
                        "ingest confirmed-noop proof invalid before raw retirement: "
                        f"{proof_error}"
                    )
                _proposal_path, confirmed_review_path = _ingest_artifact_paths(
                    str(frontier_result.get("source_key") or "")
                )
                durable_review = _load_ingest_review_artifact(
                    confirmed_review_path,
                    source_key=str(frontier_result.get("source_key") or ""),
                    proposal_sha256=str(frontier_result.get("proposal_sha256") or ""),
                )
                if (
                    durable_review is None
                    or durable_review.get("authority") != confirmed_authority
                    or durable_review.get("review") != confirmed_review
                ):
                    raise IngestApplyError(
                        "ingest confirmed-noop durable proof changed before raw retirement"
                    )
                job_store.update(
                    job_id,
                    status=JobStatus.COMPLETED,
                    completed_at=_now(),
                    pages_created=created,
                    pages_updated=updated,
                    result=job_result,
                )
                if on_complete:
                    # Raw retirement is the second half of the durable apply
                    # transaction.  Its callback publishes the completion ACK
                    # and then marks the source processed.  Swallowing an ACK
                    # failure here reports a false success and replays the
                    # already-applied mutation on the next tick.
                    on_complete()
                noop_callback_completed = True
        else:
            job_store.update(
                job_id,
                status=JobStatus.COMPLETED,
                completed_at=_now(),
                pages_created=created,
                pages_updated=updated,
                result=job_result,
            )
        # _safe_log so a log failure here can't fall through to the outer
        # except, override COMPLETED with FAILED, and skip on_complete.
        # That was the R5-Critical regression path.
        if failed_op_specs:
            _safe_log(
                f"ingest | local-consensus-final: {len(created)} created, "
                f"{len(updated)} updated, "
                f"{len(failed_op_specs)} local generation failures confirmed unnecessary "
                f"({', '.join(failed_ops[:3])}"
                + ("..." if len(failed_ops) > 3 else "")
                + ")"
            )
        else:
            _safe_log(
                f"ingest | completed: {len(created)} created, {len(updated)} updated"
            )
        runtime_status.safe_write_status(
            state="running",
            stage="complete",
            current_job_id=job_id,
            current_raw=source_raw,
            current_op=None,
            llm=None,
            last_success={
                "job_id": job_id,
                "raw": source_raw,
                "created": created,
                "updated": updated,
                "failed_ops": failed_op_specs,
                "local_consensus_status": frontier_status,
                "frontier_status": frontier_status,
                "audit": frontier_result.get("audit"),
                "failed_operations_disposition": (
                    (frontier_result.get("review") or {}).get(
                        "failed_operations_disposition"
                    )
                    if isinstance(frontier_result.get("review"), dict)
                    else None
                ),
                "read_back": read_back_result,
            },
        )

        if on_complete and not noop_callback_completed:
            on_complete()
        failed = False

    except Exception as e:
        job_store.update(
            job_id,
            status=JobStatus.FAILED,
            completed_at=_now(),
            error=str(e),
        )
        runtime_status.safe_write_status(
            state="error",
            stage="failed",
            current_job_id=job_id,
            current_raw=source_raw,
            current_op=None,
            last_error=str(e),
            llm=None,
        )
        _safe_log(f"ingest | failed: {e}")
    finally:
        if on_finally:
            try:
                on_finally(failed=failed, triage_failed=triage_failed)
            except Exception as cb_err:
                _safe_log(f"ingest | on_finally callback failed: {cb_err}")


def start_ingest(
    content: str,
    on_complete: "callable | None" = None,
    on_finally: "callable | None" = None,
    *,
    metadata: dict | None = None,
) -> str:
    """Start an async ingest job. Returns job_id.

    ``metadata`` is forwarded to :func:`run_ingest` (keyword-only) so callers
    that want raw-side context (e.g. ``raw_keywords``) propagated to the
    resulting operations can pass it through without changing positional
    argument order.
    """
    processor = "ollama" if is_available() else "unavailable"
    job = job_store.create(processor=processor)

    thread = threading.Thread(
        target=run_ingest,
        args=(content, job.job_id, on_complete, on_finally),
        kwargs={"metadata": metadata},
        daemon=True,
    )
    thread.start()

    return job.job_id


def _now() -> str:
    from datetime import datetime

    return datetime.now().isoformat()
