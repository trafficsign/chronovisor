"""Ingest engine - structures raw data into wiki pages (two-stage pipeline)."""

import ast
import base64
import copy
import hashlib
import json
import re
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher
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
from llm_wiki_mcp.ingest_schemas import (
    INGEST_FRONTIER_ARTIFACT_SCHEMA_VERSION,
    INGEST_FRONTIER_DECISION_SCHEMA,
    INGEST_FRONTIER_LEGACY_ARTIFACT_SCHEMA_VERSION as _INGEST_FRONTIER_LEGACY_ARTIFACT_SCHEMA_VERSION,
    INGEST_FRONTIER_REVIEW_ARTIFACT_SCHEMA_VERSION,
    INGEST_REVIEW_LIMIT_FIELDS as _INGEST_REVIEW_LIMIT_FIELDS,
    INGEST_REVIEW_SHARD_POLICY_VERSION,
    INGEST_REVIEW_SHARD_ROW_FIELDS as _INGEST_REVIEW_SHARD_ROW_FIELDS,
    INGEST_REVIEW_SHARD_SCHEMA_VERSION,
    MAX_INGEST_REVIEW_SHARDS as _MAX_INGEST_REVIEW_SHARDS,
    RECALL_METADATA_SCHEMA,
    TRIAGE_CATALOG_TOP_N as _TRIAGE_CATALOG_TOP_N,
    TRIAGE_MAX_FEEDBACK_BYTES as _TRIAGE_MAX_FEEDBACK_BYTES,
    TRIAGE_MAX_OPERATIONS as _TRIAGE_MAX_OPERATIONS,
    TRIAGE_MAX_OUTPUT_BYTES as _TRIAGE_MAX_OUTPUT_BYTES,
    TRIAGE_NUM_PREDICT as _TRIAGE_NUM_PREDICT,
    TRIAGE_PLAN_SCHEMA,
    TRIAGE_PLAN_VALIDATION_SCHEMA as _TRIAGE_PLAN_VALIDATION_SCHEMA,
)
from llm_wiki_mcp.ingest_transport import (
    generate_with_progress as _generate_with_progress_core,
    llm_progress_callback as _llm_progress_callback_core,
    structured_chat_transport as _structured_chat_transport_core,
    structured_generate_transport as _structured_generate_transport_core,
    supports_keyword as _supports_keyword,
)
from llm_wiki_mcp.ingest_review_plan import (
    IngestReviewBudgetExhausted,
    IngestReviewShard as _IngestReviewShard,
    IngestReviewShardCapacityError,
    IngestReviewShardPlan as _IngestReviewShardPlan,
    IngestReviewShardPlanState as _IngestReviewShardPlanState,
    build_ingest_review_shard_plan as _build_ingest_review_shard_plan_core,
    build_ingest_review_shard_proposal as _build_ingest_review_shard_proposal_core,
    measure_ingest_review_request as _measure_ingest_review_request_core,
    validate_ingest_shard_source_rows as _validate_ingest_shard_source_rows_core,
)
from llm_wiki_mcp.ingest_review import (
    normalize_ingest_frontier_review as _normalize_ingest_frontier_review_core,
    run_ingest_frontier_review as _run_ingest_frontier_review_core,
)
from llm_wiki_mcp.ingest_review_authority import (
    current_ingest_review_authority as _current_ingest_review_authority_core,
    ingest_review_authority_error as _ingest_review_authority_error_core,
    ingest_review_authority_shape_error as _ingest_review_authority_shape_error_core,
    ingest_review_shard_proof_error as _ingest_review_shard_proof_error_core,
)
from llm_wiki_mcp.ingest_review_store import (
    continuation_marker_path as _continuation_marker_path_core,
    ingest_artifact_paths as _ingest_artifact_paths_core,
    load_ingest_proposal as _load_ingest_proposal_core,
    load_ingest_review as _load_ingest_review_core,
    load_ingest_review_artifact as _load_ingest_review_artifact_core,
    repair_transition_path as _repair_transition_path_core,
    review_stall_path as _review_stall_path_core,
    sealed_ingest_review_artifact as _sealed_ingest_review_artifact_core,
    write_and_readback_ingest_review_artifact as _write_and_readback_ingest_review_artifact_core,
    write_ingest_artifact as _write_ingest_artifact_core,
)
from llm_wiki_mcp.ingest_review_recovery import (
    consume_ingest_review_continuation_marker as _consume_ingest_review_continuation_marker_core,
    load_ingest_review_continuation_marker as _load_ingest_review_continuation_marker_core,
    matching_ingest_review_stall_error as _matching_ingest_review_stall_error_core,
    persist_ingest_review_continuation_marker as _persist_ingest_review_continuation_marker_core,
    persist_ingest_review_stall as _persist_ingest_review_stall_core,
    seal_ingest_review_repair_transition as _seal_ingest_review_repair_transition_core,
)
from llm_wiki_mcp.ingest_review_execution import (
    IngestShardedReviewDeps,
    ingest_review_shard_aggregate as _ingest_review_shard_aggregate_core,
    ingest_review_shard_failure as _ingest_review_shard_failure_core,
    ingest_review_shard_manifest_artifact_payload as _ingest_review_shard_manifest_artifact_payload_core,
    ingest_review_shard_manifest_path as _ingest_review_shard_manifest_path_core,
    ingest_review_shard_review_identity as _ingest_review_shard_review_identity_core,
    persist_ingest_review_shard_manifest as _persist_ingest_review_shard_manifest_core,
    run_ingest_sharded_review as _run_ingest_sharded_review_core,
    stored_ingest_review_shard_manifest_error as _stored_ingest_review_shard_manifest_error_core,
)
from llm_wiki_mcp.entities import patch_entities_frontmatter
from llm_wiki_mcp.triage_plan import (
    collapse_exact_duplicate_operations,
    distinct_target_collisions,
)
from llm_wiki_mcp.search_types import tokenize


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
    return _generate_with_progress_core(
        generate,
        prompt,
        system=system,
        progress_callback=progress_callback,
        format=format,
        model=model,
        num_ctx=num_ctx,
        num_predict=num_predict,
        keep_alive=keep_alive,
        read_timeout_ms=read_timeout_ms,
        temperature=temperature,
        seed=seed,
        return_metadata=return_metadata,
    )


_DEFAULT_GENERATE_WITH_PROGRESS = _generate_with_progress


def _structured_generate_transport(
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ChatTransport:
    """Adapt the legacy generate fixture/progress seam to chat-style history."""

    return _structured_generate_transport_core(
        _generate_with_progress, progress_callback
    )


def _structured_chat_transport() -> ChatTransport:
    """Preserve native chat roles for production structured repair turns.

    The historical ingest seam above flattens messages into one generate
    transcript so narrow test fixtures can keep replacing
    ``_generate_with_progress``.  Production must not use that compatibility
    path: Ollama's chat endpoint retains the assistant response and the exact
    validator feedback as separate roles, which is the contract
    ``LocalStructuredSession`` repairs against.
    """

    return _structured_chat_transport_core(ollama_runtime.chat)


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
    return _llm_progress_callback_core(
        runtime_status,
        phase=phase,
        target=target,
        job_id=job_id,
        source_raw=source_raw,
        op_progress=op_progress,
    )


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
    from llm_wiki_mcp.ingest_triage import triage

    return triage(
        content,
        progress_callback=progress_callback,
        frontier_feedback=frontier_feedback,
        transport=transport,
        raise_on_failure=raise_on_failure,
    )


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
    reserved_system_keys = _reserved_system_page_collision_keys()
    for index, operation in enumerate(operations):
        filename = (
            _effective_triage_target_filename(operation)
            if resolve_effective_targets
            else operation.get("filename")
        )
        target_key = _target_page_collision_key(filename)
        if target_key is None or target_key not in reserved_system_keys:
            continue
        requested_filename = operation.get("filename")
        issues.append(
            ValidationIssue(
                pointer=f"/{index}/filename",
                keyword="reservedSystemTarget",
                expected={
                    "rule": (
                        "normal ingest cannot create or update reserved system pages"
                    ),
                    "repair": (
                        "return a complete corrected array; replace this operation "
                        "with a grounded create/update for a normal knowledge page, "
                        "or return [] only when the raw warrants no normal knowledge-"
                        "page operation; preserve every unrelated valid operation"
                    ),
                },
                received={"type": "string", "value": requested_filename},
                message=(
                    "this operation targets a reserved system page outside normal "
                    "ingest authority. Re-evaluate the complete array from the raw: "
                    "retarget the grounded fact to a normal knowledge page or use [] "
                    "only if no such operation is warranted, and do not drop or alter "
                    "unrelated valid operations."
                ),
            )
        )
    if issues:
        return issues

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
_MIN_ADAPTIVE_PAGE_NUM_PREDICT = 2_048
_MAX_COMPACT_UPDATE_OUTLINE_BYTES = 32 * 1_024


def _read_exact_utf8(path: Path) -> str:
    """Decode the exact on-disk UTF-8 bytes without newline translation."""

    return path.read_bytes().decode("utf-8")


def _read_optional_exact_utf8(path: Path) -> str | None:
    """Return exact UTF-8 text when ``path`` exists, otherwise ``None``."""

    return _read_exact_utf8(path) if path.exists() else None


@dataclass(frozen=True)
class _MarkdownSection:
    """One byte-complete Markdown section from an existing page."""

    start_line: int
    end_line: int
    heading: str | None
    content: str
    sha256: str


@dataclass(frozen=True)
class _CompactUpdateContext:
    """Read-only, hash-bound context for an append-only oversized update."""

    text: str
    page_id: str
    page_sha256: str
    page_bytes: int
    section_count: int
    selected_sections: tuple[_MarkdownSection, ...]


@dataclass(frozen=True)
class _PageGenerationBudget:
    num_ctx: int
    num_predict: int
    required_num_ctx: int


_MARKDOWN_SECTION_HEADING_RE = re.compile(
    r"^ {0,3}(?P<marks>#{1,2})[\t ]+(?P<title>.*?)(?:[\t ]+#+)?[\t ]*(?:\n)?$"
)
_MARKDOWN_FENCE_RE = re.compile(r"^ {0,3}(?P<marker>`{3,}|~{3,})")


def _markdown_sections(text: str) -> tuple[_MarkdownSection, ...]:
    """Split Markdown at H1/H2 boundaries outside fenced code blocks.

    Sections are a lossless partition of ``text``: concatenating their
    ``content`` fields reproduces the exact page bytes.  A pre-heading region
    (normally frontmatter) is represented as a section with ``heading=None``.
    Lower-level headings remain inside their enclosing H2 section so selection
    never detaches a subsection from its top-level semantic unit.
    """

    if not text:
        return ()
    lines = text.splitlines(keepends=True)
    heading_rows: list[tuple[int, str]] = []
    fence_char: str | None = None
    fence_width = 0
    for index, line in enumerate(lines):
        fence_match = _MARKDOWN_FENCE_RE.match(line)
        if fence_match is not None:
            marker = fence_match.group("marker")
            if fence_char is None:
                fence_char = marker[0]
                fence_width = len(marker)
            elif (
                marker[0] == fence_char
                and len(marker) >= fence_width
                and re.fullmatch(
                    r"[\t ]*(?:\r?\n)?",
                    line[fence_match.end() :],
                )
                is not None
            ):
                fence_char = None
                fence_width = 0
            continue
        if fence_char is not None:
            continue
        heading_match = _MARKDOWN_SECTION_HEADING_RE.match(line)
        if heading_match is not None:
            heading_rows.append(
                (
                    index,
                    f"{heading_match.group('marks')} "
                    f"{heading_match.group('title').strip()}",
                )
            )

    boundaries = [index for index, _heading in heading_rows]
    if not boundaries or boundaries[0] != 0:
        boundaries.insert(0, 0)
    boundaries.append(len(lines))
    headings_by_index = dict(heading_rows)
    sections: list[_MarkdownSection] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        content = "".join(lines[start:end])
        if not content:
            continue
        sections.append(
            _MarkdownSection(
                start_line=start + 1,
                end_line=end,
                heading=headings_by_index.get(start),
                content=content,
                sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
        )
    return tuple(sections)


def _semantic_relevance_text(raw_content: str) -> str:
    """Prefer semantic child record text over its transport metadata."""

    try:
        decoded = json.loads(raw_content)
    except json.JSONDecodeError:
        return raw_content
    if not isinstance(decoded, dict) or not isinstance(decoded.get("records"), list):
        return raw_content
    texts = [
        row.get("text")
        for row in decoded["records"]
        if isinstance(row, dict) and isinstance(row.get("text"), str)
    ]
    return "\n".join(texts) if texts else raw_content


def _section_relevance_score(
    section: _MarkdownSection,
    *,
    explicit_tokens: set[str],
    raw_tokens: set[str],
) -> int:
    section_tokens = set(tokenize(section.content))
    explicit_hits = len(section_tokens & explicit_tokens)
    overlap = len(section_tokens & raw_tokens)
    heading_overlap = (
        len(set(tokenize(section.heading or "")) & raw_tokens) if section.heading else 0
    )
    return explicit_hits * 1_000 + heading_overlap * 25 + overlap


def _render_compact_update_context(
    *,
    page_id: str,
    page_text: str,
    sections: tuple[_MarkdownSection, ...],
    selected: tuple[_MarkdownSection, ...],
) -> str:
    page_bytes = len(page_text.encode("utf-8"))
    page_sha256 = hashlib.sha256(page_text.encode("utf-8")).hexdigest()
    outline_rows = []
    for section in sections:
        # A padding-free base64url digest is the complete 256-bit SHA-256,
        # merely encoded in 43 rather than 64 bytes.  TSV avoids repeating
        # field labels for every section while retaining every heading and
        # exact line/byte range in the bounded manifest.
        digest_b64url = base64.urlsafe_b64encode(bytes.fromhex(section.sha256))
        digest_b64url_text = digest_b64url.rstrip(b"=").decode("ascii")
        heading_json = json.dumps(
            section.heading or "[preamble]",
            ensure_ascii=False,
        )
        outline_rows.append(
            f"{section.start_line}-{section.end_line}\t"
            f"{len(section.content.encode('utf-8'))}\t"
            f"{digest_b64url_text}\t{heading_json}"
        )
    selected_rows = [
        (
            f"--- BEGIN COMPLETE SECTION lines {section.start_line}-{section.end_line} "
            f"sha256={section.sha256} ---\n"
            f"{section.content}"
            f"--- END COMPLETE SECTION lines {section.start_line}-{section.end_line} "
            f"sha256={section.sha256} ---"
        )
        for section in selected
    ]
    return "\n".join(
        [
            f"--- Compact append-only context for [[{page_id}]] ---",
            "The stored page is not rewritten or truncated by this context view.",
            f"page_bytes: {page_bytes}",
            f"page_sha256: {page_sha256}",
            f"section_count: {len(sections)}",
            f"selected_section_count: {len(selected)}",
            "Complete deterministic H1/H2 section manifest (TSV):",
            "lines\tbytes\tsha256_b64url\theading_json",
            *outline_rows,
            "",
            "Selected complete sections (never partial byte ranges):",
            *(selected_rows or ["(none fit the fixed compact-context envelope)"]),
            "--- End compact append-only context ---\n",
        ]
    )


def _build_compact_update_context(
    op: dict,
    raw_content: str,
    *,
    max_selected_bytes: int,
    max_outline_bytes: int = _MAX_COMPACT_UPDATE_OUTLINE_BYTES,
) -> _CompactUpdateContext | None:
    """Build a read-only outline plus whole relevant Markdown sections.

    This path is valid only for append-only updates.  It never writes the
    stored page, never slices a selected section, and represents every omitted
    section by a stable line range, byte count, and SHA-256 in the outline.
    """

    if op.get("type") != "update" or max_selected_bytes < 1 or max_outline_bytes < 1:
        return None
    filename = str(op.get("filename") or "")
    page_id = filename.replace(".md", "").split("/")[-1]
    existing_path = find_page(page_id)
    if existing_path is None:
        return None
    page_text = _read_exact_utf8(existing_path)
    sections = _markdown_sections(page_text)
    if not sections or "".join(section.content for section in sections) != page_text:
        return None

    explicit_values = [
        op.get("title"),
        op.get("summary"),
        *(op.get("keywords") or []),
    ]
    explicit_tokens = set(
        tokenize(
            "\n".join(
                value.strip()
                for value in explicit_values
                if isinstance(value, str) and value.strip()
            )
        )
    )
    raw_tokens = set(tokenize(_semantic_relevance_text(raw_content)))
    candidates = [section for section in sections if section.heading is not None]
    ranked = sorted(
        (
            (
                section,
                _section_relevance_score(
                    section,
                    explicit_tokens=explicit_tokens,
                    raw_tokens=raw_tokens,
                ),
            )
            for section in candidates
        ),
        key=lambda item: (-item[1], -item[0].start_line),
    )
    outline_only = _render_compact_update_context(
        page_id=page_id,
        page_text=page_text,
        sections=sections,
        selected=(),
    )
    if len(outline_only.encode("utf-8")) > max_outline_bytes:
        # Never truncate the outline manifest. A pathological heading count
        # remains an operational defer rather than an unbound context view.
        return None

    if not ranked or ranked[0][1] <= 0:
        return None
    highest_relevance_section = ranked[0][0]
    if len(highest_relevance_section.content.encode("utf-8")) > max_selected_bytes:
        # Skipping the strongest match and filling the view with weaker small
        # sections would make the projection look admissible while hiding the
        # very context needed to avoid a duplicate or contradictory append.
        return None

    selected: list[_MarkdownSection] = []
    selected_bytes = 0
    for section, relevance_score in ranked:
        if relevance_score <= 0:
            continue
        section_bytes = len(section.content.encode("utf-8"))
        if section_bytes > max_selected_bytes - selected_bytes:
            continue
        selected.append(section)
        selected_bytes += section_bytes

    selected_tuple = tuple(sorted(selected, key=lambda section: section.start_line))
    if not selected_tuple:
        # An outline alone cannot protect against semantic duplication. Keep
        # the raw deferred until at least one complete section can be shown.
        return None
    rendered = _render_compact_update_context(
        page_id=page_id,
        page_text=page_text,
        sections=sections,
        selected=selected_tuple,
    )
    return _CompactUpdateContext(
        text=rendered,
        page_id=page_id,
        page_sha256=hashlib.sha256(page_text.encode("utf-8")).hexdigest(),
        page_bytes=len(page_text.encode("utf-8")),
        section_count=len(sections),
        selected_sections=selected_tuple,
    )


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


def _select_page_generation_budget(
    prompt: str,
    system: str | None,
    *,
    configured_num_predict: int,
    num_ctx: int,
    max_num_ctx: int,
) -> _PageGenerationBudget:
    """Select a lossless context bucket and one output budget for all turns.

    Every byte already selected for the prompt, plus every possible repair
    turn, remains in the transcript. Oversized append-only updates may first
    replace the full-page view with a separately hash-bound outline and whole
    sections; this function never performs that projection itself. When the
    configured output reservation alone pushes the resulting three-response
    envelope over the model's hard context ceiling, reduce only
    ``num_predict`` to the largest integer reservation that fits. Never adapt
    below the fixed 2K floor (or below an explicitly configured smaller
    reservation); callers defer instead.
    """

    if configured_num_predict < 1:
        raise ValueError("configured_num_predict must be positive")

    configured_required = _required_generate_context_tokens(
        prompt,
        system,
        num_predict=configured_num_predict,
    )
    selected_num_predict = configured_num_predict
    if configured_required > max_num_ctx:
        # The requirement is linear in one identical reservation for the
        # initial response and both possible repair responses.  Everything
        # else is immutable source/system/history overhead and must not be
        # truncated to make the request fit.
        fixed_context = (
            configured_required
            - _MAX_PAGE_GENERATION_RESPONSES * configured_num_predict
        )
        largest_safe_num_predict = (
            max_num_ctx - fixed_context
        ) // _MAX_PAGE_GENERATION_RESPONSES
        adaptive_floor = min(
            configured_num_predict,
            _MIN_ADAPTIVE_PAGE_NUM_PREDICT,
        )
        if largest_safe_num_predict < adaptive_floor:
            minimum_required = _required_generate_context_tokens(
                prompt,
                system,
                num_predict=adaptive_floor,
            )
            raise IngestContextCapacityError(
                "complete page-repair transcript requires context "
                f"{minimum_required} at minimum num_predict {adaptive_floor}, "
                f"exceeding configured max_num_ctx {max_num_ctx}"
            )
        selected_num_predict = min(
            configured_num_predict,
            largest_safe_num_predict,
        )

    required_num_ctx = _required_generate_context_tokens(
        prompt,
        system,
        num_predict=selected_num_predict,
    )
    selected_num_ctx = _select_ingest_context(
        required_num_ctx,
        num_ctx=num_ctx,
        max_num_ctx=max_num_ctx,
    )
    return _PageGenerationBudget(
        num_ctx=selected_num_ctx,
        num_predict=selected_num_predict,
        required_num_ctx=required_num_ctx,
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
    scope_rule = (
        "For UPDATE, return only the new append body inside the wrapper; do not "
        "repeat, summarize, or rewrite the existing stored page."
        if op_type == "update"
        else "For CREATE, return the complete new page body inside the wrapper."
    )
    prompt = f"""Your previous response was rejected by the deterministic page validator.

Validator errors:
- code: {failure_class}
  reason: {reason}

Return a complete replacement response for `{filename}`. Do not describe the
fix and do not return a patch. Start with exactly
`=== {expected_wrapper} PAGE: {filename} ===` and finish with the exact final
line `=== END PAGE ===`. {scope_rule} Preserve only facts grounded in the
original source.
"""
    if len(prompt.encode("utf-8")) > _MAX_PAGE_REPAIR_FEEDBACK_BYTES:
        raise RuntimeError(
            "ingest generation feedback_too_large: page validator feedback "
            "exceeded the fixed repair cap"
        )
    return prompt


def _page_generation_truncation_retry_prompt(
    *,
    op_type: str,
    filename: str,
    max_output_tokens: int,
) -> str:
    """Request a shorter complete replacement without trusting partial text.

    An Ollama ``done_reason=length`` completion has no attested terminal
    boundary, so it must never become assistant history.  The retry sees the
    original grounded prompt again plus this bounded instruction and produces
    a complete replacement from scratch.
    """

    expected_wrapper = "NEW" if op_type == "create" else "UPDATE"
    scope_rule = (
        "For UPDATE, return only the new append body inside the wrapper; do not "
        "repeat, summarize, or rewrite the existing stored page."
        if op_type == "update"
        else "For CREATE, return the complete new page body inside the wrapper."
    )
    prompt = f"""Your previous response reached the transport output limit and was discarded.

Return a shorter, complete replacement response for `{filename}` from the
original source. Use at most {max_output_tokens} output tokens. Do not continue
the discarded response, describe the fix, or return a patch. Start with exactly
`=== {expected_wrapper} PAGE: {filename} ===` and finish with the exact final
line `=== END PAGE ===`. {scope_rule} Preserve only facts grounded in the
original source.
"""
    if len(prompt.encode("utf-8")) > _MAX_PAGE_REPAIR_FEEDBACK_BYTES:
        raise RuntimeError(
            "ingest generation feedback_too_large: output truncation feedback "
            "exceeded the fixed repair cap"
        )
    return prompt


def _build_page_generation_prompt(
    *,
    context: str,
    raw_content: str,
    op_type: str,
    filename: str,
    title: str,
    summary: str,
    feedback_block: str,
    current_date: str,
) -> str:
    """Render the exact prompt shared by full and compact update contexts."""

    return f"""{context}

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


def _generate_one(
    op: dict,
    raw_content: str,
    *,
    raw_keywords: list[str] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    frontier_feedback: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict | None:
    from llm_wiki_mcp.ingest_generation import generate_one

    return generate_one(
        op,
        raw_content,
        raw_keywords=raw_keywords,
        progress_callback=progress_callback,
        frontier_feedback=frontier_feedback,
        diagnostics=diagnostics,
    )


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


def _reserved_system_page_collision_keys() -> frozenset[str]:
    """Return every page_id normal ingest must never mutate.

    The three correction-owned system pages stay reserved even if one is
    temporarily absent on disk.  Any additional installed system page is also
    protected so the triage validator and the final apply guard cannot drift.
    Resolve ``wiki.SYSTEM_DIR`` dynamically because tests and isolated runtimes
    replace it after this module is imported.
    """

    from llm_wiki_mcp import wiki as _wiki
    from llm_wiki_mcp.page_mutation import CORRECTABLE_SYSTEM_PAGE_IDS

    page_ids = set(CORRECTABLE_SYSTEM_PAGE_IDS)
    try:
        page_ids.update(
            path.stem for path in _wiki.SYSTEM_DIR.rglob("*.md") if path.is_file()
        )
    except OSError:
        # Core system ids remain fail-closed when an optional directory scan is
        # unavailable.  The apply layer's index load still fails independently
        # if it cannot prove the complete corpus state.
        pass
    return frozenset(_normalize_for_collision(page_id) for page_id in page_ids)


def _target_page_collision_key(filename: object) -> str | None:
    """Return the apply-equivalent page_id key for one safe target filename."""

    if not isinstance(filename, str):
        return None
    try:
        page_id = _safe_resolve_page_path(filename).stem
    except IngestApplyError:
        return None
    return _normalize_for_collision(page_id)


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
        # ``Path.rglob('foo.md')`` can return a query-spelled path on a
        # case-insensitive filesystem even when the directory entry is
        # actually ``Foo.md``.  Preserve the filesystem spelling because the
        # page_id is a durable identity surfaced in ingest results and logs.
        try:
            for candidate in direct.parent.iterdir():
                if candidate.is_file() and candidate.samefile(direct):
                    return candidate
        except OSError:
            pass
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
    from llm_wiki_mcp.ingest_prepare import prepare_operations

    return prepare_operations(operations, read_only=read_only)


def _apply_prepared_operations(
    planned: list[PreparedIngestOperation],
    *,
    link_totals: dict[str, int] | None = None,
    recovery_only: bool = False,
) -> tuple[list[str], list[str]]:
    from llm_wiki_mcp.ingest_apply import apply_prepared_operations

    return apply_prepared_operations(
        planned,
        link_totals=link_totals,
        recovery_only=recovery_only,
    )


def _apply_operations(operations: list[dict]) -> tuple[list[str], list[str]]:
    """Legacy/internal primitive: prepare and apply an already-approved plan.

    Production ingest must use :func:`_review_and_apply_ingest_operations`.
    Keeping this small wrapper preserves focused apply tests without creating a
    second semantic write path in the running ingest pipeline.
    """

    planned, totals = _prepare_operations(operations)
    return _apply_prepared_operations(planned, link_totals=totals)


@dataclass(frozen=True)
class _IngestReviewShardContinuation:
    """Strict pre-triage resume state for a partially approved shard plan."""

    proposal: dict[str, Any]
    planned: tuple[PreparedIngestOperation, ...]
    plan: _IngestReviewShardPlan
    approved_shards: int


def _ingest_review_router_config() -> Any:
    """Resolve the same adopted router configuration used by live review."""

    from llm_wiki_mcp.decision_router import resolve_router_policy
    from llm_wiki_mcp.runtime_config import load_decision_router_config

    config = load_decision_router_config()
    if not config.adoption_artifact.strip():
        return config
    resolution = resolve_router_policy(config)
    if resolution.error is not None:
        raise IngestReviewShardCapacityError(
            "local_decision_artifact_invalid",
            f"review router policy is unavailable: {resolution.error}",
        )
    return resolution.config


def _measure_ingest_review_request(
    proposal: dict[str, Any],
    *,
    original_operation_indices: tuple[int, ...],
    config: Any,
) -> _IngestReviewShard:
    """Compatibility seam for the extracted deterministic request planner."""

    return _measure_ingest_review_request_core(
        proposal,
        original_operation_indices=original_operation_indices,
        config=config,
    )


def _validate_ingest_shard_source_rows(
    proposal: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return _validate_ingest_shard_source_rows_core(proposal)


def _build_ingest_review_shard_proposal(
    proposal: dict[str, Any],
    *,
    group: tuple[int, ...],
    groups: tuple[tuple[int, ...], ...],
    shard_index: int,
    full_proposal_sha256: str,
) -> dict[str, Any]:
    return _build_ingest_review_shard_proposal_core(
        proposal,
        group=group,
        groups=groups,
        shard_index=shard_index,
        full_proposal_sha256=full_proposal_sha256,
    )


def _build_ingest_review_shard_plan(
    proposal: dict[str, Any],
    *,
    config: Any | None = None,
    force_review_unit: bool = False,
) -> _IngestReviewShardPlan | None:
    """Resolve the live config seam, then run the extracted pure planner."""

    return _build_ingest_review_shard_plan_core(
        proposal,
        config=config or _ingest_review_router_config(),
        force_review_unit=force_review_unit,
    )


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
    return _ingest_artifact_paths_core(PAGES_DIR, source_key)


def _ingest_review_continuation_marker_path(source_key: str) -> Path:
    return _continuation_marker_path_core(PAGES_DIR, source_key)


def _ingest_review_repair_transition_path(source_key: str) -> Path:
    return _repair_transition_path_core(PAGES_DIR, source_key)


def _seal_ingest_review_repair_transition(
    *,
    source_key: str,
    previous_full_proposal_sha256: str,
    repaired_operations_sha256: str,
) -> str | None:
    return _seal_ingest_review_repair_transition_core(
        _ingest_review_repair_transition_path(source_key),
        source_key=source_key,
        previous_full_proposal_sha256=previous_full_proposal_sha256,
        repaired_operations_sha256=repaired_operations_sha256,
    )


def _persist_ingest_review_continuation_marker(
    *,
    source_key: str,
    plan: _IngestReviewShardPlan,
    reason: str,
    previous_full_proposal_sha256: str,
    previous_authority: dict[str, Any] | str,
    current_authority: dict[str, Any],
) -> str | None:
    return _persist_ingest_review_continuation_marker_core(
        _ingest_review_continuation_marker_path(source_key),
        source_key=source_key,
        plan=plan,
        reason=reason,
        previous_full_proposal_sha256=previous_full_proposal_sha256,
        previous_authority=previous_authority,
        current_authority=current_authority,
    )


def _load_ingest_review_continuation_marker(
    *,
    source_key: str,
    plan: _IngestReviewShardPlan,
    authority: dict[str, Any],
    allow_stale_identity: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    return _load_ingest_review_continuation_marker_core(
        _ingest_review_continuation_marker_path(source_key),
        source_key=source_key,
        plan=plan,
        authority=authority,
        allow_stale_identity=allow_stale_identity,
    )


def _consume_ingest_review_continuation_marker(
    source_key: str,
    marker: dict[str, Any],
) -> str | None:
    return _consume_ingest_review_continuation_marker_core(
        _ingest_review_continuation_marker_path(source_key), marker
    )


def _ingest_review_stall_path(source_key: str) -> Path:
    return _review_stall_path_core(PAGES_DIR, source_key)


def _persist_ingest_review_stall(
    *,
    source_key: str,
    plan: _IngestReviewShardPlan,
    authority: dict[str, Any],
    approved_indices: tuple[int, ...],
) -> str | None:
    return _persist_ingest_review_stall_core(
        _ingest_review_stall_path(source_key),
        source_key=source_key,
        plan=plan,
        authority=authority,
        approved_indices=approved_indices,
    )


def _matching_ingest_review_stall_error(
    *,
    source_key: str,
    plan: _IngestReviewShardPlan,
    authority: dict[str, Any],
    approved_indices: tuple[int, ...],
) -> str | None:
    return _matching_ingest_review_stall_error_core(
        _ingest_review_stall_path(source_key),
        source_key=source_key,
        plan=plan,
        authority=authority,
        approved_indices=approved_indices,
    )


def _write_ingest_artifact(path: Path, payload: dict[str, Any]) -> None:
    _write_ingest_artifact_core(path, payload)


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
        return _read_optional_exact_utf8(item.path)

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
            current = _read_optional_exact_utf8(item.path)
        except (OSError, UnicodeDecodeError):
            return False
        if current not in {item.previous_text, item.new_body}:
            return False
    return True


def _prepared_plan_targets_reserved_system_page(
    planned: list[PreparedIngestOperation],
) -> bool:
    """Reject legacy artifacts that predate the reserved-target validator."""

    reserved = _reserved_system_page_collision_keys()
    return any(_normalize_for_collision(item.page_id) in reserved for item in planned)


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
    return _load_ingest_proposal_core(
        path,
        source_key=source_key,
        raw_content=raw_content,
        decode_prepared=_prepared_from_review_payload,
        targets_reserved_system_page=_prepared_plan_targets_reserved_system_page,
        plan_is_recoverable=_prepared_plan_is_recoverable,
    )


def _load_ingest_review_artifact(
    path: Path,
    *,
    source_key: str,
    proposal_sha256: str,
    require_integrity: bool = False,
) -> dict[str, Any] | None:
    return _load_ingest_review_artifact_core(
        path,
        source_key=source_key,
        proposal_sha256=proposal_sha256,
        authority_shape_error=_ingest_review_authority_shape_error,
        authority_error=_ingest_review_authority_error,
        require_integrity=require_integrity,
    )


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

    return _load_ingest_review_core(
        path,
        source_key=source_key,
        proposal_sha256=proposal_sha256,
        authority_shape_error=_ingest_review_authority_shape_error,
        authority_error=_ingest_review_authority_error,
    )


def _sealed_ingest_review_artifact(
    *,
    source_key: str,
    proposal_sha256: str,
    review: dict[str, Any],
    authority: dict[str, Any],
    integrity: bool = False,
) -> dict[str, Any]:
    """Build the common authority-sealed terminal ingest artifact."""

    return _sealed_ingest_review_artifact_core(
        source_key=source_key,
        proposal_sha256=proposal_sha256,
        review=review,
        authority=authority,
        integrity=integrity,
    )


def _write_and_readback_ingest_review_artifact(
    path: Path,
    *,
    source_key: str,
    proposal_sha256: str,
    review: dict[str, Any],
    authority: dict[str, Any],
    integrity: bool = False,
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
            integrity=integrity,
        )
        _write_ingest_artifact(path, sealed)
    except (OSError, ValueError) as exc:
        return None, f"frontier review artifact write failed: {exc}"
    readback = _load_ingest_review_artifact(
        path,
        source_key=source_key,
        proposal_sha256=proposal_sha256,
        require_integrity=integrity,
    )
    if readback != sealed:
        return None, "frontier review artifact readback verification failed"
    return readback, None


def _current_ingest_review_authority(
    *, reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve the exact enabled local authority allowed to affect ingest."""

    return _current_ingest_review_authority_core(injected_reviewer=reviewer is not None)


def _ingest_review_authority_error(
    review: dict[str, Any],
    authority: dict[str, Any],
) -> str | None:
    return _ingest_review_authority_error_core(review, authority)


def _ingest_review_shard_proof_error(
    review: dict[str, Any],
    authority: dict[str, Any],
) -> str | None:
    return _ingest_review_shard_proof_error_core(review, authority)


def _ingest_review_authority_shape_error(authority: dict[str, Any]) -> str | None:
    """Reject authority envelopes that cannot identify a review epoch."""

    return _ingest_review_authority_shape_error_core(authority)


def _prepared_plan_is_fully_applied(
    planned: list[PreparedIngestOperation],
) -> bool:
    """Prove that recovery can finish without installing any semantic bytes."""

    if not planned:
        return False
    for item in planned:
        try:
            current = _read_optional_exact_utf8(item.path)
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
    from llm_wiki_mcp.ingest_recovery_runtime import (
        load_strict_ingest_proposal_for_recovery,
    )

    return load_strict_ingest_proposal_for_recovery(
        path,
        source_key=source_key,
        raw_content=raw_content,
        raw_keywords=raw_keywords,
    )


def _load_pretriage_terminal_recovery(
    raw_content: str,
    raw_keywords: list[str] | None,
    *,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> dict[str, Any] | None:
    from llm_wiki_mcp.ingest_recovery_runtime import (
        load_pretriage_terminal_recovery,
    )

    return load_pretriage_terminal_recovery(
        raw_content,
        raw_keywords,
        reviewer=reviewer,
    )


def _normalize_ingest_frontier_review(
    value: object,
    *,
    proposal: dict[str, Any],
) -> dict[str, Any]:
    return _normalize_ingest_frontier_review_core(value, proposal=proposal)


def _run_ingest_frontier_review(
    proposal: dict[str, Any],
    *,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from llm_wiki_mcp.decision_lane_prompts import (
        build_ingest_reconciliation_prompt,
    )

    return _run_ingest_frontier_review_core(
        proposal,
        reviewer=reviewer,
        repo_root=Path(__file__).resolve().parents[2],
        decision_schema=INGEST_FRONTIER_DECISION_SCHEMA,
        prompt_builder=build_ingest_reconciliation_prompt,
    )


def _ingest_review_shard_manifest_path(plan: _IngestReviewShardPlan) -> Path:
    return _ingest_review_shard_manifest_path_core(
        _ingest_artifact_paths("unused")[0].parent, plan
    )


def _ingest_review_shard_review_identity(
    plan: _IngestReviewShardPlan,
    *,
    shard_index: int,
    shard: _IngestReviewShard,
) -> tuple[str, Path]:
    return _ingest_review_shard_review_identity_core(
        _ingest_artifact_paths("unused")[0].parent,
        plan,
        shard_index=shard_index,
        shard=shard,
    )


def _ingest_review_shard_failure(
    failure_class: str,
    summary: str,
) -> dict[str, Any]:
    return _ingest_review_shard_failure_core(failure_class, summary)


def _persist_ingest_review_shard_manifest(
    plan: _IngestReviewShardPlan,
    *,
    source_key: str,
) -> str | None:
    return _persist_ingest_review_shard_manifest_core(
        _ingest_review_shard_manifest_path(plan), plan, source_key=source_key
    )


def _ingest_review_shard_manifest_artifact_payload(
    plan: _IngestReviewShardPlan,
    *,
    source_key: str,
) -> dict[str, Any]:
    return _ingest_review_shard_manifest_artifact_payload_core(
        plan, source_key=source_key
    )


def _stored_ingest_review_shard_manifest_error(
    plan: _IngestReviewShardPlan,
    *,
    source_key: str,
) -> str | None:
    return _stored_ingest_review_shard_manifest_error_core(
        _ingest_review_shard_manifest_path(plan), plan, source_key=source_key
    )


def _ingest_review_shard_aggregate(
    plan: _IngestReviewShardPlan,
    shard_reviews: list[dict[str, Any]],
) -> dict[str, Any]:
    return _ingest_review_shard_aggregate_core(plan, shard_reviews)


def _run_ingest_sharded_review(
    plan: _IngestReviewShardPlan,
    *,
    source_key: str,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None,
    authority: dict[str, Any],
    frontier_budget: "_FrontierCallBudget | None" = None,
) -> dict[str, Any]:
    from llm_wiki_mcp.page_mutation import decision_authority_lock

    return _run_ingest_sharded_review_core(
        plan,
        source_key=source_key,
        reviewer=reviewer,
        authority=authority,
        frontier_budget=frontier_budget,
        deps=IngestShardedReviewDeps(
            persist_manifest=_persist_ingest_review_shard_manifest,
            inspect_plan_state=_inspect_ingest_review_shard_plan_state,
            review_identity=_ingest_review_shard_review_identity,
            run_frontier_review=_run_ingest_frontier_review,
            normalize_review=_normalize_ingest_frontier_review,
            current_authority=_current_ingest_review_authority,
            authority_error=_ingest_review_authority_error,
            write_and_readback=_write_and_readback_ingest_review_artifact,
            authority_lock=decision_authority_lock,
        ),
    )


def _ingest_sharded_review_reuse_error(
    review: dict[str, Any],
    proposal: dict[str, Any],
    authority: dict[str, Any],
) -> str | None:
    """Recompute with current limits before an ordinary unapplied reuse."""

    if "review_shard_proof" not in review:
        return None
    try:
        plan = _build_ingest_review_shard_plan(
            proposal,
            force_review_unit=True,
        )
    except (IngestReviewShardCapacityError, ValueError) as exc:
        return f"ingest shard manifest recomputation failed: {exc}"
    assert plan is not None
    return _ingest_sharded_review_plan_artifact_error(
        review,
        proposal,
        authority,
        plan,
    )


def _ingest_sharded_review_plan_artifact_error(
    review: dict[str, Any],
    proposal: dict[str, Any],
    authority: dict[str, Any],
    plan: _IngestReviewShardPlan,
) -> str | None:
    """Verify an exact plan and every already-durable shard without writes."""

    proof = review.get("review_shard_proof")
    if (
        not isinstance(proof, dict)
        or proof.get("manifest") != plan.manifest
        or proof.get("manifest_sha256") != plan.manifest_sha256
        or proof.get("full_proposal_sha256") != plan.full_proposal_sha256
    ):
        return "ingest shard terminal review does not match exact recomputation"
    shard_state = _inspect_ingest_review_shard_plan_state(
        plan,
        source_key=str(proposal.get("source_key") or ""),
        authority=authority,
    )
    if shard_state.invalid_reason is not None:
        return shard_state.invalid_reason
    if shard_state.approved_shards != len(plan.shards):
        return "ingest shard terminal review authority changed"
    proof_rows = proof.get("shard_reviews")
    if not isinstance(proof_rows, list) or len(proof_rows) != len(plan.shards):
        return "ingest shard terminal review count changed"
    for shard_index, (shard, proof_row, durable_review) in enumerate(
        zip(plan.shards, proof_rows, shard_state.reviews, strict=True)
    ):
        if not isinstance(proof_row, dict):
            return f"ingest shard {shard_index} proof is invalid"
        if (
            durable_review != proof_row.get("review")
            or proof_row.get("proposal_sha256") != shard.proposal_sha256
            or proof_row.get("shard_index") != shard_index
        ):
            return f"ingest shard {shard_index} durable proof changed"
    return None


def _historical_ingest_sharded_review_recovery_error(
    review: dict[str, Any],
    proposal: dict[str, Any],
    authority: dict[str, Any],
) -> str | None:
    """Verify a fully-applied shard proof under its sealed historical limits.

    Recovery is allowed to acknowledge exact postimages after live router limits
    change, but it must never reinterpret the old partition under those new
    limits.  The embedded limits are authority-sealed by the aggregate review;
    recomputing with them proves the same shard proposals and artifact names.
    """

    if "review_shard_proof" not in review:
        return None
    if proof_error := _ingest_review_authority_error(review, authority):
        return proof_error
    proof = review.get("review_shard_proof")
    manifest = proof.get("manifest") if isinstance(proof, dict) else None
    if proof.get("full_proposal_sha256") != _canonical_json_sha256(proposal):
        return "ingest shard historical proof does not bind the full proposal"
    try:
        plan = _ingest_review_shard_plan_from_sealed_manifest(
            proposal,
            manifest,
        )
    except (IngestReviewShardCapacityError, TypeError, ValueError) as exc:
        return f"ingest shard historical manifest recomputation failed: {exc}"
    return _ingest_sharded_review_plan_artifact_error(
        review,
        proposal,
        authority,
        plan,
    )


def _ingest_review_shard_plan_from_sealed_manifest(
    proposal: dict[str, Any],
    manifest: object,
) -> _IngestReviewShardPlan:
    """Rebuild one shard plan from the limits sealed into its manifest."""

    limits = manifest.get("review_limits") if isinstance(manifest, dict) else None
    if (
        not isinstance(limits, dict)
        or set(limits) != _INGEST_REVIEW_LIMIT_FIELDS
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in limits.values()
        )
        or limits.get("min_num_ctx", 0) > limits.get("num_ctx", 0)
    ):
        raise ValueError("ingest shard historical review limits are invalid")

    from llm_wiki_mcp.runtime_config import DecisionRouterConfig

    historical_config = DecisionRouterConfig(
        **limits,
        adoption_artifact="",
    )
    plan = _build_ingest_review_shard_plan(
        proposal,
        config=historical_config,
        force_review_unit=True,
    )
    assert plan is not None
    if plan.manifest != manifest:
        raise ValueError(
            "ingest shard historical manifest is not an exact recomputation"
        )
    return plan


def _inspect_ingest_review_shard_plan_state(
    plan: _IngestReviewShardPlan,
    *,
    source_key: str,
    authority: dict[str, Any],
) -> _IngestReviewShardPlanState:
    from llm_wiki_mcp.ingest_recovery_runtime import (
        inspect_ingest_review_shard_plan_state,
    )

    return inspect_ingest_review_shard_plan_state(
        plan,
        source_key=source_key,
        authority=authority,
    )


def _load_pretriage_ingest_shard_continuation(
    raw_content: str,
    raw_keywords: list[str] | None,
    *,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> _IngestReviewShardContinuation | None:
    from llm_wiki_mcp.ingest_recovery_runtime import (
        load_pretriage_ingest_shard_continuation,
    )

    return load_pretriage_ingest_shard_continuation(
        raw_content,
        raw_keywords,
        reviewer=reviewer,
    )


def _tombstone_current_ingest_review_plan(
    raw_content: str,
    raw_keywords: list[str] | None,
    *,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> str | None:
    """Durably stop a normally-returned continuation that made no terminal effect."""

    source_key = _ingest_source_key(raw_content, raw_keywords)
    proposal_path, review_path = _ingest_artifact_paths(source_key)
    if not proposal_path.exists() or review_path.exists():
        return None
    proposal, _planned = _load_strict_ingest_proposal_for_recovery(
        proposal_path,
        source_key=source_key,
        raw_content=raw_content,
        raw_keywords=raw_keywords,
    )
    authority, authority_error = _current_ingest_review_authority(reviewer=reviewer)
    if authority_error is not None or not isinstance(authority, dict):
        return authority_error or "ingest review stall authority is missing"
    plan = _build_ingest_review_shard_plan(
        proposal,
        force_review_unit=True,
    )
    assert plan is not None
    if not _ingest_review_shard_manifest_path(plan).exists():
        return "ingest review stall manifest is missing"
    shard_state = _inspect_ingest_review_shard_plan_state(
        plan,
        source_key=source_key,
        authority=authority,
    )
    if shard_state.invalid_reason is not None:
        return shard_state.invalid_reason
    return _persist_ingest_review_stall(
        source_key=source_key,
        plan=plan,
        authority=authority,
        approved_indices=shard_state.current_approved_indices,
    )


def _has_sharded_ingest_review_artifact_family(
    review_path: Path,
    *,
    proposal: dict[str, Any],
    source_key: str,
) -> bool:
    """Detect a sharded terminal family even when its aggregate was corrupted."""

    try:
        raw_review_artifact = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raw_review_artifact = None
    raw_review = (
        raw_review_artifact.get("review")
        if isinstance(raw_review_artifact, dict)
        else None
    )
    if isinstance(raw_review, dict) and "review_shard_proof" in raw_review:
        return True
    try:
        expected_shard_plan = _build_ingest_review_shard_plan(proposal)
    except (IngestReviewShardCapacityError, ValueError):
        expected_shard_plan = None
    if (
        expected_shard_plan is not None
        and _ingest_review_shard_manifest_path(expected_shard_plan).exists()
    ):
        return True
    shard_root = _ingest_artifact_paths(source_key)[0].parent
    for candidate_path in shard_root.glob("review-shard-manifest-*.json"):
        try:
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(candidate, dict) and candidate.get("source_key") == source_key:
            return True
    return False


def _invalid_sharded_review_result(
    *,
    source_key: str,
    proposal: dict[str, Any],
    recovered_artifact: bool,
) -> dict[str, Any]:
    summary = (
        "current sharded terminal review artifact is corrupt; "
        "refusing replacement or page mutation"
    )
    audit = proposal.get("audit_decision")
    return {
        "status": "needs_retry",
        "source_key": source_key,
        "proposal_sha256": _canonical_json_sha256(proposal),
        "review": _ingest_review_shard_failure(
            "ingest_review_shard_reuse_invalid", summary
        ),
        "summary": summary,
        "recovered_artifact": recovered_artifact,
        "reused_review": False,
        "created": [],
        "updated": [],
        "audit": dict(audit) if isinstance(audit, dict) else {},
    }


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
    shard_continuation: _IngestReviewShardContinuation | None = None,
    allow_empty_shard_continuation: bool = False,
    continuation_reseed_from_sha256: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    from llm_wiki_mcp.ingest_review_apply import (
        review_and_apply_ingest_operations,
    )

    return review_and_apply_ingest_operations(
        operations,
        raw_content=raw_content,
        raw_keywords=raw_keywords,
        source_raw=source_raw,
        triage_plan=triage_plan,
        failed_operation_specs=failed_operation_specs,
        local_disposition=local_disposition,
        reviewer=reviewer,
        force_frontier_review=force_frontier_review,
        frontier_budget=frontier_budget,
        shard_continuation=shard_continuation,
        allow_empty_shard_continuation=allow_empty_shard_continuation,
        continuation_reseed_from_sha256=continuation_reseed_from_sha256,
        dry_run=dry_run,
    )


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


def _verify_changed_pages_read_back(
    page_ids: list[str], *, top_n: int = 10
) -> dict:
    from llm_wiki_mcp.ingest_readback import verify_changed_pages_read_back

    return verify_changed_pages_read_back(page_ids, top_n=top_n)


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
    from llm_wiki_mcp.ingest_recovery_runtime import (
        complete_pretriage_terminal_recovery,
    )

    return complete_pretriage_terminal_recovery(
        recovery,
        raw_content=raw_content,
        raw_keywords=raw_keywords,
        source_raw=source_raw,
        job_id=job_id,
        on_complete=on_complete,
        reviewer=reviewer,
    )


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


def _record_ingest_shard_continuation(
    *,
    job_id: str,
    source_raw: str | None,
    result: dict[str, Any],
) -> None:
    """Finish this bounded run successfully while leaving its raw pending."""

    continuation = result.get("shard_continuation")
    if (
        result.get("status") != "shard_continuation_pending"
        or not isinstance(continuation, dict)
        or continuation.get("schema_version") != INGEST_REVIEW_SHARD_SCHEMA_VERSION
        or continuation.get("kind") != "ingest_review_shard_continuation"
        or not isinstance(continuation.get("approved_shards"), int)
        or isinstance(continuation.get("approved_shards"), bool)
        or not isinstance(continuation.get("total_shards"), int)
        or continuation.get("approved_shards", -1) < 0
        or continuation.get("approved_shards", 0) >= continuation.get("total_shards", 0)
        or continuation.get("remaining_shards")
        != continuation.get("total_shards", 0) - continuation.get("approved_shards", 0)
    ):
        raise IngestApplyError("ingest shard continuation result is invalid")
    job_store.update(
        job_id,
        status=JobStatus.COMPLETED,
        stage="authorization-continuation",
        completed_at=_now(),
        pages_created=[],
        pages_updated=[],
        result={
            "ingest_continuation": dict(continuation),
            "local_consensus": {
                "status": "shard_continuation_pending",
                "proposal_sha256": result.get("proposal_sha256"),
                "source_key": result.get("source_key"),
            },
            "audit": result.get("audit"),
        },
        error=None,
    )
    runtime_status.safe_write_status(
        state="running",
        stage="authorization-continuation",
        current_job_id=job_id,
        current_raw=source_raw,
        current_op=None,
        llm=None,
    )
    runtime_status.safe_append_event(
        "info",
        "ingest | bounded shard review continuation saved",
        source="ingest",
        raw_file=source_raw,
        job_id=job_id,
        approved_shards=continuation["approved_shards"],
        total_shards=continuation["total_shards"],
    )
    _safe_log(
        "ingest | shard review continuation: "
        f"{continuation['approved_shards']}/{continuation['total_shards']} approved; "
        "raw remains pending"
    )


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


def _structured_frontier_authority_sha256(result: dict[str, Any]) -> str | None:
    """Return the adopted router artifact identity carried by a review."""

    review = result.get("review")
    for candidate in (result, review):
        if not isinstance(candidate, dict):
            continue
        decision_policy = candidate.get("decision_policy")
        if not isinstance(decision_policy, dict):
            continue
        router_policy = decision_policy.get("router_policy")
        if not isinstance(router_policy, dict):
            continue
        artifact_sha256 = router_policy.get("artifact_sha256")
        if isinstance(artifact_sha256, str) and re.fullmatch(
            r"[0-9a-f]{64}", artifact_sha256
        ):
            return artifact_sha256
    return None


def _raise_bounded_local_consensus_nonconvergence(
    *,
    initial_authority: dict[str, Any] | None,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None,
    semantic_detail: str,
    legacy_error: str,
) -> None:
    """Turn production semantic exhaustion into one authority-bound hold.

    Dependency-injected reviewer seams retain their historical error contract.
    Production, however, must not route a bounded semantic disagreement through
    the legacy nonconvergence failure class: that class immediately quarantines
    the raw and can tear apart a multi-child projection bundle.  Re-resolve the
    adopted ingest authority at the last possible boundary and emit a semantic
    defer only while it is the exact epoch observed before triage.  Missing,
    malformed, or replaced authority is operational evidence instead.
    """

    if (
        not isinstance(initial_authority, dict)
        or initial_authority.get("source") != decision_authority.ADOPTED_LOCAL_SOURCE
    ):
        raise IngestApplyError(legacy_error)

    current_authority, current_error = _current_ingest_review_authority(
        reviewer=reviewer
    )
    authority_problem = current_error
    if current_authority is None and authority_problem is None:
        authority_problem = "decision_authority_missing: current authority is missing"
    if current_authority is not None and authority_problem is None:
        shape_error = _ingest_review_authority_shape_error(current_authority)
        if shape_error is not None:
            authority_problem = f"decision_authority_invalid: {shape_error}"
    if current_authority is not None and authority_problem is None:
        compare_error = decision_authority.compare_semantic_authority(
            initial_authority,
            current_authority,
            lane="ingest_reconciliation",
        )
        if compare_error is not None:
            authority_problem = f"decision_authority_changed: {compare_error}"
    if authority_problem is not None:
        raise IngestApplyError(
            "local consensus authority unavailable: " + authority_problem
        )

    assert current_authority is not None
    router = current_authority.get("router")
    authority_sha256 = (
        router.get("artifact_sha256") if isinstance(router, dict) else None
    )
    if (
        not isinstance(authority_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", authority_sha256) is None
    ):
        raise IngestApplyError(
            "local consensus authority unavailable: "
            "decision_authority_invalid: adopted artifact hash is invalid"
        )
    raise IngestApplyError(
        "local consensus semantic no quorum "
        f"[authority_sha256={authority_sha256}]: {semantic_detail}"
    )


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


def _review_exact_ingest_repair_once(
    operations: list[dict],
    result: dict[str, Any],
    *,
    raw_content: str,
    raw_keywords: list[str] | None,
    source_raw: str | None,
    reviewer: Callable[[dict[str, Any]], dict[str, Any]] | None,
    frontier_budget: "_FrontierCallBudget | None",
    triage_plan: list[dict] | None = None,
    failed_operation_specs: list[dict] | None = None,
    local_disposition: str = "operations_available",
) -> tuple[list[dict], dict[str, Any]]:
    """Apply one exact quorum repair and re-review its complete postimage.

    This transition is shared by fresh generation and pre-triage shard
    continuation.  In both cases the repair arrays name complete host-bound
    postimages; they never authorize a page write until the rebuilt full
    proposal receives a fresh terminal review.
    """

    repaired_operations, replaced_files = _apply_frontier_replacement_operations(
        operations,
        result,
    )
    if not replaced_files:
        return operations, result
    _safe_log(
        "ingest | exact local quorum repair replaced: " + ", ".join(replaced_files)
    )
    previous_proposal_sha256 = str(result.get("proposal_sha256") or "")
    repair_error = _seal_ingest_review_repair_transition(
        source_key=_ingest_source_key(raw_content, raw_keywords),
        previous_full_proposal_sha256=previous_proposal_sha256,
        repaired_operations_sha256=_canonical_json_sha256(repaired_operations),
    )
    if repair_error is not None:
        blocked = {
            **result,
            "status": "needs_retry",
            "summary": repair_error,
            "review": _ingest_review_shard_failure(
                "ingest_review_repair_limit_exceeded",
                repair_error,
            ),
            "created": [],
            "updated": [],
        }
        return operations, blocked
    repaired_result = _review_and_apply_ingest_operations(
        repaired_operations,
        raw_content=raw_content,
        raw_keywords=raw_keywords,
        source_raw=source_raw,
        triage_plan=triage_plan,
        failed_operation_specs=failed_operation_specs,
        local_disposition=local_disposition,
        reviewer=reviewer,
        force_frontier_review=True,
        frontier_budget=frontier_budget,
        allow_empty_shard_continuation=True,
        continuation_reseed_from_sha256=previous_proposal_sha256,
    )
    return repaired_operations, repaired_result


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


def _normalize_ingest_source_metadata(
    metadata: object,
) -> tuple[list[str] | None, str | None]:
    """Return the two supported raw side channels without fabricating values."""

    if not isinstance(metadata, dict):
        return None, None
    keywords = metadata.get("raw_keywords")
    raw_keywords = (
        list(keywords)
        if isinstance(keywords, list)
        and all(isinstance(value, str) for value in keywords)
        else None
    )
    source = metadata.get("source_raw")
    source_raw = source if isinstance(source, str) else None
    return raw_keywords, source_raw


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
    raw_keywords_for_ops, source_raw = _normalize_ingest_source_metadata(metadata)
    initial_review_authority: dict[str, Any] | None = None

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
        initial_review_authority = review_authority

        shard_continuation = _load_pretriage_ingest_shard_continuation(
            content,
            raw_keywords_for_ops,
            reviewer=frontier_reviewer,
        )
        if shard_continuation is not None:
            runtime_status.safe_write_status(
                state="running",
                stage="authorization-continuation",
                current_job_id=job_id,
                current_raw=source_raw,
                current_op=None,
                llm=None,
            )
            continuation_budget = _FrontierCallBudget()
            stored_operations = shard_continuation.proposal.get(
                "local_generated_operations"
            )
            if not isinstance(stored_operations, list) or not all(
                isinstance(operation, dict) for operation in stored_operations
            ):
                raise IngestApplyError(
                    "pre-triage shard continuation operations are invalid"
                )
            stored_triage = shard_continuation.proposal.get("triage_plan")
            stored_failed_specs = shard_continuation.proposal.get(
                "failed_operation_specs"
            )
            stored_disposition = shard_continuation.proposal.get("local_disposition")
            if (
                not isinstance(stored_triage, list)
                or not all(isinstance(item, dict) for item in stored_triage)
                or not isinstance(stored_failed_specs, list)
                or not all(isinstance(item, dict) for item in stored_failed_specs)
                or not isinstance(stored_disposition, str)
                or not stored_disposition
            ):
                raise IngestApplyError(
                    "pre-triage shard continuation proposal context is invalid"
                )
            continuation_result = _review_and_apply_ingest_operations(
                list(stored_operations),
                raw_content=content,
                raw_keywords=raw_keywords_for_ops,
                source_raw=source_raw,
                reviewer=frontier_reviewer,
                frontier_budget=continuation_budget,
                shard_continuation=shard_continuation,
            )
            stored_operations, continuation_result = _review_exact_ingest_repair_once(
                list(stored_operations),
                continuation_result,
                raw_content=content,
                raw_keywords=raw_keywords_for_ops,
                source_raw=source_raw,
                reviewer=frontier_reviewer,
                frontier_budget=continuation_budget,
                triage_plan=list(stored_triage),
                failed_operation_specs=list(stored_failed_specs),
                local_disposition=stored_disposition,
            )
            continuation_status = str(
                continuation_result.get("status") or "needs_retry"
            )
            if continuation_status == "shard_continuation_pending":
                _record_ingest_shard_continuation(
                    job_id=job_id,
                    source_raw=source_raw,
                    result=continuation_result,
                )
                failed = False
                return
            if continuation_status != "apply_available":
                stall_error = _tombstone_current_ingest_review_plan(
                    content,
                    raw_keywords_for_ops,
                    reviewer=frontier_reviewer,
                )
                if stall_error is not None:
                    raise IngestApplyError(
                        "local consensus ingest shard continuation stall could not "
                        "be sealed: " + stall_error
                    )
                raise IngestApplyError(
                    "local consensus ingest shard continuation did not converge: "
                    + _frontier_feedback_text(continuation_result)
                )
            completed_continuation = _load_pretriage_terminal_recovery(
                content,
                raw_keywords_for_ops,
                reviewer=frontier_reviewer,
            )
            if completed_continuation is None:
                raise IngestApplyError(
                    "ingest shard continuation applied without a terminal recovery proof"
                )
            recovered_result = _complete_pretriage_terminal_recovery(
                completed_continuation,
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
                    "created": list(completed_continuation.get("created") or []),
                    "updated": list(completed_continuation.get("updated") or []),
                    "frontier_status": completed_continuation.get("status"),
                    "audit": completed_continuation.get("audit"),
                    "recovery_basis": "durable_shard_continuation",
                    "model_calls": continuation_budget.used,
                    "read_back": recovered_result.get("read_back"),
                },
            )
            failed = False
            return

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
            local_disposition = (
                "triage_no_operations"
                if not plan
                else "all_generation_failed"
                if failed_ops and not all_operations
                else "partial_generation_failed"
                if failed_ops
                else "operations_available"
            )
            frontier_result = _review_and_apply_ingest_operations(
                all_operations,
                raw_content=content,
                raw_keywords=raw_keywords_for_ops,
                source_raw=source_raw,
                triage_plan=plan,
                failed_operation_specs=failed_op_specs,
                local_disposition=local_disposition,
                reviewer=frontier_reviewer,
                force_frontier_review=frontier_feedback is not None,
                frontier_budget=frontier_budget,
            )
            all_operations, frontier_result = _review_exact_ingest_repair_once(
                all_operations,
                frontier_result,
                raw_content=content,
                raw_keywords=raw_keywords_for_ops,
                source_raw=source_raw,
                reviewer=frontier_reviewer,
                frontier_budget=frontier_budget,
                triage_plan=plan,
                failed_operation_specs=failed_op_specs,
                local_disposition=local_disposition,
            )
            frontier_status = str(frontier_result.get("status") or "needs_retry")
            if frontier_status in {"apply_available", "confirmed_noop"}:
                break
            if frontier_status == "shard_continuation_pending":
                _record_ingest_shard_continuation(
                    job_id=job_id,
                    source_raw=source_raw,
                    result=frontier_result,
                )
                failed = False
                return
            if frontier_status == "frontier_budget_exhausted":
                feedback = _frontier_feedback_text(frontier_result)
                _raise_bounded_local_consensus_nonconvergence(
                    initial_authority=initial_review_authority,
                    reviewer=frontier_reviewer,
                    semantic_detail=feedback,
                    legacy_error=(
                        "local consensus ingest review did not converge after "
                        f"{frontier_budget.used} local review calls: {feedback}"
                    ),
                )
            frontier_feedback = _frontier_feedback_text(frontier_result)
            if not _frontier_retry_is_actionable(frontier_result):
                structured_failure = _structured_frontier_failure_class(frontier_result)
                if structured_failure == "local_semantic_no_quorum":
                    authority_sha256 = _structured_frontier_authority_sha256(
                        frontier_result
                    )
                    if authority_sha256 is not None:
                        raise IngestApplyError(
                            "local consensus semantic no quorum "
                            f"[authority_sha256={authority_sha256}]: "
                            + frontier_feedback
                        )
                if structured_failure in {
                    "input_too_large",
                    "context_window_exceeded",
                }:
                    raise IngestApplyError(
                        "local consensus structured failure "
                        f"[{structured_failure}]: {frontier_feedback}"
                    )
                typed_feedback = (
                    f"{structured_failure}: {frontier_feedback}"
                    if structured_failure is not None
                    else frontier_feedback
                )
                raise IngestApplyError(
                    "local consensus authority unavailable: " + typed_feedback
                )
            if frontier_budget.used >= frontier_budget.limit:
                detail = (
                    "structured review budget exhausted "
                    f"({frontier_budget.used}/{frontier_budget.limit}); "
                    + frontier_feedback
                )
                _raise_bounded_local_consensus_nonconvergence(
                    initial_authority=initial_review_authority,
                    reviewer=frontier_reviewer,
                    semantic_detail=detail,
                    legacy_error=(
                        "local consensus ingest review did not converge after "
                        f"{frontier_budget.used} local review calls: {detail}"
                    ),
                )
            _safe_log(
                "ingest | local consensus requested regeneration: "
                + frontier_feedback.replace("\n", " ")[:300]
            )
        else:
            detail = frontier_feedback or "unknown local consensus rejection"
            _raise_bounded_local_consensus_nonconvergence(
                initial_authority=initial_review_authority,
                reviewer=frontier_reviewer,
                semantic_detail=detail,
                legacy_error=(
                    "local consensus ingest review did not converge after "
                    f"{_MAX_FRONTIER_CONVERGENCE_ATTEMPTS} attempts: {detail}"
                ),
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
