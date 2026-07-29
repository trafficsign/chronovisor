"""Stage-two page generation for ingest."""

from __future__ import annotations

import hashlib
from contextlib import nullcontext
from datetime import date
from typing import Any, Callable

from chronovisor.core import ollama as ollama_runtime


def _runtime():
    from chronovisor.ingest import ingest

    return ingest


def _runtime_call(name: str):
    def call(*args: Any, **kwargs: Any) -> Any:
        return getattr(_runtime(), name)(*args, **kwargs)

    return call


_admit_ingest_context = _runtime_call("_admit_ingest_context")
_build_compact_update_context = _runtime_call("_build_compact_update_context")
_build_focused_context = _runtime_call("_build_focused_context")
_build_page_generation_prompt = _runtime_call("_build_page_generation_prompt")
_generate_with_progress = _runtime_call("_generate_with_progress")
_generation_completion_failure = _runtime_call("_generation_completion_failure")
_page_generation_repair_prompt = _runtime_call("_page_generation_repair_prompt")
_page_generation_transcript = _runtime_call("_page_generation_transcript")
_page_generation_truncation_retry_prompt = _runtime_call("_page_generation_truncation_retry_prompt")
_repair_transport_attested_page_boundary = _runtime_call("_repair_transport_attested_page_boundary")
_required_generate_context_tokens = _runtime_call("_required_generate_context_tokens")
_safe_log = _runtime_call("_safe_log")
_select_page_generation_budget = _runtime_call("_select_page_generation_budget")
_supports_keyword = _runtime_call("_supports_keyword")
_validate_generated_page_output = _runtime_call("_validate_generated_page_output")
load_ingest_config = _runtime_call("load_ingest_config")

from chronovisor.ingest.ingest import (  # noqa: E402
    GENERATE_SYSTEM_PROMPT,
    UPDATE_SYSTEM_PROMPT,
    IngestContextCapacityError,
    IngestTriageFailure,
    _CompactUpdateContext,
    _DEFAULT_GENERATE_WITH_PROGRESS,
    _MAX_PAGE_GENERATION_REPAIR_TURNS,
    _MAX_PAGE_GENERATION_RESPONSES,
    _PAGE_GENERATION_CONTEXT_SAFETY_TOKENS,
    _PageGenerationBudget,
)


def generate_one(
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
    if _supports_keyword(_runtime()._build_focused_context, "max_bytes"):
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

    prompt = _build_page_generation_prompt(
        context=context,
        raw_content=raw_content,
        op_type=op_type,
        filename=filename,
        title=title,
        summary=summary,
        feedback_block=feedback_block,
        current_date=current_date,
    )

    full_system_prompt = (
        UPDATE_SYSTEM_PROMPT if op_type == "update" else GENERATE_SYSTEM_PROMPT
    )
    system_prompt = full_system_prompt
    full_prompt = prompt
    attempts_made = 0
    generation_budget: _PageGenerationBudget | None = None
    capacity_error: IngestContextCapacityError | None = None
    compact_context: _CompactUpdateContext | None = None
    original_required_num_ctx = _required_generate_context_tokens(
        full_prompt,
        full_system_prompt,
        num_predict=config.num_predict,
    )
    if original_required_num_ctx > config.max_num_ctx and op_type == "update":
        compact_context = _build_compact_update_context(
            op,
            raw_content,
            max_selected_bytes=config.max_related_context_bytes,
        )
        if compact_context is not None:
            context = compact_context.text
            prompt = _build_page_generation_prompt(
                context=context,
                raw_content=raw_content,
                op_type=op_type,
                filename=filename,
                title=title,
                summary=summary,
                feedback_block=feedback_block,
                current_date=current_date,
            )
            system_prompt = (
                UPDATE_SYSTEM_PROMPT
                + "\n\nOversized append-only context rule:\n"
                + "- The complete stored page remains byte-identical on disk.\n"
                + "- The prompt contains a hash-bound outline and only selected "
                + "complete Markdown sections.\n"
                + "- Omitted section bytes are still present on disk; output only "
                + "a new append body and never claim that omitted content is absent.\n"
                + "- Never rewrite, summarize, or replace the stored preimage.\n"
            )
            try:
                generation_budget = _select_page_generation_budget(
                    prompt,
                    system_prompt,
                    configured_num_predict=config.num_predict,
                    num_ctx=config.num_ctx,
                    max_num_ctx=config.max_num_ctx,
                )
            except IngestContextCapacityError as compact_exc:
                capacity_error = compact_exc
            else:
                capacity_error = None
                _safe_log(
                    "ingest | compacted oversized append-only update context for "
                    f"{compact_context.page_id}: page={compact_context.page_bytes}B "
                    f"sections={len(compact_context.selected_sections)}/"
                    f"{compact_context.section_count} required="
                    f"{generation_budget.required_num_ctx}"
                )

    # A compact outline is preferred over sacrificing output budget or holding
    # the model at its hard context ceiling. If compaction is unavailable or
    # itself cannot fit, preserve the prior adaptive full-context behavior.
    if generation_budget is None:
        compact_context = None
        prompt = full_prompt
        system_prompt = full_system_prompt
        try:
            generation_budget = _select_page_generation_budget(
                prompt,
                system_prompt,
                configured_num_predict=config.num_predict,
                num_ctx=config.num_ctx,
                max_num_ctx=config.max_num_ctx,
            )
        except IngestContextCapacityError as exc:
            capacity_error = exc

    if generation_budget is None:
        exc = capacity_error or IngestContextCapacityError(
            "page generation context admission failed without a typed reason"
        )
        if diagnostics is not None:
            diagnostics.update(
                {
                    "failure_class": "context_window_exceeded",
                    "reason": str(exc),
                    "attempts": 0,
                    "original_required_num_ctx": original_required_num_ctx,
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

    selected_num_ctx = generation_budget.num_ctx
    selected_num_predict = generation_budget.num_predict
    if diagnostics is not None:
        diagnostics.update(
            {
                "num_ctx": selected_num_ctx,
                "num_predict": selected_num_predict,
                "required_num_ctx": generation_budget.required_num_ctx,
                "original_required_num_ctx": original_required_num_ctx,
            }
        )
        if compact_context is not None:
            diagnostics.update(
                {
                    "context_strategy": "append_only_outline_sections",
                    "context_page_id": compact_context.page_id,
                    "context_page_sha256": compact_context.page_sha256,
                    "context_page_bytes": compact_context.page_bytes,
                    "context_section_count": compact_context.section_count,
                    "context_selected_sections": [
                        {
                            "start_line": section.start_line,
                            "end_line": section.end_line,
                            "sha256": section.sha256,
                            "bytes": len(section.content.encode("utf-8")),
                        }
                        for section in compact_context.selected_sections
                    ],
                }
            )

    try:
        live_transport = (
            _runtime()._generate_with_progress is _DEFAULT_GENERATE_WITH_PROGRESS
        )
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
                "num_predict": selected_num_predict,
                "keep_alive": config.keep_alive,
                "read_timeout_ms": config.read_timeout_ms,
                "temperature": config.temperature,
            }
            for name, value in optional.items():
                if _supports_keyword(_runtime()._generate_with_progress, name):
                    generate_kwargs[name] = value
            if _supports_keyword(
                _runtime()._generate_with_progress, "return_metadata"
            ):
                generate_kwargs["return_metadata"] = True
            messages = [{"role": "user", "content": prompt}]
            seen_output_hashes: set[str] = set()
            output_truncation_retries = 0
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
                        if (
                            failure_class == "output_truncated"
                            and attempt_index < _MAX_PAGE_GENERATION_RESPONSES - 1
                        ):
                            # Never feed the partial response back as assistant
                            # history. Re-anchor on the original evidence and
                            # ask for a successively shorter complete
                            # replacement within this bounded ingest call.
                            output_truncation_retries += 1
                            target_tokens = min(
                                selected_num_predict,
                                max(
                                    512,
                                    selected_num_predict
                                    // (2**output_truncation_retries),
                                ),
                            )
                            retry_prompt = _page_generation_truncation_retry_prompt(
                                op_type=op_type,
                                filename=filename,
                                max_output_tokens=target_tokens,
                            )
                            messages = [
                                {"role": "user", "content": prompt},
                                {"role": "user", "content": retry_prompt},
                            ]
                            if diagnostics is not None:
                                diagnostics.update(
                                    {
                                        "output_truncation_retries": (
                                            output_truncation_retries
                                        ),
                                        "replacement_token_target": target_tokens,
                                        "truncated_output_sha256": hashlib.sha256(
                                            output.content.encode("utf-8")
                                        ).hexdigest(),
                                    }
                                )
                            _safe_log(
                                "ingest | targeted generate replacement "
                                f"{output_truncation_retries}/"
                                f"{_MAX_PAGE_GENERATION_REPAIR_TURNS} for "
                                f"{filename}: output_truncated, target<="
                                f"{target_tokens} tokens"
                            )
                            if progress_callback is not None:
                                progress_callback(
                                    {
                                        "event": "repair",
                                        "active": True,
                                        "repair_turn": attempt_index + 1,
                                        "failure_class": failure_class,
                                    }
                                )
                            continue
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
                        # history. Bounded replacements above never include
                        # it; every other incomplete transport fails closed.
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
                    if compact_context is not None:
                        # Private host binding: preparation must observe the
                        # exact page snapshot used to build the compact view.
                        result["_compact_update_preimage_sha256"] = (
                            compact_context.page_sha256
                        )
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
