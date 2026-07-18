"""Stage-one ingest triage execution."""

from __future__ import annotations

import re
from contextlib import nullcontext
from typing import Any, Callable

from llm_wiki_mcp import ollama as ollama_runtime
from llm_wiki_mcp.local_structured import ChatTransport


def _runtime():
    from llm_wiki_mcp import ingest

    return ingest


def _runtime_call(name: str):
    def call(*args: Any, **kwargs: Any) -> Any:
        return getattr(_runtime(), name)(*args, **kwargs)

    return call


_admit_ingest_context = _runtime_call("_admit_ingest_context")
_emit_triage_failure = _runtime_call("_emit_triage_failure")
_generate_with_progress = _runtime_call("_generate_with_progress")
_safe_log = _runtime_call("_safe_log")
_select_ingest_context = _runtime_call("_select_ingest_context")
_structured_chat_transport = _runtime_call("_structured_chat_transport")
_structured_generate_transport = _runtime_call("_structured_generate_transport")
_triage_plan_validation_issues = _runtime_call("_triage_plan_validation_issues")
_validate_triage_plan = _runtime_call("_validate_triage_plan")
all_pages = _runtime_call("all_pages")
load_ingest_config = _runtime_call("load_ingest_config")
page_id_from_path = _runtime_call("page_id_from_path")
required_structured_context_tokens = _runtime_call("required_structured_context_tokens")
LocalStructuredSession = _runtime_call("LocalStructuredSession")

from llm_wiki_mcp.ingest import (  # noqa: E402
    IngestContextCapacityError,
    IngestTriageFailure,
    TRIAGE_PLAN_SCHEMA,
    TRIAGE_SYSTEM_PROMPT,
    _DEFAULT_GENERATE_WITH_PROGRESS,
    _TRIAGE_CATALOG_TOP_N,
    _TRIAGE_MAX_FEEDBACK_BYTES,
    _TRIAGE_MAX_OUTPUT_BYTES,
    _TRIAGE_NUM_PREDICT,
)


def triage(
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
        {p.parent.name for p in all_pages() if p.parent != _runtime().PAGES_DIR}
    )
    catalog_lines = [
        (
            "Existing top-level folders (prefer the best semantic match for "
            f"every create): {', '.join(f'{f}/' for f in existing_folders)}"
        ),
        (
            "Create routing contract: never create directly under pages/. "
            "Use an existing folder when one fits; otherwise create a specific "
            "new kebab-case folder. Every create filename must be folder/page.md."
        ),
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
            and _runtime()._generate_with_progress
            is _DEFAULT_GENERATE_WITH_PROGRESS
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
