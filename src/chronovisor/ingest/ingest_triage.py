"""Stage-one ingest triage execution."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

from chronovisor.core import ollama as ollama_runtime
from chronovisor.decision.local_structured import ChatTransport


def _runtime():
    from chronovisor.ingest import ingest

    return ingest


def _runtime_call(name: str):
    def call(*args: Any, **kwargs: Any) -> Any:
        return getattr(_runtime(), name)(*args, **kwargs)

    return call


_admit_ingest_context = _runtime_call("_admit_ingest_context")
_emit_triage_failure = _runtime_call("_emit_triage_failure")
_generate_with_progress = _runtime_call("_generate_with_progress")
_host_phase = _runtime_call("_host_phase")
_safe_log = _runtime_call("_safe_log")
_select_ingest_context = _runtime_call("_select_ingest_context")
_structured_chat_transport = _runtime_call("_structured_chat_transport")
_structured_generate_transport = _runtime_call("_structured_generate_transport")
_triage_plan_validation_issues = _runtime_call("_triage_plan_validation_issues")
_validate_triage_plan = _runtime_call("_validate_triage_plan")
load_ingest_config = _runtime_call("load_ingest_config")
required_structured_context_tokens = _runtime_call("required_structured_context_tokens")
LocalStructuredSession = _runtime_call("LocalStructuredSession")


TRIAGE_TEXT_OUTPUT_CONTRACT = """\
Return NOOP when no durable page operation is warranted.
Otherwise return one operation per line with exactly these five pipe-separated
columns and no header:
create | folder/page-id.md | Page title | keyword one; keyword two | Brief summary
update | existing-page-id.md | Existing title | keyword one; keyword two | New facts
Use semicolons only to separate keywords. Keep every operation on one line.
"""


def _validate_effective_triage_plan(value: Any) -> list[Any]:
    with _host_phase("target-resolution"):
        return _triage_plan_validation_issues(
            value,
            resolve_effective_targets=True,
        )


def _decode_triage_output(text: str) -> list[dict[str, Any]]:
    """Materialize the model's compact rows into the validated host plan."""

    stripped = text.strip()
    lines = stripped.splitlines()
    if (
        len(lines) >= 2
        and lines[0].strip().startswith("```")
        and lines[-1].strip() == "```"
    ):
        stripped = "\n".join(lines[1:-1]).strip()
        lines = stripped.splitlines()
    if stripped.startswith(("[", "{")):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid legacy structured response: {exc.msg}") from exc
    if stripped.casefold() == "noop":
        return []

    operations: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        line = line.strip("|").strip()
        parts = [part.strip() for part in line.split("|", 4)]
        if len(parts) == 5 and parts[0].casefold() == "type":
            continue
        if parts and all(part and set(part) <= {"-", ":"} for part in parts):
            continue
        if len(parts) != 5:
            raise ValueError(
                f"line {line_number} must contain exactly five pipe-separated columns"
            )
        op_type, filename, title, keyword_text, summary = parts
        op_type = op_type.casefold()
        if op_type not in {"create", "update"}:
            raise ValueError(f"line {line_number} type must be create or update")
        operations.append(
            {
                "type": op_type,
                "filename": filename,
                "title": title,
                "keywords": [
                    keyword.strip()
                    for keyword in keyword_text.split(";")
                    if keyword.strip()
                ],
                "summary": summary,
            }
        )
    if not operations:
        raise ValueError("response must be NOOP or contain at least one operation row")
    return operations

from chronovisor.ingest.ingest import (  # noqa: E402
    _DEFAULT_GENERATE_WITH_PROGRESS,
    _TRIAGE_CATALOG_TOP_N,
    _TRIAGE_MAX_FEEDBACK_BYTES,
    _TRIAGE_MAX_OUTPUT_BYTES,
    _TRIAGE_NUM_PREDICT,
    TRIAGE_PLAN_SCHEMA,
    TRIAGE_SYSTEM_PROMPT,
    IngestContextCapacityError,
    IngestTriageFailure,
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
    store = _runtime().get_store()
    store.ensure_loaded()
    existing_folders = sorted(
        {
            parts[1]
            for key in store.all_canonical_page_keys()
            if len(parts := key.split("/")) > 2 and parts[0] == "pages"
        }
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
        from chronovisor.core.search import search as chronovisor_search
        from chronovisor.core.search import search_existing_bm25

        query_text = content[:2000]
        try:
            results, _ = chronovisor_search(
                query_text,
                top_n=_TRIAGE_CATALOG_TOP_N,
                semantic=True,
            )
        except Exception:
            results = search_existing_bm25(
                query_text,
                top_n=_TRIAGE_CATALOG_TOP_N,
            )
    except Exception as exc:
        failure = IngestTriageFailure(
            "transport_error",
            "triage catalog search unavailable after bounded lexical fallback",
        )
        _emit_triage_failure(progress_callback, failure)
        if raise_on_failure:
            raise failure from exc
        return None
    results = results[:_TRIAGE_CATALOG_TOP_N]
    for r in results:
        catalog_lines.append(f"  [[{r.page_id}]] — {r.title}")
    _safe_log(
        f"ingest | triage catalog filtered to {len(results)} pages "
        f"(of {store.page_count()} total)"
    )

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

Analyze the raw data above and return the page-operation record."""

    config = load_ingest_config()
    triage_num_predict = min(config.num_predict, _TRIAGE_NUM_PREDICT)
    required_num_ctx = required_structured_context_tokens(
        prompt,
        TRIAGE_PLAN_SCHEMA,
        system=TRIAGE_SYSTEM_PROMPT,
        num_predict=triage_num_predict,
        max_output_chars=_TRIAGE_MAX_OUTPUT_BYTES,
        max_feedback_chars=_TRIAGE_MAX_FEEDBACK_BYTES,
        plain_text_contract=TRIAGE_TEXT_OUTPUT_CONTRACT,
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
        route = (
            ollama_runtime.runtime_generation_routes(
                (ollama_runtime.INGEST_GENERATION_RUNTIME_ROLE,)
            )[0]
            if live_transport
            else None
        )
        local_ollama = (
            route is not None
            and route.provider == "ollama"
            and route.location == "local"
        )
        lease = (
            ollama_runtime.model_resource_lease(exclusive=True)
            if local_ollama
            else nullcontext()
        )
        with lease:
            if local_ollama and route is not None:
                selected_num_ctx = _admit_ingest_context(
                    config,
                    selected_num_ctx,
                    model=route.model,
                )
            session_transport = transport
            if session_transport is None and not live_transport:
                session_transport = _structured_generate_transport(progress_callback)
            result = LocalStructuredSession(
                model=route.model if route is not None else "injected",
                transport=session_transport,
                role="ingest_triage",
                runtime_role=ollama_runtime.INGEST_GENERATION_RUNTIME_ROLE,
                runtime_location=route.location if route is not None else None,
                source_data_class="raw",
                source_sensitivity="high",
                resource_managed=local_ollama,
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
                value_validator=_validate_effective_triage_plan,
                plain_text_contract=TRIAGE_TEXT_OUTPUT_CONTRACT,
                plain_text_decoder=_decode_triage_output,
            )
    except IngestContextCapacityError as exc:
        failure = IngestTriageFailure("context_window_exceeded", str(exc))
        _emit_triage_failure(progress_callback, failure)
        if raise_on_failure:
            raise failure from exc
        return None
    except ollama_runtime.RuntimeBridgeError as exc:
        failure = IngestTriageFailure(exc.category, exc.category)
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
