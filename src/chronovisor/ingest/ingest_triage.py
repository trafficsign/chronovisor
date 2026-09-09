"""Stage-one ingest triage execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from chronovisor.core import ollama as ollama_runtime
from chronovisor.core.canonical_json import canonical_json_sha256_strict
from chronovisor.core.triage_contract import (
    quote_span_payload as _c2_quote_span_payload,
)
from chronovisor.core.triage_contract import (
    unique_quote_byte_range as _c2_quote_span,
)
from chronovisor.decision.local_structured import (
    ChatTransport,
    ValidationIssue,
    structured_request_sha256,
)
from chronovisor.ingest.ingest_schemas import (
    TRIAGE_C2_MAX_QUOTE_CHARS,
    TRIAGE_C2_MAX_QUOTE_LIST,
    TRIAGE_C2_MAX_SEMANTIC_ROWS,
    TRIAGE_C2_MAX_SOURCE_RECORD_ID_CHARS,
    TRIAGE_C2_MAX_SOURCE_RECORD_TEXT_CHARS,
    TRIAGE_C2_MAX_SOURCE_RECORDS,
    TRIAGE_C2_SCHEMA,
    TRIAGE_C2_SCHEMA_VERSION,
)


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

from chronovisor.ingest.ingest import (  # noqa: E402, I001
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


TRIAGE_C2_SYSTEM_PROMPT = """\
You are the explicit C2 knowledge-wiki triage engine. Analyze the supplied
source records and return one JSON object with exactly two keys:
`operations` and `semantic_evidence`.

`operations` uses the existing five fields (type, filename, title, keywords,
summary), with the same routing and one-operation-per-target rules as v1.
`semantic_evidence` contains at most eight model judgments. Every row must
name one supplied record_id. Copy quote, subject_quote, scope_quotes, and
condition_quotes exactly from that record's text; use null or [] when unknown.
Use kind=unknown unless the text explicitly supports proposal, decision, or
result. Do not invent facts, dates, validity, hashes, or byte offsets. The
host computes source ranges and digests after validating the copied text.
Do not return any field outside the documented wrapper or row fields.
"""


@dataclass(frozen=True, slots=True)
class _C2SourceRecord:
    record_id: str
    text: str


def _normalize_c2_source_records(
    source_records: Sequence[Mapping[str, Any]],
) -> tuple[_C2SourceRecord, ...]:
    """Validate the caller-owned source-record boundary before model use."""

    if isinstance(source_records, (str, bytes, bytearray)):
        raise ValueError("C2 source_records must be a sequence of mappings")
    if not isinstance(source_records, Sequence) or not source_records:
        raise ValueError("C2 source_records must contain at least one record")
    if len(source_records) > TRIAGE_C2_MAX_SOURCE_RECORDS:
        raise ValueError("C2 source_records exceed the fixed record limit")

    normalized: list[_C2SourceRecord] = []
    seen: set[str] = set()
    for index, row in enumerate(source_records):
        if not isinstance(row, Mapping):
            raise ValueError(f"C2 source record {index} must be an object")
        record_id = row.get("record_id")
        text = row.get("text")
        if (
            not isinstance(record_id, str)
            or not record_id.strip()
            or record_id != record_id.strip()
            or len(record_id) > TRIAGE_C2_MAX_SOURCE_RECORD_ID_CHARS
            or any(ord(char) < 0x20 or char == "\x7f" for char in record_id)
        ):
            raise ValueError(f"C2 source record {index} has an invalid record_id")
        if record_id in seen:
            raise ValueError(f"C2 source record_id is duplicated: {record_id}")
        if (
            not isinstance(text, str)
            or not text
            or len(text) > TRIAGE_C2_MAX_SOURCE_RECORD_TEXT_CHARS
        ):
            raise ValueError(f"C2 source record {record_id} has invalid text")
        try:
            record_id.encode("utf-8")
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(
                f"C2 source record {record_id!r} must contain valid UTF-8 text"
            ) from exc
        seen.add(record_id)
        normalized.append(_C2SourceRecord(record_id=record_id, text=text))
    return tuple(normalized)


def _render_c2_source_records(records: Sequence[_C2SourceRecord]) -> str:
    """Render fixed source records without asking the model to derive identity."""

    chunks: list[str] = []
    for record in records:
        chunks.append(
            "--- SOURCE RECORD "
            + record.record_id
            + " ---\n"
            + record.text
            + "\n--- END SOURCE RECORD ---"
        )
    return "\n\n".join(chunks)


def _c2_source_records_sha256(records: Sequence[_C2SourceRecord]) -> str:
    return canonical_json_sha256_strict(
        [{"record_id": record.record_id, "text": record.text} for record in records]
    )


def _c2_semantic_issue(
    pointer: str,
    message: str,
    *,
    keyword: str = "sourceQuote",
) -> ValidationIssue:
    return ValidationIssue(
        pointer=pointer,
        keyword=keyword,
        expected="one unique substring of the selected source record",
        received={"type": "untrusted_semantic_value"},
        message=message,
    )


def _c2_semantic_validation_issues(
    value: Any,
    records: Sequence[_C2SourceRecord],
) -> list[ValidationIssue]:
    if not isinstance(value, Mapping):
        return [
            _c2_semantic_issue(
                "",
                "C2 output must be an object with operations and semantic_evidence",
                keyword="type",
            )
        ]
    rows = value.get("semantic_evidence")
    if not isinstance(rows, list):
        return [_c2_semantic_issue("/semantic_evidence", "semantic_evidence must be an array", keyword="type")]
    by_id = {record.record_id: record for record in records}
    issues: list[ValidationIssue] = []
    if len(rows) > TRIAGE_C2_MAX_SEMANTIC_ROWS:
        issues.append(
            _c2_semantic_issue(
                "/semantic_evidence",
                "semantic_evidence exceeds the fixed row limit",
                keyword="maxItems",
            )
        )
    for index, row in enumerate(rows[:TRIAGE_C2_MAX_SEMANTIC_ROWS]):
        pointer = f"/semantic_evidence/{index}"
        if not isinstance(row, Mapping):
            issues.append(_c2_semantic_issue(pointer, "semantic evidence row must be an object", keyword="type"))
            continue
        record_id = row.get("record_id")
        record = by_id.get(record_id) if isinstance(record_id, str) else None
        if record is None:
            issues.append(_c2_semantic_issue(f"{pointer}/record_id", "record_id is not in the fixed source set", keyword="sourceRecord"))
            continue
        quote = row.get("quote")
        if quote is not None and (
            not isinstance(quote, str)
            or not quote
            or len(quote) > TRIAGE_C2_MAX_QUOTE_CHARS
            or _c2_quote_span(record.text, quote) is None
        ):
            issues.append(_c2_semantic_issue(f"{pointer}/quote", "quote must be one unique source substring"))
        if row.get("kind") in {"proposal", "decision", "result"} and quote is None:
            issues.append(
                _c2_semantic_issue(
                    f"{pointer}/quote",
                    "non-unknown semantic evidence requires a source quote",
                )
            )
        subject_quote = row.get("subject_quote")
        if subject_quote is not None and (
            not isinstance(subject_quote, str)
            or not subject_quote
            or len(subject_quote) > TRIAGE_C2_MAX_QUOTE_CHARS
            or _c2_quote_span(record.text, subject_quote) is None
        ):
            issues.append(_c2_semantic_issue(f"{pointer}/subject_quote", "subject_quote must be one unique source substring"))
        for field in ("scope_quotes", "condition_quotes"):
            quotes = row.get(field)
            if quotes is None:
                continue
            if not isinstance(quotes, list) or len(quotes) > TRIAGE_C2_MAX_QUOTE_LIST:
                issues.append(_c2_semantic_issue(f"{pointer}/{field}", f"{field} exceeds its fixed list limit", keyword="maxItems"))
                continue
            for quote_index, item in enumerate(quotes):
                if (
                    not isinstance(item, str)
                    or not item
                    or len(item) > TRIAGE_C2_MAX_QUOTE_CHARS
                    or _c2_quote_span(record.text, item) is None
                ):
                    issues.append(_c2_semantic_issue(f"{pointer}/{field}/{quote_index}", f"{field} entries must be unique source substrings"))
    return issues


def _materialize_c2_semantic_evidence(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[_C2SourceRecord],
) -> list[dict[str, Any]]:
    """Attach deterministic spans/digests in decoded-source coordinates."""

    by_id = {record.record_id: record for record in records}
    materialized: list[dict[str, Any]] = []
    for row in rows:
        record_id = row.get("record_id")
        record = by_id.get(record_id) if isinstance(record_id, str) else None
        if record is None:
            raise ValueError("C2 semantic evidence references an unknown record")
        output: dict[str, Any] = {
            "record_id": record.record_id,
            "quote": row.get("quote"),
            "kind": row.get("kind"),
            "subject_quote": row.get("subject_quote"),
            "scope_quotes": row.get("scope_quotes"),
            "condition_quotes": row.get("condition_quotes"),
            "source_text_sha256": hashlib.sha256(record.text.encode("utf-8")).hexdigest(),
            "byte_coordinate_space": "decoded_source_text_utf8",
        }
        output["quote_span"] = _c2_quote_span_payload(record.text, row.get("quote"))
        output["subject_span"] = _c2_quote_span_payload(
            record.text, row.get("subject_quote")
        )
        for field, span_field in (
            ("scope_quotes", "scope_spans"),
            ("condition_quotes", "condition_spans"),
        ):
            quotes = row.get(field)
            output[span_field] = (
                None
                if quotes is None
                else [
                    _c2_quote_span_payload(record.text, quote)
                    for quote in quotes
                ]
            )
        materialized.append(output)
    return materialized


def _build_triage_catalog(
    content: str,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    raise_on_failure: bool,
    failure_message: str,
    log_label: str = "",
) -> tuple[Any, str] | None:
    """Build the bounded existing-page catalog shared by v1 and C2 triage."""

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
            f"every create): {', '.join(f'{folder}/' for folder in existing_folders)}"
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
        failure = IngestTriageFailure("transport_error", failure_message)
        _emit_triage_failure(progress_callback, failure)
        if raise_on_failure:
            raise failure from exc
        return None
    results = results[:_TRIAGE_CATALOG_TOP_N]
    for row in results:
        catalog_lines.append(f"  [[{row.page_id}]] — {row.title}")
    label = f" {log_label}" if log_label else ""
    _safe_log(
        f"ingest | triage{label} catalog filtered to {len(results)} pages "
        f"(of {store.page_count()} total)"
    )
    return store, "\n".join(catalog_lines)


def _run_structured_triage_session(
    prompt: str,
    schema: Mapping[str, Any],
    *,
    system: str,
    role: str,
    transport: ChatTransport | None,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    value_validator: Callable[[Any], Sequence[ValidationIssue]] | None = None,
    plain_text_contract: str | None = None,
    plain_text_decoder: Callable[[str], Any] | None = None,
    raise_on_failure: bool,
) -> Any | None:
    """Run one triage LocalStructuredSession with the shared runtime guards."""

    config = load_ingest_config()
    triage_num_predict = min(config.num_predict, _TRIAGE_NUM_PREDICT)
    required_num_ctx = required_structured_context_tokens(
        prompt,
        schema,
        system=system,
        num_predict=triage_num_predict,
        max_output_chars=_TRIAGE_MAX_OUTPUT_BYTES,
        max_feedback_chars=_TRIAGE_MAX_FEEDBACK_BYTES,
        plain_text_contract=plain_text_contract,
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
                role=role,
                runtime_role=ollama_runtime.INGEST_GENERATION_RUNTIME_ROLE,
                runtime_location=route.location if route is not None else None,
                source_data_class="raw",
                source_sensitivity="high",
                resource_managed=local_ollama,
                num_ctx=selected_num_ctx,
                num_predict=triage_num_predict,
                keep_alive=config.keep_alive,
                read_timeout_ms=config.read_timeout_ms,
                max_input_chars=selected_num_ctx,
                max_output_chars=_TRIAGE_MAX_OUTPUT_BYTES,
                max_feedback_chars=_TRIAGE_MAX_FEEDBACK_BYTES,
            ).run(
                prompt,
                schema,
                system=system,
                value_validator=value_validator,
                plain_text_contract=plain_text_contract,
                plain_text_decoder=plain_text_decoder,
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
    return result


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
    catalog_result = _build_triage_catalog(
        content,
        progress_callback=progress_callback,
        raise_on_failure=raise_on_failure,
        failure_message="triage catalog search unavailable after bounded lexical fallback",
    )
    if catalog_result is None:
        return None
    _, catalog = catalog_result

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

    result = _run_structured_triage_session(
        prompt,
        TRIAGE_PLAN_SCHEMA,
        system=TRIAGE_SYSTEM_PROMPT,
        role="ingest_triage",
        transport=transport,
        progress_callback=progress_callback,
        value_validator=_validate_effective_triage_plan,
        plain_text_contract=TRIAGE_TEXT_OUTPUT_CONTRACT,
        plain_text_decoder=_decode_triage_output,
        raise_on_failure=raise_on_failure,
    )
    if result is None:
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


def triage_c2(
    source_records: Sequence[Mapping[str, Any]],
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    frontier_feedback: str | None = None,
    transport: ChatTransport | None = None,
    raise_on_failure: bool = False,
) -> dict[str, Any] | None:
    """Run the explicit C2 triage wrapper against fixed source records.

    The model sees the supplied record text and may *judge* a bounded semantic
    row, but it never supplies hashes or offsets.  Returned semantic rows are
    host materializations in ``decoded_source_text_utf8`` coordinates; they are
    not RawEvidenceRef bindings or fact-validity assertions.  The ordinary
    :func:`triage` entry point, its five-column fallback, and its v1 bytes are
    intentionally untouched.
    """

    records = _normalize_c2_source_records(source_records)
    source_text = _render_c2_source_records(records)
    source_records_sha256 = _c2_source_records_sha256(records)

    catalog_result = _build_triage_catalog(
        source_text,
        progress_callback=progress_callback,
        raise_on_failure=raise_on_failure,
        failure_message="triage C2 catalog search unavailable after bounded lexical fallback",
        log_label="C2",
    )
    if catalog_result is None:
        return None
    _, catalog = catalog_result

    feedback_block = ""
    if frontier_feedback:
        feedback_block = f"""

---
Previous local consensus review (authoritative correction instructions):
---
{frontier_feedback}
---
Regenerate the operations and semantic judgments from the fixed source
records. Remove unsupported claims and use null/[] when a field is not
explicitly grounded in one record.
"""

    prompt = f"""{catalog}

---
Fixed source records (record_id labels are host-bound; quote only record text):
---
{source_text}
---
{feedback_block}

Return the C2 wrapper. `operations` must contain the existing five operation
fields. `semantic_evidence` may contain at most {TRIAGE_C2_MAX_SEMANTIC_ROWS}
rows. For every semantic row, copy each quote exactly from the selected record;
do not calculate or emit hashes, byte offsets, dates, or validity intervals.
"""

    def validate_c2(value: Any) -> Sequence[ValidationIssue]:
        issues = _validate_effective_triage_plan(
            value.get("operations") if isinstance(value, Mapping) else None,
        )
        issues.extend(_c2_semantic_validation_issues(value, records))
        return issues

    result = _run_structured_triage_session(
        prompt,
        TRIAGE_C2_SCHEMA,
        system=TRIAGE_C2_SYSTEM_PROMPT,
        role="ingest_triage_c2",
        transport=transport,
        progress_callback=progress_callback,
        value_validator=validate_c2,
        raise_on_failure=raise_on_failure,
    )
    if result is None:
        return None

    audit = {
        "schema_version": TRIAGE_C2_SCHEMA_VERSION,
        "schema_sha256": canonical_json_sha256_strict(TRIAGE_C2_SCHEMA),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "request_sha256": structured_request_sha256(
            prompt,
            TRIAGE_C2_SCHEMA,
            TRIAGE_C2_SYSTEM_PROMPT,
        ),
        "source_records_sha256": source_records_sha256,
        "source_record_count": len(records),
        "semantic_evidence_authority": "model_judgment",
        "local_structured": result.audit_record(),
    }
    if not result.ok:
        failure = IngestTriageFailure(
            result.failure_class or "unknown",
            result.failure_reason or "structured C2 triage failed",
        )
        _safe_log(
            "ingest | triage C2 structured session failed "
            f"({failure.failure_class}: {failure.reason[:160]})"
        )
        _emit_triage_failure(progress_callback, failure)
        if raise_on_failure:
            raise failure
        return None

    raw_value = result.value
    if not isinstance(raw_value, Mapping):
        failure = IngestTriageFailure(
            "value_validation_error",
            "C2 triage returned a non-object wrapper",
        )
        _emit_triage_failure(progress_callback, failure)
        if raise_on_failure:
            raise failure
        return None
    operations = raw_value.get("operations")
    semantic_rows = raw_value.get("semantic_evidence")
    validated_operations = _validate_triage_plan(
        operations,
        coerce_missing_updates=True,
    )
    semantic_issues = _c2_semantic_validation_issues(raw_value, records)
    if validated_operations is None or semantic_issues:
        failure = IngestTriageFailure(
            "value_validation_error",
            "C2 triage host validation rejected the wrapper",
        )
        _emit_triage_failure(progress_callback, failure)
        if raise_on_failure:
            raise failure
        return None
    assert isinstance(semantic_rows, list)
    try:
        materialized = _materialize_c2_semantic_evidence(semantic_rows, records)
    except (TypeError, ValueError) as exc:
        failure = IngestTriageFailure("value_validation_error", str(exc))
        _emit_triage_failure(progress_callback, failure)
        if raise_on_failure:
            raise failure from exc
        return None
    if progress_callback is not None:
        progress_callback({"event": "done", "active": False})
    return {
        "operations": validated_operations,
        "semantic_evidence": materialized,
        "audit": audit,
    }
