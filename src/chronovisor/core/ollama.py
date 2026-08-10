"""Ollama API facade and temporary structured-runtime migration bridge."""

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any

import httpx

from chronovisor.core import ollama_calibration as _ollama_calibration
from chronovisor.core import ollama_lease as _ollama_lease
from chronovisor.core import ollama_telemetry as _ollama_telemetry
from chronovisor.core import ollama_transport as _ollama_transport
from chronovisor.core.runtime_config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INGEST_MODEL,
    IngestConfig,
    load_embedding_config,
    load_ingest_config,
)
from chronovisor.core.store import CHRONOVISOR_ROOT

GIB = _ollama_calibration.GIB
RESIDENCY_UPSHIFT_MIN_HEADROOM_BYTES = (
    _ollama_calibration.RESIDENCY_UPSHIFT_MIN_HEADROOM_BYTES
)
RESIDENCY_UPSHIFT_HEADROOM_RATIO = _ollama_calibration.RESIDENCY_UPSHIFT_HEADROOM_RATIO
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

OLLAMA_URL = _ollama_transport.OLLAMA_URL
HEALTH_CACHE_TTL = _ollama_transport.HEALTH_CACHE_TTL
OutputTooLargeError = _ollama_transport.OutputTooLargeError
ChatResponse = _ollama_transport.ChatResponse
GenerateResponse = _ollama_transport.GenerateResponse

MODEL = DEFAULT_INGEST_MODEL


def _safe_runtime_bridge_category(value: object) -> str:
    from chronovisor.core.llm_config import LLMConfigFailureCategory
    from chronovisor.core.llm_runtime import SAFE_FAILURE_CATEGORIES

    normalized = getattr(value, "value", value)
    allowed = SAFE_FAILURE_CATEGORIES | frozenset(
        category.value for category in LLMConfigFailureCategory
    )
    if isinstance(normalized, str) and normalized in allowed:
        return normalized
    return "backend_error"


class RuntimeBridgeError(RuntimeError):
    """Safe provider-neutral failure exposed during the consumer migration."""

    def __init__(self, category: str) -> None:
        self.category = _safe_runtime_bridge_category(category)
        super().__init__(self.category)


def _runtime_bridge_category(exc: Exception) -> str:
    return _safe_runtime_bridge_category(getattr(exc, "category", None))


@dataclass(frozen=True)
class RuntimeGenerationRoute:
    role: str
    provider: str
    model: str
    location: str
    structured_output: bool


def runtime_generation_routes(roles: Sequence[str]) -> tuple[RuntimeGenerationRoute, ...]:
    """Resolve exact configured identities once without invoking a backend."""

    from chronovisor.core.llm_config import LLMConfigError, load_default_llm_runtime
    from chronovisor.core.llm_runtime import LLMRuntimeError

    try:
        runtime = load_default_llm_runtime()
        return tuple(
            RuntimeGenerationRoute(
                role=route.role,
                provider=route.provider,
                model=route.model,
                location=route.location.value,
                structured_output=route.capabilities.structured_output,
            )
            for route in (runtime.resolve_generation(role) for role in roles)
        )
    except (LLMConfigError, LLMRuntimeError) as exc:
        raise RuntimeBridgeError(_runtime_bridge_category(exc)) from None


def source_data_classification(data_class: str, sensitivity: str) -> object:
    """Build the canonical runtime classification without exporting runtime types."""

    from chronovisor.core.llm_runtime import (
        SourceDataClass,
        SourceDataClassification,
        SourceSensitivity,
    )

    try:
        return SourceDataClassification(
            SourceDataClass(data_class),
            SourceSensitivity(sensitivity),
        )
    except ValueError:
        raise RuntimeBridgeError("source_classification_required") from None


def source_data_classification_values(source: object) -> tuple[str, str]:
    """Validate and unwrap one canonical source classification."""

    from chronovisor.core.llm_runtime import (
        SourceDataClass,
        SourceDataClassification,
        SourceSensitivity,
    )

    if (
        not isinstance(source, SourceDataClassification)
        or not isinstance(source.data_class, SourceDataClass)
        or not isinstance(source.sensitivity, SourceSensitivity)
    ):
        raise RuntimeBridgeError("source_classification_required")
    return source.data_class.value, source.sensitivity.value


def runtime_generation_location(role: str) -> str:
    """Return the configured location without exposing backend controls."""

    from chronovisor.core.llm_config import LLMConfigError, load_default_llm_runtime
    from chronovisor.core.llm_runtime import LLMRuntimeError

    try:
        return load_default_llm_runtime().generation_location(role).value
    except (LLMConfigError, LLMRuntimeError) as exc:
        raise RuntimeBridgeError(_runtime_bridge_category(exc)) from None


def runtime_structured_chat(
    messages: Sequence[Mapping[str, str]],
    *,
    runtime_role: str,
    source_data_class: str,
    source_sensitivity: str,
    format: Mapping[str, Any],
    num_ctx: int,
    num_predict: int,
    keep_alive: str,
    read_timeout_ms: int,
    max_output_chars: int,
    temperature: int | float,
    seed: int,
    think: bool | str,
) -> ChatResponse:
    """Run one structured turn through the cached provider-neutral runtime."""

    from chronovisor.core.llm_config import LLMConfigError, load_default_llm_runtime
    from chronovisor.core.llm_runtime import (
        LLMRuntimeError,
        MessageGenerationRequest,
        SourceDataClass,
        SourceDataClassification,
        SourceSensitivity,
    )

    try:
        source = SourceDataClassification(
            SourceDataClass(source_data_class),
            SourceSensitivity(source_sensitivity),
        )
    except ValueError:
        raise RuntimeBridgeError("source_classification_required") from None
    try:
        result = load_default_llm_runtime().generate(
            runtime_role,
            MessageGenerationRequest(
                messages=tuple(dict(message) for message in messages),
                format=dict(format),
                source=source,
                num_ctx=num_ctx,
                max_output_tokens=num_predict,
                keep_alive=keep_alive,
                timeout_ms=read_timeout_ms,
                max_output_chars=max_output_chars,
                temperature=temperature,
                seed=seed,
                think=think,
            ),
        )
    except (LLMConfigError, LLMRuntimeError) as exc:
        raise RuntimeBridgeError(_runtime_bridge_category(exc)) from None
    return ChatResponse(
        content=result.content,
        prompt_eval_count=result.usage.input_tokens,
        eval_count=result.usage.output_tokens,
        done=result.completed,
        done_reason=result.finish_reason,
    )


# ponytail: temporary facade bridge; delete after final W6 callers own runtime types.


def _client() -> httpx.Client:
    return _ollama_transport.client(base_url=OLLAMA_URL)


def client() -> httpx.Client:
    return _client()


def _raise_for_status_with_detail(response: httpx.Response) -> None:
    return _ollama_transport._raise_for_status_with_detail(response)


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
    return _ollama_transport.is_available(
        client=_client,
        cache_ttl=HEALTH_CACHE_TTL,
    )


def model_digests(models: Sequence[str]) -> dict[str, str]:
    return _ollama_transport.model_digests(models, client=_client)


def _ollama_daemon_process_identity() -> str:
    return _ollama_calibration._ollama_daemon_process_identity()


def _ollama_engine_identity() -> str:
    return _ollama_calibration._ollama_engine_identity(
        client=_client,
        daemon_identity=_ollama_daemon_process_identity,
    )


def memory_snapshot() -> MemorySnapshot:
    return _ollama_calibration.memory_snapshot()


def macos_pressure_snapshot() -> MacOSPressureSnapshot:
    return _ollama_calibration.macos_pressure_snapshot()


def _ollama_resource_rows() -> tuple[dict[str, int], dict[str, tuple[int, int]]]:
    return _ollama_transport._ollama_resource_rows(client=_client)


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
        return _ollama_transport.unload_named_model(
            model,
            verify_timeout=verify_timeout,
            client=_client,
            resource_rows=_ollama_resource_rows,
        )


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


def _num_ctx_for_prompt(
    prompt: str,
    system: str | None,
    config: IngestConfig,
) -> int:
    return _ollama_transport._num_ctx_for_prompt(prompt, system, config)


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
    return _ollama_transport._generate_unlocked(
        prompt,
        system,
        client=_client,
        load_ingest_config=load_ingest_config,
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
    return _ollama_transport._chat_unlocked(
        messages,
        client=_client,
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

    selected_model = model or embedding_model()
    with model_resource_lease(exclusive=False):
        with model_activity(
            model=selected_model,
            operation="search",
            pipeline="recall",
        ):
            return _ollama_transport.embed(
                texts,
                model=selected_model,
                read_timeout_ms=read_timeout_ms,
                client=_client,
            )


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
- Emit at most 8 operations
- Limit filename to 200 characters, title to 300, and summary to 2000
- Include 1 to 32 keywords, each at most 200 characters
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

PRESERVE_SOURCE_FACTS_RULE = (
    "Preserve every relevant source-grounded fact; do not omit names, numbers, "
    "dates, or decisions when summarizing."
)

GENERATE_SYSTEM_PROMPT = f"""\
You are a knowledge wiki structuring engine. Generate content for a SINGLE NEW wiki page.

Rules:
- Frontmatter MUST include: title, updated, AND tags
- Use the exact current date supplied in the user prompt for `updated`
- Never invent or infer dates that are absent from the raw evidence
- {PRESERVE_SOURCE_FACTS_RULE}
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
=== NEW PAGE: {{filename}} ===
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

UPDATE_SYSTEM_PROMPT = f"""\
You are a knowledge wiki structuring engine. Append content to an EXISTING wiki page.

Rules:
- DO NOT output frontmatter (no `---`, no title:, no updated: lines). The existing page already has frontmatter; your output is appended to its body.
- Never invent or infer dates that are absent from the raw evidence. Do not add a dated heading unless that date appears explicitly in the raw evidence.
- {PRESERVE_SOURCE_FACTS_RULE}
- DO NOT repeat content that already exists on the page (it is provided in context).
- Output ONLY the new section(s) to add — Japanese prose, headings, lists, code, etc.
- Cross-references: use [[wiki-link]] notation (page ID only, no folder path)
- Focus on facts, decisions, and technical knowledge

Output exactly one block:
=== UPDATE PAGE: {{filename}} ===
New section(s) here. Markdown body only — NO frontmatter delimiters.

=== END PAGE ===

The final non-whitespace line MUST be exactly `=== END PAGE ===`. Keep the
update concise enough to emit that closing line before stopping.
"""
