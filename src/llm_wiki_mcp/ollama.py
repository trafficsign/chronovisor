"""Ollama API client for Ingest/Lint operations."""

import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

import httpx

from llm_wiki_mcp.runtime_config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INGEST_MODEL,
    IngestConfig,
    load_embedding_config,
    load_ingest_config,
)

OLLAMA_URL = "http://localhost:11434"
MODEL = DEFAULT_INGEST_MODEL

# Health check cache
_health_cache: dict = {"status": None, "checked_at": 0.0}
HEALTH_CACHE_TTL = 900  # 15 minutes on failure

# Shared httpx.Client — one per process, reused across is_available /
# generate / embed / unload. Connection pooling avoids paying TCP setup
# and DNS lookup cost on every call. Per-call timeouts are still passed
# explicitly so the long-running /api/generate doesn't inherit the short
# health-check default.
_CLIENT_LOCK = threading.Lock()
_CLIENT: httpx.Client | None = None


def _client() -> httpx.Client:
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = httpx.Client(base_url=OLLAMA_URL)
    return _CLIENT


def is_available() -> bool:
    """Check if Ollama is running (cached on failure)."""
    now = time.time()

    # If last check failed, use cache for TTL
    if _health_cache["status"] is False:
        if now - _health_cache["checked_at"] < HEALTH_CACHE_TTL:
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


def _emit_progress(callback: Callable[[dict[str, Any]], None] | None, payload: dict[str, Any]) -> None:
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        pass


def ingest_model() -> str:
    return load_ingest_config().model


def _num_ctx_for_prompt(prompt: str, system: str | None, config: IngestConfig) -> int:
    # Keep ordinary saves on a smaller MLX context, but grow for unusually long
    # raw transcripts so the old 262K ceiling remains available when needed.
    prompt_chars = len(prompt) + (len(system) if system else 0)
    estimated_prompt_tokens = max(1, (prompt_chars + 1) // 2)
    needed = estimated_prompt_tokens + config.num_predict + 1024
    return min(max(config.num_ctx, needed), config.max_num_ctx)


def generate(
    prompt: str,
    system: str | None = None,
    *,
    format: dict | str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Call Ollama generate API.

    Uses keep_alive="5m" to keep model loaded for 5 minutes after use.
    This avoids cold-start on consecutive calls (e.g. Ingest then Lint)
    while still freeing memory after a reasonable idle period.

    When ``progress_callback`` is provided, the call uses Ollama's streaming
    response and periodically emits lightweight progress dictionaries while
    still returning the final response string for existing callers.
    """
    config = load_ingest_config()
    num_ctx = _num_ctx_for_prompt(prompt, system, config)
    prompt_chars = len(prompt) + (len(system) if system else 0)
    log.info("generate num_ctx=%d prompt_chars=%d model=%s", num_ctx, prompt_chars, config.model)
    payload = {
        "model": config.model,
        "prompt": prompt,
        "stream": progress_callback is not None,
        "think": False,
        "keep_alive": config.keep_alive,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.num_predict,
            "num_ctx": num_ctx,
        },
    }
    if system:
        payload["system"] = system
    if format is not None:
        payload["format"] = format

    # Timeout: 60s for model load + 600s for generation
    timeout = httpx.Timeout(connect=10.0, read=config.read_timeout_ms / 1000, write=10.0, pool=10.0)
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
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                piece = data.get("response") or ""
                if piece:
                    pieces.append(piece)
                    chunks += 1
                    chars += len(piece)

                done = bool(data.get("done"))
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
            _emit_progress(progress_callback, {
                "event": "error",
                "active": False,
                "generated_chars": chars,
                "chunks": chunks,
                "elapsed_seconds": round(max(0.001, time.monotonic() - started), 2),
                "error": "stream ended before done",
            })
            raise RuntimeError("Ollama stream ended before done")
        return "".join(pieces)

    resp = _client().post(
        "/api/generate",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"]


EMBED_MODEL = DEFAULT_EMBEDDING_MODEL


def embedding_model() -> str:
    return load_embedding_config().model


def embed(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    """Get embedding vectors via Ollama /api/embed."""
    resp = _client().post(
        "/api/embed",
        json={"model": model or embedding_model(), "input": texts},
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def unload_model() -> None:
    """Explicitly unload model to free memory."""
    try:
        _client().post(
            "/api/generate",
            json={"model": ingest_model(), "keep_alive": 0, "prompt": ""},
            timeout=10,
        )
    except Exception:
        pass


TRIAGE_SYSTEM_PROMPT = """\
You are a knowledge wiki triage engine. Analyze raw session data and decide \
what wiki pages to create or update. Do NOT generate page content — only output a structured plan.

Rules:
- 1 entity = 1 page
- Output valid JSON array only (no markdown fences, no explanation)
- For new pages: choose folder and filename in kebab-case (English)
- For updates: reference the existing page ID in a field named "filename"
- Every update object MUST use "filename". Never emit a "page_id" field
- If the target page is not listed in the catalog, use create, not update
- Skip ephemeral conversation, greetings, and filler
- Include brief summary of what knowledge each page should contain
- Include keywords for finding related existing pages

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
    "summary": "What new information to add"
  }
]

WRONG output (do NOT do these):
- Bare keyword list: ["keyword1", "keyword2"]   ← This is a list of strings, not operations
- Single object: {"type": "create", ...}        ← Must be wrapped in an array
- Code fences around the JSON                   ← Output raw JSON only

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
"""
