"""Ollama API client for Ingest/Lint operations."""

import threading
import time

import httpx

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma4:26b"

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


def generate(prompt: str, system: str | None = None) -> str:
    """Call Ollama generate API.

    Uses keep_alive="5m" to keep model loaded for 5 minutes after use.
    This avoids cold-start on consecutive calls (e.g. Ingest then Lint)
    while still freeing memory after a reasonable idle period.
    """
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "5m",
        "options": {
            "temperature": 0.3,
            "num_predict": 8192,
        },
    }
    if system:
        payload["system"] = system

    # Timeout: 60s for model load + 600s for generation
    resp = _client().post(
        "/api/generate",
        json=payload,
        timeout=httpx.Timeout(connect=10.0, read=660.0, write=10.0, pool=10.0),
    )
    resp.raise_for_status()
    return resp.json()["response"]


EMBED_MODEL = "nomic-embed-text"


def embed(texts: list[str]) -> list[list[float]]:
    """Get embedding vectors via Ollama /api/embed."""
    resp = _client().post(
        "/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def unload_model() -> None:
    """Explicitly unload model to free memory."""
    try:
        _client().post(
            "/api/generate",
            json={"model": MODEL, "keep_alive": 0, "prompt": ""},
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
- For updates: reference existing page ID
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
- Frontmatter: only title and updated
- Cross-references: use [[wiki-link]] notation (page ID only, no folder path)
- Write content in Japanese
- Focus on facts, decisions, and technical knowledge
- Use the provided context for cross-references but do not duplicate existing content

Output exactly one page block:
=== NEW PAGE: {filename} ===
---
title: Page Title
updated: YYYY-MM-DD
---

Page content here with [[wiki-links]] to related topics.

=== END PAGE ===
"""

UPDATE_SYSTEM_PROMPT = """\
You are a knowledge wiki structuring engine. Append content to an EXISTING wiki page.

Rules:
- DO NOT output frontmatter (no `---`, no title:, no updated: lines). The existing page already has frontmatter; your output is appended to its body.
- DO NOT repeat content that already exists on the page (it is provided in context).
- Output ONLY the new section(s) to add — Japanese prose, headings, lists, code, etc.
- Cross-references: use [[wiki-link]] notation (page ID only, no folder path)
- Focus on facts, decisions, and technical knowledge

Output exactly one block:
=== UPDATE PAGE: {filename} ===
New section(s) here. Markdown body only — NO frontmatter delimiters.

=== END PAGE ===
"""
