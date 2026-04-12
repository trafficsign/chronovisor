"""Ollama API client for Ingest/Lint operations."""

import httpx
import time

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma4:26b"

# Health check cache
_health_cache: dict = {"status": None, "checked_at": 0.0}
HEALTH_CACHE_TTL = 900  # 15 minutes on failure


def is_available() -> bool:
    """Check if Ollama is running (cached on failure)."""
    now = time.time()

    # If last check failed, use cache for TTL
    if _health_cache["status"] is False:
        if now - _health_cache["checked_at"] < HEALTH_CACHE_TTL:
            return False

    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3)
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
    resp = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=httpx.Timeout(connect=10.0, read=660.0, write=10.0, pool=10.0),
    )
    resp.raise_for_status()
    return resp.json()["response"]


EMBED_MODEL = "nomic-embed-text"


def embed(texts: list[str]) -> list[list[float]]:
    """Get embedding vectors via Ollama /api/embed."""
    resp = httpx.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def unload_model() -> None:
    """Explicitly unload model to free memory."""
    try:
        httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL, "keep_alive": 0, "prompt": ""},
            timeout=10,
        )
    except Exception:
        pass


INGEST_SYSTEM_PROMPT = """\
You are a knowledge wiki structuring engine. Your job is to extract knowledge \
from raw session data and produce structured wiki pages.

Rules:
- 1 entity = 1 page
- Filename: folder/kebab-case.md (English). Choose an appropriate folder category.
- Frontmatter: only title and updated
- Cross-references: use [[wiki-link]] notation (page ID only, no folder path)
- Write content in Japanese
- Focus on facts, decisions, and technical knowledge
- Skip ephemeral conversation, greetings, and filler
- If the content relates to an existing page, output an UPDATE instruction instead of a new page

Folder guidelines:
- Choose the folder based on the page content and topic
- Use short, broad category names in English kebab-case
- Examples: career/, project/, ai/, auto-industry/, hardware/, cad/, car-spec/, engineering/
- Create new folders as needed for new topics
- Keep folder depth to 1 level (no nested subfolders)
- If existing related pages are in a specific folder, prefer the same folder for consistency

Output format:
For each page, output:
```
=== NEW PAGE: folder/filename.md ===
---
title: Page Title
updated: YYYY-MM-DD
---

Page content here with [[wiki-links]] to related topics.

=== END PAGE ===
```

For updates to existing pages:
```
=== UPDATE PAGE: existing-filename.md ===
Section or content to add/modify.
=== END PAGE ===
```
"""

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
"""

GENERATE_SYSTEM_PROMPT = """\
You are a knowledge wiki structuring engine. Generate content for a SINGLE wiki page.

Rules:
- Frontmatter: only title and updated
- Cross-references: use [[wiki-link]] notation (page ID only, no folder path)
- Write content in Japanese
- Focus on facts, decisions, and technical knowledge
- Use the provided context for cross-references but do not duplicate existing content

Output exactly one page block:
=== {type} PAGE: {filename} ===
---
title: Page Title
updated: YYYY-MM-DD
---

Page content here with [[wiki-links]] to related topics.

=== END PAGE ===
"""
