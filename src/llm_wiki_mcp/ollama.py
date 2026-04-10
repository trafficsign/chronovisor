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
    """Call Ollama generate API with keep_alive=0 (unload after use)."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0,
        "options": {
            "temperature": 0.3,
            "num_predict": 8192,
        },
    }
    if system:
        payload["system"] = system

    resp = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["response"]


INGEST_SYSTEM_PROMPT = """\
You are a knowledge wiki structuring engine. Your job is to extract knowledge \
from raw session data and produce structured wiki pages.

Rules:
- 1 entity = 1 page
- Filename: kebab-case (English), e.g. studio-display-xdr.md
- Frontmatter: only title and updated
- Cross-references: use [[wiki-link]] notation
- Write content in Japanese
- Focus on facts, decisions, and technical knowledge
- Skip ephemeral conversation, greetings, and filler
- If the content relates to an existing page, output an UPDATE instruction instead of a new page

Output format:
For each page, output:
```
=== NEW PAGE: filename.md ===
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
