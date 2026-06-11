"""Embedding helpers via the configured Ollama embedding model.

Thin layer over :func:`ollama.embed` adding a disk cache keyed by
``sha256(model|text)`` plus cosine similarity. Used by the tag
deduplication path (existing-tag preference at >= 0.80 similarity) and
intended to be reused by related-page suggestion features.

Single-process safe. The cache is content-addressed so repeated runs
(ingest, lint, backfill dry-run, distribution report) don't re-pay
Ollama latency for the same input.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from llm_wiki_mcp.ollama import embed as _ollama_embed
from llm_wiki_mcp.runtime_config import load_embedding_config
from llm_wiki_mcp.wiki import WIKI_ROOT


# Cache layout: ~/.wiki/.index/embeddings/<first-2-hex>/<hash>.json
# Sharding by the first byte keeps any single directory below ~few-hundred
# files even after thousands of unique texts, which matters because some
# filesystems (APFS, ext4) get noticeably slower on dirs with >10k entries.
_CACHE_DIR = WIKI_ROOT / ".index" / "embeddings"


def _cache_path(text: str, model: str | None = None) -> Path:
    model_id = model or load_embedding_config().model
    h = hashlib.sha256(f"{model_id}|{text}".encode("utf-8")).hexdigest()
    return _CACHE_DIR / h[:2] / f"{h}.json"


def _read_cached(path: Path) -> list[float] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, list) or not all(isinstance(v, (int, float)) for v in data):
        return None
    return [float(v) for v in data]


def _write_cached(path: Path, vec: list[float]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(vec))
    except OSError:
        # Cache is best-effort — losing a write doesn't break correctness,
        # and we don't want a disk-full event to abort a tag lookup.
        pass


def embed_text(text: str) -> list[float]:
    """Embed a single text via Ollama, caching to disk.

    Raises whatever ``_ollama_embed`` raises when Ollama is unreachable;
    callers that need a soft-fail (e.g. tag dedup falling back to literal
    string match) should catch the exception themselves.
    """
    path = _cache_path(text)
    cached = _read_cached(path)
    if cached is not None:
        return cached
    vec = _ollama_embed([text])[0]
    _write_cached(path, vec)
    return vec


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed. Only texts missing from the cache hit Ollama.

    Order matches ``texts`` exactly (including duplicates — both copies
    return the same vector via the cache).
    """
    if not texts:
        return []
    cached: list[list[float] | None] = []
    pending: list[tuple[int, str]] = []
    for i, t in enumerate(texts):
        path = _cache_path(t)
        hit = _read_cached(path)
        if hit is not None:
            cached.append(hit)
        else:
            cached.append(None)
            pending.append((i, t))

    if pending:
        # Keep the request stable (no dedup yet) so the index alignment is
        # trivial. Ollama batches efficiently enough that a few duplicates
        # in one call are cheaper than the bookkeeping to dedupe-then-fanout.
        vecs = _ollama_embed([t for _, t in pending])
        for (i, t), v in zip(pending, vecs):
            cached[i] = v
            _write_cached(_cache_path(t), v)

    # All slots filled by this point.
    return [v for v in cached if v is not None]  # type: ignore[return-value]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 for zero vectors."""
    if not a or not b:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def most_similar(
    query: str,
    candidates: list[str],
    threshold: float = 0.0,
) -> tuple[str, float] | None:
    """Best-match candidate by cosine similarity, gated by ``threshold``.

    Returns ``(candidate, similarity)`` for the top candidate iff its
    similarity is ``>= threshold``. Returns ``None`` if there are no
    candidates, or if the best similarity is below the threshold —
    callers can use the latter to mean "no existing tag is close enough,
    treat the query as a new tag".
    """
    if not candidates:
        return None
    qv = embed_text(query)
    cvs = embed_texts(candidates)
    best: tuple[str, float] | None = None
    for cand, cv in zip(candidates, cvs):
        sim = cosine(qv, cv)
        if best is None or sim > best[1]:
            best = (cand, sim)
    if best is None or best[1] < threshold:
        return None
    return best
