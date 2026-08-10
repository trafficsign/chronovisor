"""Shared provider-neutral embedding helpers with a route-bound disk cache."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from chronovisor.core import ollama
from chronovisor.core.llm_config import load_default_llm_runtime
from chronovisor.core.llm_runtime import (
    BackendContractError,
    EmbeddingPurpose,
    EmbeddingRequest,
    LLMRuntime,
    ResolvedEmbeddingRoute,
    RouteLocation,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)
from chronovisor.core.store import CHRONOVISOR_ROOT

EMBEDDING_ROLE = "knowledge.embedding"
_SOURCE = SourceDataClassification(
    SourceDataClass.DERIVED_SNIPPET,
    SourceSensitivity.HIGH,
)

# Cache layout: ~/.chronovisor/.index/embeddings/<first-2-hex>/<hash>.json
_CACHE_DIR = CHRONOVISOR_ROOT / ".index" / "embeddings"


def _route_identity(route: ResolvedEmbeddingRoute) -> dict[str, str | None]:
    digest: str | None = None
    if route.provider == "ollama" and route.location is RouteLocation.LOCAL:
        digest = ollama.model_digests([route.model]).get(route.model, "")
        if not digest:
            raise BackendContractError(
                EMBEDDING_ROLE, "embedding", "model_digest_missing"
            )
    return {
        "role": route.role,
        "provider": route.provider,
        "model": route.model,
        "location": route.location.value,
        "model_digest": digest,
    }


def _resolved_runtime() -> tuple[LLMRuntime, dict[str, str | None]]:
    runtime = load_default_llm_runtime()
    return runtime, _route_identity(runtime.resolve_embedding(EMBEDDING_ROLE))


def _cache_path(
    text: str,
    route: dict[str, str | None],
    purpose: EmbeddingPurpose = EmbeddingPurpose.DOCUMENT,
) -> Path:
    material = json.dumps(
        {
            "version": 2,
            **route,
            "purpose": purpose.value,
            "text": text,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(material.encode()).hexdigest()
    return _CACHE_DIR / digest[:2] / f"{digest}.json"


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
        # Cache is best-effort; an unwritable cache must not block inference.
        pass


def _embed_texts(
    texts: list[str],
    purpose: EmbeddingPurpose,
    runtime: LLMRuntime,
    route: dict[str, str | None],
) -> tuple[list[list[float]], dict[str, str | None]]:
    cached: list[list[float] | None] = []
    pending: list[tuple[int, str]] = []
    for index, text in enumerate(texts):
        hit = _read_cached(_cache_path(text, route, purpose))
        cached.append(hit)
        if hit is None:
            pending.append((index, text))

    if pending:
        result = runtime.embed(
            EMBEDDING_ROLE,
            EmbeddingRequest(
                tuple(text for _, text in pending),
                _SOURCE,
                purpose=purpose,
            ),
        )
        for (index, text), vector in zip(pending, result.vectors, strict=True):
            value = list(vector)
            cached[index] = value
            _write_cached(_cache_path(text, route, purpose), value)
    return [vector for vector in cached if vector is not None], route


def embed_text(text: str) -> list[float]:
    """Embed one document snippet through the fixed knowledge role."""

    runtime, route = _resolved_runtime()
    return _embed_texts([text], EmbeddingPurpose.DOCUMENT, runtime, route)[0][0]


def embed_texts(
    texts: list[str],
    *,
    return_route: bool = False,
) -> list[list[float]] | tuple[list[list[float]], dict[str, str | None]]:
    """Batch-embed document snippets, preserving input order and duplicates."""

    if not texts and not return_route:
        return []
    runtime, route = _resolved_runtime()
    result = _embed_texts(texts, EmbeddingPurpose.DOCUMENT, runtime, route)
    return result if return_route else result[0]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 for zero vectors."""

    if not a or not b:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
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
    """Return the best candidate when its cosine score meets ``threshold``."""

    if not candidates:
        return None
    runtime, route = _resolved_runtime()
    query_vector = _embed_texts([query], EmbeddingPurpose.QUERY, runtime, route)[0][0]
    candidate_vectors = _embed_texts(
        candidates, EmbeddingPurpose.DOCUMENT, runtime, route
    )[0]
    best: tuple[str, float] | None = None
    for candidate, vector in zip(candidates, candidate_vectors, strict=True):
        similarity = cosine(query_vector, vector)
        if best is None or similarity > best[1]:
            best = (candidate, similarity)
    return None if best is None or best[1] < threshold else best
