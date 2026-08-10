"""Isolated fixed-role embedding worker for the preemptible research lane."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import Any

from chronovisor.core import llm_config, llm_runtime, ollama
from chronovisor.recall.classification import ClassificationError

SCHEMA = "chronovisor.classification-embedding-worker.v2"
RUNTIME_ROLE = "classification.embedding"
_OVERRIDES = frozenset({"model", "provider", "runtime_role", "route_identity"})
_FIELDS = frozenset(
    {
        "schema",
        "texts",
        "source_data_class",
        "source_sensitivity",
        "embedding_purpose",
        "read_timeout_ms",
    }
)


def route_identity(
    route: llm_runtime.ResolvedEmbeddingRoute,
) -> dict[str, str | None]:
    digest: str | None = None
    if (
        route.provider == "ollama"
        and route.location is llm_runtime.RouteLocation.LOCAL
    ):
        digest = ollama.model_digests([route.model]).get(route.model, "")
        if not digest:
            raise llm_runtime.BackendContractError(
                RUNTIME_ROLE, "embedding", "model_digest_missing"
            )
    return {
        "role": route.role,
        "provider": route.provider,
        "model": route.model,
        "location": route.location.value,
        "model_digest": digest,
    }


def resolved_route_identity() -> dict[str, str | None]:
    return route_identity(
        llm_config.load_default_llm_runtime().resolve_embedding(RUNTIME_ROLE)
    )


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    if _OVERRIDES.intersection(payload):
        raise ClassificationError("embedding worker route overrides are forbidden")
    if set(payload) != _FIELDS:
        raise ClassificationError("embedding worker payload fields are invalid")
    if payload.get("schema") != SCHEMA:
        raise ClassificationError("unsupported embedding worker schema")
    texts = payload.get("texts")
    if (
        not isinstance(texts, list)
        or not texts
        or not all(isinstance(value, str) and value.strip() for value in texts)
    ):
        raise ClassificationError("embedding worker requires non-empty texts")
    try:
        source = llm_runtime.SourceDataClassification(
            llm_runtime.SourceDataClass(payload.get("source_data_class")),
            llm_runtime.SourceSensitivity(payload.get("source_sensitivity")),
        )
        purpose = llm_runtime.EmbeddingPurpose(payload.get("embedding_purpose"))
    except (TypeError, ValueError):
        raise ClassificationError("embedding worker source contract is invalid") from None
    read_timeout_ms = payload.get("read_timeout_ms")
    if (
        not isinstance(read_timeout_ms, int)
        or isinstance(read_timeout_ms, bool)
        or read_timeout_ms < 1
    ):
        raise ClassificationError("embedding worker timeout is invalid")
    runtime = llm_config.load_default_llm_runtime()
    route = route_identity(runtime.resolve_embedding(RUNTIME_ROLE))
    result = runtime.embed(
        RUNTIME_ROLE,
        llm_runtime.EmbeddingRequest(
            tuple(texts),
            source,
            timeout_ms=read_timeout_ms,
            purpose=purpose,
        ),
    )
    vectors = [list(vector) for vector in result.vectors]
    if len(vectors) != len(texts):
        raise ClassificationError("embedding worker returned wrong vector count")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or not dimensions or next(iter(dimensions)) < 1:
        raise ClassificationError("embedding worker returned invalid dimensions")
    return {
        "schema": SCHEMA,
        "model": route["model"],
        "route_identity": route,
        "dimensions": next(iter(dimensions)),
        "vectors": vectors,
    }


def main() -> None:
    payload = json.load(sys.stdin)
    print(json.dumps(run(payload), ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
