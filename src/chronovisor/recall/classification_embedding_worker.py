"""Isolated Ollama embedding worker for the preemptible research lane."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import Any

from chronovisor.core import ollama
from chronovisor.recall.classification import ClassificationError

SCHEMA = "chronovisor.classification-embedding-worker.v1"


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != SCHEMA:
        raise ClassificationError("unsupported embedding worker schema")
    texts = payload.get("texts")
    if (
        not isinstance(texts, list)
        or not texts
        or not all(isinstance(value, str) and value.strip() for value in texts)
    ):
        raise ClassificationError("embedding worker requires non-empty texts")
    model = str(payload.get("model") or ollama.embedding_model())
    vectors = ollama.embed(
        texts,
        model=model,
        read_timeout_ms=int(payload.get("read_timeout_ms") or 600_000),
    )
    if len(vectors) != len(texts):
        raise ClassificationError("embedding worker returned wrong vector count")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or not dimensions or next(iter(dimensions)) < 1:
        raise ClassificationError("embedding worker returned invalid dimensions")
    return {
        "schema": SCHEMA,
        "model": model,
        "dimensions": next(iter(dimensions)),
        "vectors": vectors,
    }


def main() -> None:
    payload = json.load(sys.stdin)
    print(json.dumps(run(payload), ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
