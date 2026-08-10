#!/usr/bin/env python3
"""Opt-in real Ollama smoke check against an isolated Chronovisor root."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path


def _write_config(path: Path, generation_model: str, embedding_model: str) -> None:
    payload = f'''\
[llm.providers.local]
kind = "ollama"

[llm.roles."ingest.generation"]
capability = "generation"
provider = "local"
model = {json.dumps(generation_model)}

[llm.roles."knowledge.embedding"]
capability = "embedding"
provider = "local"
model = {json.dumps(embedding_model)}
'''
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _main() -> dict[str, object]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-model", default="ornith:9b-q4_K_M")
    parser.add_argument("--embedding-model", default="bge-m3:latest")
    args = parser.parse_args()

    configured_root = os.environ.get("CHRONOVISOR_ROOT", "").strip()
    if not configured_root:
        raise ValueError("isolated_root_required")
    root = Path(configured_root).expanduser().resolve(strict=True)
    if not root.is_dir() or root == (Path.home() / ".chronovisor").resolve():
        raise ValueError("isolated_root_required")
    if next(root.iterdir(), None) is not None:
        raise ValueError("isolated_root_not_empty")
    _write_config(root / "config.toml", args.generation_model, args.embedding_model)

    from chronovisor.core.embedding import embed_text
    from chronovisor.core.llm_config import load_default_llm_runtime
    from chronovisor.core.ollama import runtime_generate

    runtime = load_default_llm_runtime()
    generation_route = runtime.resolve_generation("ingest.generation")
    embedding_route = runtime.resolve_embedding("knowledge.embedding")
    for route, model in (
        (generation_route, args.generation_model),
        (embedding_route, args.embedding_model),
    ):
        if (
            route.provider != "ollama"
            or route.location.value != "local"
            or route.model != model
        ):
            raise ValueError("route_mismatch")

    vector = embed_text("Chronovisor local Ollama end-to-end probe.")
    if not vector or not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in vector
    ):
        raise ValueError("embedding_invalid")
    generated = runtime_generate(
        "Return exactly: OK",
        runtime_role="ingest.generation",
        source_data_class="system",
        source_sensitivity="normal",
        num_ctx=2048,
        num_predict=8,
        keep_alive="0",
        read_timeout_ms=120_000,
        temperature=0,
        seed=0,
    )
    if not generated.content.strip():
        raise ValueError("generation_empty")
    return {
        "status": "ok",
        "embedding_dimensions": len(vector),
        "generation_chars": len(generated.content),
    }


if __name__ == "__main__":
    try:
        result = _main()
    except Exception as error:
        category = getattr(error, "category", "e2e_failed")
        if not isinstance(category, str) or re.fullmatch(r"[a-z0-9_]{1,64}", category) is None:
            category = "e2e_failed"
        print(
            json.dumps(
                {"status": "error", "category": category, "type": type(error).__name__},
                separators=(",", ":"),
            )
        )
        raise SystemExit(1) from None
    print(json.dumps(result, separators=(",", ":")))
