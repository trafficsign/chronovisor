#!/usr/bin/env python3.14
"""Drive the Chronovisor oMLX cutover profile through ``build_llm_runtime``.

The harness intentionally targets the dedicated loopback service on port
18125.  It builds an in-memory config, so the production
``~/.chronovisor/config.toml`` is never read or written.  Model names remain
CLI overrides for fixture tests, while the defaults match the one-model
Qwen3.8 Flash Next deployment.
"""

from __future__ import annotations

import argparse
import json
import time
import tomllib

from chronovisor.core.llm_config import build_llm_runtime, parse_llm_config
from chronovisor.core.llm_runtime import (
    EmbeddingPurpose,
    EmbeddingRequest,
    MessageGenerationRequest,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)

NORMAL_PAGE = SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.NORMAL)

DEFAULT_OMLX_ENDPOINT = "http://127.0.0.1:18125/v1"
# oMLX discovers the child directory name as the API model ID.  The stable
# layout is ~/.omlx/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp.
DEFAULT_GENERATION_MODEL = "Qwen3.8-Flash-Next-oQ4e-mtp"

GENERATION_ROLES = (
    "ingest.generation",
    "librarian.review",
    "lint.tag_repair",
    "classification.primary",
    "classification.challenger",
    "classification.tie_break",
)
EMBEDDING_ROLES = ("classification.embedding",)


def build_config(args: argparse.Namespace) -> dict[str, object]:
    endpoint = getattr(args, "endpoint", DEFAULT_OMLX_ENDPOINT)
    roles = {
        role: ("generation", args.generation_model) for role in GENERATION_ROLES
    }
    roles["classification.challenger"] = ("generation", args.challenger_model)
    roles["classification.tie_break"] = ("generation", args.tie_break_model)
    roles["recall.gate"] = ("generation", args.gate_model)
    roles["recall.processor.judge"] = ("generation", args.gate_model)
    for role in EMBEDDING_ROLES:
        roles[role] = ("embedding", args.embedding_model)
    role_tables = "\n".join(
        (
            f'[llm.roles.{json.dumps(role)}]\n'
            f"capability = {json.dumps(capability)}\n"
            f'provider = "omlx"\n'
            f"model = {json.dumps(model)}\n"
        )
        for role, (capability, model) in roles.items()
    )
    payload = f"""\
[llm.providers.omlx]
kind = "omlx"
endpoint = {json.dumps(endpoint)}

{role_tables}
"""
    return tomllib.loads(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=DEFAULT_OMLX_ENDPOINT)
    parser.add_argument("--generation-model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--gate-model", default="Ornith-1.5-9B-MLX-4bit")
    parser.add_argument("--challenger-model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--tie-break-model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--embedding-model", default="bge-m3-mlx-fp16")
    parser.add_argument(
        "--num-ctx",
        default=114_688,
        type=int,
        help="generation context sent by the harness (default: 114688)",
    )
    parser.add_argument(
        "--gate-num-ctx",
        default=2_048,
        type=int,
        help="low-latency gate context (default: 2048)",
    )
    parser.add_argument("--max-tokens", default=128, type=int)
    args = parser.parse_args()

    config = parse_llm_config(build_config(args))
    runtime = build_llm_runtime(config)

    failures = 0
    for role in GENERATION_ROLES:
        request = MessageGenerationRequest(
            messages=(
                {"role": "user", "content": "3から40までの素数をすべて列挙してください。"},
            ),
            format=None,
            source=NORMAL_PAGE,
            num_ctx=args.num_ctx,
            max_output_tokens=args.max_tokens,
            keep_alive="0",
            timeout_ms=120_000,
            max_output_chars=4000,
            temperature=0,
            seed=0,
            think=False,
        )
        print(f"[gen] {role}", flush=True)
        try:
            result = runtime.generate(role, request)
        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)
            failures += 1
            continue
        print(
            f"  {result.provider}/{result.model}: {len(result.content)} chars "
            f"tok={result.usage.output_tokens} tt={result.metadata.get('total_time')}",
            flush=True,
        )
    request = MessageGenerationRequest(
        messages=(
            {"role": "user", "content": "Reply YES or NO only: does this mention memory?"},
        ),
        format=None,
        source=NORMAL_PAGE,
        num_ctx=args.gate_num_ctx,
        max_output_tokens=16,
        keep_alive="0",
        timeout_ms=15_000,
        max_output_chars=200,
        temperature=0,
        seed=0,
        think=False,
    )
    print("[gen] recall.gate (low-latency)", flush=True)
    try:
        started = time.monotonic()
        result = runtime.generate("recall.gate", request)
        elapsed = time.monotonic() - started
        print(
            f"  {result.provider}/{result.model}: {result.content!r} "
            f"tt={result.metadata.get('total_time')} wall={elapsed:.3f}s",
            flush=True,
        )
        if elapsed > 1.5:
            print("  ERROR: recall.gate exceeded 1.5s", flush=True)
            failures += 1
    except Exception as exc:
        print(f"  ERROR: {exc}", flush=True)
        failures += 1

    for role in EMBEDDING_ROLES:
        request = EmbeddingRequest(
            texts=("Chronovisor provides local memory retrieval.", "cooking recipe"),
            source=NORMAL_PAGE,
            timeout_ms=60_000,
            purpose=EmbeddingPurpose.DOCUMENT,
        )
        print(f"[embed] {role}", flush=True)
        try:
            result = runtime.embed(role, request)
        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)
            failures += 1
            continue
        dims = len(result.vectors[0]) if result.vectors else 0
        print(f"  {result.provider}/{result.model}: {len(result.vectors)} x {dims}d", flush=True)

    print(f"RESULT: {'FAIL' if failures else 'PASS'} ({failures} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
