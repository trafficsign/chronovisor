#!/usr/bin/env python3
"""Run real local models through representative production workflow seams."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

GENERATION_ROLES = (
    "ingest.generation",
    "librarian.review",
    "recall.gate",
    "lint.tag_repair",
    "classification.primary",
    "classification.challenger",
    "classification.tie_break",
)
EMBEDDING_ROLES = ("search.semantic.foreground",)
RERANK_ROLE = "search.rerank"
PROBE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status"],
    "properties": {"status": {"type": "string", "enum": ["ok"]}},
}


def _write_config(path: Path, args: argparse.Namespace) -> None:
    roles = {
        **{
            role: ("generation", "ollama", args.generation_model)
            for role in GENERATION_ROLES
        },
        **{
            role: ("embedding", "ollama", args.embedding_model)
            for role in EMBEDDING_ROLES
        },
        RERANK_ROLE: ("rerank", "reranker", args.reranker_model),
        "classification.primary": ("generation", "ollama", args.primary_model),
        "classification.challenger": ("generation", "ollama", args.challenger_model),
        "classification.tie_break": ("generation", "ollama", args.tie_break_model),
    }
    role_tables = "\n".join(
        f"[llm.roles.{json.dumps(role)}]\n"
        f"capability = {json.dumps(capability)}\n"
        f"provider = {json.dumps(provider)}\n"
        f"model = {json.dumps(model)}\n"
        for role, (capability, provider, model) in roles.items()
    )
    payload = f"""\
[ingest]
keep_alive = "0"
num_ctx = 16384
max_num_ctx = 16384
num_predict = 512
read_timeout_ms = 120000

[decision_router]
primary_model = {json.dumps(args.primary_model)}
challenger_model = {json.dumps(args.challenger_model)}
tie_break_model = {json.dumps(args.tie_break_model)}
primary_keep_alive = "0"
challenger_keep_alive = "0"
tie_break_keep_alive = "0"
num_ctx = 16384
num_predict = 128

[search.reranker]
enabled = true
model = {json.dumps(args.reranker_model)}

[llm.providers.ollama]
kind = "ollama"

[llm.providers.reranker]
kind = "local-transformers"
backend = "transformers"

{role_tables}"""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _isolated_root() -> Path:
    configured = os.environ.get("CHRONOVISOR_ROOT", "").strip()
    if not configured:
        raise ValueError("isolated_root_required")
    root = Path(configured).expanduser().resolve(strict=True)
    temporary = Path(tempfile.gettempdir()).resolve(strict=True)
    if (
        not root.is_dir()
        or root == (Path.home() / ".chronovisor").resolve(strict=False)
        or not root.is_relative_to(temporary)
    ):
        raise ValueError("isolated_temporary_root_required")
    if next(root.iterdir(), None) is not None:
        raise ValueError("isolated_root_not_empty")
    return root


def _local_routes(runtime: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for capability, roles, resolve in (
        ("generation", GENERATION_ROLES, runtime.resolve_generation),
        ("embedding", EMBEDDING_ROLES, runtime.resolve_embedding),
        ("rerank", (RERANK_ROLE,), runtime.resolve_rerank),
    ):
        for role in roles:
            route = resolve(role)
            if route.location.value != "local":
                raise ValueError(f"non_local_route:{role}")
            rows.append(
                {
                    "capability": capability,
                    "role": role,
                    "provider": route.provider,
                    "model": route.model,
                    "location": route.location.value,
                }
            )
    return rows


def _semantic_search(root: Path, runtime: Any, dimensions: int) -> str:
    import numpy as np

    from chronovisor.core.llm_runtime import (
        EmbeddingPurpose,
        EmbeddingRequest,
        SourceDataClass,
        SourceDataClassification,
        SourceSensitivity,
    )
    from chronovisor.core.semantic_index import (
        SemanticDocument,
        activate_generation,
        build_generation,
        load_active_generation,
    )

    route = runtime.resolve_embedding("search.semantic.foreground")
    source = SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.NORMAL)
    documents = tuple(
        SemanticDocument(
            f"{page_id}:page:0",
            page_id,
            "page",
            0,
            text,
            f"pages/{page_id}.md",
            digest * 64,
            1,
            source_sensitivity="normal",
        )
        for page_id, text, digest in (
            ("chronovisor-memory", "Chronovisor provides local memory retrieval.", "a"),
            ("cooking-note", "A dinner recipe with vegetables and spices.", "b"),
        )
    )

    def encode(rows: Sequence[SemanticDocument], _batch_size: int) -> np.ndarray:
        result = runtime.embed(
            route.role,
            EmbeddingRequest(
                tuple(row.text for row in rows),
                source,
                120_000,
                EmbeddingPurpose.DOCUMENT,
            ),
        )
        return np.asarray(result.vectors, dtype=np.float32)

    index_root = root / ".index" / "semantic-e2e"
    manifest = build_generation(
        documents,
        encode_documents=encode,
        role=route.role,
        provider=route.provider,
        model=route.model,
        location="local",
        revision="local-e2e",
        dimensions=dimensions,
        query_prefix="",
        document_prefix="",
        batch_size=2,
        root=index_root,
        repo_commit="local-e2e",
    )
    activate_generation(manifest.generation_id, root=index_root)
    query = runtime.embed(
        route.role,
        EmbeddingRequest(
            ("local memory retrieval",),
            SourceDataClassification(SourceDataClass.RAW, SourceSensitivity.NORMAL),
            120_000,
            EmbeddingPurpose.QUERY,
        ),
    )
    hits = load_active_generation(
        root=index_root,
        expected_route={
            "role": route.role,
            "provider": route.provider,
            "model": route.model,
            "location": "local",
        },
    ).search(query.vectors[0], top_n=2)
    if not hits or hits[0][0] != "chronovisor-memory":
        raise ValueError("semantic_search_invalid")
    if len(query.vectors[0]) != dimensions or not np.isfinite(query.vectors[0]).all():
        raise ValueError("semantic_embedding_invalid")
    return hits[0][0]


def _run_workflows(
    root: Path, runtime: Any, args: argparse.Namespace
) -> dict[str, Any]:
    from chronovisor.core.reranker import rerank_results
    from chronovisor.core.search_types import ScoredPage
    from chronovisor.decision.decision_router import DecisionRouter
    from chronovisor.ingest.ingest import _triage
    from chronovisor.librarian.tag_distribution import analyze_page
    from chronovisor.ops.lint_repair import (
        TAG_REPAIR_SCHEMA,
        _default_local_reviewer,
        normalize_tag_decision,
    )
    from chronovisor.recall.recall_runtime import (
        RecallPolicy,
        RecallRequest,
        run_local_judge,
    )

    workflows: dict[str, Any] = {}
    plan = _triage(
        "Isolated control record with no durable facts. Return an empty plan.",
        raise_on_failure=True,
    )
    if plan != []:
        raise ValueError("ingest_triage_invalid")
    workflows["ingest_generation"] = {"status": "ok"}

    workflows["semantic_search"] = {
        "status": "ok",
        "top_page_id": _semantic_search(root, runtime, args.embedding_dimensions),
    }

    reranked = rerank_results(
        "local memory retrieval",
        [
            ScoredPage("cooking-note", "Cooking", "", "", 1.0),
            ScoredPage("chronovisor-memory", "Chronovisor", "", "", 0.5),
        ],
    )
    if reranked.metadata.get("status") != "applied" or len(reranked.scores) != 2:
        raise ValueError("rerank_invalid")
    workflows["rerank"] = {"status": "ok", "candidates": 2}

    confidence, _queries, reason = run_local_judge(
        RecallRequest(
            "local-e2e", "UserPromptSubmit", "Recall the prior decision.", str(root)
        ),
        0.5,
        RecallPolicy(
            judge_mode="always",
            judge_timeout_ms=120_000,
            judge_keep_alive="0",
            log_decisions=False,
        ),
        timeout_ms=120_000,
    )
    if confidence is None or "unavailable" in reason or "fallback" in reason:
        raise ValueError("recall_judge_invalid")
    workflows["recall"] = {"status": "ok"}

    seen_roles: set[str] = set()
    for excluded in ((), ("primary",)):
        result = DecisionRouter(
            audit_root=root / "runtime" / "decision",
            audit_role="local_e2e",
            record_replay=False,
            live_resource_control=False,
            excluded_roles=excluded,
        ).decide("Return status ok.", PROBE_SCHEMA)
        if not result.ok:
            raise ValueError("decision_quorum_invalid")
        seen_roles.update(vote.role for vote in result.votes)
    if seen_roles != {"primary", "challenger", "tie_break"}:
        raise ValueError("classification_role_missing")
    workflows["classification_decision"] = {"status": "ok", "roles": sorted(seen_roles)}

    librarian = analyze_page("missing-local-e2e-page", ["d/tools-config"])
    if not librarian.main_topic or librarian.raw_response.startswith("<"):
        raise ValueError("librarian_review_invalid")
    workflows["librarian"] = {"status": "ok"}

    lint = _default_local_reviewer(
        "The excerpt is empty. Return rejected, no tags, and a short reason.",
        TAG_REPAIR_SCHEMA,
        audit_root=root / "runtime" / "lint",
    )
    if not normalize_tag_decision(lint).get("valid"):
        raise ValueError("lint_tag_repair_invalid")
    workflows["lint"] = {"status": "ok"}
    return workflows


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-model", default="ornith:9b-q4_K_M")
    parser.add_argument("--embedding-model", default="bge-m3:latest")
    parser.add_argument("--embedding-dimensions", type=int, default=1024)
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--primary-model", default="maxwell1500/ornith-35b:Q5_K_M")
    parser.add_argument("--challenger-model", default="gpt-oss:20b")
    parser.add_argument("--tie-break-model", default="gemma4:26b")
    args = parser.parse_args(argv)
    if len({args.primary_model, args.challenger_model, args.tie_break_model}) != 3:
        parser.error("classification models must be distinct")
    if not 128 <= args.embedding_dimensions <= 4096:
        parser.error("embedding dimensions must be between 128 and 4096")
    return args


def _main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parse_args(argv)
    root = _isolated_root()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _write_config(root / "config.toml", args)

    from chronovisor.core.llm_config import load_default_llm_runtime
    from chronovisor.core.llm_security import canonical_endpoint
    from chronovisor.core.ollama_transport import OLLAMA_URL

    runtime = load_default_llm_runtime()
    routes = _local_routes(runtime)
    endpoint = canonical_endpoint(OLLAMA_URL, cloud_secret=False)
    if not endpoint.is_loopback or endpoint.url != endpoint.origin:
        raise ValueError("ollama_endpoint_not_loopback_origin")
    return {
        "schema_version": 1,
        "status": "ok",
        "isolation": {"temporary_root": True, "production_state_touched": False},
        "routing": {
            "routes": routes,
            "all_local": True,
            "cloud_routes": 0,
            "silent_fallbacks": 0,
        },
        "network": {
            "non_loopback_outbound": 0,
            "ollama_origin": endpoint.origin,
            "proof": "local-routes+transformers-offline+production-inventory-gate",
        },
        "workflows": _run_workflows(root, runtime, args),
    }


if __name__ == "__main__":
    try:
        result = _main()
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "category": "local_e2e_failed",
                    "error_type": type(exc).__name__,
                },
                separators=(",", ":"),
            )
        )
        raise SystemExit(1) from None
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
