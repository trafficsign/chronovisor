"""Observation-only Recall work, consumed by the existing background job runner."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from typing import Any


def typed_graph_trace_rows(
    query: str,
    paths: dict[str, Any],
    *,
    shadow: bool,
    query_plan: str = "",
    decision_id: str,
    session_id: str,
) -> list[dict[str, Any]]:
    query_sha = hashlib.sha256(query.encode()).hexdigest()
    session = hashlib.sha256(session_id.encode()).hexdigest()[:16] if session_id else ""
    created_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
    output = []
    for page_id, value in paths.items():
        if not isinstance(value, dict):
            continue
        path_ids = [
            str(item)
            for item in value.get("relation_ids") or []
            if isinstance(item, str)
            and (item.startswith("rel_") or item.startswith("merge_"))
        ]
        relation_ids = [item for item in path_ids if item.startswith("rel_")]
        entity_merge_ids = [item for item in path_ids if item.startswith("merge_")]
        if not path_ids:
            continue
        path_id = str(value.get("path_id") or "")
        if not path_id:
            path_id = (
                "path_"
                + hashlib.sha256("|".join([page_id, *path_ids]).encode()).hexdigest()[
                    :24
                ]
            )
        output.append(
            {
                "schema_version": 1,
                "trace_id": hashlib.sha256(
                    f"{decision_id}:{path_id}:{page_id}:{shadow}".encode()
                ).hexdigest()[:24],
                "decision_id": decision_id,
                "session_hash": session,
                "query_sha256": query_sha,
                "query_plan": str(value.get("query_plan") or query_plan),
                "page_id": str(page_id),
                "path_id": path_id,
                "page_ids": [
                    str(item)
                    for item in value.get("pages") or value.get("path") or []
                    if isinstance(item, str)
                ],
                "relation_ids": relation_ids,
                "entity_merge_ids": entity_merge_ids,
                "relations": value.get("relations") or [],
                "community_id": str(value.get("community_id") or ""),
                "activation": float(value.get("activation") or 0.0),
                "shadow": shadow,
                "supervision": "exposure",
                "candidate_generated": True,
                "created_at": created_at,
                "external_model_calls": 0,
            }
        )
    return output


def enqueue_observers(payload: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("CHRONOVISOR_READ_ONLY") == "1":
        return {"status": "skipped", "reason": "read_only"}
    from chronovisor.core.background_jobs import enqueue_job

    job = enqueue_job(
        name="recall-shadow",
        module=__name__,
        args=[],
        env={},
        stdin_text=json.dumps(payload, ensure_ascii=False),
    )
    return {"status": "queued", "job_id": job["job_id"]}


def observe(payload: dict[str, Any]) -> dict[str, Any]:
    from chronovisor.core.jsonl_write import append_jsonl_durable
    from chronovisor.core.knowledge_graph_retrieval import shadow_candidate_paths
    from chronovisor.recall.recall_compiler import append_shadow_trace, compile_query
    from chronovisor.recall.recall_runtime import TYPED_GRAPH_TRACE_FILE

    if os.environ.get("CHRONOVISOR_READ_ONLY") == "1":
        return {"status": "skipped", "reason": "read_only"}
    errors = []
    for query in payload["graph_queries"]:
        try:
            paths = shadow_candidate_paths(query["seeds"], query=query["query"])
        except Exception as exc:
            errors.append({"observer": "graph", "reason": type(exc).__name__})
            continue
        rows = typed_graph_trace_rows(
            query["query"],
            paths,
            shadow=True,
            decision_id=payload["decision_id"],
            session_id=payload["session_id"],
        )
        if rows:
            append_jsonl_durable(TYPED_GRAPH_TRACE_FILE, rows, sort_keys=True)
    try:
        compiler = compile_query(payload["prompt"])
    except Exception as exc:
        compiler = {"status": "error", "reason": type(exc).__name__, "page_ids": []}
        errors.append({"observer": "compiler", "reason": type(exc).__name__})
    append_shadow_trace(
        prompt=payload["prompt"],
        compiler=compiler,
        teacher_page_ids=payload["teacher_page_ids"],
        committed_page_ids=payload["committed_page_ids"],
    )
    return {"status": "partial" if errors else "observed", "errors": errors}


def main() -> None:
    print(json.dumps(observe(json.load(sys.stdin))))


if __name__ == "__main__":
    main()
