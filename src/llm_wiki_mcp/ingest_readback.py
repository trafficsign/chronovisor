"""Post-apply retrieval readback verification for changed ingest pages."""

from __future__ import annotations

import json
import time
from typing import Any


def _runtime():
    from llm_wiki_mcp import ingest

    return ingest


def _runtime_call(name: str):
    def call(*args: Any, **kwargs: Any) -> Any:
        return getattr(_runtime(), name)(*args, **kwargs)

    return call


_read_back_failure_log = _runtime_call("_read_back_failure_log")
_read_back_query = _runtime_call("_read_back_query")
_read_back_run_log = _runtime_call("_read_back_run_log")
_safe_log = _runtime_call("_safe_log")


def verify_changed_pages_read_back(page_ids: list[str], *, top_n: int = 10) -> dict:
    if not page_ids:
        return {"checked": 0, "passed": 0, "failed": []}
    try:
        from llm_wiki_mcp.index_store import get_store
        from llm_wiki_mcp.search import search

        store = get_store()
        store.refresh()
    except Exception as e:
        _safe_log(f"ingest | read-back unavailable: {e}")
        return {"checked": 0, "passed": 0, "failed": [{"error": str(e)}]}

    checked = 0
    passed = 0
    failed: list[dict] = []
    for page_id in page_ids:
        meta = store.meta(page_id)
        if meta is None:
            failed.append({"page_id": page_id, "reason": "missing-meta"})
            continue
        query = _read_back_query(meta, page_id)
        if not query:
            failed.append({"page_id": page_id, "reason": "empty-query"})
            continue
        checked += 1
        try:
            results, mode = search(query, top_n=top_n, semantic=True)
        except Exception as e:
            failed.append(
                {"page_id": page_id, "reason": "search-error", "error": str(e)}
            )
            continue
        rank = next(
            (
                idx + 1
                for idx, result in enumerate(results)
                if result.page_id == page_id
            ),
            None,
        )
        if rank is None:
            failed.append(
                {
                    "page_id": page_id,
                    "reason": "not-in-top-results",
                    "query": query[:180],
                    "mode": mode,
                    "top": [result.page_id for result in results[:5]],
                }
            )
        else:
            passed += 1

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "schema_version": 2,
        "cohort": "all_ingest_runs",
        "checked": checked,
        "passed": passed,
        "failed": failed,
    }
    try:
        run_path = _read_back_run_log()
        run_path.parent.mkdir(parents=True, exist_ok=True)
        with run_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass
    if failed:
        try:
            log_path = _read_back_failure_log()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
        _safe_log(
            f"ingest | read-back: {len(failed)} failed of {checked} checked",
            level="warn",
            outcome_kind="read_back_warning",
        )
    elif checked:
        _safe_log(f"ingest | read-back: {checked} checked ok")

    return {"checked": checked, "passed": passed, "failed": failed}

