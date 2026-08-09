"""Post-apply retrieval readback verification for changed ingest pages."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _runtime():
    from chronovisor.ingest import ingest

    return ingest


def _read_back_failure_log() -> Path:
    return _runtime().PAGES_DIR.parent / "runtime" / "ingest-read-back-failures.jsonl"


def _read_back_run_log() -> Path:
    return _runtime().PAGES_DIR.parent / "runtime" / "ingest-read-back-runs.jsonl"


def _read_back_query(meta: dict, page_id: str) -> str:
    questions = meta.get("recall_questions")
    if isinstance(questions, list):
        for question in questions:
            if isinstance(question, str) and question.strip():
                return question.strip()
    summary = meta.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    title = meta.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return page_id


def verify_changed_pages_read_back(page_ids: list[str], *, top_n: int = 10) -> dict:
    if not page_ids:
        return {"checked": 0, "passed": 0, "failed": []}
    runtime = _runtime()
    try:
        from chronovisor.search.index_store import get_store
        from chronovisor.search.search import search

        store = get_store()
        store.refresh()
    except Exception as e:
        runtime._safe_log(f"ingest | read-back unavailable: {e}")
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
        runtime._safe_log(
            f"ingest | read-back: {len(failed)} failed of {checked} checked",
            level="warn",
            outcome_kind="read_back_warning",
        )
    elif checked:
        runtime._safe_log(f"ingest | read-back: {checked} checked ok")

    return {"checked": checked, "passed": passed, "failed": failed}


def _refresh_ingest_derived_artifacts(
    changed_pages: list[str],
    *,
    source_raw: str | None,
) -> dict[str, Any]:
    """Refresh rebuildable indexes and return the normal read-back result."""

    runtime = _runtime()
    try:
        runtime._rebuild_index()
    except Exception as exc:
        runtime._safe_log(f"ingest | index.md rebuild failed (non-fatal): {exc}")

    try:
        from chronovisor.search.index_store import get_store

        get_store().refresh()
    except Exception as exc:
        runtime._safe_log(f"ingest | index_store refresh failed: {exc}")

    if changed_pages:
        try:
            from chronovisor.search.search import update_embeddings

            # Read-back is a correctness gate, so publication of the delta
            # index must complete before retrieval is evaluated. The previous
            # fire-and-forget enqueue produced false misses that passed when
            # the same query was repeated after the worker caught up.
            update_embeddings(page_ids=changed_pages, strict=True)
        except Exception as exc:
            runtime._safe_log(f"ingest | semantic index enqueue failed: {exc}")
        try:
            from chronovisor.search.claims import append_page_claims

            append_page_claims(
                changed_pages,
                source_raw=source_raw or "",
                op="ingest",
            )
        except Exception as exc:
            runtime._safe_log(f"ingest | claim ledger failed (non-fatal): {exc}")
        try:
            from chronovisor.ops.state_register import refresh_state_register

            refresh_state_register(changed_pages, source_raw=source_raw or "")
        except Exception as exc:
            runtime._safe_log(
                f"ingest | state register refresh failed (non-fatal): {exc}"
            )
    return verify_changed_pages_read_back(changed_pages)
