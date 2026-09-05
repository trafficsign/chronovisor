"""Read-only diagnostic checks; expected to fail on the investigated revision.

Run with the repository's .venv/bin/python. No model or service is contacted.
Only the synthetic corpus and post-recall persistence are replaced; the actual
fallback admission and QueryBatcher implementations are exercised.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from unittest.mock import patch

os.environ["CHRONOVISOR_READ_ONLY"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))


def fallback_probe() -> dict:
    from chronovisor.core.search_types import ScoredPage
    from chronovisor.recall.recall_runtime import (
        ContextItem,
        RecallPolicy,
        RecallRequest,
        run_deterministic_fallback,
    )

    request = RecallRequest(
        host="codex",
        event="UserPromptSubmit",
        prompt="CronoBisāに改善の余地はないよな。",
        cwd="/projects/personal/chronovisor",
        session_id="synthetic-diagnostic",
        decision_id="synthetic-diagnostic",
    )
    policy = RecallPolicy(max_context_chars=3000, max_total_context_chars=4602)
    candidate = ScoredPage("unrelated-cooking-note", "Cooking note", "", "", 22.0)
    item = ContextItem(candidate.page_id, candidate.title, "", candidate.score)
    with (
        patch("chronovisor.recall.recall_runtime.search_existing_bm25", return_value=[candidate]),
        patch("chronovisor.recall.recall_runtime.search_existing_lexical", return_value=([candidate], [candidate])),
        patch("chronovisor.recall.recall_runtime.context_item_from_page_id", return_value=item),
        patch("chronovisor.recall.recall_runtime.state_context_for_request", return_value=""),
        patch("chronovisor.recall.recall_runtime._capture_legacy_distillation_observation"),
    ):
        result = run_deterministic_fallback(request, policy, reason="synthetic timeout")
    return {
        "probe": "unrelated_fallback_admission",
        "confidence": result.confidence,
        "decision": result.decision,
        "unrelated_cards": len(result.context_items),
        "passes": not result.context_items,
    }


def queue_probe() -> dict:
    import numpy as np

    from chronovisor.search.semantic_service import QueryBatcher

    entered = threading.Event()
    release = threading.Event()
    calls: list[tuple[str, ...]] = []
    errors: list[BaseException] = []

    def encode(texts, _batch_size):
        calls.append(tuple(texts))
        if texts == ["first"]:
            entered.set()
            if not release.wait(timeout=2):
                raise RuntimeError("diagnostic release not signaled")
        return np.zeros((len(texts), 2), dtype=np.float32)

    batcher = QueryBatcher(
        encode=encode,
        search=lambda _vector, _top_n: [("page", 1.0)],
        window_ms=0,
        max_batch=1,
        available=lambda: True,
    )

    def first_submit():
        try:
            batcher.submit("first", 1, 1.5)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=first_submit)
    thread.start()
    timed_out = False
    try:
        assert entered.wait(timeout=1), "first encode did not start"
        try:
            batcher.submit("second", 1, 0.03)
        except TimeoutError:
            timed_out = True
        release.set()
        thread.join(timeout=2)
        assert not thread.is_alive(), "first submit did not finish"
        assert not errors, repr(errors)
        live_result = batcher.submit("third", 1, 1.0)
    finally:
        release.set()
        thread.join(timeout=2)
        batcher.close()
    expired_work_ran = any("second" in call for call in calls)
    return {
        "probe": "expired_pending_request",
        "caller_timed_out": timed_out,
        "expired_request_encoded": expired_work_ran,
        "following_request_completed": bool(live_result),
        "passes": timed_out and bool(live_result) and not expired_work_ran,
    }


if __name__ == "__main__":
    results = [fallback_probe(), queue_probe()]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    raise SystemExit(0 if all(result["passes"] for result in results) else 1)
