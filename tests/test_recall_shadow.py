from __future__ import annotations

import json
from pathlib import Path

from chronovisor.core import background_jobs, knowledge_graph_retrieval
from chronovisor.core.durable_state import exclusive_text_file_lock
from chronovisor.recall import recall_compiler, recall_runtime, recall_shadow


def payload():
    return {
        "decision_id": "decision",
        "session_id": "session",
        "prompt": "query",
        "graph_queries": [{"query": "query", "seeds": ["seed"]}],
        "teacher_page_ids": ["teacher"],
        "committed_page_ids": ["committed"],
    }


def test_queue_defers_graph_and_compiler_until_worker(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("CHRONOVISOR_READ_ONLY", raising=False)
    monkeypatch.setattr(background_jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(background_jobs, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(
        background_jobs,
        "_lock",
        lambda: exclusive_text_file_lock(tmp_path / "state.lock"),
    )
    calls = []
    monkeypatch.setattr(
        knowledge_graph_retrieval,
        "shadow_candidate_paths",
        lambda seeds, **_k: (
            calls.append(("graph", seeds))
            or {"page": {"relation_ids": ["rel_test"], "pages": ["seed", "page"]}}
        ),
    )
    monkeypatch.setattr(
        recall_compiler,
        "compile_query",
        lambda prompt: (
            calls.append(("compiler", prompt))
            or {"status": "exact", "page_ids": ["committed"]}
        ),
    )
    traces = []
    monkeypatch.setattr(
        recall_compiler, "append_shadow_trace", lambda **kw: traces.append(kw)
    )
    graph_trace = tmp_path / "graph.jsonl"
    monkeypatch.setattr(recall_runtime, "TYPED_GRAPH_TRACE_FILE", graph_trace)
    receipt = recall_shadow.enqueue_observers(payload())
    assert receipt["status"] == "queued"
    assert calls == []
    job = background_jobs._load()["jobs"][receipt["job_id"]]
    assert job["module"] == "chronovisor.recall.recall_shadow"
    recall_shadow.observe(json.loads(job["stdin"]))
    assert calls == [("graph", ["seed"]), ("compiler", "query")]
    assert traces[0]["committed_page_ids"] == ["committed"]
    row = json.loads(graph_trace.read_text())
    assert row["shadow"] is True
    assert row["decision_id"] == "decision"


def test_read_only_never_enqueues_or_observes(monkeypatch) -> None:
    monkeypatch.setenv("CHRONOVISOR_READ_ONLY", "1")

    def forbidden(*_a, **_k):
        raise AssertionError("read-only observer wrote or ran")

    monkeypatch.setattr(background_jobs, "enqueue_job", forbidden)
    monkeypatch.setattr(recall_compiler, "compile_query", forbidden)
    monkeypatch.setattr(knowledge_graph_retrieval, "shadow_candidate_paths", forbidden)
    assert recall_shadow.enqueue_observers(payload()) == {
        "status": "skipped",
        "reason": "read_only",
    }
    recall_shadow.observe(payload())


def test_graph_failure_does_not_prevent_compiler_observation(monkeypatch) -> None:
    monkeypatch.delenv("CHRONOVISOR_READ_ONLY", raising=False)
    monkeypatch.setattr(
        knowledge_graph_retrieval,
        "shadow_candidate_paths",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad snapshot")),
    )
    monkeypatch.setattr(
        recall_compiler, "compile_query", lambda _p: {"status": "fallback"}
    )
    traces = []
    monkeypatch.setattr(
        recall_compiler, "append_shadow_trace", lambda **kw: traces.append(kw)
    )
    result = recall_shadow.observe(payload())
    assert result == {
        "status": "partial",
        "errors": [{"observer": "graph", "reason": "ValueError"}],
    }
    assert traces[0]["compiler"]["status"] == "fallback"
