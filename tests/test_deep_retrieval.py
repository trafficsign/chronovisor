from __future__ import annotations

import json
from pathlib import Path

from chronovisor.core.runtime_config import DecisionRouterConfig
from chronovisor.hosts import server
from chronovisor.research import deep_retrieval
from chronovisor.search.research_config import ResearchConfig
from chronovisor.search.research_store import ResearchStore
from chronovisor.search.search import ScoredPage


def _tool(function):
    return function.fn if hasattr(function, "fn") else function


def page(page_id: str, score: float = 1.0) -> ScoredPage:
    return ScoredPage(
        page_id=page_id,
        title=page_id,
        folder="",
        updated="2026-06-11",
        score=score,
    )


class FakeStore:
    def __init__(self) -> None:
        self.outlink_map = {"alpha": ["beta"], "beta": [], "target": []}
        self.backlink_map = {"alpha": [], "beta": ["alpha"], "target": []}

    def refresh(self) -> None:
        pass

    def meta(self, page_id: str):
        if page_id not in {"alpha", "beta", "target"}:
            return None
        return {
            "title": page_id.title(),
            "updated": "2026-06-11",
            "path": f"/tmp/{page_id}.md",
            "status": "active",
        }

    def outlinks(self, page_id: str) -> list[str]:
        return list(self.outlink_map.get(page_id, []))

    def backlinks(self, page_id: str) -> list[str]:
        return list(self.backlink_map.get(page_id, []))


def test_run_deep_dive_searches_reads_links_and_requeries(tmp_path, monkeypatch) -> None:
    paths = {}
    for page_id in ("alpha", "beta", "target"):
        path = tmp_path / f"{page_id}.md"
        path.write_text(
            f"---\ntitle: {page_id.title()}\nupdated: 2026-06-11\n---\nBody for {page_id}",
            encoding="utf-8",
        )
        paths[page_id] = path

    def fake_search(query: str, top_n: int, semantic: bool):
        if query == "q1":
            return [page("alpha")], "hybrid"
        return [page("target")], "hybrid"

    monkeypatch.setattr(deep_retrieval, "get_store", lambda: FakeStore())
    monkeypatch.setattr(deep_retrieval, "find_page", lambda page_id: paths.get(page_id))
    monkeypatch.setattr(deep_retrieval, "run_search", fake_search)
    monkeypatch.setattr(deep_retrieval, "_llm_requeries", lambda *args, **kwargs: ["q2"])

    result = deep_retrieval.run_deep_dive("q1", max_iterations=2, fanout=2)

    assert [item["query"] for item in result["iterations"]] == ["q1", "q2"]
    assert result["iterations"][0]["linked_page_ids"] == ["beta"]
    assert {page["page_id"] for page in result["pages"]} == {"alpha", "beta", "target"}


def test_start_deep_dive_enqueues_durable_worker(monkeypatch) -> None:
    from chronovisor.ops import background_jobs

    recorded = []

    def enqueue(**kwargs):
        recorded.append(kwargs)
        return {"job_id": "durable-job"}

    monkeypatch.setattr(background_jobs, "enqueue_job", enqueue)

    job_id = deep_retrieval.start_deep_dive("q", max_iterations=1)

    assert job_id == "durable-job"
    assert recorded[0]["module"] == "chronovisor.research.deep_retrieval_worker"
    assert json.loads(recorded[0]["stdin_text"])["query"] == "q"


def test_chronovisor_deep_dive_sync_returns_payload(monkeypatch) -> None:
    from chronovisor.research import deep_retrieval as deep_retrieval_mod

    monkeypatch.setattr(
        deep_retrieval_mod,
        "run_deep_dive",
        lambda *args, **kwargs: {"status": "completed", "query": "q", "iterations": []},
    )

    tool_fn = server.chronovisor_deep_dive.fn if hasattr(server.chronovisor_deep_dive, "fn") else server.chronovisor_deep_dive
    payload = json.loads(tool_fn("q", background=False, engine="v1"))

    assert payload == {"status": "completed", "query": "q", "iterations": []}


def test_chronovisor_jobs_reads_durable_deep_retrieval_job(monkeypatch) -> None:
    from chronovisor.ops import background_jobs

    monkeypatch.setattr(
        background_jobs,
        "get_job",
        lambda _job_id: {
            "job_id": "durable-job",
            "name": "deep-retrieval",
            "status": "queued",
            "created_at": "2026-07-18T00:00:00+00:00",
            "updated_at": "2026-07-18T00:00:00+00:00",
            "attempts": 0,
            "output_tail": "",
        },
    )

    payload = json.loads(_tool(server.chronovisor_jobs)("durable-job"))

    assert payload["status"] == "queued"
    assert payload["processor"] == "deep-retrieval"


def test_v2_deep_dive_uses_bounded_wiki_only_kernel(tmp_path, monkeypatch) -> None:
    from chronovisor.core import research_scheduler
    from chronovisor.research import research_orchestrator
    from chronovisor.search import research_store

    scheduler_root = tmp_path / "scheduler"
    monkeypatch.setattr(research_scheduler, "SYNC_DIR", scheduler_root / "sync")
    monkeypatch.setattr(research_scheduler, "RESEARCH_LOCK", scheduler_root / "lock")
    monkeypatch.setattr(research_scheduler, "ACTIVE_FILE", scheduler_root / "active.json")
    monkeypatch.setattr(research_scheduler, "SCHEDULER_LOG", scheduler_root / "log.jsonl")
    store = ResearchStore(tmp_path / "store")
    monkeypatch.setattr(research_store, "ResearchStore", lambda: store)

    def tool(action, _context):
        if action.type.value == "chronovisor_search":
            return {
                "query": "q",
                "search_mode": "bm25",
                "results": [{"page_id": "target", "title": "Target", "score": 1.0}],
            }
        return {
            "page_id": "target",
            "title": "Target",
            "updated": "2026-07-18",
            "body": "target evidence",
            "outlinks": [],
            "backlinks": [],
        }

    monkeypatch.setattr(research_orchestrator, "execute_tool", tool)
    result = deep_retrieval.run_deep_dive_v2(
        "q",
        max_iterations=3,
        use_llm=False,
        config=ResearchConfig(enabled=True, mode="trace"),
    )

    assert result["engine"] == "v2"
    assert result["authority"] == "wiki_only"
    assert result["stop_reason"] == "completed"
    assert result["pages"][0]["page_id"] == "target"


def _router_config() -> DecisionRouterConfig:
    return DecisionRouterConfig(
        primary_model="ornith:test",
        challenger_model="gpt-oss:test",
        tie_break_model="gemma:test",
        num_ctx=16_384,
        num_predict=512,
        read_timeout_ms=5_000,
        max_input_chars=20_000,
        max_output_chars=2_000,
        max_feedback_chars=2_000,
    )


def test_llm_requeries_repairs_invalid_json_in_same_session(monkeypatch) -> None:
    from chronovisor.core import ollama

    responses = iter(
        [
            '{"queries":["next"],"extra":true}',
            '{"queries":["next query"]}',
        ]
    )
    requests = []

    def transport(request):
        requests.append(request)
        return next(responses)

    monkeypatch.setattr(ollama, "is_available", lambda: True)
    monkeypatch.setattr(deep_retrieval, "load_decision_router_config", _router_config)

    queries = deep_retrieval._llm_requeries(
        "original",
        "current",
        [{"page_id": "page-a", "title": "A", "snippet": "body"}],
        limit=2,
        transport=transport,
    )

    assert queries == ["next query"]
    assert len(requests) == 2
    assert len(requests[1].messages) == 4


def test_llm_requeries_fails_closed_after_repeated_invalid_json(monkeypatch) -> None:
    from chronovisor.core import ollama

    calls = 0

    def transport(_request):
        nonlocal calls
        calls += 1
        return '{"queries":[]}'

    monkeypatch.setattr(ollama, "is_available", lambda: True)
    monkeypatch.setattr(deep_retrieval, "load_decision_router_config", _router_config)

    queries = deep_retrieval._llm_requeries(
        "original",
        "current",
        [],
        limit=2,
        transport=transport,
    )

    assert queries == []
    assert calls == 2


def test_injected_requery_transport_does_not_pollute_production_audit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from chronovisor.core import ollama, store

    chronovisor_root = tmp_path / "wiki"
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(ollama, "is_available", lambda: True)
    monkeypatch.setattr(deep_retrieval, "load_decision_router_config", _router_config)

    queries = deep_retrieval._llm_requeries(
        "original",
        "current",
        [],
        limit=1,
        transport=lambda _request: '{"queries":["follow up"]}',
    )

    assert queries == ["follow up"]
    assert not (chronovisor_root / "runtime" / "local-consensus").exists()
