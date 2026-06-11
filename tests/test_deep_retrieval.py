from __future__ import annotations

import json
import time

from llm_wiki_mcp import deep_retrieval, server
from llm_wiki_mcp.jobs import JobStatus, job_store
from llm_wiki_mcp.search import ScoredPage


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


def test_start_deep_dive_completes_background_job(monkeypatch) -> None:
    monkeypatch.setattr(
        deep_retrieval,
        "run_deep_dive",
        lambda *args, **kwargs: {"status": "completed", "iterations": [{"iteration": 1}]},
    )

    job_id = deep_retrieval.start_deep_dive("q", max_iterations=1)

    for _ in range(50):
        job = job_store.get(job_id)
        if job and job.status == JobStatus.COMPLETED:
            break
        time.sleep(0.01)

    job = job_store.get(job_id)
    assert job is not None
    assert job.status == JobStatus.COMPLETED
    assert job.completed_ops == 1
    assert job.result == {"status": "completed", "iterations": [{"iteration": 1}]}


def test_wiki_deep_dive_sync_returns_payload(monkeypatch) -> None:
    from llm_wiki_mcp import deep_retrieval as deep_retrieval_mod

    monkeypatch.setattr(
        deep_retrieval_mod,
        "run_deep_dive",
        lambda *args, **kwargs: {"status": "completed", "query": "q", "iterations": []},
    )

    tool_fn = server.wiki_deep_dive.fn if hasattr(server.wiki_deep_dive, "fn") else server.wiki_deep_dive
    payload = json.loads(tool_fn("q", background=False))

    assert payload == {"status": "completed", "query": "q", "iterations": []}
