from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from chronovisor.core import llm_config, ollama, runtime_config, store
from chronovisor.core.llm_runtime import (
    BackendCapabilities,
    GenerationResult,
    GenerationRoute,
    LLMRuntime,
    MessageGenerationRequest,
    RouteLocation,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)
from chronovisor.core.search import ScoredPage
from chronovisor.hosts import server
from chronovisor.research import deep_retrieval
from chronovisor.search.research_config import ResearchConfig
from chronovisor.search.research_store import ResearchStore


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
    def __init__(self, paths: dict[str, Path]) -> None:
        self.paths = paths
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
            "path": str(self.paths[page_id]),
            "status": "stable",
            "relative_path": f"{page_id}.md",
            "is_system": False,
        }

    def outlinks(self, page_id: str) -> list[str]:
        return list(self.outlink_map.get(page_id, []))

    def backlinks(self, page_id: str) -> list[str]:
        return list(self.backlink_map.get(page_id, []))


def test_run_deep_dive_searches_reads_links_and_requeries(tmp_path, monkeypatch) -> None:
    paths = {}
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    for page_id in ("alpha", "beta", "target"):
        path = pages_dir / f"{page_id}.md"
        path.write_text(
            f"---\ntitle: {page_id.title()}\nupdated: 2026-06-11\n"
            f"status: stable\ntype: knowledge\n---\nBody for {page_id}",
            encoding="utf-8",
        )
        paths[page_id] = path

    def fake_search(query: str, top_n: int, semantic: bool):
        if query == "q1":
            return [page("alpha")], "hybrid"
        return [page("target")], "hybrid"

    monkeypatch.setattr(deep_retrieval, "get_store", lambda: FakeStore(paths))
    monkeypatch.setattr(deep_retrieval, "PAGES_DIR", pages_dir)
    monkeypatch.setattr(deep_retrieval, "run_search", fake_search)
    monkeypatch.setattr(deep_retrieval, "_llm_requeries", lambda *args, **kwargs: ["q2"])

    result = deep_retrieval.run_deep_dive("q1", max_iterations=2, fanout=2)

    assert [item["query"] for item in result["iterations"]] == ["q1", "q2"]
    assert result["iterations"][0]["linked_page_ids"] == ["beta"]
    assert {page["page_id"] for page in result["pages"]} == {"alpha", "beta", "target"}


@pytest.mark.parametrize("drift", ["symlink", "external_path"])
def test_research_reads_revalidate_indexed_path_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    from chronovisor.research import research_tools

    pages_dir = tmp_path / "wiki" / "pages"
    pages_dir.mkdir(parents=True)
    page = pages_dir / "secret.md"
    canonical = (
        "---\ntitle: Secret\nstatus: stable\ntype: knowledge\n---\nSECRET BODY\n"
    )
    page.write_text(canonical, encoding="utf-8")
    external = tmp_path / "external.md"
    external.write_text(canonical, encoding="utf-8")
    indexed_path = page
    if drift == "symlink":
        page.unlink()
        page.symlink_to(external)
    else:
        indexed_path = external

    class DriftedStore(FakeStore):
        def __init__(self) -> None:
            super().__init__({"secret": indexed_path})

        def meta(self, page_id: str):
            if page_id != "secret":
                return None
            return {
                "title": "Secret",
                "updated": "2026-06-11",
                "path": str(indexed_path),
                "status": "stable",
                "relative_path": "secret.md",
                "is_system": False,
            }

    store = DriftedStore()
    monkeypatch.setattr(research_tools, "get_store", lambda: store)
    monkeypatch.setattr(research_tools, "PAGES_DIR", pages_dir)
    monkeypatch.setattr(deep_retrieval, "get_store", lambda: store)
    monkeypatch.setattr(deep_retrieval, "PAGES_DIR", pages_dir)

    with pytest.raises(FileNotFoundError):
        research_tools.chronovisor_read({"page_id": "secret"}, None)  # type: ignore[arg-type]
    assert deep_retrieval._page_record("secret") is None


@pytest.mark.parametrize("status", ["draft", "deprecated"])
def test_research_reads_exclude_nonstable_model_selected_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    from chronovisor.research import research_tools

    path = tmp_path / "secret.md"
    path.write_text(
        f"---\ntitle: Secret\nstatus: {status}\ntype: knowledge\n---\nSECRET BODY",
        encoding="utf-8",
    )

    class NonStableStore:
        def refresh(self) -> None:
            return None

        def meta(self, page_id: str):
            if page_id != "secret":
                return None
            return {"status": status, "path": str(path), "title": "Secret"}

        def outlinks(self, _page_id: str) -> list[str]:
            return ["secret"]

        def backlinks(self, _page_id: str) -> list[str]:
            return []

    store = NonStableStore()
    monkeypatch.setattr(research_tools, "get_store", lambda: store)
    monkeypatch.setattr(deep_retrieval, "get_store", lambda: store)

    with pytest.raises(FileNotFoundError):
        research_tools.chronovisor_read({"page_id": "secret"}, None)  # type: ignore[arg-type]
    assert deep_retrieval._page_record("secret") is None
    assert deep_retrieval._linked_page_ids(["source"], limit=5) == []


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


def test_start_evidence_dive_reuses_durable_worker(monkeypatch) -> None:
    from chronovisor.ops import background_jobs

    recorded = []

    def enqueue(**kwargs):
        recorded.append(kwargs)
        return {"job_id": "evidence-job"}

    monkeypatch.setattr(background_jobs, "enqueue_job", enqueue)

    job_id = deep_retrieval.start_deep_dive("q", engine="evidence")

    assert job_id == "evidence-job"
    assert recorded[0]["module"] == "chronovisor.research.deep_retrieval_worker"
    assert recorded[0]["args"][-1] == "evidence"
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


def test_chronovisor_evidence_dive_sync_is_explicit(monkeypatch) -> None:
    monkeypatch.setattr(
        deep_retrieval,
        "run_evidence_dive",
        lambda query: {
            "status": "completed",
            "engine": "evidence",
            "query": query,
            "packet": {"packet_id": "packet:test"},
        },
    )

    payload = json.loads(
        _tool(server.chronovisor_deep_dive)("q", background=False, engine="evidence")
    )

    assert payload["engine"] == "evidence"
    assert payload["packet"]["packet_id"] == "packet:test"


def test_evidence_dive_activation_waits_for_campaign_x(monkeypatch) -> None:
    monkeypatch.setattr(
        deep_retrieval,
        "okf_startup_status",
        lambda _root: SimpleNamespace(
            allowed=True,
            layout="legacy",
            state="unmigrated",
        ),
    )

    result = deep_retrieval.run_evidence_dive("q")

    assert result == {
        "status": "blocked",
        "engine": "evidence",
        "reason": "campaign_x_not_finalized",
    }


def test_evidence_dive_executes_bounded_raw_gap_actions(monkeypatch) -> None:
    from chronovisor.research import evidence_runtime
    from chronovisor.search import research_config, research_store

    config = object()
    store = object()
    observed: dict[str, object] = {}
    program = SimpleNamespace(
        claim_slots=(SimpleNamespace(slot_id="answer", claim="outage cause"),)
    )
    monkeypatch.setattr(
        deep_retrieval,
        "okf_startup_status",
        lambda _root: SimpleNamespace(
            allowed=True,
            layout="okf_v0_2",
            state="finalized-v2",
        ),
    )
    monkeypatch.setattr(
        evidence_runtime, "run_projection_cycle", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        evidence_runtime,
        "compile_projection_program",
        lambda _query, _as_of: program,
    )
    monkeypatch.setattr(research_config, "load_research_config", lambda: config)
    monkeypatch.setattr(research_store, "ResearchStore", lambda: store)

    def retrieve(_program, _projection, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            stop_reason="missing_required_evidence",
            packet=SimpleNamespace(to_dict=lambda: {"abstained": True}),
            trace={},
            telemetry={"cloud_call_count": 0, "external_model_call_count": 0},
        )

    monkeypatch.setattr(evidence_runtime, "run_evidence_retrieval", retrieve)

    result = deep_retrieval.run_evidence_dive("outage cause", rebuild_projection=True)

    [(slot_id, action)] = observed["actions"]
    assert slot_id == "answer"
    assert action.type.value == "raw_search"
    assert action.arguments == {
        "query": "outage cause",
        "limit": 5,
        "scan_limit": 1_000,
    }
    assert observed["tool_context"].config is config
    assert observed["tool_context"].store is store
    assert observed["raw_dir"] == deep_retrieval.RAW_DIR
    assert result["telemetry"] == {
        "cloud_call_count": 0,
        "external_model_call_count": 0,
    }


def test_evidence_dive_sync_is_projection_only(monkeypatch) -> None:
    from chronovisor.research import evidence_reconstruction, evidence_runtime

    observed: dict[str, object] = {}
    monkeypatch.setattr(
        deep_retrieval,
        "okf_startup_status",
        lambda _root: SimpleNamespace(
            allowed=True,
            layout="okf_v0_2",
            state="finalized-v2",
        ),
    )
    monkeypatch.setattr(
        evidence_reconstruction,
        "load_episode_projection",
        lambda _path: object(),
    )
    monkeypatch.setattr(
        evidence_runtime,
        "compile_projection_program",
        lambda _query, _as_of: object(),
    )

    def retrieve(_program, _projection, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            stop_reason="coverage",
            packet=SimpleNamespace(to_dict=lambda: {"abstained": False}),
            trace={},
            telemetry={"raw_search_calls": 0},
        )

    monkeypatch.setattr(evidence_runtime, "run_evidence_retrieval", retrieve)

    result = deep_retrieval.run_evidence_dive("outage cause")

    assert observed["actions"] == ()
    assert observed["raw_dir"] is None
    assert "tool_context" not in observed
    assert result["telemetry"] == {"raw_search_calls": 0}


def test_evidence_worker_rebuilds_projection_in_background(monkeypatch) -> None:
    from chronovisor.research import deep_retrieval_worker

    observed: dict[str, object] = {}

    def run(query: str, *, rebuild_projection: bool = False):
        observed.update(query=query, rebuild_projection=rebuild_projection)
        return {"status": "completed", "engine": "evidence"}

    monkeypatch.setattr(deep_retrieval_worker, "run_evidence_dive", run)
    monkeypatch.setattr(
        deep_retrieval_worker.sys,
        "stdin",
        StringIO('{"query":"q"}'),
    )

    assert deep_retrieval_worker.main(["--run-id", "run", "--engine", "evidence"]) == 0
    assert observed == {"query": "q", "rebuild_projection": True}


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


def test_llm_requeries_repairs_invalid_json_in_same_session(monkeypatch) -> None:
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

    queries = deep_retrieval._llm_requeries(
        "original",
        "current",
        [{"page_id": "page-a", "title": "A", "snippet": "body"}],
        limit=2,
        transport=transport,
    )

    assert queries == ["next query"]
    assert len(requests) == 2
    assert requests[0].model == "injected:deep-retrieval-requery"
    assert len(requests[1].messages) == 4


def test_llm_requeries_fails_closed_after_repeated_invalid_json(monkeypatch) -> None:
    calls = 0

    def transport(_request):
        nonlocal calls
        calls += 1
        return '{"queries":[]}'

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
    chronovisor_root = tmp_path / "wiki"
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", chronovisor_root)
    monkeypatch.setattr(
        llm_config,
        "load_default_llm_runtime",
        lambda: pytest.fail("injected transport loaded runtime configuration"),
    )
    monkeypatch.setattr(
        runtime_config,
        "load_decision_router_config",
        lambda: pytest.fail("injected transport loaded Decision Router config"),
    )
    _forbid_ollama_controls(monkeypatch)

    queries = deep_retrieval._llm_requeries(
        "original",
        "current",
        [],
        limit=1,
        transport=lambda _request: '{"queries":["follow up"]}',
    )

    assert queries == ["follow up"]
    assert not (chronovisor_root / "runtime" / "local-consensus").exists()


def test_production_requery_uses_fixed_raw_high_runtime_role(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class Session:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def run(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(ok=True, value={"queries": ["follow up"]})

    monkeypatch.setattr(deep_retrieval, "LocalStructuredSession", Session)
    monkeypatch.setattr(
        runtime_config,
        "load_decision_router_config",
        lambda: pytest.fail("legacy Decision Router model selected requery execution"),
    )

    queries = deep_retrieval._llm_requeries(
        "original",
        "current",
        [],
        limit=1,
    )

    assert queries == ["follow up"]
    assert captured["model"] is None
    assert captured["runtime_role"] == deep_retrieval.REQUERY_RUNTIME_ROLE
    assert captured["source_data_class"] == "raw"
    assert captured["source_sensitivity"] == "high"
    assert captured["num_ctx"] == 114_688
    assert captured["num_predict"] == 512
    assert captured["max_output_chars"] == 2_000


class _RemoteBackend:
    provider = "remote-test"
    location = RouteLocation.REMOTE

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[tuple[MessageGenerationRequest, str]] = []

    def generate(
        self, request: MessageGenerationRequest, *, model: str
    ) -> GenerationResult:
        self.requests.append((request, model))
        if self.fail:
            raise RuntimeError("provider failed")
        return GenerationResult(
            content='{"queries":["remote follow up"]}',
            provider=self.provider,
            model=model,
            finish_reason="stop",
        )


def _remote_runtime(backend: _RemoteBackend, *, allow_egress: bool) -> LLMRuntime:
    return LLMRuntime(
        generation={
            deep_retrieval.REQUERY_RUNTIME_ROLE: GenerationRoute(
                backend,
                "remote-requery-model",
                BackendCapabilities(True, False, structured_output=True),
            )
        },
        remote_egress_opt_ins=(
            {(deep_retrieval.REQUERY_RUNTIME_ROLE, SourceDataClass.RAW)}
            if allow_egress
            else set()
        ),
    )


def _forbid_ollama_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("remote deep retrieval touched an Ollama control")

    for name in (
        "chat",
        "generate",
        "is_available",
        "model_digests",
        "model_resource_lease",
        "model_resource_lease_mode",
        "plan_model_residency",
        "resident_model_rows",
        "unload_model",
        "unload_named_model",
    ):
        monkeypatch.setattr(ollama, name, forbidden)


def test_remote_requery_succeeds_without_ollama_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _RemoteBackend()
    monkeypatch.setattr(
        llm_config,
        "load_default_llm_runtime",
        lambda: _remote_runtime(backend, allow_egress=True),
    )
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", tmp_path)
    _forbid_ollama_controls(monkeypatch)

    queries = deep_retrieval._llm_requeries(
        "original",
        "current",
        [{"page_id": "page-a", "title": "A", "snippet": "body"}],
        limit=2,
    )

    assert queries == ["remote follow up"]
    assert len(backend.requests) == 1
    request, model = backend.requests[0]
    assert model == "remote-requery-model"
    assert request.source == SourceDataClassification(
        SourceDataClass.RAW,
        SourceSensitivity.HIGH,
    )


def test_remote_egress_denial_uses_only_deterministic_requery_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _RemoteBackend()
    queries: list[str] = []

    def fake_search(query: str, top_n: int, semantic: bool):
        queries.append(query)
        return ([page("alpha")] if len(queries) == 1 else []), "bm25"

    paths: dict[str, Path] = {}
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    for page_id in ("alpha", "beta", "target"):
        path = pages_dir / f"{page_id}.md"
        path.write_text(
            f"---\ntitle: {page_id.title()}\nstatus: stable\ntype: knowledge\n---\n"
            f"Body for {page_id}",
            encoding="utf-8",
        )
        paths[page_id] = path

    monkeypatch.setattr(
        llm_config,
        "load_default_llm_runtime",
        lambda: _remote_runtime(backend, allow_egress=False),
    )
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", tmp_path)
    monkeypatch.setattr(deep_retrieval, "get_store", lambda: FakeStore(paths))
    monkeypatch.setattr(deep_retrieval, "PAGES_DIR", pages_dir)
    monkeypatch.setattr(deep_retrieval, "run_search", fake_search)
    _forbid_ollama_controls(monkeypatch)

    assert deep_retrieval._llm_requeries("q1", "q1", [], limit=2) == []
    result = deep_retrieval.run_deep_dive("q1", max_iterations=2, fanout=1)

    assert backend.requests == []
    assert queries == ["q1", "q1 Alpha Beta"]
    assert result["iterations"][0]["next_queries"] == ["q1 Alpha Beta"]


def test_remote_provider_failure_has_no_retry_or_local_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _RemoteBackend(fail=True)
    monkeypatch.setattr(
        llm_config,
        "load_default_llm_runtime",
        lambda: _remote_runtime(backend, allow_egress=True),
    )
    monkeypatch.setattr(store, "CHRONOVISOR_ROOT", tmp_path)
    _forbid_ollama_controls(monkeypatch)

    queries = deep_retrieval._llm_requeries("original", "current", [], limit=2)

    assert queries == []
    assert len(backend.requests) == 1
