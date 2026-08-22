from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chronovisor.core import llm_config, reranker
from chronovisor.core.canonical_document import CanonicalDocument, serialize_document
from chronovisor.core.llm_runtime import (
    LLMRuntime,
    RerankItem,
    RerankRequest,
    RerankResult,
    RerankRoute,
    RouteLocation,
    SourceDataClass,
    SourceDataClassification,
    SourceSensitivity,
)
from chronovisor.core.reranker import (
    QUERY_SOURCE,
    RERANK_RUNTIME_ROLE,
    RerankOutcome,
    apply_reranker_scores,
    rerank_results,
)
from chronovisor.core.runtime_config import RerankerConfig, RerankerServiceConfig
from chronovisor.core.search import ScoredPage
from chronovisor.hosts import server


def page(page_id: str, score: float = 1.0) -> ScoredPage:
    return ScoredPage(
        page_id=page_id,
        title=page_id,
        folder="",
        updated="2026-06-11",
        score=score,
    )


class FakeRerankBackend:
    provider = "fake-reranker"
    location = RouteLocation.LOCAL

    def __init__(self, scores: list[float] | None = None) -> None:
        self.scores = scores or [0.1, 0.9]
        self.requests: list[RerankRequest] = []

    def rerank(self, request: RerankRequest, *, model: str) -> RerankResult:
        self.requests.append(request)
        return RerankResult(
            tuple(
                RerankItem(index, score) for index, score in enumerate(self.scores)
            ),
            self.provider,
            model,
        )


def install_runtime(monkeypatch, backend: FakeRerankBackend) -> LLMRuntime:
    runtime = LLMRuntime(
        rerank={RERANK_RUNTIME_ROLE: RerankRoute(backend, "route-model")}
    )
    monkeypatch.setattr(llm_config, "load_default_llm_runtime", lambda: runtime)
    monkeypatch.setattr(
        reranker,
        "resolve_rerank_candidate",
        lambda candidate, **_kwargs: (
            candidate.page_id,
            SourceDataClassification(
                SourceDataClass.PAGE, SourceSensitivity.NORMAL
            ),
            ("pages", candidate.page_id, 1, 1, candidate.page_id),
        ),
    )
    return runtime


def test_rerank_results_disabled_preserves_order() -> None:
    candidates = [page("a"), page("b")]

    outcome = rerank_results("query", candidates, config=RerankerConfig(enabled=False))

    assert outcome.results == candidates
    assert outcome.metadata["status"] == "disabled"


def test_rerank_results_applies_scores_without_touching_tail(monkeypatch) -> None:
    candidates = [page("a"), page("b"), page("c")]
    backend = FakeRerankBackend()
    install_runtime(monkeypatch, backend)

    outcome = rerank_results(
        "query",
        candidates,
        config=RerankerConfig(enabled=True, top_n=2, weight=2.0),
    )

    assert [result.page_id for result in outcome.results] == ["b", "a", "c"]
    assert outcome.metadata["status"] == "applied"
    assert outcome.metadata["candidate_count"] == 2
    assert outcome.metadata["execution"] == "in_process"
    assert [row["page_id"] for row in outcome.metadata["scores"]] == ["b", "a"]
    assert [detail.raw_score for detail in outcome.scores] == [0.9, 0.1]
    assert round(outcome.scores[0].margin_to_next, 6) == 0.8
    assert backend.requests[0].source == QUERY_SOURCE
    assert backend.requests[0].candidate_sources == (
        SourceDataClassification(SourceDataClass.PAGE, SourceSensitivity.NORMAL),
    ) * 2
    assert outcome.metadata["route"] == {
        "role": RERANK_RUNTIME_ROLE,
        "provider": "fake-reranker",
        "model": "route-model",
        "location": "local",
    }


def test_rerank_results_unavailable_preserves_order(monkeypatch) -> None:
    candidates = [page("a"), page("b")]

    class FailingBackend(FakeRerankBackend):
        def rerank(self, request: RerankRequest, *, model: str) -> RerankResult:
            raise RuntimeError("SECRET backend detail")

    install_runtime(monkeypatch, FailingBackend())

    outcome = rerank_results(
        "query",
        candidates,
        config=RerankerConfig(enabled=True),
    )

    assert outcome.results == candidates
    assert outcome.metadata["status"] == "unavailable"
    assert outcome.metadata["reason"] == "backend_error"
    assert outcome.metadata["degraded"] is True
    assert "SECRET" not in repr(outcome.metadata)


def test_rerank_results_uses_resident_service_without_local_fallback(monkeypatch) -> None:
    from chronovisor.core import reranker_client

    candidates = [page("a"), page("b")]
    config = RerankerConfig(
        enabled=True,
        service=RerankerServiceConfig(enabled=True, mode="on"),
    )
    monkeypatch.setattr(
        reranker_client,
        "rerank",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            reranker_client.RerankerServiceUnavailable("transport_error")
        ),
    )
    monkeypatch.setattr(
        llm_config,
        "load_default_llm_runtime",
        lambda: pytest.fail("in-process reranker loaded"),
    )

    outcome = rerank_results("query", candidates, config=config)

    assert outcome.results is candidates
    assert outcome.metadata == {
        "status": "unavailable",
        "reason": "transport_error",
        "candidate_count": 2,
        "execution": "service",
        "degraded": True,
    }


def test_rerank_results_rejects_partial_score_vectors(monkeypatch) -> None:
    candidates = [page("a"), page("b")]
    outcome = apply_reranker_scores(
        candidates,
        [0.9],
        config=RerankerConfig(enabled=True, top_n=2),
    )

    assert outcome.results == candidates
    assert outcome.metadata["status"] == "unavailable"
    assert outcome.metadata["score_count"] == 1


def test_llm_runtime_routes_to_local_rerank_backend(
    monkeypatch,
) -> None:
    config = RerankerConfig(enabled=True, backend="transformers", model="legacy")
    seen: list[tuple[str, list[str], str]] = []

    def fake_impl(_config):
        def score(query, passages, call_config):
            seen.append((query, passages, call_config.model))
            return [0.1, 0.9]

        return score

    monkeypatch.setattr(reranker, "_score_impl", fake_impl)
    backend = reranker.LocalRerankBackend(config)
    runtime = LLMRuntime(rerank={"search": RerankRoute(backend, "reranker")})

    result = runtime.rerank(
        "search",
        RerankRequest(
            query="query",
            candidates=("first", "second"),
            source=SourceDataClassification(
                SourceDataClass.PAGE, SourceSensitivity.NORMAL
            ),
        ),
    )

    assert seen == [("query", ["first", "second"], "reranker")]
    assert result.items == (RerankItem(1, 0.9), RerankItem(0, 0.1))
    assert result.provider == "local-reranker"
    assert result.model == "reranker"
    assert result.metadata == {}


@pytest.mark.parametrize(
    ("namespace", "sensitivity", "expected_class", "expected_sensitivity"),
    [
        ("pages", "normal", SourceDataClass.PAGE, SourceSensitivity.NORMAL),
        ("pages", "high", SourceDataClass.PAGE, SourceSensitivity.HIGH),
        ("pages", None, SourceDataClass.PAGE, SourceSensitivity.HIGH),
        ("pages", "unknown", SourceDataClass.PAGE, SourceSensitivity.HIGH),
        ("system", "normal", SourceDataClass.SYSTEM, SourceSensitivity.HIGH),
    ],
)
def test_candidate_resolver_uses_canonical_namespace_and_sensitivity(
    tmp_path: Path,
    monkeypatch,
    namespace: str,
    sensitivity: str | None,
    expected_class: SourceDataClass,
    expected_sensitivity: SourceSensitivity,
) -> None:
    pages = tmp_path / "pages"
    system = tmp_path / "system"
    pages.mkdir()
    system.mkdir()
    root = system if namespace == "system" else pages
    path = root / "candidate.md"
    metadata: dict[str, Any] = {
        "title": "Canonical title",
        "status": "stable",
        "type": "knowledge",
    }
    if sensitivity is not None:
        metadata["sensitivity"] = sensitivity
    path.write_bytes(
        serialize_document(CanonicalDocument(metadata=metadata, body=b"body only"))
    )

    class Store:
        def meta(self, _page_id: str) -> dict[str, Any]:
            return {
                "namespace": namespace,
                "path": str(path),
                "title": "Canonical title",
                "sensitivity": "normal",
            }

    monkeypatch.setattr(reranker, "PAGES_DIR", pages)
    monkeypatch.setattr(reranker, "SYSTEM_DIR", system)

    passage, source, identity = reranker.resolve_rerank_candidate(
        "candidate", store=Store()  # type: ignore[arg-type]
    )

    assert passage == "Canonical title\n\nbody only"
    assert source == SourceDataClassification(expected_class, expected_sensitivity)
    assert identity[:2] == (namespace, str(path))


def test_candidate_resolver_missing_or_invalid_is_system_high() -> None:
    passage, source, identity = reranker.resolve_rerank_candidate(
        page("missing"), store=None
    )

    assert passage == "missing"
    assert source == SourceDataClassification(
        SourceDataClass.SYSTEM, SourceSensitivity.HIGH
    )
    assert identity[0] == "invalid"


def test_candidate_resolver_never_reads_symlinked_outside_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    pages = tmp_path / "pages"
    system = tmp_path / "system"
    outside = tmp_path / "outside"
    pages.mkdir()
    system.mkdir()
    outside.mkdir()
    secret = "CANARY_OUTSIDE_BYTES"
    outside_page = outside / "candidate.md"
    outside_page.write_text(secret, encoding="utf-8")
    leaf_link = pages / "candidate.md"
    descendant = pages / "linked"
    try:
        leaf_link.symlink_to(outside_page)
        descendant.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")

    class Store:
        def __init__(self, path: Path) -> None:
            self.path = path

        def meta(self, _page_id: str) -> dict[str, Any]:
            return {
                "namespace": "pages",
                "path": str(self.path),
                "title": "candidate",
                "sensitivity": "normal",
            }

    monkeypatch.setattr(reranker, "PAGES_DIR", pages)
    monkeypatch.setattr(reranker, "SYSTEM_DIR", system)
    for path in (leaf_link, descendant / "candidate.md"):
        passage, source, identity = reranker.resolve_rerank_candidate(
            "candidate", store=Store(path)  # type: ignore[arg-type]
        )
        assert secret not in passage
        assert source == SourceDataClassification(
            SourceDataClass.SYSTEM, SourceSensitivity.HIGH
        )
        assert identity[0] == "invalid"


def test_transformer_loader_prefers_complete_local_snapshot() -> None:
    calls: list[tuple[str, bool]] = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, _model, **kwargs):
            calls.append(("tokenizer", kwargs.get("local_files_only", False)))
            return "tokenizer"

    class FakeModel:
        @classmethod
        def from_pretrained(cls, _model, **kwargs):
            calls.append(("model", kwargs.get("local_files_only", False)))
            return "model"

    tokenizer, model = reranker._load_transformer_components(
        RerankerConfig(enabled=True), FakeTokenizer, FakeModel
    )

    assert (tokenizer, model) == ("tokenizer", "model")
    assert calls == [("tokenizer", True), ("model", True)]


def test_transformer_loader_allows_first_install_fallback() -> None:
    calls: list[tuple[str, bool]] = []

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, _model, **kwargs):
            local = kwargs.get("local_files_only", False)
            calls.append(("tokenizer", local))
            if local:
                raise OSError("not cached")
            return "tokenizer"

    class FakeModel:
        @classmethod
        def from_pretrained(cls, _model, **kwargs):
            calls.append(("model", kwargs.get("local_files_only", False)))
            return "model"

    tokenizer, model = reranker._load_transformer_components(
        RerankerConfig(enabled=True), FakeTokenizer, FakeModel
    )

    assert (tokenizer, model) == ("tokenizer", "model")
    assert calls[:3] == [
        ("tokenizer", True),
        ("tokenizer", False),
        ("model", False),
    ]


def test_chronovisor_search_uses_reranker_only_when_enabled(monkeypatch) -> None:
    class FakeStore:
        def refresh(self) -> None:
            pass

        def tags(self, page_id: str) -> list[str]:
            return []

        def outlinks(self, page_id: str) -> list[str]:
            return []

    def fake_search(**kwargs):
        assert kwargs["top_n"] == 10
        return [page("a"), page("b")], "hybrid"

    def fake_rerank(query, candidates, *, config):
        assert query == "needle"
        assert config.enabled is True
        return RerankOutcome(
            [candidates[1], candidates[0]],
            {
                "status": "applied",
                "candidate_count": 2,
                "weight": config.weight,
                "latency_ms": 3,
            },
        )

    from chronovisor.core import runtime_config
    from chronovisor.core import search as search_mod

    monkeypatch.setattr(search_mod, "search", fake_search)
    monkeypatch.setattr(
        runtime_config, "load_reranker_config", lambda: RerankerConfig(enabled=True)
    )
    monkeypatch.setattr(reranker, "rerank_results", fake_rerank)
    monkeypatch.setattr(server, "get_store", lambda: FakeStore())
    monkeypatch.setattr(
        server,
        "_direct_search_hits",
        lambda results, **_kwargs: [
            {"page_id": result.page_id} for result in results
        ],
    )

    tool_fn = (
        server.chronovisor_search.fn
        if hasattr(server.chronovisor_search, "fn")
        else server.chronovisor_search
    )
    payload = json.loads(tool_fn("needle", depth=0))

    assert payload["search_mode"] == "hybrid+rerank"
    assert payload["reranker"]["status"] == "applied"
    assert [hit["page_id"] for hit in payload["direct_hits"]] == ["b", "a"]


def test_chronovisor_search_reranks_after_tag_filter(monkeypatch) -> None:
    class FakeStore:
        def refresh(self) -> None:
            pass

        def tags(self, page_id: str) -> list[str]:
            return ["d/keep"] if page_id == "keep" else []

        def outlinks(self, page_id: str) -> list[str]:
            return []

    seen_candidates: list[list[str]] = []

    def fake_search(**kwargs):
        assert kwargs["top_n"] == 10
        return [page("keep"), page("drop")], "hybrid"

    def fake_rerank(query, candidates, *, config):
        seen_candidates.append([candidate.page_id for candidate in candidates])
        return RerankOutcome(
            candidates,
            {
                "status": "applied",
                "candidate_count": len(candidates),
                "weight": config.weight,
                "latency_ms": 2,
            },
        )

    from chronovisor.core import runtime_config
    from chronovisor.core import search as search_mod

    monkeypatch.setattr(search_mod, "search", fake_search)
    monkeypatch.setattr(
        runtime_config, "load_reranker_config", lambda: RerankerConfig(enabled=True)
    )
    monkeypatch.setattr(reranker, "rerank_results", fake_rerank)
    monkeypatch.setattr(server, "get_store", lambda: FakeStore())
    monkeypatch.setattr(
        server,
        "_direct_search_hits",
        lambda results, **_kwargs: [
            {"page_id": result.page_id} for result in results
        ],
    )

    tool_fn = (
        server.chronovisor_search.fn
        if hasattr(server.chronovisor_search, "fn")
        else server.chronovisor_search
    )
    payload = json.loads(tool_fn("needle", depth=0, tags=["d/keep"]))

    assert seen_candidates == [["keep"]]
    assert payload["reranker"]["status"] == "applied"
    assert [hit["page_id"] for hit in payload["direct_hits"]] == ["keep"]
