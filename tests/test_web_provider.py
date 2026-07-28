from __future__ import annotations

from chronovisor.research_config import WebConfig
from chronovisor.web_provider import (
    FederatedSearchProvider,
    FixtureSearchProvider,
    HttpSearchProvider,
    SearchResult,
    provider_from_config,
    route_source_packs,
    search_web,
)


def test_fixture_search_runs_before_live_egress_is_enabled(
    tmp_path, monkeypatch
) -> None:
    from chronovisor import web_provider

    monkeypatch.setattr(web_provider, "WEB_TRACE", tmp_path / "trace.jsonl")
    provider = FixtureSearchProvider(
        {
            "query": [
                {
                    "title": "Official",
                    "url": "https://example.com",
                    "snippet": "evidence",
                }
            ]
        }
    )

    result = search_web("query", config=WebConfig(), provider=provider)

    assert result.status == "ok"
    assert result.results[0].provider == "fixture"


def test_sensitive_query_is_blocked_before_provider_call(tmp_path, monkeypatch) -> None:
    from chronovisor import web_provider

    monkeypatch.setattr(web_provider, "WEB_TRACE", tmp_path / "trace.jsonl")

    class Forbidden:
        name = "forbidden"

        def search(self, query: str, *, limit: int):
            raise AssertionError("provider must not receive sensitive query")

    result = search_web(
        "person@example.com latest status",
        config=WebConfig(adapter_enabled=True, live_egress_enabled=True),
        provider=Forbidden(),
    )

    assert result.status == "blocked"
    assert result.policy_reason == "pii_detected"


def test_searxng_adapter_normalizes_mocked_live_results(tmp_path, monkeypatch) -> None:
    import httpx
    from chronovisor import web_provider
    from chronovisor.web_provider import HttpSearchProvider

    monkeypatch.setattr(web_provider, "WEB_TRACE", tmp_path / "trace.jsonl")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Official docs",
                            "url": "https://example.com/docs",
                            "content": "primary evidence",
                        }
                    ]
                },
            )
        )
    )
    provider = HttpSearchProvider(
        name="searxng", endpoint="https://example.com", client=client
    )

    result = search_web(
        "bounded research",
        config=WebConfig(
            adapter_enabled=True, live_egress_enabled=True, provider="searxng"
        ),
        provider=provider,
    )

    assert result.status == "ok"
    assert result.results[0].title == "Official docs"
    assert "bounded research" in (tmp_path / "trace.jsonl").read_text()


def test_searxng_partial_engine_failure_is_visible_without_losing_results(
    tmp_path, monkeypatch
) -> None:
    import httpx
    from chronovisor import web_provider
    from chronovisor.web_provider import HttpSearchProvider

    monkeypatch.setattr(web_provider, "WEB_TRACE", tmp_path / "trace.jsonl")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Healthy result",
                            "url": "https://example.com/healthy",
                            "content": "available from another engine",
                        }
                    ],
                    "unresponsive_engines": [
                        ["brave", "Too many requests"],
                        ["startpage", "parsing error"],
                    ],
                },
            )
        )
    )
    provider = HttpSearchProvider(
        name="searxng",
        endpoint="https://example.com",
        client=client,
    )

    result = search_web(
        "bounded research",
        config=WebConfig(
            adapter_enabled=True,
            live_egress_enabled=True,
            provider="searxng",
        ),
        provider=provider,
    )

    assert result.status == "degraded"
    assert [row.title for row in result.results] == ["Healthy result"]
    assert result.provider_statuses == {
        "searxng:brave": "error:Too many requests",
        "searxng:startpage": "error:parsing error",
    }
    assert result.error.startswith("partial provider degradation:")


def test_mediawiki_adapter_is_a_keyless_bounded_fallback() -> None:
    import httpx
    from chronovisor.web_provider import HttpSearchProvider

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "query": {
                        "search": [
                            {
                                "title": "人工知能",
                                "pageid": 1,
                                "snippet": "<span>AI</span> の説明",
                            }
                        ]
                    }
                },
            )
        )
    )
    provider = HttpSearchProvider(
        name="mediawiki",
        endpoint="https://ja.wikipedia.org/w/api.php",
        client=client,
    )

    rows = provider.search("人工知能", limit=3)

    assert rows[0].snippet == "AI の説明"
    assert (
        rows[0].url
        == "https://ja.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%9F%A5%E8%83%BD"
    )


def test_source_pack_router_is_bounded_and_deterministic() -> None:
    enabled = ("general", "code", "academic", "encyclopedia")

    assert route_source_packs("Gemma model release update", enabled=enabled) == (
        "code",
        "general",
    )
    assert route_source_packs("最新のtransformer論文を調査", enabled=enabled) == (
        "academic",
        "general",
    )
    assert route_source_packs("Chronovisorとは", enabled=enabled) == (
        "encyclopedia",
        "general",
    )
    assert route_source_packs("今日の一般ニュース", enabled=enabled) == ("general",)


def test_github_adapter_normalizes_repository_results() -> None:
    import httpx

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "full_name": "owner/project",
                            "html_url": "https://github.com/owner/project",
                            "description": "official source",
                            "language": "Python",
                            "stargazers_count": 42,
                            "updated_at": "2026-07-23T00:00:00Z",
                        }
                    ]
                },
            )
        )
    )
    provider = HttpSearchProvider(
        name="github",
        endpoint="https://api.github.com/search/repositories",
        client=client,
    )

    rows = provider.search("project", limit=3)

    assert rows[0].title == "owner/project"
    assert rows[0].provider == "github"
    assert "stars=42" in rows[0].snippet


def test_academic_adapters_normalize_arxiv_and_crossref() -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "export.arxiv.org":
            return httpx.Response(
                200,
                text="""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2607.00001</id>
    <title>Bounded Local Research</title>
    <summary>Evidence-first orchestration.</summary>
    <published>2026-07-01T00:00:00Z</published>
    <author><name>Researcher One</name></author>
  </entry>
</feed>""",
                headers={"content-type": "application/atom+xml"},
            )
        return httpx.Response(
            200,
            json={
                "message": {
                    "items": [
                        {
                            "DOI": "10.1000/example",
                            "URL": "https://doi.org/10.1000/example",
                            "title": ["Bounded Research Systems"],
                            "abstract": "<jats:p>Verified evidence.</jats:p>",
                            "author": [{"given": "Ada", "family": "Example"}],
                            "type": "journal-article",
                        }
                    ]
                }
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    arxiv = HttpSearchProvider(
        name="arxiv",
        endpoint="https://export.arxiv.org/api/query",
        client=client,
    )
    crossref = HttpSearchProvider(
        name="crossref",
        endpoint="https://api.crossref.org/works",
        client=client,
    )

    assert arxiv.search("local research", limit=2)[0].provider == "arxiv"
    crossref_row = crossref.search("local research", limit=2)[0]
    assert crossref_row.provider == "crossref"
    assert crossref_row.title == "Bounded Research Systems"
    assert "Verified evidence." in crossref_row.snippet


def test_federated_provider_prefers_specialist_and_deduplicates_urls() -> None:
    class Stub:
        def __init__(self, name: str, rows: list[SearchResult]) -> None:
            self.name = name
            self.endpoint = ""
            self.rows = rows
            self.queries: list[str] = []

        def search(self, query: str, *, limit: int) -> list[SearchResult]:
            self.queries.append(query)
            return self.rows[:limit]

    shared = "https://github.com/owner/project"
    github = Stub(
        "github",
        [SearchResult("Official", shared, "repository", 1, "github")],
    )
    providers = {
        "github": github,
        "searxng": Stub(
            "searxng",
            [
                SearchResult("Duplicate", shared + "/", "web", 1, "searxng"),
                SearchResult(
                    "Docs",
                    "https://example.com/docs",
                    "documentation",
                    2,
                    "searxng",
                ),
            ],
        ),
    }
    config = WebConfig(
        provider="federated",
        source_packs=("general", "code"),
        max_provider_calls=4,
    )
    provider = FederatedSearchProvider(
        providers=providers,
        source_packs=config.source_packs,
        config=config,
    )

    rows = provider.search("model release update", limit=5)

    assert [row.provider for row in rows] == ["github", "searxng"]
    assert len({row.url.rstrip("/") for row in rows}) == 2
    assert github.queries == ["model release update"]
    assert provider.last_statuses == {"github": "ok:1", "searxng": "ok:2"}


def test_unknown_source_pack_fails_closed() -> None:
    config = WebConfig(
        adapter_enabled=True,
        live_egress_enabled=True,
        provider="federated",
        source_packs=("general", "random-site-api"),
        searxng_endpoint="http://127.0.0.1:8888",
        allow_local_search_backend=True,
    )

    result = search_web("public query", config=config)

    assert provider_from_config(config) is None
    assert result.status == "degraded"
    assert result.results == ()
    assert result.error == "source packs are not adopted: random-site-api"


def test_loopback_exception_is_limited_to_local_searxng_search(
    tmp_path, monkeypatch
) -> None:
    import httpx

    from chronovisor import web_provider

    monkeypatch.setattr(web_provider, "WEB_TRACE", tmp_path / "trace.jsonl")
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"results": []})
        )
    )
    provider = HttpSearchProvider(
        name="searxng",
        endpoint="http://127.0.0.1:8888",
        client=client,
    )

    allowed = search_web(
        "public query",
        config=WebConfig(
            adapter_enabled=True,
            live_egress_enabled=True,
            provider="searxng",
            allow_local_search_backend=True,
        ),
        provider=provider,
    )
    blocked = search_web(
        "public query",
        config=WebConfig(
            adapter_enabled=True,
            live_egress_enabled=True,
            provider="searxng",
            allow_local_search_backend=False,
        ),
        provider=provider,
    )

    assert allowed.status == "ok"
    assert blocked.status == "blocked"
