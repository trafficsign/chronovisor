from __future__ import annotations

from llm_wiki_mcp.research_config import WebConfig
from llm_wiki_mcp.web_provider import FixtureSearchProvider, search_web


def test_fixture_search_runs_before_live_egress_is_enabled(tmp_path, monkeypatch) -> None:
    from llm_wiki_mcp import web_provider

    monkeypatch.setattr(web_provider, "WEB_TRACE", tmp_path / "trace.jsonl")
    provider = FixtureSearchProvider(
        {"query": [{"title": "Official", "url": "https://example.com", "snippet": "evidence"}]}
    )

    result = search_web("query", config=WebConfig(), provider=provider)

    assert result.status == "ok"
    assert result.results[0].provider == "fixture"


def test_sensitive_query_is_blocked_before_provider_call(tmp_path, monkeypatch) -> None:
    from llm_wiki_mcp import web_provider

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
    from llm_wiki_mcp import web_provider
    from llm_wiki_mcp.web_provider import HttpSearchProvider

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
        config=WebConfig(adapter_enabled=True, live_egress_enabled=True, provider="searxng"),
        provider=provider,
    )

    assert result.status == "ok"
    assert result.results[0].title == "Official docs"
    assert "bounded research" in (tmp_path / "trace.jsonl").read_text()


def test_mediawiki_adapter_is_a_keyless_bounded_fallback() -> None:
    import httpx
    from llm_wiki_mcp.web_provider import HttpSearchProvider

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
    assert rows[0].url == "https://ja.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%9F%A5%E8%83%BD"
