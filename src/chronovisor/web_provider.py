"""Provider-neutral Web search with explicit egress policy and trace."""

from __future__ import annotations

import json
import os
import time
import re
from html import unescape
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

import httpx

from chronovisor.jsonl_write import append_jsonl_durable
from chronovisor.research_config import WebConfig
from chronovisor.research_security import guard_egress_query, guard_url
from chronovisor.store import CHRONOVISOR_ROOT

WEB_TRACE = CHRONOVISOR_ROOT / "runtime" / "research" / "web-egress.jsonl"
USER_AGENT = "Chronovisor/0.1 (+https://github.com/trafficsign/chronovisor)"


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    rank: int
    provider: str


@dataclass(frozen=True)
class SearchResponse:
    status: str
    query: str
    provider: str
    results: tuple[SearchResult, ...] = ()
    latency_ms: int = 0
    error: str = ""
    policy_reason: str = "allowed"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["results"] = [asdict(item) for item in self.results]
        return payload


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, *, limit: int) -> list[SearchResult]: ...


class FixtureSearchProvider:
    name = "fixture"

    def __init__(self, fixture: dict[str, list[dict[str, str]]]) -> None:
        self.fixture = fixture

    def search(self, query: str, *, limit: int) -> list[SearchResult]:
        return [
            SearchResult(
                title=str(row.get("title") or ""),
                url=str(row.get("url") or ""),
                snippet=str(row.get("snippet") or ""),
                rank=index + 1,
                provider=self.name,
            )
            for index, row in enumerate(self.fixture.get(query, [])[:limit])
        ]


class HttpSearchProvider:
    def __init__(
        self,
        *,
        name: str,
        endpoint: str,
        api_key: str = "",
        timeout_seconds: float = 8.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.name = name
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.client = client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
            headers={"User-Agent": USER_AGENT},
        )

    def search(self, query: str, *, limit: int) -> list[SearchResult]:
        if self.name == "mediawiki":
            response = self.client.get(
                self.endpoint,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": limit,
                    "utf8": 1,
                    "format": "json",
                    "formatversion": 2,
                },
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("query", {}).get("search", []) if isinstance(payload, dict) else []
            host = urlsplit(self.endpoint).hostname or "ja.wikipedia.org"
            results: list[SearchResult] = []
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                title = str(row.get("title") or "")
                snippet = unescape(re.sub(r"<[^>]+>", "", str(row.get("snippet") or "")))
                results.append(
                    SearchResult(
                        title=title,
                        url=f"https://{host}/wiki/{quote(title.replace(' ', '_'))}",
                        snippet=snippet[:2_000],
                        rank=len(results) + 1,
                        provider=self.name,
                    )
                )
            return results
        if self.name == "brave":
            response = self.client.get(
                self.endpoint,
                params={"q": query, "count": limit},
                headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"},
            )
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("web", {}).get("results", []) if isinstance(payload, dict) else []
            return _normalize_rows(rows, provider=self.name, limit=limit, snippet_key="description")
        if self.name == "tavily":
            response = self.client.post(
                self.endpoint,
                json={"api_key": self.api_key, "query": query, "max_results": limit, "include_raw_content": False},
            )
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("results", []) if isinstance(payload, dict) else []
            return _normalize_rows(rows, provider=self.name, limit=limit, snippet_key="content")
        response = self.client.get(
            self.endpoint.rstrip("/") + "/search",
            params={"q": query, "format": "json", "language": "all"},
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("results", []) if isinstance(payload, dict) else []
        return _normalize_rows(rows, provider=self.name, limit=limit, snippet_key="content")


def _normalize_rows(
    rows: Any,
    *,
    provider: str,
    limit: int,
    snippet_key: str,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    if not isinstance(rows, list):
        return results
    for row in rows:
        if not isinstance(row, dict):
            continue
        url = row.get("url")
        if not isinstance(url, str) or not url:
            continue
        results.append(
            SearchResult(
                title=str(row.get("title") or "")[:500],
                url=url,
                snippet=str(row.get(snippet_key) or "")[:2_000],
                rank=len(results) + 1,
                provider=provider,
            )
        )
        if len(results) >= limit:
            break
    return results


def provider_from_config(config: WebConfig) -> SearchProvider | None:
    name = config.provider.casefold().strip()
    if name not in {"brave", "tavily", "searxng", "mediawiki"}:
        return None
    endpoint = config.endpoint.strip()
    if not endpoint:
        endpoint = {
            "brave": "https://api.search.brave.com/res/v1/web/search",
            "tavily": "https://api.tavily.com/search",
            "searxng": "",
            "mediawiki": "https://ja.wikipedia.org/w/api.php",
        }[name]
    if not endpoint:
        return None
    key = os.getenv(config.api_key_env, "") if config.api_key_env else ""
    if name in {"brave", "tavily"} and not key:
        return None
    return HttpSearchProvider(name=name, endpoint=endpoint, api_key=key)


def search_web(
    query: str,
    *,
    config: WebConfig,
    provider: SearchProvider | None = None,
    limit: int = 5,
) -> SearchResponse:
    started = time.monotonic()
    policy = guard_egress_query(query)
    selected = provider or provider_from_config(config)
    provider_name = getattr(selected, "name", config.provider or "none")
    live = not isinstance(selected, FixtureSearchProvider)
    status = "error"
    error = ""
    results: list[SearchResult] = []
    if not policy.allowed:
        status = "blocked"
        error = policy.reason
    elif selected is None:
        status = "degraded"
        error = "web provider is not configured"
    elif live and (not config.adapter_enabled or not config.live_egress_enabled):
        status = "blocked"
        error = "live egress disabled"
    else:
        if live and isinstance(selected, HttpSearchProvider):
            endpoint_policy, _addresses = guard_url(
                selected.endpoint,
                allow_private_network=config.allow_private_network,
            )
            if not endpoint_policy.allowed:
                status = "blocked"
                error = f"provider endpoint blocked: {endpoint_policy.reason}"
            else:
                status = "pending"
        else:
            status = "pending"
        if status == "pending":
            for attempt in range(2):
                try:
                    results = selected.search(policy.normalized, limit=max(1, min(10, limit)))
                    status = "ok"
                    break
                except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
                    error = f"{exc.__class__.__name__}: {exc}"
                    status = "degraded"
                    if attempt == 0:
                        time.sleep(0.05)
    response = SearchResponse(
        status=status,
        query=policy.normalized if policy.allowed else "",
        provider=provider_name,
        results=tuple(results),
        latency_ms=round((time.monotonic() - started) * 1000),
        error=error,
        policy_reason=policy.reason,
    )
    append_jsonl_durable(
        WEB_TRACE,
        [
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "kind": "web_search",
                "provider": provider_name,
                "query": response.query,
                "policy_reason": policy.reason,
                "status": status,
                "result_count": len(results),
                "latency_ms": response.latency_ms,
            }
        ],
    )
    return response
