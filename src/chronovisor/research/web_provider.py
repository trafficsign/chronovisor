"""Provider-neutral Web search with explicit egress policy and trace."""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from chronovisor.core.jsonl_write import append_jsonl_durable
from chronovisor.research.research_config import WebConfig
from chronovisor.research.research_security import guard_egress_query, guard_url
from chronovisor.core.store import CHRONOVISOR_ROOT

WEB_TRACE = CHRONOVISOR_ROOT / "runtime" / "research" / "web-egress.jsonl"
USER_AGENT = "Chronovisor/0.1 (+https://github.com/trafficsign/chronovisor)"
ADOPTED_SOURCE_PACKS = frozenset({"general", "code", "academic", "encyclopedia"})
_CODE_STRONG_TERMS = (
    "github",
    "repository",
    "repo ",
    "source code",
    "commit",
    "pull request",
    "changelog",
    "package",
    "library",
    "framework",
    "リポジトリ",
    "ソースコード",
    "コミット",
    "プルリク",
    "ライブラリ",
    "パッケージ",
)
_RELEASE_TERMS = ("release", "version", "update", "リリース", "バージョン", "更新")
_TECH_TERMS = (
    "software",
    "model",
    "llm",
    "api",
    "cli",
    "sdk",
    "ソフトウェア",
    "モデル",
)
_ACADEMIC_TERMS = (
    "arxiv",
    "paper",
    "preprint",
    "journal",
    "doi",
    "study",
    "research",
    "論文",
    "研究",
    "査読",
    "学術",
    "文献",
)
_ENCYCLOPEDIA_TERMS = (
    "what is",
    "who is",
    "definition",
    "history of",
    "overview",
    "とは",
    "誰",
    "定義",
    "歴史",
    "概要",
)


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
    routes: tuple[str, ...] = ()
    provider_statuses: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["results"] = [asdict(item) for item in self.results]
        payload["routes"] = list(self.routes)
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
        self.last_statuses: dict[str, str] = {}
        self.client = client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
            headers={"User-Agent": USER_AGENT},
        )

    def search(self, query: str, *, limit: int) -> list[SearchResult]:
        self.last_statuses = {}
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
            rows = (
                payload.get("query", {}).get("search", [])
                if isinstance(payload, dict)
                else []
            )
            host = urlsplit(self.endpoint).hostname or "ja.wikipedia.org"
            results: list[SearchResult] = []
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                title = str(row.get("title") or "")
                snippet = unescape(
                    re.sub(r"<[^>]+>", "", str(row.get("snippet") or ""))
                )
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
        if self.name == "github":
            headers = {
                "Accept": "application/vnd.github+json",
                "User-Agent": USER_AGENT,
                "X-GitHub-Api-Version": "2022-11-28",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = self.client.get(
                self.endpoint,
                params={"q": query, "per_page": limit},
                headers=headers,
            )
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("items", []) if isinstance(payload, dict) else []
            results: list[SearchResult] = []
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                url = row.get("html_url")
                if not isinstance(url, str) or not url:
                    continue
                details = [
                    str(row.get("description") or "").strip(),
                    f"language={row.get('language')}" if row.get("language") else "",
                    f"stars={row.get('stargazers_count')}"
                    if isinstance(row.get("stargazers_count"), int)
                    else "",
                    f"updated={row.get('updated_at')}" if row.get("updated_at") else "",
                ]
                results.append(
                    SearchResult(
                        title=str(row.get("full_name") or row.get("name") or "")[:500],
                        url=url,
                        snippet=" | ".join(item for item in details if item)[:2_000],
                        rank=len(results) + 1,
                        provider=self.name,
                    )
                )
                if len(results) >= limit:
                    break
            return results
        if self.name == "arxiv":
            response = self.client.get(
                self.endpoint,
                params={
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": limit,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
                headers={"Accept": "application/atom+xml", "User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            results: list[SearchResult] = []
            for entry in root.findall("atom:entry", namespace):
                title = " ".join(
                    (entry.findtext("atom:title", "", namespace) or "").split()
                )
                summary = " ".join(
                    (entry.findtext("atom:summary", "", namespace) or "").split()
                )
                url = (entry.findtext("atom:id", "", namespace) or "").strip()
                published = (
                    entry.findtext("atom:published", "", namespace) or ""
                ).strip()
                authors = [
                    " ".join((node.text or "").split())
                    for node in entry.findall("atom:author/atom:name", namespace)
                    if node.text
                ]
                if not url:
                    continue
                details = [
                    summary,
                    f"authors={', '.join(authors[:5])}" if authors else "",
                    f"published={published}" if published else "",
                ]
                results.append(
                    SearchResult(
                        title=title[:500],
                        url=url,
                        snippet=" | ".join(item for item in details if item)[:2_000],
                        rank=len(results) + 1,
                        provider=self.name,
                    )
                )
                if len(results) >= limit:
                    break
            return results
        if self.name == "crossref":
            response = self.client.get(
                self.endpoint,
                params={
                    "query.bibliographic": query,
                    "rows": limit,
                    "select": "DOI,URL,title,abstract,author,published,type",
                },
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            payload = response.json()
            rows = (
                payload.get("message", {}).get("items", [])
                if isinstance(payload, dict)
                else []
            )
            results: list[SearchResult] = []
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                url = row.get("URL")
                if not isinstance(url, str) or not url:
                    doi = str(row.get("DOI") or "")
                    url = f"https://doi.org/{quote(doi)}" if doi else ""
                if not url:
                    continue
                titles = row.get("title")
                title = (
                    str(titles[0])
                    if isinstance(titles, list) and titles
                    else str(titles or "")
                )
                abstract = unescape(
                    re.sub(r"<[^>]+>", "", str(row.get("abstract") or ""))
                )
                authors = row.get("author")
                author_names = []
                for author in authors if isinstance(authors, list) else []:
                    if not isinstance(author, dict):
                        continue
                    name = " ".join(
                        item
                        for item in (
                            str(author.get("given") or "").strip(),
                            str(author.get("family") or "").strip(),
                        )
                        if item
                    )
                    if name:
                        author_names.append(name)
                details = [
                    abstract,
                    f"authors={', '.join(author_names[:5])}" if author_names else "",
                    f"doi={row.get('DOI')}" if row.get("DOI") else "",
                    f"type={row.get('type')}" if row.get("type") else "",
                ]
                results.append(
                    SearchResult(
                        title=title[:500],
                        url=url,
                        snippet=" | ".join(item for item in details if item)[:2_000],
                        rank=len(results) + 1,
                        provider=self.name,
                    )
                )
                if len(results) >= limit:
                    break
            return results
        if self.name == "brave":
            response = self.client.get(
                self.endpoint,
                params={"q": query, "count": limit},
                headers={
                    "X-Subscription-Token": self.api_key,
                    "Accept": "application/json",
                },
            )
            response.raise_for_status()
            payload = response.json()
            rows = (
                payload.get("web", {}).get("results", [])
                if isinstance(payload, dict)
                else []
            )
            return _normalize_rows(
                rows, provider=self.name, limit=limit, snippet_key="description"
            )
        if self.name == "tavily":
            response = self.client.post(
                self.endpoint,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": limit,
                    "include_raw_content": False,
                },
            )
            response.raise_for_status()
            payload = response.json()
            rows = payload.get("results", []) if isinstance(payload, dict) else []
            return _normalize_rows(
                rows, provider=self.name, limit=limit, snippet_key="content"
            )
        response = self.client.get(
            self.endpoint.rstrip("/") + "/search",
            params={"q": query, "format": "json", "language": "all"},
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("results", []) if isinstance(payload, dict) else []
        if self.name == "searxng" and isinstance(payload, dict):
            unresponsive = payload.get("unresponsive_engines")
            for row in unresponsive if isinstance(unresponsive, list) else []:
                if not isinstance(row, (list, tuple)) or not row:
                    continue
                engine = str(row[0] or "unknown")
                detail = str(row[1] or "unresponsive") if len(row) > 1 else "unresponsive"
                self.last_statuses[f"searxng:{engine}"] = f"error:{detail[:160]}"
        return _normalize_rows(
            rows, provider=self.name, limit=limit, snippet_key="content"
        )


def route_source_packs(
    query: str,
    *,
    enabled: tuple[str, ...],
) -> tuple[str, ...]:
    """Choose from the four adopted packs without asking a model or adding APIs."""

    folded = unicodedata.normalize("NFKC", query).casefold()
    selected: list[str] = []
    if "academic" in enabled and any(term in folded for term in _ACADEMIC_TERMS):
        selected.append("academic")
    code_match = any(term in folded for term in _CODE_STRONG_TERMS)
    release_match = any(term in folded for term in _RELEASE_TERMS)
    tech_match = any(term in folded for term in _TECH_TERMS)
    if "code" in enabled and (code_match or (release_match and tech_match)):
        selected.append("code")
    if "encyclopedia" in enabled and any(
        term in folded for term in _ENCYCLOPEDIA_TERMS
    ):
        selected.append("encyclopedia")
    if "general" in enabled:
        selected.append("general")
    if not selected and "encyclopedia" in enabled:
        selected.append("encyclopedia")
    return tuple(dict.fromkeys(selected))


def _canonical_result_url(url: str) -> str:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    port = parsed.port
    netloc = host
    if port and port not in {80, 443}:
        netloc = f"{host}:{port}"
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((parsed.scheme.casefold(), netloc, path, parsed.query, ""))


def _adapt_query(provider_name: str, query: str) -> str:
    """Remove routing vocabulary that narrows specialist APIs to zero hits."""

    removable = {
        "github": (
            "github",
            "repository",
            "repo",
            "source code",
            "リポジトリ",
            "ソースコード",
        ),
        "arxiv": ("arxiv", "paper", "preprint", "論文", "文献"),
        "crossref": ("crossref", "doi", "paper", "journal", "論文", "文献"),
        "mediawiki": (
            "what is",
            "who is",
            "definition",
            "overview",
            "とは",
            "誰",
            "定義",
            "概要",
        ),
    }.get(provider_name, ())
    adapted = unicodedata.normalize("NFKC", query)
    for term in sorted(removable, key=len, reverse=True):
        if term.isascii() and term.replace(" ", "").isalnum():
            adapted = re.sub(
                rf"(?i)(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])",
                " ",
                adapted,
            )
        else:
            adapted = re.sub(re.escape(term), " ", adapted, flags=re.I)
    compacted = " ".join(adapted.split()).strip(" -:|")
    return compacted or query


def _merge_provider_results(
    batches: list[list[SearchResult]],
    *,
    limit: int,
) -> list[SearchResult]:
    merged: list[SearchResult] = []
    seen: set[str] = set()
    depth = 0
    while len(merged) < limit and any(depth < len(batch) for batch in batches):
        for batch in batches:
            if depth >= len(batch):
                continue
            item = batch[depth]
            key = _canonical_result_url(item.url)
            if key in seen:
                continue
            seen.add(key)
            merged.append(
                SearchResult(
                    title=item.title,
                    url=item.url,
                    snippet=item.snippet,
                    rank=len(merged) + 1,
                    provider=item.provider,
                )
            )
            if len(merged) >= limit:
                break
        depth += 1
    return merged


def _loopback_search_endpoint_allowed(
    provider: SearchProvider,
    config: WebConfig,
) -> bool:
    if not config.allow_local_search_backend:
        return False
    if getattr(provider, "name", "") != "searxng":
        return False
    endpoint = str(getattr(provider, "endpoint", ""))
    parsed = urlsplit(endpoint)
    return (
        parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "::1"}
        and not parsed.username
        and not parsed.password
    )


def _provider_endpoint_allowed(
    provider: SearchProvider,
    config: WebConfig,
) -> tuple[bool, str]:
    endpoint = str(getattr(provider, "endpoint", ""))
    if not endpoint:
        return True, "allowed"
    if _loopback_search_endpoint_allowed(provider, config):
        return True, "local_search_backend"
    policy, _addresses = guard_url(endpoint, allow_private_network=False)
    return policy.allowed, policy.reason


class FederatedSearchProvider:
    name = "federated"

    def __init__(
        self,
        *,
        providers: dict[str, SearchProvider],
        source_packs: tuple[str, ...],
        config: WebConfig,
    ) -> None:
        self.providers = providers
        self.source_packs = source_packs
        self.config = config
        self.last_statuses: dict[str, str] = {}

    def routes(self, query: str) -> tuple[str, ...]:
        return route_source_packs(query, enabled=self.source_packs)

    def _provider_names(self, query: str) -> list[str]:
        names: list[str] = []
        mapping = {
            "academic": ("arxiv", "crossref"),
            "code": ("github",),
            "encyclopedia": ("mediawiki",),
            "general": ("searxng",),
        }
        for route in self.routes(query):
            for name in mapping[route]:
                if name in self.providers and name not in names:
                    names.append(name)
                if len(names) >= self.config.max_provider_calls:
                    return names
        return names

    def search(self, query: str, *, limit: int) -> list[SearchResult]:
        self.last_statuses = {}
        names = self._provider_names(query)
        if not names:
            raise ValueError("no adopted source pack has a configured provider")
        per_provider_limit = min(
            max(1, limit),
            self.config.per_provider_limit,
        )
        batches_by_name: dict[str, list[SearchResult]] = {}
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            futures = {}
            for name in names:
                provider = self.providers[name]
                allowed, reason = _provider_endpoint_allowed(provider, self.config)
                if not allowed:
                    errors.append(f"{name}: endpoint blocked ({reason})")
                    self.last_statuses[name] = f"blocked:{reason}"
                    continue
                futures[
                    pool.submit(
                        provider.search,
                        _adapt_query(name, query),
                        limit=per_provider_limit,
                    )
                ] = name
            for future in as_completed(futures):
                name = futures[future]
                try:
                    batches_by_name[name] = future.result()
                    self.last_statuses[name] = f"ok:{len(batches_by_name[name])}"
                except (httpx.HTTPError, ValueError, ET.ParseError) as exc:
                    errors.append(f"{name}: {exc.__class__.__name__}: {exc}")
                    self.last_statuses[name] = f"error:{exc.__class__.__name__}"
        ordered_batches = [
            batches_by_name[name] for name in names if name in batches_by_name
        ]
        merged = _merge_provider_results(ordered_batches, limit=limit)
        if (
            len(merged) < min(2, limit)
            and "mediawiki" in self.providers
            and "mediawiki" not in names
            and len(names) < self.config.max_provider_calls
        ):
            fallback = self.providers["mediawiki"]
            allowed, reason = _provider_endpoint_allowed(fallback, self.config)
            if allowed:
                try:
                    fallback_rows = fallback.search(query, limit=per_provider_limit)
                    self.last_statuses["mediawiki"] = f"fallback:{len(fallback_rows)}"
                    merged = _merge_provider_results(
                        [merged, fallback_rows], limit=limit
                    )
                except (httpx.HTTPError, ValueError, ET.ParseError) as exc:
                    errors.append(f"mediawiki: {exc.__class__.__name__}: {exc}")
                    self.last_statuses["mediawiki"] = f"error:{exc.__class__.__name__}"
            else:
                errors.append(f"mediawiki: endpoint blocked ({reason})")
                self.last_statuses["mediawiki"] = f"blocked:{reason}"
        if not merged and errors:
            raise ValueError("; ".join(errors))
        return merged


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
    if name == "federated":
        unknown = sorted(set(config.source_packs) - ADOPTED_SOURCE_PACKS)
        if unknown:
            return None
        timeout = config.provider_timeout_seconds
        providers: dict[str, SearchProvider] = {}
        endpoints = {
            "searxng": config.searxng_endpoint.strip(),
            "github": config.github_endpoint.strip(),
            "arxiv": config.arxiv_endpoint.strip(),
            "crossref": config.crossref_endpoint.strip(),
            "mediawiki": config.mediawiki_endpoint.strip(),
        }
        enabled_provider_names: set[str] = set()
        if "general" in config.source_packs:
            enabled_provider_names.add("searxng")
        if "code" in config.source_packs:
            enabled_provider_names.add("github")
        if "academic" in config.source_packs:
            enabled_provider_names.update({"arxiv", "crossref"})
        if "encyclopedia" in config.source_packs:
            enabled_provider_names.add("mediawiki")
        for provider_name in sorted(enabled_provider_names):
            endpoint = endpoints[provider_name]
            if not endpoint:
                continue
            api_key = ""
            if provider_name == "github" and config.github_token_env:
                api_key = os.getenv(config.github_token_env, "")
            providers[provider_name] = HttpSearchProvider(
                name=provider_name,
                endpoint=endpoint,
                api_key=api_key,
                timeout_seconds=timeout,
            )
        if not providers:
            return None
        return FederatedSearchProvider(
            providers=providers,
            source_packs=config.source_packs,
            config=config,
        )
    if name not in {
        "brave",
        "tavily",
        "searxng",
        "mediawiki",
        "github",
        "arxiv",
        "crossref",
    }:
        return None
    endpoint = config.endpoint.strip()
    if not endpoint:
        endpoint = {
            "brave": "https://api.search.brave.com/res/v1/web/search",
            "tavily": "https://api.tavily.com/search",
            "searxng": "",
            "mediawiki": "https://ja.wikipedia.org/w/api.php",
            "github": "https://api.github.com/search/repositories",
            "arxiv": "https://export.arxiv.org/api/query",
            "crossref": "https://api.crossref.org/works",
        }[name]
    if not endpoint:
        return None
    key = os.getenv(config.api_key_env, "") if config.api_key_env else ""
    if name in {"brave", "tavily"} and not key:
        return None
    return HttpSearchProvider(
        name=name,
        endpoint=endpoint,
        api_key=key,
        timeout_seconds=config.provider_timeout_seconds,
    )


def _provider_config_error(config: WebConfig) -> str:
    if config.provider.casefold().strip() != "federated":
        return ""
    unknown = sorted(set(config.source_packs) - ADOPTED_SOURCE_PACKS)
    if unknown:
        return f"source packs are not adopted: {', '.join(unknown)}"
    return ""


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
    routes_function = getattr(selected, "routes", None)
    routes = (
        tuple(routes_function(policy.normalized))
        if policy.allowed and callable(routes_function)
        else ()
    )
    live = not isinstance(selected, FixtureSearchProvider)
    status = "error"
    error = ""
    results: list[SearchResult] = []
    provider_statuses: dict[str, str] = {}
    if not policy.allowed:
        status = "blocked"
        error = policy.reason
    elif selected is None:
        status = "degraded"
        error = _provider_config_error(config) or "web provider is not configured"
    elif live and (not config.adapter_enabled or not config.live_egress_enabled):
        status = "blocked"
        error = "live egress disabled"
    else:
        if live and isinstance(selected, HttpSearchProvider):
            endpoint_allowed, endpoint_reason = _provider_endpoint_allowed(
                selected, config
            )
            if not endpoint_allowed:
                status = "blocked"
                error = f"provider endpoint blocked: {endpoint_reason}"
            else:
                status = "pending"
        else:
            status = "pending"
        if status == "pending":
            for attempt in range(2):
                try:
                    results = selected.search(
                        policy.normalized, limit=max(1, min(10, limit))
                    )
                    observed_statuses = getattr(selected, "last_statuses", None)
                    if isinstance(observed_statuses, dict):
                        provider_statuses = {
                            str(name): str(value)
                            for name, value in observed_statuses.items()
                        }
                    partial_errors = [
                        f"{name}={value}"
                        for name, value in sorted(provider_statuses.items())
                        if value.startswith(("error:", "blocked:"))
                    ]
                    if results and partial_errors:
                        status = "degraded"
                        error = "partial provider degradation: " + "; ".join(
                            partial_errors
                        )
                    else:
                        status = "ok"
                    break
                except (
                    httpx.HTTPError,
                    ValueError,
                    json.JSONDecodeError,
                    ET.ParseError,
                ) as exc:
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
        routes=routes,
        provider_statuses=provider_statuses,
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
                "routes": list(routes),
                "result_providers": sorted({item.provider for item in results}),
                "provider_statuses": provider_statuses,
                "latency_ms": response.latency_ms,
            }
        ],
    )
    return response
