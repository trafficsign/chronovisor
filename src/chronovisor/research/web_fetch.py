"""SSRF-resistant, bounded Web fetch separate from search permission."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
import zstandard

from chronovisor.research.research_config import WebConfig
from chronovisor.research.research_security import (
    Resolver,
    external_content_metadata,
    guard_url,
    resolve_host,
)
from chronovisor.research.research_store import ResearchStore
from chronovisor.core.store import CHRONOVISOR_ROOT

CACHE_DIR = CHRONOVISOR_ROOT / "runtime" / "research" / "web-cache"
ALLOWED_MIME = (
    "text/",
    "application/json",
    "application/xml",
    "application/xhtml+xml",
)
USER_AGENT = "Chronovisor/0.1 (+https://github.com/trafficsign/chronovisor)"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in {"script", "style", "noscript", "svg"}:
            self.hidden += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style", "noscript", "svg"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


@dataclass(frozen=True)
class FetchResponse:
    status: str
    url: str
    final_url: str = ""
    title: str = ""
    text: str = ""
    mime_type: str = ""
    byte_length: int = 0
    sha256: str = ""
    artifact_id: str = ""
    cache: str = "miss"
    latency_ms: int = 0
    redirects: tuple[str, ...] = ()
    error: str = ""
    security: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["redirects"] = list(self.redirects)
        return payload


def _cache_paths(url: str, cache_dir: Path) -> tuple[Path, Path]:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json", cache_dir / f"{digest}.zst"


def _atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def _cached(url: str, *, cache_dir: Path, ttl_seconds: int) -> FetchResponse | None:
    meta_path, body_path = _cache_paths(url, cache_dir)
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        stored = datetime.fromisoformat(str(metadata["cached_at"]))
        if stored.tzinfo is None:
            stored = stored.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - stored > timedelta(seconds=max(0, ttl_seconds)):
            return None
        text = zstandard.ZstdDecompressor().decompress(body_path.read_bytes()).decode("utf-8")
        payload = dict(metadata["response"])
        payload["text"] = text
        payload["cache"] = "hit"
        payload["redirects"] = tuple(payload.get("redirects") or ())
        return FetchResponse(**payload)
    except (OSError, KeyError, ValueError, json.JSONDecodeError, zstandard.ZstdError, TypeError):
        return None


def _save_cache(response: FetchResponse, *, cache_dir: Path) -> None:
    meta_path, body_path = _cache_paths(response.url, cache_dir)
    payload = response.to_dict()
    text = str(payload.pop("text", ""))
    _atomic(body_path, zstandard.ZstdCompressor(level=6).compress(text.encode("utf-8")))
    _atomic(
        meta_path,
        (
            json.dumps(
                {"cached_at": datetime.now(timezone.utc).isoformat(), "response": payload},
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _extract_text(raw: bytes, mime_type: str, encoding: str | None) -> tuple[str, str]:
    text = raw.decode(encoding or "utf-8", errors="replace")
    title = ""
    if mime_type.startswith("text/html") or mime_type.startswith("application/xhtml"):
        parser = _TextExtractor()
        parser.feed(text)
        text = "\n".join(parser.parts)
        import re

        match = re.search(r"<title[^>]*>(.*?)</title>", raw.decode("utf-8", errors="ignore"), re.I | re.S)
        if match:
            title = " ".join(match.group(1).split())[:500]
    return text, title


def fetch_web(
    url: str,
    *,
    config: WebConfig,
    client: httpx.Client | None = None,
    resolver: Resolver = resolve_host,
    store: ResearchStore | None = None,
    cache_dir: Path = CACHE_DIR,
    max_redirects: int = 3,
) -> FetchResponse:
    started = time.monotonic()
    cached = _cached(url, cache_dir=cache_dir, ttl_seconds=config.cache_ttl_seconds)
    if cached is not None:
        return cached
    if not config.adapter_enabled or not config.live_egress_enabled:
        return FetchResponse(
            status="blocked",
            url=url,
            error="live egress disabled",
            latency_ms=round((time.monotonic() - started) * 1000),
        )
    own_client = client is None
    http = client or httpx.Client(
        timeout=10.0,
        follow_redirects=False,
        headers={"User-Agent": USER_AGENT},
    )
    current = url
    redirects: list[str] = []
    visited: set[str] = set()
    origin_host = urlsplit(url).hostname
    try:
        for _attempt in range(max_redirects + 1):
            if current in visited:
                return FetchResponse(status="blocked", url=url, error="redirect_loop", redirects=tuple(redirects))
            visited.add(current)
            policy, before_addresses = guard_url(
                current,
                allow_private_network=config.allow_private_network,
                resolver=resolver,
            )
            if not policy.allowed:
                return FetchResponse(status="blocked", url=url, error=policy.reason, redirects=tuple(redirects))
            with http.stream(
                "GET",
                current,
                headers={
                    "Accept": "text/html,text/plain,application/json,application/xml",
                    "User-Agent": USER_AGENT,
                },
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        return FetchResponse(status="error", url=url, error="redirect_without_location")
                    target = urljoin(current, location)
                    if urlsplit(target).hostname != origin_host:
                        return FetchResponse(status="blocked", url=url, error="cross_host_redirect", redirects=tuple([*redirects, target]))
                    redirects.append(target)
                    current = target
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0].casefold().strip()
                if not any(content_type.startswith(prefix) for prefix in ALLOWED_MIME):
                    return FetchResponse(status="blocked", url=url, error="unsupported_mime", mime_type=content_type)
                declared = response.headers.get("content-length")
                if declared and int(declared) > config.max_fetch_bytes:
                    return FetchResponse(status="blocked", url=url, error="declared_body_too_large", mime_type=content_type)
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > config.max_fetch_bytes:
                        return FetchResponse(status="blocked", url=url, error="body_too_large", mime_type=content_type)
                    chunks.append(chunk)
                raw = b"".join(chunks)
                final_policy, after_addresses = guard_url(
                    current,
                    allow_private_network=config.allow_private_network,
                    resolver=resolver,
                )
                if not final_policy.allowed or set(before_addresses) != set(after_addresses):
                    return FetchResponse(status="blocked", url=url, error="dns_rebinding_detected")
                text, title = _extract_text(raw, content_type, response.encoding)
                digest = hashlib.sha256(raw).hexdigest()
                artifact_id = ""
                if store is not None:
                    artifact = store.put_artifact(
                        text,
                        source_type="web",
                        source_uri=current,
                        title=title,
                        mime_type=content_type,
                        citation=current,
                        trust="untrusted",
                        metadata=external_content_metadata(text),
                    )
                    artifact_id = artifact.artifact_id
                result = FetchResponse(
                    status="ok",
                    url=url,
                    final_url=current,
                    title=title,
                    text=text,
                    mime_type=content_type,
                    byte_length=len(raw),
                    sha256=digest,
                    artifact_id=artifact_id,
                    latency_ms=round((time.monotonic() - started) * 1000),
                    redirects=tuple(redirects),
                    security=external_content_metadata(text),
                )
                _save_cache(result, cache_dir=cache_dir)
                return result
        return FetchResponse(status="blocked", url=url, error="too_many_redirects", redirects=tuple(redirects))
    except (httpx.HTTPError, ValueError, UnicodeError) as exc:
        return FetchResponse(
            status="degraded",
            url=url,
            error=f"{exc.__class__.__name__}: {exc}",
            latency_ms=round((time.monotonic() - started) * 1000),
            redirects=tuple(redirects),
        )
    finally:
        if own_client:
            http.close()
