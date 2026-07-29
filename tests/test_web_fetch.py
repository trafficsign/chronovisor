from __future__ import annotations

import gzip

import httpx

from chronovisor.research.research_config import WebConfig
from chronovisor.research.web_fetch import fetch_web


def PUBLIC(_host, _port):
    return ["93.184.216.34"]


def test_fetch_extracts_text_and_uses_ttl_cache(tmp_path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<title>Example</title><p>Evidence</p><script>ignore me</script>",
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler), follow_redirects=False
    )
    config = WebConfig(adapter_enabled=True, live_egress_enabled=True)
    first = fetch_web(
        "https://example.com/page",
        config=config,
        client=client,
        resolver=PUBLIC,
        cache_dir=tmp_path,
    )
    second = fetch_web(
        "https://example.com/page",
        config=config,
        client=client,
        resolver=PUBLIC,
        cache_dir=tmp_path,
    )

    assert first.status == "ok"
    assert "Evidence" in first.text and "ignore me" not in first.text
    assert second.cache == "hit"
    assert len(calls) == 1


def test_fetch_blocks_cross_host_redirect_binary_and_oversized_body(tmp_path) -> None:
    responses = {
        "/redirect": httpx.Response(
            302, headers={"location": "https://evil.example/x"}
        ),
        "/binary": httpx.Response(
            200, headers={"content-type": "application/octet-stream"}, content=b"x"
        ),
        "/large": httpx.Response(
            200, headers={"content-type": "text/plain"}, content=b"x" * 20
        ),
    }
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: responses[request.url.path]),
        follow_redirects=False,
    )
    config = WebConfig(
        adapter_enabled=True, live_egress_enabled=True, max_fetch_bytes=10
    )

    assert (
        fetch_web(
            "https://example.com/redirect",
            config=config,
            client=client,
            resolver=PUBLIC,
            cache_dir=tmp_path / "a",
        ).error
        == "cross_host_redirect"
    )
    assert (
        fetch_web(
            "https://example.com/binary",
            config=config,
            client=client,
            resolver=PUBLIC,
            cache_dir=tmp_path / "b",
        ).error
        == "unsupported_mime"
    )
    assert (
        fetch_web(
            "https://example.com/large",
            config=config,
            client=client,
            resolver=PUBLIC,
            cache_dir=tmp_path / "c",
        ).error
        == "declared_body_too_large"
    )


def test_fetch_blocks_redirect_loop_dns_rebinding_and_compression_bomb(
    tmp_path,
) -> None:
    compressed = gzip.compress(b"x" * 200)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/loop":
            return httpx.Response(302, headers={"location": "/loop"})
        if request.url.path == "/rebind":
            return httpx.Response(
                200, headers={"content-type": "text/plain"}, content=b"small"
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/plain", "content-encoding": "gzip"},
            content=compressed,
        )

    client = httpx.Client(
        transport=httpx.MockTransport(handler), follow_redirects=False
    )
    config = WebConfig(
        adapter_enabled=True, live_egress_enabled=True, max_fetch_bytes=50
    )
    calls = 0

    def rebind(_host, _port):
        nonlocal calls
        calls += 1
        return ["93.184.216.34"] if calls == 1 else ["127.0.0.1"]

    assert (
        fetch_web(
            "https://example.com/loop",
            config=config,
            client=client,
            resolver=PUBLIC,
            cache_dir=tmp_path / "loop",
        ).error
        == "redirect_loop"
    )
    assert (
        fetch_web(
            "https://example.com/rebind",
            config=config,
            client=client,
            resolver=rebind,
            cache_dir=tmp_path / "rebind",
        ).error
        == "dns_rebinding_detected"
    )
    assert (
        fetch_web(
            "https://example.com/bomb",
            config=config,
            client=client,
            resolver=PUBLIC,
            cache_dir=tmp_path / "bomb",
        ).error
        == "body_too_large"
    )


def test_local_search_backend_exception_does_not_open_web_fetch_ssrf(tmp_path) -> None:
    config = WebConfig(
        adapter_enabled=True,
        live_egress_enabled=True,
        allow_local_search_backend=True,
        allow_private_network=False,
    )

    result = fetch_web(
        "http://127.0.0.1:8888/private",
        config=config,
        cache_dir=tmp_path,
    )

    assert result.status == "blocked"
    assert result.error == "private_or_special_address"
