from __future__ import annotations

import gzip
import inspect

import httpx

from chronovisor.research import research_tools
from chronovisor.research.web_fetch import fetch_web
from chronovisor.search.research_config import WebConfig


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
        == "private_or_special_address"
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


def test_disabled_fetch_does_no_dns_and_cache_hit_rechecks_policy(tmp_path) -> None:
    def forbidden_resolver(_host, _port):
        raise AssertionError("disabled fetch must not resolve DNS")

    disabled = fetch_web(
        "https://example.com/page",
        config=WebConfig(),
        resolver=forbidden_resolver,
        cache_dir=tmp_path / "disabled",
    )
    assert disabled.error == "live egress disabled"

    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                content=b"cached",
            )
        )
    )
    config = WebConfig(adapter_enabled=True, live_egress_enabled=True)
    assert (
        fetch_web(
            "https://example.com/page",
            config=config,
            client=client,
            resolver=PUBLIC,
            cache_dir=tmp_path / "cache",
        ).status
        == "ok"
    )
    blocked = fetch_web(
        "https://example.com/page",
        config=config,
        client=client,
        resolver=lambda _host, _port: ["127.0.0.1"],
        cache_dir=tmp_path / "cache",
    )
    assert blocked.error == "private_or_special_address"


def test_production_web_tool_never_injects_unpinned_client() -> None:
    assert "client=" not in inspect.getsource(research_tools.web_fetch)


def test_public_only_fetch_does_not_reuse_private_policy_cache(tmp_path) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text=f"response-{calls}",
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    private = fetch_web(
        "http://internal.example/page",
        config=WebConfig(
            adapter_enabled=True,
            live_egress_enabled=True,
            allow_private_network=True,
        ),
        client=client,
        resolver=lambda _host, _port: ["127.0.0.1"],
        cache_dir=tmp_path,
    )
    public = fetch_web(
        "http://internal.example/page",
        config=WebConfig(adapter_enabled=True, live_egress_enabled=True),
        client=client,
        resolver=PUBLIC,
        cache_dir=tmp_path,
    )

    assert private.network_policy == "private_network_allowed"
    assert public.network_policy == "public_only"
    assert public.text == "response-2"
    assert calls == 2
