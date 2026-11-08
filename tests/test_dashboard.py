"""Capture dashboard and retained LAN boundary regression tests."""

from __future__ import annotations

import base64
import json
import os
import socket
import ssl
import subprocess
import threading
import time
from datetime import date, datetime
from pathlib import Path

import httpx
import pytest

from chronovisor.ops import dashboard


def test_dashboard_private_client_scope_rejects_public_addresses() -> None:
    assert dashboard._private_client_scope("127.0.0.1") == "loopback"
    assert dashboard._private_client_scope("192.168.1.22") == "private"
    assert dashboard._private_client_scope("10.1.2.3") == "private"
    assert dashboard._private_client_scope("169.254.1.2") == "private"
    assert dashboard._private_client_scope("192.0.0.1") == "public"
    assert dashboard._private_client_scope("100.64.0.1") == "public"
    assert dashboard._private_client_scope("8.8.8.8") == "public"
    assert dashboard._private_client_scope("not-an-ip") == "invalid"


def test_dashboard_static_path_resolver_rejects_directory_and_symlink_escapes(
    tmp_path: Path,
) -> None:
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    inside = static_dir / "app.js"
    inside.write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.js"
    outside.write_text("outside", encoding="utf-8")
    (static_dir / "linked.js").symlink_to(outside)

    assert dashboard._resolve_static_path(static_dir, "/static/app.js") == inside
    assert dashboard._resolve_static_path(static_dir, "/static/../outside.js") is None
    assert dashboard._resolve_static_path(static_dir, "/static/linked.js") is None


def test_dashboard_host_allowlist_uses_loopback_names_and_actual_port() -> None:
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        responses = [
            httpx.get(
                f"http://{host}:{port}/api/lan-access",
                headers={"Host": allowed},
                timeout=2,
            )
            for allowed in (
                f"localhost:{port}",
                f"127.0.0.1:{port}",
                f"[::1]:{port}",
            )
        ]
        wrong_name = httpx.get(
            f"http://{host}:{port}/api/lan-access",
            headers={"Host": f"attacker.example:{port}"},
            timeout=2,
        )
        wrong_port = httpx.get(
            f"http://{host}:{port}/api/lan-access",
            headers={"Host": "127.0.0.1:1"},
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert wrong_name.status_code == 421
    assert wrong_port.status_code == 421
    assert wrong_name.headers["x-frame-options"] == "DENY"


def test_dashboard_http_origin_is_same_origin_or_absent() -> None:
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    request_host = f"127.0.0.1:{port}"
    url = f"http://{host}:{port}/api/lan-access"
    try:
        missing = httpx.get(url, headers={"Host": request_host}, timeout=2)
        same = httpx.get(
            url,
            headers={"Host": request_host, "Origin": f"http://{request_host}"},
            timeout=2,
        )
        cross = httpx.get(
            url,
            headers={"Host": request_host, "Origin": "http://attacker.example"},
            timeout=2,
        )
        other_loopback_origin = httpx.get(
            url,
            headers={
                "Host": request_host,
                "Origin": f"http://localhost:{port}",
            },
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert missing.status_code == 200
    assert same.status_code == 200
    assert cross.status_code == 403
    assert other_loopback_origin.status_code == 403


def test_dashboard_websocket_requires_same_origin() -> None:
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address

    def websocket_status(origin: str | None) -> int:
        origin_header = f"Origin: {origin}\r\n" if origin else ""
        request = (
            "GET /api/cortex/events HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            f"{origin_header}\r\n"
        ).encode("ascii")
        with socket.create_connection((host, port), timeout=2) as connection:
            connection.sendall(request)
            status_line = connection.recv(4096).split(b"\r\n", 1)[0]
        return int(status_line.split()[1])

    try:
        missing = websocket_status(None)
        cross = websocket_status("http://attacker.example")
        same = websocket_status(f"http://127.0.0.1:{port}")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert missing == 403
    assert cross == 403
    assert same == 101


def test_dashboard_disables_lan_share_cookie_and_non_loopback_bind() -> None:
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        response = httpx.get(
            f"http://{host}:{port}/api/lan-access?access_token={'a' * 43}",
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.json() == {
        "enabled": False,
        "urls": [],
        "trusted_lan_only": False,
    }
    assert "set-cookie" not in response.headers
    with pytest.raises(ValueError, match="explicit private IPv4"):
        dashboard.serve("127.0.0.1", 0, lan=True)
    with pytest.raises(ValueError, match="explicit private IPv4"):
        dashboard.serve("0.0.0.0", 0, lan=True)
    with pytest.raises(ValueError, match="dashboard host must be"):
        dashboard.serve("0.0.0.0", 0)
    with pytest.raises(ValueError, match="dashboard host must be"):
        dashboard.serve("::1", 0)


def test_dashboard_lan_requires_basic_then_uses_secure_bounded_session(
    tmp_path: Path, monkeypatch
) -> None:
    credentials_path = tmp_path / "dashboard-credentials.json"
    dashboard._write_dashboard_credentials(
        credentials_path, "admin", "correct horse battery staple"
    )
    with pytest.raises(ValueError, match="safe ASCII"):
        dashboard._write_dashboard_credentials(
            credentials_path, "管理者", "correct horse battery staple"
        )
    credentials = dashboard._load_dashboard_credentials(credentials_path)
    assert not dashboard._dashboard_credentials_match(
        credentials, "管理者", "correct horse battery staple"
    )
    monkeypatch.setattr(dashboard, "_private_client_scope", lambda _value: "private")
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    server.lan_access_enabled = True
    server.dashboard_public_host = "192.168.50.20"
    server.dashboard_credentials = credentials
    server.dashboard_auth_lock = threading.Lock()
    server.dashboard_auth_slots = threading.BoundedSemaphore(1)
    server.dashboard_stream_slots = threading.BoundedSemaphore(
        dashboard.DASHBOARD_STREAM_LIMIT
    )
    server.dashboard_login_attempts = {}
    server.dashboard_sessions = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    headers = {
        "Host": f"192.168.50.20:{port}",
        "Origin": f"https://192.168.50.20:{port}",
    }
    try:
        with httpx.Client(follow_redirects=False, timeout=2) as client:
            challenge = client.get(f"http://{host}:{port}/", headers=headers)
            rejected = client.get(
                f"http://{host}:{port}/", headers=headers, auth=("admin", "wrong")
            )
            assert server.dashboard_auth_slots.acquire(blocking=False)
            saturated = client.get(
                f"http://{host}:{port}/",
                headers=headers,
                auth=("admin", "correct horse battery staple"),
            )
            server.dashboard_auth_slots.release()
            accepted = client.get(
                f"http://{host}:{port}/",
                headers=headers,
                auth=("admin", "correct horse battery staple"),
            )
            session_cookie = accepted.headers["set-cookie"].split(";", 1)[0]
            page = client.get(
                f"http://{host}:{port}/",
                headers={**headers, "Cookie": session_cookie},
            )
            duplicate_cookie = client.get(
                f"http://{host}:{port}/",
                headers=[
                    *headers.items(),
                    ("Cookie", session_cookie),
                    ("Cookie", session_cookie),
                ],
            )
            server.dashboard_sessions.clear()
            server.dashboard_login_attempts[host] = [
                time.monotonic()
            ] * dashboard.DASHBOARD_LOGIN_ATTEMPT_LIMIT
            rate_limited = client.get(
                f"http://{host}:{port}/",
                headers=headers,
                auth=("admin", "correct horse battery staple"),
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert challenge.status_code == 401
    assert rejected.status_code == 401
    assert saturated.status_code == 429
    assert accepted.status_code == 200
    assert "Chronovisor Observatory" in accepted.text
    assert "location" not in accepted.headers
    assert "Secure" in accepted.headers["set-cookie"]
    assert "HttpOnly" in accepted.headers["set-cookie"]
    assert "SameSite=Strict" in accepted.headers["set-cookie"]
    assert page.status_code == 200
    assert duplicate_cookie.status_code == 401
    assert rate_limited.status_code == 429
    assert len(server.dashboard_sessions) == 0
    assert credentials_path.stat().st_mode & 0o777 == 0o600
    assert "correct horse battery staple" not in credentials_path.read_text()


def test_dashboard_lan_tls_server_and_private_key_permissions(
    tmp_path: Path, monkeypatch
) -> None:
    key = tmp_path / "dashboard.key"
    cert = tmp_path / "dashboard.crt"
    credentials_path = tmp_path / "dashboard-credentials.json"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-subj",
            "/CN=192.168.50.20",
        ],
        check=True,
        capture_output=True,
    )
    key.chmod(0o600)
    dashboard._write_dashboard_credentials(
        credentials_path, "admin", "correct horse battery staple"
    )
    monkeypatch.setattr(dashboard, "_private_client_scope", lambda _value: "private")
    server = dashboard._ThreadingHTTPSServer(
        ("127.0.0.1", 0), dashboard.DashboardHandler, certfile=cert, keyfile=key
    )
    server.lan_access_enabled = True
    server.dashboard_public_host = "192.168.50.20"
    server.dashboard_credentials = dashboard._load_dashboard_credentials(
        credentials_path
    )
    server.dashboard_auth_lock = threading.Lock()
    server.dashboard_auth_slots = threading.BoundedSemaphore(
        dashboard.DASHBOARD_AUTH_CONCURRENCY_LIMIT
    )
    server.dashboard_stream_slots = threading.BoundedSemaphore(
        dashboard.DASHBOARD_STREAM_LIMIT
    )
    server.dashboard_login_attempts = {}
    server.dashboard_sessions = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        response = httpx.get(
            f"https://{host}:{port}/",
            headers={
                "Host": f"192.168.50.20:{port}",
                "Origin": f"https://192.168.50.20:{port}",
            },
            auth=("admin", "correct horse battery staple"),
            verify=False,
            follow_redirects=False,
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.status_code == 200
    assert "set-cookie" in response.headers

    key.chmod(0o644)
    with pytest.raises(RuntimeError, match="0600"):
        dashboard._read_private_file(key, "dashboard TLS private key")
    link = tmp_path / "linked.key"
    link.symlink_to(key)
    with pytest.raises(RuntimeError, match="unavailable"):
        dashboard._read_private_file(link, "dashboard TLS private key")


def test_dashboard_tls_slow_handshake_does_not_block_and_handler_slots_recover(
    tmp_path: Path, monkeypatch
) -> None:
    key = tmp_path / "dashboard.key"
    cert = tmp_path / "dashboard.crt"
    credentials_path = tmp_path / "dashboard-credentials.json"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-subj",
            "/CN=192.168.50.20",
        ],
        check=True,
        capture_output=True,
    )
    key.chmod(0o600)
    dashboard._write_dashboard_credentials(
        credentials_path, "admin", "correct horse battery staple"
    )
    monkeypatch.setattr(dashboard, "_private_client_scope", lambda _value: "private")
    server = dashboard._ThreadingHTTPSServer(
        ("127.0.0.1", 0),
        dashboard.DashboardHandler,
        certfile=cert,
        keyfile=key,
        handshake_timeout=2,
        io_timeout=0.2,
        handler_limit=2,
    )
    server.lan_access_enabled = True
    server.dashboard_public_host = "192.168.50.20"
    server.dashboard_credentials = dashboard._load_dashboard_credentials(
        credentials_path
    )
    server.dashboard_auth_lock = threading.Lock()
    server.dashboard_auth_slots = threading.BoundedSemaphore(1)
    server.dashboard_stream_slots = threading.BoundedSemaphore(2)
    server.dashboard_login_attempts = {}
    server.dashboard_sessions = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    slow = socket.create_connection((host, port), timeout=2)
    try:
        response = httpx.get(
            f"https://{host}:{port}/",
            headers={
                "Host": f"192.168.50.20:{port}",
                "Origin": f"https://192.168.50.20:{port}",
            },
            auth=("admin", "correct horse battery staple"),
            verify=False,
            follow_redirects=False,
            timeout=1,
        )
        assert response.status_code == 200
        assert server.dashboard_handler_slots.acquire(timeout=1)
        assert not server.dashboard_handler_slots.acquire(blocking=False)
        rejected = socket.create_connection((host, port), timeout=2)
        rejected.settimeout(2)
        assert rejected.recv(1) == b""
        rejected.close()
        server.dashboard_handler_slots.release()
    finally:
        slow.close()

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        acquired = [
            server.dashboard_handler_slots.acquire(blocking=False) for _ in range(2)
        ]
        for value in acquired:
            if value:
                server.dashboard_handler_slots.release()
        if all(acquired):
            break
        time.sleep(0.01)
    client_context = ssl._create_unverified_context()
    idle = client_context.wrap_socket(
        socket.create_connection((host, port), timeout=2),
        server_hostname="192.168.50.20",
    )
    time.sleep(0.3)
    idle.close()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        post_handshake = [
            server.dashboard_handler_slots.acquire(blocking=False) for _ in range(2)
        ]
        for value in post_handshake:
            if value:
                server.dashboard_handler_slots.release()
        if all(post_handshake):
            break
        time.sleep(0.01)
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    assert all(acquired)
    assert all(post_handshake)


def test_dashboard_lan_websocket_accepts_basic_only_on_exact_https_origin(
    tmp_path: Path, monkeypatch
) -> None:
    class ClosingCursor:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def poll_payload(self) -> dict[str, object]:
            raise BrokenPipeError

    credentials_path = tmp_path / "dashboard-credentials.json"
    dashboard._write_dashboard_credentials(
        credentials_path, "admin", "correct horse battery staple"
    )
    monkeypatch.setattr(dashboard, "_private_client_scope", lambda _value: "private")
    monkeypatch.setattr(dashboard, "CortexEventCursor", ClosingCursor)
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    server.lan_access_enabled = True
    server.dashboard_public_host = "192.168.50.20"
    server.dashboard_credentials = dashboard._load_dashboard_credentials(
        credentials_path
    )
    server.dashboard_auth_lock = threading.Lock()
    server.dashboard_auth_slots = threading.BoundedSemaphore(
        dashboard.DASHBOARD_AUTH_CONCURRENCY_LIMIT
    )
    server.dashboard_stream_slots = threading.BoundedSemaphore(2)
    server.dashboard_login_attempts = {}
    server.dashboard_sessions = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    authorization = base64.b64encode(b"admin:correct horse battery staple").decode(
        "ascii"
    )
    session_token = "browser-session-token"
    server.dashboard_sessions[session_token] = time.monotonic() + 60

    def websocket_status(
        *,
        origin: str,
        auth: str | None,
        cookie: str | None = None,
        key: str = "dGhlIHNhbXBsZSBub25jZQ==",
    ) -> int:
        auth_header = f"Authorization: Basic {auth}\r\n" if auth else ""
        cookie_header = f"Cookie: {cookie}\r\n" if cookie else ""
        request = (
            "GET /api/cortex/events HTTP/1.1\r\n"
            f"Host: 192.168.50.20:{port}\r\n"
            f"Origin: {origin}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"{auth_header}{cookie_header}\r\n"
        ).encode("ascii")
        with socket.create_connection((host, port), timeout=2) as connection:
            connection.sendall(request)
            # A response header precedes the handler's stream-slot cleanup.
            # The closing cursor makes EOF the request-completion boundary.
            response = bytearray()
            while chunk := connection.recv(4096):
                response.extend(chunk)
            return int(response.split(b"\r\n", 1)[0].split()[1])

    try:
        missing = websocket_status(origin=f"https://192.168.50.20:{port}", auth=None)
        wrong_origin = websocket_status(
            origin="https://attacker.example", auth=authorization
        )
        invalid_key = websocket_status(
            origin=f"https://192.168.50.20:{port}",
            auth=authorization,
            key="invalid",
        )
        assert server.dashboard_stream_slots.acquire(blocking=False)
        assert server.dashboard_stream_slots.acquire(blocking=False)
        server.dashboard_stream_slots.release()
        server.dashboard_stream_slots.release()
        accepted = websocket_status(
            origin=f"https://192.168.50.20:{port}", auth=authorization
        )
        assert server.dashboard_stream_slots.acquire(blocking=False)
        assert server.dashboard_stream_slots.acquire(blocking=False)
        stream_saturated = websocket_status(
            origin=f"https://192.168.50.20:{port}", auth=authorization
        )
        sse_saturated = httpx.get(
            f"http://{host}:{port}/api/activity-stream",
            headers={
                "Host": f"192.168.50.20:{port}",
                "Origin": f"https://192.168.50.20:{port}",
                "Cookie": (f"{dashboard.DASHBOARD_SESSION_COOKIE}={session_token}"),
            },
            timeout=2,
        )
        server.dashboard_stream_slots.release()
        server.dashboard_stream_slots.release()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            acquired = [
                server.dashboard_stream_slots.acquire(blocking=False) for _ in range(2)
            ]
            for value in acquired:
                if value:
                    server.dashboard_stream_slots.release()
            if all(acquired):
                break
            time.sleep(0.01)
        else:
            pytest.fail("stream slots were not released")
        session_accepted = websocket_status(
            origin=f"https://192.168.50.20:{port}",
            auth=None,
            cookie=f"{dashboard.DASHBOARD_SESSION_COOKIE}={session_token}",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert missing == 401
    assert wrong_origin == 403
    assert invalid_key == 400
    assert accepted == 101
    assert stream_saturated == 429
    assert sse_saturated.status_code == 404
    assert session_accepted == 101


@pytest.mark.parametrize(
    ("age_seconds", "pid_alive", "expected_loaded"),
    [
        (0, True, True),
        (dashboard.SERVICE_STATUS_MAX_AGE_SECONDS + 1, True, False),
        (0, False, False),
    ],
)
def test_service_model_status_requires_fresh_live_process(
    tmp_path: Path,
    monkeypatch,
    age_seconds: float,
    pid_alive: bool,
    expected_loaded: bool,
) -> None:
    pid = os.getpid() if pid_alive else os.getpid() + 100_000
    path = tmp_path / "semantic-service-status.json"
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "ready": True,
                "pid": pid,
                "observed_at_epoch": time.time() - age_seconds,
                "routes": {
                    "search.semantic.foreground": {
                        "model": "nvidia/Nemotron-3-Embed-1B-BF16",
                        "provider": "nemotron",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dashboard,
        "_pid_alive",
        lambda value: value == os.getpid(),
    )

    rows = dashboard._service_model_rows(path, capability="embedding")

    assert len(rows) == 1
    assert rows[0]["loaded"] is expected_loaded


def test_service_model_status_rejects_invalid_time_and_uses_flat_model(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "semantic-service-status.json"
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "ready": True,
                "pid": os.getpid(),
                "observed_at_epoch": float("nan"),
                "model": "nvidia/Nemotron-3-Embed-1B-BF16",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dashboard,
        "_pid_alive",
        lambda value: value == os.getpid(),
    )

    rows = dashboard._service_model_rows(path, capability="embedding")

    assert rows == [
        {
            "name": "nvidia/Nemotron-3-Embed-1B-BF16",
            "model": "nvidia/Nemotron-3-Embed-1B-BF16",
            "provider": "nemotron",
            "loaded": False,
            "size": None,
            "size_vram": None,
            "processor": "nemotron",
            "details": {
                "format": "service",
                "revision": None,
                "runtime_model": None,
                "runtime_revision": None,
            },
            "capabilities": ["embedding"],
            "_service_only": True,
        }
    ]


def test_knowledge_mix_snapshot_groups_pages_by_category(
    tmp_path: Path, monkeypatch
) -> None:
    chronovisor_root = tmp_path / "wiki"
    pages_dir = chronovisor_root / "pages"
    (pages_dir / "ai").mkdir(parents=True)
    (pages_dir / "macos").mkdir()
    stable_texts = {
        pages_dir / "ai" / "agent-memory.md": (
            "---\ntitle: Agent memory\nstatus: stable\ntype: knowledge\n---\n"
            + "a" * 20
        ),
        pages_dir / "ai" / "evals.md": (
            "---\ntitle: Evals\nstatus: stable\ntype: knowledge\n---\n" + "b" * 10
        ),
        pages_dir / "macos" / "display.md": (
            "---\ntitle: Display\nstatus: stable\ntype: knowledge\n---\n" + "c" * 15
        ),
    }
    for path, content in stable_texts.items():
        path.write_text(content, encoding="utf-8")
    (pages_dir / "ai" / "draft.md").write_text(
        "---\ntitle: Draft\nstatus: draft\ntype: knowledge\n---\nexcluded",
        encoding="utf-8",
    )
    (pages_dir / "macos" / "invalid.md").write_text(
        "---\ntitle: Invalid\nstatus: stable\n---\nexcluded",
        encoding="utf-8",
    )
    for relative in (
        "index.md",
        "log.md",
        "schema.md",
        "ai/index.md",
        "ai/log.md",
        "ai/schema.md",
    ):
        (pages_dir / relative).write_text("reserved", encoding="utf-8")
    outside = chronovisor_root / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (pages_dir / "outside-link.md").symlink_to(outside)

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", chronovisor_root)

    mix = dashboard._knowledge_mix_snapshot()

    assert mix["total_pages"] == 3
    assert mix["total_bytes"] == sum(
        len(content.encode("utf-8")) for content in stable_texts.values()
    )
    assert [row["id"] for row in mix["categories"]] == ["ai", "macos"]
    assert mix["categories"][0]["label"] == "AI"
    assert mix["categories"][0]["pages"] == 2
    assert mix["categories"][0]["bytes"] == sum(
        len(stable_texts[path].encode("utf-8"))
        for path in stable_texts
        if path.parent.name == "ai"
    )
    assert "ai/agent-memory.md" in mix["categories"][0]["samples"]


def test_dashboard_config_retains_local_and_lan_ports_and_secret_paths(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", tmp_path)
    assert dashboard._dashboard_config() == {"host": "127.0.0.1", "port": 8765}
    assert dashboard._dashboard_config(lan=True)["port"] == 8766
    with pytest.raises(SystemExit):
        dashboard.main(["--lan"])
    with pytest.raises(SystemExit):
        dashboard.main(["--host", "0.0.0.0"])
    (tmp_path / "config.toml").write_text("""[dashboard]
host = "localhost"
port = 9876
[dashboard_lan]
host = "192.168.1.2"
port = 8766
tls_cert_file = "/tmp/existing.crt"
tls_key_file = "/tmp/existing.key"
credentials_file = "/tmp/existing-credentials.json"
""")
    args = dashboard.build_parser().parse_args([])
    assert (args.host, args.port) == ("localhost", 9876)
    calls = []
    monkeypatch.setattr(dashboard, "serve", lambda *a, **k: calls.append((a, k)))
    assert dashboard.main(["--lan"]) == 0
    assert calls == [
        (
            ("192.168.1.2", 8766),
            {
                "lan": True,
                "tls_cert_file": Path("/tmp/existing.crt"),
                "tls_key_file": Path("/tmp/existing.key"),
                "credentials_file": Path("/tmp/existing-credentials.json"),
            },
        )
    ]


def test_capture_snapshot_filters_retired_jobs_and_never_exposes_input(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", tmp_path)
    state = tmp_path / "runtime/background-jobs/state.json"
    state.parent.mkdir(parents=True)
    state.write_text(
        json.dumps(
            {
                "jobs": {
                    "capture": {
                        "module": "chronovisor.hosts.codex_record",
                        "status": "completed",
                        "exit_code": 0,
                        "updated_at": "2026-11-08T02:00:00+00:00",
                        "stdin": "private payload",
                        "output": "secret",
                    },
                    "queue": {
                        "module": "chronovisor.hosts.claude_code_record",
                        "status": "queued",
                    },
                    "retry": {
                        "module": "chronovisor.hosts.codex_record",
                        "status": "retry_wait",
                    },
                    "retired": {
                        "module": "chronovisor.ops.recall_auditor",
                        "status": "queued",
                    },
                }
            }
        )
    )
    before = state.read_bytes(), state.stat().st_mtime_ns
    capture = dashboard._capture_snapshot()
    assert capture["queued"] == 2
    assert capture["by_status"] == {"completed": 1, "queued": 1, "retry_wait": 1}
    assert capture["last_run"]["exit_code"] == 0
    assert "private payload" not in json.dumps(capture) and "secret" not in json.dumps(
        capture
    )
    assert before == (state.read_bytes(), state.stat().st_mtime_ns)


def test_last_capture_uses_the_current_job_producer_schema(tmp_path, monkeypatch):
    from chronovisor.core import background_jobs

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", tmp_path)
    job_dir = tmp_path / "runtime/background-jobs"
    monkeypatch.setattr(background_jobs, "JOB_DIR", job_dir)
    monkeypatch.setattr(background_jobs, "STATE_FILE", job_dir / "state.json")
    monkeypatch.setattr(background_jobs, "LOCK_FILE", job_dir / "state.lock")
    job_dir.mkdir(parents=True)
    background_jobs.STATE_FILE.write_text(
        json.dumps(
            {
                "jobs": {
                    "capture": {
                        "module": "chronovisor.hosts.codex_record",
                        "status": "running",
                        "attempts": 1,
                    }
                }
            }
        )
    )
    finished = background_jobs._finish("capture", exit_code=0, output="private content")
    capture = dashboard._capture_snapshot()
    assert capture["last_run"] == {
        "timestamp": dashboard._dashboard_local_datetime(
            finished["updated_at"]
        ).isoformat(),
        "status": "completed",
        "exit_code": 0,
    }
    assert "private content" not in json.dumps(capture)


def test_save_history_only_counts_raw_captures_without_materializing(
    tmp_path, monkeypatch
):
    from chronovisor.core.raw_segment import append_capture

    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "codex-20261108.md").write_text("saved")
    append_capture(
        raw_dir=raw,
        raw_id="save-test.md",
        idempotency_key="test",
        host="codex",
        session_key="0123456789abcdef01234567",
        session_id="test",
        source_file=tmp_path / "session.jsonl",
        after_line=0,
        until_line=1,
        source_bytes=b'{"test":true}\n',
        record_count=1,
        now=datetime(2026, 11, 8, tzinfo=dashboard.DASHBOARD_TIMEZONE),
    )
    before = {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    history = dashboard._save_history_snapshot(today=date(2026, 11, 8))
    assert history["totals"]["raw_saved"] == 2
    assert history["totals"]["raw_bytes"] == 19
    assert all(
        set(day)
        == {"date", "raw_saved", "raw_bytes", "sources", "raw_samples", "raw_segments"}
        for day in history["days"]
    )
    assert {
        p: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in tmp_path.rglob("*")
        if p.is_file()
    } == before
    assert not (tmp_path / "runtime/raw-projections").exists()


def test_save_history_is_empty_and_bounds_segment_details(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", tmp_path)
    assert dashboard._save_history_snapshot(days=7)["totals"]["raw_saved"] == 0
    raw = tmp_path / "raw"
    raw.mkdir()
    for index in range(70):
        (raw / f"codex-20261108-{index}.md").write_bytes(b"x" * 10)
    result = dashboard._save_history_snapshot(today=date(2026, 11, 8))
    assert result["totals"]["raw_saved"] == 70
    assert result["totals"]["raw_bytes"] == 700
    assert result["days"][-1]["raw_segments"] == [
        {"name": "70 Raw saves", "bytes": 700, "source": "aggregate"}
    ]


def test_snapshot_models_only_retained_roles_and_live_heartbeats(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name, model in (("semantic", "Nemotron"), ("reranker", "StartLux")):
        (runtime / f"{name}-service-status.json").write_text(
            json.dumps(
                {
                    "status": "ready",
                    "ready": True,
                    "pid": os.getpid(),
                    "observed_at_epoch": time.time(),
                    "model": model,
                }
            )
        )
    snapshot = dashboard.build_snapshot()
    assert set(snapshot) == {
        "generated_at",
        "status",
        "save_history",
        "knowledge_mix",
        "model_status",
        "health",
        "events",
    }
    assert snapshot["model_status"]["summary"]["loaded"] == 2
    assert [row["roles"] for row in snapshot["model_status"]["models"]] == [
        ["embed"],
        ["rerank"],
    ]
    monkeypatch.setattr(dashboard, "_pid_alive", lambda _pid: False)
    assert dashboard.build_snapshot()["model_status"]["summary"]["loaded"] == 0


@pytest.mark.parametrize("semantic_status_exists", [False, True])
def test_disabled_semantic_is_optional_with_or_without_status(
    tmp_path, monkeypatch, semantic_status_exists
):
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", tmp_path)
    (tmp_path / "config.toml").write_text("[search.embedding]\nenabled = false\n")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    ready = {
        "status": "ready",
        "ready": True,
        "pid": os.getpid(),
        "observed_at_epoch": time.time(),
        "model": "StartLux",
    }
    (runtime / "reranker-service-status.json").write_text(json.dumps(ready))
    if semantic_status_exists:
        (runtime / "semantic-service-status.json").write_text(
            json.dumps({**ready, "model": "Nemotron"})
        )
    snapshot = dashboard.build_snapshot()
    semantic = snapshot["model_status"]["models"][0]
    assert semantic["status"] == "disabled"
    assert not semantic["running"]
    assert snapshot["model_status"]["summary"]["enabled"] == 1
    assert snapshot["model_status"]["summary"]["loaded"] == 1
    assert snapshot["status"]["services"]["semantic"]["enabled"] is False
    assert not any("semantic" in event["message"] for event in snapshot["events"])
    assert not list(runtime.glob("*.sock"))


def test_retired_dashboard_routes_are_absent_and_errors_do_not_disclose(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dashboard, "CHRONOVISOR_ROOT", tmp_path)
    (tmp_path / "config.toml").write_text("[search.embedding]\nenabled = false\n")
    server = dashboard.ThreadingHTTPServer(("127.0.0.1", 0), dashboard.DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        for path in (
            "activity",
            "activity-stream",
            "local-consensus",
            "model-stream",
            "self-heal",
            "recall",
            "recall-improvement",
            "model-lab",
            "typed-graph",
            "cortex/field",
            "cortex/relations",
            "metrics",
            "frontier-repair",
            "fast-snapshot",
        ):
            assert httpx.get(f"{url}/api/{path}").status_code == 404
        for path in (
            "snapshot",
            "status",
            "events",
            "save-history",
            "knowledge-mix",
            "health",
            "model-status",
            "cortex/graph",
        ):
            assert httpx.get(f"{url}/api/{path}").status_code == 200

        def fail():
            raise RuntimeError("sensitive internal path")

        monkeypatch.setattr(dashboard, "build_snapshot", fail)
        response = httpx.get(f"{url}/api/snapshot")
        assert response.status_code == 500
        assert response.json()["error"] == "Dashboard request failed."
        assert "sensitive" not in response.text
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
