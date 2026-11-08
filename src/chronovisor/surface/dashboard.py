"""Read-only capture and frozen-memory dashboard."""

from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import getpass
import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import ssl
import stat
import threading
import time
from datetime import date, datetime, timedelta
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from chronovisor.capture import capture_status
from chronovisor.capture.background_jobs import CAPTURE_MODULES
from chronovisor.common.config import load_toml_file
from chronovisor.common.paths import CHRONOVISOR_ROOT
from chronovisor.pages import store
from chronovisor.raw.raw_segment import CAPTURE_TIMEZONE
from chronovisor.raw.raw_store import RawStore
from chronovisor.surface.cortex import CortexEventCursor, build_cortex_graph
from chronovisor.surface.cortex_stream import websocket_accept, websocket_text_frame
from chronovisor.surface.dashboard_http import (
    _file_response,
    _json_response,
    _send_security_headers,
)
from chronovisor.surface.dashboard_static import STATIC_DIR, _resolve_static_path

RAW_DATE_RE = re.compile(r"(?:^|[^0-9])(?P<stamp>20\d{6})(?:[^0-9]|$)")
DASHBOARD_TIMEZONE = CAPTURE_TIMEZONE
SAVE_HISTORY_SEGMENT_DETAIL_DAYS = 30
SAVE_HISTORY_MAX_SEGMENTS_PER_DAY = 64
_LOCK_TYPE = type(threading.Lock())
_SNAPSHOT_LOCK = threading.Lock()
_SNAPSHOT_CACHE: dict[str, Any] = {}

DASHBOARD_CREDENTIAL_VERSION = 2
DASHBOARD_SESSION_COOKIE = "__Host-chronovisor_dashboard_session"
DASHBOARD_SESSION_MAX_AGE_SECONDS = 12 * 60 * 60
DASHBOARD_SESSION_LIMIT = 64
DASHBOARD_PASSWORD_ALGORITHM = "scrypt"
DASHBOARD_PASSWORD_MIN_LENGTH = 8
DASHBOARD_PASSWORD_SCRYPT_N = 2**14
DASHBOARD_PASSWORD_SCRYPT_R = 8
DASHBOARD_PASSWORD_SCRYPT_P = 1
DASHBOARD_PASSWORD_DKLEN = 32
DASHBOARD_LOGIN_ATTEMPT_LIMIT = 5
DASHBOARD_LOGIN_ATTEMPT_WINDOW_SECONDS = 5 * 60
DASHBOARD_TLS_HANDSHAKE_TIMEOUT_SECONDS = 5.0
DASHBOARD_IO_TIMEOUT_SECONDS = 30.0
DASHBOARD_HANDLER_LIMIT = 32
DASHBOARD_AUTH_CONCURRENCY_LIMIT = 2
DASHBOARD_STREAM_LIMIT = 8


def _read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _raw_file_date(path: Path) -> date | None:
    match = RAW_DATE_RE.search(path.name)
    if match:
        try:
            return datetime.strptime(match.group("stamp"), "%Y%m%d").date()
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(
            path.stat().st_mtime, tz=DASHBOARD_TIMEZONE
        ).date()
    except OSError:
        return None


def _knowledge_category_label(category: str) -> str:
    special = {
        "ai": "AI",
        "bom": "BOM",
        "cad": "CAD",
        "jt": "JT",
        "jttok": "JTTOK",
        "qmk": "QMK",
        "vis": "VIS",
    }
    if category in special:
        return special[category]
    return category.replace("-", " ").replace("_", " ").title()


def _knowledge_mix_snapshot() -> dict[str, Any]:
    pages_dir = CHRONOVISOR_ROOT / "pages"
    categories: dict[str, dict[str, Any]] = {}
    if pages_dir.exists():
        for resolved in store.canonical_document_paths(
            pages_dir, require_stable=True
        ):
            try:
                rel = resolved.relative_to(pages_dir.resolve())
                page_bytes = resolved.stat().st_size
            except Exception:
                continue
            category = rel.parts[0] if len(rel.parts) > 1 else "root"
            row = categories.setdefault(
                category,
                {
                    "id": category,
                    "label": _knowledge_category_label(category),
                    "pages": 0,
                    "bytes": 0,
                    "samples": [],
                },
            )
            row["pages"] += 1
            row["bytes"] += page_bytes
            _add_sample(row, "samples", str(rel), limit=4)

    total_pages = sum(int(row["pages"]) for row in categories.values())
    total_bytes = sum(int(row["bytes"]) for row in categories.values())
    rows = sorted(
        categories.values(), key=lambda row: (-int(row["bytes"]), str(row["label"]))
    )
    for row in rows:
        row["share"] = (int(row["bytes"]) / total_bytes) if total_bytes else 0.0

    return {
        "total_pages": total_pages,
        "total_bytes": total_bytes,
        "categories": rows,
        "top": rows[:8],
        "paths": {
            "pages_dir": str(pages_dir),
        },
    }


def _add_sample(row: dict[str, Any], key: str, value: str, limit: int = 6) -> None:
    samples = row.setdefault(key, [])
    if isinstance(samples, list) and value not in samples and len(samples) < limit:
        samples.append(value)


def _dashboard_local_datetime(value: object) -> datetime | None:
    """Parse one timestamp in the dashboard's capture timezone."""

    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=DASHBOARD_TIMEZONE)
    return parsed.astimezone(DASHBOARD_TIMEZONE)


def _raw_source_label(filename: str) -> str:
    for host in ("claude-code", "codex"):
        if host in filename.lower():
            return host
    return "manual"


def _save_history_snapshot(
    days: int = 371, today: date | None = None
) -> dict[str, Any]:
    end = today or datetime.now(DASHBOARD_TIMEZONE).date()
    start = end - timedelta(days=max(1, days) - 1)
    rows = {}
    for offset in range((end - start).days + 1):
        day = (start + timedelta(days=offset)).isoformat()
        rows[day] = {
            "date": day,
            "raw_saved": 0,
            "raw_bytes": 0,
            "sources": {},
            "raw_samples": [],
            "raw_segments": [],
        }
    for unit in RawStore(CHRONOVISOR_ROOT / "raw").iter_units():
        captured = _dashboard_local_datetime(unit.captured_at)
        day = captured.date() if captured else _raw_file_date(unit.path)
        if day is None or not start <= day <= end:
            continue
        row = rows[day.isoformat()]
        source = _raw_source_label(unit.raw_id)
        row["raw_saved"] += 1
        row["raw_bytes"] += unit.length
        row["sources"][source] = row["sources"].get(source, 0) + 1
        _add_sample(row, "raw_samples", unit.raw_id)
        if day >= end - timedelta(days=SAVE_HISTORY_SEGMENT_DETAIL_DAYS - 1):
            row["raw_segments"].append(
                {"name": unit.raw_id, "bytes": unit.length, "source": source}
            )
    source_totals: dict[str, int] = {}
    for row in rows.values():
        for source, count in row["sources"].items():
            source_totals[source] = source_totals.get(source, 0) + count
        row["sources"] = [
            {"name": name, "count": count}
            for name, count in sorted(row["sources"].items())
        ]
        if len(row["raw_segments"]) > SAVE_HISTORY_MAX_SEGMENTS_PER_DAY:
            row["raw_segments"] = [
                {
                    "name": f"{row['raw_saved']} Raw saves",
                    "bytes": row["raw_bytes"],
                    "source": "aggregate",
                }
            ]
    days_list = list(rows.values())
    return {
        "range": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "days": len(days_list),
        },
        "days": days_list,
        "recent": [row for row in days_list if row["raw_saved"]][-14:],
        "totals": {
            "raw_saved": sum(row["raw_saved"] for row in days_list),
            "raw_bytes": sum(row["raw_bytes"] for row in days_list),
            "days_with_saves": sum(bool(row["raw_saved"]) for row in days_list),
        },
        "sources": [
            {"name": name, "count": count}
            for name, count in sorted(source_totals.items())
        ],
    }


def _capture_snapshot() -> dict[str, Any]:
    state = (
        _read_json_file(CHRONOVISOR_ROOT / "runtime/background-jobs/state.json") or {}
    )
    values = state.get("jobs", {})
    jobs = (
        [
            job
            for job in values.values()
            if isinstance(job, dict) and job.get("module") in CAPTURE_MODULES
        ]
        if isinstance(values, dict)
        else []
    )
    counts: dict[str, int] = {}
    events = []
    runs = []
    for job in jobs:
        status = str(job.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
        timestamp = _dashboard_local_datetime(job.get("updated_at"))
        exit_code = job.get("exit_code")
        if timestamp is not None and isinstance(exit_code, int):
            run = {
                "timestamp": timestamp.isoformat(),
                "exit_code": exit_code,
                "status": status,
            }
            runs.append(run)
            events.append(
                {
                    "timestamp": run["timestamp"],
                    "level": "info" if exit_code == 0 else "error",
                    "message": f"Capture · {status} · exit {exit_code}",
                }
            )
    runs.sort(key=lambda run: run["timestamp"])
    observed = capture_status.snapshot(
        capture_status.status_path(CHRONOVISOR_ROOT)
    )
    if observed.get("available"):
        pending = observed["pending"]
        failed = observed["failed"]
        last_error = observed["last_error"]
        last_success_at = observed["last_success_at"]
    else:
        pending = sum(counts.get(key, 0) for key in ("queued", "retry_wait"))
        failed = sum(counts.get(key, 0) for key in ("failed", "quarantined"))
        last_error = "none" if not failed else "capture_failed"
        last_success_at = None
    return {
        "by_status": counts,
        "queued": pending,
        "pending": pending,
        "failed": failed,
        "last_error": last_error,
        "last_success_at": last_success_at,
        "status_available": observed.get("available", False),
        "last_run": runs[-1] if runs else None,
        "events": events,
    }


def build_snapshot() -> dict[str, Any]:
    # Frozen pages need one scan; captures refresh independently of the page index.
    root = str(CHRONOVISOR_ROOT)
    with _SNAPSHOT_LOCK:
        if _SNAPSHOT_CACHE.get("root") != root:
            _SNAPSHOT_CACHE.clear()
            _SNAPSHOT_CACHE.update(root=root, knowledge_mix=_knowledge_mix_snapshot())
        if time.monotonic() - _SNAPSHOT_CACHE.get("saved_at", -60) >= 10:
            history = _save_history_snapshot()
            _SNAPSHOT_CACHE.update(save_history=history, saved_at=time.monotonic())
        history = _SNAPSHOT_CACHE["save_history"]
        knowledge = _SNAPSHOT_CACHE["knowledge_mix"]
    capture = _capture_snapshot()
    counts = capture["by_status"]
    events = list(capture.pop("events"))
    return {
        "generated_at": datetime.now(DASHBOARD_TIMEZONE).isoformat(),
        "status": {
            "state": "running" if counts.get("running") else "idle",
            "saves_today": history["days"][-1]["raw_saved"],
            "page_count": knowledge["total_pages"],
            "capture": capture,
        },
        "save_history": history,
        "knowledge_mix": knowledge,
        "health": {"capture": capture, "queue": counts},
        "events": sorted(events, key=lambda event: event["timestamp"])[-80:],
    }


def _dashboard_config(*, lan: bool = False) -> dict[str, Any]:
    data = load_toml_file(CHRONOVISOR_ROOT / "config.toml")
    section = data.get("dashboard_lan" if lan else "dashboard")
    section = section if isinstance(section, dict) else {}
    host = section.get("host")
    host = host.strip() if isinstance(host, str) else ""
    if not lan and host.casefold() not in {"localhost", "127.0.0.1"}:
        host = "127.0.0.1"
    port = section.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        port = 8766 if lan else 8765
    result: dict[str, Any] = {"host": host, "port": port}
    if lan:
        for key in ("tls_cert_file", "tls_key_file", "credentials_file"):
            value = section.get(key)
            path = (
                Path(value).expanduser() if isinstance(value, str) and value else None
            )
            result[key] = str(path) if path is not None and path.is_absolute() else None
    return result


def _write_private_file(path: Path, payload: bytes) -> None:
    if not path.is_absolute():
        raise ValueError("dashboard secret paths must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _read_private_file(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"{label} is unavailable: {path}") from exc
    try:
        metadata = os.fstat(fd)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise RuntimeError(f"{label} must be a regular 0600 file: {path}")
        if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
            raise RuntimeError(f"{label} must be owned by the dashboard user: {path}")
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            return handle.read()
    finally:
        if fd >= 0:
            os.close(fd)


def _password_digest(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=DASHBOARD_PASSWORD_SCRYPT_N,
        r=DASHBOARD_PASSWORD_SCRYPT_R,
        p=DASHBOARD_PASSWORD_SCRYPT_P,
        dklen=DASHBOARD_PASSWORD_DKLEN,
        maxmem=64 * 1024 * 1024,
    )


def _write_dashboard_credentials(path: Path, username: str, password: str) -> None:
    username = username.strip()
    if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", username) is None:
        raise ValueError(
            "dashboard username must contain 1 to 64 safe ASCII characters"
        )
    if len(password) < DASHBOARD_PASSWORD_MIN_LENGTH:
        raise ValueError(
            f"dashboard password must contain at least {DASHBOARD_PASSWORD_MIN_LENGTH} characters"
        )
    salt = secrets.token_bytes(16)
    payload = {
        "version": DASHBOARD_CREDENTIAL_VERSION,
        "username": username,
        "password": {
            "algorithm": DASHBOARD_PASSWORD_ALGORITHM,
            "salt": base64.b64encode(salt).decode("ascii"),
            "digest": base64.b64encode(_password_digest(password, salt)).decode(
                "ascii"
            ),
            "n": DASHBOARD_PASSWORD_SCRYPT_N,
            "r": DASHBOARD_PASSWORD_SCRYPT_R,
            "p": DASHBOARD_PASSWORD_SCRYPT_P,
            "dklen": DASHBOARD_PASSWORD_DKLEN,
        },
        "created_at": datetime.now().astimezone().isoformat(),
    }
    _write_private_file(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(),
    )


def _load_dashboard_credentials(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(_read_private_file(path, "dashboard credentials"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"dashboard credentials are malformed: {path}") from exc
    password = payload.get("password") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("version") != DASHBOARD_CREDENTIAL_VERSION
        or not isinstance(payload.get("username"), str)
        or re.fullmatch(r"[A-Za-z0-9._-]{1,64}", payload["username"]) is None
        or not isinstance(password, dict)
        or password.get("algorithm") != DASHBOARD_PASSWORD_ALGORITHM
    ):
        raise RuntimeError(f"dashboard credentials are malformed: {path}")
    try:
        salt = base64.b64decode(str(password["salt"]), validate=True)
        digest = base64.b64decode(str(password["digest"]), validate=True)
    except (KeyError, ValueError, binascii.Error) as exc:
        raise RuntimeError(f"dashboard credentials are malformed: {path}") from exc
    expected = {
        "n": DASHBOARD_PASSWORD_SCRYPT_N,
        "r": DASHBOARD_PASSWORD_SCRYPT_R,
        "p": DASHBOARD_PASSWORD_SCRYPT_P,
        "dklen": DASHBOARD_PASSWORD_DKLEN,
    }
    if (
        len(salt) != 16
        or len(digest) != DASHBOARD_PASSWORD_DKLEN
        or any(password.get(key) != value for key, value in expected.items())
    ):
        raise RuntimeError(f"dashboard credentials are malformed: {path}")
    return payload


def _dashboard_credentials_match(
    credentials: dict[str, Any], username: str, password: str
) -> bool:
    if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", username) is None:
        return False
    stored_password = credentials.get("password")
    if not isinstance(stored_password, dict):
        return False
    try:
        salt = base64.b64decode(str(stored_password["salt"]), validate=True)
        expected = base64.b64decode(str(stored_password["digest"]), validate=True)
        actual = _password_digest(password, salt)
    except KeyError, ValueError, binascii.Error:
        return False
    return hmac.compare_digest(
        username, str(credentials.get("username") or "")
    ) and hmac.compare_digest(actual, expected)


_PRIVATE_LAN_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
)
_LINK_LOCAL_NETWORKS = tuple(
    ipaddress.ip_network(value) for value in ("169.254.0.0/16", "fe80::/10")
)


class _ThreadingHTTPSServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        *,
        certfile: Path,
        keyfile: Path,
        handshake_timeout: float = DASHBOARD_TLS_HANDSHAKE_TIMEOUT_SECONDS,
        io_timeout: float = DASHBOARD_IO_TIMEOUT_SECONDS,
        handler_limit: int = DASHBOARD_HANDLER_LIMIT,
    ) -> None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.set_alpn_protocols(["http/1.1"])
        context.load_cert_chain(certfile, keyfile)
        super().__init__(server_address, handler)
        self.dashboard_tls_context = context
        self.dashboard_tls_handshake_timeout = handshake_timeout
        self.dashboard_io_timeout = io_timeout
        self.dashboard_handler_slots = threading.BoundedSemaphore(handler_limit)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self.dashboard_handler_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.dashboard_handler_slots.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        secure_request = None
        try:
            request.settimeout(self.dashboard_tls_handshake_timeout)
            secure_request = self.dashboard_tls_context.wrap_socket(
                request, server_side=True
            )
            secure_request.settimeout(self.dashboard_io_timeout)
            super().process_request_thread(secure_request, client_address)
        except Exception:
            self.shutdown_request(secure_request or request)
        finally:
            self.dashboard_handler_slots.release()


def _normalized_client_ip(
    value: object,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        address = ipaddress.ip_address(str(value))
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        return address.ipv4_mapped
    return address


def _private_client_scope(value: object) -> str:
    address = _normalized_client_ip(value)
    if address is None:
        return "invalid"
    if address.is_loopback:
        return "loopback"
    if any(
        address.version == network.version and address in network
        for network in (*_PRIVATE_LAN_NETWORKS, *_LINK_LOCAL_NETWORKS)
    ):
        return "private"
    return "public"


def _dashboard_bind_host_is_loopback(host: str) -> bool:
    return host.casefold() in {"localhost", "127.0.0.1"}


def _dashboard_bind_host_is_private(host: str) -> bool:
    address = _normalized_client_ip(host)
    return (
        isinstance(address, ipaddress.IPv4Address)
        and _private_client_scope(host) == "private"
    )


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "LLMWikiDashboard/0.1"

    def end_headers(self) -> None:
        session_cookie = getattr(self, "_dashboard_session_cookie", None)
        if session_cookie:
            self.send_header("Set-Cookie", session_cookie)
            self._dashboard_session_cookie = None
        super().end_headers()

    def _is_loopback(self) -> bool:
        return _private_client_scope(self.client_address[0]) == "loopback"

    def _lan_enabled(self) -> bool:
        return bool(getattr(self.server, "lan_access_enabled", False))

    def _deny_remote(self, status: HTTPStatus, message: str) -> None:
        body = message.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        _send_security_headers(self)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _deny_basic_auth(self, status: HTTPStatus = HTTPStatus.UNAUTHORIZED) -> None:
        body = b"Authentication required."
        self.send_response(status)
        if status == HTTPStatus.UNAUTHORIZED:
            self.send_header(
                "WWW-Authenticate",
                'Basic realm="Chronovisor Dashboard", charset="UTF-8"',
            )
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        _send_security_headers(self)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _login_state(
        self,
    ) -> tuple[threading.Lock, dict[str, list[float]], dict[str, float]]:
        lock = getattr(self.server, "dashboard_auth_lock", None)
        attempts = getattr(self.server, "dashboard_login_attempts", None)
        sessions = getattr(self.server, "dashboard_sessions", None)
        if (
            not isinstance(lock, _LOCK_TYPE)
            or not isinstance(attempts, dict)
            or not isinstance(sessions, dict)
        ):
            raise RuntimeError("dashboard LAN authentication is not initialized")
        return lock, attempts, sessions

    def _login_rate_limited(self) -> bool:
        lock, attempts, _sessions = self._login_state()
        client = str(self.client_address[0])
        cutoff = time.monotonic() - DASHBOARD_LOGIN_ATTEMPT_WINDOW_SECONDS
        with lock:
            recent = [stamp for stamp in attempts.get(client, []) if stamp >= cutoff]
            attempts[client] = recent
            return len(recent) >= DASHBOARD_LOGIN_ATTEMPT_LIMIT

    def _record_login_failure(self) -> None:
        lock, attempts, _sessions = self._login_state()
        client = str(self.client_address[0])
        cutoff = time.monotonic() - DASHBOARD_LOGIN_ATTEMPT_WINDOW_SECONDS
        with lock:
            recent = [stamp for stamp in attempts.get(client, []) if stamp >= cutoff]
            recent.append(time.monotonic())
            attempts[client] = recent

    def _basic_authorized(self) -> bool | None:
        headers = self.headers.get_all("Authorization", [])
        if len(headers) != 1 or len(headers[0]) > 2048:
            return False
        scheme, separator, encoded = headers[0].partition(" ")
        if not separator or scheme.casefold() != "basic" or not encoded:
            return False
        try:
            decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
        except UnicodeError, ValueError, binascii.Error:
            return False
        username, separator, password = decoded.partition(":")
        credentials = getattr(self.server, "dashboard_credentials", None)
        slots = getattr(self.server, "dashboard_auth_slots", None)
        if not separator or not isinstance(credentials, dict):
            return False
        if not isinstance(slots, threading.BoundedSemaphore):
            raise RuntimeError("dashboard LAN authentication is not initialized")
        if not slots.acquire(blocking=False):
            return None
        try:
            return _dashboard_credentials_match(credentials, username, password)
        finally:
            slots.release()

    def _session_authorized(self) -> bool:
        try:
            cookie_headers = self.headers.get_all("Cookie", [])
            if len(cookie_headers) != 1:
                return False
            cookies = SimpleCookie()
            cookies.load(cookie_headers[0])
            supplied = cookies.get(DASHBOARD_SESSION_COOKIE)
        except Exception:
            return False
        if supplied is None or len(supplied.value) > 128:
            return False
        lock, _attempts, sessions = self._login_state()
        now = time.monotonic()
        with lock:
            expired = [token for token, expiry in sessions.items() if expiry <= now]
            for token in expired:
                sessions.pop(token, None)
            return sessions.get(supplied.value, 0) > now

    def _establish_session(self) -> None:
        lock, attempts, sessions = self._login_state()
        token = secrets.token_urlsafe(32)
        now = time.monotonic()
        with lock:
            attempts.pop(str(self.client_address[0]), None)
            if len(sessions) >= DASHBOARD_SESSION_LIMIT:
                sessions.pop(min(sessions, key=lambda token: sessions[token]))
            sessions[token] = now + DASHBOARD_SESSION_MAX_AGE_SECONDS
        self._dashboard_session_cookie = (
            f"{DASHBOARD_SESSION_COOKIE}={token}; Path=/; "
            f"Max-Age={DASHBOARD_SESSION_MAX_AGE_SECONDS}; Secure; HttpOnly; SameSite=Strict"
        )

    def _lan_authorized(self) -> bool:
        if self._session_authorized():
            return True
        if self._login_rate_limited():
            self._deny_basic_auth(HTTPStatus.TOO_MANY_REQUESTS)
            return False
        authorized = self._basic_authorized()
        if authorized is None:
            self._deny_basic_auth(HTTPStatus.TOO_MANY_REQUESTS)
            return False
        if not authorized:
            if self.headers.get("Authorization"):
                self._record_login_failure()
            self._deny_basic_auth(
                HTTPStatus.TOO_MANY_REQUESTS
                if self._login_rate_limited()
                else HTTPStatus.UNAUTHORIZED
            )
            return False
        if self.headers.get("Upgrade", "").casefold() == "websocket":
            return True
        self._establish_session()
        return True

    def _acquire_stream_slot(self) -> bool:
        if not self._lan_enabled():
            return True
        slots = getattr(self.server, "dashboard_stream_slots", None)
        if not isinstance(slots, threading.BoundedSemaphore):
            raise RuntimeError("dashboard LAN streams are not initialized")
        return slots.acquire(blocking=False)

    def _release_stream_slot(self) -> None:
        if not self._lan_enabled():
            return
        slots = getattr(self.server, "dashboard_stream_slots", None)
        if isinstance(slots, threading.BoundedSemaphore):
            slots.release()

    def _browser_boundary_allows(self) -> bool:
        lan_enabled = self._lan_enabled()
        scope = _private_client_scope(self.client_address[0])
        if (lan_enabled and scope != "private") or (
            not lan_enabled and scope != "loopback"
        ):
            self._deny_remote(HTTPStatus.FORBIDDEN, "Client network is not allowed.")
            return False
        port = int(self.server.server_address[1])  # type: ignore[index]
        allowed_hosts = (
            {f"{getattr(self.server, 'dashboard_public_host', '')}:{port}"}
            if lan_enabled
            else {f"localhost:{port}", f"127.0.0.1:{port}", f"[::1]:{port}"}
        )
        hosts = self.headers.get_all("Host", [])
        host = hosts[0].strip().casefold() if len(hosts) == 1 else ""
        if host not in allowed_hosts:
            self._deny_remote(HTTPStatus.MISDIRECTED_REQUEST, "Request rejected.")
            return False
        origins = self.headers.get_all("Origin", [])
        origin = origins[0].strip().casefold() if len(origins) == 1 else ""
        websocket = self.headers.get("Upgrade", "").casefold() == "websocket"
        scheme = "https" if lan_enabled else "http"
        if len(origins) > 1 or (origin and origin != f"{scheme}://{host}"):
            self._deny_remote(HTTPStatus.FORBIDDEN, "Request rejected.")
            return False
        if websocket and not origin:
            self._deny_remote(HTTPStatus.FORBIDDEN, "Request rejected.")
            return False
        return self._lan_authorized() if lan_enabled else True

    def _lan_access_response(self) -> None:
        port = int(self.server.server_address[1])  # type: ignore[index]
        host = str(getattr(self.server, "dashboard_public_host", ""))
        _json_response(
            self,
            {
                "enabled": self._lan_enabled(),
                "urls": [f"https://{host}:{port}/"] if self._lan_enabled() else [],
                "trusted_lan_only": self._lan_enabled(),
            },
        )

    def _cortex_graph_response(self) -> None:
        _json_response(
            self,
            build_cortex_graph(CHRONOVISOR_ROOT),
        )

    def _cortex_events_response(self, query: str) -> None:
        if not self._acquire_stream_slot():
            self._deny_remote(HTTPStatus.TOO_MANY_REQUESTS, "Stream limit reached.")
            return
        try:
            self._cortex_events_response_with_slot(query)
        finally:
            self._release_stream_slot()

    def _cortex_events_response_with_slot(self, query: str) -> None:
        upgrade = self.headers.get("Upgrade", "").casefold()
        connection = {
            item.strip().casefold()
            for item in self.headers.get("Connection", "").split(",")
        }
        key = self.headers.get("Sec-WebSocket-Key", "")
        if upgrade != "websocket" or "upgrade" not in connection or not key:
            self.send_error(HTTPStatus.BAD_REQUEST, "WebSocket upgrade required")
            return
        try:
            accept = websocket_accept(key)
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid WebSocket key")
            return

        self.send_response(HTTPStatus.SWITCHING_PROTOCOLS)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        self.close_connection = True

        cursor = CortexEventCursor(CHRONOVISOR_ROOT)
        last_heartbeat = 0.0
        try:
            while True:
                payload = cursor.poll_payload()
                events = (
                    payload.get("events") if payload.get("type") == "events" else []
                )
                now = time.monotonic()
                if events or payload.get("type") in {"resync", "session_changed"}:
                    pass
                elif now - last_heartbeat >= 15:
                    payload = {"type": "heartbeat"}
                else:
                    time.sleep(0.25)
                    continue
                self.wfile.write(websocket_text_frame(payload))
                self.wfile.flush()
                last_heartbeat = now
                if not events:
                    time.sleep(0.25)
        except BrokenPipeError, ConnectionError, OSError:
            return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if not self._browser_boundary_allows():
            return
        try:
            if path == "/":
                _file_response(self, STATIC_DIR / "index.html")
            elif path == "/cortex":
                _file_response(self, STATIC_DIR / "cortex.html")
            elif path == "/api/lan-access":
                self._lan_access_response()
            elif path == "/api/snapshot":
                _json_response(self, build_snapshot())
            elif path in {
                "/api/status",
                "/api/events",
                "/api/save-history",
                "/api/knowledge-mix",
                "/api/health",
            }:
                key = path.removeprefix("/api/").replace("-", "_")
                _json_response(self, {key: build_snapshot()[key]})
            elif path == "/api/cortex/graph":
                self._cortex_graph_response()
            elif path == "/api/cortex/events":
                self._cortex_events_response(parsed.query)
            elif path.startswith("/static/"):
                target = _resolve_static_path(STATIC_DIR, path)
                if target is None:
                    self.send_error(HTTPStatus.FORBIDDEN)
                    return
                _file_response(self, target)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except BrokenPipeError, ConnectionError, OSError:
            return
        except Exception:
            if path.startswith("/api/"):
                _json_response(
                    self,
                    {
                        "status": "error",
                        "category": "internal_error",
                        "error": "Dashboard request failed.",
                    },
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            raise

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def serve(
    host: str,
    port: int,
    *,
    lan: bool = False,
    tls_cert_file: Path | None = None,
    tls_key_file: Path | None = None,
    credentials_file: Path | None = None,
) -> None:
    if lan and not _dashboard_bind_host_is_private(host):
        raise ValueError("LAN dashboard host must be an explicit private IPv4 address")
    if not lan and not _dashboard_bind_host_is_loopback(host):
        raise ValueError("dashboard host must be localhost or 127.0.0.1")
    server: ThreadingHTTPServer
    if lan:
        if tls_cert_file is None or tls_key_file is None or credentials_file is None:
            raise ValueError(
                "LAN dashboard requires TLS certificate, key, and credentials"
            )
        _read_private_file(tls_key_file, "dashboard TLS private key")
        credentials = _load_dashboard_credentials(credentials_file)
        try:
            cert_stat = tls_cert_file.stat(follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(
                f"dashboard TLS certificate is unavailable: {tls_cert_file}"
            ) from exc
        if tls_cert_file.is_symlink() or not stat.S_ISREG(cert_stat.st_mode):
            raise RuntimeError("dashboard TLS certificate must be a regular file")
        server = _ThreadingHTTPSServer(
            (host, port),
            DashboardHandler,
            certfile=tls_cert_file,
            keyfile=tls_key_file,
        )
        server.dashboard_credentials = credentials  # type: ignore[attr-defined]
        server.dashboard_auth_lock = threading.Lock()  # type: ignore[attr-defined]
        server.dashboard_auth_slots = threading.BoundedSemaphore(  # type: ignore[attr-defined]
            DASHBOARD_AUTH_CONCURRENCY_LIMIT
        )
        server.dashboard_stream_slots = threading.BoundedSemaphore(  # type: ignore[attr-defined]
            DASHBOARD_STREAM_LIMIT
        )
        server.dashboard_login_attempts = {}  # type: ignore[attr-defined]
        server.dashboard_sessions = {}  # type: ignore[attr-defined]
    else:
        server = ThreadingHTTPServer((host, port), DashboardHandler)
    server.lan_access_enabled = lan  # type: ignore[attr-defined]
    server.dashboard_public_host = host  # type: ignore[attr-defined]
    print(f"Chronovisor dashboard: {'https' if lan else 'http'}://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    config = _dashboard_config()
    parser = argparse.ArgumentParser(description="Run the Chronovisor local dashboard.")
    parser.add_argument("--host", default=config["host"])
    parser.add_argument("--port", type=int, default=config["port"])
    parser.add_argument(
        "--lan",
        action="store_true",
        help="Use the separately configured TLS-authenticated private-LAN endpoint.",
    )
    parser.add_argument(
        "--set-lan-credentials",
        action="store_true",
        help="Prompt for and store the configured LAN password, then exit.",
    )
    parser.add_argument("--username", default="admin")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the ``chronovisor-dashboard`` command-line entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    lan_config = (
        _dashboard_config(lan=True) if args.lan or args.set_lan_credentials else None
    )
    if args.set_lan_credentials:
        credentials_path = lan_config.get("credentials_file") if lan_config else None
        if credentials_path is None:
            parser.error("[dashboard_lan].credentials_file must be an absolute path")
        credentials_file = Path(str(credentials_path))
        password = getpass.getpass(
            f"Dashboard LAN password (minimum {DASHBOARD_PASSWORD_MIN_LENGTH} characters): "
        )
        confirmation = getpass.getpass("Confirm dashboard LAN password: ")
        if password != confirmation:
            parser.error("dashboard passwords do not match")
        try:
            _write_dashboard_credentials(credentials_file, args.username, password)
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Dashboard LAN credentials stored: {credentials_file}")
        return 0
    if args.lan:
        if lan_config is None or not all(
            (
                lan_config.get("host"),
                lan_config.get("tls_cert_file"),
                lan_config.get("tls_key_file"),
                lan_config.get("credentials_file"),
            )
        ):
            parser.error(
                "[dashboard_lan] requires host, tls_cert_file, tls_key_file, and credentials_file"
            )
        serve(
            str(lan_config["host"]),
            int(lan_config["port"]),
            lan=True,
            tls_cert_file=Path(str(lan_config["tls_cert_file"])),
            tls_key_file=Path(str(lan_config["tls_key_file"])),
            credentials_file=Path(str(lan_config["credentials_file"])),
        )
        return 0
    if not _dashboard_bind_host_is_loopback(args.host):
        parser.error("--host must be localhost or 127.0.0.1")
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
