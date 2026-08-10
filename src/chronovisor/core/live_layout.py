"""Small sealed proof for resumable fresh final-layout publication."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO

from chronovisor.core.canonical_json import canonical_json_line_bytes_strict
from chronovisor.core.durable_state import (
    StateSealError,
    atomic_write_bytes_at,
    file_lock,
    open_regular_nofollow,
    seal_object,
    verify_sealed_object,
)

LIVE_LAYOUT_SCHEMA = "chronovisor.live-layout.v1"
LIVE_LAYOUT_VERSION = 1
LIVE_LAYOUT_PROOF = "bootstrap-layout.json"
LIVE_LAYOUT_LOCK = "bootstrap-layout.lock"
INDEX_RENDERER_VERSION = 1
_INDEX_PREFIX = b"---\nokf_version: '0.2'\n---\n# Chronovisor pages\n"


@contextmanager
def pinned_layout_directories(root: Path) -> Iterator[tuple[int, int]]:
    """Pin the canonical root and runtime directories without following links."""

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    root_fd = os.open(root, flags)
    runtime_fd = -1
    try:
        runtime_fd = os.open("runtime", flags, dir_fd=root_fd)
        yield root_fd, runtime_fd
    finally:
        if runtime_fd >= 0:
            os.close(runtime_fd)
        os.close(root_fd)


@contextmanager
def bootstrap_layout_lock(root: Path) -> Iterator[tuple[int, int]]:
    """Serialize the complete fresh-layout publication across processes."""

    with pinned_layout_directories(root) as (root_fd, runtime_fd):
        with file_lock(Path(LIVE_LAYOUT_LOCK), dir_fd=runtime_fd):
            yield root_fd, runtime_fd


def write_live_layout_proof(
    root: Path,
    *,
    state: str,
    runtime_fd: int | None = None,
) -> None:
    """Atomically publish one sealed in-progress or ready bootstrap proof."""

    if state not in {"in-progress", "ready"}:
        raise ValueError("live layout proof state is invalid")
    from chronovisor.core.reserved_documents import render_pages_log
    from chronovisor.core.store import SCHEMA_CONTENT

    payload = {
        "schema": LIVE_LAYOUT_SCHEMA,
        "version": LIVE_LAYOUT_VERSION,
        "state": state,
        "paths": {
            "index": "pages/index.md",
            "log": "pages/log.md",
            "schema": "system/schema.md",
            "activity": "runtime/activity.jsonl",
        },
        "index_renderer_version": INDEX_RENDERER_VERSION,
        "log_sha256": hashlib.sha256(render_pages_log()).hexdigest(),
        "schema_sha256": hashlib.sha256(SCHEMA_CONTENT.encode("utf-8")).hexdigest(),
        "activity_prefix": {
            "length": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        },
    }
    raw = canonical_json_line_bytes_strict(seal_object(payload))
    if runtime_fd is not None:
        atomic_write_bytes_at(runtime_fd, LIVE_LAYOUT_PROOF, raw)
    else:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        flags |= getattr(os, "O_CLOEXEC", 0)
        root_fd = os.open(root, flags)
        fallback_runtime_fd = -1
        try:
            fallback_runtime_fd = os.open("runtime", flags, dir_fd=root_fd)
            atomic_write_bytes_at(fallback_runtime_fd, LIVE_LAYOUT_PROOF, raw)
        finally:
            if fallback_runtime_fd >= 0:
                os.close(fallback_runtime_fd)
            os.close(root_fd)


def read_live_layout_proof(
    root: Path,
    *,
    runtime_fd: int | None = None,
) -> dict[str, object] | None:
    """Read and validate the exact bounded live-layout proof."""

    path = root / "runtime" / LIVE_LAYOUT_PROOF
    try:
        if runtime_fd is None:
            with open_regular_nofollow(path) as handle:
                raw = _bounded_proof_bytes(handle)
        else:
            flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(LIVE_LAYOUT_PROOF, flags, dir_fd=runtime_fd)
            try:
                with os.fdopen(descriptor, "rb", closefd=False) as handle:
                    raw = _bounded_proof_bytes(handle)
            finally:
                os.close(descriptor)
        proof = verify_sealed_object(json.loads(raw))
    except (OSError, TypeError, ValueError, json.JSONDecodeError, StateSealError):
        return None
    expected_keys = {
        "schema",
        "version",
        "state",
        "paths",
        "index_renderer_version",
        "log_sha256",
        "schema_sha256",
        "activity_prefix",
        "seal_sha256",
    }
    if set(proof) != expected_keys:
        return None
    if (
        proof.get("schema") != LIVE_LAYOUT_SCHEMA
        or proof.get("version") != LIVE_LAYOUT_VERSION
        or proof.get("state") not in {"in-progress", "ready"}
        or proof.get("index_renderer_version") != INDEX_RENDERER_VERSION
        or proof.get("paths")
        != {
            "index": "pages/index.md",
            "log": "pages/log.md",
            "schema": "system/schema.md",
            "activity": "runtime/activity.jsonl",
        }
        or proof.get("activity_prefix")
        != {"length": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    ):
        return None
    from chronovisor.core.reserved_documents import render_pages_log
    from chronovisor.core.store import SCHEMA_CONTENT

    if proof.get("log_sha256") != hashlib.sha256(render_pages_log()).hexdigest():
        return None
    if proof.get("schema_sha256") != hashlib.sha256(
        SCHEMA_CONTENT.encode("utf-8")
    ).hexdigest():
        return None
    return proof


def _bounded_proof_bytes(handle: IO[bytes]) -> bytes:
    snapshot = os.fstat(handle.fileno())
    if not stat.S_ISREG(snapshot.st_mode) or snapshot.st_size > 4096:
        raise ValueError("live layout proof is unsafe")
    return handle.read(snapshot.st_size + 1)


def file_sha256_nofollow(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> str:
    """Hash one bounded regular file through a no-follow descriptor."""

    return _file_sha256_nofollow(path, max_bytes=max_bytes)


def valid_index_shape_nofollow(path: Path) -> bool:
    """Check the bounded renderer signature without scanning the page corpus."""

    try:
        with open_regular_nofollow(path) as handle:
            snapshot = os.fstat(handle.fileno())
            if not stat.S_ISREG(snapshot.st_mode) or snapshot.st_size > 16 * 1024 * 1024:
                return False
            return handle.read(len(_INDEX_PREFIX)) == _INDEX_PREFIX
    except OSError:
        return False


def _file_sha256_nofollow(
    path: Path,
    *,
    max_bytes: int = 16 * 1024 * 1024,
) -> str:
    with open_regular_nofollow(path) as handle:
        snapshot = os.fstat(handle.fileno())
        if not stat.S_ISREG(snapshot.st_mode) or snapshot.st_size > max_bytes:
            raise ValueError("live layout file is unsafe or exceeds its bound")
        digest = hashlib.sha256()
        remaining = snapshot.st_size
        while remaining:
            chunk = handle.read(min(64 * 1024, remaining))
            if not chunk:
                raise ValueError("live layout file changed during hashing")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.fstat(handle.fileno()).st_size != snapshot.st_size:
            raise ValueError("live layout file changed during hashing")
        return digest.hexdigest()
