"""Opaque SHA-256 helpers with explicit input and output contracts."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_bytes(value: bytes) -> str:
    """Hash bytes and return an unprefixed lowercase digest."""
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash UTF-8 text and return an unprefixed lowercase digest."""
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_prefixed_bytes(value: bytes) -> str:
    """Hash bytes and return the legacy ``sha256:``-prefixed form."""
    return f"sha256:{sha256_bytes(value)}"


def sha256_prefixed_text(value: str) -> str:
    """Hash UTF-8 text and return the legacy prefixed form."""
    return sha256_prefixed_bytes(value.encode("utf-8"))


def is_sha256(value: object) -> bool:
    """Return whether a value is an unprefixed lowercase SHA-256 digest."""
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None
