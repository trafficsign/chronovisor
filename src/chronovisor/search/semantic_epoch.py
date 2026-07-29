"""Pure construction and validation of structured-review cache epochs."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from chronovisor.core.canonical_json import canonical_json_sha256_strict
from chronovisor.decision.decision_authority import semantic_authority_shape_error


STRUCTURED_REVIEW_HOLD_EPOCH_VERSION = 2
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_EPOCH_FIELDS = frozenset(
    {
        "epoch_version",
        "lane",
        "authority_sha256",
        "schema_sha256",
        "prompt_sha256",
        "system_sha256",
        "system_kind",
        "effective_request_sha256",
        "resolver_sha256",
    }
)


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def opaque_text_sha256(value: str | None) -> str:
    marker = b"none\0" if value is None else b"text\0" + value.encode("utf-8")
    return hashlib.sha256(marker).hexdigest()


def build_structured_review_epoch(
    *,
    lane: str,
    authority: Mapping[str, Any],
    schema_sha256: str,
    prompt: str,
    system: str | None,
    effective_request_sha256: str,
    resolver_sha256: str,
) -> dict[str, Any]:
    """Build an opaque, plaintext-free exact review identity."""

    authority_error = semantic_authority_shape_error(authority, lane=lane)
    if authority_error is not None:
        raise ValueError(authority_error)
    if not isinstance(prompt, str):
        raise ValueError("structured review prompt must be text")
    for name, digest in (
        ("schema", schema_sha256),
        ("effective request", effective_request_sha256),
        ("resolver", resolver_sha256),
    ):
        if not is_sha256(digest):
            raise ValueError(f"structured review {name} digest is invalid")
    return {
        "epoch_version": STRUCTURED_REVIEW_HOLD_EPOCH_VERSION,
        "lane": lane,
        "authority_sha256": canonical_json_sha256_strict(authority),
        "schema_sha256": schema_sha256,
        "prompt_sha256": opaque_text_sha256(prompt),
        "system_sha256": opaque_text_sha256(system),
        "system_kind": "none" if system is None else "text",
        "effective_request_sha256": effective_request_sha256,
        "resolver_sha256": resolver_sha256,
    }


def structured_review_epoch_error(
    epoch: object,
    *,
    lane: str,
    authority: Mapping[str, Any],
) -> str | None:
    """Validate an opaque review epoch without request plaintext."""

    if not isinstance(epoch, Mapping):
        return "structured review hold epoch is missing"
    if set(epoch) != _EPOCH_FIELDS:
        return "structured review hold epoch fields are invalid"
    authority_error = semantic_authority_shape_error(authority, lane=lane)
    if authority_error is not None:
        return authority_error
    if (
        epoch.get("epoch_version") != STRUCTURED_REVIEW_HOLD_EPOCH_VERSION
        or epoch.get("lane") != lane
        or epoch.get("authority_sha256")
        != canonical_json_sha256_strict(authority)
        or epoch.get("system_kind") not in {"none", "text"}
    ):
        return "structured review hold epoch identity is invalid"
    for field in (
        "authority_sha256",
        "schema_sha256",
        "prompt_sha256",
        "system_sha256",
        "effective_request_sha256",
        "resolver_sha256",
    ):
        if not is_sha256(epoch.get(field)):
            return f"structured review hold epoch {field} is invalid"
    return None
