"""Full-YAML frontmatter helpers with byte-stable Markdown bodies."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from chronovisor.core.canonical_document import (
    CanonicalDocument,
    parse_document,
    patch_document_metadata,
    serialize_document,
)
from chronovisor.core.canonical_json import canonical_json_stringifying_strict


def parse(text: str) -> tuple[dict[str, Any], str]:
    """Parse canonical full-YAML frontmatter and preserve the body exactly."""

    if not _has_frontmatter(text):
        return {}, text
    document = parse_document(text.encode("utf-8"))
    return document.metadata, document.body.decode("utf-8")


def patch(text: str, updates: dict[str, Any], deletes: list[str] | None = None) -> str:
    """Patch top-level full-YAML metadata without changing the body."""

    if not _has_frontmatter(text):
        if not updates:
            return text
        data = serialize_document(
            CanonicalDocument(metadata={}, body=text.encode("utf-8"))
        )
    else:
        data = text.encode("utf-8")
    return patch_document_metadata(data, updates, delete=deletes or ()).decode("utf-8")


def canonicalize(text: str) -> str:
    """Canonicalize full-YAML frontmatter without touching the body."""

    if not _has_frontmatter(text):
        return text
    return serialize_document(parse_document(text.encode("utf-8"))).decode("utf-8")


def _has_frontmatter(text: str) -> bool:
    return text.startswith("---\n") or text.startswith("---\r\n")


def normalize_nested(text: str) -> tuple[str, dict[str, Any]]:
    """Merge one accidentally nested frontmatter block without losing fields.

    Ingest used to prepend metadata around a model response that already had
    frontmatter.  The inner block then became searchable body text.  We merge
    only when both blocks parse and the inner block has a real title.  Conflicting
    values are left untouched so a semantic reviewer can decide them.
    """
    outer, body = parse(text)
    if not outer:
        return text, {"changed": False, "reason": "no_outer_frontmatter"}
    inner, inner_body = parse(body.lstrip())
    if not inner or not str(inner.get("title") or "").strip():
        return text, {"changed": False, "reason": "no_nested_frontmatter"}
    conflicts = {
        key: {
            "outer": review_value(outer[key]),
            "inner": review_value(value),
        }
        for key, value in inner.items()
        if key in outer and outer[key] != value
    }
    if conflicts:
        return text, {
            "changed": False,
            "reason": "conflicting_nested_frontmatter",
            "conflicts": conflicts,
        }
    merged = {**outer, **inner}
    normalized = patch(inner_body, merged)
    return normalized, {
        "changed": normalized != text,
        "reason": "merged_nested_frontmatter",
        "merged_keys": sorted(inner),
    }


def propose_nested_resolution(text: str) -> tuple[str, dict[str, Any]]:
    """Build an exact proposal for a conflicting nested block.

    The newer outer scalar wins. Lists retain the outer order and append
    inner-only values. A frontier reviewer must approve this proposal before
    it is written.
    """
    outer, body = parse(text)
    inner, inner_body = parse(body.lstrip()) if outer else ({}, body)
    if not outer or not inner or not str(inner.get("title") or "").strip():
        return text, {"changed": False, "reason": "no_nested_frontmatter"}
    merged = dict(inner)
    for key, value in outer.items():
        if isinstance(value, list) and isinstance(merged.get(key), list):
            merged[key] = _equality_stable_union(value, merged[key])
        else:
            merged[key] = value
    proposed = patch(inner_body, merged)
    conflict_keys = [key for key in inner if key in outer and inner[key] != outer[key]]
    return proposed, {
        "changed": proposed != text,
        "reason": "frontier_required_conflict_resolution",
        "policy": "outer scalar wins; lists are outer-first stable unions",
        "outer_keys": sorted(outer),
        "inner_keys": sorted(inner),
        "merged_keys": sorted(merged),
        "conflicts": {
            key: {
                "outer": review_value(outer[key]),
                "inner": review_value(inner[key]),
                "merged": review_value(merged[key]),
            }
            for key in conflict_keys
        },
    }


def _equality_stable_union(outer: list[Any], inner: list[Any]) -> list[Any]:
    """Preserve outer-first order for both hashable and unhashable YAML values."""

    merged: list[Any] = []
    for candidate in (*outer, *inner):
        if not any(candidate == existing for existing in merged):
            merged.append(candidate)
    return merged


def _json_safe_review_value(value: Any) -> Any:
    """Return deterministic strict-JSON content or a canonical-YAML hash receipt."""

    if not _contains_yaml_set(value):
        try:
            return json.loads(canonical_json_stringifying_strict(value))
        except (TypeError, ValueError):
            pass
    rendered = serialize_document(CanonicalDocument(metadata={"value": value}, body=b""))
    return {
        "kind": "canonical_yaml",
        "utf8_bytes": len(rendered),
        "sha256": hashlib.sha256(rendered).hexdigest(),
    }


def _contains_yaml_set(value: Any, seen: set[int] | None = None) -> bool:
    """Detect nested SafeLoader sets before JSON's ``default=str`` can see them."""

    if isinstance(value, set):
        return True
    if not isinstance(value, (dict, list, tuple)):
        return False
    seen = set() if seen is None else seen
    marker = id(value)
    if marker in seen:
        return False
    seen.add(marker)
    if isinstance(value, dict):
        return any(
            _contains_yaml_set(key, seen) or _contains_yaml_set(item, seen)
            for key, item in value.items()
        )
    return any(_contains_yaml_set(item, seen) for item in value)


def review_value(value: Any) -> Any:
    """Return bounded deterministic JSON-safe review content for YAML values."""
    if not isinstance(value, list):
        return _json_safe_review_value(value)
    safe_items = [_json_safe_review_value(item) for item in value]
    encoded = canonical_json_stringifying_strict(safe_items)
    return {
        "kind": "list",
        "count": len(value),
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "sample": safe_items[:8],
    }
