"""Frontmatter parsing and patching utilities.

Patch-centric YAML-flavored frontmatter parser/serializer for wiki pages
and raw entries. `parse(text)` extracts metadata + body, `patch(text, updates)`
updates the frontmatter region while preserving the body verbatim.

Design tradeoffs:
- Bit-exact round-trip is NOT a goal. Body is preserved verbatim, unknown
  keys are preserved, but key order may shift (updates are applied in
  insertion order over the existing dict).
- Supported value forms: scalar `key: value`, inline list `key: [a, b, c]`,
  block list `key:\n  - a\n  - b`. Complex YAML features (anchors,
  multi-line scalars, nested maps, flow maps) are NOT supported.
- No external YAML parser dependency. Minimal hand-rolled parser intended
  for the limited shapes used by this codebase.
- Behaves as a strict superset of the legacy scalar-only parsers in
  `server._parse_frontmatter` and `index_store._parse_frontmatter`: any
  document that worked before continues to work, and the returned dict
  has scalar values as `str` (consumers depending on `.get("title", ...)`
  remain correct).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_FM_DELIM = "---"
_FM_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


def parse(text: str) -> tuple[dict[str, Any], str]:
    """Extract frontmatter and body from a markdown document.

    Returns ``(meta, body)``. If no frontmatter is present, returns
    ``({}, text)``.

    Supported value forms:
      - scalar: ``key: value`` → ``str``
      - inline list: ``key: [a, b, c]`` → ``list[str]``
      - block list::

          key:
            - a
            - b

        → ``list[str]``

    Quoted scalars (double or single) are supported; matching outer quotes
    are stripped. Trailing whitespace in scalar values is stripped.
    """
    if not text.startswith(_FM_DELIM):
        return {}, text

    m = _FM_RE.match(text)
    if not m:
        return {}, text

    fm_text = m.group(1)
    body = text[m.end():]

    meta: dict[str, Any] = {}
    lines = fm_text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        if value == "":
            # Possibly a block list (next lines indented with "- ").
            block_items: list[str] = []
            j = i + 1
            while j < n:
                next_line = lines[j]
                stripped = next_line.lstrip()
                if not stripped.startswith("- "):
                    break
                item = stripped[2:].strip()
                block_items.append(_unquote(item))
                j += 1
            if block_items:
                meta[key] = block_items
                i = j
                continue
            # Empty value, no block — treat as empty string for legacy parity.
            meta[key] = ""
            i += 1
            continue

        # Inline list: [a, b, c]
        if value.startswith("[") and value.endswith("]"):
            try:
                decoded = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                decoded = None
            if isinstance(decoded, list):
                meta[key] = [str(item) for item in decoded]
                i += 1
                continue
            inner = value[1:-1].strip()
            if inner == "":
                meta[key] = []
            else:
                items = [_unquote(item) for item in _split_inline_list(inner)]
                meta[key] = items
            i += 1
            continue

        # Plain scalar.
        meta[key] = _unquote(value)
        i += 1

    return meta, body


def patch(text: str, updates: dict[str, Any], deletes: list[str] | None = None) -> str:
    """Update frontmatter and return the new document text.

    - ``updates``: keys to add or replace.
    - ``deletes``: keys to remove.
    - Body is preserved verbatim.
    - If no frontmatter exists, a new one is prepended.
    - Existing keys keep their position; new keys are appended in
      ``updates`` insertion order.
    """
    deletes = deletes or []
    meta, body = parse(text)

    for k in deletes:
        meta.pop(k, None)

    for k, v in updates.items():
        meta[k] = v

    if not meta:
        return body

    out = [_FM_DELIM]
    for k, v in meta.items():
        out.append(_serialize_kv(k, v))
    out.append(_FM_DELIM)
    return "\n".join(out) + "\n" + body


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
        key: {"outer": outer[key], "inner": value}
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
            merged[key] = list(dict.fromkeys([*value, *merged[key]]))
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
                "outer": _review_value(outer[key]),
                "inner": _review_value(inner[key]),
                "merged": _review_value(merged[key]),
            }
            for key in conflict_keys
        },
    }


def _review_value(value: Any) -> Any:
    """Bound large metadata lists without weakening exact diff/hash review."""
    if not isinstance(value, list):
        return value
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return {
        "kind": "list",
        "count": len(value),
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "sample": value[:8],
    }


def _unquote(value: str) -> str:
    """Strip a matching pair of outer quotes (double or single) from a scalar."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]
        return decoded if isinstance(decoded, str) else str(decoded)
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1]
    return value


def _split_inline_list(inner: str) -> list[str]:
    """Split inline-list content on top-level commas, respecting quoted strings.

    Quoted strings preserve commas inside them. Supports both double and
    single quotes; quote characters are kept on the items (stripped later
    by ``_unquote``).
    """
    items: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for ch in inner:
        if quote is not None:
            current.append(ch)
            if ch == quote:
                quote = None
        elif ch in ('"', "'"):
            quote = ch
            current.append(ch)
        elif ch == ",":
            items.append("".join(current))
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail or items:
        items.append("".join(current))
    return items


def _serialize_kv(key: str, value: Any) -> str:
    """Serialize a single key-value pair for the frontmatter region.

    Lists are emitted in inline form. Scalars that are unsafe as YAML plain
    values are JSON-quoted (JSON strings are valid YAML double-quoted
    scalars), while ordinary values retain the compact legacy representation.
    """
    if isinstance(value, list):
        if not value:
            return f"{key}: []"
        items = ", ".join(_serialize_scalar(v, flow=True) for v in value)
        return f"{key}: [{items}]"
    return f"{key}: {_serialize_scalar(value)}"


def _serialize_scalar(value: Any, *, flow: bool = False) -> str:
    text = str(value)
    yaml_indicator = "-?:,[]{}#&*!|>'\"%@`"
    unsafe = (
        not text
        or text != text.strip()
        or any(char in text for char in "\n\r\t")
        or text[0] in yaml_indicator
        or ":" in text
        or " #" in text
        or (
            flow
            and (
                any(char.isspace() for char in text)
                or any(char in text for char in ",[]{}#?")
            )
        )
    )
    return json.dumps(text, ensure_ascii=False) if unsafe else text


def canonicalize(text: str) -> str:
    """Serialize supported frontmatter as strict YAML without touching the body."""

    meta, body = parse(text)
    if not meta:
        return text
    rendered = [_FM_DELIM]
    rendered.extend(_serialize_kv(key, value) for key, value in meta.items())
    rendered.append(_FM_DELIM)
    return "\n".join(rendered) + "\n" + body
