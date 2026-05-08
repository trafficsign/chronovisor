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


def _unquote(value: str) -> str:
    """Strip a matching pair of outer quotes (double or single) from a scalar."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
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

    Lists are emitted in inline form. Items are joined with ``, ``; the
    caller is responsible for ensuring items don't contain characters
    that would break the inline form (commas, brackets, etc.). For
    ``raw_keywords`` and similar list fields, the writer at the
    boundary (``wiki_save_raw``) rejects unsafe characters before they
    reach this layer.
    """
    if isinstance(value, list):
        if not value:
            return f"{key}: []"
        items = ", ".join(str(v) for v in value)
        return f"{key}: [{items}]"
    return f"{key}: {value}"
