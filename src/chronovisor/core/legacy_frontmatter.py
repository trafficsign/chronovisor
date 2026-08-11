"""Historical limited frontmatter parser for Raw and offline migration only."""

from __future__ import annotations

import json
import re
from typing import Any

_FM_DELIM = "---"
_FM_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


def parse(text: str) -> tuple[dict[str, Any], str]:
    """Parse the historical scalar/list frontmatter subset."""

    if not text.startswith(_FM_DELIM):
        return {}, text

    match = _FM_RE.match(text)
    if not match:
        return {}, text

    fm_text = match.group(1)
    body = text[match.end() :]

    meta: dict[str, Any] = {}
    lines = fm_text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip() or ":" not in line:
            index += 1
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        if value == "":
            block_items: list[str] = []
            next_index = index + 1
            while next_index < len(lines):
                stripped = lines[next_index].lstrip()
                if not stripped.startswith("- "):
                    break
                block_items.append(_unquote(stripped[2:].strip()))
                next_index += 1
            if block_items:
                meta[key] = block_items
                index = next_index
                continue
            meta[key] = ""
            index += 1
            continue

        if value.startswith("[") and value.endswith("]"):
            try:
                decoded = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                decoded = None
            if isinstance(decoded, list):
                meta[key] = [str(item) for item in decoded]
                index += 1
                continue
            inner = value[1:-1].strip()
            meta[key] = (
                []
                if inner == ""
                else [_unquote(item) for item in _split_inline_list(inner)]
            )
            index += 1
            continue

        meta[key] = _unquote(value)
        index += 1

    return meta, body


def patch(text: str, updates: dict[str, Any], deletes: list[str] | None = None) -> str:
    """Patch historical scalar/list frontmatter without changing its body."""

    meta, body = parse(text)
    for key in deletes or []:
        meta.pop(key, None)
    meta.update(updates)
    if not meta:
        return body
    rendered = [_FM_DELIM]
    rendered.extend(_serialize_kv(key, value) for key, value in meta.items())
    rendered.append(_FM_DELIM)
    return "\n".join(rendered) + "\n" + body


def _unquote(value: str) -> str:
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
    items: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for character in inner:
        if quote is not None:
            current.append(character)
            if character == quote:
                quote = None
        elif character in ('"', "'"):
            quote = character
            current.append(character)
        elif character == ",":
            items.append("".join(current))
            current = []
        else:
            current.append(character)
    tail = "".join(current).strip()
    if tail or items:
        items.append("".join(current))
    return items


def _serialize_kv(key: str, value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return f"{key}: []"
        items = ", ".join(_serialize_scalar(item, flow=True) for item in value)
        return f"{key}: [{items}]"
    return f"{key}: {_serialize_scalar(value)}"


def _serialize_scalar(value: Any, *, flow: bool = False) -> str:
    text = str(value)
    yaml_indicator = "-?:,[]{}#&*!|>'\"%@`"
    unsafe = (
        not text
        or text != text.strip()
        or any(character in text for character in "\n\r\t")
        or text[0] in yaml_indicator
        or ":" in text
        or " #" in text
        or (
            flow
            and (
                any(character.isspace() for character in text)
                or any(character in text for character in ",[]{}#?")
            )
        )
    )
    return json.dumps(text, ensure_ascii=False) if unsafe else text
