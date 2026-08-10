"""Full-YAML canonical Markdown documents with byte-stable bodies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, cast

import yaml  # type: ignore[import-untyped]

from chronovisor.core.link_fix import position_in_spans, protected_spans

_MARKDOWN_LINK_RE = re.compile(
    r"(?<![!\\])\[[^\]\n]*\]\(\s*"
    r"(?P<target><[^>\n]+>|(?:[^\s()\n]+|\([^()\n]*\))+)"
    r"(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^()\n]*\)))?\s*\)"
)


class CanonicalDocumentError(ValueError):
    """The document is not valid canonical Markdown."""


class _UniqueKeyLoader(yaml.SafeLoader):  # type: ignore[misc]
    pass


def _construct_unique_mapping(
    loader: Any, node: Any, *, deep: bool = False
) -> dict[Any, Any]:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=False)
        if key in seen:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key {key!r}",
                key_node.start_mark,
            )
        seen.add(key)
    loader.flatten_mapping(node)
    return cast(dict[Any, Any], yaml.SafeLoader.construct_mapping(loader, node, deep))


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(slots=True)
class CanonicalDocument:
    metadata: dict[str, Any]
    body: bytes


def parse_document(data: bytes) -> CanonicalDocument:
    """Parse full YAML frontmatter while retaining the body bytes exactly."""

    opening_end = _opening_end(data)
    cursor = opening_end
    while cursor <= len(data):
        line_end = data.find(b"\n", cursor)
        if line_end < 0:
            line_end = len(data)
            next_line = line_end
        else:
            next_line = line_end + 1
        if data[cursor:line_end].removesuffix(b"\r") == b"---":
            metadata = _load_metadata(data[opening_end:cursor])
            return CanonicalDocument(metadata=metadata, body=data[next_line:])
        if line_end == len(data):
            break
        cursor = next_line
    raise CanonicalDocumentError("closing frontmatter delimiter is missing")


def serialize_document(document: CanonicalDocument) -> bytes:
    """Canonicalize only the YAML frontmatter and append the original body."""

    try:
        rendered = cast(
            str,
            yaml.safe_dump(
                document.metadata,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            ),
        )
    except (TypeError, UnicodeError, yaml.YAMLError) as exc:
        raise CanonicalDocumentError(
            f"frontmatter cannot be serialized: {exc}"
        ) from exc
    return b"---\n" + rendered.encode("utf-8") + b"---\n" + document.body


def extract_markdown_links(body: bytes | str) -> tuple[str, ...]:
    """Return inline Markdown link destinations, without resolving them."""

    try:
        text = body.decode("utf-8") if isinstance(body, bytes) else body
    except UnicodeDecodeError as exc:
        raise CanonicalDocumentError("Markdown body is not UTF-8") from exc
    spans = protected_spans(text)
    return tuple(
        match.group("target").removeprefix("<").removesuffix(">")
        for match in _MARKDOWN_LINK_RE.finditer(text)
        if not position_in_spans(match.start(), spans)
    )


def _opening_end(data: bytes) -> int:
    if data.startswith(b"---\n"):
        return 4
    if data.startswith(b"---\r\n"):
        return 5
    raise CanonicalDocumentError("opening frontmatter delimiter is missing")


def _load_metadata(data: bytes) -> dict[str, Any]:
    try:
        loaded = yaml.load(data.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise CanonicalDocumentError(f"frontmatter is not valid YAML: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise CanonicalDocumentError("frontmatter must be a mapping with string keys")
    return cast(dict[str, Any], loaded)
