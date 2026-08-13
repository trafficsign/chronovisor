"""Full-YAML canonical Markdown documents with byte-stable bodies."""

from __future__ import annotations

import posixpath
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal, cast
from urllib.parse import quote, unquote, urlsplit

import yaml  # type: ignore[import-untyped]

_MARKDOWN_LINK_RE = re.compile(
    r"(?<![!\\])\[(?P<label>(?:\\[^\n]|[^\]\\\n])*)\]\(\s*"
    r"(?P<target><[^>\n]+>|(?:[^\s()\n]+|\([^()\n]*\))+)"
    r"(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^()\n]*\)))?\s*\)"
)
_FENCE_OPEN_RE = re.compile(r"(?m)^[ \t]{0,3}(?P<fence>`{3,}|~{3,})[^\n]*(?:\n|$)")
_INLINE_CODE_RE = re.compile(r"`+[^`\n]*`+")
_WIKILINK_RE = re.compile(r"(?<!\\)\[\[[^\]\n]*\]\]")

Namespace = Literal["pages", "system"]
_NAMESPACES = frozenset({"pages", "system"})
PAGE_STATUSES = frozenset({"draft", "stable", "deprecated"})


class CanonicalDocumentError(ValueError):
    """The document is not valid canonical Markdown."""


class _UniqueKeyLoader(yaml.SafeLoader):  # type: ignore[misc]
    pass


class _CanonicalDumper(yaml.SafeDumper):  # type: ignore[misc]
    pass


def _represent_deterministic_set(dumper: Any, value: set[Any]) -> Any:
    """Order SafeLoader-compatible scalar set keys without sorting mappings."""

    nodes = []
    for item in value:
        key_node = dumper.represent_data(item)
        if not isinstance(key_node, yaml.nodes.ScalarNode):
            raise yaml.representer.RepresenterError(
                "canonical YAML sets require scalar keys"
            )
        nodes.append(
            (
                key_node,
                yaml.nodes.ScalarNode(
                    tag="tag:yaml.org,2002:null",
                    value="null",
                ),
            )
        )
    nodes.sort(key=lambda row: (row[0].tag, row[0].value, row[0].style or ""))
    return yaml.nodes.MappingNode(tag="tag:yaml.org,2002:set", value=nodes)


_CanonicalDumper.add_representer(set, _represent_deterministic_set)


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


@dataclass(frozen=True, slots=True)
class ResolvedMarkdownLink:
    namespace: Namespace
    path: str
    fragment: str | None = None


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
            yaml.dump(
                document.metadata,
                Dumper=_CanonicalDumper,
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


def patch_document_metadata(
    data: bytes,
    updates: Mapping[str, Any],
    *,
    delete: Iterable[str] = (),
) -> bytes:
    """Patch top-level full-YAML metadata without changing body bytes."""

    delete_fields = tuple(delete)
    if not all(isinstance(key, str) for key in updates) or not all(
        isinstance(key, str) for key in delete_fields
    ):
        raise CanonicalDocumentError("metadata patch keys must be strings")
    overlap = set(updates).intersection(delete_fields)
    if overlap:
        raise CanonicalDocumentError(
            f"metadata patch cannot update and delete {sorted(overlap)!r}"
        )
    document = parse_document(data)
    metadata = dict(document.metadata)
    metadata.update(updates)
    for key in delete_fields:
        metadata.pop(key, None)
    return serialize_document(CanonicalDocument(metadata=metadata, body=document.body))


def validate_canonical_document(
    data: bytes,
    *,
    namespace: Namespace,
    path: str,
    require_stable: bool = False,
    allowed_targets: set[tuple[Namespace, str]] | None = None,
) -> CanonicalDocument:
    """Validate one production document and its internal Markdown links."""

    document = parse_document(data)
    status = document.metadata.get("status")
    if status not in PAGE_STATUSES:
        raise CanonicalDocumentError(
            "status must be one of draft, stable, or deprecated"
        )
    if require_stable and status != "stable":
        raise CanonicalDocumentError("writer target must have status: stable")
    page_type = document.metadata.get("type")
    if namespace == "pages" and (
        not isinstance(page_type, str) or not page_type.strip()
    ):
        raise CanonicalDocumentError("pages documents require a non-empty type")
    text = _decode_body(document.body)
    spans = _protected_spans(text)
    if any(
        not _position_in_spans(match.start(), spans)
        for match in _WIKILINK_RE.finditer(text)
    ):
        raise CanonicalDocumentError("legacy wikilinks are not canonical Markdown")
    links = resolve_internal_markdown_links(
        document.body,
        source_namespace=namespace,
        source_path=path,
    )
    if allowed_targets is not None:
        missing = sorted(
            {
                (link.namespace, link.path)
                for link in links
                if (link.namespace, link.path) not in allowed_targets
            }
        )
        if missing:
            targets = ", ".join(f"{ns}/{target}" for ns, target in missing)
            raise CanonicalDocumentError(f"missing Markdown link target: {targets}")
    return document


def format_internal_markdown_link(
    label: str,
    *,
    source_namespace: Namespace,
    source_path: str,
    target_namespace: Namespace,
    target_path: str,
) -> str:
    """Render one canonical internal link with a relative same-namespace target."""

    source = _source_document_path(source_namespace, source_path)
    target = _source_document_path(target_namespace, target_path)
    rendered_target = _format_internal_target(
        source_namespace=source.namespace,
        source_path=source.path,
        target_namespace=target.namespace,
        target_path=target.path,
        fragment=None,
    )
    if "\r" in label or "\n" in label:
        raise CanonicalDocumentError("Markdown link label cannot contain newlines")
    rendered_label = label.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    rendered = f"[{rendered_label}](<{rendered_target}>)"
    resolved = resolve_internal_markdown_link(
        rendered_target,
        source_namespace=source.namespace,
        source_path=source.path,
    )
    if resolved != target:
        raise CanonicalDocumentError("rendered Markdown link changed its target")
    return rendered


def extract_markdown_links(body: bytes | str) -> tuple[str, ...]:
    """Return inline-link destinations; reference-style links are out of scope."""

    text = _decode_body(body) if isinstance(body, bytes) else body
    spans = _protected_spans(text)
    return tuple(
        match.group("target").removeprefix("<").removesuffix(">")
        for match in _MARKDOWN_LINK_RE.finditer(text)
        if not _position_in_spans(match.start(), spans)
    )


def markdown_link_spans(body: bytes | str) -> tuple[tuple[int, int], ...]:
    """Return exact inline-link spans outside frontmatter and code."""

    text = _decode_body(body) if isinstance(body, bytes) else body
    protected = _protected_spans(text)
    return tuple(
        match.span()
        for match in _MARKDOWN_LINK_RE.finditer(text)
        if not _position_in_spans(match.start(), protected)
    )


def _decode_body(body: bytes) -> str:
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CanonicalDocumentError("Markdown body is not UTF-8") from exc


def resolve_internal_markdown_link(
    target: str,
    *,
    source_namespace: Namespace,
    source_path: str,
) -> ResolvedMarkdownLink | None:
    """Purely resolve one inline target, excluding external and same-doc links."""

    source = _source_document_path(source_namespace, source_path)
    target = target.removeprefix("<").removesuffix(">")
    try:
        parsed = urlsplit(target)
    except ValueError as exc:
        raise CanonicalDocumentError(
            f"invalid Markdown link target: {target!r}"
        ) from exc
    if parsed.scheme or parsed.netloc or parsed.query or not parsed.path:
        return None
    decoded_path = _decode_url_component(parsed.path, target)
    if source_namespace == "pages" and not decoded_path.startswith("/"):
        _reject_page_namespace_escape(decoded_path, source.path)
    if decoded_path.startswith("/"):
        absolute = decoded_path.removeprefix("/")
        first = PurePosixPath(absolute).parts[:1]
        combined = (
            absolute
            if first and first[0] in _NAMESPACES
            else posixpath.join(source.namespace, absolute)
        )
    else:
        combined = posixpath.join(
            source.namespace,
            PurePosixPath(source.path).parent.as_posix(),
            decoded_path,
        )
    normalized = posixpath.normpath(combined)
    parts = PurePosixPath(normalized).parts
    if normalized in {".", ".."} or normalized.startswith("../"):
        raise CanonicalDocumentError(
            f"Markdown link escapes canonical roots: {target!r}"
        )
    if not parts or parts[0] not in _NAMESPACES or len(parts) == 1:
        raise CanonicalDocumentError(
            f"Markdown link escapes canonical roots: {target!r}"
        )
    namespace = cast(Namespace, parts[0])
    if source.namespace == "pages" and namespace != "pages":
        raise CanonicalDocumentError(f"pages link crosses into system: {target!r}")
    path = PurePosixPath(*parts[1:]).as_posix()
    if PurePosixPath(path).suffix.casefold() != ".md":
        return None
    if namespace == source.namespace and path == source.path:
        return None
    fragment = (
        _decode_url_component(parsed.fragment, target) if parsed.fragment else None
    )
    return ResolvedMarkdownLink(namespace, path, fragment or None)


def resolve_internal_markdown_links(
    body: bytes | str,
    *,
    source_namespace: Namespace,
    source_path: str,
) -> tuple[ResolvedMarkdownLink, ...]:
    """Extract and resolve internal document links from canonical Markdown."""

    return tuple(
        resolved
        for target in extract_markdown_links(body)
        if (
            resolved := resolve_internal_markdown_link(
                target,
                source_namespace=source_namespace,
                source_path=source_path,
            )
        )
        is not None
    )


InternalLinkRewrite = Callable[
    [ResolvedMarkdownLink, str], ResolvedMarkdownLink | str | None
]
InvalidInternalLinkRewrite = Callable[[str, str, CanonicalDocumentError], str | None]


def rewrite_internal_markdown_links(
    body: bytes | str,
    *,
    source_namespace: Namespace,
    source_path: str,
    rewrite: InternalLinkRewrite,
    on_invalid: InvalidInternalLinkRewrite | None = None,
    output_namespace: Namespace | None = None,
    output_path: str | None = None,
) -> tuple[str, int]:
    """Rewrite internal links, preserving protected Markdown and invalid defaults."""

    text = _decode_body(body) if isinstance(body, bytes) else body
    spans = _protected_spans(text)
    rendered_namespace = output_namespace or source_namespace
    rendered_path = output_path or source_path
    changed = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        if _position_in_spans(match.start(), spans):
            return match.group(0)
        raw_target = match.group("target").removeprefix("<").removesuffix(">")
        try:
            resolved = resolve_internal_markdown_link(
                raw_target,
                source_namespace=source_namespace,
                source_path=source_path,
            )
        except CanonicalDocumentError as exc:
            if on_invalid is None:
                raise
            rendered = on_invalid(raw_target, match.group("label"), exc)
            if rendered is None:
                raise
            if rendered != match.group(0):
                changed += 1
            return rendered
        if resolved is None:
            return match.group(0)
        replacement = rewrite(resolved, match.group("label"))
        if replacement is None:
            return match.group(0)
        if isinstance(replacement, str):
            rendered = replacement
        else:
            target = _format_internal_target(
                source_namespace=rendered_namespace,
                source_path=rendered_path,
                target_namespace=replacement.namespace,
                target_path=replacement.path,
                fragment=replacement.fragment,
            )
            target_start = match.start("target") - match.start()
            target_end = match.end("target") - match.start()
            rendered = match.group(0)[:target_start] + f"<{target}>" + match.group(0)[target_end:]
        if rendered != match.group(0):
            changed += 1
        return rendered

    return _MARKDOWN_LINK_RE.sub(replace, text), changed


def _format_internal_target(
    *,
    source_namespace: Namespace,
    source_path: str,
    target_namespace: Namespace,
    target_path: str,
    fragment: str | None,
) -> str:
    source = _source_document_path(source_namespace, source_path)
    target = _source_document_path(target_namespace, target_path)
    if source.namespace == "pages" and target.namespace == "system":
        raise CanonicalDocumentError("pages links cannot cross into system")
    if source.namespace == target.namespace:
        destination = posixpath.relpath(
            target.path,
            start=PurePosixPath(source.path).parent.as_posix(),
        )
    else:
        destination = f"/{target.namespace}/{target.path}"
    rendered = quote(destination, safe="/-._~")
    if fragment:
        rendered += f"#{quote(fragment, safe='-._~')}"
    return rendered


def _protected_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    frontmatter_end = _text_frontmatter_end(text)
    if frontmatter_end:
        spans.append((0, frontmatter_end))
    cursor = frontmatter_end
    while opening := _FENCE_OPEN_RE.search(text, cursor):
        fence = opening.group("fence")
        closing = re.compile(
            rf"(?m)^[ \t]{{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*(?:\n|$)"
        ).search(text, opening.end())
        end = closing.end() if closing else len(text)
        spans.append((opening.start(), end))
        cursor = end
    spans.extend(match.span() for match in _INLINE_CODE_RE.finditer(text))
    return sorted(spans)


def _text_frontmatter_end(text: str) -> int:
    if not (text.startswith("---\n") or text.startswith("---\r\n")):
        return 0
    cursor = text.find("\n") + 1
    while cursor < len(text):
        line_end = text.find("\n", cursor)
        end = len(text) if line_end < 0 else line_end
        if text[cursor:end].removesuffix("\r") == "---":
            return len(text) if line_end < 0 else line_end + 1
        cursor = end + 1
    return len(text)


def _position_in_spans(position: int, spans: list[tuple[int, int]]) -> bool:
    for start, end in spans:
        if position < start:
            return False
        if position < end:
            return True
    return False


def _source_document_path(namespace: Namespace, value: str) -> ResolvedMarkdownLink:
    if namespace not in _NAMESPACES:
        raise CanonicalDocumentError(f"invalid canonical namespace: {namespace!r}")
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() == "."
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise CanonicalDocumentError(f"invalid source document path: {value!r}")
    return ResolvedMarkdownLink(namespace, path.as_posix())


def _decode_url_component(value: str, target: str) -> str:
    decoded = value
    for _ in range(16):
        try:
            next_value = unquote(decoded, errors="strict")
        except UnicodeDecodeError as exc:
            raise CanonicalDocumentError(
                f"invalid encoded Markdown link target: {target!r}"
            ) from exc
        if next_value == decoded:
            break
        decoded = next_value
    else:
        raise CanonicalDocumentError(f"over-encoded Markdown link target: {target!r}")
    if "\\" in decoded or any(
        unicodedata.category(character) == "Cc" for character in decoded
    ):
        raise CanonicalDocumentError(f"unsafe Markdown link target: {target!r}")
    return decoded


def _reject_page_namespace_escape(target_path: str, source_path: str) -> None:
    depth = len(PurePosixPath(source_path).parent.parts)
    if PurePosixPath(source_path).parent.as_posix() == ".":
        depth = 0
    for part in PurePosixPath(target_path).parts:
        if part == "..":
            if depth == 0:
                raise CanonicalDocumentError(
                    f"Markdown link escapes pages namespace: {target_path!r}"
                )
            depth -= 1
        elif part != ".":
            depth += 1


def _opening_end(data: bytes) -> int:
    if data.startswith(b"---\n"):
        return 4
    if data.startswith(b"---\r\n"):
        return 5
    raise CanonicalDocumentError("opening frontmatter delimiter is missing")


def _load_metadata(data: bytes) -> dict[str, Any]:
    try:
        loaded = yaml.load(data.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (TypeError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise CanonicalDocumentError(f"frontmatter is not valid YAML: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise CanonicalDocumentError("frontmatter must be a mapping with string keys")
    return cast(dict[str, Any], loaded)
