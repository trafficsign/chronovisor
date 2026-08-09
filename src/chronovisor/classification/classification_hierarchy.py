"""Deterministic navigation graph over primary UDC Summary concepts."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from chronovisor.recall.classification import ClassificationError, UDCPackage

NAVIGATION_GRAPH_SCHEMA = "chronovisor.udc-navigation-graph.v1"
ROOT_NOTATIONS = ("0", "1", "2", "3", "5", "6", "7", "8", "9")
_MARKDOWN_HEADING = re.compile(r"^\s*#{1,6}\s+")
_FRONTMATTER_FENCE = re.compile(r"^\s*---\s*$")


def is_primary_navigation_concept(row: Mapping[str, Any]) -> bool:
    """Keep main-table concepts and ranges, but not auxiliary notation headers."""

    notation = str(row.get("notation") or "")
    label = str(row.get("label_en") or row.get("label") or "")
    return bool(
        notation
        and notation[0] in "012356789"
        and not any(char in notation for char in ('"', "'", "`", "(", ")", "="))
        and "special auxiliary" not in label.casefold()
    )


@dataclass(frozen=True)
class NavigationNode:
    uri: str
    notation: str
    label_en: str
    label_ja: str
    parent_uri: str | None
    children_uris: tuple[str, ...]
    is_range: bool

    def card(self) -> dict[str, Any]:
        return {
            "notation": self.notation,
            "label_en": self.label_en,
            "label_ja": self.label_ja,
            "has_children": bool(self.children_uris),
            "is_range": self.is_range,
        }


@dataclass(frozen=True)
class NavigationGraph:
    schema: str
    release: str
    checksum: str
    roots: tuple[str, ...]
    nodes: Mapping[str, NavigationNode]
    notation_to_uri: Mapping[str, str]
    contracted_parent_count: int

    def by_notation(self, notation: str) -> NavigationNode | None:
        uri = self.notation_to_uri.get(notation)
        return self.nodes.get(uri) if uri else None

    def children(self, notation: str | None) -> tuple[NavigationNode, ...]:
        uris = self.roots if notation is None else (
            self.by_notation(notation).children_uris
            if self.by_notation(notation)
            else ()
        )
        return tuple(self.nodes[uri] for uri in uris)

    def ancestors(self, notation: str) -> tuple[NavigationNode, ...]:
        node = self.by_notation(notation)
        output = []
        seen = set()
        while node is not None and node.uri not in seen:
            output.append(node)
            seen.add(node.uri)
            node = self.nodes.get(node.parent_uri) if node.parent_uri else None
        output.reverse()
        return tuple(output)

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return ancestor in {
            node.notation for node in self.ancestors(descendant)
        }


def build_navigation_graph(package: UDCPackage) -> NavigationGraph:
    """Build a connected graph, contracting excluded auxiliary header nodes."""

    valid_rows = {
        str(row.get("uri") or ""): row
        for row in package.concepts.values()
        if is_primary_navigation_concept(row)
    }
    if not valid_rows:
        raise ClassificationError("UDC navigation graph has no primary concepts")

    parent_by_uri: dict[str, str | None] = {}
    contracted = 0
    for uri, row in valid_rows.items():
        parent_uri = str(row.get("broader_uri") or "")
        seen = {uri}
        skipped = False
        while parent_uri and parent_uri not in valid_rows:
            if parent_uri in seen:
                raise ClassificationError("UDC navigation graph contains a cycle")
            seen.add(parent_uri)
            parent = package.concepts.get(parent_uri)
            if parent is None:
                parent_uri = ""
                break
            skipped = True
            parent_uri = str(parent.get("broader_uri") or "")
        if skipped:
            contracted += 1
        parent_by_uri[uri] = parent_uri or None

    children: dict[str, list[str]] = defaultdict(list)
    for uri, parent_uri in parent_by_uri.items():
        if parent_uri:
            children[parent_uri].append(uri)
    for values in children.values():
        values.sort(key=lambda value: str(valid_rows[value].get("notation") or ""))

    nodes = {
        uri: NavigationNode(
            uri=uri,
            notation=str(row.get("notation") or ""),
            label_en=str(row.get("label_en") or row.get("label") or ""),
            label_ja=str(row.get("label_ja") or ""),
            parent_uri=parent_by_uri[uri],
            children_uris=tuple(children.get(uri, ())),
            is_range="/" in str(row.get("notation") or ""),
        )
        for uri, row in valid_rows.items()
    }
    notation_to_uri = {node.notation: uri for uri, node in nodes.items()}
    roots = tuple(
        notation_to_uri[notation]
        for notation in ROOT_NOTATIONS
        if notation in notation_to_uri
    )
    if len(roots) != len(ROOT_NOTATIONS):
        raise ClassificationError("UDC navigation graph root set is incomplete")

    reachable = set()
    queue = deque(roots)
    while queue:
        uri = queue.popleft()
        if uri in reachable:
            continue
        reachable.add(uri)
        queue.extend(nodes[uri].children_uris)
    if reachable != set(nodes):
        missing = sorted(nodes[uri].notation for uri in set(nodes) - reachable)
        raise ClassificationError(
            "UDC navigation graph contains unreachable primary concepts: "
            + ", ".join(missing[:10])
        )
    return NavigationGraph(
        schema=NAVIGATION_GRAPH_SCHEMA,
        release=package.release,
        checksum=package.checksum,
        roots=roots,
        nodes=nodes,
        notation_to_uri=notation_to_uri,
        contracted_parent_count=contracted,
    )


def deterministic_evidence_capsule(
    page: Mapping[str, Any],
    *,
    max_excerpt_chars: int = 1_200,
) -> dict[str, str]:
    """Keep raw correction evidence without frontmatter or repeated headings."""

    title = str(page.get("title") or "").strip()
    summary = str(page.get("summary") or "").strip()
    excerpt = str(page.get("excerpt") or "")
    lines = excerpt.splitlines()
    in_frontmatter = False
    body_lines = []
    for line in lines:
        if _FRONTMATTER_FENCE.match(line):
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            continue
        cleaned = _MARKDOWN_HEADING.sub("", line).strip()
        if not cleaned or cleaned == title:
            continue
        body_lines.append(cleaned)
    body = "\n".join(body_lines)[:max_excerpt_chars].strip()
    return {"title": title, "summary": summary, "evidence_excerpt": body}


def navigation_cards(
    graph: NavigationGraph,
    parent_notations: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Return a stable deduplicated option set for one or more beam parents."""

    parents = list(parent_notations or [])
    nodes = graph.children(None) if not parents else tuple(
        child
        for parent in parents
        for child in graph.children(parent)
    )
    by_notation = {node.notation: node for node in nodes}
    return [by_notation[key].card() for key in sorted(by_notation)]
