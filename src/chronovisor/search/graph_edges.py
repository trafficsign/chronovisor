"""Reusable typed graph-neighbor generation for Search and Recall Field."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TypedEdge:
    target: str
    weight: float
    edge_type: str
    supervision: str = ""


def typed_neighbors(
    store: Any,
    page_id: str,
    *,
    limit: int = 12,
    include_exposure_cofire: bool = True,
    include_positive_cofire: bool = True,
    degree_normalize: bool = False,
) -> list[TypedEdge]:
    """Return degree-normalized, deterministic neighbors for one page."""

    edges: dict[str, TypedEdge] = {}

    def add(
        target: str,
        weight: float,
        edge_type: str,
        supervision: str = "",
    ) -> None:
        if not target or target == page_id:
            return
        edge = TypedEdge(
            target=target,
            weight=max(0.0, min(1.0, weight)),
            edge_type=edge_type,
            supervision=supervision,
        )
        current = edges.get(target)
        if current is None or (edge.weight, edge.edge_type, edge.target) > (
            current.weight,
            current.edge_type,
            current.target,
        ):
            edges[target] = edge

    outlinks = list(store.outlinks(page_id))
    out_degree = max(1, len(outlinks))
    for target in outlinks:
        add(
            target,
            1.0 / math.sqrt(out_degree) if degree_normalize else 1.0,
            "wikilink",
        )

    backlinks = list(store.backlinks(page_id))
    back_degree = max(1, len(backlinks))
    for target in backlinks:
        add(
            target,
            0.85 / math.sqrt(back_degree) if degree_normalize else 0.85,
            "backlink",
        )

    for tag in store.tags(page_id):
        related = list(store.pages_for_tag(tag))
        degree = max(1, len(related) - 1)
        weight = (
            0.55 / math.sqrt(degree)
            if degree_normalize
            else 0.55 / math.sqrt(max(1.0, degree / 4.0))
        )
        for target in related[:12]:
            add(target, weight, "tag")

    meta = store.meta(page_id) or {}
    for entity in meta.get("entities", []):
        related = list(store.pages_for_entity(str(entity)))
        degree = max(1, len(related) - 1)
        weight = (
            0.75 / math.sqrt(degree)
            if degree_normalize
            else 0.75 / math.sqrt(max(1.0, degree / 4.0))
        )
        for target in related[:12]:
            add(target, weight, "entity")

    try:
        from chronovisor.search.cofire import neighbors as cofire_neighbors

        for row in cofire_neighbors(
            page_id,
            limit=8,
            positive_weight=1.0,
            exposure_weight=0.05 if include_exposure_cofire else 0.0,
        ):
            signals = row.get("signals")
            positive = isinstance(signals, list) and "positive_used" in signals
            if positive and not include_positive_cofire:
                continue
            if not positive and not include_exposure_cofire:
                continue
            add(
                str(row.get("page_id") or ""),
                float(row.get("weight") or 0.0),
                "cofire",
                "positive_used" if positive else "exposure",
            )
    except Exception:
        pass

    return sorted(
        (edge for edge in edges.values() if edge.weight > 0),
        key=lambda edge: (-edge.weight, edge.edge_type, edge.target),
    )[: max(0, limit)]
