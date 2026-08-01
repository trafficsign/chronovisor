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
    relation_id: str = ""
    predicate: str = ""
    direction: str = ""
    lifecycle: str = ""
    evidence_refs: tuple[dict[str, Any], ...] = ()


def typed_neighbors(
    store: Any,
    page_id: str,
    *,
    limit: int = 12,
    include_exposure_cofire: bool = True,
    include_positive_cofire: bool = True,
    degree_normalize: bool = False,
    include_typed_relations: bool = True,
    typed_relations_for_field: bool = False,
    rollout_key: str = "",
) -> list[TypedEdge]:
    """Return degree-normalized, deterministic neighbors for one page."""

    edges: dict[str, TypedEdge] = {}

    def add(
        target: str,
        weight: float,
        edge_type: str,
        supervision: str = "",
        *,
        relation_id: str = "",
        predicate: str = "",
        direction: str = "",
        lifecycle: str = "",
        evidence_refs: tuple[dict[str, Any], ...] = (),
    ) -> None:
        if not target or target == page_id:
            return
        edge = TypedEdge(
            target=target,
            weight=max(0.0, min(1.0, weight)),
            edge_type=edge_type,
            supervision=supervision,
            relation_id=relation_id,
            predicate=predicate,
            direction=direction,
            lifecycle=lifecycle,
            evidence_refs=evidence_refs,
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

    if include_typed_relations:
        try:
            from chronovisor.knowledge_graph.retrieval import (
                entity_merge_neighbors,
                relation_neighbors,
            )

            relation_rows = relation_neighbors(
                page_id,
                for_field=typed_relations_for_field,
                rollout_key=rollout_key,
            )
            entity_rows = entity_merge_neighbors(
                page_id,
                for_field=typed_relations_for_field,
                rollout_key=rollout_key,
            )
            for typed_row in [*relation_rows, *entity_rows]:
                add(
                    typed_row.target,
                    typed_row.weight,
                    "typed_entity_merge"
                    if typed_row.supervision == "entity_merge"
                    else "typed_relation",
                    typed_row.supervision,
                    relation_id=typed_row.relation_id,
                    predicate=typed_row.predicate,
                    direction=typed_row.direction,
                    lifecycle=typed_row.status,
                    evidence_refs=typed_row.evidence_refs,
                )
        except Exception:
            # A derived/candidate graph can never take down baseline search.
            pass

    return sorted(
        (edge for edge in edges.values() if edge.weight > 0),
        key=lambda edge: (-edge.weight, edge.edge_type, edge.target),
    )[: max(0, limit)]
