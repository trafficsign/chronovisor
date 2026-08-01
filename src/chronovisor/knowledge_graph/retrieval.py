"""Typed relation provider, query planning, and browser-safe path traces."""

from __future__ import annotations

import hashlib
import re
import threading
from collections import Counter, deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from typing import Any

from chronovisor.core.durable_state import DurableStateError, read_sealed_json
from chronovisor.core.store import CHRONOVISOR_ROOT
from chronovisor.knowledge_graph.config import KnowledgeGraphConfig, load_config
from chronovisor.knowledge_graph.rollout import selected_for_canary
from chronovisor.knowledge_graph.store import KnowledgeGraphStore

GLOBAL_QUERY_RE = re.compile(
    r"(?:全体|まとめ|傾向|横断|共通|overview|overall|across|synthesize)", re.IGNORECASE
)
MULTIHOP_QUERY_RE = re.compile(
    r"(?:関係|経由|つなが|なぜ.*から|組み合わせ|relation|connect|between|multi.?hop)",
    re.IGNORECASE,
)
_MODE_OVERRIDE = threading.local()


@dataclass(frozen=True)
class RelationNeighbor:
    target: str
    weight: float
    relation_id: str
    predicate: str
    direction: str
    status: str
    evidence_refs: tuple[dict[str, Any], ...]
    supervision: str


@dataclass(frozen=True)
class CommunityCandidate:
    page_id: str
    score: float
    community_id: str
    relation_ids: tuple[str, ...]
    source_digests: tuple[str, ...]
    summary_sha256: str


def classify_query(query: str) -> str:
    if GLOBAL_QUERY_RE.search(query):
        return "global"
    if MULTIHOP_QUERY_RE.search(query):
        return "local"
    terms = [term for term in re.split(r"\s+", query.strip()) if term]
    return "direct" if len(terms) <= 6 else "mixed"


def authority_statuses(mode: str, *, for_field: bool = False) -> frozenset[str]:
    if mode in {"off", "shadow"}:
        return frozenset()
    if mode == "candidate":
        return frozenset(
            {"repeatedly_used"}
            if for_field
            else {"verified", "repeatedly_used", "authoritative"}
        )
    return frozenset({"authoritative"} if for_field else {"authoritative"})


def effective_retrieval_mode(configured: str, *, rollout_key: str = "") -> str:
    override = str(getattr(_MODE_OVERRIDE, "value", ""))
    if override in {"off", "shadow", "candidate", "active"}:
        return override
    if configured in {"off", "active"}:
        return configured
    try:
        promotion = read_sealed_json(
            CHRONOVISOR_ROOT / "runtime" / "typed-graph" / "promotion.json",
            recover_backup=True,
        )
    except (DurableStateError, OSError, ValueError):
        return configured
    mode = str(promotion.get("mode") or "shadow")
    percent = int(promotion.get("canary_percent") or 0)
    if mode == "active" and percent >= 100:
        return "active"
    if mode == "candidate" and selected_for_canary(rollout_key, percent):
        return "candidate"
    return configured


@contextmanager
def retrieval_mode_override(mode: str) -> Iterator[None]:
    if mode not in {"off", "shadow", "candidate", "active"}:
        raise ValueError("invalid retrieval override")
    previous = getattr(_MODE_OVERRIDE, "value", "")
    _MODE_OVERRIDE.value = mode
    try:
        yield
    finally:
        _MODE_OVERRIDE.value = previous


def relation_neighbors(
    page_id: str,
    *,
    store: KnowledgeGraphStore | None = None,
    config: KnowledgeGraphConfig | None = None,
    for_field: bool = False,
    limit: int | None = None,
    rollout_key: str = "",
) -> list[RelationNeighbor]:
    cfg = config or load_config()
    effective_mode = effective_retrieval_mode(
        cfg.retrieval.mode, rollout_key=rollout_key
    )
    statuses = authority_statuses(effective_mode, for_field=for_field)
    if not cfg.enabled or not statuses:
        return []
    graph_store = store or KnowledgeGraphStore()
    rows = graph_store.relations(statuses=statuses)
    degree: Counter[str] = Counter()
    for row in rows:
        degree[row.source_page_id] += 1
        degree[row.target_page_id] += 1
    per_predicate: Counter[str] = Counter()
    neighbors: list[RelationNeighbor] = []
    for row in rows:
        target = ""
        direction = row.direction
        if row.source_page_id == page_id:
            target = row.target_page_id
        elif row.target_page_id == page_id and row.direction in {
            "reverse",
            "bidirectional",
        }:
            target = row.source_page_id
            direction = "reverse"
        if not target or target == page_id:
            continue
        if per_predicate[row.predicate] >= cfg.retrieval.per_predicate_cap:
            continue
        per_predicate[row.predicate] += 1
        hub = max(1, degree[page_id], degree[target])
        weight = row.confidence / (1.0 + cfg.retrieval.hub_penalty * (hub - 1))
        neighbors.append(
            RelationNeighbor(
                target=target,
                weight=max(0.0, min(1.0, weight)),
                relation_id=row.relation_id,
                predicate=row.predicate,
                direction=direction,
                status=row.status,
                evidence_refs=tuple(
                    {
                        "page_id": evidence.page_id,
                        "content_sha256": evidence.content_sha256,
                        "span_sha256": evidence.span_sha256,
                        "source_line": evidence.source_line,
                    }
                    for evidence in row.evidence
                ),
                supervision="typed_relation",
            )
        )
    cap = cfg.retrieval.max_relations_per_node if limit is None else limit
    return sorted(
        neighbors, key=lambda row: (-row.weight, row.predicate, row.relation_id)
    )[: max(0, cap)]


def entity_merge_neighbors(
    page_id: str,
    *,
    store: KnowledgeGraphStore | None = None,
    config: KnowledgeGraphConfig | None = None,
    for_field: bool = False,
    limit: int | None = None,
    rollout_key: str = "",
) -> list[RelationNeighbor]:
    """Expose reversible verified alias merges as separately supervised edges."""

    cfg = config or load_config()
    mode = effective_retrieval_mode(cfg.retrieval.mode, rollout_key=rollout_key)
    statuses = authority_statuses(mode, for_field=for_field)
    if not cfg.enabled or not statuses:
        return []
    graph_store = store or KnowledgeGraphStore()
    try:
        payload = read_sealed_json(
            graph_store.entity_snapshot_file, recover_backup=True
        )
    except (DurableStateError, OSError, ValueError):
        return []
    candidate_values = payload.get("candidates")
    merge_values = payload.get("merge_candidates")
    candidates = candidate_values if isinstance(candidate_values, dict) else {}
    merges = merge_values if isinstance(merge_values, dict) else {}
    output: list[RelationNeighbor] = []
    for merge_id, merge in sorted(merges.items()):
        if not isinstance(merge, dict) or str(merge.get("status") or "") not in statuses:
            continue
        members = [
            candidates.get(str(candidate_id))
            for candidate_id in merge.get("member_candidate_ids") or []
        ]
        rows = [row for row in members if isinstance(row, dict)]
        page_rows = [row for row in rows if str(row.get("page_id") or "") == page_id]
        if not page_rows:
            continue
        targets = sorted(
            {
                str(row.get("page_id") or "")
                for row in rows
                if str(row.get("page_id") or "") not in {"", page_id}
            }
        )
        weight = 0.72 / max(1.0, (len(targets) + 1) ** 0.5)
        evidence_refs = tuple(
            {
                "page_id": str(row.get("page_id") or ""),
                "content_sha256": str(row.get("content_sha256") or ""),
                "span_sha256": str(row.get("alias_evidence_sha256") or ""),
                "source_line": 0,
            }
            for row in rows
        )
        for target in targets:
            output.append(
                RelationNeighbor(
                    target=target,
                    weight=weight,
                    relation_id=str(merge_id),
                    predicate="same_entity_alias",
                    direction="bidirectional",
                    status=str(merge.get("status") or "verified"),
                    evidence_refs=evidence_refs,
                    supervision="entity_merge",
                )
            )
    cap = cfg.retrieval.max_relations_per_node if limit is None else limit
    return sorted(output, key=lambda row: (-row.weight, row.relation_id, row.target))[
        : max(0, cap)
    ]


def community_candidates(
    seed_page_ids: list[str],
    *,
    query: str = "",
    store: KnowledgeGraphStore | None = None,
    config: KnowledgeGraphConfig | None = None,
    rollout_key: str = "",
    limit: int | None = None,
) -> list[CommunityCandidate]:
    """Return members of seed-overlapping communities for global questions only."""

    cfg = config or load_config()
    mode = effective_retrieval_mode(cfg.retrieval.mode, rollout_key=rollout_key)
    statuses = authority_statuses(mode)
    if not cfg.enabled or not statuses or not seed_page_ids:
        return []
    graph_store = store or KnowledgeGraphStore()
    eligible_relation_ids = {
        row.relation_id for row in graph_store.relations(statuses=statuses)
    }
    try:
        payload = read_sealed_json(
            graph_store.community_snapshot_file, recover_backup=True
        )
    except (DurableStateError, OSError, ValueError):
        return []
    values = payload.get("communities")
    if not isinstance(values, dict):
        return []
    seeds = set(seed_page_ids[:20])
    query_terms = {
        term.casefold()
        for term in re.findall(r"[\w-]{2,}", query, flags=re.UNICODE)
    }
    output: dict[str, CommunityCandidate] = {}
    for community_id, value in sorted(values.items()):
        if not isinstance(value, dict):
            continue
        members = tuple(str(item) for item in value.get("member_page_ids") or [])
        relation_ids = tuple(
            relation_id
            for relation_id in (str(item) for item in value.get("relation_ids") or [])
            if relation_id in eligible_relation_ids
        )
        overlap = seeds.intersection(members)
        if not overlap or not relation_ids:
            continue
        summary = str(value.get("summary") or "")
        summary_terms = {
            term.casefold()
            for term in re.findall(r"[\w-]{2,}", summary, flags=re.UNICODE)
        }
        lexical = len(query_terms & summary_terms) / max(1, len(query_terms))
        base = len(overlap) / max(1.0, len(members) ** 0.5)
        score = min(1.0, base + 0.2 * lexical)
        source_digests = tuple(
            str(item) for item in value.get("source_digests") or []
        )
        for page_id in members:
            if page_id in seeds:
                continue
            candidate = CommunityCandidate(
                page_id=page_id,
                score=score,
                community_id=str(community_id),
                relation_ids=relation_ids,
                source_digests=source_digests,
                summary_sha256=str(value.get("summary_sha256") or ""),
            )
            current = output.get(page_id)
            if current is None or candidate.score > current.score:
                output[page_id] = candidate
    cap = cfg.retrieval.max_candidate_pages if limit is None else limit
    return sorted(
        output.values(), key=lambda row: (-row.score, row.community_id, row.page_id)
    )[: max(0, cap)]


def trace_paths(
    seeds: list[str],
    *,
    store: KnowledgeGraphStore | None = None,
    config: KnowledgeGraphConfig | None = None,
) -> dict[str, dict[str, Any]]:
    cfg = config or load_config()
    if cfg.retrieval.mode == "off":
        return {}
    queue: deque[tuple[str, int, tuple[str, ...], tuple[dict[str, Any], ...]]] = (
        deque((seed, 0, (seed,), ()) for seed in seeds[:20])
    )
    best: dict[str, float] = {seed: 1.0 for seed in seeds[:20]}
    traces: dict[str, dict[str, Any]] = {}
    visited_edges = 0
    while queue and visited_edges < 200:
        page_id, hop, pages, relations = queue.popleft()
        if hop >= cfg.retrieval.max_hops:
            continue
        for edge in [
            *relation_neighbors(page_id, store=store, config=cfg),
            *entity_merge_neighbors(page_id, store=store, config=cfg),
        ]:
            visited_edges += 1
            if edge.target in pages:
                continue
            score = best.get(page_id, 1.0) * edge.weight * (0.72 ** (hop + 1))
            if score < 0.005 or score <= best.get(edge.target, 0.0):
                continue
            best[edge.target] = score
            next_pages = (*pages, edge.target)
            relation_trace = {
                "relation_id": edge.relation_id,
                "predicate": edge.predicate,
                "direction": edge.direction,
                "status": edge.status,
                "evidence_refs": list(edge.evidence_refs),
                "weight": round(edge.weight, 6),
            }
            next_relations = (*relations, relation_trace)
            traces[edge.target] = {
                "path_id": "path_"
                + hashlib.sha256(
                    "|".join(
                        [*next_pages, *(row["relation_id"] for row in next_relations)]
                    ).encode()
                ).hexdigest()[:24],
                "pages": list(next_pages),
                "relations": list(next_relations),
                "hops": hop + 1,
                "activation": round(score, 6),
            }
            queue.append((edge.target, hop + 1, next_pages, next_relations))
            if len(traces) >= cfg.retrieval.max_candidate_pages:
                return traces
    return traces


def shadow_candidate_paths(
    seeds: list[str],
    *,
    query: str,
    store: KnowledgeGraphStore | None = None,
    config: KnowledgeGraphConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Observe candidate GraphRAG paths without changing production ranking."""

    cfg = config or load_config()
    query_plan = classify_query(query)
    if not cfg.enabled or cfg.retrieval.mode != "shadow" or query_plan == "direct":
        return {}
    candidate_cfg = dataclass_replace(
        cfg,
        retrieval=dataclass_replace(cfg.retrieval, mode="candidate"),
    )
    graph_store = store or KnowledgeGraphStore()
    if query_plan == "global":
        output: dict[str, dict[str, Any]] = {}
        for row in community_candidates(
            seeds,
            query=query,
            store=graph_store,
            config=candidate_cfg,
        ):
            pages = [seeds[0], row.page_id] if seeds else [row.page_id]
            output[row.page_id] = {
                "path_id": "path_"
                + hashlib.sha256(
                    "|".join([row.community_id, *pages, *row.relation_ids]).encode()
                ).hexdigest()[:24],
                "pages": pages,
                "relations": [],
                "relation_ids": list(row.relation_ids),
                "community_id": row.community_id,
                "source_digests": list(row.source_digests),
                "summary_sha256": row.summary_sha256,
                "hops": 0,
                "activation": round(row.score, 6),
                "query_plan": query_plan,
                "shadow": True,
            }
        return output
    return {
        page_id: {
            **value,
            "relation_ids": [
                str(row.get("relation_id") or "")
                for row in value.get("relations", [])
                if isinstance(row, dict) and row.get("relation_id")
            ],
            "query_plan": query_plan,
            "shadow": True,
        }
        for page_id, value in trace_paths(
            seeds, store=graph_store, config=candidate_cfg
        ).items()
    }
