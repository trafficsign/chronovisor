"""Core search engine with BM25 and semantic RRF fusion."""

import contextlib
import hashlib
import heapq
import json
import math
import os
import re
import threading
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

from chronovisor.core.lexical_index import LexicalIndex
from chronovisor.core.negative_feedback import apply_penalties, penalties_for_query
from chronovisor.core.pipeline import (
    PipelineDependencies,
    production_pipeline_config,
    run_search_pipeline,
)
from chronovisor.core.runtime_config import (
    load_negative_feedback_config,
    load_search_embedding_config,
)
from chronovisor.core.search_types import ScoredPage
from chronovisor.core.store import (
    CHRONOVISOR_ROOT,
    PAGES_DIR,
)


def searchable_pages() -> list[Path]:
    """Return stable canonical pages from the shared metadata snapshot."""

    from chronovisor.core.index_store import get_store

    store = get_store()
    _refresh_store_for_search(store)
    paths = []
    for page_id in store.all_page_ids(include_system=True):
        meta = store.meta(page_id)
        path = meta.get("path") if meta is not None else None
        if isinstance(path, str):
            paths.append(Path(path))
    return sorted(paths)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

_BM25_CACHE_FILE = CHRONOVISOR_ROOT / ".index" / "lexical.sqlite"
_LEGACY_BM25_CACHE_FILE = CHRONOVISOR_ROOT / ".index" / "bm25.json"
_STABLE_STATUS = "stable"
_VALID_LIFECYCLE_STATUSES = {"draft", _STABLE_STATUS, "deprecated"}
_REFERENCE_PAGE_TYPE = "reference"
_VALID_PAGE_TYPES = {
    "knowledge",
    _REFERENCE_PAGE_TYPE,
    "episodic",
    "semantic",
    "procedural",
    "state",
    "lesson",
    "decision",
}
_VALID_SENSITIVITY_TIERS = {"normal", "high"}


def _normalize_lifecycle_status(value: object) -> str:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if normalized in _VALID_LIFECYCLE_STATUSES:
        return normalized
    return "draft"


def _normalize_page_type(value: object, *, folder: str = "") -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _VALID_PAGE_TYPES:
            return normalized
    if folder == "car-spec":
        return _REFERENCE_PAGE_TYPE
    return "knowledge"


def _normalize_sensitivity(value: object, *, folder: str = "") -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _VALID_SENSITIVITY_TIERS:
            return normalized
    if folder == "career":
        return "high"
    return "normal"


def _is_stable_result(result: ScoredPage) -> bool:
    return _normalize_lifecycle_status(result.status) == _STABLE_STATUS


def _is_reference_result(result: ScoredPage) -> bool:
    return (
        _normalize_page_type(result.page_type, folder=result.folder)
        == _REFERENCE_PAGE_TYPE
    )


def _folder_from_meta(meta: dict[str, Any]) -> str:
    try:
        parent = Path(meta["path"]).parent
        if parent != PAGES_DIR:
            return parent.name
    except (KeyError, TypeError):
        pass
    return ""


def _meta_page_type(meta: dict[str, Any], *, folder: str = "") -> str:
    return _normalize_page_type(meta.get("page_type"), folder=folder)


def _meta_sensitivity(meta: dict[str, Any], *, folder: str = "") -> str:
    return _normalize_sensitivity(meta.get("sensitivity"), folder=folder)


def _refresh_store_for_search(store: object) -> None:
    """Share a short read snapshot while preserving injected-store compatibility."""

    refresh_if_stale = getattr(store, "refresh_if_stale", None)
    if callable(refresh_if_stale):
        refresh_if_stale()
        return
    refresh = cast(Any, store).refresh
    refresh()


# ---------------------------------------------------------------------------
# BM25 singleton — shared across `search()` and `ingest._search_related_pages`
# ---------------------------------------------------------------------------


class BM25Index(LexicalIndex):
    """Compatibility name for the adopted SQLite inverted lexical index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        # k1/b remain accepted so older debug/test callers do not break.
        del k1, b
        super().__init__(path=_BM25_CACHE_FILE, pages=searchable_pages)

    def build(self, *, force: bool = False) -> None:
        super().build(force=force)
        if os.environ.get("CHRONOVISOR_READ_ONLY") != "1":
            _LEGACY_BM25_CACHE_FILE.unlink(missing_ok=True)


def lexical_cache_paths() -> tuple[Path, ...]:
    return (
        _BM25_CACHE_FILE,
        Path(f"{_BM25_CACHE_FILE}-wal"),
        Path(f"{_BM25_CACHE_FILE}-shm"),
        _LEGACY_BM25_CACHE_FILE,
    )


_BM25_LOCK = threading.Lock()
_BM25_SINGLETON: BM25Index | None = None


def get_bm25() -> BM25Index:
    """Return the process-wide BM25Index instance.

    All callers go through this so that the disk-cache load + persisted
    state are paid at most once per process; subsequent `build()` calls
    are O(stat) when nothing changed.
    """
    global _BM25_SINGLETON
    if _BM25_SINGLETON is None:
        with _BM25_LOCK:
            if _BM25_SINGLETON is None:
                _BM25_SINGLETON = BM25Index()
    return _BM25_SINGLETON


def search_existing_bm25(query: str, *, top_n: int = 20) -> list[ScoredPage]:
    """Read the existing lexical projection without refresh, build, or writes."""

    return apply_filters(get_bm25().query_existing(query, top_n=top_n))


def search_existing_anchors(query: str, *, top_n: int = 20) -> list[ScoredPage]:
    """Read exact-anchor matches without refreshing, building, or writing."""

    return apply_filters(get_bm25().anchor_query_existing(query, top_n=top_n))


def search_existing_lexical(
    query: str, *, top_n: int = 20
) -> tuple[list[ScoredPage], list[ScoredPage]]:
    """Return read-only ``(anchor, BM25)`` candidates from one projection."""

    index = get_bm25()
    return (
        apply_filters(index.anchor_query_existing(query, top_n=top_n)),
        apply_filters(index.query_existing(query, top_n=top_n)),
    )


# Semantic document projection shared by the immutable service index.
MAX_CHUNKS_PER_PAGE = 8
MAX_CHUNK_CHARS = 900


def _recall_questions(meta: dict[str, Any]) -> list[str]:
    questions = meta.get("recall_questions")
    if not isinstance(questions, list):
        return []
    return [q for q in questions if isinstance(q, str) and q.strip()][:8]


def _markdown_chunks(body: str, title: str, meta: dict[str, Any]) -> list[str]:
    chunks: list[str] = []
    heading = title
    buffer: list[str] = []

    summary_value = meta.get("description") or meta.get("summary")
    summary = summary_value if isinstance(summary_value, str) else ""
    entities = meta.get("entities") if isinstance(meta.get("entities"), list) else []
    page_type = meta.get("page_type") if isinstance(meta.get("page_type"), str) else ""
    updated = meta.get("updated") if isinstance(meta.get("updated"), str) else ""
    context_lines = [f"Page: {title}"]
    if summary.strip():
        context_lines.append(f"Summary: {summary.strip()}")
    if entities:
        context_lines.append(
            "Entities: " + ", ".join(str(item) for item in entities[:8])
        )
    if page_type or updated:
        context_lines.append(
            f"Metadata: type={page_type or 'knowledge'} updated={updated or 'unknown'}"
        )
    context = "\n".join(context_lines)

    def flush() -> None:
        nonlocal buffer
        text = re.sub(r"\s+", " ", "\n".join(buffer)).strip()
        buffer = []
        if not text:
            return
        prefix = (
            f"{context}\nHeading: {heading}"
            if heading and heading != title
            else context
        )
        while text:
            piece = text[:MAX_CHUNK_CHARS].strip()
            if len(text) > MAX_CHUNK_CHARS and " " in piece:
                piece = (
                    piece.rsplit(" ", 1)[0].strip() or text[:MAX_CHUNK_CHARS].strip()
                )
            chunks.append(f"{prefix}\n\n{piece}")
            text = text[len(piece) :].strip()
            if len(chunks) >= MAX_CHUNKS_PER_PAGE:
                return

    for line in body.splitlines():
        stripped = line.strip()
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            flush()
            heading = heading_match.group(2).strip()
            if len(chunks) >= MAX_CHUNKS_PER_PAGE:
                break
            continue
        if not stripped:
            flush()
            if len(chunks) >= MAX_CHUNKS_PER_PAGE:
                break
            continue
        buffer.append(stripped)
        if sum(len(item) for item in buffer) >= MAX_CHUNK_CHARS:
            flush()
            if len(chunks) >= MAX_CHUNKS_PER_PAGE:
                break
    if len(chunks) < MAX_CHUNKS_PER_PAGE:
        flush()
    if not chunks and title.strip():
        chunks.append(title.strip())
    return chunks[:MAX_CHUNKS_PER_PAGE]


def update_embeddings(
    page_ids: list[str] | None = None, *, strict: bool = False
) -> int:
    """Refresh the configured semantic-service index.

    Nemotron writes are durable jobs.  Correctness-critical callers may use
    ``strict=True`` to wait for generation-scoped delta publication.
    """

    config = load_search_embedding_config()
    if not config.enabled:
        return 0
    from chronovisor.core.semantic_jobs import enqueue_pages, enqueue_rebuild

    if page_ids is None:
        enqueue_rebuild()
        return 0
    unique = sorted({page_id for page_id in page_ids if page_id})
    if strict:
        from chronovisor.core.semantic_client import index_pages

        response = index_pages(unique, config, wait=True)
        return int(response.get("pages_updated") or 0)

    from chronovisor.core.semantic_index import extract_page_documents
    from chronovisor.core.store import SYSTEM_DIR, find_page

    hashes: dict[str, str] = {}
    for page_id in unique:
        path = find_page(page_id)
        if path is None:
            system_path = SYSTEM_DIR / f"{page_id}.md"
            path = system_path if system_path.is_file() else None
        documents = extract_page_documents(path) if path is not None else []
        hashes[page_id] = documents[0].source_sha256 if documents else ""
    enqueue_pages(unique, source_hashes=hashes)
    return len(unique)


def semantic_search(
    query: str,
    top_n: int = 20,
    *,
    include_reference: bool = False,
    strict: bool = False,
    timeout_ms: int | None = None,
) -> list[ScoredPage]:
    """Search through the selected execution topology."""

    config = load_search_embedding_config()
    if not config.enabled:
        return []
    from chronovisor.core import semantic_client

    use_new = semantic_client.selected_for_rollout(query, config)
    if config.rollout_mode == "shadow":
        with contextlib.suppress(Exception):
            semantic_client.search(
                query,
                top_n,
                include_reference=include_reference,
                config=config,
                timeout_ms=timeout_ms,
            )
        return []
    if not use_new:
        return []
    return semantic_client.search(
        query,
        top_n,
        include_reference=include_reference,
        config=config,
        timeout_ms=timeout_ms,
    )


def semantic_verify(
    query: str,
    page_ids: list[str],
    *,
    timeout_ms: int | None = None,
) -> list[ScoredPage]:
    """Run full-dimensional verification for graph-generated candidates."""

    config = load_search_embedding_config()
    if not page_ids or not config.enabled:
        return []
    from chronovisor.core import semantic_client

    if not semantic_client.selected_for_rollout(query, config):
        return []
    rows = semantic_client.verify(
        query,
        page_ids,
        config=config,
        timeout_ms=timeout_ms,
    )
    return [page for page in rows if float(page.score) >= float(config.min_top_score)]


def context_seed_results(query: str, *, limit: int = 4) -> list[ScoredPage]:
    """Use only explicit recall-used evidence as a weak independent entrance."""

    try:
        from chronovisor.core.index_store import get_store
        from chronovisor.core.prefetch import prefetch_page_ids

        page_ids = prefetch_page_ids(
            host="",
            cwd="",
            queries=[query],
            prompt=query,
            limit=max(1, min(8, limit)),
            positive_weight=1,
            exposure_weight=0,
        )
        if not page_ids:
            return []
        store = get_store()
        _refresh_store_for_search(store)
    except Exception:
        return []
    out: list[ScoredPage] = []
    for rank, page_id in enumerate(page_ids):
        meta = store.meta(page_id)
        if meta is None or _normalize_lifecycle_status(meta.get("status")) != "stable":
            continue
        folder = _folder_from_meta(meta)
        out.append(
            ScoredPage(
                page_id=page_id,
                title=str(meta.get("title") or page_id),
                folder=folder,
                updated=str(meta.get("updated") or ""),
                score=1.0 / (1.0 + rank),
                status=_normalize_lifecycle_status(meta.get("status")),
                superseded_by=(
                    str(meta.get("superseded_by") or "")
                    if isinstance(meta.get("superseded_by"), str)
                    else ""
                ),
                page_type=_meta_page_type(meta, folder=folder),
                sensitivity=_meta_sensitivity(meta, folder=folder),
            )
        )
    return out


# ---------------------------------------------------------------------------
# RRF Fusion + Filter + Sort
# ---------------------------------------------------------------------------

DEFAULT_FUSION_WEIGHTS: dict[str, float] = {
    "anchor": 0.9,
    "bm25": 1.0,
    "semantic": 0.6,
    "graph": 0.3,
    "context": 0.25,
    "usage_prior": 0.0,
    "bm25_score_bonus": 0.005,
    "bm25_rank_bonus": 0.006,
    "bm25_rank_decay": 0.006,
    "semantic_min_top_score": 0.45,
    "semantic_min_margin": 0.002,
    "semantic_low_confidence_weight": 0.25,
    "usage_prior_decay": 0.98,
    "usage_prior_cap": 3.0,
    "retention_prior": 0.015,
}

ACTIVE_SEARCH_POLICY_FILE = CHRONOVISOR_ROOT / "recall" / "search-policy.json"


def load_active_fusion_weights(path: Path | None = None) -> dict[str, float]:
    """Load the validated search policy, falling back safely to defaults.

    The artifact is deliberately separate from user configuration: evaluation
    can atomically replace it after holdout validation, while malformed or
    stale artifacts can never make production search unusable.
    """
    policy_path = path or ACTIVE_SEARCH_POLICY_FILE
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_FUSION_WEIGHTS)
    if (
        not isinstance(payload, dict)
        or payload.get("version") != 1
        or payload.get("source") != "search_eval.self_tune"
        or not isinstance(payload.get("holdout"), dict)
    ):
        return dict(DEFAULT_FUSION_WEIGHTS)
    raw = payload.get("weights")
    if not isinstance(raw, dict) or set(raw) != set(DEFAULT_FUSION_WEIGHTS):
        return dict(DEFAULT_FUSION_WEIGHTS)
    weights: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool):
            return dict(DEFAULT_FUSION_WEIGHTS)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return dict(DEFAULT_FUSION_WEIGHTS)
        if not math.isfinite(numeric) or numeric < 0:
            return dict(DEFAULT_FUSION_WEIGHTS)
        weights[key] = numeric
    if not any(
        weights[channel] > 0
        for channel in (
            "anchor",
            "bm25",
            "semantic",
            "graph",
            "context",
            "usage_prior",
        )
    ):
        return dict(DEFAULT_FUSION_WEIGHTS)
    for bounded in (
        "semantic_min_top_score",
        "semantic_min_margin",
        "semantic_low_confidence_weight",
        "usage_prior_decay",
    ):
        if weights[bounded] > 1:
            return dict(DEFAULT_FUSION_WEIGHTS)
    return weights


def _semantic_reliability_multiplier(
    semantic_results: list[ScoredPage],
    weights: dict[str, float],
) -> float:
    if not semantic_results:
        return 0.0
    min_top = max(0.0, float(weights.get("semantic_min_top_score", 0.0)))
    min_margin = max(0.0, float(weights.get("semantic_min_margin", 0.0)))
    low_weight = max(0.0, float(weights.get("semantic_low_confidence_weight", 1.0)))
    top1 = max(0.0, float(semantic_results[0].score))
    top2 = (
        max(0.0, float(semantic_results[1].score)) if len(semantic_results) > 1 else 0.0
    )
    if top1 < min_top or (top1 - top2) < min_margin:
        return low_weight
    return 1.0


def fuse_results(
    bm25_results: list[ScoredPage],
    semantic_results: list[ScoredPage],
    graph_results: list[ScoredPage] | None = None,
    usage_results: list[ScoredPage] | None = None,
    k: int = 60,
    weights: dict[str, float] | None = None,
    *,
    anchor_results: list[ScoredPage] | None = None,
    context_results: list[ScoredPage] | None = None,
) -> list[ScoredPage]:
    """Weighted Reciprocal Rank Fusion of result lists."""
    scores: dict[str, float] = {}
    meta: dict[str, ScoredPage] = {}
    weights = {**DEFAULT_FUSION_WEIGHTS, **(weights or {})}
    bm25_score_bonus = max(0.0, float(weights.get("bm25_score_bonus", 0.0)))
    bm25_rank_bonus = max(0.0, float(weights.get("bm25_rank_bonus", 0.0)))
    bm25_rank_decay = max(0.0, float(weights.get("bm25_rank_decay", 0.0)))
    semantic_multiplier = _semantic_reliability_multiplier(semantic_results, weights)

    def add_results(results: list[ScoredPage], channel: str) -> None:
        weight = max(0.0, float(weights.get(channel, 1.0)))
        if channel == "semantic":
            weight *= semantic_multiplier
        if weight == 0:
            return
        top_raw = max((float(page.score) for page in results), default=0.0)
        for rank, page in enumerate(results):
            score = weight / (k + rank)
            if channel == "bm25" and top_raw > 0:
                score += (
                    weight * bm25_score_bonus * (max(0.0, float(page.score)) / top_raw)
                )
                score += weight * max(0.0, bm25_rank_bonus - (bm25_rank_decay * rank))
            scores[page.page_id] = scores.get(page.page_id, 0) + score
            if page.page_id not in meta:
                meta[page.page_id] = page

    add_results(anchor_results or [], "anchor")
    add_results(bm25_results, "bm25")
    add_results(semantic_results, "semantic")
    add_results(graph_results or [], "graph")
    add_results(context_results or [], "context")
    add_results(usage_results or [], "usage_prior")

    retention_weight = max(0.0, float(weights.get("retention_prior", 0.0)))
    if retention_weight:
        try:
            from chronovisor.core.retention import retention_score

            for page_id in list(scores):
                prior = retention_score(page_id)
                if prior > 0:
                    scores[page_id] = scores.get(page_id, 0.0) + (
                        retention_weight * prior
                    )
        except Exception:
            pass

    fused = []
    for pid, score in scores.items():
        p = meta[pid]
        fused.append(
            ScoredPage(
                page_id=p.page_id,
                title=p.title,
                folder=p.folder,
                updated=p.updated,
                score=score,
                status=p.status,
                superseded_by=p.superseded_by,
                page_type=p.page_type,
                sensitivity=p.sensitivity,
            )
        )

    fused.sort(key=lambda x: x.score, reverse=True)
    return fused


_GRAPH_TRACE = threading.local()
_GRAPH_QUERY = threading.local()
_GRAPH_ROLLOUT = threading.local()


def graph_expansion_trace() -> dict[str, dict[str, object]]:
    return dict(getattr(_GRAPH_TRACE, "paths", {}))


@contextlib.contextmanager
def graph_query_context(query: str, *, rollout_key: str = "") -> Iterator[None]:
    previous_query = getattr(_GRAPH_QUERY, "value", "")
    previous_rollout = getattr(_GRAPH_ROLLOUT, "value", "")
    _GRAPH_QUERY.value = query
    _GRAPH_ROLLOUT.value = rollout_key
    try:
        yield
    finally:
        _GRAPH_QUERY.value = previous_query
        _GRAPH_ROLLOUT.value = previous_rollout


def graph_expand_results(
    results: list[ScoredPage], *, decay: float = 0.5, limit: int = 50
) -> list[ScoredPage]:
    """Bounded two-hop associative spreading from independently found seeds."""

    _GRAPH_TRACE.paths = {}
    if decay <= 0 or not results:
        return []
    from chronovisor.core.index_store import get_store
    from chronovisor.core.knowledge_graph_retrieval import (
        classify_query,
        community_candidates,
    )

    store = get_store()
    _refresh_store_for_search(store)
    seeds = results[:20]
    seed_ids = {result.page_id for result in seeds}
    output_limit = min(max(1, limit), 50)
    best_activation: dict[str, float] = {
        result.page_id: 1.0 / (1.0 + (rank * 0.25)) for rank, result in enumerate(seeds)
    }
    frontier: list[tuple[float, int, str, tuple[str, ...], str]] = []
    for _rank, result in enumerate(seeds):
        activation = best_activation[result.page_id]
        heapq.heappush(
            frontier,
            (-activation, 0, result.page_id, (result.page_id,), "seed"),
        )

    expanded: dict[str, ScoredPage] = {}
    trace: dict[str, dict[str, Any]] = {}
    visited_states = 0
    from chronovisor.core.graph_edges import typed_neighbors

    query = str(getattr(_GRAPH_QUERY, "value", ""))
    rollout_key = str(getattr(_GRAPH_ROLLOUT, "value", ""))
    query_plan = classify_query(query)

    if query_plan == "global":
        for candidate in community_candidates(
            [result.page_id for result in seeds],
            query=query,
            rollout_key=rollout_key,
            limit=output_limit,
        ):
            if candidate.page_id in seed_ids or len(expanded) >= output_limit:
                continue
            meta = store.meta(candidate.page_id)
            if (
                meta is None
                or _normalize_lifecycle_status(meta.get("status")) != "stable"
            ):
                continue
            folder = _folder_from_meta(meta)
            expanded[candidate.page_id] = ScoredPage(
                page_id=candidate.page_id,
                title=str(meta.get("title") or candidate.page_id),
                folder=folder,
                updated=str(meta.get("updated") or ""),
                score=candidate.score * decay,
                status=_normalize_lifecycle_status(meta.get("status")),
                superseded_by=(
                    str(meta.get("superseded_by") or "")
                    if isinstance(meta.get("superseded_by"), str)
                    else ""
                ),
                page_type=_meta_page_type(meta, folder=folder),
                sensitivity=_meta_sensitivity(meta, folder=folder),
            )
            trace[candidate.page_id] = {
                "path": [candidate.page_id],
                "path_id": "path_"
                + hashlib.sha256(
                    "|".join(
                        [
                            candidate.community_id,
                            candidate.page_id,
                            *candidate.relation_ids,
                        ]
                    ).encode()
                ).hexdigest()[:24],
                "hops": 0,
                "signal": "typed_community",
                "activation": round(candidate.score, 6),
                "community_id": candidate.community_id,
                "relation_ids": list(candidate.relation_ids),
                "source_digests": list(candidate.source_digests),
                "summary_sha256": candidate.summary_sha256,
            }

    while frontier and visited_states < 200:
        negative, hop, page_id, path, incoming_signal = heapq.heappop(frontier)
        activation = -negative
        if activation + 1e-12 < best_activation.get(page_id, 0.0):
            continue
        visited_states += 1
        if hop >= 2:
            continue
        for edge in typed_neighbors(
            store,
            page_id,
            limit=12,
            include_typed_relations=query_plan in {"local", "mixed"},
            rollout_key=rollout_key,
        ):
            target = edge.target
            edge_weight = edge.weight
            signal = (
                f"{edge.edge_type}:{edge.supervision}"
                if edge.supervision
                else edge.edge_type
            )
            next_hop = hop + 1
            next_activation = activation * edge_weight * (0.72**next_hop)
            if next_activation < 0.005:
                continue
            if next_activation <= best_activation.get(target, 0.0):
                continue
            if (
                target not in seed_ids
                and target not in expanded
                and len(expanded) >= output_limit
            ):
                continue
            meta = store.meta(target)
            if (
                meta is None
                or _normalize_lifecycle_status(meta.get("status")) != "stable"
            ):
                continue
            best_activation[target] = next_activation
            next_path = (*path, target)
            prior_trace = trace.get(page_id)
            prior_relation_ids = (
                list(prior_trace.get("relation_ids") or [])
                if isinstance(prior_trace, dict)
                else []
            )
            prior_relations = (
                list(prior_trace.get("relations") or [])
                if isinstance(prior_trace, dict)
                else []
            )
            relation_ids = [
                *prior_relation_ids,
                *([edge.relation_id] if edge.relation_id else []),
            ]
            relation_steps = [
                *prior_relations,
                *(
                    [
                        {
                            "relation_id": edge.relation_id,
                            "predicate": edge.predicate,
                            "direction": edge.direction,
                            "lifecycle": edge.lifecycle,
                            "evidence_refs": list(edge.evidence_refs),
                            "weight_components": {
                                "edge": round(edge_weight, 6),
                                "hop_decay": round(0.72**next_hop, 6),
                                "source_activation": round(activation, 6),
                            },
                        }
                    ]
                    if edge.relation_id
                    else []
                ),
            ]
            heapq.heappush(
                frontier,
                (-next_activation, next_hop, target, next_path, signal),
            )
            if target in seed_ids:
                continue
            folder = _folder_from_meta(meta)
            expanded[target] = ScoredPage(
                page_id=target,
                title=str(meta.get("title") or target),
                folder=folder,
                updated=str(meta.get("updated") or ""),
                score=next_activation * decay,
                status=_normalize_lifecycle_status(meta.get("status")),
                superseded_by=(
                    str(meta.get("superseded_by") or "")
                    if isinstance(meta.get("superseded_by"), str)
                    else ""
                ),
                page_type=_meta_page_type(meta, folder=folder),
                sensitivity=_meta_sensitivity(meta, folder=folder),
            )
            trace[target] = {
                "path": list(next_path),
                "path_id": "path_"
                + hashlib.sha256(
                    "|".join([*next_path, *relation_ids]).encode()
                ).hexdigest()[:24],
                "hops": next_hop,
                "signal": signal or incoming_signal,
                "activation": round(next_activation, 6),
                "relation_id": edge.relation_id,
                "predicate": edge.predicate,
                "direction": edge.direction,
                "lifecycle": edge.lifecycle,
                "evidence_refs": list(edge.evidence_refs),
                "relation_ids": relation_ids,
                "relations": relation_steps,
            }
    _GRAPH_TRACE.paths = trace
    _GRAPH_TRACE.query_plan = query_plan
    return sorted(expanded.values(), key=lambda page: page.score, reverse=True)[
        :output_limit
    ]


def usage_prior_results(
    candidate_ids: set[str],
    *,
    limit: int = 50,
    decay: float = 0.98,
    cap: float = 3.0,
) -> list[ScoredPage]:
    if not candidate_ids:
        return []
    feedback_file = CHRONOVISOR_ROOT / "recall" / "feedback.jsonl"
    try:
        with feedback_file.open(encoding="utf-8") as f:
            lines = deque(f, maxlen=1000)
    except OSError:
        return []
    scores: dict[str, float] = {}
    decay = max(0.0, min(1.0, float(decay)))
    cap = max(0.0, float(cap))
    for age, line in enumerate(reversed(lines)):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("kind") != "injection_used":
            continue
        weight = decay**age
        for page_id in record.get("expected_pages", []) or record.get(
            "injected_pages", []
        ):
            if isinstance(page_id, str) and page_id in candidate_ids:
                scores[page_id] = min(cap, scores.get(page_id, 0.0) + weight)
    if not scores:
        return []
    from chronovisor.core.index_store import get_store

    store = get_store()
    _refresh_store_for_search(store)
    out: list[ScoredPage] = []
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    for page_id, score in ranked:
        meta = store.meta(page_id)
        if meta is None or _normalize_lifecycle_status(meta.get("status")) != "stable":
            continue
        folder = _folder_from_meta(meta)
        out.append(
            ScoredPage(
                page_id=page_id,
                title=meta["title"],
                folder=folder,
                updated=meta["updated"],
                score=float(score),
                status=_normalize_lifecycle_status(meta.get("status")),
                superseded_by=meta.get("superseded_by", "")
                if isinstance(meta.get("superseded_by", ""), str)
                else "",
                page_type=_meta_page_type(meta, folder=folder),
                sensitivity=_meta_sensitivity(meta, folder=folder),
            )
        )
    return out


def apply_filters(
    results: list[ScoredPage],
    folder: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
) -> list[ScoredPage]:
    """Filter results by folder and date range."""
    for result in results:
        result.status = _normalize_lifecycle_status(result.status)
    filtered = [r for r in results if _is_stable_result(r)]
    if not folder:
        filtered = [r for r in filtered if not _is_reference_result(r)]
    if folder:
        filtered = [r for r in filtered if r.folder.startswith(folder)]
    if updated_after:
        filtered = [
            r for r in filtered if r.updated >= updated_after and r.updated != "unknown"
        ]
    if updated_before:
        filtered = [
            r
            for r in filtered
            if r.updated <= updated_before and r.updated != "unknown"
        ]
    return filtered


def apply_sort(
    results: list[ScoredPage], sort_by: str = "relevance"
) -> list[ScoredPage]:
    """Sort results."""
    if sort_by == "date":
        return sorted(results, key=lambda x: x.updated, reverse=True)
    elif sort_by == "title":
        return sorted(results, key=lambda x: x.title)
    return results  # relevance is already sorted


# ---------------------------------------------------------------------------
# Public search API
# ---------------------------------------------------------------------------

_SEARCH_TRACE = threading.local()


def last_search_trace() -> dict[str, object]:
    return dict(getattr(_SEARCH_TRACE, "value", {}))


def _pipeline_dependencies() -> PipelineDependencies:
    return PipelineDependencies(
        get_bm25=get_bm25,
        context_seed_results=context_seed_results,
        semantic_search=semantic_search,
        semantic_verify=semantic_verify,
        graph_expand_results=graph_expand_results,
        usage_prior_results=usage_prior_results,
        fuse_results=fuse_results,
        apply_filters=apply_filters,
        apply_sort=apply_sort,
        load_negative_feedback_config=load_negative_feedback_config,
        penalties_for_query=penalties_for_query,
        apply_penalties=apply_penalties,
    )


def search(
    query: str,
    top_n: int = 20,
    folder: str | None = None,
    updated_after: str | None = None,
    updated_before: str | None = None,
    sort_by: str = "relevance",
    semantic: bool = True,
    fusion_weights: dict[str, float] | None = None,
    semantic_timeout_ms: int | None = None,
    rollout_key: str = "",
) -> tuple[list[ScoredPage], str]:
    """Run search and return (results, search_mode)."""
    weights = (
        {**DEFAULT_FUSION_WEIGHTS, **fusion_weights}
        if fusion_weights is not None
        else load_active_fusion_weights()
    )
    search_embedding = load_search_embedding_config()
    if (
        fusion_weights is None
        and search_embedding.enabled
    ):
        from chronovisor.core.semantic_client import selected_for_rollout

        if selected_for_rollout(query, search_embedding):
            weights.update(
                {
                    "semantic": search_embedding.fusion_weight,
                    "semantic_min_top_score": search_embedding.min_top_score,
                    "semantic_min_margin": search_embedding.min_margin,
                    "semantic_low_confidence_weight": (
                        search_embedding.low_confidence_weight
                    ),
                }
            )
    _GRAPH_QUERY.value = query
    _GRAPH_ROLLOUT.value = rollout_key
    stage_timings_ms: dict[str, int] = {}
    _SEARCH_TRACE.value = {"stage_timings_ms": stage_timings_ms}
    try:
        result = run_search_pipeline(
            query,
            config=production_pipeline_config(
                top_n=top_n,
                folder=folder,
                updated_after=updated_after,
                updated_before=updated_before,
                sort_by=sort_by,
                semantic=semantic,
                fusion_weights=weights,
                include_reference=folder is not None,
                semantic_timeout_ms=semantic_timeout_ms,
            ),
            deps=_pipeline_dependencies(),
            stage_timings_ms=stage_timings_ms,
        )
    except BaseException:
        _SEARCH_TRACE.value = {"stage_timings_ms": dict(stage_timings_ms)}
        raise
    finally:
        _GRAPH_QUERY.value = ""
        _GRAPH_ROLLOUT.value = ""
    graph_ids = {page.page_id for page in result.graph_results}
    semantic_ids = {page.page_id for page in result.semantic_results}
    _SEARCH_TRACE.value = {
        "strategy": "multi_seed_associative",
        "budgets": {
            "anchor": 20,
            "lexical": max(top_n * 5, 100),
            "semantic": max(top_n * 5, 100),
            "context": 4,
            "graph_hops": 2,
            "graph_nodes": 50,
        },
        "channels": {
            "anchor": [page.page_id for page in result.anchor_results],
            "lexical": [page.page_id for page in result.bm25_results],
            "semantic": [page.page_id for page in result.semantic_results],
            "context": [page.page_id for page in result.context_results],
            "graph": [page.page_id for page in result.graph_results],
        },
        "verified_graph": sorted(graph_ids & semantic_ids),
        "paths": graph_expansion_trace(),
        "query_plan": getattr(_GRAPH_TRACE, "query_plan", "direct"),
        "stage_timings_ms": dict(result.stage_timings_ms or {}),
    }
    return result.results, result.search_mode
