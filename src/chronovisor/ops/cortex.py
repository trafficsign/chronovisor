"""Data and event transport for the Synaptic Cortex dashboard view."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from chronovisor.core.canonical_document import (
    Namespace,
    parse_document,
    resolve_internal_markdown_links,
)
from chronovisor.core.durable_state import DurableStateError, read_sealed_json
from chronovisor.core.index_store import (
    canonical_document_paths,
)
from chronovisor.core.knowledge_graph_store import KnowledgeGraphStore
from chronovisor.core.raw_segment import RawSegmentCommit, RawSegmentCorrupt
from chronovisor.ops.cortex_stream import (
    websocket_accept as websocket_accept,
)
from chronovisor.ops.cortex_stream import (
    websocket_text_frame as websocket_text_frame,
)

_GRAPH_CACHE_LOCK = threading.Lock()
_GRAPH_CACHE: dict[str, dict[str, Any]] = {}
_CORTEX_EVENT_SCHEMA = "chronovisor.cortex.event.v2"
_CORTEX_PAGE_ID_MAX_LENGTH = 240  # Keep aligned with CortexTransportPolicy.
_CORTEX_EVENT_BATCH_LIMIT = 32
_RELATION_DETAIL_LIMIT = 24
_RELATION_EVIDENCE_LIMIT = 12
_RELATION_VOTE_LIMIT = 8
_ENTITY_DETAIL_MEMBER_SCAN_LIMIT = 256
_RECALL_TRANSPORT_KINDS = {"recall", "auto_recall", "read", "search", "used"}
_INGEST_PAGE_RE = re.compile(
    r"(?:^|\s)ingest \| (?P<operation>created|updated) (?P<page>[^\r\n]+)$"
)
_INGEST_GENERATE_RE = re.compile(
    r"(?:^|\s)ingest \| generating \d+/\d+: (?P<page>[^\r\n]+)$"
)
_INGEST_STAGE_RE = re.compile(r"(?:^|\s)ingest \| stage 1: triage started$")
_INGEST_AUTH_RE = re.compile(
    r"(?:^|\s)ingest \| authorization: [^\r\n]+ -> apply_available$"
)
_INGEST_COMPLETE_RE = re.compile(r"(?:^|\s)ingest \| completed(?:\s|$)")
_ENTRYPOINT_PAGES = {
    "claude-code",
    "current-state",
    "lessons-learned",
    "user-profile",
}
_FIELD_SESSION_RE = re.compile(r"^[0-9a-f]{16}$")
_FIELD_EVENT_KEYS = {
    "seq",
    "timestamp_epoch",
    "session_hash",
    "topic_epoch",
    "kind",
    "page_id",
    "source_page_id",
    "target_page_id",
    "edge_type",
    "delta",
    "activation",
    "reason_code",
    "certificate_id",
    "components",
}
_FIELD_COMPONENT_KEYS = {
    "direct",
    "spread",
    "negative",
    "inhibition",
    "anti_index",
    "hub_penalty",
}


def _browser_text(value: Any, limit: int) -> str:
    return str(value or "")[:limit]


def _browser_nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _browser_confidence(value: Any) -> float:
    try:
        return round(max(0.0, min(1.0, float(value or 0.0))), 4)
    except (TypeError, ValueError):
        return 0.0


def _project_cortex_page_ids(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    projected: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        page_id = value[:_CORTEX_PAGE_ID_MAX_LENGTH]
        if not page_id or page_id in seen:
            continue
        seen.add(page_id)
        projected.append(page_id)
        if len(projected) >= 24:
            break
    return projected


@dataclass(frozen=True)
class _Page:
    path: Path
    page_id: str
    canonical_key: str
    category: str
    title: str
    updated: str
    tags: tuple[str, ...]
    line_count: int
    byte_count: int
    targets: tuple[str, ...]


def _page_sources(root: Path) -> list[tuple[Path, str]]:
    return [
        (path, "system" if root / "system" in path.parents else "pages")
        for path in canonical_document_paths(
            root / "pages",
            system_dir=root / "system",
            require_stable=True,
        )
    ]


def _source_fingerprint(
    root: Path,
    sources: list[tuple[Path, str]],
    *,
    commit: str,
) -> str:
    rows: list[tuple[str, int, int]] = []
    for path, _source_kind in sources:
        try:
            stat = path.stat()
        except OSError:
            continue
        rows.append(
            (
                str(path.relative_to(root)),
                stat.st_size,
                stat.st_mtime_ns,
            )
        )
    for relative in (
        "knowledge-graph/relation-snapshot.json",
        "knowledge-graph/community-snapshot.json",
        "runtime/typed-graph/status.json",
        "runtime/typed-graph/promotion.json",
        "runtime/recall-rubric/status.json",
    ):
        path = root / relative
        try:
            stat = path.stat()
        except OSError:
            continue
        rows.append((relative, stat.st_size, stat.st_mtime_ns))
    encoded = json.dumps(
        {"commit": commit, "sources": rows},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _string_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item).strip())
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()


def _read_page(path: Path, source_kind: str, root: Path) -> _Page | None:
    try:
        raw = path.read_bytes()
        content = raw.decode("utf-8")
        document = parse_document(raw)
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    metadata = document.metadata
    namespace: Namespace = "system" if source_kind == "system" else "pages"
    namespace_root = root / namespace
    relative_path = path.relative_to(namespace_root).as_posix()
    if source_kind == "system":
        category = "system"
    else:
        relative = path.relative_to(root / "pages")
        category = relative.parts[0] if len(relative.parts) > 1 else "root"
    page_id = path.stem
    title = str(metadata.get("title") or page_id)
    updated = str(metadata.get("updated") or "")
    return _Page(
        path=path,
        page_id=page_id,
        canonical_key=(
            f"{namespace}/{PurePosixPath(relative_path).with_suffix('').as_posix()}"
        ),
        category=category,
        title=title,
        updated=updated,
        tags=_string_list(metadata.get("tags")),
        line_count=max(1, content.count("\n") + 1),
        byte_count=len(content.encode("utf-8")),
        targets=tuple(
            f"{link.namespace}/{PurePosixPath(link.path).with_suffix('').as_posix()}"
            for link in resolve_internal_markdown_links(
                document.body,
                source_namespace=namespace,
                source_path=relative_path,
            )
        ),
    )


def _build_graph(
    root: Path,
    sources: list[tuple[Path, str]],
    *,
    commit: str,
    generated: str,
) -> dict[str, Any]:
    pages = [
        page
        for path, source_kind in sources
        if (page := _read_page(path, source_kind, root)) is not None
    ]
    pages.sort(key=lambda page: (page.category.casefold(), page.page_id.casefold()))

    index_by_key: dict[str, int] = {}
    ambiguous_keys: set[str] = set()
    for index, page in enumerate(pages):
        key = page.canonical_key.casefold()
        if key in index_by_key:
            ambiguous_keys.add(key)
        else:
            index_by_key[key] = index
    for key in ambiguous_keys:
        index_by_key.pop(key, None)
    page_key_by_id: dict[str, str] = {}
    ambiguous_page_ids: set[str] = set()
    for page in pages:
        page_id = page.page_id.casefold()
        if page_id in page_key_by_id:
            ambiguous_page_ids.add(page_id)
        else:
            page_key_by_id[page_id] = page.canonical_key.casefold()
    for page_id in ambiguous_page_ids:
        page_key_by_id.pop(page_id, None)
    page_id_by_key = {
        key: page.page_id
        for page in pages
        if (key := page.canonical_key.casefold()) in index_by_key
        and page_key_by_id.get(page.page_id.casefold()) == key
    }

    edges: list[list[int]] = []
    edge_keys: set[tuple[int, int]] = set()
    unresolved = 0
    fan_in = [0] * len(pages)
    fan_out = [0] * len(pages)
    for source_index, page in enumerate(pages):
        for target in page.targets:
            target_index = index_by_key.get(target.casefold())
            if target_index is None:
                unresolved += 1
                continue
            if target_index == source_index:
                continue
            edge_key = (source_index, target_index)
            if edge_key in edge_keys:
                continue
            edge_keys.add(edge_key)
            edges.append([source_index, target_index, 0])
            fan_out[source_index] += 1
            fan_in[target_index] += 1

    category_counts: dict[str, int] = {}
    for page in pages:
        category_counts[page.category] = category_counts.get(page.category, 0) + 1
    categories = [
        {"id": category, "count": count}
        for category, count in sorted(
            category_counts.items(),
            key=lambda item: (-item[1], item[0].casefold()),
        )
    ]
    nodes = [
        {
            "id": page.page_id,
            "pkg": page.category,
            "l": page.line_count,
            "b": page.byte_count,
            "fi": fan_in[index],
            "fo": fan_out[index],
            "ep": int(page.page_id in _ENTRYPOINT_PAGES),
            "title": page.title,
            "updated": page.updated,
            "tags": list(page.tags),
        }
        for index, page in enumerate(pages)
    ]
    typed_graph = _typed_graph_projection(
        root,
        index_by_key=index_by_key,
        page_key_by_id=page_key_by_id,
        page_id_by_key=page_id_by_key,
    )
    memberships = typed_graph.pop("memberships")
    for node in nodes:
        node["communities"] = memberships.get(node["id"], [])
    short_commit = commit[:7] if commit else "local"
    return {
        "meta": {
            "generated": generated,
            "commit": short_commit,
            "totalLines": sum(page.line_count for page in pages),
            "static": len(edges),
            "deferred": unresolved,
            "spawn": 0,
            "entrypoints": sum(node["ep"] for node in nodes),
            "source": "local-wiki",
        },
        "nodes": nodes,
        "links": edges,
        "categories": categories,
        "typedGraph": typed_graph,
    }


def _safe_sealed(path: Path) -> dict[str, Any]:
    try:
        value = read_sealed_json(path, recover_backup=True)
    except (DurableStateError, OSError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _graph_page_key(
    value: str,
    index_by_key: dict[str, int],
    page_key_by_id: Mapping[str, str],
) -> str | None:
    """Map legacy KG page paths to exact canonical Cortex node keys."""

    normalized = value.strip().removeprefix("/").removesuffix(".md").casefold()
    if normalized.startswith(("pages/", "system/")):
        page_id = PurePosixPath(normalized).name
        return (
            normalized
            if normalized in index_by_key and page_key_by_id.get(page_id) == normalized
            else None
        )
    return page_key_by_id.get(PurePosixPath(normalized).name)


def _typed_graph_projection(
    root: Path,
    *,
    index_by_key: dict[str, int],
    page_key_by_id: Mapping[str, str],
    page_id_by_key: Mapping[str, str],
) -> dict[str, Any]:
    """Return browser-safe relation topology without lazy detail payloads."""

    store = KnowledgeGraphStore(root / "knowledge-graph")
    try:
        records = store.relations()
    except (DurableStateError, OSError, TypeError, ValueError):
        records = []
    relations: list[dict[str, Any]] = []
    for record in records:
        source_key = _graph_page_key(
            record.source_page_id, index_by_key, page_key_by_id
        )
        target_key = _graph_page_key(
            record.target_page_id, index_by_key, page_key_by_id
        )
        source_index = index_by_key.get(source_key) if source_key else None
        target_index = index_by_key.get(target_key) if target_key else None
        if source_index is None or target_index is None:
            continue
        relation = {
            "relation_id": record.relation_id,
            "source": source_index,
            "target": target_index,
            "source_page_id": record.source_page_id,
            "target_page_id": record.target_page_id,
            "predicate": record.predicate[:128],
            "direction": record.direction,
            "status": record.status,
            "producer_role": record.producer_role,
            "confidence": round(record.confidence, 4),
            "used_count": record.used_count,
            "used_sessions": len(record.used_sessions),
            "reason_code": record.reason_code[:160],
            "detail_available": bool(record.evidence or record.consensus),
        }
        relations.append(relation)
    entity_payload = _safe_sealed(store.entity_snapshot_file)
    candidate_values = entity_payload.get("candidates")
    merge_values = entity_payload.get("merge_candidates")
    entity_candidates = candidate_values if isinstance(candidate_values, dict) else {}
    if isinstance(merge_values, dict):
        for merge_id, merge in sorted(merge_values.items()):
            if len(relations) >= 2_000 or not isinstance(merge, dict):
                break
            member_values = merge.get("member_candidate_ids")
            member_ids = member_values if isinstance(member_values, list) else []
            members = [
                entity_candidates.get(_browser_text(candidate_id, 256))
                for candidate_id in member_ids[:_ENTITY_DETAIL_MEMBER_SCAN_LIMIT]
            ]
            rows = [value for value in members if isinstance(value, dict)]
            page_ids = sorted(
                {_browser_text(value.get("page_id"), 240) for value in rows}
                - {""}
            )
            merge_consensus_value = merge.get("consensus")
            merge_consensus: dict[str, Any] = (
                merge_consensus_value if isinstance(merge_consensus_value, dict) else {}
            )
            relation_limit_reached = False
            for source_offset, source_page_id in enumerate(page_ids):
                for target_page_id in page_ids[source_offset + 1 :]:
                    if len(relations) >= 2_000:
                        relation_limit_reached = True
                        break
                    source_key = _graph_page_key(
                        source_page_id, index_by_key, page_key_by_id
                    )
                    target_key = _graph_page_key(
                        target_page_id, index_by_key, page_key_by_id
                    )
                    source_index = index_by_key.get(source_key) if source_key else None
                    target_index = index_by_key.get(target_key) if target_key else None
                    if source_index is None or target_index is None:
                        continue
                    relation = {
                        "relation_id": str(merge_id),
                        "source": source_index,
                        "target": target_index,
                        "source_page_id": source_page_id,
                        "target_page_id": target_page_id,
                        "predicate": "same_entity_alias",
                        "direction": "bidirectional",
                        "status": str(merge.get("status") or "proposed"),
                        "producer_role": "entity_local_consensus",
                        "confidence": 1.0,
                        "used_count": int(merge.get("used_count") or 0),
                        "used_sessions": len(merge.get("used_sessions") or []),
                        "reason_code": str(
                            merge.get("reason_code") or merge.get("reason") or ""
                        )[:160],
                        "detail_available": bool(rows or merge_consensus),
                    }
                    relations.append(relation)
                if relation_limit_reached:
                    break
    community_payload = _safe_sealed(store.community_snapshot_file)
    community_values = community_payload.get("communities")
    communities: list[dict[str, Any]] = []
    memberships: dict[str, list[str]] = {}
    if isinstance(community_values, dict):
        for community_id, value in sorted(community_values.items()):
            if not isinstance(value, dict):
                continue
            community_members = []
            for page_id in value.get("member_page_ids") or []:
                key = _graph_page_key(str(page_id), index_by_key, page_key_by_id)
                if key is not None and (node_id := page_id_by_key.get(key)) is not None:
                    community_members.append(node_id)
            for member_page_id in community_members:
                memberships.setdefault(member_page_id, []).append(str(community_id))
            communities.append(
                {
                    "community_id": str(community_id),
                    "member_page_ids": community_members,
                    "relation_ids": [
                        str(relation_id)
                        for relation_id in value.get("relation_ids") or []
                    ],
                    "source_digests": [
                        str(digest) for digest in value.get("source_digests") or []
                    ],
                    "summary_sha256": str(value.get("summary_sha256") or ""),
                    "generated_at": str(value.get("generated_at") or ""),
                }
            )
    status = _safe_sealed(root / "runtime" / "typed-graph" / "status.json")
    promotion = _safe_sealed(root / "runtime" / "typed-graph" / "promotion.json")
    rubric = _safe_sealed(root / "runtime" / "recall-rubric" / "status.json")
    return {
        "relations": relations,
        "communities": communities,
        "memberships": memberships,
        "status": {
            "mode": str(status.get("mode") or "shadow"),
            "engineering_complete": status.get("engineering_complete") is True,
            "engineering_gates": status.get("engineering_gates") or {},
            "authority_mature": status.get("authority_mature") is True,
            "relation_counts": status.get("relation_counts") or {},
            "builder": status.get("builder") or {},
            "consensus": status.get("consensus") or {},
            "entities": status.get("entities") or {},
            "community_summary": status.get("community_summary") or {},
            "evaluation": status.get("evaluation") or {},
            "four_arm": status.get("four_arm") or {},
            "rubric_gold": status.get("rubric_gold") or {},
            "authority": status.get("authority") or {},
            "external_model_calls": int(status.get("external_model_calls") or 0),
            "rollout": {
                "mode": str(promotion.get("mode") or "shadow"),
                "canary_percent": int(promotion.get("canary_percent") or 0),
                "reason": str(promotion.get("reason") or "not_evaluated")[:160],
                "gates": promotion.get("gates") or {},
                "sample_count": int(promotion.get("sample_count") or 0),
                "sample_unit": str(promotion.get("sample_unit") or ""),
            },
            "rubric": {
                "status": str(rubric.get("status") or "builtin"),
                "rubric_id": str(rubric.get("rubric_id") or "builtin-v1"),
                "gates": rubric.get("gates") or {},
                "samples": int(rubric.get("samples") or 0),
                "judge_metrics": rubric.get("judge_metrics") or {},
            },
        },
    }


def build_cortex_relation_details(
    root: Path,
    relation_keys: list[tuple[str, str, str]],
) -> list[dict[str, Any]]:
    """Return bounded details for stable, explicitly requested relation keys."""

    requested = {
        (
            str(relation_id)[:256],
            str(source_page_id)[:_CORTEX_PAGE_ID_MAX_LENGTH],
            str(target_page_id)[:_CORTEX_PAGE_ID_MAX_LENGTH],
        )
        for relation_id, source_page_id, target_page_id in relation_keys[
            :_RELATION_DETAIL_LIMIT
        ]
        if str(relation_id) and str(source_page_id) and str(target_page_id)
    }
    if not requested:
        return []
    resolved_root = root.expanduser().resolve()
    store = KnowledgeGraphStore(resolved_root / "knowledge-graph")
    relation_ids = {key[0] for key in requested}
    records: dict[str, dict[str, Any]] = {}
    try:
        snapshot = store.load_snapshot()
        values = snapshot.get("relations")
        if isinstance(values, dict):
            for relation_id_value in relation_ids:
                value = values.get(relation_id_value)
                if isinstance(value, dict):
                    records[relation_id_value] = value
    except (DurableStateError, OSError, TypeError, ValueError):
        records = {}

    details: dict[tuple[str, str, str], dict[str, Any]] = {}
    for key in requested:
        record = records.get(key[0])
        if record is None or key != (
            str(record.get("relation_id") or ""),
            str(record.get("source_page_id") or ""),
            str(record.get("target_page_id") or ""),
        ):
            continue
        evidence_value = record.get("evidence")
        evidence = evidence_value if isinstance(evidence_value, list) else []
        consensus_value = record.get("consensus")
        consensus = consensus_value if isinstance(consensus_value, dict) else None
        votes_value = consensus.get("votes") if consensus is not None else None
        votes = votes_value if isinstance(votes_value, list) else []
        details[key] = {
            "relation_id": key[0],
            "source_page_id": key[1],
            "target_page_id": key[2],
            "evidence_refs": [
                {
                    "page_id": _browser_text(row.get("page_id"), 240),
                    "content_sha256": _browser_text(
                        row.get("content_sha256"), 64
                    ),
                    "span_sha256": _browser_text(row.get("span_sha256"), 64),
                    "source_line": _browser_nonnegative_int(
                        row.get("source_line")
                    ),
                    "raw_sha256": _browser_text(row.get("raw_sha256"), 64),
                }
                for row in evidence[:_RELATION_EVIDENCE_LIMIT]
                if isinstance(row, dict)
            ],
            "consensus": (
                {
                    "receipt_id": _browser_text(consensus.get("receipt_id"), 256),
                    "producer_role": _browser_text(
                        consensus.get("producer_role"), 64
                    ),
                    "quorum": _browser_nonnegative_int(consensus.get("quorum")),
                    "outcome": _browser_text(consensus.get("outcome"), 32),
                    "hold_reason": _browser_text(
                        consensus.get("hold_reason"), 160
                    ),
                    "votes": [
                        {
                            "role": _browser_text(vote.get("role"), 64),
                            "model_sha256": _browser_text(
                                vote.get("model_sha256"), 64
                            ),
                            "decision": _browser_text(
                                vote.get("decision") or "abstain", 32
                            ),
                            "confidence": _browser_confidence(
                                vote.get("confidence")
                            ),
                            "vote_sha256": _browser_text(
                                vote.get("vote_sha256"), 64
                            ),
                        }
                        for vote in votes[:_RELATION_VOTE_LIMIT]
                        if isinstance(vote, dict)
                    ],
                }
                if consensus is not None
                else None
            ),
        }

    missing = requested - details.keys()
    if missing:
        entity_payload = _safe_sealed(store.entity_snapshot_file)
        candidate_values = entity_payload.get("candidates")
        merge_values = entity_payload.get("merge_candidates")
        candidates = candidate_values if isinstance(candidate_values, dict) else {}
        merges = merge_values if isinstance(merge_values, dict) else {}
        for key in missing:
            merge = merges.get(key[0])
            if not isinstance(merge, dict):
                continue
            member_values = merge.get("member_candidate_ids")
            member_ids = member_values if isinstance(member_values, list) else []
            requested_pages = {key[1], key[2]}
            if len(requested_pages) != 2:
                continue
            rows: list[dict[str, Any]] = []
            found_pages: set[str] = set()
            for candidate_id in member_ids[:_ENTITY_DETAIL_MEMBER_SCAN_LIMIT]:
                value = candidates.get(_browser_text(candidate_id, 256))
                if not isinstance(value, dict):
                    continue
                page_id = _browser_text(value.get("page_id"), 240)
                if page_id not in requested_pages:
                    continue
                found_pages.add(page_id)
                if len(rows) < _RELATION_EVIDENCE_LIMIT:
                    rows.append(value)
                if found_pages == requested_pages:
                    break
            if found_pages != requested_pages:
                continue
            consensus_value = merge.get("consensus")
            consensus = consensus_value if isinstance(consensus_value, dict) else {}
            votes_value = consensus.get("votes")
            votes = votes_value if isinstance(votes_value, list) else []
            details[key] = {
                "relation_id": key[0],
                "source_page_id": key[1],
                "target_page_id": key[2],
                "evidence_refs": [
                    {
                        "page_id": _browser_text(value.get("page_id"), 240),
                        "content_sha256": _browser_text(
                            value.get("content_sha256"), 64
                        ),
                        "span_sha256": _browser_text(
                            value.get("alias_evidence_sha256"), 64
                        ),
                        "source_line": 0,
                        "raw_sha256": "",
                    }
                    for value in rows
                ],
                "consensus": {
                    "receipt_id": _browser_text(
                        consensus.get("receipt_id")
                        or merge.get("receipt_id")
                        or "",
                        256,
                    ),
                    "producer_role": _browser_text(
                        consensus.get("producer_role")
                        or "entity_local_consensus",
                        64,
                    ),
                    "quorum": _browser_nonnegative_int(
                        consensus.get("quorum") or 2
                    ),
                    "outcome": _browser_text(
                        consensus.get("outcome")
                        or merge.get("status")
                        or "proposed",
                        32,
                    ),
                    "hold_reason": _browser_text(
                        consensus.get("hold_reason")
                        or merge.get("reason_code")
                        or "",
                        160,
                    ),
                    "votes": [
                        {
                            "role": _browser_text(vote.get("role"), 64),
                            "model_sha256": _browser_text(
                                vote.get("model_sha256"), 64
                            ),
                            "decision": _browser_text(
                                vote.get("decision") or "abstain", 32
                            ),
                            "confidence": _browser_confidence(
                                vote.get("confidence")
                            ),
                            "vote_sha256": _browser_text(
                                vote.get("vote_sha256"), 64
                            ),
                        }
                        for vote in votes[:_RELATION_VOTE_LIMIT]
                        if isinstance(vote, dict)
                    ],
                },
            }

    return [details[key] for key in sorted(details)[:_RELATION_DETAIL_LIMIT]]


def build_cortex_graph(
    root: Path,
    *,
    commit: str = "",
    generated: str | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Build the browser-safe Wiki graph without exposing page bodies."""

    resolved_root = root.expanduser().resolve()
    sources = _page_sources(resolved_root)
    fingerprint = _source_fingerprint(resolved_root, sources, commit=commit)
    cache_key = str(resolved_root)
    if use_cache:
        with _GRAPH_CACHE_LOCK:
            cached = _GRAPH_CACHE.get(cache_key)
            if cached and cached.get("fingerprint") == fingerprint:
                cached_graph = cached.get("graph")
                if isinstance(cached_graph, dict):
                    return cached_graph

    graph = _build_graph(
        resolved_root,
        sources,
        commit=commit,
        generated=generated
        or datetime.now().astimezone().isoformat(timespec="seconds"),
    )
    if use_cache:
        with _GRAPH_CACHE_LOCK:
            _GRAPH_CACHE[cache_key] = {
                "fingerprint": fingerprint,
                "graph": graph,
            }
    return graph


def _read_sealed_field_snapshot(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("field snapshot must be an object")
    seal = value.get("snapshot_sha256")
    payload = {key: item for key, item in value.items() if key != "snapshot_sha256"}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if not isinstance(seal, str) or seal != hashlib.sha256(encoded).hexdigest():
        raise ValueError("field snapshot seal mismatch")
    return payload


def _project_field_event(value: Any) -> dict[str, Any] | None:
    """Return the strict browser-safe subset of one durable Field event."""

    if not isinstance(value, dict):
        return None
    session = value.get("session_hash")
    seq = value.get("seq")
    kind = value.get("kind")
    if (
        not isinstance(session, str)
        or not _FIELD_SESSION_RE.fullmatch(session)
        or not isinstance(seq, int)
        or seq < 1
        or not isinstance(kind, str)
    ):
        return None
    projected: dict[str, Any] = {
        key: value.get(key) for key in _FIELD_EVENT_KEYS if key in value
    }
    components = value.get("components")
    safe_components = components if isinstance(components, dict) else {}
    projected_components: dict[str, Any] = {
        key: round(float(safe_components.get(key) or 0.0), 6)
        for key in sorted(_FIELD_COMPONENT_KEYS)
        if isinstance(safe_components.get(key, 0.0), int | float)
    }
    relation_id = safe_components.get("relation_id")
    if isinstance(relation_id, str) and relation_id.startswith("rel_"):
        projected_components["relation_id"] = relation_id[:128]
    projected["components"] = projected_components
    projected["source"] = "stateful-recall-field"
    return projected


def _read_field_events(
    event_root: Path,
    session_hash: str,
    *,
    limit: int = 256,
) -> list[dict[str, Any]]:
    if not _FIELD_SESSION_RE.fullmatch(session_hash):
        return []
    path = event_root / f"{session_hash}.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in lines[-max(1, limit) :]:
        try:
            projected = _project_field_event(json.loads(line))
        except json.JSONDecodeError:
            continue
        if projected is not None and projected["session_hash"] == session_hash:
            events.append(projected)
    return sorted(events, key=lambda row: int(row["seq"]))


def _field_recall_metrics(
    root: Path,
    session_hash: str,
    *,
    limit: int = 400,
) -> dict[str, Any]:
    """Aggregate Field latency and teacher agreement without exposing prompts."""

    path = root / "recall" / "recall-log.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    latencies: list[float] = []
    teacher_total = 0
    teacher_agreed = 0
    for line in lines[-max(1, limit) :]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        features = row.get("evidence_features")
        field = features.get("field_shadow") if isinstance(features, dict) else None
        if not isinstance(field, dict) or field.get("session_hash") != session_hash:
            continue
        latency = field.get("latency_ms")
        if isinstance(latency, int | float) and latency >= 0:
            latencies.append(float(latency))
        candidates = {
            str(page_id)
            for page_id in field.get("candidate_page_ids") or []
            if isinstance(page_id, str)
        }
        pages = {
            str(page_id)
            for page_id in row.get("pages") or []
            if isinstance(page_id, str)
        }
        if pages:
            teacher_total += len(pages)
            teacher_agreed += len(pages & candidates)
    latencies.sort()

    def percentile(fraction: float) -> float | None:
        if not latencies:
            return None
        index = min(len(latencies) - 1, round((len(latencies) - 1) * fraction))
        return round(latencies[index], 1)

    return {
        "samples": len(latencies),
        "latency_ms": {
            "p50": percentile(0.5),
            "p95": percentile(0.95),
            "max": round(max(latencies), 1) if latencies else None,
        },
        "teacher_agreement": (
            round(teacher_agreed / teacher_total, 4) if teacher_total else None
        ),
        "teacher_pages": teacher_total,
    }


def _safe_metric_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _field_growth_summary(root: Path) -> dict[str, Any]:
    path = root / "runtime" / "recall-field" / "growth-state.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    labels = metrics.get("labels") if isinstance(metrics, dict) else None
    thresholds = payload.get("thresholds") if isinstance(payload, dict) else None
    candidate = metrics.get("candidate") if isinstance(metrics, dict) else None
    processor_used = (
        metrics.get("processor_used") if isinstance(metrics, dict) else None
    )
    return {
        "stage": str(payload.get("stage") or "not_started"),
        "field_learning_allowed": payload.get("field_learning_allowed") is True,
        "positive_learning_allowed": (
            payload.get("positive_learning_allowed") is True
            if "positive_learning_allowed" in payload
            else payload.get("field_learning_allowed") is True
        ),
        "policy_update_allowed": payload.get("policy_update_allowed") is True,
        "authority_enabled": payload.get("authority_enabled") is True,
        "canary_percent": _safe_metric_int(payload.get("canary_percent")),
        "strong_positive": _safe_metric_int(
            labels.get("strong_positive") or 0 if isinstance(labels, dict) else 0
        ),
        "strong_positive_target": _safe_metric_int(
            thresholds.get("strong_positive") or 200
            if isinstance(thresholds, dict)
            else 200
        ),
        "strong_sessions": _safe_metric_int(
            labels.get("strong_positive_sessions") or 0
            if isinstance(labels, dict)
            else 0
        ),
        "strong_sessions_target": _safe_metric_int(
            thresholds.get("strong_positive_sessions") or 20
            if isinstance(thresholds, dict)
            else 20
        ),
        "candidate_traces": _safe_metric_int(
            candidate.get("traces") or 0 if isinstance(candidate, dict) else 0
        ),
        "processor_used_episodes": _safe_metric_int(
            processor_used.get("episodes") or 0
            if isinstance(processor_used, dict)
            else 0
        ),
    }


def build_cortex_field_projection(
    root: Path,
    *,
    session_hash: str = "",
    now: float | None = None,
    event_limit: int = 256,
) -> dict[str, Any]:
    """Build a browser-safe projection of recent Stateful Recall Field state."""

    from chronovisor.recall.recall_field_schema import load_recall_field_config
    from chronovisor.recall.recall_field_store import RecallFieldStore

    observed = time.time() if now is None else now
    field_root = root.expanduser().resolve() / "recall" / "field"
    config = load_recall_field_config()
    field_store = RecallFieldStore(root=field_root, config=config)
    session_root = field_store.session_root
    growth = _field_growth_summary(root)
    effective_mode = config.mode
    if config.auto_promote and config.mode not in {"off", "shadow"}:
        effective_mode = "active" if growth["authority_enabled"] else "candidate"
    sessions: list[tuple[dict[str, Any], dict[str, Any]]] = []
    corrupt_snapshots = 0
    try:
        paths = sorted(
            session_root.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        paths = []
    for path in paths[:24]:
        if not _FIELD_SESSION_RE.fullmatch(path.stem):
            continue
        try:
            payload = _read_sealed_field_snapshot(path)
        except (OSError, ValueError, json.JSONDecodeError):
            corrupt_snapshots += 1
            continue
        if payload.get("session_hash") != path.stem:
            corrupt_snapshots += 1
            continue
        mode = effective_mode
        buffer_name = "shadow" if mode == "shadow" else "active"
        buffer = payload.get(buffer_name)
        if not isinstance(buffer, dict):
            buffer = {}
        updated = float(payload.get("updated_at_epoch") or 0.0)
        sessions.append(
            (
                {
                    "session_hash": path.stem,
                    "host": str(payload.get("host") or ""),
                    "updated_at_epoch": updated,
                    "topic_epoch": int(payload.get("topic_epoch") or 0),
                    "turn": int(payload.get("turn") or 0),
                    "seq": int(payload.get("seq") or 0),
                    "mode": mode,
                    "nodes": len(buffer),
                },
                payload,
            )
        )
    requested = session_hash if _FIELD_SESSION_RE.fullmatch(session_hash) else ""
    selected = next(
        (row for row in sessions if row[0]["session_hash"] == requested),
        sessions[0] if sessions else None,
    )
    if selected is None:
        return {
            "status": "fault" if corrupt_snapshots else "offline",
            "source": "stateful-recall-field",
            "mode": effective_mode,
            "session_hash": "",
            "sessions": [],
            "snapshot": None,
            "events": [],
            "summary": {
                "active": 0,
                "candidate": 0,
                "commit": 0,
                "reject": 0,
                "teacher_agreement": None,
                "latency_ms": {"p50": None, "p95": None, "max": None},
                "stale": True,
                "corrupt_snapshots": corrupt_snapshots,
                "growth": growth,
            },
        }

    session, payload = selected
    mode = session["mode"]
    buffer_name = "shadow" if mode == "shadow" else "active"
    raw_buffer = payload.get(buffer_name)
    buffer = raw_buffer if isinstance(raw_buffer, dict) else {}
    nodes: list[dict[str, Any]] = []
    for page_id, value in buffer.items():
        if not isinstance(page_id, str) or not isinstance(value, dict):
            continue
        activation = value.get("activation")
        if not isinstance(activation, int | float):
            continue
        components = {
            key: round(float(value.get(key) or 0.0), 6) for key in _FIELD_COMPONENT_KEYS
        }
        nodes.append(
            {
                "page_id": page_id,
                "activation": round(float(activation), 6),
                "components": components,
                "last_seq": int(value.get("last_seq") or 0),
            }
        )
    nodes.sort(key=lambda row: (-row["activation"], row["page_id"]))
    nodes = nodes[: config.max_active_nodes]
    events = _read_field_events(
        field_store.event_root,
        session["session_hash"],
        limit=event_limit,
    )
    counts: dict[str, int] = {}
    for event in events:
        kind = str(event.get("kind") or "")
        counts[kind] = counts.get(kind, 0) + 1
    metrics = _field_recall_metrics(root, session["session_hash"])
    stale_after_seconds = max(
        60,
        min(600, config.wall_half_life_seconds * 2),
    )
    age_seconds = max(0.0, observed - session["updated_at_epoch"])
    stale = age_seconds > stale_after_seconds
    status = "fault" if corrupt_snapshots else ("stale" if stale else "online")
    return {
        "status": status,
        "source": "stateful-recall-field",
        "mode": mode,
        "session_hash": session["session_hash"],
        "sessions": [row[0] for row in sessions[:12]],
        "snapshot": {
            "session_hash": session["session_hash"],
            "host": session["host"],
            "topic_epoch": session["topic_epoch"],
            "turn": session["turn"],
            "seq": session["seq"],
            "updated_at_epoch": session["updated_at_epoch"],
            "full_search_fallback": payload.get("full_search_fallback") is not False,
            "nodes": nodes,
        },
        "events": events,
        "summary": {
            "active": sum(node["activation"] >= 0.05 for node in nodes),
            "candidate": min(len(nodes), config.working_set_size),
            "commit": counts.get("commit_queued", 0) + counts.get("commit_applied", 0),
            "reject": counts.get("reject", 0) + counts.get("inhibit", 0),
            "teacher_agreement": metrics["teacher_agreement"],
            "latency_ms": metrics["latency_ms"],
            "stale": stale,
            "age_seconds": round(age_seconds, 1),
            "corrupt_snapshots": corrupt_snapshots,
            "growth": growth,
        },
    }


class CortexEventCursor:
    """Tail durable Chronovisor telemetry and expose browser-safe firing events."""

    def __init__(
        self,
        root: Path,
        *,
        recall_log: Path | None = None,
        pull_log: Path | None = None,
        activity_log: Path | None = None,
        field_session: str = "",
        follow_field_sessions: bool = False,
        after_seq: int = 0,
        field_batch_limit: int = 32,
        event_batch_limit: int = _CORTEX_EVENT_BATCH_LIMIT,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.recall_log = recall_log or self.root / "recall" / "recall-log.jsonl"
        self.pull_log = pull_log or self.root / "recall" / "pull-log.jsonl"
        self.activity_log = activity_log or self.root / "log.md"
        self.raw_dir = self.root / "raw"
        self.field_session = (
            field_session if _FIELD_SESSION_RE.fullmatch(field_session) else ""
        )
        self.follow_field_sessions = bool(follow_field_sessions)
        from chronovisor.recall.recall_field_store import RecallFieldStore

        self.field_store = RecallFieldStore(root=self.root / "recall" / "field")
        self.field_event_root = self.field_store.event_root
        self.field_event_log = None
        self._field_after_seq = max(0, int(after_seq))
        self._field_batch_limit = max(1, min(32, int(field_batch_limit)))
        self._event_batch_limit = max(
            1,
            min(_CORTEX_EVENT_BATCH_LIMIT, int(event_batch_limit)),
        )
        self._pending_events: deque[dict[str, Any]] = deque()
        self._field_control: dict[str, Any] | None = None
        if self.follow_field_sessions and not self.field_session:
            self.field_session = self._latest_followed_field_session()
        self._offsets = {
            self.recall_log: self._file_size(self.recall_log),
            self.pull_log: self._file_size(self.pull_log),
            self.activity_log: self._file_size(self.activity_log),
        }
        self._remainders: dict[Path, bytes] = {}
        self._raw_commit_files: dict[Path, tuple[int, int]] = {}
        self._known_raw_commit_files: set[tuple[int, int]] = set()
        self._raw_commit_rebaselines: set[Path] = set()
        self._seen_raw_commit_ids: set[str] = set()
        for path in self._raw_commit_paths():
            identity = self._file_identity(path)
            if identity is None:
                continue
            self._offsets[path] = self._file_size(path)
            self._raw_commit_files[path] = identity
            self._known_raw_commit_files.add(identity)
        self._raw_snapshot = self._raw_file_snapshot()

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

    @staticmethod
    def _file_identity(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        return stat.st_dev, stat.st_ino

    def _tail_lines(self, path: Path) -> list[str]:
        size = self._file_size(path)
        offset = self._offsets.get(path, 0)
        if size < offset:
            offset = 0
            self._remainders.pop(path, None)
        if size == offset:
            return []
        try:
            with path.open("rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
        except OSError:
            return []
        self._offsets[path] = offset + len(chunk)
        data = self._remainders.pop(path, b"") + chunk
        if data and not data.endswith(b"\n"):
            data, remainder = data.rsplit(b"\n", 1) if b"\n" in data else (b"", data)
            self._remainders[path] = remainder
        return data.decode("utf-8", errors="replace").splitlines()

    def _raw_file_snapshot(self) -> dict[str, tuple[int, int]]:
        snapshot: dict[str, tuple[int, int]] = {}
        try:
            paths = self.raw_dir.glob("*.md")
        except OSError:
            return snapshot
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[path.name] = (stat.st_size, stat.st_mtime_ns)
        return snapshot

    def _raw_commit_paths(self) -> list[Path]:
        pattern = "[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]/*.commits.jsonl"
        try:
            return sorted(self.raw_dir.glob(pattern))
        except OSError:
            return []

    def _raw_segment_commits(self) -> list[RawSegmentCommit]:
        commits: list[RawSegmentCommit] = []
        for path in self._raw_commit_paths():
            identity = self._file_identity(path)
            if identity is None:
                continue
            size = self._file_size(path)
            previous_identity = self._raw_commit_files.get(path)
            if previous_identity is None:
                self._offsets[path] = (
                    size if identity in self._known_raw_commit_files else 0
                )
                self._raw_commit_files[path] = identity
                self._known_raw_commit_files.add(identity)
            elif previous_identity != identity:
                self._offsets[path] = size
                self._remainders.pop(path, None)
                self._raw_commit_files[path] = identity
                self._known_raw_commit_files.add(identity)
                self._raw_commit_rebaselines.discard(path)
                continue
            elif size < self._offsets.get(path, 0):
                # A same-inode truncation may be followed by historical journal
                # restoration. Baseline both the shrink and its first regrowth.
                self._offsets[path] = size
                self._remainders.pop(path, None)
                self._raw_commit_rebaselines.add(path)
                continue
            elif path in self._raw_commit_rebaselines:
                if size > self._offsets.get(path, 0):
                    self._offsets[path] = size
                    self._remainders.pop(path, None)
                    self._raw_commit_rebaselines.discard(path)
                continue
            for line in self._tail_lines(path):
                try:
                    commit = RawSegmentCommit.from_dict(json.loads(line))
                except (
                    json.JSONDecodeError,
                    RawSegmentCorrupt,
                    TypeError,
                    ValueError,
                ):
                    continue
                if commit.raw_id in self._seen_raw_commit_ids:
                    continue
                self._seen_raw_commit_ids.add(commit.raw_id)
                commits.append(commit)
        return commits

    @staticmethod
    def _event(
        kind: str,
        page_ids: list[Any],
        label: str,
        *,
        origin: str,
        **details: str | int,
    ) -> dict[str, Any]:
        phase_value = details.get("phase")
        if isinstance(phase_value, str) and phase_value:
            phase = phase_value
        elif kind in {"save", "capture"}:
            phase = "capture"
        elif kind == "search":
            phase = "triage"
        elif kind == "used":
            phase = "apply"
        else:
            phase = "generate"
        if kind in {"save", "capture"}:
            lane_key = "raw_buffer"
        elif kind == "ingest":
            lane_key = "ingest"
        elif kind in _RECALL_TRANSPORT_KINDS:
            lane_key = "recall"
        else:
            lane_key = "audit"
        priority_class = (
            "protected"
            if kind in {"save", "capture"} | _RECALL_TRANSPORT_KINDS
            else "standard"
        )
        event: dict[str, Any] = {
            "schema": _CORTEX_EVENT_SCHEMA,
            "family": "transport",
            "origin": origin,
            "mode": "live",
            "kind": kind,
            "page_ids": _project_cortex_page_ids(page_ids),
            "label": label[:160],
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": "telemetry-fallback",
            "presentation": {
                "lane_key": lane_key,
                "phase": phase,
                "channel_key": kind,
                "priority_class": priority_class,
            },
        }
        for key in (
            "phase",
            "operation",
            "file_name",
            "raw_id",
            "capture_id",
            "byte_count",
            "raw_count",
        ):
            value = details.get(key)
            if isinstance(value, str):
                event[key] = value[:160]
            elif isinstance(value, int):
                event[key] = max(0, value)
        return event

    def _field_events(self) -> list[dict[str, Any]]:
        if not self.field_session:
            return []
        rows = self.field_store.read_events(
            self.field_session,
            after_seq=self._field_after_seq,
        )
        snapshot_status, committed_seq = self._field_commit_watermark(
            self.field_session
        )
        if snapshot_status == "corrupt":
            self._field_control = {
                "type": "resync",
                "session_hash": self.field_session,
                "after_seq": self._field_after_seq,
                "committed_seq": 0,
                "reason": "field_snapshot_corrupt",
            }
            return []
        if snapshot_status == "missing":
            # Browser-safe test fixtures and pre-v2 sessions may have only an
            # event journal. Production sessions are gated by their snapshot.
            committed_seq = max(
                (int(row.get("seq") or 0) for row in rows if isinstance(row, dict)),
                default=self._field_after_seq,
            )
        elif self._field_after_seq > committed_seq:
            self._field_control = {
                "type": "resync",
                "session_hash": self.field_session,
                "after_seq": self._field_after_seq,
                "committed_seq": committed_seq,
                "reason": "field_watermark_ahead",
            }
            return []
        events: list[dict[str, Any]] = []
        expected = self._field_after_seq + 1
        for row in rows:
            event = _project_field_event(row)
            if event is None or event["session_hash"] != self.field_session:
                continue
            sequence = int(event["seq"])
            if sequence > committed_seq:
                continue
            if sequence != expected:
                self._field_control = {
                    "type": "resync",
                    "session_hash": self.field_session,
                    "after_seq": self._field_after_seq,
                    "committed_seq": committed_seq,
                    "reason": "field_sequence_gap",
                }
                return []
            events.append(event)
            expected += 1
            if len(events) >= self._field_batch_limit:
                break
        if not events and committed_seq > self._field_after_seq:
            self._field_control = {
                "type": "resync",
                "session_hash": self.field_session,
                "after_seq": self._field_after_seq,
                "committed_seq": committed_seq,
                "reason": "field_retention_gap",
            }
            return []
        if events:
            self._field_after_seq = int(events[-1]["seq"])
        return events

    def _field_commit_watermark(self, session_hash: str) -> tuple[str, int]:
        snapshot_path = self.field_store.session_root / f"{session_hash}.json"
        try:
            snapshot = _read_sealed_field_snapshot(snapshot_path)
            return "valid", max(0, int(snapshot.get("seq") or 0))
        except FileNotFoundError:
            return "missing", 0
        except (OSError, ValueError, json.JSONDecodeError):
            return "corrupt", 0

    def _field_event_paths(self) -> list[Path]:
        try:
            return sorted(self.field_event_root.glob("*.jsonl"))
        except OSError:
            return []

    def _latest_followed_field_session(self) -> str:
        latest = self.field_store.latest_session_hash(max_age_seconds=float("inf"))
        if latest:
            return latest
        paths = self._field_event_paths()
        try:
            return max(paths, key=lambda path: path.stat().st_mtime_ns).stem
        except (OSError, ValueError):
            return ""

    def _followed_field_events(self) -> list[dict[str, Any]]:
        latest = self._latest_followed_field_session()
        if latest and latest != self.field_session:
            previous = self.field_session
            self.field_session = latest
            snapshot_status, committed_seq = self._field_commit_watermark(latest)
            if snapshot_status == "corrupt":
                self._field_control = {
                    "type": "resync",
                    "session_hash": latest,
                    "previous_session_hash": previous,
                    "after_seq": self._field_after_seq,
                    "committed_seq": 0,
                    "reason": "field_snapshot_corrupt",
                }
                return []
            if snapshot_status == "missing":
                committed_seq = max(
                    (
                        int(row.get("seq") or 0)
                        for row in self.field_store.read_events(latest)
                        if isinstance(row, dict)
                    ),
                    default=0,
                )
            self._field_after_seq = committed_seq
            self._field_control = {
                "type": "session_changed",
                "session_hash": latest,
                "previous_session_hash": previous,
                "committed_seq": self._field_after_seq,
            }
            return []
        return self._field_events()

    def _automatic_recall_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in self._tail_lines(self.recall_log):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            page_ids = [
                str(page_id)
                for page_id in row.get("pages") or []
                if isinstance(page_id, str) and page_id
            ]
            if (
                row.get("stage") != "injected"
                or row.get("status") != "ok"
                or row.get("decision") != "read"
                or not page_ids
            ):
                continue
            events.append(
                self._event(
                    "auto_recall",
                    page_ids,
                    f"AUTO RECALL · {len(page_ids)} page"
                    f"{'' if len(page_ids) == 1 else 's'}",
                    origin="recall-log",
                )
            )
        return events

    def _pull_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in self._tail_lines(self.pull_log):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_type = row.get("type")
            if event_type == "read" and row.get("page_id"):
                events.append(
                    self._event(
                        "read",
                        [row["page_id"]],
                        "MCP READ",
                        origin="pull-log",
                    )
                )
            elif event_type == "search":
                page_ids = [
                    page_id for page_id in row.get("direct_pages") or [] if page_id
                ]
                if page_ids:
                    events.append(
                        self._event(
                            "search", page_ids, "MCP SEARCH", origin="pull-log"
                        )
                    )
            elif event_type == "used":
                page_ids = [
                    page_id for page_id in row.get("page_ids") or [] if page_id
                ]
                if page_ids:
                    events.append(
                        self._event(
                            "used", page_ids, "RECALL USED", origin="pull-log"
                        )
                    )
        return events

    def _save_events(self) -> list[dict[str, Any]]:
        commits = self._raw_segment_commits()
        snapshot = self._raw_file_snapshot()
        changed = [
            name
            for name, identity in snapshot.items()
            if self._raw_snapshot.get(name) != identity
        ]
        self._raw_snapshot = snapshot
        if not commits and not changed:
            return []
        changed.sort(key=lambda name: (snapshot[name][1], name), reverse=True)
        byte_count = sum(commit.length for commit in commits) + sum(
            snapshot[name][0] for name in changed
        )
        identities = [
            f"raw:{commit.raw_id}:{commit.sha256}" for commit in commits
        ] + [
            f"legacy:{name}:{snapshot[name][0]}:{snapshot[name][1]}"
            for name in changed
        ]
        capture_id = hashlib.sha256(
            "\n".join(sorted(identities)).encode("utf-8")
        ).hexdigest()[:12]
        details: dict[str, str | int] = {
            "phase": "capture",
            "capture_id": capture_id,
            "byte_count": byte_count,
            "raw_count": len(commits) + len(changed),
        }
        if commits:
            newest_commit = max(
                commits,
                key=lambda commit: (commit.captured_at, commit.raw_id),
            )
            details["raw_id"] = newest_commit.raw_id
        if changed:
            details["file_name"] = changed[0]
        if commits and changed:
            origin = "raw-journal+raw-snapshot"
        elif commits:
            origin = "raw-journal"
        else:
            origin = "raw-snapshot"
        return [
            self._event(
                "save",
                [],
                f"CAPTURED · {byte_count} B · ID {capture_id}",
                origin=origin,
                **details,
            )
        ]

    def _ingest_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for line in self._tail_lines(self.activity_log):
            match = _INGEST_PAGE_RE.search(line)
            if match:
                page_id = Path(match.group("page").strip()).stem
                operation = match.group("operation")
                events.append(
                    self._event(
                        "ingest",
                        [page_id],
                        f"MEMORY {operation.upper()} · {page_id}",
                        origin="activity-log",
                        phase="apply",
                        operation=operation,
                    )
                )
                continue
            match = _INGEST_GENERATE_RE.search(line)
            if match:
                page_id = Path(match.group("page").strip()).stem
                events.append(
                    self._event(
                        "ingest",
                        [page_id],
                        f"INGEST GENERATE · {page_id}",
                        origin="activity-log",
                        phase="generate",
                    )
                )
            elif _INGEST_STAGE_RE.search(line):
                events.append(
                    self._event(
                        "ingest",
                        [],
                        "INGEST TRIAGE · raw inspection",
                        origin="activity-log",
                        phase="triage",
                    )
                )
            elif _INGEST_AUTH_RE.search(line):
                events.append(
                    self._event(
                        "ingest",
                        [],
                        "INGEST CONSENSUS · apply available",
                        origin="activity-log",
                        phase="consensus",
                    )
                )
            elif _INGEST_COMPLETE_RE.search(line):
                events.append(
                    self._event(
                        "ingest",
                        [],
                        "INGEST COMPLETE · memory consolidated",
                        origin="activity-log",
                        phase="complete",
                    )
                )
        return events

    def poll(self) -> list[dict[str, Any]]:
        payload = self.poll_payload()
        return payload.get("events", []) if payload.get("type") == "events" else []

    def poll_payload(self) -> dict[str, Any]:
        """Return one bounded event envelope or an explicit resync control."""

        self._field_control = None
        if self._pending_events:
            return {
                "type": "events",
                "events": [
                    self._pending_events.popleft()
                    for _ in range(
                        min(self._event_batch_limit, len(self._pending_events))
                    )
                ],
            }
        field_events: list[dict[str, Any]] = []
        if self.follow_field_sessions:
            field_events = self._followed_field_events()
        elif self.field_session:
            field_events = self._field_events()
        if self._field_control is not None:
            return self._field_control
        self._pending_events.extend(
            [
            *field_events,
            *self._automatic_recall_events(),
            *self._pull_events(),
            *self._save_events(),
            *self._ingest_events(),
            ]
        )
        return {
            "type": "events",
            "events": [
                self._pending_events.popleft()
                for _ in range(
                    min(self._event_batch_limit, len(self._pending_events))
                )
            ],
        }
