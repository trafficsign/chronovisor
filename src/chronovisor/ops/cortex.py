"""Data and event transport for the Synaptic Cortex dashboard view."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from chronovisor.core.durable_state import DurableStateError, read_sealed_json
from chronovisor.core.frontmatter import parse as parse_frontmatter
from chronovisor.core.link_fix import extract_targets
from chronovisor.knowledge_graph.store import KnowledgeGraphStore

_GRAPH_CACHE_LOCK = threading.Lock()
_GRAPH_CACHE: dict[str, dict[str, Any]] = {}
_WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
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


@dataclass(frozen=True)
class _Page:
    path: Path
    page_id: str
    category: str
    title: str
    updated: str
    tags: tuple[str, ...]
    line_count: int
    byte_count: int
    targets: tuple[str, ...]


def _page_sources(root: Path) -> list[tuple[Path, str]]:
    sources: list[tuple[Path, str]] = []
    pages_dir = root / "pages"
    system_dir = root / "system"
    if pages_dir.exists():
        sources.extend((path, "pages") for path in pages_dir.rglob("*.md"))
    if system_dir.exists():
        sources.extend((path, "system") for path in system_dir.glob("*.md"))
    return sorted(sources, key=lambda item: str(item[0]))


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
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    metadata, _body = parse_frontmatter(content)
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
        category=category,
        title=title,
        updated=updated,
        tags=_string_list(metadata.get("tags")),
        line_count=max(1, content.count("\n") + 1),
        byte_count=len(content.encode("utf-8")),
        targets=tuple(extract_targets(content, strip=True)),
    )


def _target_key(value: str) -> str:
    normalized = value.strip().removesuffix(".md")
    return normalized.rsplit("/", 1)[-1].casefold()


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
        key = page.page_id.casefold()
        if key in index_by_key:
            ambiguous_keys.add(key)
        else:
            index_by_key[key] = index
    for key in ambiguous_keys:
        index_by_key.pop(key, None)

    edges: list[list[int]] = []
    edge_keys: set[tuple[int, int]] = set()
    unresolved = 0
    fan_in = [0] * len(pages)
    fan_out = [0] * len(pages)
    for source_index, page in enumerate(pages):
        for target in page.targets:
            target_index = index_by_key.get(_target_key(target))
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
    typed_graph = _typed_graph_projection(root, index_by_key=index_by_key)
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


def _typed_graph_projection(
    root: Path,
    *,
    index_by_key: dict[str, int],
) -> dict[str, Any]:
    """Return IDs, digests, votes and state only; never expose evidence text."""

    store = KnowledgeGraphStore(root / "knowledge-graph")
    try:
        records = store.relations()
    except (DurableStateError, OSError, TypeError, ValueError):
        records = []
    relations: list[dict[str, Any]] = []
    for record in records:
        source_index = index_by_key.get(_target_key(record.source_page_id))
        target_index = index_by_key.get(_target_key(record.target_page_id))
        if source_index is None or target_index is None:
            continue
        consensus = record.consensus
        relations.append(
            {
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
                "evidence_refs": [
                    {
                        "page_id": row.page_id,
                        "content_sha256": row.content_sha256,
                        "span_sha256": row.span_sha256,
                        "source_line": row.source_line,
                        "raw_sha256": row.raw_sha256,
                    }
                    for row in record.evidence
                ],
                "consensus": (
                    {
                        "receipt_id": consensus.receipt_id,
                        "producer_role": consensus.producer_role,
                        "quorum": consensus.quorum,
                        "outcome": consensus.outcome,
                        "hold_reason": consensus.hold_reason[:160],
                        "votes": [
                            {
                                "role": vote.role,
                                "model_sha256": vote.model_sha256,
                                "decision": vote.decision,
                                "confidence": round(vote.confidence, 4),
                                "vote_sha256": vote.vote_sha256,
                            }
                            for vote in consensus.votes
                        ],
                    }
                    if consensus is not None
                    else None
                ),
            }
        )
    entity_payload = _safe_sealed(store.entity_snapshot_file)
    candidate_values = entity_payload.get("candidates")
    merge_values = entity_payload.get("merge_candidates")
    entity_candidates = candidate_values if isinstance(candidate_values, dict) else {}
    if isinstance(merge_values, dict):
        for merge_id, merge in sorted(merge_values.items()):
            if len(relations) >= 2_000 or not isinstance(merge, dict):
                break
            members = [
                entity_candidates.get(str(candidate_id))
                for candidate_id in merge.get("member_candidate_ids") or []
            ]
            rows = [value for value in members if isinstance(value, dict)]
            page_ids = sorted(
                {str(value.get("page_id") or "") for value in rows} - {""}
            )
            evidence_refs = [
                {
                    "page_id": str(value.get("page_id") or ""),
                    "content_sha256": str(value.get("content_sha256") or ""),
                    "span_sha256": str(value.get("alias_evidence_sha256") or ""),
                    "source_line": 0,
                    "raw_sha256": "",
                }
                for value in rows
            ]
            merge_consensus_value = merge.get("consensus")
            merge_consensus: dict[str, Any] = (
                merge_consensus_value if isinstance(merge_consensus_value, dict) else {}
            )
            merge_votes_value = merge_consensus.get("votes")
            merge_votes = (
                merge_votes_value if isinstance(merge_votes_value, list) else []
            )
            for source_offset, source_page_id in enumerate(page_ids):
                for target_page_id in page_ids[source_offset + 1 :]:
                    source_index = index_by_key.get(_target_key(source_page_id))
                    target_index = index_by_key.get(_target_key(target_page_id))
                    if source_index is None or target_index is None:
                        continue
                    relations.append(
                        {
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
                            "evidence_refs": evidence_refs,
                            "consensus": {
                                "receipt_id": str(
                                    merge_consensus.get("receipt_id")
                                    or merge.get("receipt_id")
                                    or ""
                                ),
                                "producer_role": str(
                                    merge_consensus.get("producer_role")
                                    or "entity_local_consensus"
                                ),
                                "quorum": int(merge_consensus.get("quorum") or 2),
                                "outcome": str(
                                    merge_consensus.get("outcome")
                                    or merge.get("status")
                                    or "proposed"
                                ),
                                "hold_reason": str(
                                    merge_consensus.get("hold_reason")
                                    or merge.get("reason_code")
                                    or ""
                                )[:160],
                                "votes": [
                                    {
                                        "role": str(vote.get("role") or ""),
                                        "model_sha256": str(
                                            vote.get("model_sha256") or ""
                                        ),
                                        "decision": str(
                                            vote.get("decision") or "abstain"
                                        ),
                                        "confidence": round(
                                            float(vote.get("confidence") or 0.0), 4
                                        ),
                                        "vote_sha256": str(
                                            vote.get("vote_sha256") or ""
                                        ),
                                    }
                                    for vote in merge_votes
                                    if isinstance(vote, dict)
                                ],
                            },
                        }
                    )
    community_payload = _safe_sealed(store.community_snapshot_file)
    community_values = community_payload.get("communities")
    communities: list[dict[str, Any]] = []
    memberships: dict[str, list[str]] = {}
    if isinstance(community_values, dict):
        for community_id, value in sorted(community_values.items()):
            if not isinstance(value, dict):
                continue
            community_members: list[str] = [
                _target_key(str(page_id))
                for page_id in value.get("member_page_ids") or []
                if _target_key(str(page_id)) in index_by_key
            ]
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


def websocket_accept(key: str) -> str:
    """Return the RFC 6455 accept token for a validated browser key."""

    try:
        raw = base64.b64decode(key, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid Sec-WebSocket-Key") from exc
    if len(raw) != 16:
        raise ValueError("invalid Sec-WebSocket-Key")
    digest = hashlib.sha1(f"{key}{_WEBSOCKET_GUID}".encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def websocket_text_frame(payload: dict[str, Any]) -> bytes:
    """Encode one unmasked server-to-browser JSON text frame."""

    body = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    size = len(body)
    if size < 126:
        header = bytes((0x81, size))
    elif size <= 0xFFFF:
        header = bytes((0x81, 126)) + size.to_bytes(2, "big")
    else:
        header = bytes((0x81, 127)) + size.to_bytes(8, "big")
    return header + body


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

        self.field_event_root = RecallFieldStore(
            root=self.root / "recall" / "field"
        ).event_root
        self.field_event_log = (
            self.field_event_root / f"{self.field_session}.jsonl"
            if self.field_session
            else None
        )
        self._offsets = {
            self.recall_log: self._file_size(self.recall_log),
            self.pull_log: self._file_size(self.pull_log),
            self.activity_log: self._file_size(self.activity_log),
        }
        if self.field_event_log is not None:
            self._offsets[self.field_event_log] = self._file_size(self.field_event_log)
        if self.follow_field_sessions:
            for path in self._field_event_paths():
                self._offsets[path] = self._file_size(path)
        self._remainders: dict[Path, bytes] = {}
        self._raw_snapshot = self._raw_file_snapshot()

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except OSError:
            return 0

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

    @staticmethod
    def _event(
        kind: str,
        page_ids: list[str],
        label: str,
        **details: str | int,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "kind": kind,
            "page_ids": list(dict.fromkeys(page_ids))[:24],
            "label": label[:160],
            "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": "telemetry-fallback",
        }
        for key in (
            "phase",
            "operation",
            "file_name",
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
        if self.field_event_log is None:
            return []
        events: list[dict[str, Any]] = []
        for line in self._tail_lines(self.field_event_log):
            try:
                event = _project_field_event(json.loads(line))
            except json.JSONDecodeError:
                continue
            if event is None or event["session_hash"] != self.field_session:
                continue
            events.append(event)
        return events

    def _field_event_paths(self) -> list[Path]:
        try:
            return sorted(self.field_event_root.glob("*.jsonl"))
        except OSError:
            return []

    def _followed_field_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for path in self._field_event_paths():
            if not _FIELD_SESSION_RE.fullmatch(path.stem):
                continue
            if path not in self._offsets:
                self._offsets[path] = 0
            for line in self._tail_lines(path):
                try:
                    event = _project_field_event(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if event is None or event["session_hash"] != path.stem:
                    continue
                events.append(event)
        return sorted(
            events,
            key=lambda event: (
                float(event.get("timestamp_epoch") or 0.0),
                str(event.get("session_hash") or ""),
                int(event.get("seq") or 0),
            ),
        )

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
                events.append(self._event("read", [str(row["page_id"])], "MCP READ"))
            elif event_type == "search":
                page_ids = [
                    str(page_id) for page_id in row.get("direct_pages") or [] if page_id
                ]
                if page_ids:
                    events.append(self._event("search", page_ids, "MCP SEARCH"))
            elif event_type == "used":
                page_ids = [
                    str(page_id) for page_id in row.get("page_ids") or [] if page_id
                ]
                if page_ids:
                    events.append(self._event("used", page_ids, "RECALL USED"))
        return events

    def _save_events(self) -> list[dict[str, Any]]:
        snapshot = self._raw_file_snapshot()
        if snapshot == self._raw_snapshot:
            return []
        changed = [
            name
            for name, identity in snapshot.items()
            if self._raw_snapshot.get(name) != identity
        ]
        self._raw_snapshot = snapshot
        if not changed:
            return []
        changed.sort(key=lambda name: snapshot[name][1], reverse=True)
        newest = changed[0]
        byte_count = sum(snapshot[name][0] for name in changed)
        identity = f"{newest}:{snapshot[newest][0]}:{snapshot[newest][1]}"
        capture_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        return [
            self._event(
                "save",
                [],
                f"CAPTURED · {byte_count} B · ID {capture_id}",
                phase="capture",
                file_name=newest,
                capture_id=capture_id,
                byte_count=byte_count,
                raw_count=len(changed),
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
                        phase="generate",
                    )
                )
            elif _INGEST_STAGE_RE.search(line):
                events.append(
                    self._event(
                        "ingest",
                        [],
                        "INGEST TRIAGE · raw inspection",
                        phase="triage",
                    )
                )
            elif _INGEST_AUTH_RE.search(line):
                events.append(
                    self._event(
                        "ingest",
                        [],
                        "INGEST CONSENSUS · apply available",
                        phase="consensus",
                    )
                )
            elif _INGEST_COMPLETE_RE.search(line):
                events.append(
                    self._event(
                        "ingest",
                        [],
                        "INGEST COMPLETE · memory consolidated",
                        phase="complete",
                    )
                )
        return events

    def poll(self) -> list[dict[str, Any]]:
        if self.follow_field_sessions:
            return [
                *self._followed_field_events(),
                *self._save_events(),
                *self._ingest_events(),
            ]
        if self.field_session:
            return self._field_events()
        return [
            *self._automatic_recall_events(),
            *self._pull_events(),
            *self._save_events(),
            *self._ingest_events(),
        ]
