from __future__ import annotations

import json
import math
import os
import subprocess
import threading
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest

from chronovisor.core.durable_state import write_sealed_json
from chronovisor.core.knowledge_graph_schema import (
    EvidenceRef,
    RelationRecord,
    relation_id,
    sha256,
)
from chronovisor.core.knowledge_graph_store import KnowledgeGraphStore
from chronovisor.core.raw_segment import RawSegmentReceipt, append_capture
from chronovisor.ops import cortex, dashboard
from chronovisor.recall.recall_field_schema import (
    ActivationNode,
    FieldEvent,
    RecallFieldConfig,
)
from chronovisor.recall.recall_field_store import RecallFieldStore


def _write_page(
    path: Path,
    *,
    title: str,
    body: str,
    tags: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tag_line = f"tags: [{', '.join(tags or [])}]\n" if tags else ""
    path.write_text(
        f"---\ntitle: {title}\nstatus: stable\ntype: knowledge\nupdated: 2026-07-29\n{tag_line}---\n{body}\n",
        encoding="utf-8",
    )


def _append_v2_capture(root: Path, *, after_line: int) -> RawSegmentReceipt:
    session_key = "0123456789abcdef01234567"
    until_line = after_line + 1
    idempotency_key = (
        f"codex-{session_key}-from{after_line}-to{until_line}"
    )
    with patch(
        "chronovisor.core.store.okf_runtime_operation",
        lambda *_args, **_kwargs: nullcontext(),
    ):
        return append_capture(
            raw_dir=root / "raw",
            raw_id=f"save-{idempotency_key}.md",
            idempotency_key=idempotency_key,
            host="codex",
            session_key=session_key,
            session_id="cortex-test-session",
            source_file=root / "session.jsonl",
            after_line=after_line,
            until_line=until_line,
            source_bytes=f'{{"line":{until_line}}}\n'.encode(),
            record_count=1,
        )


def test_build_cortex_graph_uses_local_wiki_without_exposing_bodies(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    _write_page(
        root / "pages" / "alpha" / "page-a.md",
        title="Alpha <private>",
        body="Links to [page-b](<../beta/page-b.md>) and [missing-page](<missing-page.md>).",
        tags=["d/example"],
    )
    _write_page(
        root / "pages" / "beta" / "page-b.md",
        title="Beta",
        body="Back to [page-a](<../alpha/page-a.md>).",
    )
    _write_page(
        root / "system" / "current-state.md",
        title="Current State",
        body="System state.",
    )

    graph = cortex.build_cortex_graph(
        root,
        commit="0123456789abcdef",
        generated="2026-07-29T22:00:00+09:00",
        use_cache=False,
    )

    assert graph["meta"] == {
        "generated": "2026-07-29T22:00:00+09:00",
        "commit": "0123456",
        "totalLines": 25,
        "static": 2,
        "deferred": 1,
        "spawn": 0,
        "entrypoints": 1,
        "source": "local-wiki",
    }
    assert [row["id"] for row in graph["nodes"]] == [
        "page-a",
        "page-b",
        "current-state",
    ]
    assert graph["categories"] == [
        {"id": "alpha", "count": 1},
        {"id": "beta", "count": 1},
        {"id": "system", "count": 1},
    ]
    page_a = graph["nodes"][0]
    page_b = graph["nodes"][1]
    current_state = graph["nodes"][2]
    assert page_a["title"] == "Alpha <private>"
    assert page_a["tags"] == ["d/example"]
    assert page_a["fi"] == 1
    assert page_a["fo"] == 1
    assert page_b["fi"] == 1
    assert page_b["fo"] == 1
    assert current_state["ep"] == 1
    assert graph["links"] == [[0, 1, 0], [1, 0, 0]]
    assert all("content" not in node and "body" not in node for node in graph["nodes"])


def test_build_cortex_graph_skips_descendant_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "chronovisor"
    _write_page(
        root / "pages" / "safe.md",
        title="Safe",
        body="Safe body.",
    )
    outside = tmp_path / "outside.md"
    _write_page(outside, title="Outside secret", body="Outside secret body.")
    inside = root / "pages" / "inside.private"
    _write_page(inside, title="Inside secret", body="Inside secret body.")
    (root / "pages" / "outside-link.md").symlink_to(outside)
    (root / "pages" / "inside-link.md").symlink_to(inside)

    graph = cortex.build_cortex_graph(root, use_cache=False)
    encoded = json.dumps(graph)

    assert [node["id"] for node in graph["nodes"]] == ["safe"]
    assert "Outside secret" not in encoded
    assert "Inside secret" not in encoded


def test_build_cortex_graph_excludes_reserved_documents_by_namespace(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    _write_page(
        root / "pages" / "concept-index.md",
        title="Concept index",
        body="A normal concept.",
    )
    for relative in (
        "pages/index.md",
        "pages/log.md",
        "pages/schema.md",
        "pages/nested/index.md",
        "pages/nested/log.md",
        "pages/nested/schema.md",
        "system/index.md",
        "system/log.md",
    ):
        _write_page(root / relative, title="Reserved", body="Reserved body.")
    _write_page(
        root / "system" / "schema.md",
        title="System schema",
        body="Privileged system schema.",
    )

    graph = cortex.build_cortex_graph(root, use_cache=False)

    assert [node["id"] for node in graph["nodes"]] == ["concept-index", "schema"]
    assert graph["categories"] == [
        {"id": "root", "count": 1},
        {"id": "system", "count": 1},
    ]


def test_build_cortex_graph_skips_symlinked_namespace_roots(tmp_path: Path) -> None:
    root = tmp_path / "chronovisor"
    root.mkdir()
    outside_pages = tmp_path / "outside-pages"
    _write_page(
        outside_pages / "outside.md",
        title="Outside root secret",
        body="Outside root body.",
    )
    inside_system = root / "private-system"
    _write_page(
        inside_system / "inside.md",
        title="Inside root secret",
        body="Inside root body.",
    )
    (root / "pages").symlink_to(outside_pages, target_is_directory=True)
    (root / "system").symlink_to(inside_system, target_is_directory=True)

    graph = cortex.build_cortex_graph(root, use_cache=False)
    encoded = json.dumps(graph)

    assert graph["nodes"] == []
    assert "Outside root secret" not in encoded
    assert "Inside root secret" not in encoded


def test_build_cortex_graph_cache_invalidates_when_a_page_changes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    page = root / "pages" / "page-a.md"
    _write_page(page, title="Alpha", body="First.")

    first = cortex.build_cortex_graph(root, use_cache=True)
    _write_page(page, title="Alpha", body="First.\nSecond.")
    stat = page.stat()
    os.utime(page, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    second = cortex.build_cortex_graph(root, use_cache=True)

    assert first is not second
    assert second["nodes"][0]["l"] > first["nodes"][0]["l"]


def test_cortex_projects_entity_consensus_votes_without_mentions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    _write_page(root / "pages" / "a.md", title="A", body="Alpha")
    _write_page(root / "pages" / "b.md", title="B", body="Beta")
    _write_page(root / "pages" / "c.md", title="C", body="Gamma")
    store = KnowledgeGraphStore(root / "knowledge-graph")
    store.write_derived_snapshot(
        "entities",
        {
            "schema_version": 1,
            "candidates": {
                "one": {
                    "candidate_id": "one",
                    "mention": "secret mention one",
                    "page_id": "a",
                    "content_sha256": "a" * 64,
                    "alias_evidence_sha256": "b" * 64,
                },
                "two": {
                    "candidate_id": "two",
                    "mention": "secret mention two",
                    "page_id": "b",
                    "content_sha256": "c" * 64,
                    "alias_evidence_sha256": "d" * 64,
                },
                "three": {
                    "candidate_id": "three",
                    "mention": "secret mention three",
                    "page_id": "c",
                    "content_sha256": "1" * 64,
                    "alias_evidence_sha256": "2" * 64,
                },
                **{
                    f"extra-{index}": {
                        "candidate_id": f"extra-{index}",
                        "mention": f"secret mention extra {index}",
                        "page_id": "a" if index % 2 == 0 else "b",
                        "content_sha256": f"{index + 10:064x}",
                        "alias_evidence_sha256": f"{index + 100:064x}",
                    }
                    for index in range(20)
                },
            },
            "merge_candidates": {
                "merge_one": {
                    "member_candidate_ids": [
                        "one",
                        "two",
                        "three",
                        *(f"extra-{index}" for index in range(20)),
                    ],
                    "status": "verified",
                    "receipt_id": "receipt-one",
                    "consensus": {
                        "receipt_id": "receipt-one",
                        "producer_role": "tie_break",
                        "quorum": 2,
                        "outcome": "verified",
                        "hold_reason": "",
                        "votes": [
                            {
                                "role": "primary",
                                "model_sha256": "e" * 64,
                                "decision": "approve",
                                "confidence": 0.91,
                                "vote_sha256": "f" * 64,
                            },
                            *(
                                {
                                    "role": f"extra-{index}",
                                    "model_sha256": f"{index + 200:064x}",
                                    "decision": "approve",
                                    "confidence": 0.8,
                                    "vote_sha256": f"{index + 300:064x}",
                                }
                                for index in range(20)
                            ),
                        ],
                    },
                },
            },
        },
    )
    write_sealed_json(
        root / "runtime" / "typed-graph" / "status.json",
        {"engineering_complete": True, "authority_mature": False},
    )

    graph = cortex.build_cortex_graph(root, use_cache=False)
    relation = graph["typedGraph"]["relations"][0]

    assert relation["relation_id"] == "merge_one"
    assert "consensus" not in relation
    assert "evidence_refs" not in relation
    details = cortex.build_cortex_relation_details(
        root,
        [
            (
                relation["relation_id"],
                relation["source_page_id"],
                relation["target_page_id"],
            )
        ],
    )
    assert details[0]["consensus"]["votes"][0]["role"] == "primary"
    assert details[0]["consensus"]["votes"][0]["decision"] == "approve"
    assert len(details[0]["evidence_refs"]) == 2
    assert len(details[0]["consensus"]["votes"]) == 8
    shared_relation = next(
        row
        for row in graph["typedGraph"]["relations"]
        if row["source_page_id"] == "b" and row["target_page_id"] == "c"
    )
    shared_details = cortex.build_cortex_relation_details(
        root,
        [(shared_relation["relation_id"], "b", "c")],
    )
    assert shared_details[0]["source_page_id"] == "b"
    assert shared_details[0]["target_page_id"] == "c"
    assert "secret mention" not in json.dumps(graph, ensure_ascii=False)
    assert "secret mention" not in json.dumps(details, ensure_ascii=False)


def test_cortex_entity_projection_only_emits_detail_resolvable_member_pairs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    for page_id in ("a", "b", "c"):
        _write_page(
            root / "pages" / f"{page_id}.md",
            title=page_id.upper(),
            body=page_id,
        )
    member_ids = ["candidate-a", "candidate-b"] + [
        f"candidate-extra-{index}" for index in range(254)
    ] + ["candidate-c"]
    candidates = {
        "candidate-a": {
            "page_id": "a",
            "content_sha256": "a" * 64,
            "alias_evidence_sha256": "1" * 64,
        },
        "candidate-b": {
            "page_id": "b",
            "content_sha256": "b" * 64,
            "alias_evidence_sha256": "2" * 64,
        },
        "candidate-c": {
            "page_id": "c",
            "content_sha256": "c" * 64,
            "alias_evidence_sha256": "3" * 64,
        },
        **{
            f"candidate-extra-{index}": {
                "page_id": "a",
                "content_sha256": f"{index + 10:064x}",
                "alias_evidence_sha256": f"{index + 300:064x}",
            }
            for index in range(254)
        },
    }
    store = KnowledgeGraphStore(root / "knowledge-graph")
    store.write_derived_snapshot(
        "entities",
        {
            "schema_version": 1,
            "candidates": candidates,
            "merge_candidates": {
                "merge-bounded": {
                    "member_candidate_ids": member_ids,
                    "status": "verified",
                    "consensus": {"outcome": "verified", "votes": []},
                }
            },
        },
    )

    graph = cortex.build_cortex_graph(root, use_cache=False)
    relations = [
        row
        for row in graph["typedGraph"]["relations"]
        if row["relation_id"] == "merge-bounded"
    ]

    assert [
        (row["source_page_id"], row["target_page_id"])
        for row in relations
    ] == [("a", "b")]
    details = cortex.build_cortex_relation_details(
        root,
        [
            (
                row["relation_id"],
                row["source_page_id"],
                row["target_page_id"],
            )
            for row in relations
        ],
    )
    assert [(row["source_page_id"], row["target_page_id"]) for row in details] == [
        ("a", "b")
    ]


def test_cortex_entity_projection_caps_pairs_within_one_large_merge(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    page_ids = [f"page-{index:03}" for index in range(256)]
    for page_id in page_ids:
        _write_page(
            root / "pages" / f"{page_id}.md",
            title=page_id,
            body=page_id,
        )
    candidate_ids = [f"candidate-{index:03}" for index in range(256)]
    KnowledgeGraphStore(root / "knowledge-graph").write_derived_snapshot(
        "entities",
        {
            "schema_version": 1,
            "candidates": {
                candidate_id: {
                    "page_id": page_id,
                    "content_sha256": f"{index:064x}",
                    "alias_evidence_sha256": f"{index + 256:064x}",
                }
                for index, (candidate_id, page_id) in enumerate(
                    zip(candidate_ids, page_ids, strict=True)
                )
            },
            "merge_candidates": {
                "merge-256": {
                    "member_candidate_ids": candidate_ids,
                    "status": "verified",
                }
            },
        },
    )

    graph = cortex.build_cortex_graph(root, use_cache=False)
    relations = graph["typedGraph"]["relations"]

    assert len(relations) == 2_000
    assert {row["relation_id"] for row in relations} == {"merge-256"}


def test_cortex_projects_typed_relations_with_nested_page_ids(tmp_path: Path) -> None:
    root = tmp_path / "chronovisor"
    _write_page(root / "pages" / "topic" / "a.md", title="A", body="Alpha")
    _write_page(root / "pages" / "b.md", title="B", body="Beta")
    evidence = EvidenceRef(
        page_id="topic/a",
        content_sha256="a" * 64,
        span_sha256="b" * 64,
        source_line=3,
    )
    record = RelationRecord(
        relation_id=relation_id(
            source_page_id="topic/a",
            target_page_id="b",
            predicate="references",
            evidence_sha256=sha256([evidence.__dict__]),
            model_sha256="c" * 64,
            rubric_sha256="d" * 64,
        ),
        source_page_id="topic/a",
        target_page_id="b",
        predicate="references",
        direction="forward",
        status="verified",
        evidence=(evidence,),
        model_sha256="c" * 64,
        rubric_sha256="d" * 64,
        producer_role="local_consensus",
        confidence=0.9,
    )
    KnowledgeGraphStore(root / "knowledge-graph").append(record, action="propose")

    graph = cortex.build_cortex_graph(root, use_cache=False)

    relation = graph["typedGraph"]["relations"][0]
    assert graph["nodes"][relation["source"]]["id"] == "a"
    assert graph["nodes"][relation["target"]]["id"] == "b"
    assert relation["source_page_id"] == "topic/a"


def test_cortex_community_projection_uses_stem_ids_for_nodes_and_hulls(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    _write_page(root / "pages" / "topic" / "a.md", title="A", body="Alpha")
    _write_page(root / "pages" / "b.md", title="B", body="Beta")
    KnowledgeGraphStore(root / "knowledge-graph").write_derived_snapshot(
        "communities",
        {
            "communities": {
                "community-one": {
                    "member_page_ids": ["pages/topic/a", "pages/b"],
                    "relation_ids": [],
                    "source_digests": [],
                }
            }
        },
    )

    graph = cortex.build_cortex_graph(root, use_cache=False)

    community = graph["typedGraph"]["communities"][0]
    assert community["member_page_ids"] == ["a", "b"]
    nodes = {node["id"]: node for node in graph["nodes"]}
    assert nodes["a"]["communities"] == ["community-one"]
    assert nodes["b"]["communities"] == ["community-one"]
    assert all(page_id in nodes for page_id in community["member_page_ids"])


def test_cortex_relation_detail_lookup_is_bounded_and_sorted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evidence = tuple(
        EvidenceRef(
            page_id="a",
            content_sha256=f"{index:064x}",
            span_sha256=f"{index + 100:064x}",
            source_line=index + 1,
        )
        for index in range(20)
    )
    relations = []
    for index in range(30):
        predicate = f"bounded_{index:02}"
        relations.append(
            RelationRecord(
                relation_id=relation_id(
                    source_page_id="a",
                    target_page_id="b",
                    predicate=predicate,
                    evidence_sha256=sha256([row.__dict__ for row in evidence]),
                    model_sha256="c" * 64,
                    rubric_sha256="d" * 64,
                ),
                source_page_id="a",
                target_page_id="b",
                predicate=predicate,
                direction="forward",
                status="verified",
                evidence=evidence,
                model_sha256="c" * 64,
                rubric_sha256="d" * 64,
                producer_role="local_consensus",
                confidence=0.9,
            )
        )
    snapshot_relations = {
        relation.relation_id: json.loads(json.dumps(relation.to_dict()))
        for relation in relations
    }
    snapshot_relations[relations[-1].relation_id]["evidence"][12] = {
        "invalid_beyond_browser_limit": True
    }
    # Invalid unrequested rows prove the detail path does not instantiate the
    # full relation snapshot before filtering stable requested IDs.
    snapshot_relations.update(
        {f"unrequested-{index}": {"invalid": True} for index in range(2_111)}
    )
    monkeypatch.setattr(
        KnowledgeGraphStore,
        "load_snapshot",
        lambda _store: {"relations": snapshot_relations},
    )
    keys = [
        (relation.relation_id, relation.source_page_id, relation.target_page_id)
        for relation in reversed(relations)
    ]

    details = cortex.build_cortex_relation_details(tmp_path, keys)

    assert len(details) == 24
    assert [row["relation_id"] for row in details] == sorted(
        relation.relation_id for relation in relations[6:]
    )
    assert all(len(row["evidence_refs"]) == 12 for row in details)
    assert len(json.dumps(details, separators=(",", ":"))) < 120_000


def test_cortex_entity_relation_detail_bounds_member_work_and_strings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class CountingCandidates(dict[str, dict[str, object]]):
        def __init__(self) -> None:
            super().__init__(
                {
                    "candidate-source": {
                        "page_id": "source",
                        "content_sha256": "c" * 10_000,
                        "alias_evidence_sha256": "a" * 10_000,
                    },
                    "candidate-target": {
                        "page_id": "target",
                        "content_sha256": "d" * 10_000,
                        "alias_evidence_sha256": "b" * 10_000,
                    },
                }
            )
            self.lookups = 0

        def get(self, key: str, default=None):  # type: ignore[no-untyped-def]
            self.lookups += 1
            return super().get(key, default)

    candidates = CountingCandidates()
    long_value = "x" * 10_000
    votes = [
        {
            "role": long_value,
            "model_sha256": long_value,
            "decision": long_value,
            "confidence": 4.2,
            "vote_sha256": long_value,
        }
        for _index in range(20)
    ]
    entity_payload = {
        "candidates": candidates,
        "merge_candidates": {
            "merge-long": {
                "member_candidate_ids": [
                    "candidate-source",
                    "candidate-target",
                    *(f"unrequested-{index}" for index in range(10_000)),
                ],
                "status": long_value,
                "receipt_id": long_value,
                "reason_code": long_value,
                "consensus": {
                    "receipt_id": long_value,
                    "producer_role": long_value,
                    "quorum": -3,
                    "outcome": long_value,
                    "hold_reason": long_value,
                    "votes": votes,
                },
            }
        },
    }
    monkeypatch.setattr(
        KnowledgeGraphStore,
        "load_snapshot",
        lambda _store: {"relations": {}},
    )
    monkeypatch.setattr(cortex, "_safe_sealed", lambda _path: entity_payload)

    details = cortex.build_cortex_relation_details(
        tmp_path,
        [("merge-long", "source", "target")],
    )

    assert candidates.lookups == 2
    assert len(details) == 1
    detail = details[0]
    assert len(detail["evidence_refs"]) == 2
    assert all(len(row["page_id"]) <= 240 for row in detail["evidence_refs"])
    assert all(len(row["content_sha256"]) == 64 for row in detail["evidence_refs"])
    assert all(len(row["span_sha256"]) == 64 for row in detail["evidence_refs"])
    consensus = detail["consensus"]
    assert len(consensus["receipt_id"]) == 256
    assert len(consensus["producer_role"]) == 64
    assert consensus["quorum"] == 0
    assert len(consensus["outcome"]) == 32
    assert len(consensus["hold_reason"]) == 160
    assert len(consensus["votes"]) == 8
    assert all(len(vote["role"]) == 64 for vote in consensus["votes"])
    assert all(len(vote["model_sha256"]) == 64 for vote in consensus["votes"])
    assert all(len(vote["decision"]) == 32 for vote in consensus["votes"])
    assert all(vote["confidence"] == 1.0 for vote in consensus["votes"])
    assert all(len(vote["vote_sha256"]) == 64 for vote in consensus["votes"])


def test_websocket_helpers_follow_rfc6455() -> None:
    assert (
        cortex.websocket_accept("dGhlIHNhbXBsZSBub25jZQ==")
        == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
    )

    frame = cortex.websocket_text_frame(
        {"type": "events", "events": [{"kind": "recall"}]}
    )

    assert frame[0] == 0x81
    assert frame[1] == len(frame) - 2
    assert json.loads(frame[2:]) == {
        "type": "events",
        "events": [{"kind": "recall"}],
    }


def test_cortex_event_cursor_maps_durable_activity_to_firing_events(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    recall_log = root / "recall" / "recall-log.jsonl"
    pull_log = root / "recall" / "pull-log.jsonl"
    activity_log = root / "log.md"
    raw_dir = root / "raw"
    pull_log.parent.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    recall_log.write_text("", encoding="utf-8")
    pull_log.write_text("", encoding="utf-8")
    activity_log.write_text("", encoding="utf-8")
    cursor = cortex.CortexEventCursor(
        root,
        recall_log=recall_log,
        pull_log=pull_log,
        activity_log=activity_log,
    )
    assert cursor.poll() == []
    raw_mtime = raw_dir.stat().st_mtime_ns

    recall_log.write_text(
        json.dumps(
            {
                "event": "UserPromptSubmit",
                "stage": "injected",
                "status": "ok",
                "decision": "read",
                "pages": ["current-state", "page-a"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pull_log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "read",
                        "stage": "read",
                        "page_id": "current-state",
                    }
                ),
                json.dumps(
                    {
                        "type": "search",
                        "stage": "returned",
                        "direct_pages": ["page-a"],
                        "expanded_pages": ["page-b"],
                    }
                ),
                json.dumps(
                    {
                        "type": "used",
                        "stage": "used",
                        "page_ids": ["page-b"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (raw_dir / "capture.md").write_text("raw", encoding="utf-8")
    os.utime(raw_dir, ns=(raw_mtime + 1, raw_mtime + 1))
    activity_log.write_text(
        "- [22:14:00] ingest | updated folder/page-c.md\n",
        encoding="utf-8",
    )

    events = cursor.poll()

    assert [event["kind"] for event in events] == [
        "auto_recall",
        "read",
        "search",
        "used",
        "save",
        "ingest",
    ]
    assert events[0]["page_ids"] == ["current-state", "page-a"]
    assert events[1]["page_ids"] == ["current-state"]
    assert events[2]["page_ids"] == ["page-a"]
    assert events[3]["page_ids"] == ["page-b"]
    assert events[4]["page_ids"] == []
    assert events[4]["phase"] == "capture"
    assert events[4]["byte_count"] == 3
    assert events[4]["raw_count"] == 1
    assert len(events[4]["capture_id"]) == 12
    assert events[5]["page_ids"] == ["page-c"]
    assert events[5]["phase"] == "apply"
    assert events[5]["operation"] == "updated"
    assert all(event["source"] == "telemetry-fallback" for event in events)
    assert all(
        event["schema"] == "chronovisor.cortex.event.v2" for event in events
    )
    assert all(event["family"] == "transport" for event in events)
    assert all(event["mode"] == "live" for event in events)
    assert all("duration" not in event for event in events)
    assert [event["origin"] for event in events] == [
        "recall-log",
        "pull-log",
        "pull-log",
        "pull-log",
        "raw-snapshot",
        "activity-log",
    ]
    assert events[0]["presentation"] == {
        "lane_key": "recall",
        "phase": "generate",
        "channel_key": "auto_recall",
        "priority_class": "protected",
    }
    assert events[4]["presentation"] == {
        "lane_key": "raw_buffer",
        "phase": "capture",
        "channel_key": "save",
        "priority_class": "protected",
    }
    assert events[5]["presentation"] == {
        "lane_key": "ingest",
        "phase": "apply",
        "channel_key": "ingest",
        "priority_class": "standard",
    }


def test_cortex_event_cursor_projects_ingest_lifecycle_phases(tmp_path: Path) -> None:
    root = tmp_path / "chronovisor"
    activity_log = root / "log.md"
    activity_log.parent.mkdir(parents=True)
    activity_log.write_text("", encoding="utf-8")
    cursor = cortex.CortexEventCursor(root, activity_log=activity_log)

    activity_log.write_text(
        "\n".join(
            [
                "- [22:10:00] ingest | stage 1: triage started",
                "- [22:10:01] ingest | generating 1/1: semantic-projection/page-d",
                "- [22:10:02] ingest | authorization: local -> apply_available",
                "- [22:10:03] ingest | created semantic-projection/page-d.md",
                "- [22:10:04] ingest | completed in 4.0s",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    events = cursor.poll()

    assert [event["phase"] for event in events] == [
        "triage",
        "generate",
        "consensus",
        "apply",
        "complete",
    ]
    assert events[1]["page_ids"] == ["page-d"]
    assert events[3]["operation"] == "created"
    assert all(event["origin"] == "activity-log" for event in events)
    assert [event["presentation"]["phase"] for event in events] == [
        "triage",
        "generate",
        "consensus",
        "apply",
        "complete",
    ]


def test_cortex_event_cursor_bounds_and_sanitizes_page_ids(tmp_path: Path) -> None:
    root = tmp_path / "chronovisor"
    pull_log = root / "recall" / "pull-log.jsonl"
    pull_log.parent.mkdir(parents=True)
    pull_log.write_text("", encoding="utf-8")
    cursor = cortex.CortexEventCursor(root, pull_log=pull_log)
    huge_page_id = "x" * 300
    pull_log.write_text(
        json.dumps(
            {
                "type": "search",
                "direct_pages": [
                    "page-a",
                    42,
                    None,
                    "",
                    "page-a",
                    huge_page_id,
                    huge_page_id,
                    *[f"page-{index}" for index in range(30)],
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    events = cursor.poll()

    assert len(events) == 1
    assert len(events[0]["page_ids"]) == 24
    assert events[0]["page_ids"][:2] == ["page-a", "x" * 240]
    assert all(isinstance(page_id, str) for page_id in events[0]["page_ids"])
    assert all(len(page_id) <= 240 for page_id in events[0]["page_ids"])


def test_cortex_latest_field_stream_includes_real_memory_io(tmp_path: Path) -> None:
    root = tmp_path / "chronovisor"
    event_root = root / "recall" / "field" / "events-v2"
    event_root.mkdir(parents=True)
    recall_log = root / "recall" / "recall-log.jsonl"
    pull_log = root / "recall" / "pull-log.jsonl"
    activity_log = root / "log.md"
    recall_log.write_text("", encoding="utf-8")
    pull_log.write_text("", encoding="utf-8")
    activity_log.write_text("", encoding="utf-8")
    event_path = event_root / "0123456789abcdef.jsonl"
    event_path.write_text("", encoding="utf-8")
    _append_v2_capture(root, after_line=0)
    cursor = cortex.CortexEventCursor(
        root,
        activity_log=activity_log,
        follow_field_sessions=True,
    )

    event_path.write_text(
        json.dumps(
            {
                "seq": 1,
                "timestamp_epoch": 10.0,
                "session_hash": event_path.stem,
                "topic_epoch": 0,
                "kind": "stimulus",
                "page_id": "page-a",
                "delta": 1.0,
                "activation": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    recall_log.write_text(
        json.dumps(
            {
                "stage": "injected",
                "status": "ok",
                "decision": "read",
                "pages": ["page-recalled"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pull_log.write_text(
        "\n".join(
            [
                json.dumps({"type": "read", "page_id": "page-read"}),
                json.dumps(
                    {"type": "search", "direct_pages": ["page-searched"]}
                ),
                json.dumps({"type": "used", "page_ids": ["page-used"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = _append_v2_capture(root, after_line=1)
    activity_log.write_text(
        "- [22:10:03] ingest | updated page-b.md\n",
        encoding="utf-8",
    )

    events = cursor.poll()

    assert [event["kind"] for event in events] == [
        "stimulus",
        "auto_recall",
        "read",
        "search",
        "used",
        "save",
        "ingest",
    ]
    assert events[0]["source"] == "stateful-recall-field"
    assert events[1]["page_ids"] == ["page-recalled"]
    assert events[2]["page_ids"] == ["page-read"]
    assert events[3]["page_ids"] == ["page-searched"]
    assert events[4]["page_ids"] == ["page-used"]
    assert events[5]["phase"] == "capture"
    assert events[5]["byte_count"] == receipt.commit.length
    assert events[5]["raw_count"] == 1
    assert events[5]["raw_id"] == receipt.commit.raw_id
    assert events[5]["origin"] == "raw-journal"
    assert len(events[5]["capture_id"]) == 12
    assert events[6]["phase"] == "apply"
    assert all(
        event["source"] == "telemetry-fallback" for event in events[1:]
    )


def test_cortex_raw_v2_journal_truncation_does_not_replay_capture(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    receipt = _append_v2_capture(root, after_line=0)
    journal = receipt.commit_path.read_text(encoding="utf-8")
    cursor = cortex.CortexEventCursor(root)

    assert cursor.poll() == []
    receipt.commit_path.write_text("", encoding="utf-8")
    assert cursor.poll() == []
    receipt.commit_path.write_text(journal, encoding="utf-8")
    assert cursor.poll() == []
    new_receipt = _append_v2_capture(root, after_line=1)
    events = cursor.poll()
    assert [event["kind"] for event in events] == ["save"]
    assert events[0]["raw_id"] == new_receipt.commit.raw_id
    assert events[0]["origin"] == "raw-journal"


def test_cortex_save_event_reports_mixed_journal_and_snapshot_origin(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    cursor = cortex.CortexEventCursor(root)

    _append_v2_capture(root, after_line=0)
    legacy_capture = root / "raw" / "legacy-capture.md"
    legacy_capture.parent.mkdir(parents=True, exist_ok=True)
    legacy_capture.write_text("legacy raw", encoding="utf-8")

    events = cursor.poll()

    assert [event["kind"] for event in events] == ["save"]
    assert events[0]["origin"] == "raw-journal+raw-snapshot"
    assert events[0]["raw_count"] == 2


def test_cortex_field_projection_is_sealed_session_scoped_and_browser_safe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "chronovisor"
    session = "0123456789abcdef"
    field_root = root / "recall" / "field"
    store = RecallFieldStore(
        root=field_root,
        config=RecallFieldConfig(mode="shadow"),
    )

    def mutate(state):
        state.topic_epoch = 2
        state.turn = 7
        state.updated_at_epoch = 200.0
        state.shadow["page-a"] = ActivationNode(
            activation=0.72,
            direct=0.6,
            spread=0.2,
            inhibition=0.08,
            last_seq=2,
        )
        state.seq = 2
        return state, [
            FieldEvent(
                seq=1,
                timestamp_epoch=199.0,
                session_hash=session,
                topic_epoch=2,
                kind="stimulus",
                page_id="page-a",
                delta=0.6,
                activation=0.6,
                reason_code="exact_page",
            ),
            FieldEvent(
                seq=2,
                timestamp_epoch=200.0,
                session_hash=session,
                topic_epoch=2,
                kind="commit_queued",
                page_id="page-a",
                reason_code="teacher_commit_next_turn",
                certificate_id="cert-safe",
            ),
        ]

    store.transact(session, mutate, now=200.0)
    monkeypatch.setattr(
        "chronovisor.recall.recall_field_schema.load_recall_field_config",
        lambda: RecallFieldConfig(mode="shadow"),
    )

    projection = cortex.build_cortex_field_projection(
        root,
        session_hash=session,
        now=210.0,
    )

    assert projection["status"] == "online"
    assert projection["session_hash"] == session
    assert projection["snapshot"]["seq"] == 2
    assert projection["snapshot"]["nodes"] == [
        {
            "page_id": "page-a",
            "activation": 0.72,
            "components": {
                "direct": 0.6,
                "spread": 0.2,
                "negative": 0.0,
                "inhibition": 0.08,
                "anti_index": 0.0,
                "hub_penalty": 0.0,
            },
            "last_seq": 2,
        }
    ]
    assert projection["events"][-1]["certificate_id"] == "cert-safe"
    encoded = json.dumps(projection)
    assert "prompt" not in encoded
    assert "body" not in encoded
    assert projection["summary"]["commit"] == 1


def test_cortex_growth_summary_is_bounded_and_corruption_tolerant(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    path = root / "runtime" / "recall-field" / "growth-state.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "stage": "canary",
                "field_learning_allowed": True,
                "positive_learning_allowed": True,
                "policy_update_allowed": True,
                "authority_enabled": True,
                "canary_percent": "5",
                "thresholds": {
                    "strong_positive": 200,
                    "strong_positive_sessions": 20,
                },
                "metrics": {
                    "labels": {
                        "strong_positive": 210,
                        "strong_positive_sessions": 21,
                    },
                    "candidate": {"traces": {"corrupt": True}},
                    "processor_used": {"episodes": 52},
                },
                "private": "must-not-leak",
            }
        ),
        encoding="utf-8",
    )

    assert cortex._field_growth_summary(root) == {
        "stage": "canary",
        "field_learning_allowed": True,
        "positive_learning_allowed": True,
        "policy_update_allowed": True,
        "authority_enabled": True,
        "canary_percent": 5,
        "strong_positive": 210,
        "strong_positive_target": 200,
        "strong_sessions": 21,
        "strong_sessions_target": 20,
        "candidate_traces": 0,
        "processor_used_episodes": 52,
    }


def test_cortex_event_cursor_tails_only_selected_field_session(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    session = "0123456789abcdef"
    event_path = root / "recall" / "field" / "events-v2" / f"{session}.jsonl"
    event_path.parent.mkdir(parents=True)
    event_path.write_text("", encoding="utf-8")
    recall_log = root / "recall" / "recall-log.jsonl"
    pull_log = root / "recall" / "pull-log.jsonl"
    activity_log = root / "log.md"
    recall_log.write_text("", encoding="utf-8")
    pull_log.write_text("", encoding="utf-8")
    activity_log.write_text("", encoding="utf-8")
    cursor = cortex.CortexEventCursor(root, field_session=session)

    event_path.write_text(
        json.dumps(
            {
                "seq": 1,
                "timestamp_epoch": 10.0,
                "session_hash": session,
                "topic_epoch": 0,
                "kind": "spread",
                "source_page_id": "page-a",
                "target_page_id": "page-b",
                "edge_type": "wikilink",
                "delta": 0.4,
                "activation": 0.5,
                "components": {"spread": 0.5, "private": 99},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    recall_log.write_text(
        json.dumps(
            {
                "stage": "injected",
                "status": "ok",
                "decision": "read",
                "pages": ["page-recalled"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pull_log.write_text(
        "\n".join(
            [
                json.dumps({"type": "read", "page_id": "page-read"}),
                json.dumps(
                    {"type": "search", "direct_pages": ["page-searched"]}
                ),
                json.dumps({"type": "used", "page_ids": ["page-used"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = _append_v2_capture(root, after_line=0)
    invalid_commit = receipt.commit.to_dict()
    invalid_commit["after_line"] = "not-an-integer"
    receipt.commit_path.write_text(
        "{malformed\n"
        + json.dumps(invalid_commit)
        + "\n"
        + receipt.commit_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    activity_log.write_text(
        "- [22:10:03] ingest | updated page-ingested.md\n",
        encoding="utf-8",
    )

    events = cursor.poll()

    assert [event["kind"] for event in events] == [
        "spread",
        "auto_recall",
        "read",
        "search",
        "used",
        "save",
        "ingest",
    ]
    assert events[0] == {
        "seq": 1,
        "timestamp_epoch": 10.0,
        "session_hash": session,
        "topic_epoch": 0,
        "kind": "spread",
        "source_page_id": "page-a",
        "target_page_id": "page-b",
        "edge_type": "wikilink",
        "delta": 0.4,
        "activation": 0.5,
        "components": {
            "direct": 0.0,
            "spread": 0.5,
            "negative": 0.0,
            "inhibition": 0.0,
            "anti_index": 0.0,
            "hub_penalty": 0.0,
        },
        "source": "stateful-recall-field",
    }
    assert events[1]["page_ids"] == ["page-recalled"]
    assert events[2]["page_ids"] == ["page-read"]
    assert events[3]["page_ids"] == ["page-searched"]
    assert events[4]["page_ids"] == ["page-used"]
    assert events[5]["raw_id"] == receipt.commit.raw_id
    assert events[5]["byte_count"] == receipt.commit.length
    assert events[6]["page_ids"] == ["page-ingested"]


def test_cortex_event_cursor_follows_activity_across_field_sessions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    event_root = root / "recall" / "field" / "events-v2"
    event_root.mkdir(parents=True)
    first = event_root / "0123456789abcdef.jsonl"
    first.write_text("", encoding="utf-8")
    cursor = cortex.CortexEventCursor(root, follow_field_sessions=True)

    first.write_text(
        json.dumps(
            {
                "seq": 1,
                "timestamp_epoch": 10.0,
                "session_hash": first.stem,
                "topic_epoch": 0,
                "kind": "stimulus",
                "page_id": "page-a",
                "delta": 1.0,
                "activation": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    second = event_root / "fedcba9876543210.jsonl"
    second.write_text(
        json.dumps(
            {
                "seq": 1,
                "timestamp_epoch": 11.0,
                "session_hash": second.stem,
                "topic_epoch": 0,
                "kind": "stimulus",
                "page_id": "page-b",
                "delta": 1.0,
                "activation": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = cursor.poll_payload()

    assert payload == {
        "type": "session_changed",
        "session_hash": second.stem,
        "previous_session_hash": first.stem,
        "committed_seq": 1,
    }
    assert cursor.poll_payload() == {"type": "events", "events": []}


def test_cortex_field_cursor_uses_committed_seq_batches_and_survives_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    session = "0123456789abcdef"
    store = RecallFieldStore(root=root / "recall" / "field")

    def seed(state):
        state.seq = 100
        state.updated_at_epoch = 100.0
        return state, [
            FieldEvent(
                seq=seq,
                timestamp_epoch=float(seq),
                session_hash=session,
                topic_epoch=0,
                kind="stimulus",
                page_id=f"page-{seq}",
                delta=0.1,
                activation=0.1,
            )
            for seq in range(1, 101)
        ]

    store.transact(session, seed, now=100.0)
    cursor = cortex.CortexEventCursor(
        root,
        field_session=session,
        after_seq=0,
    )
    batches = [cursor.poll_payload() for _index in range(4)]

    assert [len(batch["events"]) for batch in batches] == [32, 32, 32, 4]
    assert [
        event["seq"] for batch in batches for event in batch["events"]
    ] == list(range(1, 101))

    def append_one(state):
        state.seq = 101
        state.updated_at_epoch = 101.0
        return state, [
            FieldEvent(
                seq=101,
                timestamp_epoch=101.0,
                session_hash=session,
                topic_epoch=0,
                kind="stimulus",
                page_id="page-101",
            )
        ]

    # RecallFieldStore replaces the journal inode atomically. Logical sequence
    # reads still deliver the new event exactly once regardless of byte size.
    store.transact(session, append_one, now=101.0)
    assert [event["seq"] for event in cursor.poll_payload()["events"]] == [101]
    assert cursor.poll_payload() == {"type": "events", "events": []}

    advanced = cortex.CortexEventCursor(
        root,
        field_session=session,
        after_seq=999,
    )
    assert advanced.poll_payload() == {
        "type": "resync",
        "session_hash": session,
        "after_seq": 999,
        "committed_seq": 101,
        "reason": "field_watermark_ahead",
    }


def test_cortex_field_cursor_closes_projection_to_websocket_race_and_switches(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    first = "0123456789abcdef"
    second = "fedcba9876543210"
    store = RecallFieldStore(root=root / "recall" / "field")

    def write_event(session_hash: str, seq: int, page_id: str):
        def mutate(state):
            state.seq = seq
            state.updated_at_epoch = float(seq)
            return state, [
                FieldEvent(
                    seq=seq,
                    timestamp_epoch=float(seq),
                    session_hash=session_hash,
                    topic_epoch=0,
                    kind="stimulus",
                    page_id=page_id,
                )
            ]

        store.transact(session_hash, mutate, now=float(seq))

    write_event(first, 1, "page-a")
    projection = cortex.build_cortex_field_projection(
        root,
        session_hash=first,
        now=1.0,
    )
    write_event(first, 2, "page-race")
    cursor = cortex.CortexEventCursor(
        root,
        field_session=first,
        after_seq=projection["snapshot"]["seq"],
    )
    assert [event["seq"] for event in cursor.poll_payload()["events"]] == [2]

    follower = cortex.CortexEventCursor(
        root,
        field_session=first,
        follow_field_sessions=True,
        after_seq=2,
    )
    write_event(second, 1, "page-b")
    changed = follower.poll_payload()
    assert changed == {
        "type": "session_changed",
        "session_hash": second,
        "previous_session_hash": first,
        "committed_seq": 1,
    }
    assert follower.poll_payload() == {"type": "events", "events": []}


def test_cortex_field_cursor_is_independent_of_atomic_file_size_changes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    session = "0123456789abcdef"
    store = RecallFieldStore(
        root=root / "recall" / "field",
        config=RecallFieldConfig(event_retention=3),
    )

    def append_next() -> None:
        def mutate(state):
            sequence = state.seq + 1
            state.seq = sequence
            state.updated_at_epoch = float(sequence)
            return state, [
                FieldEvent(
                    seq=sequence,
                    timestamp_epoch=float(sequence),
                    session_hash=session,
                    topic_epoch=0,
                    kind="stimulus",
                    page_id=f"page-{sequence}",
                )
            ]

        store.transact(session, mutate)

    append_next()
    cursor = cortex.CortexEventCursor(root, field_session=session, after_seq=1)
    event_path = store.event_root / f"{session}.jsonl"
    delivered = []
    for _index in range(3):
        append_next()
        delivered.extend(
            event["seq"] for event in cursor.poll_payload()["events"]
        )
    before_shrink = event_path.stat().st_size
    store.config = RecallFieldConfig(event_retention=1)
    append_next()
    after_shrink = event_path.stat().st_size
    delivered.extend(event["seq"] for event in cursor.poll_payload()["events"])

    assert delivered == [2, 3, 4, 5]
    assert after_shrink < before_shrink


def test_cortex_event_cursor_bounds_whole_envelope_and_preserves_tail_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    recall_log = root / "recall" / "recall-log.jsonl"
    recall_log.parent.mkdir(parents=True)
    recall_log.write_text("", encoding="utf-8")
    cursor = cortex.CortexEventCursor(root, recall_log=recall_log)
    recall_log.write_text(
        "".join(
            json.dumps(
                {
                    "stage": "injected",
                    "status": "ok",
                    "decision": "read",
                    "pages": [f"page-{index:03}"],
                }
            )
            + "\n"
            for index in range(100)
        ),
        encoding="utf-8",
    )

    batches = [cursor.poll_payload() for _index in range(4)]

    assert [len(batch["events"]) for batch in batches] == [32, 32, 32, 4]
    assert [
        event["page_ids"][0]
        for batch in batches
        for event in batch["events"]
    ] == [f"page-{index:03}" for index in range(100)]
    assert cursor.poll_payload() == {"type": "events", "events": []}


def test_cortex_field_cursor_fails_closed_for_corrupt_existing_snapshot(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    session = "0123456789abcdef"
    field_root = root / "recall" / "field"
    event_root = field_root / "events-v2"
    session_root = field_root / "sessions-v2"
    event_root.mkdir(parents=True)
    session_root.mkdir(parents=True)
    event = {
        "seq": 1,
        "timestamp_epoch": 1.0,
        "session_hash": session,
        "topic_epoch": 0,
        "kind": "stimulus",
        "page_id": "page-a",
        "delta": 1.0,
        "activation": 1.0,
    }
    (event_root / f"{session}.jsonl").write_text(
        json.dumps(event) + "\n",
        encoding="utf-8",
    )
    missing_snapshot = cortex.CortexEventCursor(root, field_session=session)
    assert [
        row["seq"] for row in missing_snapshot.poll_payload()["events"]
    ] == [1]

    (session_root / f"{session}.json").write_text(
        '{"seq":1,"snapshot_sha256":"invalid"}\n',
        encoding="utf-8",
    )
    corrupt_snapshot = cortex.CortexEventCursor(root, field_session=session)
    assert corrupt_snapshot.poll_payload() == {
        "type": "resync",
        "session_hash": session,
        "after_seq": 0,
        "committed_seq": 0,
        "reason": "field_snapshot_corrupt",
    }


def test_cortex_projection_center_tracks_mode_bar_safe_area() -> None:
    script = (
        dashboard.STATIC_DIR / "cortex.js"
    ).read_text(encoding="utf-8")
    center_contract = script.split(
        "  function updateProjectionCenter()", 1
    )[1].split("\n  function resize()", 1)[0]
    project_all = script.split(
        "  function projectAll()", 1
    )[1].split("\n  function fog(", 1)[0]

    assert 'const modeBar = document.getElementById("modeBar");' in script
    assert "const stageBounds = stage.getBoundingClientRect();" in center_contract
    assert "const modeBarBounds = modeBar.getBoundingClientRect();" in center_contract
    assert "(modeBarBounds.bottom - stageBounds.top)" in center_contract
    assert "(height / stageBounds.height)" in center_contract
    assert "projectionCenterX = width / 2;" in center_contract
    assert "projectionCenterY = height / 2;" in center_contract
    assert "projectionCenterY = (projectionTopInset + height) / 2;" in center_contract
    assert "updateProjectionCenter();" in script
    assert "projectionCenterX = width / 2;" not in project_all
    assert "projectionCenterY = height / 2;" not in project_all
    assert "getBoundingClientRect" not in project_all
    assert "resizeObserver.observe(stage);" in script
    assert "resizeObserver.observe(modeBar);" in script
    assert "target.dataset.projectionCenterX" in script
    assert "target.dataset.projectionCenterY" in script
    assert "target.dataset.projectionTopInset" in script


def test_cortex_sphere_mode_dom_persistence_simulation_and_renderer_contract() -> None:
    static = dashboard.STATIC_DIR
    html = (static / "cortex.html").read_text(encoding="utf-8")
    style = (static / "cortex.css").read_text(encoding="utf-8")
    script = (static / "cortex.js").read_text(encoding="utf-8")
    runtime_script = (static / "cortex-runtime.js").read_text(encoding="utf-8")
    webgl_script = (static / "cortex-webgl.js").read_text(encoding="utf-8")
    tick = script[
        script.index("function tick()") : script.index("function sleepSimulation()")
    ]
    set_mode = script[
        script.index("function setMode(") : script.index("let factIndex")
    ]
    fog = script[
        script.index("function fog(") : script.index("function graphRadius()")
    ]
    draw_edges = script[
        script.index("function drawEdges()") : script.index(
            "function drawTypedRelations()"
        )
    ]
    draw_typed_relations = script[
        script.index("function drawTypedRelations()") : script.index(
            "function drawCommunityHulls()"
        )
    ]
    fit_view = script[
        script.index("function fitView()") : script.index(
            "function scheduleModeFit("
        )
    ]

    assert html.index(">ORGANIC</button>") < html.index(">SPHERE</button>")
    assert html.index(">SPHERE</button>") < html.index(">CLUSTERS</button>")
    assert 'class="seg modeSeg" role="group" aria-label="Graph layout"' in html
    assert 'id="mOrganic" class="on" type="button" aria-pressed="true"' in html
    assert 'id="mSphere" type="button" aria-pressed="false"' in html
    assert 'id="mCluster" type="button" aria-pressed="false"' in html
    assert ".modeSeg button" in style
    assert "min-height: 28px" in style
    assert ".seg button:focus-visible" in style
    assert "grid-template-columns: repeat(4, minmax(0, 1fr))" in style
    assert "@media (max-width: 420px)" in style
    assert "flex-flow: row nowrap;" in style
    assert "#zoomCtl button" in style
    assert "width: 32px;" in style
    assert "height: 32px;" in style
    assert "Runtime.normalizeLayoutMode(value.mode)" in script
    assert '["mSphere", "sphere"]' in script
    assert 'control.setAttribute("aria-pressed", String(active));' in script
    assert 'document.getElementById("mSphere").addEventListener' in script
    assert 'setMode("sphere");' in script
    assert "Runtime.normalizeLayoutMode(nextMode)" in set_mode
    assert 'reheat(mode === "sphere" ? 0.9 : 0.7);' in set_mode
    assert "syncViewPreferenceControls();" in set_mode
    assert "saveViewPreferences();" in set_mode
    assert "scheduleModeFit(mode);" in set_mode
    assert "window.clearTimeout(modeFitTimer);" in script
    assert "if (mode === expectedMode) {" in script
    assert 'invalidate("mode-fit");' in script
    assert 'if (mode === "sphere") scheduleModeFit(mode);' in script
    assert 'mode: "organic"' in script
    assert "Runtime.createSphereTargets(nodes)" in script
    assert "nodes[index].sphereTarget = target" in script
    assert 'const sphereMode = mode === "sphere"' in tick
    assert "SPHERE_TARGET_FORCE" in tick
    assert "SPHERE_RADIAL_FORCE" in tick
    assert "SPHERE_LINK_SPRING" in tick
    assert "const centerForce = sphereMode ? 0" in tick
    assert "for (let index = 0; index < simulationLinks.length" in tick
    assert "SPHERE_BACK_HEMISPHERE_FADE" in fog
    assert "sphereQuality.targetRadius.max" in fog
    assert "Runtime.fitSphereCamera({" in fit_view
    assert "sphereRadius: graphRadiusFrom(center)" in fit_view
    assert "focalLength: projectionFocalLength()" in fit_view
    assert "topInset: projectionTopInset" in fit_view
    assert "padding: SPHERE_FIT_PADDING_PX" in fit_view
    assert "Math.max(600, graphRadius() * 2.55)" in fit_view
    assert "normalEdgeScale: mode === \"sphere\"" in script
    assert "opacity *= SPHERE_NORMAL_EDGE_SCALE" in script
    assert "const depthFade = fog(depth);" in draw_edges
    assert "const depthBandCount = sphereMode ? 4 : 3;" in draw_edges
    assert "Runtime.sphereFogBand(depthFade)" in draw_edges
    assert "Runtime.sphereFogOpacity(band)" in draw_edges
    assert draw_edges.count("context.stroke(paths[batch])") == 1
    assert 'const sphereMode = mode === "sphere";' in draw_typed_relations
    assert "Runtime.sphereFogBand(depthFade)" in draw_typed_relations
    assert "Runtime.sphereFogOpacity(depthBand)" in draw_typed_relations
    assert draw_typed_relations.count("context.stroke(batch.path)") == 1
    assert "Number(scene.normalEdgeScale)" in webgl_script
    assert 'mode === "sphere" ? left.viewDepth - right.viewDepth' in script
    assert "sphereQuality = Runtime.measureSphereQuality" in script
    assert "target.dataset.layoutMode = mode" in script
    assert "target.dataset.sphereCoreNodes" in script
    assert "target.dataset.sphereTargetErrorMean" in script
    assert "function createSphereTargets(" in runtime_script
    assert "function measureSphereQuality(" in runtime_script
    assert "function fitSphereCamera(" in runtime_script
    assert "function sphereFogBand(" in runtime_script
    assert "function sphereFogOpacity(" in runtime_script
    assert "function convexHull(" in runtime_script
    assert "Runtime.convexHull(community)" in script
    # The established ORGANIC/CLUSTERS package-anchor handedness stays intact.
    assert "Math.PI * (1 + Math.sqrt(5)) * offset" in script


def test_cortex_staged_renderer_lazy_explorer_and_relation_detail_contract() -> None:
    static = dashboard.STATIC_DIR
    html = (static / "cortex.html").read_text(encoding="utf-8")
    style = (static / "cortex.css").read_text(encoding="utf-8")
    script = (static / "cortex.js").read_text(encoding="utf-8")
    webgl_script = (static / "cortex-webgl.js").read_text(encoding="utf-8")
    build_tree = script[
        script.index("function buildTree()") : script.index(
            "function renderTreeSelection()"
        )
    ]
    draw_relations = script[
        script.index("function drawTypedRelations()") : script.index(
            "function drawCommunityHulls()"
        )
    ]
    draw_nodes = script[
        script.index("function drawNodes(") : script.index(
            "function drawCameraPivot("
        )
    ]
    core_block = draw_nodes[
        draw_nodes.index("// WebGL owns the full core.") : draw_nodes.index(
            "if (excitation > 0.35)"
        )
    ]
    overflow_key = script[
        script.index("function transportOverflowKey(") : script.index(
            "function mergeTransportOverflow("
        )
    ]
    terminal_control = script[
        script.index("function handleTerminalFieldControl(") : script.index(
            "function processQueuedEvents("
        )
    ]

    assert 'id="gl" class="cortexBase"' in html
    assert 'id="overlay" class="cortexOverlay"' in html
    assert "/static/cortex-runtime.js" in html
    assert "/static/cortex-webgl.js" in html
    assert html.index("/static/cortex-runtime.js") < html.index("/static/cortex.js")
    assert html.index("/static/cortex-webgl.js") < html.index("/static/cortex.js")
    assert "#stage canvas.cortexBase" in style
    assert "#stage canvas.cortexOverlay" in style
    assert "const EXPLORER_CHUNK_SIZE = 120;" in script
    assert "function materializePackage(" in script
    assert 'tree.addEventListener("click"' in build_tree
    assert 'tree.addEventListener("dblclick"' in build_tree
    assert "moduleRow.addEventListener" not in build_tree
    assert "materializePackage(packageName);" in build_tree
    assert "Runtime.relationBatchKey(" in draw_relations
    assert "context.save();" in draw_relations
    assert draw_relations.count("context.stroke(") == 1
    assert "relationDetailsByKey" in script
    assert "String(relation.source_page_id" in script
    assert "String(relation.target_page_id" in script
    assert 'fetch(\n        `/api/cortex/relations?keys=' in script
    assert "relationDetailRequest" in script
    assert "AbortController" in script
    assert "EVENT_DRAIN_PER_FRAME = 32" in script
    assert "VISUAL_DRAIN_PER_FRAME = 4" in script
    assert "overflowCoalesceKey: transportOverflowKey" in script
    assert "const incomingFieldEvents = Runtime.createEventQueue" in script
    assert "const incomingTransportEvents = Runtime.createEventQueue" in script
    assert "incomingFieldEvents.clear();" in script
    resync_block = script[
        script.index("if (fieldResyncPending) {") : script.index(
            "if (panelRenderPending) {"
        )
    ]
    assert "incomingTransportEvents.clear" not in resync_block
    assert "coalesced_count: coalescedCount" in script
    assert "String(event.kind" in overflow_key
    assert "String(event.phase" in overflow_key
    assert "String(event.channel_key" in overflow_key
    assert "transportByKind: {}" in script
    assert "cortexMetrics.transportByKind[transportKind]" in script
    assert 'payload.reason !== "field_snapshot_corrupt"' in terminal_control
    assert "terminalFieldControlGate.accept(generation" in terminal_control
    assert "reconnectSuppressedGeneration = generation" in terminal_control
    assert "fieldResyncPending = false" in terminal_control
    assert "incomingTransportEvents.clear" not in terminal_control
    assert "if (reconnectSuppressedGeneration === generation) return;" in script
    assert 'data-retry-field' in script
    assert "target.dataset.longAnimationFrames" in script
    assert "target.dataset.projectionBaseP95Ms" not in script  # emitted generically
    assert 'stageMetrics.record("projectionBase"' in script
    assert 'stageMetrics.record("overlayEffects"' in script
    assert 'stageMetrics.record("domEventFlush"' in script
    assert "1.0 - smoothstep(0.28, 0.5, radius)" in webgl_script
    assert "fallbackLatched = true;" in webgl_script
    assert "if (!overlayOnly)" in core_block
    assert "context.fill();" in core_block
    assert 'canvas.style.opacity = visible ? "1" : "0"' in webgl_script


def test_cortex_static_view_preserves_fable_layout_and_uses_live_data() -> None:
    static = dashboard.STATIC_DIR
    html = (static / "cortex.html").read_text(encoding="utf-8")
    style = (static / "cortex.css").read_text(encoding="utf-8")
    script = (static / "cortex.js").read_text(encoding="utf-8")
    runtime_script = (static / "cortex-runtime.js").read_text(encoding="utf-8")
    policy_script = (static / "cortex-transport-policy.js").read_text(
        encoding="utf-8"
    )
    field_script = (static / "cortex-field.js").read_text(encoding="utf-8")
    observatory = (static / "index.html").read_text(encoding="utf-8")
    activity_style = (static / "activity-bar.css").read_text(encoding="utf-8")

    assert "CHRONOVISOR // SYNAPTIC CORTEX" in html
    assert 'id="side"' in html
    assert 'id="stage"' in html
    assert 'id="hud"' in html
    assert 'id="mOrganic"' in html
    assert 'id="mCluster"' in html
    assert 'id="tLive"' in html
    assert 'id="tMotion"' in html
    assert 'id="sessionSelect"' in html
    assert 'id="processingLanesMonitor"' in html
    assert 'id="resetCenter"' in html
    assert "◎ RESET CENTER" in html
    assert 'id="tAuto"' not in html
    assert 'id="tSnd"' in html
    assert 'id="tReset"' in html
    assert "grid-template-columns: 242px 1fr 296px;" in style
    assert "--amber: #ffb454;" in style
    assert "repeating-linear-gradient" in style
    assert 'fetch("/api/cortex/graph"' in script
    assert "/api/cortex/events${queryString}" in script
    assert "function firePageIds(" not in script
    assert "function fire(" not in script
    assert "function visualizeFieldEvent(event)" in script
    assert "function visualizeTransportEvent(event)" in script
    assert "function drawTransportEffects(time)" in script
    assert 'new EventSource("/api/activity-stream")' in script
    assert "function connectProcessingActivity()" in script
    assert "function applyProcessingActivity(snapshot)" in script
    assert "TransportPolicy.processingEffectPhase" in script
    assert "function pulseActiveProcessingLanes()" in script
    assert "function processingTargetNode(laneKey)" in script
    assert "function drawProcessingNodeBlink(" in script
    assert 'effect.laneKey === "ingest"' in script
    assert "processingTargetLocked" in script
    assert "cortexMetrics.processingNodeBlinks += 1;" in script
    assert "cortexMetrics.processingTargetNodeIndex = target.node.index;" in script
    assert (
        "const PROCESSING_EFFECT_PULSE_MS = TransportPolicy.PROCESSING_EFFECT_PULSE_MS;"
        in script
    )
    assert "function graphCenter()" in script
    assert "function setNodeAsCameraPivot(index)" in script
    assert "function resetCameraPivot(announce = true)" in script
    assert "function drawCameraPivot(time)" in script
    assert "camera.pivotNodeIndex = nodeIndex;" in script
    assert "node.x - camera.pivotX" in script
    assert "node.y - camera.pivotY" in script
    assert "node.z - camera.pivotZ" in script
    assert "select(index, index >= 0);" in script
    assert 'document.getElementById("resetCenter").addEventListener' in script
    assert "resetCameraPivot();" in script
    assert "drawCameraPivot(time);" in script
    assert "target.dataset.cameraPivotNodeIndex" in script
    assert ".seg button.centerReset.on" in style
    assert 'kind: "processing"' in policy_script
    assert '`processing:${laneKey || "audit"}`' in policy_script
    assert "connectProcessingActivity();" in script
    assert "function drawCaptureWaveTrain(" in script
    assert "function captureWaveGeometry(" in script
    assert "function captureWaveState(" in runtime_script
    assert "Runtime.captureWaveState(" in script
    assert "function drawCaptureHeartbeat(" in script
    assert "function memoryStarGeometry(" in script
    assert "captureWaveDurationMs: 3600" in policy_script
    assert "captureWaveDurationMs: 5000" in policy_script
    assert "function drawCaptureComets(" not in script
    assert "function captureCometPoint(" not in script
    assert "function captureSafeRect(" not in script
    assert "captureCometDurationMs" not in policy_script
    assert "turns: 2.85" not in script
    assert "function drawCaptureElectricity(" not in script
    assert "function drawTriageFormation(" in script
    assert "function drawGenerateFormation(" in script
    assert "function drawConsensusFormation(" in script
    assert "function ingestFormationCandidates(" in script
    assert "function drawIngestElectricity(" not in script
    assert "function transportAnchor(" not in script
    assert "function retireSupersededIngestEffects(" in script
    assert "function drawApplyFormation(" in script
    assert "function visibleConsolidationNeighbors(node)" in script
    assert 'phase: "triage"' in script
    assert 'phase: "consensus"' in script
    assert "scheduleDemoTransport(3300" in script
    assert "cortexMetrics.captureWaveEffects = paintedWaveCount;" in script
    assert "cortexMetrics.captureWavePeak = Math.max(" in script
    assert "cortexMetrics.captureHeartbeatPeak = peak;" in script
    assert "cortexMetrics.triageCandidates += 1;" in script
    assert "cortexMetrics.generateParticles += 1;" in script
    assert "cortexMetrics.consensusOrbits += 1;" in script
    assert "cortexMetrics.consolidationEdges += 1;" in script
    assert "incomingFieldEvents.push(" in script
    assert "incomingTransportEvents.push(" in script
    assert "pendingTransportVisuals" in script
    assert "TransportPolicy.isTransportKind(event.kind)" in script
    assert "drawTransportEffects(time);" in script
    assert "PROCESSING LANES" in html
    assert (
        '.processingLane[data-phase="generate"] { --lane-color: #9b7cff; }' in style
    )
    assert (
        '.processingLane[data-phase="consensus"] { --lane-color: #ffe6ae; }' in style
    )
    assert "violet · memory synthesis" in script
    assert "platinum · local agreement" in script
    assert '.processingLane[data-state="active"]' in style
    assert "function ensureActualEdge(event)" in script
    assert "function drawEdges()" in script
    assert "const ACTIVE_LABEL_LIMIT = 5;" in script
    assert "const NODE_FLASH_ATTACK_MS = 90;" in script
    assert "const NODE_FLASH_HOLD_MS = 150;" in script
    assert "const NODE_FLASH_DECAY_MS = 1450;" in script
    assert "const EDGE_AFTERGLOW_MS = 1550;" in script
    assert "const ELECTRIC_TRAVEL_MIN_MS = 420;" in script
    assert "const ELECTRIC_TRAVEL_MAX_MS = 760;" in script
    assert "const MAX_ELECTRIC_PATHS = TransportPolicy.MAX_ELECTRIC_PATHS;" in script
    assert "const NODE_STIMULUS_SCALE = 0.38;" in script
    assert "const NODE_ARRIVAL_SCALE = 0.28;" in script
    assert "const NODE_CORE_SCALE = 1;" in script
    assert "const NODE_GLOW_MAX_PADDING_PX = 4;" in script
    assert "const NODE_EFFECT_MAX_PADDING_PX = 3;" in script
    assert 'const VIEW_PREFERENCES_KEY = "chronovisor.cortex.preferences.v1";' in script
    assert "function sanitizeViewPreferences(candidate)" in script
    assert "function loadViewPreferences()" in script
    assert "function saveViewPreferences()" in script
    assert "function resetViewPreferences()" in script
    assert "applyViewPreferences(loadViewPreferences());" in script
    assert "window.localStorage.setItem(" in script
    assert "window.localStorage.removeItem(VIEW_PREFERENCES_KEY);" in script
    assert 'window.addEventListener("pointerdown", unlockSound' in script
    assert "function excitationLevel(node, time)" in script
    assert "function exciteNode(node, delta, time)" in script
    assert "function drawCompactGlow(" in script
    assert "const radius = baseRadius * NODE_CORE_SCALE;" in script
    assert "radius + NODE_EFFECT_MAX_PADDING_PX" in script
    assert "target.dataset.maxCoreScale" in script
    assert "target.dataset.maxGlowPadding" in script
    assert "target.dataset.cameraTheta = camera.theta.toFixed(4);" in script
    assert "target.dataset.cameraPhi = camera.phi.toFixed(4);" in script
    assert "camera.theta += pointer.movementX * 0.0045;" in script
    assert "camera.theta -= event.movementX * 0.0045;" not in script
    assert "camera.phi - pointer.movementY * 0.0045" in script
    assert "camera.phi + event.movementY * 0.0045" not in script
    assert "radius + progress * 26" not in script
    assert "radius + (1 - progress) * 28" not in script
    assert "radius + progress * 18" not in script
    assert "radius + progress * 13" not in script
    assert "function electricPathPoints(" in runtime_script
    assert "function electricPathPrefix(" in runtime_script
    assert "Runtime.electricPathPoints(" in script
    assert "Runtime.electricPathPrefix(" in script
    assert "function queueElectricPulse(" in script
    assert "function drawEdgeAfterglows(time)" in script
    assert "target.dataset.electricEdges" in script
    assert "target.dataset.electricPeak" in script
    assert "function publishVisualMetrics(time)" in script
    assert "cortexMetrics.violetNodes += 1;" in script
    assert "if (insertAt >= ACTIVE_LABEL_LIMIT) return;" in script
    assert "Math.min(ACTIVE_LABEL_LIMIT, activeLabelNodes.length + 1)" in script
    assert "liveEventsEnabled = true" in script
    assert "followLatestSession = true" in script
    assert 'params.set("follow", "latest")' in script
    assert 'params.set("after_seq"' in script
    assert "LIVE · follow activity" in script
    assert "window.CortexField.applyEvents(fieldState" in script
    assert "const MAX_EVENTS = 256;" in field_script
    assert "event.seq !== state.seq + 1" in field_script
    assert "/static/cortex-field.js" in html
    assert "/static/cortex-transport-policy.js" in html
    assert html.index("/static/cortex-field.js") < html.index("/static/cortex.js")
    assert html.index("/static/cortex-transport-policy.js") < html.index(
        "/static/cortex.js"
    )
    assert "DEMO · RECALL" in html
    assert "@media (prefers-reduced-motion: reduce)" in style
    assert 'id="recall-field-summary"' in observatory
    assert 'fetch("/api/cortex/field"' in (static / "app-client.js").read_text(
        encoding="utf-8"
    )
    assert "function ambient(" not in script
    assert "function autoTick(" not in script
    assert 'setTimeout(() => stimulate("recall")' not in script
    assert "if (performance.now() <= tickerHold) return;" in script
    assert "/static/cortex_graph.json" not in script
    assert "3d-force-graph" not in html
    assert 'class="has-activity-bar"' in observatory
    assert 'class="has-activity-bar"' in html
    assert 'class="activity-view is-active" data-view="observatory"' in observatory
    assert 'class="activity-view" data-view="cortex"' in observatory
    assert 'class="activity-view" data-view="observatory"' in html
    assert 'class="activity-view is-active" data-view="cortex"' in html
    assert 'aria-label="Chronovisor views"' in observatory
    assert 'aria-label="Chronovisor views"' in html
    assert "--activity-bar-width: 48px;" in activity_style
    assert "body.has-activity-bar {" in activity_style
    assert ".activity-bar {" in activity_style
    assert ".has-activity-bar .shell {" in activity_style


def test_cortex_save_capture_uses_bounded_radial_wave_and_heartbeat() -> None:
    script = (dashboard.STATIC_DIR / "cortex.js").read_text(encoding="utf-8")
    runtime_path = dashboard.STATIC_DIR / "cortex-runtime.js"
    runtime_script = runtime_path.read_text(encoding="utf-8")
    policy_script = (
        dashboard.STATIC_DIR / "cortex-transport-policy.js"
    ).read_text(encoding="utf-8")
    capture_block = script[
        script.index("function captureWaveGeometry()") : script.index(
            "function ingestFormationCandidates(",
        )
    ]
    capture_runtime_block = runtime_script[
        runtime_script.index("function captureWaveState(") : runtime_script.index(
            "function normalizeLayoutMode("
        )
    ]

    assert "const CAPTURE_WAVE_NODE_LIMIT = 112;" in script
    assert "const CAPTURE_WAVE_TAIL_RATIO = 0.42;" in script
    front_opacity = 1.0
    front_padding = 5.5
    tail_opacity = 0.66
    tail_padding = 3.6
    assert (
        f"const CAPTURE_WAVE_FRONT_GLOW_OPACITY = {front_opacity:g};" in script
    )
    assert (
        f"const CAPTURE_WAVE_FRONT_GLOW_PADDING_PX = {front_padding:g};" in script
    )
    assert f"const CAPTURE_WAVE_TAIL_GLOW_OPACITY = {tail_opacity:g};" in script
    assert (
        f"const CAPTURE_WAVE_TAIL_GLOW_PADDING_PX = {tail_padding:g};" in script
    )
    assert front_opacity > tail_opacity
    assert front_padding > tail_padding
    assert "const CAPTURE_WAVE_TRAVEL_END = 0.72;" in runtime_script
    assert "Runtime.CAPTURE_WAVE_TRAVEL_END" in script
    assert "const CAPTURE_WAVE_MAX_VISIBLE = 4;" in script
    assert "const CAPTURE_WAVE_MIN_INTERVAL_MS = 420;" in script
    assert "const CAPTURE_HEARTBEAT_SETTLE_MS = 220;" in script
    assert "const CAPTURE_HEARTBEAT_DURATION_MS = 720;" in script
    assert "captureWaveDistances = new Float32Array(nodes.length);" in capture_block
    assert "node.screenX - centerX" in capture_block
    assert "node.screenY - centerY" in capture_block
    assert "radius: Math.max(maxDistance + 8, shortest * 0.22)" in capture_block
    assert "Math.hypot(width, height) * 0.62" not in capture_block
    assert "function drawCaptureWaveBand(" not in script
    assert "function drawCaptureWaveTrain(" in capture_block
    assert "function captureWaveNodeIntensity(" in capture_block
    assert "function captureWaveSampleStride(" in capture_runtime_block
    assert "greatestCommonDivisor(stride, length)" in capture_runtime_block
    assert "Runtime.captureWaveSampleStride(" in capture_block
    assert "(start + offset * stride) % nodes.length" in capture_block
    assert "if (drawnNodes >= CAPTURE_WAVE_NODE_LIMIT) break;" in capture_block
    assert ".sort(" not in capture_block
    assert "function drawCaptureHeartbeat(" in capture_block
    assert 'phase: "wave"' in capture_runtime_block
    assert 'cortexMetrics.capturePhase = "heartbeat";' in capture_block
    assert 'phase: "settle"' in capture_runtime_block
    settle_block = capture_runtime_block.split('phase: "settle"', 1)[1].split(
        "function captureHeartbeatAlpha(", 1
    )[0]
    assert "wavePeak: 0" in settle_block
    assert "heartbeatPeak" not in settle_block
    assert 'phase: "static"' in capture_runtime_block
    assert "reducedMotion.matches || !motionEnabled" in capture_block
    assert "context.createRadialGradient" not in capture_block
    assert "context.stroke(" not in capture_block
    assert "context.arc(" not in capture_block
    assert "context.fill(" not in capture_block
    assert "camera." not in capture_block
    assert "node.x =" not in capture_block
    assert "node.y =" not in capture_block
    assert "node.z =" not in capture_block
    assert "comet" not in capture_block.lower()
    assert "orbit" not in capture_block.lower()
    assert "captureComet" not in script
    assert "captureCometDurationMs" not in policy_script
    assert "captureWaveDurationMs: 3600" in policy_script
    assert "captureWaveDurationMs: 5000" in policy_script
    assert "const captureWaveTrain = [];" in script
    assert "const captureHeartbeatBurst = {" in script
    assert "function admitCaptureWave(" in script
    assert "function armCaptureHeartbeat(" in script
    assert "function scheduleCaptureHeartbeatWake(" in script
    assert "TransportPolicy.supersedeCaptureVisuals(transportEffects);" not in script
    assert "let paintedThisFrame = false;" in script
    assert "paintedThisFrame && !effect.paintedAt" in script
    assert "effect.heartbeatRecorded" not in capture_block
    assert "cortexMetrics.captureHeartbeatCount += 1;" in capture_block
    assert "const staticMotion = reducedMotion.matches || !motionEnabled;" in (
        capture_block
    )
    assert (
        "const peak = staticMotion ? 0.3 : Runtime.captureHeartbeatPeak(progress);"
        in capture_block
    )
    assert "if (staticMotion && !captureHeartbeatBurst.wakeTimer)" in capture_block
    assert "motionEnabled\n      && !reducedMotion.matches" in capture_block
    node_wave_block = capture_block.split(
        "function drawCaptureWaveTrain(", 1
    )[1].split("function drawCaptureHeartbeat(", 1)[0]
    assert "drawCaptureWaveNodeGlow(node, combinedIntensity, 1, inTail);" in (
        node_wave_block
    )
    assert node_wave_block.count(
        "for (let offset = 0; offset < nodes.length; offset += 1)"
    ) == 1
    assert "for (let index = 0; index < captureWaveTrain.length; index += 1)" in (
        node_wave_block
    )
    assert "context.drawImage(" not in node_wave_block
    assert "const inTail = distance > strongestWave.frameState.waveRadius;" in (
        node_wave_block
    )
    assert "falloffWidth * CAPTURE_WAVE_TAIL_RATIO" in capture_block
    node_glow_block = capture_block.split(
        "function drawCaptureWaveNodeGlow(", 1
    )[1].split("function drawCaptureWaveTrain(", 1)[0]
    assert node_glow_block.count("context.drawImage(") == 1
    assert "CAPTURE_WAVE_FRONT_GLOW_OPACITY" in node_glow_block
    assert "CAPTURE_WAVE_FRONT_GLOW_PADDING_PX" in node_glow_block
    assert "CAPTURE_WAVE_TAIL_GLOW_OPACITY" in node_glow_block
    assert "CAPTURE_WAVE_TAIL_GLOW_PADDING_PX" in node_glow_block
    assert "context.stroke(" not in node_glow_block
    assert "context.arc(" not in node_glow_block
    assert "context.fill(" not in node_glow_block
    heartbeat_draw_block = capture_block.split(
        "function drawCaptureHeartbeat(", 1
    )[1].split("function drawCaptureHeartbeatBurst(", 1)[0]
    assert heartbeat_draw_block.count("context.drawImage(") == 1
    assert "glowCapture" in heartbeat_draw_block
    assert "context.stroke(" not in heartbeat_draw_block
    assert "context.arc(" not in heartbeat_draw_block
    assert "context.fill(" not in heartbeat_draw_block
    assert "context.beginPath(" not in heartbeat_draw_block
    assert "captureWaveTrain.length >= CAPTURE_WAVE_MAX_VISIBLE" in script
    assert "newest.coalescedCount += eventMultiplicity;" in script
    assert "newest.startedAt =" not in script
    assert (
        "Math.max(\n          captureHeartbeatBurst.dueAt,\n"
        "          now + CAPTURE_HEARTBEAT_SETTLE_MS,\n        )"
    ) in script
    assert "combinedIntensity = 1 - (1 - combinedIntensity) * (1 - intensity);" in (
        node_wave_block
    )
    draw_transport_start = script.index("function drawTransportEffects(time)")
    draw_transport_block = script[draw_transport_start : script.index(
        "function compareActiveLabels(", draw_transport_start
    )]
    assert "drawCaptureEffects(time);" in draw_transport_block
    assert 'effect.phase === "capture"' not in draw_transport_block
    draw_effects_block = capture_block.split("function drawCaptureEffects(", 1)[1]
    assert draw_effects_block.index("pruneCaptureWaveTrain(time);") < (
        draw_effects_block.index("captureWaveGeometry()")
    )
    assert "cyan · inward memory wave → heartbeat" in script
    visualize_block = script[
        script.index("function visualizeTransportEvent(event)") : script.index(
            "function visualizeFieldEvent(event)",
        )
    ]
    assert "admitCaptureWave(effect, eventMultiplicity, now);" in visualize_block
    assert "transportEffects.push(effect);" in visualize_block

    for metric in (
        "captureEffectCount",
        "captureEffectSeq",
        "captureEffectCaptureId",
        "captureWaveEffects",
        "captureWaveNodes",
        "captureAfterglowNodes",
        "captureWavePeak",
        "captureHeartbeatPeak",
        "captureCenterX",
        "captureCenterY",
        "captureRadius",
        "captureBandWidth",
        "captureProgress",
        "capturePhase",
        "captureHeartbeatCount",
        "captureHeartbeatSeq",
    ):
        assert f"target.dataset.{metric}" in script

    geometry_source = script[
        script.index("function visibleMemoryNode(") : script.index(
            "function captureWaveNodeIntensity(",
        )
    ]
    geometry_scenario = f"""
const width = 1000;
const height = 600;
const nodes = [
  ...Array.from({{ length: 10 }}, (_, index) => ({{
    index,
    screenX: 0,
    screenY: 100,
    screenScale: 2.5,
    fanIn: 0,
    fanOut: 0,
    viewDepth: 1,
  }})),
  {{
    index: 10,
    screenX: 1040,
    screenY: 568,
    screenScale: 0.3,
    fanIn: 0,
    fanOut: 0,
    viewDepth: 1,
  }},
];
const nodeState = new Uint8Array(nodes.length).fill(2);
let captureWaveDistances = new Float32Array(0);
function clamp(value, minimum = 0, maximum = 1) {{
  return Math.max(minimum, Math.min(maximum, value));
}}
{geometry_source}
const geometry = captureWaveGeometry();
const farthest = Math.max(...captureWaveDistances);
process.stdout.write(JSON.stringify({{
  geometry,
  farthest,
  reachesFarthest: geometry.radius >= farthest + 8 - 0.001,
  exceedsFormerDiagonalCap:
    geometry.radius > Math.hypot(width, height) * 0.62,
}}));
"""
    geometry_completed = subprocess.run(
        ["node", "-e", geometry_scenario],
        check=True,
        capture_output=True,
        text=True,
    )
    geometry_result = json.loads(geometry_completed.stdout)
    assert geometry_result["reachesFarthest"] is True
    assert geometry_result["exceedsFormerDiagonalCap"] is True
    assert geometry_result["geometry"]["radius"] == pytest.approx(
        geometry_result["farthest"] + 8,
        abs=0.001,
    )

    intensity_source = script[
        script.index("function captureWaveNodeIntensity(") : script.index(
            "function drawCaptureWaveNodeGlow(",
        )
    ]
    intensity_scenario = f"""
const CAPTURE_WAVE_TAIL_RATIO = 0.42;
function clamp(value, minimum = 0, maximum = 1) {{
  return Math.max(minimum, Math.min(maximum, value));
}}
function smoothstep(value) {{
  const unit = clamp(value);
  return unit * unit * (3 - 2 * unit);
}}
{intensity_source}
const width = 80;
process.stdout.write(JSON.stringify({{
  peak: captureWaveNodeIntensity(100, 100, width, 1),
  innerFalloff: captureWaveNodeIntensity(60, 100, width, 1),
  outerTail: captureWaveNodeIntensity(120, 100, width, 1),
  innerEdge: captureWaveNodeIntensity(20, 100, width, 1),
  outerTailEdge: captureWaveNodeIntensity(
    100 + width * CAPTURE_WAVE_TAIL_RATIO,
    100,
    width,
    1,
  ),
  outerAfterInwardTravel: captureWaveNodeIntensity(100, 40, width, 1),
  innerAtInwardFront: captureWaveNodeIntensity(40, 40, width, 1),
  continuousInnerEdge: captureWaveNodeIntensity(20.0001, 100, width, 1),
}}));
"""
    intensity_completed = subprocess.run(
        ["node", "-e", intensity_scenario],
        check=True,
        capture_output=True,
        text=True,
    )
    intensity_result = json.loads(intensity_completed.stdout)
    assert intensity_result["peak"] == 1
    assert 0 < intensity_result["outerTail"] < intensity_result["innerFalloff"] < 1
    assert intensity_result["innerEdge"] == 0
    assert intensity_result["outerTailEdge"] == pytest.approx(0, abs=1e-12)
    assert intensity_result["outerAfterInwardTravel"] == 0
    assert intensity_result["innerAtInwardFront"] == 1
    assert 0 < intensity_result["continuousInnerEdge"] < 0.000001

    sampling_scenario = f"""
const Runtime = require({json.dumps(str(runtime_path))});
function greatestCommonDivisor(left, right) {{
  let a = Math.abs(Math.trunc(left));
  let b = Math.abs(Math.trunc(right));
  while (b) {{
    const remainder = a % b;
    a = b;
    b = remainder;
  }}
  return a;
}}
const effect = {{ captureId: "capture-sampling", label: "SAVE", seq: 41 }};
const results = Array.from({{ length: 63 }}, (_, offset) => {{
  const length = offset + 2;
  const stride = Runtime.captureWaveSampleStride(length, effect);
  const visited = new Set();
  for (let index = 0; index < length; index += 1) {{
    visited.add((index * stride) % length);
  }}
  return {{
    length,
    stride,
    divisor: greatestCommonDivisor(stride, length),
    visited: visited.size,
  }};
}});
process.stdout.write(JSON.stringify(results));
"""
    sampling_completed = subprocess.run(
        ["node", "-e", sampling_scenario],
        check=True,
        capture_output=True,
        text=True,
    )
    sampling_result = json.loads(sampling_completed.stdout)
    assert all(item["divisor"] == 1 for item in sampling_result)
    assert all(item["visited"] == item["length"] for item in sampling_result)


def test_cortex_capture_heartbeat_envelope_is_single_and_continuous() -> None:
    runtime_path = dashboard.STATIC_DIR / "cortex-runtime.js"
    scenario = f"""
const Runtime = require({json.dumps(str(runtime_path))});
process.stdout.write(JSON.stringify({{
  wave: Runtime.captureWaveState(0.5, 100, false),
  atCenter: Runtime.captureWaveState(0.72, 100, false),
  settle: Runtime.captureWaveState(0.97, 100, false),
  staticState: Runtime.captureWaveState(0.5, 100, true),
  peakStart: Runtime.captureHeartbeatPeak(0),
  peakMiddle: Runtime.captureHeartbeatPeak(0.5),
  peakBeforeEnd: Runtime.captureHeartbeatPeak(0.999999),
  peakEnd: Runtime.captureHeartbeatPeak(1),
  alphaZeroPeak: Runtime.captureHeartbeatAlpha(0, 1),
  alphaZeroFade: Runtime.captureHeartbeatAlpha(1, 0),
  alphaTiny: Runtime.captureHeartbeatAlpha(0.000001, 1),
  alphaFull: Runtime.captureHeartbeatAlpha(1, 1),
  beforeEndAlpha: Runtime.captureHeartbeatAlpha(
    Runtime.captureHeartbeatPeak(0.999999),
    1,
  ),
  atEndAlpha: Runtime.captureHeartbeatAlpha(Runtime.captureHeartbeatPeak(1), 1),
}}));
"""
    completed = subprocess.run(
        ["node", "-e", scenario],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert "heartbeatPeak" not in result["wave"]
    assert result["atCenter"]["phase"] == "settle"
    assert result["atCenter"]["wavePeak"] == 0
    assert result["settle"]["wavePeak"] == 0
    assert result["peakStart"] == 0
    assert result["peakMiddle"] == pytest.approx(1.0)
    assert 0 < result["peakBeforeEnd"] < 0.001
    assert result["peakEnd"] == 0
    assert result["alphaZeroPeak"] == 0
    assert result["alphaZeroFade"] == 0
    assert 0 < result["alphaTiny"] < 0.000001
    assert result["alphaFull"] == pytest.approx(0.46)
    assert 0 <= result["beforeEndAlpha"] < 0.001
    assert result["atEndAlpha"] == 0
    assert result["staticState"]["phase"] == "static"
    assert "heartbeatPeak" not in result["staticState"]


def test_cortex_capture_wave_train_finishes_rapid_burst_without_revival() -> None:
    script = (dashboard.STATIC_DIR / "cortex.js").read_text(encoding="utf-8")
    policy_path = dashboard.STATIC_DIR / "cortex-transport-policy.js"
    runtime_path = dashboard.STATIC_DIR / "cortex-runtime.js"
    admission_source = script[
        script.index("function scheduleCaptureHeartbeatWake(") : script.index(
            "function processingTargetNode(",
        )
    ]
    heartbeat_source = script[
        script.index("function drawCaptureHeartbeatBurst(") : script.index(
            "function captureHeartbeatHasRenderWork(",
        )
    ]
    heartbeat_work_source = script[
        script.index("function captureHeartbeatHasRenderWork(") : script.index(
            "function drawCaptureEffects(",
        )
    ]
    scenario = f"""
const policy = require({json.dumps(str(policy_path))});
const CAPTURE_WAVE_MAX_VISIBLE = 4;
const CAPTURE_WAVE_MIN_INTERVAL_MS = 420;
const CAPTURE_HEARTBEAT_SETTLE_MS = 220;
const CAPTURE_HEARTBEAT_DURATION_MS = 720;
const captureWaveTrain = [];
const captureHeartbeatBurst = {{
  finalSeq: -1,
  finalCaptureId: "",
  dueAt: 0,
  generation: 0,
  recorded: true,
  painted: false,
  wakeTimer: 0,
}};
let observedNow = 0;
let nextTimer = 1;
const timers = new Map();
const invalidations = [];
const window = {{
  setTimeout(callback, delay) {{
    const id = nextTimer++;
    timers.set(id, {{ callback, delay }});
    return id;
  }},
  clearTimeout(id) {{ timers.delete(id); }},
}};
const Runtime = require({json.dumps(str(runtime_path))});
const performance = {{ now: () => observedNow }};
function invalidate(reason) {{ invalidations.push(reason); }}
{admission_source}
const accepted = [];
const firstWaveProgress = [];
let maximumVisible = 0;
let staleWake = null;
for (let index = 0; index < 20; index += 1) {{
  observedNow = index * 120;
  const effect = {{
    captureId: `cap-${{index}}`,
    latestCaptureId: `cap-${{index}}`,
    seq: index + 1,
    startedAt: observedNow,
    duration: 3600,
    arrivalAt: observedNow + 3600 * Runtime.CAPTURE_WAVE_TRAVEL_END,
    coalescedCount: 1,
    paintedAt: 0,
  }};
  if (admitCaptureWave(effect, 1, observedNow)) accepted.push(effect);
  if (index === 0) staleWake = timers.get(captureHeartbeatBurst.wakeTimer).callback;
  maximumVisible = Math.max(maximumVisible, captureWaveTrain.length);
  firstWaveProgress.push((observedNow - accepted[0].startedAt) / 3600);
}}
const waveSnapshot = captureWaveTrain.map((wave) => ({{
  seq: wave.seq,
  startedAt: wave.startedAt,
  duration: wave.duration,
  arrivalAt: wave.arrivalAt,
  coalescedCount: wave.coalescedCount,
  latestCaptureId: wave.latestCaptureId,
}}));
const scheduledFinalSeq = captureHeartbeatBurst.finalSeq;
const scheduledDueAt = captureHeartbeatBurst.dueAt;
const pendingWakeCount = timers.size;
staleWake();
const invalidationsAfterStaleWake = invalidations.length;
const liveWake = timers.get(captureHeartbeatBurst.wakeTimer);
observedNow = scheduledDueAt;
liveWake.callback();
const invalidationsAfterLiveWake = invalidations.length;
const remainingAfterArrivals = [];
for (const wave of waveSnapshot) {{
  pruneCaptureWaveTrain(wave.arrivalAt + 0.001);
  remainingAfterArrivals.push(captureWaveTrain.length);
}}
const cortexMetrics = {{
  captureHeartbeatCount: 0,
  captureHeartbeatSeq: -1,
  captureHeartbeatPeak: 0,
  captureEffectSeq: -1,
  captureCenterX: 0,
  captureCenterY: 0,
  captureRadius: 0,
  capturePhase: "idle",
  transportElectricPeak: 0,
}};
const reducedMotion = {{ matches: false }};
let motionEnabled = true;
function drawCaptureHeartbeat(_star, peak) {{ return peak > 0; }}
{heartbeat_source}
{heartbeat_work_source}
const star = {{ x: 500, y: 300, radius: 240 }};
drawCaptureHeartbeatBurst(scheduledDueAt, star);
drawCaptureHeartbeatBurst(
  scheduledDueAt + CAPTURE_HEARTBEAT_DURATION_MS / 2,
  star,
);
drawCaptureHeartbeatBurst(
  scheduledDueAt + CAPTURE_HEARTBEAT_DURATION_MS,
  star,
);
drawCaptureHeartbeatBurst(
  scheduledDueAt + CAPTURE_HEARTBEAT_DURATION_MS + 1,
  star,
);
const firstHeartbeatCount = cortexMetrics.captureHeartbeatCount;
captureHeartbeatBurst.dueAt = scheduledDueAt + 2000;
captureHeartbeatBurst.recorded = false;
captureHeartbeatBurst.painted = false;
const boundaryFrameAt = (
  captureHeartbeatBurst.dueAt + CAPTURE_HEARTBEAT_DURATION_MS - 0.001
);
drawCaptureHeartbeatBurst(boundaryFrameAt, star);
const schedulerAfterBoundary = (
  captureHeartbeatBurst.dueAt + CAPTURE_HEARTBEAT_DURATION_MS + 0.001
);
const boundaryNeedsFinalFrame = captureHeartbeatHasRenderWork(
  schedulerAfterBoundary,
);
const boundaryRecordedBeforeFinalFrame = captureHeartbeatBurst.recorded;
drawCaptureHeartbeatBurst(schedulerAfterBoundary, star);
const boundaryHasWorkAfterFinalFrame = captureHeartbeatHasRenderWork(
  schedulerAfterBoundary,
);
const fullQueue = Array.from({{ length: 18 }}, (_, index) => ({{
  id: `protected-${{index}}`,
  phase: "generate",
  seq: index + 1,
  startedAt: 0,
  duration: 8000,
  retainedUntil: 8000 + index,
}}));
fullQueue.push({{
  id: "first-capture",
  phase: "capture",
  seq: 30,
  startedAt: 1000,
  duration: 3600,
  retainedUntil: 4600,
}});
policy.pruneAndBoundTransportEffects(fullQueue, 1000);
const zeroLimitQueue = [{{
  id: "only-capture",
  phase: "capture",
  seq: 70,
  startedAt: 1000,
  duration: 3600,
  retainedUntil: 4600,
}}];
policy.pruneAndBoundTransportEffects(zeroLimitQueue, 1000, 0);
process.stdout.write(JSON.stringify({{
  accepted: accepted.map((wave) => wave.seq),
  waveSnapshot,
  allWaveProgressAtFinalInput: waveSnapshot.map(
    (wave) => (19 * 120 - wave.startedAt) / wave.duration,
  ),
  maximumVisible,
  firstWaveProgress,
  scheduledFinalSeq,
  scheduledFinalCaptureId: captureHeartbeatBurst.finalCaptureId,
  scheduledDueAt,
  finalInputQuietUntil: 19 * 120 + CAPTURE_HEARTBEAT_SETTLE_MS,
  pendingWakeCount,
  invalidationsAfterStaleWake,
  invalidationsAfterLiveWake,
  remainingAfterArrivals,
  firstHeartbeatCount,
  heartbeatCount: cortexMetrics.captureHeartbeatCount,
  heartbeatSeq: cortexMetrics.captureHeartbeatSeq,
  finalRecorded: captureHeartbeatBurst.recorded,
  boundaryNeedsFinalFrame,
  boundaryRecordedBeforeFinalFrame,
  boundaryHasWorkAfterFinalFrame,
  fullQueueLength: fullQueue.length,
  firstCaptureRetained: fullQueue.some((effect) => effect.id === "first-capture"),
  firstProtectedEvicted: !fullQueue.some((effect) => effect.id === "protected-0"),
  zeroLimitLength: zeroLimitQueue.length,
}}));
"""
    completed = subprocess.run(
        ["node", "-e", scenario],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["accepted"] == [1, 5, 9, 13]
    assert result["maximumVisible"] == 4
    assert len(result["waveSnapshot"]) == 4
    assert result["waveSnapshot"][0]["startedAt"] == 0
    assert [wave["startedAt"] for wave in result["waveSnapshot"]] == [
        0,
        480,
        960,
        1440,
    ]
    assert result["allWaveProgressAtFinalInput"] == pytest.approx(
        [2280 / 3600, 1800 / 3600, 1320 / 3600, 840 / 3600]
    )
    assert result["waveSnapshot"][-1]["latestCaptureId"] == "cap-19"
    assert all(
        later >= earlier
        for earlier, later in zip(
            result["firstWaveProgress"],
            result["firstWaveProgress"][1:],
            strict=False,
        )
    )
    assert result["scheduledFinalSeq"] == 20
    assert result["scheduledFinalCaptureId"] == "cap-19"
    assert result["scheduledDueAt"] == pytest.approx(4252)
    assert result["scheduledDueAt"] >= result["finalInputQuietUntil"]
    assert result["pendingWakeCount"] == 1
    assert result["invalidationsAfterStaleWake"] == 0
    assert result["invalidationsAfterLiveWake"] == 1
    assert result["remainingAfterArrivals"] == [3, 2, 1, 0]
    assert result["firstHeartbeatCount"] == 1
    assert result["heartbeatCount"] == 2
    assert result["heartbeatSeq"] == 20
    assert result["finalRecorded"] is True
    assert result["boundaryNeedsFinalFrame"] is True
    assert result["boundaryRecordedBeforeFinalFrame"] is False
    assert result["boundaryHasWorkAfterFinalFrame"] is False
    assert result["fullQueueLength"] == 18
    assert result["firstCaptureRetained"] is True
    assert result["firstProtectedEvicted"] is True
    assert result["zeroLimitLength"] == 0


def test_cortex_processing_lanes_monitor_has_fixed_independent_routing() -> None:
    static = dashboard.STATIC_DIR
    html = (static / "cortex.html").read_text(encoding="utf-8")
    style = (static / "cortex.css").read_text(encoding="utf-8")
    script = (static / "cortex.js").read_text(encoding="utf-8")
    policy_script = (static / "cortex-transport-policy.js").read_text(
        encoding="utf-8"
    )

    lane_keys = (
        "raw_buffer",
        "ingest",
        "recall",
        "audit",
        "improve",
        "repair",
        "typed_graph",
    )
    assert html.count('class="processingLane"') == len(lane_keys)
    assert html.count('class="processingLaneDetail"') == len(lane_keys)
    for lane_key in lane_keys:
        assert f'id="processingLane-{lane_key}"' in html
        assert f'data-lane="{lane_key}"' in html
    for label in (
        "RAW BUFFER",
        "INGEST",
        "RECALL",
        "AUDIT",
        "IMPROVE",
        "REPAIR",
        "TYPED GRAPH",
    ):
        assert f'<span class="processingLaneName">{label}</span>' in html

    monitor_start = html.index('<section id="processingLanesMonitor"')
    monitor_end = html.index("</section>", monitor_start)
    panel_body = html.index('<div id="panelBody"></div>')
    assert monitor_start < monitor_end < panel_body
    assert 'aria-label="Processing lanes monitor"' in html
    assert 'id="processingLanesAnnounce"' in html
    assert 'aria-live="polite"' in html
    assert 'aria-atomic="true"' in html
    assert 'id="memoryIngress"' not in html
    assert "#memoryIngress" not in style
    assert "updateMemoryIngress" not in script

    assert "#processingLanesMonitor {" in style
    assert "flex: none;" in style[style.index("#processingLanesMonitor {") :]
    assert "#panelBody {" in style
    assert "min-height: 0;" in style[style.index("#panelBody {") :]
    assert "--lane-color: #61708a;" in style
    assert "border-left: 2px solid #344057;" in style
    assert '.processingLane[data-state="active"]' in style
    assert '.processingLane[data-state="complete"]' in style
    assert '.processingLane[data-phase="capture"] { --lane-color: #4fe4ff; }' in style
    assert '.processingLane[data-phase="triage"] { --lane-color: #ffb454; }' in style
    assert '.processingLane[data-phase="generate"] { --lane-color: #9b7cff; }' in style
    assert (
        '.processingLane[data-phase="consensus"] { --lane-color: #ffe6ae; }'
        in style
    )
    assert '.processingLane[data-phase="complete"] { --lane-color: #45d49b; }' in style

    lane_style = style[
        style.index(".processingLane {") : style.index(".processingLane::after {")
    ]
    assert "display: grid;" in lane_style
    assert "grid-template-rows: 11px 9px;" in lane_style
    assert "row-gap: 1px;" in lane_style
    assert "box-sizing: border-box;" in lane_style
    assert "height: 30px;" in lane_style
    assert "padding: 4px 7px 3px;" in lane_style
    assert "min-height:" not in lane_style

    detail_style = style[
        style.index(".processingLaneDetail {") : style.index(
            '.processingLane[data-phase="capture"]'
        )
    ]
    assert "visibility: hidden;" in detail_style
    assert "min-width: 0;" in detail_style
    assert "overflow: hidden;" in detail_style
    assert "text-overflow: ellipsis;" in detail_style
    assert "white-space: nowrap;" in detail_style
    assert "font: 7.5px/9px var(--mono);" in detail_style
    assert "display: none;" not in detail_style
    assert (
        '.processingLane[data-state="active"] .processingLaneDetail,\n'
        '.processingLane[data-state="complete"] .processingLaneDetail {' in style
    )
    assert (
        "visibility: visible;"
        in style[
            style.index(
                '.processingLane[data-state="active"] .processingLaneDetail,'
            ) : style.index('.processingLane[data-state="complete"]::after')
        ]
    )

    assert "const processingLaneMonitorStates = new Map(" in script
    assert "PROCESSING_LANE_KEYS.map((laneKey) =>" in script
    assert "revision: 0," in script
    assert "resetTimer: 0," in script
    assert 'mode: "idle",' in script
    assert "monitorState.revision += 1;" in script
    assert "window.clearTimeout(monitorState.resetTimer);" in script
    assert "revision !== monitorState.revision" in script
    assert "resetProcessingLaneMonitor(laneKey, revision)" in script
    assert "projection.holdMs" in script
    assert "const PROCESSING_LANE_COMPLETE_HOLD_MS = 1800" in policy_script
    assert "const PROCESSING_LANE_ACTIVE_HOLD_MS = 4800" in policy_script

    assert 'kind === "save" || kind === "capture"' in policy_script
    assert 'return "raw_buffer";' in policy_script
    assert 'if (kind === "ingest") return "ingest";' in policy_script
    assert "if (isRecallKind(kind)) return \"recall\";" in policy_script
    assert 'PROCESSING_ACTIVITY_LANE_KEY_SET.has(laneKey) ? laneKey : "audit"' in policy_script
    assert 'if (kind === "search") return "triage";' in policy_script
    assert 'if (kind === "used") return "apply";' in policy_script
    assert '["recall", "auto_recall", "read"].includes(kind)' in policy_script
    assert 'event.kind === "processing"' in policy_script
    assert '.replaceAll("_", " ")' in policy_script
    assert "[event.model, event.role]" in policy_script
    assert "formatBytes(event.byte_count)" in policy_script
    assert "function renderProcessingLaneMonitor(" in script
    assert 'row.querySelector(".processingLaneStatus").textContent = status;' in script
    assert 'row.querySelector(".processingLaneDetail").textContent = detail;' in script
    assert "const shouldAnnounce =" in script
    assert "if (announce && shouldAnnounce)" in script
    assert 'processingLaneEvent(state.lane, "complete")' in script
    assert "updateProcessingLaneMonitor(event, phase);" in script
    assert "generation !== demoTransportGeneration" in script
    assert "demo_sequence_started_at: sequence.startedAt" in script
    assert "TransportPolicy.processingLaneUpdateDecision(" in script
    assert "monitorState.lastLiveAt = updateDecision.lastLiveAt;" in script
    assert 'if (event.mode === "live") clearDemoTransportTimers();' not in script
    assert "TransportPolicy.normalizeEvent(event);" in script


def test_cortex_sleeps_only_layout_physics_and_keeps_rendering_live() -> None:
    script = (dashboard.STATIC_DIR / "cortex.js").read_text(encoding="utf-8")

    assert "const SIMULATION_ALPHA_FLOOR = 0.012;" in script
    assert "const SIMULATION_SLEEP_VELOCITY = 0.02;" in script
    assert "const SIMULATION_SLEEP_TICKS = 36;" in script
    assert "const SIMULATION_MAX_STEPS_PER_FRAME = 4;" in script
    assert "function sleepSimulation()" in script
    assert "simulationAwake = false;" in script
    assert "simulationAccumulator = 0;" in script
    assert "nodes[index].vx = 0;" in script
    assert "simulationAwake = true;" in script
    assert "if (simulationAwake) {" in script
    assert "simulationSteps < SIMULATION_MAX_STEPS_PER_FRAME" in script
    assert (
        "simulationLinks = links.filter((link) => link.kind !== 2 && !link.typed);"
        in script
    )
    ensure_edge = script[
        script.index("function ensureActualEdge(") : script.index(
            "function syncFieldNodes("
        )
    ]
    assert "simulationLinks.push" not in ensure_edge
    assert "EVENT_EDGE_TTL_MS" in ensure_edge

    frame = script[
        script.index("function frame(\n") : script.index("function pick(")
    ]
    assert "tick();" in frame
    assert "projectAll();" in frame
    assert "draw(now, webglBaseRendered);" in frame
    assert "baseRenderer.render({" in frame
    assert frame.index("tick();") < frame.index("projectAll();")
    assert "Runtime.createRenderScheduler" in script


def test_cortex_reuses_hot_path_state_without_reducing_visual_limits() -> None:
    script = (dashboard.STATIC_DIR / "cortex.js").read_text(encoding="utf-8")

    assert "const METRICS_PUBLISH_INTERVAL_MS = 200;" in script
    assert "now - lastCortexMetricsPublished < METRICS_PUBLISH_INTERVAL_MS" in script
    assert "const FRAME_DURATION_CAPACITY = 240;" in script
    assert "function recentFrameDurations(" in script
    assert "function recentFrameCadences(" in script
    assert "frameDurationCursor = (frameDurationCursor + 1)" in script
    assert "performance.now() - frameWorkStartedAt" in script
    assert "frameState.sinceLastRender !== null" in script
    assert "frameState.continuation" in script
    assert "|| cadence <= FRAME_CADENCE_IDLE_GAP_MS" in script
    assert "1000 / cadenceMean" in script
    assert "Runtime.createCooldownWake" in script
    assert "const labelCandidates = [];" in script
    assert "const activeLabelNodes = [];" in script
    assert "const occupiedLabels = [];" in script
    assert "function collectLabelCandidates(time)" in script
    assert "labelCandidateMarks = new Uint32Array(nodeCount);" in script
    assert "context.font = labelFont;" in script
    assert "community.points = [];" in script
    assert "neighborsByConnectivity = neighbors.map(" in script
    assert "neighborsByFanIn = neighbors.map(" in script
    assert "if (!memoryStar) memoryStar = memoryStarGeometry();" in script
    assert "transportFallbacks.length = 0;" in script
    assert "const ACTIVE_LABEL_LIMIT = 5;" in script
    assert "const MAX_ELECTRIC_PATHS = TransportPolicy.MAX_ELECTRIC_PATHS;" in script
    policy_script = (dashboard.STATIC_DIR / "cortex-transport-policy.js").read_text(
        encoding="utf-8"
    )
    assert "const MAX_TRANSPORT_EFFECTS = 18;" in policy_script


def test_cortex_runtime_sphere_layout_is_deterministic_tiered_and_finite() -> None:
    runtime_path = dashboard.STATIC_DIR / "cortex-runtime.js"
    scenario = f"const runtime = require({json.dumps(str(runtime_path))});\n" + r"""
const nodes = [
  { id: "current-state", packageName: "system", entrypoint: 1, fanIn: 30, fanOut: 5 },
  { id: "hub", packageName: "alpha", fanIn: 80, fanOut: 40 },
  { id: "alpha-a", packageName: "alpha", fanIn: 2, fanOut: 1 },
  { id: "alpha-b", packageName: "alpha", fanIn: 1, fanOut: 1 },
  { id: "beta-a", packageName: "beta", fanIn: 3, fanOut: 2 },
  { id: "beta-b", packageName: "beta", fanIn: 0, fanOut: 1 },
];
const first = runtime.createSphereTargets(nodes);
const second = runtime.createSphereTargets(nodes.map((node) => ({ ...node })));
const reversedNodes = [...nodes].reverse();
const reversed = runtime.createSphereTargets(reversedNodes);
const byId = Object.fromEntries(nodes.map((node, index) => [node.id, first[index]]));
const reversedById = Object.fromEntries(
  reversedNodes.map((node, index) => [node.id, reversed[index]]),
);
const positioned = first.map((target) => ({
  x: target.x,
  y: target.y,
  z: target.z,
}));
const exactQuality = runtime.measureSphereQuality(positioned, first);
const opposite = first.map((target) => ({
  x: -target.x,
  y: -target.y,
  z: -target.z,
}));
const oppositeQuality = runtime.measureSphereQuality(opposite, first);
const largeNodes = Array.from({ length: 3863 }, (_, index) => ({
  id: `node-${String(index).padStart(4, "0")}`,
  packageName: `package-${index % 17}`,
  entrypoint: index < 4 ? 1 : 0,
  fanIn: (index * 37) % 113,
  fanOut: (index * 19) % 71,
}));
const largeTargets = runtime.createSphereTargets(largeNodes);
const largeQuality = runtime.measureSphereQuality(largeTargets, largeTargets);
const singleton = runtime.createSphereTargets([
  { id: "only", packageName: "single", fanIn: 0, fanOut: 0 },
])[0];
const largeFinite = largeTargets.every((target) =>
  [target.x, target.y, target.z, target.radius].every(Number.isFinite)
);
const radii = Object.fromEntries(
  ["core", "middle", "outer"].map((tier) => {
    const values = first.filter((target) => target.tier === tier).map((target) => target.radius);
    return [tier, { min: Math.min(...values), max: Math.max(...values), count: values.length }];
  }),
);
const storage = new Map();
storage.set("view", JSON.stringify({ mode: runtime.normalizeLayoutMode("sphere") }));
const reloadedMode = runtime.normalizeLayoutMode(JSON.parse(storage.get("view")).mode);
storage.delete("view");
const resetMode = runtime.normalizeLayoutMode(
  storage.has("view") ? JSON.parse(storage.get("view")).mode : undefined,
);
process.stdout.write(JSON.stringify({
  deterministic: JSON.stringify(first) === JSON.stringify(second),
  orderIndependent: nodes.every((node) => {
    const left = byId[node.id];
    const right = reversedById[node.id];
    return left.tier === right.tier
      && left.radius === right.radius
      && Math.hypot(left.x - right.x, left.y - right.y, left.z - right.z) < 1e-9;
  }),
  finite: first.every((target) =>
    [target.x, target.y, target.z, target.radius].every(Number.isFinite)
  ),
  tiers: exactQuality.tiers,
  radii,
  exactQuality,
  oppositeQuality,
  largeCount: largeTargets.length,
  largeFinite,
  largeQuality,
  singleton,
  modes: [
    runtime.normalizeLayoutMode("organic"),
    runtime.normalizeLayoutMode("sphere"),
    runtime.normalizeLayoutMode("cluster"),
    runtime.normalizeLayoutMode("invalid"),
    reloadedMode,
    resetMode,
  ],
}));
"""
    completed = subprocess.run(
        ["node", "-e", scenario],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["deterministic"] is True
    assert result["orderIndependent"] is True
    assert result["finite"] is True
    assert all(result["tiers"][tier] > 0 for tier in ("core", "middle", "outer"))
    assert result["radii"]["core"]["max"] < result["radii"]["middle"]["min"]
    assert result["radii"]["middle"]["max"] < result["radii"]["outer"]["min"]
    assert result["exactQuality"]["targetError"]["max"] == pytest.approx(0)
    assert result["exactQuality"]["radialError"]["max"] == pytest.approx(0)
    assert result["oppositeQuality"]["targetError"]["mean"] > 300
    assert result["oppositeQuality"]["radialError"]["max"] == pytest.approx(0)
    assert result["largeCount"] == 3863
    assert result["largeFinite"] is True
    assert result["largeQuality"]["targetCentroid"]["normalizedOffset"] <= 0.05
    assert result["largeQuality"]["occupiedOctants"] == 8
    singleton_radius = (
        result["singleton"]["x"] ** 2
        + result["singleton"]["y"] ** 2
        + result["singleton"]["z"] ** 2
    ) ** 0.5
    assert singleton_radius == pytest.approx(result["singleton"]["radius"])
    assert singleton_radius > 100
    assert result["modes"] == [
        "organic",
        "sphere",
        "cluster",
        "organic",
        "sphere",
        "organic",
    ]


def test_cortex_runtime_sphere_camera_fit_and_fog_are_bounded() -> None:
    runtime_path = dashboard.STATIC_DIR / "cortex-runtime.js"
    scenario = f"""
const runtime = require({json.dumps(str(runtime_path))});
const cases = [
  {{ viewportWidth: 1024, viewportHeight: 716, topInset: 126, padding: 28,
     sphereRadius: 562, focalLength: 716 * 1.12, minimumDistance: 600 }},
  {{ viewportWidth: 680, viewportHeight: 448, topInset: 130, padding: 28,
     sphereRadius: 562, focalLength: 448 * 1.12, minimumDistance: 600 }},
  {{ viewportWidth: 375, viewportHeight: 320, topInset: 164, padding: 28,
     sphereRadius: 562, focalLength: 320 * 1.12, minimumDistance: 600 }},
  {{ viewportWidth: 320, viewportHeight: 320, topInset: 178, padding: 28,
     sphereRadius: 562, focalLength: 320 * 1.12, minimumDistance: 600 }},
];
const fits = cases.map((options) => runtime.fitSphereCamera(options));
const fog = [1, 0.74, 0.49, 0.24, 0].map((depthFade) => {{
  const band = runtime.sphereFogBand(depthFade);
  return {{ depthFade, band, opacity: runtime.sphereFogOpacity(band) }};
}});
process.stdout.write(JSON.stringify({{ fits, fog }}));
"""
    completed = subprocess.run(
        ["node", "-e", scenario],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    for fit in result["fits"]:
        projected_radius = (
            fit["focalLength"]
            * fit["sphereRadius"]
            / math.sqrt(fit["distance"] ** 2 - fit["sphereRadius"] ** 2)
        )
        assert math.isfinite(fit["distance"])
        assert fit["distance"] > fit["sphereRadius"]
        assert fit["projectedRadius"] == pytest.approx(projected_radius)
        assert fit["projectedRadius"] <= fit["safeRadius"] + 1e-8
        assert (
            fit["projectedBounds"]["left"]
            >= fit["safeBounds"]["left"] - 1e-8
        )
        assert (
            fit["projectedBounds"]["right"]
            <= fit["safeBounds"]["right"] + 1e-8
        )
        assert (
            fit["projectedBounds"]["top"]
            >= fit["safeBounds"]["top"] - 1e-8
        )
        assert (
            fit["projectedBounds"]["bottom"]
            <= fit["safeBounds"]["bottom"] + 1e-8
        )

    assert [bucket["band"] for bucket in result["fog"]] == [0, 1, 2, 3, 3]
    assert [bucket["opacity"] for bucket in result["fog"]] == [
        0.875,
        0.625,
        0.375,
        0.2,
        0.2,
    ]
    assert result["fog"][-1]["opacity"] < result["fog"][0]["opacity"] * 0.25


def test_cortex_runtime_pulse_scheduler_metrics_burst_and_webgl_fallback() -> None:
    runtime_path = dashboard.STATIC_DIR / "cortex-runtime.js"
    webgl_path = dashboard.STATIC_DIR / "cortex-webgl.js"
    scenario = (
        f"const runtime = require({json.dumps(str(runtime_path))});\n"
        f"const webgl = require({json.dumps(str(webgl_path))});\n"
        + r"""
const pulses = [
  { id: "A", expired: true },
  { id: "B", expired: true },
  { id: "C", expired: true },
  { id: "D", expired: false },
];
const completed = [];
const completionCount = runtime.drainExpiredPulses(
  pulses,
  (pulse) => pulse.expired,
  (pulse) => {
    completed.push(pulse.id);
    // A callback may touch the already-compacted queue without invalidating
    // the original walk; complete callbacks are not nested in that walk.
    pulses.reverse();
  },
);

// Browser frame identifiers may legally wrap to zero. The scheduler must
// still treat id 0 as an outstanding request and coalesce invalidations.
let nextFrame = 0;
const frames = new Map();
let hidden = false;
    let active = false;
    let rendered = 0;
    const frameContinuations = [];
    const frameIntervals = [];
const scheduler = runtime.createRenderScheduler({
  requestFrame(callback) {
    const id = nextFrame++;
    frames.set(id, callback);
    return id;
  },
  cancelFrame(id) { frames.delete(id); },
  isHidden: () => hidden,
  onFrame: (_now, _reasons, frameState) => {
    rendered += 1;
    frameContinuations.push(frameState.continuation);
    frameIntervals.push(frameState.sinceLastRender);
  },
  hasWork: () => active,
});
function runNext(now) {
  const [id, callback] = frames.entries().next().value;
  frames.delete(id);
  callback(now);
}
scheduler.invalidate("one");
scheduler.invalidate("two");
scheduler.invalidate("three");
const coalescedFrames = frames.size;
runNext(10);
const idleFrames = frames.size;
active = true;
scheduler.invalidate("effect");
runNext(20);
const activeFrames = frames.size;
active = false;
runNext(30);
hidden = true;
scheduler.invalidate("hidden");
const hiddenFrames = frames.size;
hidden = false;
scheduler.visibilityChanged();
const wakeFrames = frames.size;
runNext(40);
scheduler.invalidate("pointer-after-idle");
runNext(1000);
scheduler.invalidate("adjacent-wheel");
runNext(1016);
active = true;
scheduler.invalidate("long-work-start");
runNext(1026);
active = false;
runNext(1400);

const ring = runtime.createDurationRing(3);
[100, 1, 2, 3].forEach((value) => ring.record(value));
const ringSnapshot = ring.snapshot();

let overflowed = 0;
const eventQueue = runtime.createEventQueue({
  maximum: 256,
  protectedEvent: (event) => event.family === "field",
  onOverflow: () => { overflowed += 1; },
});
eventQueue.push(Array.from({ length: 100 }, (_, index) => ({
  family: "field",
  seq: index + 1,
})));
const drainSizes = [];
const drainedSeqs = [];
while (eventQueue.state().length) {
  const batch = eventQueue.drain(32);
  drainSizes.push(batch.length);
  drainedSeqs.push(...batch.map((event) => event.seq));
}
const overflowQueue = runtime.createEventQueue({
  maximum: 2,
  protectedEvent: () => true,
  onOverflow: () => { overflowed += 1; },
});
overflowQueue.push([
  { family: "field", seq: 1 },
  { family: "field", seq: 2 },
  { family: "field", seq: 3 },
]);

const listeners = new Map();
const removed = [];
const rendererStates = [];
const canvas = {
  style: {},
  dataset: {},
  getContext: () => null,
  addEventListener: (name, callback) => listeners.set(name, callback),
  removeEventListener: (name, callback) => {
    if (listeners.get(name) === callback) listeners.delete(name);
    removed.push(name);
  },
};
const renderer = webgl.createRenderer(canvas, {
  onStateChange: (state) => rendererStates.push(state),
});
const fallbackRendered = renderer.render({ width: 10, height: 10 });
let prevented = 0;
listeners.get("webglcontextlost")({ preventDefault: () => { prevented += 1; } });
listeners.get("webglcontextlost")({ preventDefault: () => { prevented += 1; } });
renderer.dispose();

process.stdout.write(JSON.stringify({
  completionCount,
  completed,
  remainingPulses: pulses.map((pulse) => pulse.id),
  coalescedFrames,
  idleFrames,
  activeFrames,
  hiddenFrames,
  wakeFrames,
  rendered,
  frameContinuations,
  frameIntervals,
  scheduler: scheduler.state(),
  ring: ringSnapshot,
  drainSizes,
  drainedSeqs,
  overflowed,
  fallbackRendered,
  rendererStates,
  rendererMode: renderer.snapshot().mode,
  rendererOpacity: canvas.style.opacity,
  rendererState: canvas.dataset.rendererState,
  prevented,
  remainingListeners: listeners.size,
  removed,
}));
"""
    )
    completed = subprocess.run(
        ["node", "-e", scenario],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["completionCount"] == 3
    assert result["completed"] == ["A", "B", "C"]
    assert result["remainingPulses"] == ["D"]
    assert result["coalescedFrames"] == 1
    assert result["idleFrames"] == 0
    assert result["activeFrames"] == 1
    assert result["hiddenFrames"] == 0
    assert result["wakeFrames"] == 1
    assert result["rendered"] == 8
    assert result["frameContinuations"] == [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
    ]
    assert result["frameIntervals"] == [None, 10, 10, 10, 960, 16, 10, 374]
    assert result["scheduler"]["pending"] == 0
    assert result["ring"]["samples"] == [1, 2, 3]
    assert result["ring"]["mean"] == 2
    assert result["ring"]["max"] == 3
    assert result["ring"]["lifetimeMax"] == 100
    assert result["drainSizes"] == [32, 32, 32, 4]
    assert result["drainedSeqs"] == list(range(1, 101))
    assert result["overflowed"] == 1
    assert result["fallbackRendered"] is False
    assert result["rendererStates"] == ["fallback", "lost"]
    assert result["rendererMode"] == "canvas2d"
    assert result["rendererOpacity"] == "0"
    assert result["rendererState"] == "lost"
    assert result["prevented"] == 2
    assert result["remainingListeners"] == 0
    assert sorted(result["removed"]) == [
        "webglcontextlost",
        "webglcontextrestored",
    ]


def test_cortex_runtime_cooldown_spatial_pick_and_transport_backpressure() -> None:
    runtime_path = dashboard.STATIC_DIR / "cortex-runtime.js"
    scenario = f"const runtime = require({json.dumps(str(runtime_path))});\n" + r"""
let observedNow = 0;
let lastInteraction = 0;
let enabled = true;
let nextTimer = 0;
let wakes = 0;
const timers = new Map();
const cooldown = runtime.createCooldownWake({
  now: () => observedNow,
  setTimer(callback, delay) {
    const id = nextTimer++;
    timers.set(id, { callback, delay });
    return id;
  },
  clearTimer(id) { timers.delete(id); },
  readyAt: () => lastInteraction + 2600,
  isEnabled: () => enabled,
  isHidden: () => false,
  onReady: () => { wakes += 1; },
});
const initiallyReady = cooldown.sync();
const zeroTimerPending = cooldown.state().pending;
const staleCallback = timers.get(0).callback;
lastInteraction = 1000;
observedNow = 100;
cooldown.sync();
const rescheduledDelay = timers.get(1).delay;
staleCallback();
const wakesAfterStale = wakes;
observedNow = 3600;
timers.get(1).callback();
const readyAfterCooldown = cooldown.ready();
lastInteraction = 4000;
observedNow = 4000;
cooldown.sync();
const disposedCallback = timers.get(2).callback;
cooldown.dispose();
disposedCallback();

const spatial = runtime.createSpatialIndex(48);
const nodes = [
  { screenX: 0, screenY: 0, hitRadius: 120 },
  { screenX: 500, screenY: 0, hitRadius: 8 },
];
spatial.rebuild(nodes, () => true, (node) => node.hitRadius);
const picked = spatial.pick(
  110,
  0,
  nodes,
  (node, x, y) => Math.hypot(node.screenX - x, node.screenY - y) - node.hitRadius,
);

let fieldResyncs = 0;
const fieldQueue = runtime.createEventQueue({
  maximum: 256,
  protectedEvent: () => true,
  onOverflow: () => { fieldResyncs += 1; },
});
const transportQueue = runtime.createEventQueue({
  maximum: 256,
  protectedEvent: () => true,
});
fieldQueue.push(Array.from({ length: 257 }, (_, index) => ({
  family: "field",
  seq: index + 1,
})));
transportQueue.push([{ family: "transport", kind: "save", capture_id: "cap-1" }]);
if (fieldResyncs) fieldQueue.clear();
const transportVisible = transportQueue.drain(32);
const transportAfterVisible = transportQueue.drain(32);

const typedQueue = runtime.createEventQueue({
  maximum: 3,
  protectedEvent: () => true,
  overflowCoalesceKey: (event) => [event.kind, event.phase, event.channel_key].join(":"),
  mergeOverflow: (previous, event) => ({
    ...event,
    coalesced_count: (previous.coalesced_count || 1) + 1,
    page_ids: [...new Set([...(previous.page_ids || []), ...(event.page_ids || [])])],
  }),
});
typedQueue.push([
  { family: "transport", kind: "save", phase: "capture", channel_key: "save", page_ids: ["save-a"] },
  { family: "transport", kind: "read", phase: "generate", channel_key: "read", page_ids: ["read-a"] },
  { family: "transport", kind: "used", phase: "apply", channel_key: "used", page_ids: ["used-a"] },
]);
typedQueue.push([
  { family: "transport", kind: "save", phase: "capture", channel_key: "save", page_ids: ["save-b"], capture_id: "latest-save" },
  { family: "transport", kind: "read", phase: "generate", channel_key: "read", page_ids: ["read-b"] },
  { family: "transport", kind: "used", phase: "apply", channel_key: "used", page_ids: ["used-b"] },
]);
const typedState = typedQueue.state();
const typedEvents = typedQueue.drain(3);

const captureOverflowQueue = runtime.createEventQueue({
  maximum: 256,
  protectedEvent: () => true,
  overflowCoalesceKey: (event) => [
    event.kind,
    event.phase,
    event.channel_key,
  ].join(":"),
  mergeOverflow: (previous, event) => ({
    ...event,
    coalesced_count:
      (previous.coalesced_count || 1) + (event.coalesced_count || 1),
  }),
});
captureOverflowQueue.push(Array.from({ length: 300 }, (_, index) => ({
  family: "transport",
  kind: "save",
  phase: "capture",
  channel_key: "save",
  capture_id: `cap-${index}`,
  coalesced_count: 1,
})));
const captureOverflowState = captureOverflowQueue.state();
const captureOverflowEvents = [];
while (captureOverflowQueue.state().length) {
  captureOverflowEvents.push(...captureOverflowQueue.drain(32));
}

const gate = runtime.createGenerationGate();
const gateDecisions = [
  gate.accept(7, "field_snapshot_corrupt"),
  gate.accept(7, "field_snapshot_corrupt"),
  gate.accept(8, "field_snapshot_corrupt"),
];

process.stdout.write(JSON.stringify({
  initiallyReady,
  zeroTimerPending,
  rescheduledDelay,
  wakesAfterStale,
  wakes,
  readyAfterCooldown,
  cooldownState: cooldown.state(),
  picked,
  maximumHitRadius: spatial.maximumHitRadius(),
  fieldResyncs,
  remainingField: fieldQueue.state().length,
  transportVisible,
  transportAfterVisible,
  typedState,
  typedEvents,
  captureOverflowState,
  captureOverflowIds: captureOverflowEvents.map((event) => event.capture_id),
  captureOverflowMultiplicity: captureOverflowEvents.reduce(
    (total, event) => total + event.coalesced_count,
    0,
  ),
  gateDecisions,
}));
"""
    completed = subprocess.run(
        ["node", "-e", scenario],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["initiallyReady"] is False
    assert result["zeroTimerPending"] == 1
    assert result["rescheduledDelay"] == 3500
    assert result["wakesAfterStale"] == 0
    assert result["wakes"] == 1
    assert result["readyAfterCooldown"] is True
    assert result["cooldownState"]["pending"] == 0
    assert result["cooldownState"]["disposed"] is True
    assert result["picked"] == 0
    assert result["maximumHitRadius"] == 120
    assert result["fieldResyncs"] == 1
    assert result["remainingField"] == 0
    assert result["transportVisible"] == [
        {"family": "transport", "kind": "save", "capture_id": "cap-1"}
    ]
    assert result["transportAfterVisible"] == []
    assert result["typedState"] == {
        "length": 3,
        "maximum": 3,
        "overflowCount": 3,
        "droppedCount": 0,
        "coalescedCount": 3,
    }
    assert [event["kind"] for event in result["typedEvents"]] == [
        "save",
        "read",
        "used",
    ]
    assert all(event["coalesced_count"] == 2 for event in result["typedEvents"])
    assert result["typedEvents"][0]["capture_id"] == "latest-save"
    assert result["typedEvents"][0]["page_ids"] == ["save-a", "save-b"]
    assert result["captureOverflowState"] == {
        "length": 256,
        "maximum": 256,
        "overflowCount": 44,
        "droppedCount": 0,
        "coalescedCount": 44,
    }
    assert result["captureOverflowIds"] == [
        f"cap-{index}" for index in range(44, 300)
    ]
    assert result["captureOverflowIds"][-1] == "cap-299"
    assert result["captureOverflowMultiplicity"] == 300
    assert result["gateDecisions"] == [True, False, True]


def test_cortex_webgl_batches_base_nodes_edges_and_typed_relations() -> None:
    webgl_path = dashboard.STATIC_DIR / "cortex-webgl.js"
    scenario = f"const webgl = require({json.dumps(str(webgl_path))});\n" + r"""
const draws = [];
const uploads = [];
let throwDraw = false;
let drawInvocations = 0;
const gl = {
  VERTEX_SHADER: 1,
  FRAGMENT_SHADER: 2,
  COMPILE_STATUS: 3,
  LINK_STATUS: 4,
  ARRAY_BUFFER: 5,
  FLOAT: 6,
  DYNAMIC_DRAW: 7,
  BLEND: 8,
  SRC_ALPHA: 9,
  ONE_MINUS_SRC_ALPHA: 10,
  COLOR_BUFFER_BIT: 11,
  LINES: 12,
  POINTS: 13,
  createShader: () => ({}),
  shaderSource() {},
  compileShader() {},
  getShaderParameter: () => true,
  getShaderInfoLog: () => "",
  deleteShader() {},
  createProgram: () => ({}),
  attachShader() {},
  linkProgram() {},
  getProgramParameter: () => true,
  getProgramInfoLog: () => "",
  deleteProgram() {},
  createBuffer: () => ({}),
  getAttribLocation: (_program, name) => ({
    a_position: 0,
    a_color: 1,
    a_size: 2,
  })[name],
  getUniformLocation: () => ({}),
  useProgram() {},
  bindBuffer() {},
  enableVertexAttribArray() {},
  vertexAttribPointer() {},
  enable() {},
  blendFunc() {},
  bufferData() {},
  bufferSubData(_target, _offset, data) { uploads.push(Array.from(data)); },
  viewport() {},
  clearColor() {},
  clear() {},
  uniform1i() {},
  drawArrays(mode, first, count) {
    draws.push([mode, first, count]);
    drawInvocations += 1;
    if (throwDraw) throw new Error("runtime draw failure");
  },
  deleteBuffer() {},
};
const listeners = new Map();
const states = [];
const stateVisuals = [];
const canvas = {
  style: {},
  dataset: {},
  getContext: (name) => name === "webgl2" ? gl : null,
  addEventListener: (name, callback) => listeners.set(name, callback),
  removeEventListener: (name) => listeners.delete(name),
};
const renderer = webgl.createRenderer(canvas, {
  onStateChange: (state) => {
    states.push(state);
    stateVisuals.push([state, canvas.style.opacity]);
  },
});
renderer.resize(800, 600, 2);
const nodes = [
  { screenX: 100, screenY: 100, viewDepth: 1000, radius: 3, screenScale: 1, base: [100, 120, 140], fieldActivation: 0 },
  { screenX: 200, screenY: 200, viewDepth: 1100, radius: 4, screenScale: 1, base: [100, 120, 140], fieldActivation: 0 },
];
const scene = {
  now: 10,
  width: 800,
  height: 600,
  pixelRatio: 2,
  nodes,
  excitations: new Float32Array([0.2, 0.6]),
  links: [
    { source: 0, target: 1, typed: false },
    { source: 1, target: 0, typed: true, lifecycle: "verified", relationId: "r1" },
  ],
  nodeState: new Uint8Array([2, 2]),
  edgeState: new Uint8Array([2, 2]),
  activeRelations: new Set(),
  relationsVisible: true,
  edgeVisibility: 1,
  mode: "sphere",
  normalEdgeScale: 0.28,
  selected: 0,
  hovered: 1,
  fog: () => 1,
};
const rendered = renderer.render(scene);
const edgeUpload = uploads[0];
const nodeUpload = uploads[1];
throwDraw = true;
const failedRender = renderer.render(scene);
const failureSnapshot = renderer.snapshot();
const drawsAfterFailure = drawInvocations;
const repeatedFailureRender = renderer.render(scene);
const drawsAfterRepeatedFailure = drawInvocations;
throwDraw = false;
const retryReady = renderer.retry();
const recoveredRender = renderer.render(scene);
const beforeLoss = renderer.snapshot();
listeners.get("webglcontextlost")({ preventDefault() {} });
const duringLoss = renderer.render({ width: 800, height: 600 });
listeners.get("webglcontextrestored")();
const afterRestore = renderer.snapshot();
renderer.dispose();
process.stdout.write(JSON.stringify({
  rendered,
  edgeUpload,
  nodeUpload,
  failedRender,
  failureSnapshot,
  drawsAfterFailure,
  repeatedFailureRender,
  drawsAfterRepeatedFailure,
  retryReady,
  recoveredRender,
  draws,
  beforeLoss,
  duringLoss,
  afterRestore,
  states,
  stateVisuals,
  width: canvas.width,
  height: canvas.height,
  listenerCount: listeners.size,
  disposedOpacity: canvas.style.opacity,
}));
"""
    completed = subprocess.run(
        ["node", "-e", scenario],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["rendered"] is True
    assert result["draws"] == [
        [12, 0, 14],
        [13, 0, 2],
        [12, 0, 14],
        [12, 0, 14],
        [13, 0, 2],
    ]
    assert result["failedRender"] is False
    assert result["failureSnapshot"]["mode"] == "canvas2d"
    assert result["failureSnapshot"]["fallbackLatched"] is True
    assert result["drawsAfterFailure"] == 3
    assert result["repeatedFailureRender"] is False
    assert result["drawsAfterRepeatedFailure"] == 3
    assert result["retryReady"] is True
    assert result["recoveredRender"] is True
    assert result["beforeLoss"]["mode"] == "webgl2"
    assert result["beforeLoss"]["edgeCount"] == 7
    assert result["beforeLoss"]["nodeCount"] == 2
    assert result["duringLoss"] is False
    assert result["afterRestore"]["mode"] == "webgl2"
    assert result["states"] == [
        "ready",
        "fallback",
        "ready",
        "lost",
        "ready",
        "restored",
    ]
    assert result["stateVisuals"] == [
        ["ready", "1"],
        ["fallback", "0"],
        ["ready", "1"],
        ["lost", "0"],
        ["ready", "1"],
        ["restored", "1"],
    ]
    assert result["disposedOpacity"] == "0"
    edge_upload = result["edgeUpload"]
    assert edge_upload[5] == pytest.approx(0.075 * 0.28)
    assert edge_upload[12] == pytest.approx(0.075 * 0.28)
    assert edge_upload[19] == pytest.approx(0.18)
    node_upload = result["nodeUpload"]
    violet_mix = 0.2 * 0.72
    expected_violet = [
        base + (violet - base) * violet_mix
        for base, violet in zip([100, 120, 140], [155, 124, 255], strict=True)
    ]
    hot_mix = (0.6 - 0.55) / 0.7
    expected_hot = [
        base + (hot - base) * hot_mix
        for base, hot in zip([100, 120, 140], [255, 243, 221], strict=True)
    ]
    assert node_upload[2:5] == pytest.approx(
        [channel / 255 for channel in expected_violet]
    )
    assert node_upload[5] == pytest.approx(0.72)
    assert node_upload[9:12] == pytest.approx(
        [channel / 255 for channel in expected_hot]
    )
    assert node_upload[12] == pytest.approx(0.82)
    assert result["width"] == 1600
    assert result["height"] == 1200
    assert result["listenerCount"] == 0


def test_cortex_transport_policy_normalizes_v1_v2_and_processing_activity() -> None:
    policy_path = dashboard.STATIC_DIR / "cortex-transport-policy.js"
    scenario = f"const policy = require({json.dumps(str(policy_path))});\n" + """
const legacyRecall = policy.normalizeEvent({
  kind: "read",
  source: "telemetry-fallback",
  page_ids: ["page-a", "page-a"],
});
const envelope = policy.normalizeEvent({
  schema: "chronovisor.cortex.event.v2",
  family: "transport",
  origin: "pull-log",
  mode: "live",
  source: "telemetry-fallback",
  kind: "used",
  presentation: {
    lane_key: "recall",
    phase: "apply",
    channel_key: "recall-used",
    priority_class: "protected",
  },
});
const legacyField = policy.normalizeEvent({
  kind: "spread",
  source: "stateful-recall-field",
  session_hash: "0123456789abcdef",
});
const legacyUnknownTelemetry = policy.normalizeEvent({
  kind: "unknown_activity",
  source: "telemetry-fallback",
});
const v2Field = policy.normalizeEvent({
  schema: policy.EVENT_SCHEMA,
  family: "field",
  origin: "recall-field",
  mode: "live",
  source: "stateful-recall-field",
  kind: "spread",
});
const processing = policy.normalizeProcessingActivity({
  key: "unknown-lane",
  label: "UNKNOWN",
  current_step: "verify",
  model: "model-a",
  role: "role-a",
});
const pageIds = policy.normalizePageIds([
  "page-a",
  42,
  null,
  "",
  "page-a",
  "x".repeat(300),
  "x".repeat(300),
  ...Array.from({ length: 30 }, (_, index) => `page-${index}`),
]);
const rejected = {
  unknownSchema: policy.normalizeEvent({
    schema: "chronovisor.cortex.event.v99",
    family: "transport",
    kind: "read",
    source: "telemetry-fallback",
  }),
  invalidFamily: policy.normalizeEvent({
    schema: policy.EVENT_SCHEMA,
    family: "other",
    kind: "read",
    source: "telemetry-fallback",
  }),
  transportAsField: policy.normalizeEvent({
    schema: policy.EVENT_SCHEMA,
    family: "field",
    kind: "read",
    source: "stateful-recall-field",
  }),
  fieldAsTransport: policy.normalizeEvent({
    schema: policy.EVENT_SCHEMA,
    family: "transport",
    kind: "spread",
    source: "telemetry-fallback",
  }),
  badSource: policy.normalizeEvent({
    schema: policy.EVENT_SCHEMA,
    family: "transport",
    kind: "read",
    source: "stateful-recall-field",
  }),
};
process.stdout.write(JSON.stringify({
  legacyRecall,
  envelope,
  legacyField,
  legacyUnknownTelemetry,
  v2Field,
  processing,
  pageIds,
  pageIdMaxLength: policy.PAGE_ID_MAX_LENGTH,
  rejected,
}));
"""
    completed = subprocess.run(
        ["node", "-e", scenario],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["legacyRecall"] | {
        "schema": "chronovisor.cortex.event.v2",
        "family": "transport",
        "origin": "telemetry-fallback",
        "mode": "live",
        "lane_key": "recall",
        "phase": "generate",
        "channel_key": "read",
        "priority_class": "protected",
        "page_ids": ["page-a"],
    } == result["legacyRecall"]
    assert result["envelope"]["presentation"] == {
        "lane_key": "recall",
        "phase": "apply",
        "channel_key": "recall-used",
        "priority_class": "protected",
    }
    assert result["legacyField"]["family"] == "field"
    assert result["legacyField"]["source"] == "stateful-recall-field"
    assert result["legacyUnknownTelemetry"]["family"] == "telemetry"
    assert result["legacyUnknownTelemetry"]["kind"] == "unknown_activity"
    assert result["v2Field"]["family"] == "field"
    assert result["v2Field"]["kind"] == "spread"
    assert result["processing"]["schema"] == "chronovisor.cortex.event.v2"
    assert result["processing"]["family"] == "transport"
    assert result["processing"]["origin"] == "activity-stream"
    assert result["processing"]["lane_key"] == "audit"
    assert result["processing"]["phase"] == "consensus"
    assert result["processing"]["channel_key"] == "processing:audit"
    assert result["pageIdMaxLength"] == 240
    assert len(result["pageIds"]) == 24
    assert result["pageIds"][:2] == ["page-a", "x" * 240]
    assert all(isinstance(page_id, str) for page_id in result["pageIds"])
    assert all(len(page_id) <= 240 for page_id in result["pageIds"])
    assert result["rejected"] == {
        "unknownSchema": None,
        "invalidFamily": None,
        "transportAsField": None,
        "fieldAsTransport": None,
        "badSource": None,
    }


def test_cortex_processing_lane_projection_is_pure_and_uses_mode_holds() -> None:
    policy_path = dashboard.STATIC_DIR / "cortex-transport-policy.js"
    scenario = f"const policy = require({json.dumps(str(policy_path))});\n" + """
const demoCapture = policy.projectProcessingLane({
  kind: "save",
  source: "demo",
  byte_count: 18841,
  raw_count: 3,
  capture_id: "9f3a7c21",
});
const liveCapture = policy.projectProcessingLane({
  kind: "save",
  source: "telemetry-fallback",
  byte_count: 2048,
  capture_id: "live-capture",
});
const active = policy.projectProcessingLane(
  policy.normalizeProcessingActivity({
    key: "recall",
    current_step: "rerank",
    model: "model-a",
    role: "recall_judge",
  }),
);
const complete = policy.projectProcessingLane(active.event, "complete");
const unknown = policy.projectProcessingLane({
  kind: "processing",
  lane_key: "not-real",
  step: "work",
  source: "processing-activity",
});
const demoBeforeLive = policy.projectProcessingLane({
  kind: "ingest",
  phase: "generate",
  source: "demo",
  demo_sequence_started_at: 100,
});
const demoAfterLive = policy.projectProcessingLane({
  kind: "ingest",
  phase: "apply",
  source: "demo",
  demo_sequence_started_at: 300,
});
const liveDecision = policy.processingLaneUpdateDecision(active, 0, 200);
const staleDemoDecision = policy.processingLaneUpdateDecision(
  demoBeforeLive,
  liveDecision.lastLiveAt,
  250,
);
const newerDemoDecision = policy.processingLaneUpdateDecision(
  demoAfterLive,
  liveDecision.lastLiveAt,
  350,
);
process.stdout.write(JSON.stringify({
  laneKeys: policy.PROCESSING_LANE_KEYS,
  demoCapture,
  liveCapture,
  active,
  complete,
  unknown,
  liveDecision,
  staleDemoDecision,
  newerDemoDecision,
}));
"""
    completed = subprocess.run(
        ["node", "-e", scenario],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["laneKeys"] == [
        "raw_buffer",
        "ingest",
        "recall",
        "audit",
        "improve",
        "repair",
        "typed_graph",
    ]
    assert result["demoCapture"] | {
        "laneKey": "raw_buffer",
        "phase": "capture",
        "status": "CAPTURE",
        "detail": "18.4 KB · 3 raw · ID 9f3a7c21",
        "state": "active",
        "holdMs": 4000,
        "mode": "demo",
    } == result["demoCapture"]
    assert result["liveCapture"]["holdMs"] == 5400
    assert result["active"] | {
        "laneKey": "recall",
        "phase": "generate",
        "status": "RERANK",
        "detail": "model-a · recall_judge",
        "state": "active",
        "holdMs": 4800,
        "mode": "live",
    } == result["active"]
    assert result["complete"]["state"] == "complete"
    assert result["complete"]["status"] == "COMPLETE"
    assert result["complete"]["holdMs"] == 1800
    assert result["unknown"]["laneKey"] == "audit"
    assert result["liveDecision"] == {"accept": True, "lastLiveAt": 200}
    assert result["staleDemoDecision"] == {
        "accept": False,
        "lastLiveAt": 200,
    }
    assert result["newerDemoDecision"] == {
        "accept": True,
        "lastLiveAt": 200,
    }


def test_cortex_transport_retention_resists_processing_storm_and_stays_bounded(
) -> None:
    script = (dashboard.STATIC_DIR / "cortex.js").read_text(encoding="utf-8")
    assert "TransportPolicy.transportTiming(" in script
    assert "TransportPolicy.pruneAndBoundTransportEffects(" in script
    policy_path = dashboard.STATIC_DIR / "cortex-transport-policy.js"
    scenario = f"const policy = require({json.dumps(str(policy_path))});\n" + """
const save = {
  id: "save",
  kind: "save",
  startedAt: 0,
  duration: 5000,
  retainedUntil: 5000,
};
const recall = {
  id: "recall",
  kind: "recall",
  startedAt: 0,
  duration: 3200,
  retainedUntil: 3200,
};
const processingStorm = Array.from({ length: 18 }, (_, index) => ({
  id: `processing-${index}`,
  kind: "processing",
  startedAt: 900 + index,
  duration: 2300,
  retainedUntil: 900 + index,
}));
const retained = policy.pruneAndBoundTransportEffects(
  [save, recall, ...processingStorm],
  1000,
);
const expired = policy.pruneAndBoundTransportEffects([
  { id: "expired", kind: "processing", startedAt: 0, duration: 100 },
  { id: "live", kind: "processing", startedAt: 950, duration: 1000 },
], 1000);
const protectedBurst = Array.from({ length: 19 }, (_, index) => ({
  id: `memory-${index}`,
  kind: "recall",
  startedAt: index,
  duration: 5000,
  retainedUntil: 4000 + index,
}));
policy.pruneAndBoundTransportEffects(protectedBurst, 1000);
process.stdout.write(JSON.stringify({
  demoCaptureTiming: policy.transportTiming(
    { kind: "save", source: "demo" }, "capture", 1000,
  ),
  liveCaptureTiming: policy.transportTiming(
    { kind: "save", source: "transport" }, "capture", 1000,
  ),
  recallTiming: policy.transportTiming(
    { kind: "read", source: "transport" }, "generate", 1000,
  ),
  usedTiming: policy.transportTiming(
    { kind: "used", source: "transport" }, "apply", 1000,
  ),
  retainedIds: retained.map((effect) => effect.id),
  retainedLength: retained.length,
  expiredIds: expired.map((effect) => effect.id),
  protectedLength: protectedBurst.length,
  oldestProtectedEvicted: !protectedBurst.some((effect) => effect.id === "memory-0"),
}));
"""
    completed = subprocess.run(
        ["node", "-e", scenario],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["demoCaptureTiming"] == {
        "duration": 3600,
        "retainedUntil": 4600,
    }
    assert result["liveCaptureTiming"] == {
        "duration": 5000,
        "retainedUntil": 6000,
    }
    assert result["recallTiming"] == {
        "duration": 3200,
        "retainedUntil": 4200,
    }
    assert result["usedTiming"] == {
        "duration": 3600,
        "retainedUntil": 4200,
    }
    assert result["retainedLength"] == 18
    assert "save" in result["retainedIds"]
    assert "recall" in result["retainedIds"]
    assert result["expiredIds"] == ["live"]
    assert result["protectedLength"] == 18
    assert result["oldestProtectedEvicted"] is True


def test_cortex_recall_electricity_survives_field_storm_and_stays_bounded(
) -> None:
    script = (dashboard.STATIC_DIR / "cortex.js").read_text(encoding="utf-8")
    assert "TransportPolicy.recallVisualProfile(event)" in script
    assert "TransportPolicy.liveRecallElectricTiming(" in script
    assert "duration: electricTiming.duration" in script
    assert "retainedUntil: electricTiming.retainedUntil" in script
    policy_path = dashboard.STATIC_DIR / "cortex-transport-policy.js"
    scenario = f"const policy = require({json.dumps(str(policy_path))});\n" + """
const pulses = [];
function queue(pulse) {
  pulses.push(pulse);
  policy.pruneAndBoundElectricPulses(pulses, 1000);
}
for (let index = 0; index < 12; index += 1) {
  queue({
    id: `field-${index}`,
    kind: "field",
    startedAt: 900,
    duration: 760,
    delta: 1 - index * 0.01,
    seq: index,
  });
}
for (let index = 0; index < 3; index += 1) {
  queue({
    id: `recall-${index}`,
    kind: "recall",
    startedAt: 1000 + index,
    duration: 2800,
    retainedUntil: 3800 + index,
    delta: 0.1,
    seq: 100 + index,
  });
}
for (let index = 0; index < 40; index += 1) {
  queue({
    id: `late-field-${index}`,
    kind: "field",
    startedAt: 1000,
    duration: 760,
    delta: 1,
    seq: 200 + index,
  });
}
const protectedBurst = Array.from({ length: 13 }, (_, index) => ({
  id: `protected-${index}`,
  startedAt: 1000,
  duration: 2800,
  retainedUntil: 3800 + index,
  delta: 0.5,
  seq: index,
}));
policy.pruneAndBoundElectricPulses(protectedBurst, 1000);
const expired = [
  { id: "expired", startedAt: 0, duration: 100, delta: 1, seq: 1 },
  { id: "live", startedAt: 950, duration: 1000, delta: 0.5, seq: 2 },
];
policy.pruneAndBoundElectricPulses(expired, 1000);
process.stdout.write(JSON.stringify({
  demoProfile: policy.profileFor("demo"),
  recallProfile: policy.recallVisualProfile({ kind: "recall", mode: "live" }),
  electricTiming: policy.liveRecallElectricTiming(
    { kind: "recall", mode: "live" },
    1000,
  ),
  pulseIds: pulses.map((pulse) => pulse.id),
  pulseLength: pulses.length,
  protectedLength: protectedBurst.length,
  earliestProtectedEvicted: !protectedBurst.some(
    (pulse) => pulse.id === "protected-0",
  ),
  expiredIds: expired.map((pulse) => pulse.id),
}));
"""
    completed = subprocess.run(
        ["node", "-e", scenario],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["demoProfile"] == {
        "mode": "demo",
        "captureWaveDurationMs": 3600,
        "recallNodeDurationMs": 650,
        "recallElectricDurationMs": None,
        "recallTransportMinVisibleMs": 3200,
    }
    assert result["recallProfile"] == {
        "mode": "live",
        "scale": 0.6,
        "nodeDurationMs": 3000,
        "electricDurationMs": 2800,
        "electricRetainMs": 2800,
    }
    assert result["electricTiming"] == {
        "duration": 2800,
        "retainedUntil": 3800,
    }
    assert result["pulseLength"] == 12
    assert {"recall-0", "recall-1", "recall-2"}.issubset(result["pulseIds"])
    assert result["protectedLength"] == 12
    assert result["earliestProtectedEvicted"] is True
    assert result["expiredIds"] == ["live"]


def test_dashboard_serves_cortex_graph_api(monkeypatch) -> None:
    expected = {
        "meta": {"commit": "abc1234"},
        "nodes": [{"id": "page-a"}],
        "links": [],
        "categories": [],
    }
    monkeypatch.setattr(
        dashboard,
        "runtime_identity",
        lambda: {"commit_id": "abc123456789"},
    )
    monkeypatch.setattr(
        dashboard,
        "build_cortex_graph",
        lambda root, commit: expected,
    )
    server = dashboard.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        dashboard.DashboardHandler,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        response = dashboard.httpx.get(
            f"http://{host}:{port}/api/cortex/graph",
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.status_code == 200
    assert response.json() == expected


def test_dashboard_serves_bounded_cortex_relation_details(monkeypatch) -> None:
    expected = [
        {
            "relation_id": "relation-a",
            "source_page_id": "page-a",
            "target_page_id": "page-b",
        }
    ]
    observed: dict[str, object] = {}

    def details(root, relation_keys):
        observed.update(root=root, relation_keys=relation_keys)
        return expected

    monkeypatch.setattr(dashboard, "build_cortex_relation_details", details)
    server = dashboard.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        dashboard.DashboardHandler,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        response = dashboard.httpx.get(
            f"http://{host}:{port}/api/cortex/relations",
            params={"keys": json.dumps([["relation-a", "page-a", "page-b"]])},
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.status_code == 200
    assert response.json() == {"relations": expected}
    assert observed == {
        "root": dashboard.CHRONOVISOR_ROOT,
        "relation_keys": [("relation-a", "page-a", "page-b")],
    }


def test_dashboard_serves_session_scoped_cortex_field_api(monkeypatch) -> None:
    expected = {
        "status": "online",
        "session_hash": "0123456789abcdef",
        "snapshot": {"seq": 4},
        "events": [],
    }
    observed: dict[str, object] = {}

    def projection(root, *, session_hash="", event_limit=256):
        observed.update(
            root=root,
            session_hash=session_hash,
            event_limit=event_limit,
        )
        return expected

    monkeypatch.setattr(dashboard, "build_cortex_field_projection", projection)
    server = dashboard.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        dashboard.DashboardHandler,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        response = dashboard.httpx.get(
            f"http://{host}:{port}/api/cortex/field?session=0123456789abcdef",
            timeout=2,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert response.status_code == 200
    assert response.json() == expected
    assert observed == {
        "root": dashboard.CHRONOVISOR_ROOT,
        "session_hash": "0123456789abcdef",
        "event_limit": 256,
    }
