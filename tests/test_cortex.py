from __future__ import annotations

import json
from pathlib import Path

from chronovisor.core.raw_segment import RawSegmentReceipt, append_capture
from chronovisor.ops import cortex
from chronovisor.ops.cortex_stream import websocket_accept, websocket_text_frame


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
    idempotency_key = f"codex-{session_key}-from{after_line}-to{until_line}"
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


def test_websocket_helpers_follow_rfc6455() -> None:
    assert (
        websocket_accept("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
    )

    frame = websocket_text_frame({"type": "events", "events": [{"kind": "recall"}]})

    assert frame[0] == 0x81
    assert frame[1] == len(frame) - 2
    assert json.loads(frame[2:]) == {
        "type": "events",
        "events": [{"kind": "recall"}],
    }


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


def test_cortex_graph_uses_backlinks_without_old_graph_cache_or_writes(tmp_path):
    _write_page(tmp_path / "pages/a.md", title="A", body="No outgoing text link")
    _write_page(tmp_path / "pages/b.md", title="B", body="Frozen page")
    (tmp_path / ".index").mkdir()
    (tmp_path / ".index/backlinks.json").write_text(json.dumps({"edges": {"b": ["a"]}}))
    old = tmp_path / "runtime/dashboard-materialized/cortex-graph.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"graph": {"nodes": [{"id": "retired"}]}}))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    graph = cortex.build_cortex_graph(tmp_path)
    assert graph["links"] == [[0, 1, 0]]
    assert graph["meta"]["static"] == 1
    assert "typedGraph" not in graph
    assert cortex.build_cortex_graph(tmp_path) is graph
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_cortex_capture_only_and_legacy_symlink_is_ignored(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    outside = tmp_path / "private.md"
    outside.write_text("private")
    (raw / "link.md").symlink_to(outside)
    cursor = cortex.CortexEventCursor(tmp_path)
    (raw / "codex.md").write_text("new capture")
    events = cursor.poll()
    assert len(events) == 1 and events[0]["kind"] == "save"
    assert events[0]["byte_count"] == len("new capture")
    assert cursor.poll() == []
