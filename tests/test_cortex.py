from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from chronovisor.ops import cortex, dashboard


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
        f"---\ntitle: {title}\nupdated: 2026-07-29\n{tag_line}---\n{body}\n",
        encoding="utf-8",
    )


def test_build_cortex_graph_uses_local_wiki_without_exposing_bodies(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    _write_page(
        root / "pages" / "alpha" / "page-a.md",
        title="Alpha <private>",
        body="Links to [[page-b]] and [[missing-page]].",
        tags=["d/example"],
    )
    _write_page(
        root / "pages" / "beta" / "page-b.md",
        title="Beta",
        body="Back to [[page-a]].",
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
        "totalLines": 19,
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
    assert events[5]["page_ids"] == ["page-c"]


def test_cortex_static_view_preserves_fable_layout_and_uses_live_data() -> None:
    static = dashboard.STATIC_DIR
    html = (static / "cortex.html").read_text(encoding="utf-8")
    style = (static / "cortex.css").read_text(encoding="utf-8")
    script = (static / "cortex.js").read_text(encoding="utf-8")
    observatory = (static / "index.html").read_text(encoding="utf-8")
    activity_style = (static / "activity-bar.css").read_text(encoding="utf-8")

    assert "CHRONOVISOR // SYNAPTIC CORTEX" in html
    assert 'id="side"' in html
    assert 'id="stage"' in html
    assert 'id="hud"' in html
    assert 'id="mOrganic"' in html
    assert 'id="mCluster"' in html
    assert 'id="tLive"' in html
    assert 'id="tAuto"' not in html
    assert 'id="tSnd"' in html
    assert "grid-template-columns: 242px 1fr 296px;" in style
    assert "--amber: #ffb454;" in style
    assert "repeating-linear-gradient" in style
    assert 'fetch("/api/cortex/graph"' in script
    assert 'new WebSocket(`${protocol}//${window.location.host}/api/cortex/events`)' in script
    assert "function firePageIds(pageIds, kind, label)" in script
    assert "function drawEdges()" in script
    assert "liveEventsEnabled = true" in script
    assert "function ambient(" not in script
    assert "function autoTick(" not in script
    assert 'setTimeout(() => stimulate("recall")' not in script
    assert "if (performance.now() <= tickerHold) return;" in script
    assert "/static/cortex_graph.json" not in script
    assert "3d-force-graph" not in html
    assert 'class="has-activity-bar"' in observatory
    assert 'class="has-activity-bar"' in html
    assert (
        'class="activity-view is-active" data-view="observatory"'
        in observatory
    )
    assert 'class="activity-view" data-view="cortex"' in observatory
    assert 'class="activity-view" data-view="observatory"' in html
    assert 'class="activity-view is-active" data-view="cortex"' in html
    assert 'aria-label="Chronovisor views"' in observatory
    assert 'aria-label="Chronovisor views"' in html
    assert "--activity-bar-width: 48px;" in activity_style
    assert "body.has-activity-bar {" in activity_style
    assert ".activity-bar {" in activity_style
    assert ".has-activity-bar .shell {" in activity_style


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
