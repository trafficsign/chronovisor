from __future__ import annotations

import json
import os
import threading
from pathlib import Path

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
    assert all(event["source"] == "telemetry-fallback" for event in events)


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
                },
                "private": "must-not-leak",
            }
        ),
        encoding="utf-8",
    )

    assert cortex._field_growth_summary(root) == {
        "stage": "canary",
        "field_learning_allowed": True,
        "authority_enabled": True,
        "canary_percent": 5,
        "strong_positive": 210,
        "strong_positive_target": 200,
        "strong_sessions": 21,
        "strong_sessions_target": 20,
        "candidate_traces": 0,
    }


def test_cortex_event_cursor_tails_only_selected_field_session(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    session = "0123456789abcdef"
    event_path = root / "recall" / "field" / "events" / f"{session}.jsonl"
    event_path.parent.mkdir(parents=True)
    event_path.write_text("", encoding="utf-8")
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

    assert cursor.poll() == [
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
    ]


def test_cortex_event_cursor_follows_activity_across_field_sessions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chronovisor"
    event_root = root / "recall" / "field" / "events"
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

    events = cursor.poll()

    assert [(event["session_hash"], event["page_id"]) for event in events] == [
        (first.stem, "page-a"),
        (second.stem, "page-b"),
    ]


def test_cortex_static_view_preserves_fable_layout_and_uses_live_data() -> None:
    static = dashboard.STATIC_DIR
    html = (static / "cortex.html").read_text(encoding="utf-8")
    style = (static / "cortex.css").read_text(encoding="utf-8")
    script = (static / "cortex.js").read_text(encoding="utf-8")
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
    assert "function ensureActualEdge(event)" in script
    assert "function drawEdges()" in script
    assert "const ACTIVE_LABEL_LIMIT = 5;" in script
    assert "const NODE_FLASH_ATTACK_MS = 90;" in script
    assert "const NODE_FLASH_HOLD_MS = 150;" in script
    assert "const NODE_FLASH_DECAY_MS = 1450;" in script
    assert "const EDGE_AFTERGLOW_MS = 1550;" in script
    assert "const ELECTRIC_TRAVEL_MIN_MS = 420;" in script
    assert "const ELECTRIC_TRAVEL_MAX_MS = 760;" in script
    assert "const MAX_ELECTRIC_PATHS = 12;" in script
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
    assert "radius + progress * 26" not in script
    assert "radius + (1 - progress) * 28" not in script
    assert "radius + progress * 18" not in script
    assert "radius + progress * 13" not in script
    assert "function electricPathPoints(" in script
    assert "function electricPathPrefix(" in script
    assert "function queueElectricPulse(" in script
    assert "function drawEdgeAfterglows(time)" in script
    assert "target.dataset.electricEdges" in script
    assert "target.dataset.electricPeak" in script
    assert "function publishVisualMetrics(time)" in script
    assert "cortexMetrics.violetNodes += 1;" in script
    assert ".slice(0, ACTIVE_LABEL_LIMIT)" in script
    assert "liveEventsEnabled = true" in script
    assert "followLatestSession = true" in script
    assert '"?follow=latest"' in script
    assert "LIVE · follow activity" in script
    assert "window.CortexField.applyEvents(fieldState" in script
    assert "const MAX_EVENTS = 256;" in field_script
    assert "event.seq !== state.seq + 1" in field_script
    assert "/static/cortex-field.js" in html
    assert html.index("/static/cortex-field.js") < html.index("/static/cortex.js")
    assert "DEMO · RECALL" in html
    assert "@media (prefers-reduced-motion: reduce)" in style
    assert 'id="recall-field-summary"' in observatory
    assert 'fetch("/api/cortex/field"' in (static / "app.js").read_text(
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
