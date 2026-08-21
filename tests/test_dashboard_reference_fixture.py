from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
import threading
from html.parser import HTMLParser
from http.server import ThreadingHTTPServer
from pathlib import Path

from chronovisor.ops import dashboard

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/dashboard_decision_trace_states.json"


class _DashboardMarkup(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))


def test_dashboard_reference_contract_and_state_matrix() -> None:
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
    reference = contract["reference"]
    image = ROOT / reference["path"]
    raw = image.read_bytes()

    assert hashlib.sha256(raw).hexdigest() == reference["sha256"]
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", raw[16:24]) == (
        reference["width"],
        reference["height"],
    )
    assert reference["viewport"] == {"width": 1586, "height": 992}
    assert reference["capture_selector"] == "#processing-panel"

    cases = {row["id"]: row for row in contract["cases"]}
    assert list(cases) == list("ABCDEFGH")
    assert contract["topology"]["artifact_nodes"] == 1
    assert contract["topology"]["decision_nodes"] == 1
    assert contract["topology"]["dynamic_paths"] == []
    assert contract["topology"]["artifact_join_state_order"] == [
        "single",
        "pair_yes",
        "tie_input",
        "gate",
        "gate_yes",
        "no",
    ]
    assert contract["topology"]["fixed_paths"][-9:] == [
        "single-artifact",
        "pair-tie-break",
        "pair-artifact-join",
        "tie_break-quorum",
        "quorum-artifact-join",
        "artifact-seal",
        "seal-decision",
        "quorum-hold",
        "seal-hold",
    ]

    assert cases["A"]["active_lane"] == "primary"
    assert cases["B"]["active_lane"] == "challenger"
    assert cases["C"]["branch"] == "pair-agreement"
    assert cases["C"]["lane_states"][2] == "skipped"
    assert cases["D"]["active_lane"] == "tie_break"
    assert cases["E"]["branch"] == "tie-break-agreement"
    assert cases["F"]["hold"] == "before-artifact"
    assert cases["F"]["artifact"] == "skipped"
    assert cases["F"]["decision"] == "skipped"
    assert cases["G"]["hold"] == "seal-failed"
    assert cases["G"]["artifact"] == "error"
    assert cases["G"]["decision"] == "skipped"
    assert cases["H"]["state"] == "idle"
    assert cases["H"]["active_lane"] is None
    assert [cases[key]["artifact_join_states"] for key in cases] == [
        ["pending", "pending", "pending", "pending", "pending", "pending"],
        ["pending", "pending", "pending", "pending", "pending", "pending"],
        ["pending", "done", "pending", "pending", "pending", "pending"],
        ["pending", "pending", "pending", "pending", "pending", "pending"],
        ["pending", "pending", "done", "done", "done", "pending"],
        ["pending", "pending", "error", "error", "pending", "error"],
        ["pending", "done", "pending", "pending", "pending", "pending"],
        ["pending", "pending", "pending", "pending", "pending", "pending"],
    ]

    for row in cases.values():
        assert len(row["lane_states"]) == 3
        assert len(row["lane_think"]) == 3
        assert len(row["lane_context_tokens"]) == 3


def test_dashboard_reference_svg_has_one_fixed_safe_topology() -> None:
    parser = _DashboardMarkup()
    markup = (ROOT / "src/chronovisor/dashboard_static/index.html").read_text(
        encoding="utf-8"
    )
    parser.feed(markup)
    elements = parser.elements

    def matching(key: str, value: str) -> list[dict[str, str | None]]:
        return [attrs for _tag, attrs in elements if attrs.get(key) == value]

    assert len(matching("id", "processing-panel")) == 1
    assert len(matching("class", "processing-lanes-card")) == 1
    assert len(matching("id", "decision-trace-panel")) == 1
    assert not matching("id", "processing-trace-connector")
    assert not matching("id", "processing-trace-connector-path")
    assert not matching("data-trace-entry", "true")
    assert len(matching("id", "decision-trace-harness")) == 1
    assert matching("id", "decision-trace-harness")[0]["viewbox"] == ("0 0 1500 650")
    assert matching("id", "decision-trace-harness")[0]["height"] == "607"
    assert matching("id", "trace-arrow")[0]["refx"] == "6"

    assert len(matching("data-trace-key", "artifact")) == 1
    assert len(matching("data-trace-key", "quorum")) == 1
    assert len(matching("data-trace-key", "seal")) == 1
    assert len(matching("data-trace-key", "decision")) == 1
    assert len(matching("data-trace-key", "hold")) == 1
    assert [
        attrs.get("data-overall-key")
        for _tag, attrs in elements
        if attrs.get("data-overall-key")
    ] == [
        "packet",
        "preflight",
        "execution_plan",
    ]
    for key in ("packet", "preflight", "execution_plan"):
        node = markup.split(f'data-overall-key="{key}"', 1)[1].split("</g>", 1)[0]
        assert '<circle r="10"></circle>' in node
    style = (ROOT / "src/chronovisor/dashboard_static/style.css").read_text(
        encoding="utf-8"
    )
    assert ".trace-overall .trace-node circle" not in style
    assert len(matching("data-context-tokens", "32768")) == 1
    assert len(matching("data-context-tokens", "65536")) == 1
    assert len(matching("data-context-tokens", "98304")) == 1
    assert len(matching("data-context-tokens", "131072")) == 1
    for tokens in ("32768", "65536", "98304", "131072"):
        option = markup.split(f'data-context-tokens="{tokens}"', 1)[1].split(
            "</g>", 1
        )[0]
        assert '<circle r="10"></circle>' in option
    assert not matching("class", "trace-plan-frame")
    assert {
        attrs["data-decision-lane"]: attrs.get("transform")
        for _tag, attrs in elements
        if attrs.get("data-decision-lane")
    } == {
        "primary": "translate(0 325)",
        "challenger": "translate(0 450)",
        "tie_break": "translate(0 575)",
    }
    primary_lane_markup = markup.split('data-decision-lane="primary"', 1)[1].split(
        'data-decision-lane="challenger"', 1
    )[0]
    assert (
        'data-decision-lane-step="vote" transform="translate(1096 35)"><circle r="10"></circle><text x="-25" y="-17">Vote</text>'
        in primary_lane_markup
    )
    assert len(matching("data-seal-yes-label", "true")) == 1
    assert len(matching("data-seal-no-label", "true")) == 1
    assert len(matching("data-pair-no-label", "true")) == 1

    paths = {
        attrs["data-path-key"]: attrs.get("d")
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-path-key")
    }
    assert {
        "single-artifact",
        "pair-tie_break",
        "pair-artifact-join",
        "tie_break-quorum",
        "quorum-artifact-join",
        "quorum-hold",
        "artifact-seal",
        "seal-decision",
        "seal-hold",
    }.issubset(paths)
    assert all(paths[key] for key in paths)
    assert not {
        "pair-artifact",
        "tie_break-artifact",
        "pair-hold",
        "tie_break-hold",
        "pair-quorum",
        "quorum-artifact",
    } & paths.keys()
    assert any(
        tag == "path" and attrs.get("d") == "M1208 410 1248 450 1208 490 1168 450Z"
        for tag, attrs in elements
    )
    context_generate_rails = matching("data-lane-path", "context-generate")
    assert len(context_generate_rails) == 3
    assert all(
        attrs.get("d") == "M575 0 H733"
        and attrs.get("marker-end") == "url(#trace-arrow)"
        for attrs in context_generate_rails
    )
    primary_validate_vote = markup.split(
        'data-decision-lane="primary"', 1
    )[1].split('data-decision-lane="challenger"', 1)[0]
    assert (
        'data-lane-path="validate-vote" '
        'd="M920 0 H1086 Q1096 0 1096 10 V25" '
        'marker-end="url(#trace-arrow)"'
        in primary_validate_vote
    )

    expected_boundary_paths = {
        "packet-preflight": "M106 10 H288",
        "preflight-execution_plan": "M308 10 H482 Q492 10 492 20 V36",
        "execution-plan-context": (
            "M492 56 V76 Q492 86 482 86 H452 Q442 86 442 96 V140"
        ),
        "plan-dispatch": (
            "M1380 164 H1466 Q1476 164 1476 174 V252 Q1476 262 1466 262 "
            "H190 Q180 262 180 272 V315 Q180 325 190 325 H220"
        ),
        "primary-challenger": (
            "M1096 370 V377 Q1096 387 1086 387 H190 Q180 387 180 397 "
            "V440 Q180 450 190 450 H220"
        ),
        "single-artifact": "M1106 360 H1300 Q1310 360 1310 370 V393",
        "challenger-agree": "M1106 450 H1168",
        "pair-tie_break": (
            "M1208 490 V502 Q1208 512 1198 512 "
            "H190 Q180 512 180 522 V565 Q180 575 190 575 H220"
        ),
        "pair-artifact-join": (
            "M1248 450 H1278 Q1288 450 1288 440 "
            "V414 Q1288 404 1298 404 H1299"
        ),
        "tie_break-quorum": "M1106 575 H1217",
        "quorum-artifact-join": (
            "M1248 545 V524 Q1248 514 1258 514 "
            "H1300 Q1310 514 1310 504 V415"
        ),
        "quorum-hold": (
            "M1278 575 H1408 Q1418 575 1418 585 V590 Q1418 600 1428 600 H1448"
        ),
        "artifact-seal": (
            "M1321 404 H1360 Q1370 404 1370 414 V470"
        ),
        "seal-decision": "M1404 500 H1449",
        "seal-hold": "M1370 530 V540 Q1370 550 1380 550 H1450 Q1460 550 1460 560 V588",
    }
    assert {key: paths[key] for key in expected_boundary_paths} == (
        expected_boundary_paths
    )
    assert all(
        matching("data-path-key", key)[0].get("marker-end") == "url(#trace-arrow)"
        for key in (
            "single-artifact",
            "pair-artifact-join",
            "quorum-artifact-join",
        )
    )
    assert not {"quorum-artifact-trunk", "artifact-input"} & paths.keys()

    repair_paths = [
        attrs
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-repair-lane")
    ]
    assert len(repair_paths) == 3
    assert all(
        attrs.get("d") == "M743 10 V30 Q743 40 753 40 H900 Q910 40 910 30 V10"
        and attrs.get("marker-start") == "url(#trace-arrow)"
        and attrs.get("marker-end") is None
        for attrs in repair_paths
    )

    reasoning_guides = [
        attrs
        for tag, attrs in elements
        if tag == "path" and attrs.get("class") == "trace-reasoning-guide"
    ]
    assert len(reasoning_guides) == 1
    assert [attrs.get("d") for attrs in reasoning_guides] == [
        "M812 118 V58 Q812 48 822 48 H903 "
        "M921 48 H1320 Q1330 48 1330 58 V118 "
        "M858 164 H882 Q892 164 892 154 V116 Q892 106 902 106 H903 "
        "M921 106 H1245 Q1255 106 1255 116 V154 Q1255 164 1265 164 H1270 "
        "M858 164 H903 M921 164 H1270 "
        "M858 164 H882 Q892 164 892 174 V212 Q892 222 902 222 H903 "
        "M921 222 H1245 Q1255 222 1255 212 V174 Q1255 164 1265 164 H1270",
    ]
    assert [
        (
            attrs.get("data-reasoning-output"),
            attrs.get("d"),
            attrs.get("marker-end"),
        )
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-reasoning-output")
    ] == [
        (
            "off",
            "M921 48 H1320 Q1330 48 1330 58 V118",
            "url(#trace-arrow)",
        ),
        (
            "low",
            "M921 106 H1245 Q1255 106 1255 116 V154 Q1255 164 1265 164 H1270",
            None,
        ),
        ("medium", "M921 164 H1270", None),
        (
            "high",
            "M921 222 H1245 Q1255 222 1255 212 V174 Q1255 164 1265 164 H1270",
            None,
        ),
    ]
    assert paths["reasoning-off"] == "M812 118 V58 Q812 48 822 48 H903"
    assert matching("data-path-key", "plan-fit")[0].get("class") == (
        "trace-path trace-reasoning-path"
    )
    assert all(
        attrs.get("marker-end") == "url(#trace-arrow)"
        for tag, attrs in elements
        if tag == "path"
        and attrs.get("data-path-key")
        in {
            "reasoning-off",
            "reasoning-low",
            "reasoning-medium",
            "reasoning-high",
            "plan-fit",
        }
    )
    assert all(
        attrs.get("y") == "28"
        for tag, attrs in elements
        if tag == "text" and attrs.get("data-lane-result")
    )
    assert len(matching("class", "trace-label-backdrop")) == 1
    assert len(matching("class", "trace-selected-core")) == 8
    assert {
        attrs["data-reasoning-key"]: attrs.get("transform")
        for _tag, attrs in elements
        if attrs.get("data-reasoning-key")
    } == {
        "off": "translate(912 48)",
        "low": "translate(912 106)",
        "medium": "translate(912 164)",
        "high": "translate(912 222)",
    }
    assert all(
        attrs.get("y") == "34"
        for tag, attrs in elements
        if tag == "text" and attrs.get("class") == "trace-reasoning-detail"
    )
    assert matching("data-plan-value", "fit")[0]["y"] == "174"
    assert matching("data-plan-fit-pass", "true")[0]["y"] == "190"
    assert matching("class", "trace-branch-node-label")[0]["x"] == "16"
    assert matching("class", "trace-branch-node-label")[0]["y"] == "4"
    assert matching("data-overall-key", "execution_plan")[0]["transform"] == (
        "translate(492 46)"
    )
    assert matching("class", "trace-artifact-label")[0]["x"] == "-16"
    assert matching("class", "trace-artifact-label")[0]["text-anchor"] == "end"
    assert [
        attrs.get("y")
        for _tag, attrs in elements
        if attrs.get("class") == "trace-path-label"
    ] == ["258", "381"]
    assert 325 - 262 == 450 - 387 == 575 - 512 == 63
    assert matching("class", "trace-branch-label trace-yes-label")[0]["y"] == "438"
    assert matching("class", "trace-branch-label trace-no-label")[0]["y"] == "506"
    pair_yes_label = matching("data-pair-yes-label", "true")[0]
    assert (pair_yes_label["x"], pair_yes_label["y"]) == ("1258", "438")
    assert matching("data-pair-no-label", "true")[0]["x"] == "1180"
    assert matching("data-quorum-yes-label", "true")[0]["x"] == "1268"
    assert matching("data-quorum-no-label", "true")[0]["x"] == "1336"
    assert matching("data-seal-yes-label", "true")[0]["x"] == "1422"
    assert matching("data-seal-no-label", "true")[0]["x"] == "1382"
    assert paths["plan-context"] == (
        "M442 160 V204 Q442 214 452 214 H724 Q734 214 734 204 "
        "V174 Q734 164 744 164 H766"
    )
    assert "plan-headroom" not in paths
    assert next(
        attrs.get("marker-end")
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-path-key") == "plan-context"
    ) == "url(#trace-arrow)"
    context_guides = [
        attrs
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-context-guide")
    ]
    assert len(context_guides) == 1
    assert context_guides[0]["data-context-guide"] == "all"
    assert context_guides[0]["d"] == (
        "M492 56 V76 Q492 86 482 86 H352 Q342 86 342 96 V140 "
        "M492 56 V76 Q492 86 482 86 H452 Q442 86 442 96 V140 "
        "M492 56 V76 Q492 86 502 86 H532 Q542 86 542 96 V140 "
        "M492 56 V76 Q492 86 502 86 H632 Q642 86 642 96 V140 "
        "M442 160 V204 Q442 214 452 214 "
        "M542 160 V204 Q542 214 552 214 "
        "M642 160 V204 Q642 214 652 214 "
        "M342 160 V204 Q342 214 352 214 H724 Q734 214 734 204 "
        "V174 Q734 164 744 164 H766"
    )
    assert context_guides[0].get("marker-end") is None
    assert not matching("class", "trace-context-rail")
    assert len(matching("class", "trace-label-backdrop trace-context-kicker-backdrop")) == 1
    assert all(
        attrs.get("x") == "15" and attrs.get("y") == "27"
        for attrs in matching("data-context-label", "true")
    )
    assert [
        attrs.get("transform") for attrs in matching("data-context-option", "true")
    ] == [
        "translate(342 150)",
        "translate(442 150)",
        "translate(542 150)",
        "translate(642 150)",
    ]
    assert matching("data-plan-value", "context-selection")[0]["y"] == "239"


def test_dashboard_reference_quorum_hold_paths_and_label_stay_in_bounds() -> None:
    parser = _DashboardMarkup()
    parser.feed(
        (ROOT / "src/chronovisor/dashboard_static/index.html").read_text(
            encoding="utf-8"
        )
    )
    elements = parser.elements
    paths = {
        attrs["data-path-key"]: attrs.get("d")
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-path-key")
    }

    assert paths["quorum-hold"] == (
        "M1278 575 H1408 Q1418 575 1418 585 "
        "V590 Q1418 600 1428 600 H1448"
    )
    assert paths["seal-hold"] == (
        "M1370 530 V540 Q1370 550 1380 550 "
        "H1450 Q1460 550 1460 560 V588"
    )

    hold_reason = next(
        attrs
        for _tag, attrs in elements
        if attrs.get("data-hold-reason") == "true"
    )
    assert hold_reason["class"] == "trace-hold-reason"
    assert hold_reason["x"] == "28"
    assert 1460 + int(hold_reason["x"]) == 1488 < 1500
    assert hold_reason["y"] == "43"
    assert 600 + int(hold_reason["y"]) == 643 < 650

    style = (ROOT / "src/chronovisor/dashboard_static/style.css").read_text(
        encoding="utf-8"
    )
    hold_reason_style = style.split(
        ".decision-trace-harness .trace-hold-reason {", 1
    )[1].split("}", 1)[0]
    assert "text-anchor: end;" in hold_reason_style


def test_dashboard_reference_keeps_selection_and_bucket_truth() -> None:
    static = ROOT / "src/chronovisor/dashboard_static"
    page = (static / "index.html").read_text(encoding="utf-8")
    style = (static / "style.css").read_text(encoding="utf-8")
    app = (static / "app.js").read_text(encoding="utf-8")
    renderer = (static / "app-renderer.js").read_text(encoding="utf-8")

    assert "processing-trace-connector" not in page
    assert "processingTraceConnector" not in app
    assert "updateProcessingTraceConnector" not in renderer
    assert "processingTraceResizeObserver" not in renderer
    assert "ResizeObserver" not in renderer
    assert ".processing-trace-connector" not in style
    assert 'node.className = `processing-step ${fmt(step.status, "pending")}`;' in renderer
    assert "setDecisionSvgState(node, milestoneStates[node.dataset.overallKey]);" in renderer
    processing_active_style = style.rsplit(
        ".processing-step.active::before {", 1
    )[1].split("}", 1)[0]
    assert "border-color: #ff9d00;" in processing_active_style
    assert "background: #ff9d00;" in processing_active_style

    assert [
        token
        for token in ("32768", "65536", "98304", "131072")
        if f'data-context-tokens="{token}"' in page
    ] == ["32768", "65536", "98304", "131072"]
    assert '<p id="work-summary">Waiting · 0 / 3</p>' in page
    assert (
        page.index('data-context-guide="all"')
        < page.index('data-path-key="execution-plan-context"')
        < page.index("trace-context-kicker-backdrop")
        < page.index(">CONTEXT WINDOW<")
    )
    assert (
        page.index('data-context-guide="all"')
        < page.index('data-path-key="plan-context"')
        < page.index('data-plan-value="context-selection"')
    )
    unselected_track_style = style.split(
        '.processing-lane[aria-selected="false"] .processing-track {', 1
    )[1].split("}", 1)[0]
    assert "opacity: 0.42;" in unselected_track_style
    assert "visibility: hidden;" not in unselected_track_style
    assert '.processing-lane.active[aria-selected="false"] .processing-track {' in style
    assert "opacity: 0.72;" in style
    assert '.processing-lane[aria-selected="true"] {' in style
    assert "box-shadow: inset 2px 0 #898dff;" in style
    pending_repair_style = style.split(
        ".decision-trace-harness .trace-repair-loop.pending {", 1
    )[1].split("}", 1)[0]
    assert "stroke-dasharray: 4 5;" in pending_repair_style
    assert "opacity: 0.66;" in pending_repair_style
    assert "filter: none;" in pending_repair_style
    reference_style = style.split("/* ---------- reference-first", 1)[1]
    processing_panel_style = reference_style.split(".processing-panel {", 1)[1].split("}", 1)[0]
    assert "height: auto;" in processing_panel_style
    assert "height: 1030px;" not in style
    assert "height: 802px;" not in style
    assert style.count("height: 607px;") == 2
    assert "trace-check" not in page
    assert ".trace-check" not in style
    assert (
        ".decision-trace-harness .trace-overall .trace-node.done circle,\n"
        ".decision-trace-harness .done.decision-lane-step circle,\n"
        ".decision-trace-harness .decision-lane-step.done circle {\n"
        "  fill: url(#trace-active-core);\n"
        "  stroke: #9296ff;\n"
        "  stroke-width: 2.1;"
    ) in style
    assert (
        ".decision-trace-harness .done.trace-shared-node circle,\n"
        ".decision-trace-harness .trace-shared-node.done circle {\n"
        "  fill: url(#trace-active-core);\n"
        "  stroke: #9296ff;"
    ) in style
    done_node_style = style.split(
        ".decision-trace-harness .done.trace-node circle,", 1
    )[1].split("}", 1)[0]
    assert "filter: url(#trace-glow-violet);" in done_node_style
    assert ".decision-trace-harness .trace-single-path.pending" not in style
    assert ".decision-trace-harness .trace-hold-path.pending" not in style
    assert ".decision-trace-harness .skipped.trace-path" not in style
    assert ".decision-trace-harness .trace-tie-path.pending" not in style
    assert ".decision-trace-harness .done .trace-path" not in style
    assert (
        ".decision-trace-harness .trace-context-guide,\n"
        ".decision-trace-harness .trace-reasoning-guide {\n"
        "  fill: none;\n"
        "  stroke: #7d8992;"
    ) in style
    assert (
        '.decision-trace-harness .trace-diamond [data-plan-value="fit"] {'
        not in style
    )
    assert "headroom OK ·" not in renderer
    assert '? "headroom OK"' in renderer
    active_path_style = style.split(
        ".decision-trace-harness .active.trace-path,", 1
    )[1].split("}", 1)[0]
    assert "filter: drop-shadow(" in active_path_style
    assert "filter: url(" not in active_path_style
    assert "stroke-dasharray: none;" in active_path_style
    for state in ("done", "error"):
        reached_path_style = style.split(
            f".decision-trace-harness .{state}.trace-path,", 1
        )[1].split("}", 1)[0]
        assert "stroke-dasharray: none;" in reached_path_style
    assert (
        ".decision-trace-harness .trace-path {\n"
        "  stroke: #7d8992;\n"
        "  stroke-dasharray: 4 5;\n"
        "  opacity: 0.66;\n"
        "}"
    ) in style
    assert ".decision-trace-harness .trace-lane-rail.pending" not in style
    assert (
        ".decision-trace-harness .decision-lane-step.active circle {\n"
        "  fill: url(#trace-processing-core);\n"
        "  stroke: #ff9d00;\n"
        "  filter: url(#trace-glow-orange);"
    ) in style
    assert (
        ".decision-trace-harness .trace-lane-rail.active {\n"
        "  stroke: #ff8a00;\n"
        "  filter: drop-shadow(0 0 3px rgba(255, 138, 0, 0.68));"
    ) in style
    active_rail_style = style.split(
        ".decision-trace-harness .trace-lane-rail.active {", 1
    )[1].split("}", 1)[0]
    assert "filter: url(" not in active_rail_style
    assert (
        ".decision-trace-harness .decision-lane-step.active circle,\n"
        "  .decision-trace-harness .trace-lane-rail.active,\n"
        "  .decision-generation-track.indeterminate i {\n"
        "    animation: none;"
    ) in style
    assert '<radialGradient id="trace-processing-core">' in page
    assert (
        ".decision-trace-harness .trace-context-option.selected [data-context-label] {\n"
        "  fill: #ffb340;"
    ) in style
    assert (
        ".decision-trace-harness .trace-reasoning.selected [data-reasoning-label] {\n"
        "  fill: #ffb340;"
    ) in style
    assert (
        ".decision-trace-harness .trace-yes-label,\n"
        ".decision-trace-harness .trace-no-label {\n"
        "  fill: #7d8992;\n"
        "  opacity: 0.18;"
    ) in style
    assert (
        ".decision-trace-harness .trace-yes-label.active,\n"
        ".decision-trace-harness .trace-yes-label.done {\n"
        "  fill: #74d84f;\n"
        "  opacity: 1;"
    ) in style
    assert (
        ".decision-trace-harness .trace-no-label.active,\n"
        ".decision-trace-harness .trace-no-label.done,\n"
        ".decision-trace-harness .trace-no-label.error {\n"
        "  fill: #ff3948;\n"
        "  opacity: 1;"
    ) in style
    assert ".decision-trace-harness .trace-hold-node.pending" not in style
    assert (
        ".decision-trace-harness .trace-hold-node.error text {\n"
        "  fill: #ff3948;"
    ) in style
    assert "filter: url(#trace-glow-violet);" in style.split(
        ".decision-trace-harness .active.trace-node circle,", 1
    )[1].split("}", 1)[0]
    assert ".decision-console-chrome {\n  display: grid;" in style
    assert ".decision-console-lights i:first-child {\n  background: #ff5f57;" in style
    assert "grid-template-rows: 32px 32px 142px 55px;" in style
    assert "min-height: 261px;" in style
    assert ".decision-transition-feed {\n  display: grid;" in style
    assert ".decision-transition-event {\n  display: grid;" in style
    assert ".decision-generation-meter {\n  display: grid;" in style
    assert ".decision-generation-track.indeterminate i {" in style
    assert "@keyframes decision-console-scan" in style
    assert ".decision-events" not in page
    frame = renderer.split("function renderDecisionTraceFrame", 1)[1].split(
        "function setDecisionTransitionState", 1
    )[0]
    assert "updateProcessingTraceSelection(trace);" in frame
    harness = renderer.split("function updateDecisionSvgHarness", 1)[1].split(
        "function decisionEventText", 1
    )[0]
    assert 'decision: decisionOverallState(overall, "decision")' in harness
    assert 'decision: safeNoQuorum ? "skipped"' not in harness
    assert "decisionSealStates(" in harness
    assert 'harness.querySelector("[data-seal-yes-label]")' in harness
    assert 'harness.querySelector("[data-seal-no-label]")' in harness
    assert "const pairBranches = decisionPairBranchStates(trace, tieLane);" in harness
    assert 'const pairAgreement = pairBranches.yes === "done";' in harness
    assert "const pairNoState = pairBranches.no;" in harness
    assert 'harness.querySelector("[data-pair-no-label]"),\n    pairNoState' in harness
    assert 'const safeQuorumReached = artifactReached && !singleModel;' in harness
    assert 'quorum: safeNoQuorum ? "error" : tieAgreement ? "done" : "pending"' in harness
    assert '["pair-artifact-join", pairAgreement ? "done" : "pending"]' in harness
    assert '["tie_break-quorum", safeNoQuorum && tieUsed ? "error"' in harness
    assert '["quorum-artifact-join", quorumYesState]' in harness
    assert "quorum-artifact-trunk" not in harness
    assert "artifact-input" not in harness
    assert '["quorum-hold", quorumNoState]' in harness
    assert '["artifact-seal", sealStates.input]' in harness
    assert '["seal-decision", sealYesState]' in harness
    assert '["seal-hold", sealNoState]' in harness
    assert '["challenger-agree", completedStepState("challenger", "vote")]' in harness
    assert "const reasoning = decisionReasoningPlanState(activeLane, planState);" in harness
    assert "const fitState = reasoning.fit;" in harness
    assert '["plan-fit", actualThink === "off" ? "pending" : fitState]' in harness
    assert '["context-generate", "generate"]' in harness
    assert "fmt(laneSteps.get(phase)?.status, \"pending\")" in harness
    assert "Math.floor(tokens / 1000)" in renderer
    assert "Math.sign(contextX - 492)" in harness
    assert "Math.abs(contextX - 492) / 2" in harness
    assert '`M492 56 V${86 - contextRadius} Q492 86 ${492 +' in harness
    assert '${86 + contextRadius} V140`' in harness
    assert '`M${contextX} 160 V204 Q${contextX} 214 ${contextX + 10} 214 `' in harness
    assert '+ "H724 Q734 214 734 204 V174 Q734 164 744 164 H766"' in harness
    assert 'data-reasoning-output="${mode}"' in harness
    assert '["reasoning-off", "reasoning-low", "reasoning-medium", "reasoning-high"]' in harness
    assert 'node.classList.toggle("selected", value === selectedContextValue);' in harness
    assert 'node.classList.toggle("selected", node.dataset.reasoningKey === actualThink);' in harness
    assert "decisionRepairState(traceState, lane, repairs, focusEvent)" in harness
    assert 'observedPhases.add("load")' not in renderer


def test_dashboard_reference_repair_state_is_event_backed_for_every_lane() -> None:
    renderer = (ROOT / "src/chronovisor/dashboard_static/app-renderer.js").read_text(
        encoding="utf-8"
    )
    helper = "function decisionRepairState" + renderer.split(
        "function decisionRepairState", 1
    )[1].split("function updateDecisionSvgHarness", 1)[0]
    scenario = f"""
const vm = require("node:vm");
const sandbox = {{}};
vm.createContext(sandbox);
vm.runInContext({json.dumps(helper)} + "\\nthis.repairState = decisionRepairState;", sandbox);
const states = Object.fromEntries(["primary", "challenger", "tie_break"].map((key) => [key, [
  sandbox.repairState("active", {{ key, phase: "validate" }}, [], null),
  sandbox.repairState("active", {{ key, phase: "repair" }}, [{{ phase: "repair" }}], null),
  sandbox.repairState("agreed", {{ key, phase: "vote" }}, [{{ phase: "repair" }}], null),
]]));
process.stdout.write(JSON.stringify(states));
"""
    completed = subprocess.run(
        ["node", "-"],
        input=scenario,
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    assert json.loads(completed.stdout) == {
        lane: ["pending", "active", "done"]
        for lane in ("primary", "challenger", "tie_break")
    }


def test_dashboard_reference_seal_paths_follow_success_failure_and_no_quorum() -> None:
    renderer = (ROOT / "src/chronovisor/dashboard_static/app-renderer.js").read_text(
        encoding="utf-8"
    )
    helper = "function decisionSealStates" + renderer.split(
        "function decisionSealStates", 1
    )[1].split("function updateDecisionSvgHarness", 1)[0]
    scenario = f"""
const vm = require("node:vm");
const sandbox = {{}};
vm.createContext(sandbox);
vm.runInContext({json.dumps(helper)} + "\\nthis.sealStates = decisionSealStates;", sandbox);
process.stdout.write(JSON.stringify({{
  success: sandbox.sealStates("agreed", "done", false, false),
  ready: sandbox.sealStates("ready", "done", false, false),
  failure: sandbox.sealStates("quarantined", "error", true, false),
  noQuorum: sandbox.sealStates("quarantined", "skipped", false, true),
}}));
"""
    completed = subprocess.run(
        ["node", "-"],
        input=scenario,
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    assert json.loads(completed.stdout) == {
        "success": {
            "gate": "done",
            "input": "done",
            "yes": "done",
            "no": "pending",
            "artifact": "done",
        },
        "ready": {
            "gate": "done",
            "input": "done",
            "yes": "done",
            "no": "pending",
            "artifact": "done",
        },
        "failure": {
            "gate": "error",
            "input": "error",
            "yes": "pending",
            "no": "error",
            "artifact": "error",
        },
        "noQuorum": {
            "gate": "pending",
            "input": "pending",
            "yes": "pending",
            "no": "pending",
            "artifact": "skipped",
        },
    }


def test_dashboard_reference_bypass_skips_the_central_fit_input() -> None:
    renderer = (ROOT / "src/chronovisor/dashboard_static/app-renderer.js").read_text(
        encoding="utf-8"
    )
    scenario = f"""
const vm = require("node:vm");
const node = () => ({{
  dataset: {{}},
  classList: {{ remove() {{}}, add() {{}}, toggle() {{}} }},
  querySelector: () => node(),
  querySelectorAll: () => [],
  setAttribute() {{}},
  textContent: "",
}});
const planFit = node();
const harness = node();
harness.querySelector = (selector) => selector === '[data-path-key="plan-fit"]'
  ? planFit
  : node();
const sandbox = {{
  window: {{ matchMedia: () => ({{ matches: true }}) }},
  els: {{ decisionTraceHarness: harness }},
}};
vm.createContext(sandbox);
vm.runInContext({json.dumps(renderer)}, sandbox);
const render = (think) => {{
  sandbox.updateDecisionSvgHarness({{
    state: "agreed",
    lanes: [{{ key: "primary", state: "done", think, steps: [] }}],
    overall: [{{ key: "dispatch", status: "done" }}],
  }});
  return planFit.dataset.state;
}};
process.stdout.write(JSON.stringify({{ off: render("off"), medium: render("medium") }}));
"""
    completed = subprocess.run(
        ["node", "-"],
        input=scenario,
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    assert json.loads(completed.stdout) == {"off": "pending", "medium": "done"}


def test_dashboard_reference_no_quorum_leaves_decision_unreached() -> None:
    renderer = (ROOT / "src/chronovisor/dashboard_static/app-renderer.js").read_text(
        encoding="utf-8"
    )
    scenario = f"""
const vm = require("node:vm");
const node = (traceKey = null) => ({{
  dataset: traceKey ? {{ traceKey }} : {{}},
  classList: {{ remove() {{}}, add() {{}}, toggle() {{}} }},
  querySelector: () => node(),
  querySelectorAll: () => [],
  setAttribute() {{}},
  textContent: "",
}});
const traceNodes = Object.fromEntries(
  ["artifact", "decision", "hold", "agree", "quorum"].map((key) => [key, node(key)])
);
const harness = node();
harness.querySelectorAll = (selector) => selector === "[data-trace-key]"
  ? Object.values(traceNodes)
  : [];
harness.querySelector = () => node();
const sandbox = {{
  window: {{ matchMedia: () => ({{ matches: true }}) }},
  els: {{ decisionTraceHarness: harness }},
}};
vm.createContext(sandbox);
vm.runInContext({json.dumps(renderer)}, sandbox);
sandbox.updateDecisionSvgHarness({{
  state: "quarantined",
  quorum_attempted: true,
  quorum_flow: true,
  outcome: {{ kind: "error", code: "no_safe_quorum" }},
  overall: [
    {{ key: "artifact", status: "skipped" }},
    {{ key: "decision", status: "skipped" }},
  ],
}});
process.stdout.write(JSON.stringify(Object.fromEntries(
  Object.entries(traceNodes).map(([key, value]) => [key, value.dataset.state])
)));
"""
    completed = subprocess.run(
        ["node", "-"],
        input=scenario,
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    assert json.loads(completed.stdout) == {
        "artifact": "skipped",
        "decision": "skipped",
        "hold": "error",
        "agree": "error",
        "quorum": "error",
    }


def test_six_processing_inputs_keep_real_dashboard_paths_connected(
    monkeypatch, tmp_path: Path
) -> None:
    cases = [
        {**case, "lane_context_tokens": [131_072, None, None]}
        if case["id"] == "A"
        else case
        for case in json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"][:6]
    ]
    roles = {
        "ingest": "ingest_reconciliation",
        "recall": "recall_auto_apply",
        "audit": "content_correction_classification",
        "improve": "model_eval",
        "repair": "local_repair",
        "typed_graph": "relation_extract",
    }
    models = {
        "primary": "qwen3.8:27b-axq4",
        "challenger": "challenger:model",
        "tie_break": "tie:model",
    }
    monkeypatch.setattr(dashboard, "_decision_trace_models", lambda: dict(models))

    payload = []
    for case_index, case in enumerate(cases):
        request = str(case_index + 1) * 64
        role = roles[case["workflow"]]
        activities = []
        history = []
        active_phases = {"A": "trigger", "B": "validate", "D": "vote"}
        for lane_index, lane in enumerate(dashboard._DECISION_TRACE_ROLES):
            state = case["lane_states"][lane_index]
            tokens = case["lane_context_tokens"][lane_index]
            think = case["lane_think"][lane_index]
            observed = {
                "request_sha256": request,
                "role": f"{role}:{lane}",
                "model": models[lane],
                "think": True if case["id"] == "A" and lane == "primary" else think,
                "required_context_tokens": tokens,
                "requested_context_tokens": tokens,
                "context_tokens": tokens,
            }
            timestamp = f"2026-08-15T00:00:0{lane_index + 1}Z"
            if state == "active":
                activities.append(
                    {
                        **observed,
                        "phase": active_phases[case["id"]],
                        "attempt": 0,
                        "elapsed_seconds": case["elapsed_seconds"],
                        "started_at": timestamp,
                        "updated_at": timestamp,
                    }
                )
            elif state in {"done", "error"}:
                history.append(
                    {
                        **observed,
                        "kind": "session",
                        "timestamp": timestamp,
                        "ok": state == "done",
                        "first_pass_valid": state == "done",
                        "repair_turns": 0,
                    }
                )

        decision = None
        if case["state"] in {"agreed", "quarantined"}:
            tie_used = case["branch"] in {"tie-break-agreement", "no-safe-quorum"}
            decision = {
                "kind": "decision",
                "timestamp": "2026-08-15T00:00:09Z",
                "request_sha256": request,
                "role": role,
                "status": case["state"],
                "pair_agreement": case["branch"] == "pair-agreement",
                "tie_break_used": tie_used,
                "vote_count": 3 if tie_used else 2,
                "valid_votes": 3 if tie_used else 2,
                "vote_roles": list(models) if tie_used else ["primary", "challenger"],
                "models": list(models.values()),
            }
            if case["state"] == "quarantined":
                decision["quarantine_reason"] = (
                    "local_models_did_not_reach_two_vote_quorum"
                )
            history.append(decision)

        trace = dashboard._decision_trace_snapshot(
            activities,
            history,
            decision,
            preferred_request_sha256=request,
        )
        assert trace["state"] == case["state"]
        assert [lane["state"] for lane in trace["lanes"]] == case["lane_states"]
        assert [
            None if lane["think"] == "—" else lane["think"]
            for lane in trace["lanes"]
        ] == case["lane_think"]
        assert [lane["context_tokens"] for lane in trace["lanes"]] == case[
            "lane_context_tokens"
        ]
        payload.append({"case": case, "trace": trace})

    harness = """
const fixtures = __FIXTURES__;
const workflowKeys = fixtures.map(({ case: fixture }) => fixture.workflow);
const selectedPipelines = [];
function browserFailure(detail) {
  fetch("/fixture-error", {
    method: "POST",
    headers: { "Content-Type": "text/plain" },
    body: String(detail).slice(0, 2000),
  }).catch(() => {});
}
addEventListener("error", (event) => browserFailure(event.error?.stack || event.message));
addEventListener("unhandledrejection", (event) => browserFailure(event.reason?.stack || event.reason));
addEventListener("chronovisor:processing-lane-select", (event) => {
  selectedPipelines.push(event.detail?.pipeline || "");
});
addEventListener("DOMContentLoaded", () => {
  const api = window.__chronovisorDashboardTest;
  const renderFixture = (fixture, trace, revision) => {
    api.renderProcessingActivity({
      generated_at: `2026-08-15T00:01:${revision}Z`,
      revision,
      active_count: trace.active ? 1 : 0,
      lanes: workflowKeys.map((key) => ({
        key,
        label: key,
        state: key === fixture.workflow && trace.active ? "active" : "idle",
        current_step: key === fixture.workflow ? fixture.processing_stage : null,
        steps: [],
      })),
    });
    api.renderDecisionTraceFrame(trace);
  };
  const results = fixtures.map(({ case: fixture, trace }, index) => {
    renderFixture(fixture, trace, `0${index}`);
    const pathState = (node) => {
      const style = getComputedStyle(node);
      return {
        state: node.dataset.state,
        dash: style.strokeDasharray,
        length: node.getTotalLength(),
      };
    };
    const scroller = document.querySelector("#decision-trace-scroll");
    const harness = document.querySelector("#decision-trace-harness");
    scroller.scrollLeft = scroller.scrollWidth;
    const scrollerBounds = scroller.getBoundingClientRect();
    const harnessBounds = harness.getBoundingClientRect();
    const decisionBounds = document.querySelector("#decision-trace-panel").getBoundingClientRect();
    const saveHistoryBounds = document.querySelector("#save-history-panel").getBoundingClientRect();
    const scrollerRight = scrollerBounds.right;
    const harnessRight = harnessBounds.right;
    const layout = {
      clientWidth: scroller.clientWidth,
      scrollWidth: scroller.scrollWidth,
      scrollLeft: scroller.scrollLeft,
      rightReachable: harnessRight <= scrollerRight + 1,
      fitsWidth: Math.abs(harnessBounds.width - scrollerBounds.width) <= 1,
      nextPanelGap: saveHistoryBounds.top - decisionBounds.bottom,
    };
    scroller.scrollLeft = 0;
    return {
      id: fixture.id,
      layout,
      selectionEvent: selectedPipelines.at(-1),
      selected: [...document.querySelectorAll('[data-processing-lane][aria-selected="true"]')]
        .map((node) => node.dataset.processingLane),
      context: document.querySelector("[data-context-option].selected")?.dataset.contextTokens,
      reasoning: document.querySelector("[data-reasoning-key].selected")?.dataset.reasoningKey,
      paths: Object.fromEntries([...document.querySelectorAll("[data-path-key]")]
        .map((node) => [node.dataset.pathKey, pathState(node)])),
      rails: Object.fromEntries([...document.querySelectorAll("[data-decision-lane]")]
        .map((lane) => [lane.dataset.decisionLane, [...lane.querySelectorAll("[data-lane-path]")]
          .map((node) => ({ key: node.dataset.lanePath, ...pathState(node) }))])),
    };
  });
  renderFixture(fixtures[0].case, fixtures[0].trace, "59");
  document.querySelector('[data-processing-lane="recall"]').click();
  results[0].tabClick = {
    event: selectedPipelines.at(-1),
    selected: document.querySelector('[data-processing-lane][aria-selected="true"]')
      ?.dataset.processingLane,
    tabCount: document.querySelectorAll('[data-processing-lane][role="tab"]').length,
    panelRole: document.querySelector("#decision-trace-panel")?.getAttribute("role"),
  };
  document.querySelector(
    `[data-processing-lane="${fixtures[0].case.workflow}"]`
  ).click();
  fetch("/fixture-result", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(results),
  }).catch(browserFailure);
});
""".replace("__FIXTURES__", json.dumps(payload, ensure_ascii=False))
    page = (dashboard.STATIC_DIR / "index.html").read_text(encoding="utf-8").replace(
        '<script src="/static/app-client.js"></script>',
        '<script src="/fixture-harness.js"></script>',
    )
    page_path = tmp_path / "index.html"
    harness_path = tmp_path / "fixture-harness.js"
    screenshot_path = tmp_path / "six-case-dashboard.png"
    page_path.write_text(page, encoding="utf-8")
    harness_path.write_text(harness, encoding="utf-8")
    result_ready = threading.Event()
    browser_results = []
    browser_errors = []

    class Handler(dashboard.DashboardHandler):
        def do_GET(self) -> None:
            if self.path in {"/", "/fixture-harness.js"}:
                if self._browser_boundary_allows():
                    dashboard._file_response(
                        self,
                        page_path if self.path == "/" else harness_path,
                    )
                return
            super().do_GET()

        def do_POST(self) -> None:
            if self.path not in {"/fixture-error", "/fixture-result"}:
                self.send_error(404)
                return
            if not self._browser_boundary_allows():
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65_536:
                self.send_error(400)
                return
            body = self.rfile.read(length)
            if self.path == "/fixture-error":
                browser_errors.append(body.decode(errors="replace"))
            else:
                browser_results.extend(json.loads(body))
                result_ready.set()
            self.send_response(204)
            self.end_headers()

        def log_message(self, _format: str, *_args: object) -> None:
            return

    chrome = next(
        (
            candidate
            for candidate in (
                shutil.which("google-chrome"),
                shutil.which("chromium"),
                shutil.which("chromium-browser"),
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            )
            if candidate and Path(candidate).is_file()
        ),
        None,
    )
    assert chrome is not None, "headless Chrome is required for Dashboard DOM tests"

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    process = None
    browser_exit = None
    received = False
    try:
        process = subprocess.Popen(
            [
                chrome,
                "--headless=new",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-background-networking",
                "--disable-component-update",
                "--no-first-run",
                "--force-prefers-reduced-motion=reduce",
                "--window-size=1050,992",
                f"--screenshot={screenshot_path}",
                f"--user-data-dir={tmp_path / 'chrome-profile'}",
                f"http://127.0.0.1:{server.server_port}/",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        received = result_ready.wait(60)
        if received and process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        browser_exit = process.poll()
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)

    stderr = process.stderr.read()[-4000:] if process and process.stderr else ""
    diagnostics = json.dumps(
        {
            "browser_exit": browser_exit,
            "browser_stderr": stderr,
            "browser_errors": browser_errors,
            "browser_results": browser_results,
        },
        indent=2,
    )
    assert received, diagnostics
    assert not browser_errors, diagnostics
    assert len(browser_results) == 6, diagnostics
    assert screenshot_path.is_file() and screenshot_path.stat().st_size > 0

    branches = {
        "A": [],
        "B": ["primary-challenger"],
        "C": [
            "primary-challenger",
            "challenger-agree",
            "pair-artifact-join",
            "artifact-seal",
            "seal-decision",
        ],
        "D": ["primary-challenger", "challenger-agree", "pair-tie_break"],
        "E": [
            "primary-challenger",
            "challenger-agree",
            "pair-tie_break",
            "tie_break-quorum",
            "quorum-artifact-join",
            "artifact-seal",
            "seal-decision",
        ],
        "F": [
            "primary-challenger",
            "challenger-agree",
            "pair-tie_break",
            "tie_break-quorum",
            "quorum-hold",
        ],
    }
    active_rail_counts = {"A": 0, "B": 4, "D": 5}
    by_id = {result["id"]: result for result in browser_results}
    for case in cases:
        result = by_id[case["id"]]
        assert result["selected"] == [case["workflow"]]
        assert result["selectionEvent"] == case["workflow"]
        assert result["layout"]["scrollWidth"] == result["layout"]["clientWidth"]
        assert result["layout"]["rightReachable"] is True
        assert result["layout"]["fitsWidth"] is True
        assert result["layout"]["nextPanelGap"] == 12
        selected_index = (
            case["lane_states"].index("active")
            if "active" in case["lane_states"]
            else max(
                index
                for index, state in enumerate(case["lane_states"])
                if state == "done"
            )
        )
        assert result["context"] == str(case["lane_context_tokens"][selected_index])
        assert result["reasoning"] == case["lane_think"][selected_index]

        reached_paths = [
            "execution-plan-context",
            "plan-context",
            f"reasoning-{case['lane_think'][selected_index]}",
            "plan-fit",
            "plan-dispatch",
            *branches[case["id"]],
        ]
        for path_key in reached_paths:
            path = result["paths"][path_key]
            assert path["state"] in {"active", "done", "error"}
            assert path["dash"] in {"none", ""}
            assert path["length"] > 0

        for lane_index, lane in enumerate(dashboard._DECISION_TRACE_ROLES):
            state = case["lane_states"][lane_index]
            if state not in {"active", "done"}:
                continue
            count = active_rail_counts.get(case["id"], 5) if state == "active" else 5
            for rail in result["rails"][lane][:count]:
                assert rail["state"] in {"active", "done", "error"}
                assert rail["dash"] in {"none", ""}
                assert rail["length"] > 0

    assert by_id[cases[0]["id"]]["tabClick"] == {
        "event": "recall",
        "selected": "recall",
        "tabCount": 6,
        "panelRole": "tabpanel",
    }
