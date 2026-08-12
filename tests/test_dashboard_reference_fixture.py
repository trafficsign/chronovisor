from __future__ import annotations

import hashlib
import json
import struct
import subprocess
from html.parser import HTMLParser
from pathlib import Path

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
        "pair_yes",
        "tie_input",
        "gate",
        "gate_yes",
        "quorum_trunk",
        "artifact_input",
        "no",
    ]
    assert contract["topology"]["fixed_paths"][-9:] == [
        "pair-artifact-join",
        "tie_break-quorum",
        "quorum-artifact-join",
        "quorum-artifact-trunk",
        "artifact-input",
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
    assert cases["G"]["hold"] == "seal-failed"
    assert cases["G"]["artifact"] == "error"
    assert cases["G"]["decision"] == "skipped"
    assert cases["H"]["state"] == "idle"
    assert cases["H"]["active_lane"] is None
    assert [cases[key]["artifact_join_states"] for key in cases] == [
        ["pending", "pending", "pending", "pending", "pending", "pending", "pending"],
        ["pending", "pending", "pending", "pending", "pending", "pending", "pending"],
        ["done", "pending", "pending", "pending", "pending", "done", "pending"],
        ["pending", "pending", "pending", "pending", "pending", "pending", "pending"],
        ["pending", "done", "done", "done", "done", "done", "pending"],
        ["pending", "error", "error", "pending", "pending", "pending", "error"],
        ["done", "pending", "pending", "pending", "pending", "done", "pending"],
        ["pending", "pending", "pending", "pending", "pending", "pending", "pending"],
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
        "quorum-artifact-trunk",
        "artifact-input",
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

    expected_boundary_paths = {
        "packet-preflight": "M106 10 H288",
        "preflight-execution_plan": "M308 10 H500 Q510 10 510 20 V36",
        "execution-plan-context": (
            "M510 56 V76 Q510 86 500 86 H452 Q442 86 442 96 V140"
        ),
        "plan-dispatch": (
            "M1380 164 H1466 Q1476 164 1476 174 V265 Q1476 275 1466 275 "
            "H190 Q180 275 180 285 V315 Q180 325 190 325 H220"
        ),
        "primary-challenger": (
            "M1096 335 V377 Q1096 387 1086 387 H190 Q180 387 180 397 "
            "V440 Q180 450 190 450 H220"
        ),
        "single-artifact": "M1106 325 H1300 Q1310 325 1310 335 V393",
        "challenger-agree": "M1106 450 H1168",
        "pair-tie_break": (
            "M1208 490 V502 Q1208 512 1198 512 "
            "H190 Q180 512 180 522 V565 Q180 575 190 575 H220"
        ),
        "pair-artifact-join": "M1248 450 H1300 Q1310 450 1310 440",
        "tie_break-quorum": "M1106 575 H1217",
        "quorum-artifact-join": (
            "M1248 545 V524 Q1248 514 1258 514 H1300 Q1310 514 1310 504"
        ),
        "quorum-artifact-trunk": "M1310 504 V440",
        "artifact-input": "M1310 440 V415",
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
        matching("data-path-key", key)[0].get("marker-end") is None
        for key in (
            "pair-artifact-join",
            "quorum-artifact-join",
            "quorum-artifact-trunk",
        )
    )
    assert matching("data-path-key", "artifact-input")[0]["marker-end"] == (
        "url(#trace-arrow)"
    )

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
        "M858 164 H872 Q882 164 882 154 V72 Q882 62 892 62 H903 "
        "M921 62 H1235 Q1245 62 1245 72 V144 Q1245 154 1255 154 "
        "H1260 Q1270 154 1270 164 "
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
            "M921 62 H1235 Q1245 62 1245 72 V144 Q1245 154 1255 154 "
            "H1260 Q1270 154 1270 164",
            None,
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
        "off": "translate(912 62)",
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
        "translate(510 46)"
    )
    assert matching("class", "trace-artifact-label")[0]["x"] == "-16"
    assert matching("class", "trace-artifact-label")[0]["text-anchor"] == "end"
    assert [
        attrs.get("y")
        for _tag, attrs in elements
        if attrs.get("class") == "trace-path-label"
    ] == ["271", "381"]
    assert matching("class", "trace-branch-label trace-yes-label")[0]["y"] == "443"
    assert matching("class", "trace-branch-label trace-no-label")[0]["y"] == "506"
    assert matching("data-pair-yes-label", "true")[0]["x"] == "1272"
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
        "M510 56 V76 Q510 86 500 86 H352 Q342 86 342 96 V140 "
        "M510 56 V76 Q510 86 500 86 H452 Q442 86 442 96 V140 "
        "M510 56 V76 Q510 86 520 86 H532 Q542 86 542 96 V140 "
        "M510 56 V76 Q510 86 520 86 H632 Q642 86 642 96 V140 "
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
    assert (
        '.processing-lane[aria-expanded="false"] .processing-track {\n'
        "  visibility: hidden;"
    ) in style
    pending_repair_style = style.split(
        ".decision-trace-harness .trace-repair-loop.pending {", 1
    )[1].split("}", 1)[0]
    assert "stroke-dasharray: 4 5;" in pending_repair_style
    assert "opacity: 0.66;" in pending_repair_style
    assert "filter: none;" in pending_repair_style
    assert style.count("height: 1030px;") == 3
    assert style.count("height: 802px;") == 3
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
        "  .decision-trace-harness .trace-lane-rail.active {\n"
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
    assert "filter: url(#trace-glow-violet);" in style.split(
        ".decision-trace-harness .active.trace-node circle,", 1
    )[1].split("}", 1)[0]
    assert (
        ".decision-transition-bar:has(.decision-events[open]) {\n"
        "  flex-basis: auto;\n"
        "  align-items: start;\n"
        "  min-height: 240px;\n"
        "  padding-block: 15px;"
    ) in style
    assert (
        ".decision-trace-panel:has(.decision-events[open]) {\n"
        "  height: 930px;"
    ) in style
    event_feed_style = style.split(
        ".decision-events .decision-transition-feed {", 1
    )[1].split("}", 1)[0]
    assert "position: absolute;" not in event_feed_style
    assert "bottom:" not in event_feed_style
    assert "margin-top: 8px;" in event_feed_style
    assert (
        ".decision-transition-bar:has(.decision-events[open]) {\n"
        "    grid-template-columns: 1fr;"
    ) in style
    assert (
        ".decision-events .decision-transition-feed {\n"
        "    width: auto;"
    ) in style
    frame = renderer.split("function renderDecisionTraceFrame", 1)[1].split(
        "function setDecisionTransitionState", 1
    )[0]
    assert "updateProcessingTraceSelection(trace);" in frame
    harness = renderer.split("function updateDecisionSvgHarness", 1)[1].split(
        "function decisionEventText", 1
    )[0]
    assert 'decision: decisionOverallState(overall, "decision")' in harness
    assert 'decision: noSafeQuorum ? "skipped"' not in harness
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
    assert '["quorum-artifact-trunk", quorumYesState]' in harness
    assert '["artifact-input", safeQuorumReached ? "done" : "pending"]' in harness
    assert '["quorum-hold", quorumNoState]' in harness
    assert '["artifact-seal", sealStates.input]' in harness
    assert '["seal-decision", sealYesState]' in harness
    assert '["seal-hold", sealNoState]' in harness
    assert '["challenger-agree", completedStepState("challenger", "vote")]' in harness
    assert "const reasoning = decisionReasoningPlanState(activeLane, planState);" in harness
    assert "const fitState = reasoning.fit;" in harness
    assert '["context-generate", "generate"]' in harness
    assert "fmt(laneSteps.get(phase)?.status, \"pending\")" in harness
    assert "Math.floor(tokens / 1000)" in renderer
    assert '`M510 56 V${86 - contextRadius}' in harness
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
