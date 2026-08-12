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
    assert contract["topology"]["dynamic_paths"] == ["processing-entry"]
    assert contract["topology"]["fixed_paths"][-5:] == [
        "tie-break-artifact",
        "artifact-seal",
        "seal-decision",
        "no-safe-quorum-hold",
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

    for row in cases.values():
        assert len(row["lane_states"]) == 3
        assert len(row["lane_think"]) == 3
        assert len(row["lane_context_tokens"]) == 3


def test_dashboard_reference_svg_has_one_fixed_safe_topology() -> None:
    parser = _DashboardMarkup()
    parser.feed(
        (ROOT / "src/chronovisor/dashboard_static/index.html").read_text(
            encoding="utf-8"
        )
    )
    elements = parser.elements

    def matching(key: str, value: str) -> list[dict[str, str | None]]:
        return [attrs for _tag, attrs in elements if attrs.get(key) == value]

    assert len(matching("id", "processing-panel")) == 1
    assert len(matching("class", "processing-lanes-card")) == 1
    assert len(matching("id", "decision-trace-panel")) == 1
    assert len(matching("id", "processing-trace-connector-path")) == 1
    assert len(matching("id", "decision-trace-harness")) == 1
    assert matching("id", "decision-trace-harness")[0]["viewbox"] == ("0 0 1500 650")
    assert matching("id", "decision-trace-harness")[0]["height"] == "607"

    assert len(matching("data-trace-key", "artifact")) == 1
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
    assert len(matching("data-context-tokens", "32768")) == 1
    assert len(matching("data-context-tokens", "65536")) == 1
    assert len(matching("data-context-tokens", "98304")) == 1
    assert len(matching("data-context-tokens", "131072")) == 1
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

    paths = {
        attrs["data-path-key"]: attrs.get("d")
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-path-key")
    }
    assert {
        "single-artifact",
        "pair-artifact",
        "pair-tie_break",
        "tie_break-artifact",
        "pair-hold",
        "tie_break-hold",
        "artifact-seal",
        "seal-decision",
        "seal-hold",
    }.issubset(paths)
    assert all(paths[key] for key in paths)
    assert paths["pair-artifact"] != paths["tie_break-artifact"]
    assert paths["seal-hold"] == "M1401 434 V482 Q1401 492 1411 492 H1460"
    assert paths["pair-hold"] == "M1208 490 V494 Q1208 504 1218 504 H1448"
    assert paths["tie_break-hold"] == (
        "M1096 575 H1408 Q1418 575 1418 565 "
        "V514 Q1418 504 1428 504 H1448"
    )
    assert any(
        tag == "path" and attrs.get("d") == "M1208 410 1248 450 1208 490 1168 450Z"
        for tag, attrs in elements
    )
    context_generate_rails = matching("data-lane-path", "context-generate")
    assert len(context_generate_rails) == 3
    assert all(
        attrs.get("d") == "M565 0 H733"
        and attrs.get("marker-end") == "url(#trace-arrow)"
        for attrs in context_generate_rails
    )

    expected_boundary_paths = {
        "packet-preflight": "M96 10 H289",
        "preflight-execution_plan": "M298 10 H501",
        "plan-dispatch": (
            "M1380 164 H1466 Q1476 164 1476 174 V265 Q1476 275 1466 275 "
            "H190 Q180 275 180 285 V315 Q180 325 190 325 H220"
        ),
        "primary-challenger": (
            "M1096 325 V377 Q1096 387 1086 387 H190 Q180 387 180 397 "
            "V440 Q180 450 190 450 H220"
        ),
        "single-artifact": "M1096 325 H1332 Q1342 325 1342 335 V393",
        "challenger-agree": "M1096 450 H1168",
        "pair-artifact": (
            "M1248 450 H1266 Q1276 450 1276 440 V414 Q1276 404 1286 404 H1331"
        ),
        "pair-tie_break": (
            "M1208 490 V502 Q1208 512 1198 512 "
            "H190 Q180 512 180 522 V565 Q180 575 190 575 H220"
        ),
        "tie_break-artifact": "M1096 575 H1332 Q1342 575 1342 565 V415",
        "artifact-seal": "M1342 404 H1369",
        "seal-decision": "M1433 404 H1449",
        "seal-hold": "M1401 434 V482 Q1401 492 1411 492 H1460",
    }
    assert {key: paths[key] for key in expected_boundary_paths} == (
        expected_boundary_paths
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
        and attrs.get("marker-end") == "url(#trace-arrow)"
        for attrs in repair_paths
    )

    reasoning_guides = [
        attrs
        for tag, attrs in elements
        if tag == "path" and attrs.get("class") == "trace-reasoning-guide"
    ]
    assert len(reasoning_guides) == 1
    assert [attrs.get("d") for attrs in reasoning_guides] == [
        "M858 164 H1270 M892 164 V106 H1255 V164 M892 222 V164 "
        "M892 222 H1255 M1255 222 V164",
    ]
    assert [
        (attrs.get("data-reasoning-output"), attrs.get("marker-end"))
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-reasoning-output")
    ] == [("low", None), ("medium", None), ("high", None)]
    assert all(
        attrs.get("marker-end") == "url(#trace-arrow)"
        for tag, attrs in elements
        if tag == "path"
        and attrs.get("data-path-key")
        in {"reasoning-low", "reasoning-medium", "reasoning-high", "plan-fit"}
    )
    assert all(
        attrs.get("y") == "28"
        for tag, attrs in elements
        if tag == "text" and attrs.get("data-lane-result")
    )
    assert len(matching("class", "trace-label-backdrop")) == 1
    assert len(matching("class", "trace-selected-core")) == 3
    assert {
        attrs["data-reasoning-key"]: attrs.get("transform")
        for _tag, attrs in elements
        if attrs.get("data-reasoning-key")
    } == {
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
    assert matching("class", "trace-artifact-label")[0]["x"] == "-16"
    assert matching("class", "trace-artifact-label")[0]["text-anchor"] == "end"
    assert [
        attrs.get("y")
        for _tag, attrs in elements
        if attrs.get("class") == "trace-path-label"
    ] == ["271", "381"]
    assert matching("class", "trace-branch-label trace-yes-label")[0]["y"] == "431"
    assert matching("class", "trace-branch-label trace-no-label")[0]["y"] == "506"
    assert matching("data-seal-yes-label", "true")[0]["x"] == "1438"
    assert matching("data-seal-no-label", "true")[0]["x"] == "1412"
    assert paths["plan-context"] == (
        "M442 164 V204 Q442 214 452 214 H724 Q734 214 734 204 V164"
    )
    context_guides = [
        attrs
        for tag, attrs in elements
        if tag == "path" and attrs.get("data-context-guide")
    ]
    assert len(context_guides) == 1
    assert context_guides[0]["data-context-guide"] == "all"
    assert context_guides[0]["d"] == (
        "M510 19 V86 M342 86 H642 M342 86 V164 M442 86 V164 "
        "M542 86 V164 M642 86 V164 M342 164 V214 M442 164 V214 "
        "M542 164 V214 M642 164 V214 M342 214 H734 M734 214 V164 H766"
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
        "translate(342 164)",
        "translate(442 164)",
        "translate(542 164)",
        "translate(642 164)",
    ]
    assert matching("data-plan-value", "context-selection")[0]["y"] == "239"


def test_dashboard_reference_keeps_selection_and_bucket_truth() -> None:
    static = ROOT / "src/chronovisor/dashboard_static"
    page = (static / "index.html").read_text(encoding="utf-8")
    style = (static / "style.css").read_text(encoding="utf-8")
    renderer = (static / "app-renderer.js").read_text(encoding="utf-8")

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
    assert 'const artifactReached = traceState === "agreed" || sealFailure;' in harness
    assert '["artifact-seal", sealStates.input]' in harness
    assert '["seal-decision", sealStates.yes]' in harness
    assert '["seal-hold", sealStates.no]' in harness
    assert '["challenger-agree", completedStepState("challenger", "vote")]' in harness
    assert 'const fitState = reasoningSelected ? planState : "pending";' in harness
    assert '["context-generate", "generate"]' in harness
    assert "fmt(laneSteps.get(phase)?.status, \"pending\")" in harness
    assert 'querySelector("[data-trace-entry] circle")' in renderer
    assert 'getComputedStyle(source, "::before")' in renderer
    assert "const y2 = targetBox.top - panelBox.top;" in renderer
    assert "Math.floor(tokens / 1000)" in renderer
    assert '`M${contextX} 164 V204 Q${contextX} 214 ${contextX + 10} 214 `' in harness
    assert '+ "H724 Q734 214 734 204 V164"' in harness
    assert 'data-reasoning-output="${mode}"' in harness
    assert 'node.classList.toggle("selected", value === selectedContextValue);' in harness
    assert 'node.classList.toggle("selected", node.dataset.reasoningKey === actualThink);' in harness
    assert "decisionRepairState(traceState, lane, repairs, focusEvent)" in harness


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
